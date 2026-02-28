import os
import json
import logging
from pathlib import Path

# Read .env file directly
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(f"GROQ_API_KEY not found in .env file at {env_path}")

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
client = Groq(api_key=api_key)

documents = []
doc_names = []
vectorizer = None
doc_vectors = None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def _chunk_text(text, max_len=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks for better retrieval."""
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_len:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) > max_len:
                for i in range(0, len(p), max_len - overlap):
                    chunk = p[i : i + max_len].strip()
                    if chunk:
                        chunks.append(chunk)
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks

def load_documents():
    global documents, doc_names
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return
    
    documents = []
    doc_names = []
    
    try:
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            if not os.path.isfile(file_path):
                continue
            
            try:
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            for chunk in _chunk_text(content):
                                if chunk:
                                    documents.append(chunk)
                                    doc_names.append(file)
                            logger.info(f"Loaded: {file}")
                
                elif file.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        content = json.dumps(json_data, indent=2)
                        if content.strip():
                            for chunk in _chunk_text(content, max_len=1200):
                                if chunk:
                                    documents.append(chunk)
                                    doc_names.append(file)
                            logger.info(f"Loaded: {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if documents:
            logger.info(f"Loaded {len(documents)} chunks from documents")
    except Exception as e:
        logger.error(f"Error: {e}")

def initialize_rag():
    global vectorizer, doc_vectors
    load_documents()
    
    if not documents:
        logger.warning("No documents")
        return
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            min_df=1,
            max_df=0.95
        )
        doc_vectors = vectorizer.fit_transform(documents)
        logger.info("RAG initialized")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

# ~4 chars/token; free tier TPM 6000 - reserve ~500 for prompt+query+response
MAX_CONTEXT_CHARS = 14000

def retrieve_context(query, top_k=5):
    global vectorizer, doc_vectors
    
    if vectorizer is None or doc_vectors is None:
        return ""
    
    try:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context_parts = []
        total_len = 0
        for idx in top_indices:
            if similarities[idx] <= 0.05:
                continue
            chunk = f"[{doc_names[idx]}]\n{documents[idx]}"
            if total_len + len(chunk) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total_len - 80  # reserve for "..."
                if remain > 500:
                    chunk = chunk[:remain] + "\n...[truncated]"
                    context_parts.append(chunk)
                break
            context_parts.append(chunk)
            total_len += len(chunk)
        
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""

def generate_answer(query):
    try:
        context = retrieve_context(query)
        
        if not context.strip():
            return (
                "I don't have that information. Please provide your phone number and we'll have someone contact you.",
                True,
                True
            )
        
        system_prompt = f"""You are the official voice assistant for Institute of Space Technology. You answer callers by phone.

STRICT RULES:
- Answer ONLY from the CONTEXT below. NEVER invent, assume, guess, or add information not in the context. Stick strictly to the knowledge base. List ONLY what is explicitly stated.
- If the answer is NOT in CONTEXT: (a) For simple yes/no questions (e.g. "Does IST offer X?", "Is there Y?"): answer "No, that's not in our records." (b) For complex questions needing detailed answers: say "I don't have that information. Please provide your phone number and we'll have someone contact you."
- If the caller says "you're wrong", "that's incorrect", or challenges you: respond with "This information comes from the official sources of the university. I'm sharing what our records show." Do not change your answer.
- Answer DIRECTLY—state the figures and facts yourself. NEVER say "check the file" or "visit the website". The caller cannot see files.
- Use amounts in lakh and thousand (e.g., 1 lakh 48 thousand rupees).
- Keep responses 1-3 short sentences, conversational and natural for speech.
- For aggregate calculation: give ONLY the number in one sentence. Example: "Your aggregate is about 89.6."
- Be professional and friendly.

CONTEXT:
{context}"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        reply = response.choices[0].message.content.strip()
        escalated = any(p in reply.lower() for p in ["technical issue", "cannot find", "unable"])
        ask_phone = "phone" in reply.lower() and any(p in reply.lower() for p in ["don't have", "that information", "specific information", "contact you", "connect you"])
        if ask_phone:
            escalated = True
        return reply, escalated, ask_phone
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return (
            "Technical issue. Let me connect you with admissions. Phone number?",
            True,
            True
        )