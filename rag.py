import os
import json
import logging
from pathlib import Path
from openai import OpenAI

# Read .env file directly
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(f"OPENROUTER_API_KEY not found. Please set it in .env or environment variables.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# OpenRouter client - uses OpenAI-compatible API
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0
)

# Best free model on OpenRouter - no daily limit, just per-minute rate limit
MODEL = "google/gemini-2.0-flash-exp:free"

documents = []
doc_names = []
vectorizer = None
doc_vectors = None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _chunk_text(text, max_len=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
                    chunk = p[i: i + max_len].strip()
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
        logger.error(f"Error loading documents: {e}")


def initialize_rag():
    global vectorizer, doc_vectors
    load_documents()

    if not documents:
        logger.warning("No documents found")
        return

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            min_df=1,
            max_df=0.95
        )
        doc_vectors = vectorizer.fit_transform(documents)
        logger.info("RAG initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise


MAX_CONTEXT_CHARS = 10000


def _fix_stt_errors(text):
    """Fix common Whisper STT mishearings for IST domain."""
    replacements = {
        "mephee": "fee",
        "mifi": "fee",
        "mefi": "fee",
        "harassment": "hostel",
        "hostile": "hostel",
        "hotel": "hostel",
        "isp ": "IST ",
        "isd ": "IST ",
        "i.s.t": "IST",
    }
    lower = text.lower()
    for wrong, correct in replacements.items():
        lower = lower.replace(wrong, correct)
    return lower


def _expand_query_for_retrieval(query):
    q = query.lower()
    extra = []
    if any(w in q for w in ["cost", "price", "tuition", "fee", "mifi", "mephee", "fees"]):
        extra.append("fee structure tuition semester charges rupees")
    if any(w in q for w in ["merit", "closing", "aggregate", "calculate"]):
        extra.append("merit aggregate matric FSC entry test engineering")
    if any(w in q for w in ["electrical"]) and any(w in q for w in ["program", "offer", "department"]):
        extra.append("electrical engineering department programs BS Computer Engineering")
    if any(w in q for w in ["hostel", "accommodation", "boarding", "harassment"]):
        extra.append("hostel charges accommodation fee")
    if any(w in q for w in ["transport", "bus", "shuttle"]):
        extra.append("transport bus shuttle routes fee charges")
    if "last year" in q:
        extra.append("2024")
    if extra:
        return query + " " + " ".join(extra)
    return query


def retrieve_context(query, top_k=6):
    global vectorizer, doc_vectors

    if vectorizer is None or doc_vectors is None:
        return ""

    try:
        expanded = _expand_query_for_retrieval(query)
        query_vec = vectorizer.transform([expanded])
        similarities = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        context_parts = []
        total_len = 0
        for idx in top_indices:
            if similarities[idx] <= 0.03:
                continue
            chunk = f"[{doc_names[idx]}]\n{documents[idx]}"
            if total_len + len(chunk) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total_len - 80
                if remain > 500:
                    chunk = chunk[:remain] + "\n...[truncated]"
                    context_parts.append(chunk)
                break
            context_parts.append(chunk)
            total_len += len(chunk)

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""


def generate_answer(query):
    # Fix STT mishearings first
    query = _fix_stt_errors(query)

    context = retrieve_context(query)

    if not context.strip():
        return (
            "I don't have that information. Please provide your phone number and we will contact you.",
            True
        )

    system_prompt = f"""You are the official voice assistant for Institute of Space Technology (IST). You answer callers by phone.

STRICT RULES:
- Answer ONLY from the CONTEXT below. NEVER invent or add information.
- Electrical Engineering programs: Say ONLY "BS Electrical Engineering and BS Computer Engineering". Never mention others.
- Questions NOT in CONTEXT: Say "I don't have that information. Please provide your phone number and we will contact you."
- If caller challenges you: say "This information comes from official university sources."
- Answer DIRECTLY with facts. NEVER say "check the file" or "visit the website".
- Use amounts in lakh and thousand (e.g., 1 lakh 48 thousand rupees).
- Keep responses 1-3 short sentences, conversational for speech.
- For aggregate: Formula is (Matric/1100 x 10) + (FSC/1100 x 40) + (EntryTest/100 x 50). Calculate immediately when marks are given.
- Be professional and friendly.

CONTEXT:
{context}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            max_tokens=150
        )

        reply = response.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")

        escalated = any(p in reply.lower() for p in [
            "technical issue", "cannot find", "unable",
            "phone number", "provide your phone", "contact you"
        ])
        return reply, escalated

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate" in error_str or "quota" in error_str:
            logger.error(f"OpenRouter rate limit: {e}")
            return ("Our lines are momentarily busy. Please try again in a few seconds.", False)
        if "timeout" in error_str:
            logger.error(f"OpenRouter timeout: {e}")
            return ("Taking too long to respond. Please try again.", False)
        logger.error(f"OpenRouter error: {e}")
        return ("Technical issue. Please provide your phone number.", True)