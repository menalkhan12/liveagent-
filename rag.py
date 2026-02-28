import os
import json
import time
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

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(f"GEMINI_API_KEY not found in .env file at {env_path}")

import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

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


MAX_CONTEXT_CHARS = 14000


def _expand_query_for_retrieval(query):
    q = query.lower()
    extra = []
    if any(w in q for w in ["cost", "price", "tuition", "fee", "mifi", "mephee", "fees"]):
        extra.append("fee structure tuition semester charges rupees")
    if any(w in q for w in ["merit", "closing", "aggregate", "calculate"]):
        extra.append("merit aggregate matric FSC entry test engineering")
    if any(w in q for w in ["electrical", "electrical engineering"]) and any(w in q for w in ["program", "offer", "department"]):
        extra.append("electrical engineering department programs BS Computer Engineering")
    if any(w in q for w in ["hostel", "harassment", "charges", "accommodation", "boarding"]):
        extra.append("hostel charges accommodation transport fee")
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
    # Fix common STT mishearings before processing
    query = _fix_stt_errors(query)

    context = retrieve_context(query)

    if not context.strip():
        return (
            "I don't have that information. Please provide your phone number and we will contact you.",
            True
        )

    prompt = f"""You are the official voice assistant for Institute of Space Technology (IST). You answer callers by phone.

STRICT RULES:
- Answer ONLY from the CONTEXT below. NEVER invent or add information.
- Electrical Engineering department programs: Say ONLY "BS Electrical Engineering and BS Computer Engineering". Never mention others.
- Questions NOT in CONTEXT: Say "I don't have that information. Please provide your phone number and we will contact you."
- If caller challenges you: say "This information comes from the official sources of the university."
- Answer DIRECTLY with facts. NEVER say "check the file" or "visit the website".
- Use amounts in lakh and thousand (e.g., 1 lakh 48 thousand rupees).
- Keep responses 1-3 short sentences, conversational for speech.
- For aggregate calculation: Formula is (Matric/1100 x 10) + (FSC/1100 x 40) + (EntryTest/100 x 50). Calculate immediately when marks are given.
- Be professional and friendly.

CONTEXT:
{context}

User query: {query}

Answer:"""

    for attempt in range(3):
        try:
            response = gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=150
                )
            )

            reply = response.text.strip()
            escalated = any(p in reply.lower() for p in [
                "technical issue", "cannot find", "unable",
                "phone number", "provide your phone", "contact you"
            ])
            return reply, escalated

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                if attempt < 2:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue
                logger.error(f"Rate limit after 3 attempts: {e}")
                return ("I'm temporarily busy. Please call back in a minute.", False)
            if "timeout" in error_str:
                logger.error(f"Gemini timeout: {e}")
                return ("Taking too long to respond. Please try again.", False)
            logger.error(f"Gemini error: {e}")
            return ("Technical issue. Please provide your phone number.", True)

    return ("I'm temporarily unavailable. Please call back in a moment.", False)


def _fix_stt_errors(text):
    """Fix common Whisper STT mishearings for IST domain."""
    replacements = {
        "mephee": "fee",
        "mifi": "fee",
        "mefi": "fee",
        "the fee structure": "fee structure",
        "harassment": "hostel",        # common mishearing
        "hostile": "hostel",
        "hotel": "hostel",
        "isp ": "IST ",
        "ist ": "IST ",
        "space technology": "IST",
        "institute of space": "IST",
    }
    lower = text.lower()
    for wrong, correct in replacements.items():
        lower = lower.replace(wrong, correct)
    # Restore proper casing for IST
    lower = lower.replace("ist", "IST")
    return lower