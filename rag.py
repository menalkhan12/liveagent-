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

from groq_utils import get_client, num_keys, GROQ_KEYS
if not GROQ_KEYS:
    raise ValueError("GROQ_API_KEY or GROQ_API_KEYS not found. Set in .env or Render Environment.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Groq free tier: 30 RPM, 6K TPM, 500K TPD for 8B; 30 RPM, 12K TPM, 100K TPD for 70B
MODELS = [
    "llama-3.1-8b-instant",      # Fast, generous limits
    "llama-3.3-70b-versatile",  # Fallback if 8B fails
]

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
    if any(w in q for w in ["cost", "price", "tuition", "fee", "fees"]):
        extra.append("fee structure tuition semester charges rupees")
    if any(w in q for w in ["merit", "closing", "aggregate", "calculate"]):
        extra.append("merit aggregate matric FSC entry test engineering")
    if any(w in q for w in ["electrical"]) and any(w in q for w in ["program", "offer", "department"]):
        extra.append("electrical engineering department programs BS Computer Engineering")
    if any(w in q for w in ["hostel", "accommodation", "boarding"]):
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


def _is_thanks_or_compliment(query):
    q = query.lower().strip()
    thanks = any(x in q for x in ["thank", "thanks", "thx", "ok thanks", "okay thanks", "ty "])
    compliment = any(x in q for x in ["you're good", "youre good", "great job", "well done", "you're helpful", "youre helpful", "good job", "nice ", "awesome", "excellent"])
    return thanks, compliment

def generate_answer(query, conversation_history=None):
    query = _fix_stt_errors(query)
    q_lower = query.lower().strip()

    # Fast path: thanks / compliments
    is_thanks, is_compliment = _is_thanks_or_compliment(query)
    if is_thanks:
        return ("You're welcome.", False)
    if is_compliment:
        return ("Thank you.", False)

    # Resolve continuation: "its fee", "that program", etc. using recent conversation
    user_message = query
    retrieval_query = query
    if conversation_history:
        hist_str = "\n".join([f"User: {u}\nAgent: {a}" for u, a in conversation_history])
        user_message = f"""Previous conversation:
{hist_str}

Current query (may refer to above, e.g. "its fee" = fee of what we just discussed): {query}"""
        # Improve retrieval for vague follow-ups: add keywords from last agent response
        last_agent = conversation_history[-1][1] if conversation_history else ""
        if any(w in query.lower() for w in ["its", "that", "it", "this", "same"]):
            retrieval_query = f"{query} {last_agent}"

    context = retrieve_context(retrieval_query)

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
- POLITE: "Thank you"/"thanks"/"ok thanks" -> "You're welcome" or "Welcome". Compliments ("you're good", "great job") -> "Thank you."
- FEE: If asked for a SPECIFIC program's fee, give ONLY that program. If asked "same fee for all?" say "Fee varies by department. Tell me which program."
- CONTINUATION: When the current query refers to the previous answer (e.g. "what is its fee?", "that program's merit?"), resolve "it"/"its"/"that" from the conversation above and answer accordingly.
- Be professional and friendly.

CONTEXT:
{context}"""

    for key_idx in range(num_keys()):
        client = get_client(key_idx)
        for model in MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.2,
                    max_tokens=150
                )
                reply = response.choices[0].message.content
                if not reply or not reply.strip():
                    logger.warning(f"Empty reply from {model}, trying next model...")
                    continue
                reply = reply.strip()
                logger.info(f"LLM reply from {model}: {reply}")
                escalated = any(p in reply.lower() for p in [
                    "technical issue", "cannot find", "unable",
                    "phone number", "provide your phone", "contact you"
                ])
                return reply, escalated
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(f"Key {key_idx+1} rate limited, trying next key...")
                    break
                if "401" in err_str or "invalid" in err_str or "unauthorized" in err_str:
                    logger.warning(f"Key {key_idx+1} invalid, trying next key...")
                    break
                logger.error(f"Model {model} failed: {e}, trying next model...")
                continue

    # All models failed
    logger.error("All models failed")
    return (
        "I'm sorry, I'm having technical difficulties. Please provide your phone number and we will call you back.",
        True
    )