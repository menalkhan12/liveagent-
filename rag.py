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
    "llama-3.3-70b-versatile",   # Fallback if 8B fails
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


MAX_CONTEXT_CHARS = 12000  # Increased slightly


# ─── Keyword → file mapping ───────────────────────────────────────────────────
# If ANY keyword matches the query, the listed files are ALWAYS injected first.
KEYWORD_FILE_MAP = [
    # Closing merit / last merit / aggregate history
    (
        ["closing merit", "last merit", "last year merit", "last year closing",
         "previous merit", "closing aggregate", "what was merit", "merit history",
         "merit 2024", "merit 2023"],
        ["CLOSING_MERIT_HISTORY.txt", "RAG_QUICK_REF.txt"]
    ),
    # Fee / cost / charges
    (
        ["fee structure", "fee of", "fees of", "fee for", "fees for",
         "tuition fee", "semester fee", "how much fee", "fee avionics",
         "fee aerospace", "fee electrical", "fee mechanical", "fee computer",
         "fee space", "fee mathematics", "fee physics", "fee biotechnology",
         "fee materials", "fee software", "cost of", "charges"],
        ["FEE_STRUCTURE.txt", "RAG_QUICK_REF.txt"]
    ),
    # Department programs
    (
        ["programs under", "program under", "programs offered", "programs in",
         "offered by", "offered under", "department offer", "department programs",
         "what programs", "which programs", "courses under", "under mechanical",
         "under electrical", "under avionics", "under aerospace", "under computing",
         "under materials", "under space"],
        ["IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt", "RAG_QUICK_REF.txt",
         "ELECTRICAL_DEPARTMENT_PROGRAMS.txt"]
    ),
    # Admission dates / open / closed / last date
    (
        ["admission open", "admissions open", "when do admission", "last date",
         "last date to apply", "admission close", "admission deadline",
         "when to apply", "application deadline", "admission date",
         "when start admission", "when open admission"],
        ["ADMISSION_DATES_AND_STATUS.txt", "ADMISSION_INFO.txt"]
    ),
    # Merit calculation / aggregate formula / will I get admission
    (
        ["calculate merit", "calculate aggregate", "my aggregate", "my merit",
         "will i get admission", "what is aggregate", "aggregate formula",
         "merit formula", "merit calculation", "calculate my"],
        ["MERIT_CRITERIA_AND_AGGREGATE.txt"]
    ),
    # Transport / bus
    (
        ["transport", "bus", "shuttle", "pick and drop", "route"],
        ["TRANSPORT_HOSTEL_FAQS.txt"]
    ),
    # Hostel / accommodation
    (
        ["hostel", "accommodation", "boarding", "dorm", "room", "stay on campus"],
        ["TRANSPORT_HOSTEL_FAQS.txt", "ADMISSION_FAQS_COMPLETE.txt"]
    ),
    # Eligibility / who can apply / DAE / pre-medical / ICS
    (
        ["eligible", "eligibility", "can i apply", "who can apply",
         "pre-medical", "pre medical", "ics student", "dae", "a-level",
         "a level", "o-level", "o level"],
        ["PROGRAMS_ELIGIBILITY_WHATSAPP_SEATS.txt", "ADMISSION_FAQS_COMPLETE.txt"]
    ),
    # Diploma confusion — handle firmly
    (
        ["diploma", "diplomas"],
        ["PROGRAMS_ELIGIBILITY_WHATSAPP_SEATS.txt", "IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt"]
    ),
    # Foreign / international
    (
        ["foreign", "international student", "overseas", "non-pakistani",
         "other country", "from abroad", "study in pakistan"],
        ["FOREIGN_ADMISSION.txt"]
    ),
    # Scholarships / financial aid
    (
        ["scholarship", "financial aid", "need based", "merit based scholarship",
         "honhaar", "peef", "fee waiver"],
        ["ADMISSION_FAQS_COMPLETE.txt", "TRANSPORT_HOSTEL_FAQS.txt"]
    ),
    # Announcements / challan / fee submission deadline
    (
        ["challan", "fee submission", "last date to submit fee",
         "submit fee", "fee deadline", "challan deadline"],
        ["ANNOUNCEMENTS.txt", "FEE_STRUCTURE.txt"]
    ),
    # Space science closing merit specifically
    (
        ["space science merit", "space science closing", "merit space science",
         "bs space science merit"],
        ["CLOSING_MERIT_HISTORY.txt", "RAG_QUICK_REF.txt"]
    ),
]
# ──────────────────────────────────────────────────────────────────────────────


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
        "iesp": "IST",
        "i.e.s.p": "IST",
    }
    lower = text.lower()
    for wrong, correct in replacements.items():
        lower = lower.replace(wrong, correct)
    return lower


def _get_forced_files_for_query(query_lower):
    """Return list of filenames that must be injected based on keyword matching."""
    forced = []
    seen = set()
    for keywords, files in KEYWORD_FILE_MAP:
        if any(kw in query_lower for kw in keywords):
            for f in files:
                if f not in seen:
                    forced.append(f)
                    seen.add(f)
    return forced


def _get_chunks_from_files(filenames):
    """Return concatenated chunks from specific files."""
    file_set = set(filenames)
    parts = []
    total = 0
    for idx, name in enumerate(doc_names):
        if name in file_set:
            chunk = f"[{name}]\n{documents[idx]}"
            if total + len(chunk) <= MAX_CONTEXT_CHARS - 1000:
                parts.append(chunk)
                total += len(chunk)
    return "\n\n---\n\n".join(parts) if parts else ""


def _expand_query_for_retrieval(query):
    q = query.lower()
    extra = []
    if any(w in q for w in ["cost", "price", "tuition", "fee", "fees"]):
        extra.append("fee structure tuition semester charges rupees lakh thousand")
    if any(w in q for w in ["avionics"]) and any(w in q for w in ["fee", "cost"]):
        extra.append("avionics 1 lakh 48 thousand fee per semester")
    if any(w in q for w in ["aerospace"]) and any(w in q for w in ["fee", "cost"]):
        extra.append("aerospace 1 lakh 48 thousand fee per semester")
    if any(w in q for w in ["space science", "physics", "mathematics", "biotechnology"]) and any(w in q for w in ["fee", "cost"]):
        extra.append("BS Space Science BS Physics 1 lakh 2 thousand semester")
    if any(w in q for w in ["merit", "closing", "aggregate", "calculate", "last year"]):
        extra.append("merit aggregate closing 2024 2023 Computer Science Space Science engineering")
    if any(w in q for w in ["computer science", "computer engineering"]) and any(w in q for w in ["merit", "closing"]):
        extra.append("BS Computer Science 77.4 closing merit 2024")
    if any(w in q for w in ["space science"]) and any(w in q for w in ["merit", "closing"]):
        extra.append("BS Space Science 69.8 closing merit 2024")
    if any(w in q for w in ["electrical"]) and any(w in q for w in ["program", "offer", "department"]):
        extra.append("electrical engineering department programs BS Computer Engineering")
    if any(w in q for w in ["mechanical"]) and any(w in q for w in ["department", "program", "offer"]):
        extra.append("Mechanical Engineering department BS Mechanical Engineering only")
    if any(w in q for w in ["hostel", "accommodation", "boarding"]):
        extra.append("hostel charges accommodation fee 45000")
    if any(w in q for w in ["transport", "bus", "shuttle"]):
        extra.append("transport bus shuttle routes fee charges 03000544707")
    if any(w in q for w in ["eligibility", "eligible", "whatsapp", "seats", "focus", "dae"]):
        extra.append("eligibility WhatsApp seats focus areas DAE")
    if any(w in q for w in ["foreign", "international", "overseas"]):
        extra.append("foreign admission international USD fee eligibility NOC HEC")
    if extra:
        return query + " " + " ".join(extra)
    return query


def retrieve_context(query, top_k=12):
    global vectorizer, doc_vectors

    if vectorizer is None or doc_vectors is None:
        return ""

    try:
        q_lower = query.lower()

        # Step 1: Always inject forced files based on keyword matching
        forced_files = _get_forced_files_for_query(q_lower)
        forced_context = ""
        if forced_files:
            forced_context = _get_chunks_from_files(forced_files)
            logger.info(f"Forced files injected: {forced_files}")

        # Step 2: TF-IDF retrieval for additional context
        expanded = _expand_query_for_retrieval(query)
        query_vec = vectorizer.transform([expanded])
        similarities = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Track which files are already in forced context
        forced_file_set = set(forced_files)

        tfidf_parts = []
        total_len = len(forced_context)
        sim_threshold = 0.005  # Very low — we rely on forced injection for accuracy

        for idx in top_indices:
            if similarities[idx] <= sim_threshold:
                continue
            # Skip if this file was already injected via forced path
            if doc_names[idx] in forced_file_set:
                continue
            chunk = f"[{doc_names[idx]}]\n{documents[idx]}"
            if total_len + len(chunk) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total_len - 80
                if remain > 400:
                    chunk = chunk[:remain] + "\n...[truncated]"
                    tfidf_parts.append(chunk)
                break
            tfidf_parts.append(chunk)
            total_len += len(chunk)

        tfidf_context = "\n\n---\n\n".join(tfidf_parts)

        # Combine: forced files first (most reliable), then TF-IDF extras
        if forced_context and tfidf_context:
            final_context = forced_context + "\n\n---\n\n" + tfidf_context
        elif forced_context:
            final_context = forced_context
        else:
            final_context = tfidf_context

        logger.info(f"Context length: {len(final_context)} chars, forced_files: {forced_files}")
        return final_context

    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""


def _is_thanks_or_compliment(query):
    q = query.lower().strip()
    thanks = any(x in q for x in ["thank", "thanks", "thx", "ok thanks", "okay thanks", "ty "])
    compliment = any(x in q for x in ["you're good", "youre good", "great job", "well done",
                                       "you're helpful", "youre helpful", "good job", "nice ",
                                       "awesome", "excellent"])
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

    # Resolve continuation using recent conversation
    user_message = query
    retrieval_query = query
    if conversation_history:
        hist_str = "\n".join([f"User: {u}\nAgent: {a}" for u, a in conversation_history])
        user_message = f"""Previous conversation:
{hist_str}

Current query (resolve "it"/"its"/"their"/"them"/"that" from above): {query}"""
        last_user, last_agent = conversation_history[-1] if conversation_history else ("", "")
        continuation_words = ["its", "their", "them", "they", "that", "those", "it", "this",
                               "same", "can you tell", "please tell"]
        is_continuation = any(w in query.lower() for w in continuation_words)
        if is_continuation and (last_agent or last_user):
            retrieval_query = f"{query} {last_user} {last_agent}"

    context = retrieve_context(retrieval_query)

    if not context.strip():
        return (
            "I don't have that information. Please provide your phone number and we will contact you.",
            True
        )

    system_prompt = f"""You are the official voice assistant for Institute of Space Technology (IST). You answer callers by phone.

STRICT RULES:
1. Answer ONLY from the CONTEXT below. NEVER invent or add information not in CONTEXT.
2. DIPLOMA RULE (CRITICAL): IST does NOT offer any diploma programs. IST ONLY accepts DAE (Diploma of Associate Engineering) holders as applicants. When asked "what diplomas does IST offer", say: "IST does not offer diploma programs. IST is a university offering BS, MS, and PhD degrees. However, DAE holders can apply for BS programs at IST." Never list DAE specializations as IST's diploma offerings.
3. Electrical Engineering programs: ONLY "BS Electrical Engineering and BS Computer Engineering". Never mention others.
4. Mechanical Engineering department: ONLY "BS Mechanical Engineering". Never add other programs.
5. CLOSING MERIT: When CONTEXT has closing merit (e.g. BS Computer Science 2024: 77.4), give that number. "Last year" = 2024.
6. FEE: When CONTEXT has fee for a program (e.g. Avionics 1 lakh 48 thousand), give it directly.
7. Answer DIRECTLY with facts. Never say "check the website" or "visit the website".
8. Keep responses 1-3 short sentences, conversational for phone.
9. Use amounts in lakh and thousand (e.g., 1 lakh 48 thousand rupees).
10. AGGREGATE FORMULA (Engineering): (Matric/1100 × 10) + (FSC/1100 × 40) + (EntryTest/100 × 50).
11. AGGREGATE FORMULA (Non-Engineering): (Matric/1100 × 50) + (FSC/1100 × 50). No entry test.
12. If CONTEXT truly does not contain the answer, say: "I don't have that information. Please provide your phone number and we will contact you."
13. CONTINUATION: Resolve "it"/"its"/"their"/"them"/"that" from the previous turn context.
14. Be professional and friendly.

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
                    temperature=0.1,
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