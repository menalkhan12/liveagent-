import os
import re
import json
import uuid
import threading
import logging
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv

from rag import generate_answer, initialize_rag
from stt import transcribe_audio
from tts import generate_tts, get_and_clear, stream_tts_chunks, get_full_audio_bytes
from livekit_utils import generate_livekit_token
from utils import (
    init_call_record,
    update_call_record,
    end_call_record,
    append_lead_log,
    detect_phone_number,
    get_last_user_query,
    get_recent_turns,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
log_lock = threading.Lock()

try:
    initialize_rag()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")


def _is_ios(user_agent: str) -> bool:
    ua = (user_agent or "").lower()
    # iPhone, iPad, iPod — also catch iOS Chrome (CriOS) and Firefox (FxiOS)
    return any(x in ua for x in ("iphone", "ipad", "ipod", "crios", "fxios"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "OK"}), 200


# ── TTS Stream endpoint ────────────────────────────────────────────────────
@app.route("/api/tts_stream/<token>")
def tts_stream(token):
    """
    Streams TTS audio to the client.

    Android/Chrome: true chunked streaming — first bytes in ~300ms.
    iOS Safari:     buffers full audio in memory then sends as one response.
                    (iOS doesn't support chunked-transfer MP3 reliably)

    Either way this is faster than the old disk-write approach.
    """
    text = get_and_clear(token)
    if not text:
        logger.warning(f"TTS token not found: {token[:8]}")
        return Response("Token not found or expired", status=404)

    ua = request.headers.get("User-Agent", "")
    ios = _is_ios(ua)

    if ios:
        # Buffer entire audio in memory, send as complete response
        # Fast because: no disk I/O, no file-write wait loop
        logger.info(f"TTS buffered mode (iOS): token {token[:8]}")
        try:
            audio_bytes = get_full_audio_bytes(text)
            if not audio_bytes:
                return Response("TTS generation failed", status=500)
            return Response(
                audio_bytes,
                mimetype="audio/mpeg",
                headers={
                    "Content-Length": str(len(audio_bytes)),
                    "Cache-Control": "no-cache",
                    "Accept-Ranges": "bytes",  # iOS needs this for seeking/playback
                }
            )
        except Exception as e:
            logger.error(f"iOS TTS buffer error: {e}")
            return Response("TTS error", status=500)
    else:
        # True streaming for Android/Chrome — audio starts playing immediately
        logger.info(f"TTS streaming mode (non-iOS): token {token[:8]}")
        def generate():
            try:
                for chunk in stream_tts_chunks(text):
                    yield chunk
            except Exception as e:
                logger.error(f"TTS chunk error: {e}")

        return Response(
            stream_with_context(generate()),
            mimetype="audio/mpeg",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Content-Type-Options": "nosniff",
                "Accept-Ranges": "none",
            }
        )


@app.route("/api/start_call", methods=["POST"])
def start_call():
    try:
        session_id = str(uuid.uuid4())
        room_name = f"room_{session_id}"
        init_call_record(session_id)
        token = generate_livekit_token(room_name, session_id)
        greeting_text = "Hello, this is Institute of Space Technology. How can I help you today?"
        audio_url = generate_tts(greeting_text, session_id)
        logger.info(f"Call started: {session_id}")
        return jsonify({
            "session_id": session_id,
            "livekit_token": token,
            "room_name": room_name,
            "greeting_audio_url": audio_url
        }), 200
    except Exception as e:
        logger.error(f"Error starting call: {e}")
        return jsonify({"error": "Failed to start call"}), 500


@app.route("/api/query", methods=["POST"])
def query():
    try:
        session_id = request.form.get("session_id")
        audio_file = request.files.get("audio")

        if not session_id or not audio_file:
            return jsonify({"error": "Invalid request"}), 400

        # Detect correct extension for Groq Whisper — wrong ext = transcription failure
        # iOS Safari → audio/mp4 (.mp4/.m4a)
        # Android/Chrome → audio/webm (.webm)
        # Some browsers → audio/wav (.wav) or audio/ogg (.ogg)
        filename = audio_file.filename or ""
        content_type = audio_file.content_type or ""

        if filename.endswith(".wav") or "wav" in content_type:
            ext = "wav"
        elif filename.endswith(".mp4") or filename.endswith(".m4a") or "mp4" in content_type or "m4a" in content_type:
            ext = "mp4"
        elif filename.endswith(".ogg") or "ogg" in content_type:
            ext = "ogg"
        else:
            ext = "webm"  # default for Chrome/Android

        temp_path = f"temp_{uuid.uuid4()}.{ext}"
        audio_file.save(temp_path)

        try:
            user_text = transcribe_audio(temp_path)
            logger.info(f"User query [{session_id}]: {user_text}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not user_text or len(user_text.strip()) < 3:
            return jsonify({
                "error": "Could not hear you clearly. Please speak again.",
                "user_text": user_text or ""
            }), 200

        # Phone number
        phone = detect_phone_number(user_text)
        if phone:
            unanswered_query = get_last_user_query(session_id) or user_text
            append_lead_log(session_id, phone, unanswered_query)
            reply = "Thank you. Our admissions office will contact you soon."
            audio_url = generate_tts(reply, session_id)
            update_call_record(session_id, user_text, reply, escalated=True, phone=phone)
            logger.info(f"Phone detected: {phone}")
            return jsonify({"user_text": user_text, "response": reply, "audio_url": audio_url}), 200

        # RAG
        history = get_recent_turns(session_id, n=5)
        reply, escalated = generate_answer(user_text, conversation_history=history)
        audio_url = generate_tts(reply, session_id)
        update_call_record(session_id, user_text, reply, escalated=escalated)
        logger.info(f"Agent response [{session_id}]: {reply} (escalated: {escalated})")

        return jsonify({"user_text": user_text, "response": reply, "audio_url": audio_url}), 200

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "Failed to process query"}), 500


@app.route("/api/end_call", methods=["POST"])
def end_call():
    try:
        session_id = request.json.get("session_id")
        if not session_id:
            return jsonify({"error": "Invalid request"}), 400
        end_call_record(session_id)
        logger.info(f"Call ended: {session_id}")
        return jsonify({"status": "ended"}), 200
    except Exception as e:
        logger.error(f"Error ending call: {e}")
        return jsonify({"error": "Failed to end call"}), 500


def _split_sentences(text):
    """Split reply into sentences for streaming TTS — reduces perceived latency."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


@app.route("/api/query_stream", methods=["POST"])
def query_stream():
    """
    SSE endpoint for Android/Desktop:
    1. Transcribes audio
    2. Gets RAG reply
    3. Splits into sentences, generates TTS per sentence
    4. Streams SSE events: transcript → sentence(s) → done
    Client plays sentence 1 audio while sentence 2 TTS is still generating.
    """
    session_id = request.form.get("session_id")
    audio_file  = request.files.get("audio")

    if not session_id or not audio_file:
        return jsonify({"error": "Invalid request"}), 400

    filename     = audio_file.filename or ""
    content_type = audio_file.content_type or ""
    if filename.endswith(".wav") or "wav" in content_type:
        ext = "wav"
    elif filename.endswith(".mp4") or filename.endswith(".m4a") or "mp4" in content_type:
        ext = "mp4"
    elif filename.endswith(".ogg") or "ogg" in content_type:
        ext = "ogg"
    else:
        ext = "webm"

    temp_path = f"temp_{uuid.uuid4()}.{ext}"
    audio_file.save(temp_path)

    try:
        user_text = transcribe_audio(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def generate():
        try:
            if not user_text or len(user_text.strip()) < 3:
                yield f"data: {json.dumps({'type': 'error', 'text': 'Could not hear you clearly. Please try again.'})}

"
                return

            yield f"data: {json.dumps({'type': 'transcript', 'text': user_text})}

"

            # Phone number check
            from utils import detect_phone_number, get_last_user_query, append_lead_log, get_recent_turns, update_call_record
            phone = detect_phone_number(user_text)
            if phone:
                unanswered_query = get_last_user_query(session_id) or user_text
                append_lead_log(session_id, phone, unanswered_query)
                reply = "Thank you. Our admissions office will contact you soon."
                update_call_record(session_id, user_text, reply, escalated=True, phone=phone)
                audio_url = generate_tts(reply, session_id)
                yield f"data: {json.dumps({'type': 'sentence', 'text': reply, 'audio_url': audio_url})}

"
                yield f"data: {json.dumps({'type': 'done'})}

"
                return

            history = get_recent_turns(session_id, n=5)
            reply, escalated = generate_answer(user_text, conversation_history=history)
            update_call_record(session_id, user_text, reply, escalated=escalated)
            logger.info(f"[stream] Agent reply [{session_id}]: {reply}")

            # Split into sentences and stream TTS per sentence
            sentences = _split_sentences(reply)
            for sentence in sentences:
                audio_url = generate_tts(sentence, session_id)
                yield f"data: {json.dumps({'type': 'sentence', 'text': sentence, 'audio_url': audio_url})}

"

            yield f"data: {json.dumps({'type': 'done'})}

"

        except Exception as e:
            logger.error(f"query_stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': 'Server error'})}

"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering — critical for SSE
            "Connection": "keep-alive",
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV") == "development")