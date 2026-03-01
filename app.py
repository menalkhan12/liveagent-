import os
import uuid
import json
import threading
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from rag import generate_answer, initialize_rag
from stt import transcribe_audio
from tts import generate_tts
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
log_lock = threading.Lock()

# Initialize RAG at startup
try:
    initialize_rag()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "OK"}), 200

@app.route("/api/start_call", methods=["POST"])
def start_call():
    try:
        session_id = str(uuid.uuid4())
        room_name = f"room_{session_id}"
        
        init_call_record(session_id)
        
        token = generate_livekit_token(room_name, session_id)
        
        greeting_text = "Hello, this is Institute of Space Technology. How can I help you today?"
        audio_url = generate_tts(greeting_text, session_id)
        
        logger.info(f"Call started with session_id: {session_id}")
        
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
        
        # Save temporary audio file (supports webm from mic and wav from VAD barge-in)
        ext = "wav" if audio_file.filename and audio_file.filename.endswith(".wav") else "webm"
        temp_path = f"temp_{uuid.uuid4()}.{ext}"
        audio_file.save(temp_path)
        
        try:
            # Transcribe audio
            user_text = transcribe_audio(temp_path)
            logger.info(f"User query [{session_id}]: {user_text}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Skip empty or very short transcription
        if not user_text or len(user_text.strip()) < 3:
            return jsonify({
                "error": "Could not hear you clearly. Please speak again.",
                "user_text": user_text or ""
            }), 200
        
        # Check if user provided phone number
        phone = detect_phone_number(user_text)
        if phone:
            unanswered_query = get_last_user_query(session_id) or user_text
            append_lead_log(session_id, phone, unanswered_query)
            reply = "Thank you. Our admissions office will contact you soon."
            audio_url = generate_tts(reply, session_id)
            update_call_record(session_id, user_text, reply, escalated=True, phone=phone)
            logger.info(f"Phone number detected: {phone}")
            payload = {"user_text": user_text, "response": reply}
            if audio_url:
                payload["audio_url"] = audio_url
            return jsonify(payload), 200
        
        # Generate answer using RAG (with conversation history for continuity)
        history = get_recent_turns(session_id, n=5)
        reply, escalated = generate_answer(user_text, conversation_history=history)
        
        # Generate TTS response
        audio_url = generate_tts(reply, session_id)
        update_call_record(session_id, user_text, reply, escalated=escalated)
        
        logger.info(f"Agent response [{session_id}]: {reply} (escalated: {escalated})")
        
        payload = {"user_text": user_text, "response": reply}
        if audio_url:
            payload["audio_url"] = audio_url
        return jsonify(payload), 200
    
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV") == "development")