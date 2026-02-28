import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

def transcribe_audio(file_path):
    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            logger.error("GROQ_API_KEY is not set in environment variables!")
            return ""

        client = Groq(api_key=api_key, timeout=45.0)

        # Log file info for debugging
        file_size = os.path.getsize(file_path)
        logger.info(f"Transcribing file: {file_path}, size: {file_size} bytes")

        if file_size < 500:
            logger.warning(f"Audio file too small ({file_size} bytes) — likely empty recording")
            return ""

        with open(file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio),
                model="whisper-large-v3",
                language="en"
            )

        text = transcription.text.strip()
        text = text.replace("ees", "fees")

        logger.info(f"Transcription result: '{text}'")
        return text

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return ""