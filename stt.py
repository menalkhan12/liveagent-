import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=45.0)

def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3"
            )
        text = transcription.text
        text = text.replace("ees", "fees")
        return text
    except Exception as e:
        logger.error(f"STT Error: {e}")
        return ""