import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Hint Whisper about domain-specific words to reduce mishearings
WHISPER_PROMPT = (
    "IST, Institute of Space Technology, fee structure, hostel charges, "
    "transport, merit, aggregate, admission, BS Electrical Engineering, "
    "BS Aerospace Engineering, BS Avionics Engineering, semester fee, "
    "lakh, rupees, FSC, matric, entry test, closing merit"
)

def transcribe_audio(file_path):
    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            logger.error("GROQ_API_KEY is not set in environment variables!")
            return ""

        client = Groq(api_key=api_key, timeout=45.0)

        file_size = os.path.getsize(file_path)
        logger.info(f"Transcribing file: {file_path}, size: {file_size} bytes")

        if file_size < 500:
            logger.warning(f"Audio file too small ({file_size} bytes) — likely empty recording")
            return ""

        with open(file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio),
                model="whisper-large-v3",
                language="en",
                prompt=WHISPER_PROMPT  # helps Whisper recognize domain words
            )

        text = transcription.text.strip()
        logger.info(f"Transcription result: '{text}'")
        return text

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return ""