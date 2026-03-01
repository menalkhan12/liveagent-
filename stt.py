import os
import logging
from groq_utils import get_client, num_keys, GROQ_KEYS

logger = logging.getLogger(__name__)

WHISPER_PROMPT = (
    "IST, Institute of Space Technology, fee structure, hostel charges, "
    "transport, merit, aggregate, admission, BS Electrical Engineering, "
    "BS Aerospace Engineering, BS Avionics Engineering, semester fee, "
    "lakh, rupees, FSC, matric, entry test, closing merit"
)

def transcribe_audio(file_path):
    try:
        if not GROQ_KEYS:
            logger.error("GROQ_API_KEY / GROQ_API_KEYS not set")
            return ""

        file_size = os.path.getsize(file_path)
        if file_size < 500:
            logger.warning(f"Audio file too small ({file_size} bytes)")
            return ""

        for key_idx in range(num_keys()):
            try:
                client = get_client(key_idx)
                with open(file_path, "rb") as audio:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(file_path), audio),
                        model="whisper-large-v3",
                        language="en",
                        prompt=WHISPER_PROMPT
                    )
                text = transcription.text.strip()
                logger.info(f"Transcription result: '{text}'")
                return text
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(f"Key {key_idx+1} rate limited, trying next...")
                    continue
                raise

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return ""