import os
import logging
from groq_utils import get_client, num_keys, GROQ_KEYS

logger = logging.getLogger(__name__)

# Whisper uses this prompt to bias recognition toward domain-specific vocabulary.
WHISPER_PROMPT = (
    "IST, Institute of Space Technology, Islamabad. "
    "fee structure, fee of electrical engineering, fee of aerospace engineering, "
    "fee of avionics engineering, fee of mechanical engineering, fee of computer engineering, "
    "fee of software engineering, fee of computer science, fee of space science, "
    "fee of mathematics, fee of biotechnology, fee of materials science, "
    "fee of data science, fee of artificial intelligence, semester fee, per semester fee, "
    "how much is the fee, total fee, annual fee, "
    "merit based scholarship, need based scholarship, financial aid, scholarship at IST, "
    "does IST offer scholarship, is scholarship available, "
    "closing merit, last year merit, merit 2024, merit 2023, merit aggregate, "
    "calculate merit, aggregate formula, merit calculation, "
    "BS Electrical Engineering, BS Computer Engineering, BS Aerospace Engineering, "
    "BS Avionics Engineering, BS Mechanical Engineering, BS Software Engineering, "
    "BS Computer Science, BS Space Science, BS Physics, BS Mathematics, "
    "BS Data Science, BS Artificial Intelligence, BS Biotechnology, "
    "BS Materials Science and Engineering, "
    "MS Aerospace, MS Electrical, MS Computer Science, PhD program, "
    "Electrical Engineering department, Computing department, "
    "Aeronautics and Astronautics department, Avionics department, "
    "Mechanical Engineering department, Space Science department, "
    "Applied Mathematics department, Humanities department, "
    "admission open, admission closed, last date to apply, application deadline, "
    "entry test, FSc, matric, pre-engineering, pre-medical, ICS, DAE, A-level, "
    "hostel, hostel charges, hostel fee, accommodation, transport, bus route, "
    "lakh, thousand rupees, Pakistani rupees, "
    "research lab, Space Systems Lab, Astronomy Resource Center, telescope, "
    "NCFA, remote sensing, CubeSat, iCube, "
    "programs offered, departments at IST, how many departments, "
    "eligibility criteria, who can apply, can I apply, "
    "what is IST, about IST, IST history, IST location, IST accredited"
)

# Minimum file size to attempt transcription (bytes)
MIN_AUDIO_BYTES = 1000


def transcribe_audio(file_path):
    try:
        if not GROQ_KEYS:
            logger.error("GROQ_API_KEY / GROQ_API_KEYS not set")
            return ""

        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return ""

        file_size = os.path.getsize(file_path)
        logger.info(f"Audio file size: {file_size} bytes, path: {file_path}")

        if file_size < MIN_AUDIO_BYTES:
            logger.warning(f"Audio file too small ({file_size} bytes) — likely silence or mic not captured. Skipping.")
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
                logger.error(f"STT error (key {key_idx+1}): {e}")  # Always log full error

                if "400" in err_str or "bad request" in err_str:
                    # 400 = audio file is empty, corrupt, or unsupported format
                    # This is NOT a key problem — no point trying other keys
                    logger.error(
                        f"Groq returned 400 Bad Request. "
                        f"Audio file is likely empty, too short, or in wrong format. "
                        f"File size: {file_size} bytes"
                    )
                    return ""

                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(f"Key {key_idx+1} rate limited, trying next...")
                    continue

                if "401" in err_str or "invalid" in err_str or "unauthorized" in err_str:
                    logger.warning(f"Key {key_idx+1} invalid/unauthorized, trying next...")
                    continue

                logger.error(f"Unknown STT error on key {key_idx+1}: {e}")
                continue

        logger.error("All keys exhausted for STT")
        return ""

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return ""