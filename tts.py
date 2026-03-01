import asyncio
import threading
import time
import uuid
import os
import logging
import edge_tts

logger = logging.getLogger(__name__)

VOICE = "en-US-JennyNeural"
MIN_FILE_SIZE = 1000  # bytes — any real MP3 is larger than this


def generate_tts(text, session_id):
    try:
        os.makedirs("static/audio", exist_ok=True)
        filename = f"static/audio/{session_id}_{uuid.uuid4()}.mp3"

        async def _run():
            communicate = edge_tts.Communicate(text, VOICE, rate="+0%", pitch="+0Hz")
            await communicate.save(filename)

        result = {"error": None}

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run())
            except Exception as e:
                result["error"] = e
            finally:
                loop.close()

        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join(timeout=28)

        if t.is_alive():
            logger.error("TTS timed out")
            return None

        if result["error"]:
            logger.error(f"TTS Error: {result['error']}")
            return None

        # ── Wait until the file is fully written ──────────────────────────
        # iOS fetches the MP3 immediately and gets 2 bytes if the file
        # isn't ready yet — this loop waits until the file has real content.
        waited = 0.0
        while waited < 3.0:
            if os.path.exists(filename) and os.path.getsize(filename) >= MIN_FILE_SIZE:
                break
            time.sleep(0.05)
            waited += 0.05

        if not os.path.exists(filename):
            logger.error("TTS file not found after wait")
            return None

        final_size = os.path.getsize(filename)
        if final_size < MIN_FILE_SIZE:
            logger.error(f"TTS file too small ({final_size} bytes) — incomplete write")
            return None

        logger.info(f"Generated TTS: {filename} ({final_size} bytes)")
        return "/" + filename

    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None