import asyncio
import threading
import uuid
import os
import logging
import edge_tts

logger = logging.getLogger(__name__)

# Natural neural voice (Microsoft Edge TTS)
VOICE = "en-US-JennyNeural"

def generate_tts(text, session_id):
    try:
        os.makedirs("static/audio", exist_ok=True)
        filename = f"static/audio/{session_id}_{uuid.uuid4()}.mp3"

        async def _run():
            communicate = edge_tts.Communicate(text, VOICE, rate="+0%", pitch="+0Hz")
            await communicate.save(filename)

        # Gunicorn workers may already have a running event loop.
        # Always spin up a fresh loop in a dedicated thread to avoid conflicts.
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
        t.join(timeout=25)  # Don't wait forever

        if t.is_alive():
            logger.error("TTS timed out")
            return None

        if result["error"]:
            logger.error(f"TTS Error: {result['error']}")
            return None

        if os.path.exists(filename):
            logger.info(f"Generated TTS: {filename}")
            return "/" + filename

        return None

    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None