import asyncio
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

        asyncio.run(_run())
        logger.info(f"Generated TTS: {filename}")
        return "/" + filename
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None