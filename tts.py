import asyncio
import uuid
import os
import logging

logger = logging.getLogger(__name__)

# Natural neural voice (Microsoft Edge TTS)
VOICE = "en-US-JennyNeural"

def _tts_edge(text, filename):
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, VOICE, rate="+0%", pitch="+0Hz")
        await communicate.save(filename)
    asyncio.run(_run())

def _tts_gtts(text, filename):
    from gtts import gTTS
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)

def generate_tts(text, session_id):
    try:
        os.makedirs("static/audio", exist_ok=True)
        filename = f"static/audio/{session_id}_{uuid.uuid4()}.mp3"
        try:
            _tts_edge(text, filename)
        except Exception as e:
            logger.warning(f"edge-tts failed ({e}), using gTTS fallback")
            _tts_gtts(text, filename)
        logger.info(f"Generated TTS: {filename}")
        return "/" + filename
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None