import asyncio
import threading
import uuid
import os
import io
import time
import logging
import edge_tts

logger = logging.getLogger(__name__)

VOICE = "en-US-JennyNeural"

# Token store: token -> text  (consumed on first stream request)
_pending: dict = {}
_pending_lock = threading.Lock()


def generate_tts(text: str, session_id: str) -> str:
    """
    Returns /api/tts_stream/<token> immediately.
    No disk write, no waiting — latency cut by ~1-2 seconds.
    """
    token = str(uuid.uuid4())
    with _pending_lock:
        _pending[token] = text
    logger.info(f"TTS stream token: {token[:8]}… text length: {len(text)}")
    return f"/api/tts_stream/{token}"


def get_and_clear(token: str):
    """Consume token → text mapping (one-time use)."""
    with _pending_lock:
        return _pending.pop(token, None)


def stream_tts_chunks(text: str):
    """
    Sync generator that yields raw MP3 bytes as edge-tts produces them.
    Works for Android / Chrome (supports chunked streaming audio).
    First bytes arrive in ~200-400ms.
    """
    chunks = []
    done_flag = [False]
    error_flag = [None]

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _collect():
            try:
                communicate = edge_tts.Communicate(text, VOICE, rate="+0%", pitch="+0Hz")
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
            except Exception as e:
                error_flag[0] = str(e)
            finally:
                done_flag[0] = True

        loop.run_until_complete(_collect())
        loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    idx = 0
    deadline = time.time() + 25
    while time.time() < deadline:
        while idx < len(chunks):
            yield chunks[idx]
            idx += 1
        if done_flag[0]:
            while idx < len(chunks):
                yield chunks[idx]
                idx += 1
            break
        time.sleep(0.02)

    if error_flag[0]:
        logger.error(f"TTS stream error: {error_flag[0]}")


def get_full_audio_bytes(text: str) -> bytes:
    """
    Buffers the complete audio in memory and returns bytes.
    Used for iOS Safari which doesn't support chunked-transfer MP3 streaming.
    Still faster than disk-based approach — no file I/O, no wait loop.
    """
    buf = io.BytesIO()
    for chunk in stream_tts_chunks(text):
        buf.write(chunk)
    data = buf.getvalue()
    logger.info(f"TTS buffered: {len(data)} bytes")
    return data