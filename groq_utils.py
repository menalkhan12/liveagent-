"""Shared Groq client with multi-key fallback. Use GROQ_API_KEYS (comma-separated) or GROQ_API_KEY in env."""
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

def _get_keys():
    keys_str = os.getenv("GROQ_API_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.replace("\n", ",").split(",") if k.strip()]
        if keys:
            return keys
    single = os.getenv("GROQ_API_KEY")
    return [single.strip()] if single and single.strip() else []

GROQ_KEYS = _get_keys()

def get_client(key_index=0):
    """Get Groq client for given key index. Wraps around if index >= len(keys)."""
    if not GROQ_KEYS:
        raise ValueError("No GROQ API keys. Set GROQ_API_KEY or GROQ_API_KEYS in env.")
    idx = key_index % len(GROQ_KEYS)
    # 1 key: allow retries (only option is to wait on 429). 2+ keys: fail fast, try next key.
    max_retries = 2 if len(GROQ_KEYS) == 1 else 0
    return Groq(api_key=GROQ_KEYS[idx], timeout=30.0, max_retries=max_retries)

def num_keys():
    return len(GROQ_KEYS)
