"""
src/voice/tts.py
-------------------
NEW — Multilingual Voice AI feature.

Turns text into speech using gTTS, so the AI can read interview questions
and the final loan decision aloud in the customer's language. Returns raw
MP3 bytes ready for st.audio(); the Streamlit pages own displaying the
player, this module has no Streamlit dependency.

Degrades the same way as OCR/STT: if gTTS can't reach the network or the
language isn't supported, `.synthesize()` returns None instead of raising,
and the calling page simply skips the audio player.
"""

from __future__ import annotations

import io

from src.utils.logger import get_logger

logger = get_logger(__name__)

# gTTS language codes for our five supported languages happen to match
# our ISO codes exactly (en, hi, te, ta, kn are all valid gTTS codes).
_SUPPORTED_TTS_LANGUAGES = {"en", "hi", "te", "ta", "kn"}


def synthesize(text: str, lang: str = "en") -> bytes | None:
    """Return MP3 audio bytes for `text` spoken in `lang`, or None if
    synthesis isn't possible (unsupported language, empty text, or the
    TTS engine/network is unavailable)."""
    if not text or not text.strip():
        return None

    tts_lang = lang if lang in _SUPPORTED_TTS_LANGUAGES else "en"

    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text=text, lang=tts_lang).write_to_fp(buffer)
        buffer.seek(0)
        logger.info("Synthesized %d chars of speech in lang=%s", len(text), tts_lang)
        return buffer.read()
    except Exception as exc:  # noqa: BLE001 — TTS failure should never crash the page
        logger.warning("gTTS synthesis failed (%s) — audio playback skipped.", exc)
        return None
