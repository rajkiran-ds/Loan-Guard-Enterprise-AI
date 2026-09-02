"""
src/utils/language.py
------------------------
NEW — Multilingual Voice AI feature.

Detects which of the five supported languages a piece of customer text is
in, and returns its ISO 639-1 code. Used by the voice-first interview and
the multilingual RAG assistant to decide what language to reply in.

Detection strategy (cheapest and most reliable check first):
1. Unicode script range check — Telugu, Hindi (Devanagari), Tamil, and
   Kannada each use a distinct Unicode block, so a script match is a
   near-certain signal and needs no ML model.
2. `langdetect` for Latin-script text, to tell English apart from a
   transliterated supported language typed in Latin characters.
3. LLM fallback (via Groq, reusing the same client the RAG assistant
   already uses) for short or ambiguous text where steps 1-2 are
   inconclusive — e.g. a two-word answer typed in Latin script.

Never raises: any failure quietly falls back to "en" so a language-
detection hiccup never blocks the loan interview.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
}

# (start, end, iso_code) Unicode block ranges for each non-Latin script.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0C00, 0x0C7F, "te"),  # Telugu
    (0x0900, 0x097F, "hi"),  # Devanagari (Hindi)
    (0x0B80, 0x0BFF, "ta"),  # Tamil
    (0x0C80, 0x0CFF, "kn"),  # Kannada
]


def _detect_by_script(text: str) -> str | None:
    for ch in text:
        code_point = ord(ch)
        for start, end, lang in _SCRIPT_RANGES:
            if start <= code_point <= end:
                return lang
    return None


def _detect_by_langdetect(text: str) -> str | None:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic results
        code = detect(text)
        return code if code in SUPPORTED_LANGUAGES else None
    except Exception as exc:  # noqa: BLE001 — detection must never crash the app
        logger.warning("langdetect unavailable or failed (%s)", exc)
        return None


def _detect_by_llm(text: str) -> str | None:
    """Last-resort fallback for short/ambiguous Latin-script text — asks the
    same Groq LLM the RAG assistant uses to classify the language."""
    try:
        from config import GROQ_API_KEY, GROQ_MODEL
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            return None

        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
        prompt = (
            "Identify the language of this text. Reply with exactly one of "
            "these ISO codes and nothing else: en, te, hi, ta, kn.\n\nText: "
            f"{text}"
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        code = response.content.strip().lower()[:2]
        return code if code in SUPPORTED_LANGUAGES else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM language-detection fallback failed (%s)", exc)
        return None


def detect_language(text: str, default: str = "en") -> str:
    """Return the ISO 639-1 code of `text`'s language, restricted to
    SUPPORTED_LANGUAGES. Falls back to `default` if detection is
    inconclusive or `text` is empty."""
    if not text or not text.strip():
        return default

    script_match = _detect_by_script(text)
    if script_match:
        logger.info("Language detected via script range: %s", script_match)
        return script_match

    langdetect_match = _detect_by_langdetect(text)
    if langdetect_match:
        logger.info("Language detected via langdetect: %s", langdetect_match)
        return langdetect_match

    if len(text.strip()) <= 25:
        llm_match = _detect_by_llm(text)
        if llm_match:
            logger.info("Language detected via LLM fallback: %s", llm_match)
            return llm_match

    logger.info("Language detection inconclusive for %r — defaulting to %s", text, default)
    return default


def language_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, "English")
