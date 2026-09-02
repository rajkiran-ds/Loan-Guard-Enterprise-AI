"""
src/voice/stt.py
-------------------
NEW — Multilingual Voice AI feature.

Speech-to-text for the voice-first interview. Uses faster-whisper (CPU-
friendly CTranslate2 build of Whisper) as the primary engine, matching
the graceful-degradation pattern already used for OCR
(src/ocr/extractor.py): if the engine can't be imported or initialized,
`.transcribe()` returns an empty, `engine_available=False` result instead
of crashing the app, and the UI falls back to typed text.

Whisper's own language identification is used as a first-pass hint, but
the caller (the interview page) re-confirms with src.utils.language for
Telugu/Hindi/Tamil/Kannada/English before deciding the reply language.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Whisper's own ISO codes line up with our SUPPORTED_LANGUAGES codes for
# all five supported languages, so no remapping table is needed.
_WHISPER_MODEL_SIZE = "small"  # good multilingual accuracy/latency trade-off on CPU


@dataclass
class TranscriptionResult:
    text: str = ""
    detected_language: str = ""
    engine_available: bool = True


class SpeechToText:
    _model = None
    _init_attempted = False
    _backend = None  # "faster_whisper" | "openai_whisper"

    @classmethod
    def _get_model(cls):
        if cls._model is not None or cls._init_attempted:
            return cls._model
        cls._init_attempted = True

        try:
            from faster_whisper import WhisperModel

            cls._model = WhisperModel(_WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            cls._backend = "faster_whisper"
            logger.info("faster-whisper model loaded (size=%s).", _WHISPER_MODEL_SIZE)
            return cls._model
        except Exception as exc:  # noqa: BLE001
            logger.warning("faster-whisper unavailable (%s) — trying openai-whisper.", exc)

        try:
            import whisper

            cls._model = whisper.load_model(_WHISPER_MODEL_SIZE)
            cls._backend = "openai_whisper"
            logger.info("openai-whisper model loaded (size=%s).", _WHISPER_MODEL_SIZE)
            return cls._model
        except Exception as exc:  # noqa: BLE001
            logger.warning("openai-whisper also unavailable (%s) — voice input disabled.", exc)
            cls._model = None
            return None

    @classmethod
    def transcribe(cls, audio_bytes: bytes) -> TranscriptionResult:
        """audio_bytes: raw audio (e.g. from st.audio_input, WAV-encoded)."""
        model = cls._get_model()
        if model is None:
            return TranscriptionResult(engine_available=False)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)

            try:
                if cls._backend == "faster_whisper":
                    segments, info = model.transcribe(str(tmp_path))
                    text = " ".join(segment.text for segment in segments).strip()
                    detected_language = info.language or ""
                else:  # openai_whisper
                    result = model.transcribe(str(tmp_path))
                    text = (result.get("text") or "").strip()
                    detected_language = result.get("language") or ""
            except Exception as exc:  # noqa: BLE001 — a bad recording should never crash the page
                logger.warning("Transcription failed (%s).", exc)
                return TranscriptionResult(engine_available=True, text="", detected_language="")

        logger.info("Transcribed %d chars, whisper-detected language=%s", len(text), detected_language)
        return TranscriptionResult(text=text, detected_language=detected_language, engine_available=True)
