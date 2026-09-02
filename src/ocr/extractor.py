"""
src/ocr/extractor.py
-----------------------
FEATURE 3 — OCR.

Wraps PaddleOCR to extract raw text from uploaded PAN / Aadhaar / salary
slip images, then applies light regex parsing on top to auto-fill the
customer profile fields used by the AI interview (FEATURE 2).

PaddleOCR's model weights are large and download on first use, which is
often unavailable in locked-down/offline environments. To keep the rest
of the app usable even then, `OCRExtractor` degrades gracefully: if
PaddleOCR can't be imported or initialized, `.extract()` returns an empty
result with `engine_available=False` instead of crashing the app, and the
Streamlit UI falls back to manual entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

PAN_PATTERN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
INCOME_PATTERN = re.compile(r"(?:net pay|net salary|take[- ]home)[^\d]{0,15}([\d,]+)", re.IGNORECASE)


@dataclass
class OCRResult:
    raw_text: str = ""
    confidence: float = 0.0
    extracted_fields: dict = field(default_factory=dict)
    engine_available: bool = True


class OCRExtractor:
    _ocr_engine = None
    _init_attempted = False

    @classmethod
    def _get_engine(cls):
        if cls._ocr_engine is not None or cls._init_attempted:
            return cls._ocr_engine
        cls._init_attempted = True
        try:
            from paddleocr import PaddleOCR  # heavy import, done lazily

            cls._ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("PaddleOCR engine initialized.")
        except Exception as exc:  # noqa: BLE001 — any init failure should degrade, not crash
            logger.warning("PaddleOCR unavailable (%s) — OCR will be disabled.", exc)
            cls._ocr_engine = None
        return cls._ocr_engine

    def extract(self, image_path: str | Path) -> OCRResult:
        engine = self._get_engine()
        if engine is None:
            return OCRResult(engine_available=False)

        result = engine.ocr(str(image_path), cls=True)
        lines = []
        confidences = []
        for page in result or []:
            for _box, (text, conf) in page:
                lines.append(text)
                confidences.append(conf)

        raw_text = "\n".join(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        ocr_result = OCRResult(
            raw_text=raw_text,
            confidence=round(avg_confidence, 4),
            extracted_fields=self._parse_fields(raw_text),
            engine_available=True,
        )
        logger.info("OCR extracted %d lines, avg confidence=%.2f", len(lines), avg_confidence)
        return ocr_result

    @staticmethod
    def _parse_fields(raw_text: str) -> dict:
        fields: dict = {}

        pan_match = PAN_PATTERN.search(raw_text.upper())
        if pan_match:
            fields["pan"] = pan_match.group(0)

        aadhaar_match = AADHAAR_PATTERN.search(raw_text)
        if aadhaar_match:
            fields["aadhaar"] = re.sub(r"\s", "", aadhaar_match.group(0))

        income_match = INCOME_PATTERN.search(raw_text)
        if income_match:
            fields["monthly_income"] = float(income_match.group(1).replace(",", ""))

        return fields
