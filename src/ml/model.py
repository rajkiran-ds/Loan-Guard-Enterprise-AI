"""
src/ml/model.py
------------------
Inference-only wrapper around the trained XGBoost model. This is what the
decision pipeline (src/decision/rule_engine.py) imports — it never touches
XGBoost or joblib directly, only this class's `.predict()` method. That
keeps the ML model MUT NOT approve loans directly boundary from FEATURE 6
clean: this file only ever returns a probability + risk label, never a
decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from config import (
    ML_FEATURE_COLUMNS,
    MODEL_PATH,
    RISK_LOW_MAX_DEFAULT_PROB,
    RISK_MEDIUM_MAX_DEFAULT_PROB,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAssessment:
    default_probability: float
    risk_category: str  # "Low" | "Medium" | "High"


def categorize_risk(default_probability: float) -> str:
    if default_probability <= RISK_LOW_MAX_DEFAULT_PROB:
        return "Low"
    if default_probability <= RISK_MEDIUM_MAX_DEFAULT_PROB:
        return "Medium"
    return "High"


class RiskModel:
    """Lazy-loading singleton-style wrapper so Streamlit reruns don't
    reload the model from disk on every interaction."""

    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No trained model found at {MODEL_PATH}. "
                    f"Run `python -m src.ml.train_model` first."
                )
            cls._model = joblib.load(MODEL_PATH)
            logger.info("Loaded risk model from %s", MODEL_PATH)
        return cls._model

    @classmethod
    def predict(cls, features: dict) -> RiskAssessment:
        model = cls._load()
        row = pd.DataFrame([{col: features[col] for col in ML_FEATURE_COLUMNS}])
        default_probability = float(model.predict_proba(row)[0, 1])
        risk_category = categorize_risk(default_probability)
        logger.info(
            "Risk prediction: probability=%.4f category=%s", default_probability, risk_category
        )
        return RiskAssessment(default_probability=default_probability, risk_category=risk_category)
