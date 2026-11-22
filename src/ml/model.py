"""
src/ml/model.py
------------------
Inference-only wrapper around the trained PyTorch MLP.

This is what the decision pipeline (src/decision/rule_engine.py) imports —
it never touches PyTorch or joblib directly, only this class's `.predict()`
method. That keeps the ML model boundary from FEATURE 6 clean: this file
only ever returns a probability + risk label, never a decision.

The MLP was trained on standardised features, so inference MUST apply the
same StandardScaler fit during training. Both the model weights and the
scaler are saved by src/ml/train_model.py and loaded lazily here.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np

from config import (
    ML_FEATURE_COLUMNS,
    MLP_MODEL_PATH,
    MLP_SCALER_PATH,
    RISK_LOW_MAX_DEFAULT_PROB,
    RISK_MEDIUM_MAX_DEFAULT_PROB,
)
from src.ml.mlp import load_mlp, predict_mlp
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
    _scaler = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            if not MLP_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No trained MLP found at {MLP_MODEL_PATH}. "
                    f"Run `python -m src.ml.train_model` first."
                )
            if not MLP_SCALER_PATH.exists():
                raise FileNotFoundError(
                    f"No scaler found at {MLP_SCALER_PATH}. "
                    f"Run `python -m src.ml.train_model` first."
                )
            cls._scaler = joblib.load(MLP_SCALER_PATH)
            cls._model = load_mlp(
                n_features=len(ML_FEATURE_COLUMNS),
                path=str(MLP_MODEL_PATH),
            )
            logger.info("Loaded MLP + scaler from %s", MLP_MODEL_PATH.parent)
        return cls._model, cls._scaler

    @classmethod
    def predict(cls, features: dict) -> RiskAssessment:
        model, scaler = cls._load()

        # Build a single-row feature matrix in the exact order the model
        # expects (config.ML_FEATURE_COLUMNS).
        x = np.array(
            [[float(features[col]) for col in ML_FEATURE_COLUMNS]],
            dtype=np.float32,
        )

        # Apply the SAME scaling used during training — critical for the
        # MLP to produce the same predictions it produced on the val set.
        x_scaled = scaler.transform(x).astype(np.float32)

        default_probability = float(predict_mlp(model, x_scaled)[0])
        risk_category = categorize_risk(default_probability)

        logger.info(
            "Risk prediction: probability=%.4f category=%s",
            default_probability,
            risk_category,
        )
        return RiskAssessment(
            default_probability=default_probability,
            risk_category=risk_category,
        )