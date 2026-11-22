"""
src/ml/train_model.py
------------------------
FEATURE 5 — ML Risk Model (PyTorch MLP).

Trains a PyTorch MLP classifier on Loan_Status, evaluates it on a held-out
test set (ROC AUC, precision, recall, F1, confusion matrix, classification
report), and persists the model + scaler + metrics to src/ml/artifacts/.

This script is the one and only place the model is trained — src/ml/predict.py
and src/ml/model.py only ever load what this script saves.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    FEATURE_SCHEMA_PATH,
    ML_FEATURE_COLUMNS,
    ML_RANDOM_STATE,
    ML_TARGET_COLUMN,
    ML_TEST_SIZE,
    MLP_MODEL_PATH,
    MLP_SCALER_PATH,
    MODEL_METRICS_PATH,
)
from src.ml.generate_sample_dataset import generate_dataset
from src.ml.mlp import predict_mlp, save_mlp, train_mlp
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_training_data() -> pd.DataFrame:
    if DATASET_PATH.exists():
        logger.info("Loading dataset from %s", DATASET_PATH)
        df = pd.read_csv(DATASET_PATH)
    else:
        logger.warning(
            "No dataset found at %s — generating a synthetic one.", DATASET_PATH
        )
        df = generate_dataset()
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)

    missing = set(ML_FEATURE_COLUMNS + [ML_TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"If you loaded the raw Kaggle CSV, rename its columns to match "
            f"config.ML_FEATURE_COLUMNS first."
        )
    return df


def _encode_target(y_raw: pd.Series) -> np.ndarray:
    """
    Return a 0/1 numpy array regardless of whether the CSV stores ints or
    strings ('Y'/'N', 'Yes'/'No', 'Approved'/'Rejected', etc.).

    Convention: index 1 is the positive class ("approved / no-default").
    Sorting alphabetically means 'N' < 'Y' and 'No' < 'Yes', which gives
    the correct polarity. Watch the logged mapping if your labels aren't
    alphabetical in that direction.
    """
    if y_raw.dtype == object:
        uniques = sorted(y_raw.dropna().unique())
        if len(uniques) != 2:
            raise ValueError(f"Expected binary target, got {uniques}")
        logger.info("Mapping target: %r → 0, %r → 1", uniques[0], uniques[1])
        return (y_raw == uniques[1]).astype(int).values
    return y_raw.fillna(0).astype(int).values


def train() -> dict:
    df = load_training_data()

    X = (
        df[ML_FEATURE_COLUMNS]
        .fillna(df[ML_FEATURE_COLUMNS].median())
        .values.astype(np.float32)
    )
    y = _encode_target(df[ML_TARGET_COLUMN])

    # ------------------------------------------------------------------
    # 1) Held-out test split — never touched until final evaluation
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # 2) Validation split from train — used for early stopping only
    # ------------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=ML_RANDOM_STATE,
        stratify=y_train,
    )

    # ------------------------------------------------------------------
    # 3) Scale features — fit on train split only, apply to val/test
    #    (fitting on train prevents leakage into validation statistics)
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ------------------------------------------------------------------
    # 4) Train the MLP with early stopping
    # ------------------------------------------------------------------
    logger.info(
        "Training PyTorchMLP — train=%d, val=%d, test=%d",
        len(X_tr_s),
        len(X_val_s),
        len(X_test_s),
    )
    model = train_mlp(X_tr_s, y_tr, X_val_s, y_val)

    # ------------------------------------------------------------------
    # 5) Evaluate on the untouched test set
    # ------------------------------------------------------------------
    y_proba = predict_mlp(model, X_test_s)
    y_pred = (y_proba >= 0.5).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Default/Rejected (0)", "No-default/Approved (1)"],
        digits=4,
        zero_division=0,
    )

    metrics = {
        "model": "PyTorchMLP",
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(
            float(f1_score(y_test, y_pred, zero_division=0)), 4
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
        "n_train": int(len(X_tr_s)),
        "n_val": int(len(X_val_s)),
        "n_test": int(len(X_test_s)),
    }

    # ------------------------------------------------------------------
    # 6) Persist artifacts
    # ------------------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_mlp(model, str(MLP_MODEL_PATH))
    joblib.dump(scaler, MLP_SCALER_PATH)
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    FEATURE_SCHEMA_PATH.write_text(
        json.dumps({"features": ML_FEATURE_COLUMNS}, indent=2)
    )

    logger.info("MLP weights saved → %s", MLP_MODEL_PATH)
    logger.info("Scaler saved → %s", MLP_SCALER_PATH)
    logger.info("Metrics: AUC=%.4f F1=%.4f", metrics["roc_auc"], metrics["f1_score"])
    return metrics


if __name__ == "__main__":
    result = train()
    # Print everything except the multi-line classification report as JSON,
    # then print the report cleanly at the end.
    printable = {k: v for k, v in result.items() if k != "classification_report"}
    print(json.dumps(printable, indent=2))
    print("\n" + result["classification_report"])