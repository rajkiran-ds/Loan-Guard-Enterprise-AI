"""
src/ml/train_model.py
------------------------
FEATURE 5 — ML Risk Model.

Trains an XGBoost classifier on Loan_Status, evaluates it with ROC AUC,
precision, recall, F1, and a confusion matrix, and persists the model +
metrics to src/ml/artifacts/. This script is the one and only place the
model is trained — src/ml/model.py only ever loads what this script saves.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    FEATURE_SCHEMA_PATH,
    ML_FEATURE_COLUMNS,
    ML_RANDOM_STATE,
    ML_TARGET_COLUMN,
    ML_TEST_SIZE,
    MODEL_METRICS_PATH,
    MODEL_PATH,
)
from src.ml.generate_sample_dataset import generate_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_training_data() -> pd.DataFrame:
    if DATASET_PATH.exists():
        logger.info("Loading dataset from %s", DATASET_PATH)
        df = pd.read_csv(DATASET_PATH)
    else:
        logger.warning("No dataset found at %s — generating a synthetic one.", DATASET_PATH)
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


def train() -> dict:
    df = load_training_data()
    X = df[ML_FEATURE_COLUMNS]
    y = df[ML_TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=ML_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    FEATURE_SCHEMA_PATH.write_text(json.dumps({"features": ML_FEATURE_COLUMNS}, indent=2))

    logger.info("Model trained and saved to %s", MODEL_PATH)
    logger.info("Metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
