"""
src/ml/experimentation.py
-------------------------
Compares SIX candidate models on stratified 5-fold cross-validation
for the LoanGuard default-risk classifier:

    1. LogisticRegression
    2. RandomForestClassifier
    3. GradientBoostingClassifier
    4. XGBClassifier             (current production model)
    5. PyTorchMLP                (from src/ml/mlp.py)
    6. TensorFlowMLP             (Keras Sequential, same architecture)

For each model, a FULL per-class classification report is printed to the
terminal. The final leaderboard is saved as a CSV and all classification
reports are saved as a text file for later reference.

Production pipeline unchanged: XGBoost (train_model.py) remains the served
model. This file is a comparison harness — the MLP and TF models are
candidates, not replacements.

Run (from project root, with venv active):
    python -m src.ml.experimentation

Outputs:
    src/ml/artifacts/model_comparison.csv          (scalar metrics, 6 rows)
    src/ml/artifacts/classification_reports.txt    (full per-model reports)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    ML_FEATURE_COLUMNS,
    ML_RANDOM_STATE,
    ML_TARGET_COLUMN,
)
from src.ml.mlp import predict_mlp, train_mlp
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_SPLITS = 5
COMPARISON_CSV = ARTIFACTS_DIR / "model_comparison.csv"
REPORTS_TXT = ARTIFACTS_DIR / "classification_reports.txt"

warnings.filterwarnings("ignore", category=UserWarning)


# ================================================================
# TensorFlow / Keras MLP — mirrors PyTorchMLP architecture exactly
# ================================================================
def build_keras_mlp(n_features: int):
    """Same shape as PyTorchMLP: n → 32 → 16 → 1, ReLU, Dropout 0.2, sigmoid."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def keras_fit_predict(X_tr, y_tr, X_val, y_val):
    """fit_predict wrapper for Keras — same interface as the other models."""
    from tensorflow import keras

    model = build_keras_mlp(X_tr.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        verbose=0,
        callbacks=[early_stop],
    )
    return model.predict(X_val, verbose=0).ravel()


# ================================================================
# fit_predict factories — same interface for every model
# ================================================================
def sklearn_fit_predict(factory: Callable):
    """Wraps a sklearn estimator factory into a (train, val) → proba function."""

    def _run(X_tr, y_tr, X_val, y_val):
        est = factory()
        est.fit(X_tr, y_tr)
        return est.predict_proba(X_val)[:, 1]

    return _run


def mlp_fit_predict(X_tr, y_tr, X_val, y_val):
    """PyTorch MLP: train with early stopping, return val probabilities."""
    model = train_mlp(X_tr, y_tr, X_val, y_val)
    return predict_mlp(model, X_val)


# ================================================================
# Cross-validation loop with per-class classification report
# ================================================================
def evaluate_model(name: str, fit_predict, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Run stratified K-fold CV and return:
        - mean scalar metrics across folds (AUC, precision, recall, F1)
        - a full per-class classification report built from out-of-fold
          (OOF) predictions.

    Why OOF aggregation?
        Averaging five per-fold reports hides per-class detail and produces
        odd decimals that don't correspond to any real confusion matrix.
        Instead we concatenate every validation prediction from every fold
        and build ONE report on that combined set — the standard way to
        summarise performance on imbalanced binary targets.
    """
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=ML_RANDOM_STATE
    )
    aucs, precs, recs, f1s = [], [], [], []
    oof_true: list[int] = []
    oof_pred: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        try:
            y_proba = fit_predict(X_tr, y_tr, X_val, y_val)
        except Exception as e:
            logger.error("Model %s failed on fold %d: %s", name, fold_idx, e)
            raise

        y_pred = (y_proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_val, y_proba))
        precs.append(precision_score(y_val, y_pred, zero_division=0))
        recs.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))

        oof_true.extend(y_val.tolist())
        oof_pred.extend(y_pred.tolist())

    report = classification_report(
        oof_true,
        oof_pred,
        target_names=["Default/Rejected (0)", "No-default/Approved (1)"],
        digits=4,
        zero_division=0,
    )

    return {
        "model": name,
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "precision": round(float(np.mean(precs)), 4),
        "recall": round(float(np.mean(recs)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "report": report,
    }


# ================================================================
# Dataset loading
# ================================================================
def load_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATASET_PATH)

    missing = [
        c for c in ML_FEATURE_COLUMNS + [ML_TARGET_COLUMN] if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Dataset {DATASET_PATH} is missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    X_df = df[ML_FEATURE_COLUMNS].fillna(df[ML_FEATURE_COLUMNS].median())
    X = X_df.values.astype(np.float32)

    y_raw = df[ML_TARGET_COLUMN]
    if y_raw.dtype == object:
        uniques = sorted(y_raw.dropna().unique())
        if len(uniques) != 2:
            raise ValueError(f"Expected binary target, got {uniques}")
        logger.info("Mapping target: %r → 0, %r → 1", uniques[0], uniques[1])
        y = (y_raw == uniques[1]).astype(int).values
    else:
        y = y_raw.fillna(0).astype(int).values

    return X, y


# ================================================================
# Main
# ================================================================
def main() -> None:
    logger.info("Loading dataset from %s", DATASET_PATH)
    X, y = load_data()
    logger.info(
        "Dataset: X=%s, y=%s, positive_rate=%.3f",
        X.shape,
        y.shape,
        float(y.mean()),
    )

    # Scaling: MLP + LogisticRegression require it. Trees are invariant.
    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)

    # Optional TensorFlow — skip gracefully if not installed
    try:
        import tensorflow  # noqa: F401

        tf_available = True
    except ImportError:
        logger.warning(
            "TensorFlow not installed — skipping TensorFlowMLP candidate"
        )
        tf_available = False

    candidates: list[tuple[str, Callable]] = [
        (
            "LogisticRegression",
            sklearn_fit_predict(
                lambda: LogisticRegression(
                    max_iter=1000, random_state=ML_RANDOM_STATE
                )
            ),
        ),
        (
            "RandomForest",
            sklearn_fit_predict(
                lambda: RandomForestClassifier(
                    n_estimators=200,
                    random_state=ML_RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
        (
            "GradientBoosting",
            sklearn_fit_predict(
                lambda: GradientBoostingClassifier(
                    random_state=ML_RANDOM_STATE
                )
            ),
        ),
        (
            "XGBoost",
            sklearn_fit_predict(
                lambda: XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    eval_metric="logloss",
                    random_state=ML_RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
        ("PyTorchMLP", mlp_fit_predict),
    ]
    if tf_available:
        candidates.append(("TensorFlowMLP", keras_fit_predict))

    results: list[dict] = []
    reports_for_file: list[str] = []

    for name, fit_predict in candidates:
        logger.info("Evaluating %s ...", name)
        metrics = evaluate_model(name, fit_predict, X_scaled, y)
        results.append(metrics)

        # Compact one-line log summary
        logger.info(
            "  %-20s AUC=%.4f ± %.4f  P=%.4f  R=%.4f  F1=%.4f",
            name,
            metrics["auc_mean"],
            metrics["auc_std"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        # Full per-class report — terminal
        print(f"\n{'=' * 62}")
        print(f"  CLASSIFICATION REPORT — {name}")
        print(f"{'=' * 62}")
        print(metrics["report"])

        # Save to the aggregate file
        reports_for_file.append(
            f"{'=' * 62}\n"
            f"  {name}\n"
            f"  AUC={metrics['auc_mean']} ± {metrics['auc_std']}  "
            f"P={metrics['precision']}  R={metrics['recall']}  "
            f"F1={metrics['f1']}\n"
            f"{'=' * 62}\n"
            f"{metrics['report']}\n"
        )

    # ------------------------------------------------------------------
    # Save CSV (scalar metrics only) and reports text file
    # ------------------------------------------------------------------
    csv_rows = [
        {k: v for k, v in r.items() if k != "report"} for r in results
    ]
    results_df = pd.DataFrame(csv_rows).sort_values(
        "auc_mean", ascending=False
    )
    results_df.to_csv(COMPARISON_CSV, index=False)

    REPORTS_TXT.write_text(
        "\n".join(reports_for_file),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  FINAL LEADERBOARD (sorted by ROC-AUC)")
    print("=" * 62)
    print(results_df.to_string(index=False))

    print("\n=== Winner ===")
    print(results_df.iloc[0].to_dict())
    print(f"\nComparison CSV:    {COMPARISON_CSV}")
    print(f"Full reports file: {REPORTS_TXT}")


if __name__ == "__main__":
    main()