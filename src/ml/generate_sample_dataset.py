"""
src/ml/generate_sample_dataset.py
-----------------------------------
The project is specified against this Kaggle dataset:
    https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

Kaggle requires authenticated download (kaggle.json credentials), so it
can't be fetched automatically inside this environment. This script
generates a realistic, schema-compatible synthetic dataset so the ML
pipeline is runnable end-to-end out of the box.

To use the REAL dataset instead:
    1. kaggle datasets download -d architsharma01/loan-approval-prediction-dataset
    2. Unzip and place the CSV at data/loan_dataset.csv
    3. Ensure column names match config.ML_FEATURE_COLUMNS + config.ML_TARGET_COLUMN
       (rename with pandas if the Kaggle column headers differ)
    4. Re-run src/ml/train_model.py — no other code changes needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATASET_PATH, ML_RANDOM_STATE


def generate_dataset(n_rows: int = 4000, seed: int = ML_RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cibil_score = rng.integers(300, 901, n_rows)
    annual_income = rng.integers(200_000, 5_000_001, n_rows)
    employment_years = np.round(rng.uniform(0, 30, n_rows), 1)
    loan_amount = rng.integers(50_000, 4_000_001, n_rows)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 84, 120], n_rows)
    residential_assets = rng.integers(0, 8_000_001, n_rows)
    commercial_assets = rng.integers(0, 5_000_001, n_rows)
    existing_liabilities = rng.integers(0, 3_000_001, n_rows)

    # Latent default-risk score used only to *generate* a plausible label —
    # the model never sees this; it only sees the features above.
    risk_score = (
        -0.010 * (cibil_score - 600)
        - 0.0000015 * annual_income
        - 0.05 * employment_years
        + 0.0000025 * loan_amount
        + 0.0000015 * existing_liabilities
        - 0.0000008 * (residential_assets + commercial_assets)
        + rng.normal(0, 1.4, n_rows)
    )
    default_prob = 1 / (1 + np.exp(-risk_score))
    loan_status = (rng.uniform(0, 1, n_rows) < default_prob).astype(int)
    # loan_status: 1 = defaulted / rejected risk, 0 = healthy / approved risk

    df = pd.DataFrame(
        {
            "cibil_score": cibil_score,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "residential_assets": residential_assets,
            "commercial_assets": commercial_assets,
            "existing_liabilities": existing_liabilities,
            "loan_status": loan_status,
        }
    )
    return df


def main() -> None:
    df = generate_dataset()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Synthetic dataset written to {DATASET_PATH} ({len(df)} rows, "
          f"{df['loan_status'].mean():.1%} default rate)")


if __name__ == "__main__":
    main()
