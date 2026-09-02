"""
config.py
---------
Single source of truth for every configurable constant in LoanGuard Enterprise AI.
Nothing in src/ or app.py should hard-code a path, threshold, or model name —
it should import it from here. This is what makes the system config-driven
instead of scattered with magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"
ARTIFACTS_DIR = BASE_DIR / "src" / "ml" / "artifacts"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = DATA_DIR / "loan_dataset.csv"
POLICY_PDF_PATH = DATA_DIR / "bank_policy.pdf"
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", str(DATA_DIR / "faiss_index")))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "loanguard.db")))
MODEL_PATH = ARTIFACTS_DIR / "xgb_risk_model.joblib"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"

for _dir in (DATA_DIR, ASSETS_DIR, DOCS_DIR, ARTIFACTS_DIR, UPLOADS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM / RAG configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 120
RAG_TOP_K = 4
RAG_NOT_FOUND_MESSAGE = "I couldn't find this in the bank policy."

# ---------------------------------------------------------------------------
# Mock CIBIL API
# ---------------------------------------------------------------------------
CIBIL_API_MODE = os.getenv("CIBIL_API_MODE", "mock")
CIBIL_API_BASE_URL = os.getenv("CIBIL_API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# ML model configuration
# ---------------------------------------------------------------------------
ML_TARGET_COLUMN = "loan_status"
ML_RANDOM_STATE = 42
ML_TEST_SIZE = 0.2
ML_FEATURE_COLUMNS = [
    "cibil_score",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "residential_assets",
    "commercial_assets",
    "existing_liabilities",
]

RISK_LOW_MAX_DEFAULT_PROB = 0.15
RISK_MEDIUM_MAX_DEFAULT_PROB = 0.40
# above RISK_MEDIUM_MAX_DEFAULT_PROB => High risk


# ---------------------------------------------------------------------------
# Business rule engine — lending policy (see src/decision/rule_engine.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LendingRule:
    name: str
    min_cibil: int
    max_dti: float  # debt-to-income, expressed as a fraction (0.30 = 30%)
    allowed_risk: tuple[str, ...]
    max_loan_multiplier: float  # multiple of monthly salary
    interest_rate: float
    requires_review: bool = False
    min_employment_years: float = 0.0


LENDING_RULES: tuple[LendingRule, ...] = (
    LendingRule(
        name="Rule 1 — Prime",
        min_cibil=780,
        max_dti=0.30,
        allowed_risk=("Low",),
        max_loan_multiplier=10.0,
        interest_rate=0.095,
    ),
    LendingRule(
        name="Rule 2 — Standard",
        min_cibil=720,
        max_dti=0.40,
        allowed_risk=("Low", "Medium"),
        max_loan_multiplier=8.0,
        interest_rate=0.11,
    ),
    LendingRule(
        name="Rule 3 — Conditional",
        min_cibil=650,
        max_dti=1.0,  # not the binding constraint for this rule
        allowed_risk=("Medium",),
        max_loan_multiplier=5.0,
        interest_rate=0.135,
        requires_review=True,
        min_employment_years=2.0,
    ),
)

REJECT_CIBIL_THRESHOLD = 650
REJECT_LATE_PAYMENTS_THRESHOLD = 3

# Interest rate adjustment logic
BASE_INTEREST_RATE = 0.09
RISK_RATE_ADJUSTMENT = {"Low": 0.0, "Medium": 0.02, "High": 0.04}
DTI_HIGH_THRESHOLD = 0.40
DTI_HIGH_ADJUSTMENT = 0.01
CIBIL_LOW_THRESHOLD = 700
CIBIL_LOW_ADJUSTMENT = 0.015
MAX_INTEREST_RATE = 0.18
MIN_INTEREST_RATE = 0.09

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "LoanGuard Enterprise AI"
APP_SUBTITLE = "Intelligent Personal Loan Origination System"
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

CUSTOMER_INTERVIEW_FIELDS: list[str] = [
    "full_name",
    "age",
    "pan",
    "aadhaar",
    "monthly_income",
    "employment_type",
    "company_name",
    "existing_emis",
    "loan_purpose",
    "requested_loan_amount",
]