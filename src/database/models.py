"""
src/database/models.py
------------------------
SQLAlchemy ORM models. One Application row = one full, end-to-end loan
origination record: customer profile, OCR results, CIBIL bureau pull,
ML risk output, business-rule decision, and a timestamp — exactly what
FEATURE 8 (Database) requires.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- Customer profile (Feature 2: AI interview) ---
    full_name: Mapped[str] = mapped_column(String(120))
    age: Mapped[int] = mapped_column(Integer)
    pan: Mapped[str] = mapped_column(String(10))
    aadhaar: Mapped[str] = mapped_column(String(12))
    monthly_income: Mapped[float] = mapped_column(Float)
    employment_type: Mapped[str] = mapped_column(String(40))
    company_name: Mapped[str] = mapped_column(String(120), nullable=True)
    existing_emis: Mapped[float] = mapped_column(Float, default=0.0)
    loan_purpose: Mapped[str] = mapped_column(String(120), nullable=True)
    requested_loan_amount: Mapped[float] = mapped_column(Float)

    # --- Documents / OCR (Feature 3) ---
    documents_uploaded: Mapped[dict] = mapped_column(JSON, default=dict)
    ocr_confidence: Mapped[float] = mapped_column(Float, nullable=True)

    # --- CIBIL bureau pull (Feature 4) ---
    cibil_score: Mapped[int] = mapped_column(Integer, nullable=True)
    active_loans: Mapped[int] = mapped_column(Integer, nullable=True)
    late_payments: Mapped[int] = mapped_column(Integer, nullable=True)
    credit_utilization: Mapped[float] = mapped_column(Float, nullable=True)
    credit_history_years: Mapped[float] = mapped_column(Float, nullable=True)

    # --- ML risk model output (Feature 5) ---
    default_probability: Mapped[float] = mapped_column(Float, nullable=True)
    risk_category: Mapped[str] = mapped_column(String(20), nullable=True)

    # --- Business rule engine decision (Feature 6) ---
    decision: Mapped[str] = mapped_column(String(30), nullable=True)  # Approved / Rejected / Review
    rule_triggered: Mapped[str] = mapped_column(String(80), nullable=True)
    approved_loan_amount: Mapped[float] = mapped_column(Float, nullable=True)
    interest_rate: Mapped[float] = mapped_column(Float, nullable=True)
    decision_reasons: Mapped[list] = mapped_column(JSON, default=list)
    requires_manual_review: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
