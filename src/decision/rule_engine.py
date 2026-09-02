"""
src/decision/rule_engine.py
------------------------------
FEATURE 6 — Business Rule Engine.

This is the ONLY module in the codebase allowed to output a lending
decision. The ML model (src/ml/model.py) supplies a default probability
and a risk category; this engine applies deterministic bank policy on
top of that, exactly the way a real underwriting desk separates "model
score" from "credit policy". That separation is what makes the system
auditable and compliant — a regulator can review LENDING_RULES in
config.py without needing to understand gradient boosting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import (
    BASE_INTEREST_RATE,
    CIBIL_LOW_ADJUSTMENT,
    CIBIL_LOW_THRESHOLD,
    DTI_HIGH_ADJUSTMENT,
    DTI_HIGH_THRESHOLD,
    LENDING_RULES,
    LendingRule,
    MAX_INTEREST_RATE,
    MIN_INTEREST_RATE,
    REJECT_CIBIL_THRESHOLD,
    REJECT_LATE_PAYMENTS_THRESHOLD,
    RISK_RATE_ADJUSTMENT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ApplicantSnapshot:
    monthly_income: float
    existing_emis: float
    employment_years: float
    cibil_score: int
    late_payments: int
    risk_category: str          # "Low" | "Medium" | "High" — from RiskModel
    default_probability: float  # from RiskModel


@dataclass
class Decision:
    outcome: str  # "Approved" | "Approved with Review" | "Rejected"
    rule_triggered: str
    approved_loan_amount: float
    interest_rate: float
    requires_manual_review: bool
    reasons: list[str] = field(default_factory=list)


def _debt_to_income(applicant: ApplicantSnapshot) -> float:
    if applicant.monthly_income <= 0:
        return 1.0
    return applicant.existing_emis / applicant.monthly_income


def _compute_interest_rate(applicant: ApplicantSnapshot, dti: float) -> float:
    """Base Rate 9% + risk adjustment + DTI adjustment + low-CIBIL adjustment,
    clamped to [MIN_INTEREST_RATE, MAX_INTEREST_RATE]."""
    rate = BASE_INTEREST_RATE
    rate += RISK_RATE_ADJUSTMENT.get(applicant.risk_category, 0.04)
    if dti > DTI_HIGH_THRESHOLD:
        rate += DTI_HIGH_ADJUSTMENT
    if applicant.cibil_score < CIBIL_LOW_THRESHOLD:
        rate += CIBIL_LOW_ADJUSTMENT
    return max(MIN_INTEREST_RATE, min(MAX_INTEREST_RATE, round(rate, 4)))


def _hard_reject(applicant: ApplicantSnapshot) -> list[str]:
    reasons = []
    if applicant.cibil_score < REJECT_CIBIL_THRESHOLD:
        reasons.append(f"CIBIL score ({applicant.cibil_score}) is below the minimum threshold of {REJECT_CIBIL_THRESHOLD}.")
    if applicant.risk_category == "High":
        reasons.append("ML model classifies this applicant as High default risk.")
    if applicant.late_payments > REJECT_LATE_PAYMENTS_THRESHOLD:
        reasons.append(f"Applicant has {applicant.late_payments} late payments, exceeding the policy limit of {REJECT_LATE_PAYMENTS_THRESHOLD}.")
    return reasons


def _rule_matches(rule: LendingRule, applicant: ApplicantSnapshot, dti: float) -> bool:
    if applicant.cibil_score < rule.min_cibil:
        return False
    if applicant.risk_category not in rule.allowed_risk:
        return False
    if dti > rule.max_dti:
        return False
    if applicant.employment_years < rule.min_employment_years:
        return False
    return True


def evaluate(applicant: ApplicantSnapshot) -> Decision:
    """Apply Rules 1-4 in priority order and return a fully-explained Decision."""
    dti = _debt_to_income(applicant)
    interest_rate = _compute_interest_rate(applicant, dti)

    hard_reject_reasons = _hard_reject(applicant)
    if hard_reject_reasons:
        decision = Decision(
            outcome="Rejected",
            rule_triggered="Rule 4 — Reject / Manual Review",
            approved_loan_amount=0.0,
            interest_rate=0.0,
            requires_manual_review=True,
            reasons=hard_reject_reasons,
        )
        logger.info("Decision: Rejected (hard reject) reasons=%s", hard_reject_reasons)
        return decision

    for rule in LENDING_RULES:
        if _rule_matches(rule, applicant, dti):
            max_loan = round(applicant.monthly_income * rule.max_loan_multiplier, 2)
            reasons = _approval_reasons(applicant, dti, rule)
            outcome = "Approved with Review" if rule.requires_review else "Approved"
            decision = Decision(
                outcome=outcome,
                rule_triggered=rule.name,
                approved_loan_amount=max_loan,
                interest_rate=rule.interest_rate,
                requires_manual_review=rule.requires_review,
                reasons=reasons,
            )
            logger.info("Decision: %s via %s, loan=%.2f rate=%.3f", outcome, rule.name, max_loan, rule.interest_rate)
            return decision

    # No rule matched and no hard-reject trigger fired -> policy gap, route to manual review
    decision = Decision(
        outcome="Rejected",
        rule_triggered="No matching rule",
        approved_loan_amount=0.0,
        interest_rate=0.0,
        requires_manual_review=True,
        reasons=[
            "Applicant profile does not satisfy any automated approval rule "
            "(borderline CIBIL, DTI, or employment tenure). Routed to a human underwriter."
        ],
    )
    logger.info("Decision: Rejected (no rule matched)")
    return decision


def _approval_reasons(applicant: ApplicantSnapshot, dti: float, rule: LendingRule) -> list[str]:
    reasons = []
    if applicant.cibil_score >= 780:
        reasons.append("Excellent CIBIL score.")
    elif applicant.cibil_score >= 720:
        reasons.append("Strong CIBIL score above the standard-approval threshold.")
    else:
        reasons.append("CIBIL score meets the conditional-approval threshold.")

    if applicant.employment_years >= 2:
        reasons.append("Stable employment history.")

    reasons.append(f"Debt-to-income ratio ({dti:.0%}) is within the policy threshold ({rule.max_dti:.0%}).")
    reasons.append(f"ML default-risk model classifies applicant as {applicant.risk_category} risk "
                    f"({applicant.default_probability:.1%} default probability).")
    if rule.requires_review:
        reasons.append("Approved conditionally — routed to a human underwriter for final sign-off.")
    return reasons
