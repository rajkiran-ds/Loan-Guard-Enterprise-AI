"""
src/decision/explainer.py
----------------------------
Turns a rule_engine.Decision + ml.model.RiskAssessment into the
human-readable explanation blocks used by the Streamlit UI and the PDF
approval report. Deliberately template-based rather than LLM-generated:
an approval/rejection explanation must be exactly reproducible and
auditable, which an LLM call would not guarantee.
"""

from __future__ import annotations

from src.decision.rule_engine import Decision
from src.ml.model import RiskAssessment


def build_explanation(decision: Decision, risk: RiskAssessment) -> dict:
    verdict_line = {
        "Approved": "✅ Loan Approved",
        "Approved with Review": "🟡 Approved — Pending Manual Review",
        "Rejected": "❌ Loan Rejected",
    }.get(decision.outcome, decision.outcome)

    summary = (
        f"{verdict_line} under {decision.rule_triggered}. "
        f"ML risk model estimated a {risk.default_probability:.1%} default probability "
        f"({risk.risk_category} risk)."
    )

    return {
        "verdict": verdict_line,
        "outcome": decision.outcome,
        "summary": summary,
        "rule_triggered": decision.rule_triggered,
        "approved_loan_amount": decision.approved_loan_amount,
        "interest_rate": decision.interest_rate,
        "requires_manual_review": decision.requires_manual_review,
        "reasons": decision.reasons,
        "risk_category": risk.risk_category,
        "default_probability": risk.default_probability,
    }


def format_reasons_markdown(explanation: dict) -> str:
    header = "Approved because:" if explanation["outcome"].startswith("Approved") else "Rejected because:"
    bullet_points = "\n".join(f"- {reason}" for reason in explanation["reasons"])
    return f"**{header}**\n\n{bullet_points}"
