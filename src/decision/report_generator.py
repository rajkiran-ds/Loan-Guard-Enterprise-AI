"""
src/decision/report_generator.py
------------------------------------
FEATURE 9 — Executive PDF Approval Report.

Renders one Application record (customer profile + CIBIL + risk + decision)
into a professionally formatted PDF using ReportLab. Called from the
"Approval Report" Streamlit page after a decision has been reached and
saved to the database.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import APP_TITLE, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_STYLES = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "ReportTitle", parent=_STYLES["Title"], textColor=colors.HexColor("#0B1F3A")
)
_SECTION_STYLE = ParagraphStyle(
    "SectionHeader",
    parent=_STYLES["Heading2"],
    textColor=colors.HexColor("#0B1F3A"),
    spaceBefore=14,
    spaceAfter=6,
)
_BODY_STYLE = _STYLES["BodyText"]

_VERDICT_COLOR = {
    "Approved": colors.HexColor("#1B7A3D"),
    "Approved with Review": colors.HexColor("#B8860B"),
    "Rejected": colors.HexColor("#B00020"),
}


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(rows, colWidths=[6.5 * cm, 9.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0B1F3A")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
            ]
        )
    )
    return table


def generate_approval_report(application: dict) -> Path:
    """application: a dict of Application ORM columns (see src/database/models.py).
    Returns the path to the generated PDF."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"loan_report_{application.get('id', 'draft')}_{dt.date.today().isoformat()}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )

    story = []
    story.append(Paragraph(APP_TITLE, _TITLE_STYLE))
    story.append(Paragraph("Executive Loan Approval Report", _STYLES["Heading3"]))
    story.append(
        Paragraph(
            f"Generated on {dt.datetime.now():%d %B %Y, %H:%M} &nbsp;|&nbsp; "
            f"Application ID: {application.get('id', 'N/A')}",
            _BODY_STYLE,
        )
    )
    story.append(Spacer(1, 10))

    verdict = application.get("decision", "Pending")
    verdict_color = _VERDICT_COLOR.get(verdict, colors.HexColor("#333333"))
    verdict_style = ParagraphStyle(
        "Verdict", parent=_STYLES["Heading1"], textColor=verdict_color, fontSize=18
    )
    story.append(Paragraph(f"Decision: {verdict}", verdict_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Customer Profile", _SECTION_STYLE))
    story.append(
        _kv_table(
            [
                ("Full Name", str(application.get("full_name", "-"))),
                ("Age", str(application.get("age", "-"))),
                ("PAN", str(application.get("pan", "-"))),
                ("Employment Type", str(application.get("employment_type", "-"))),
                ("Company", str(application.get("company_name", "-") or "-")),
                ("Loan Purpose", str(application.get("loan_purpose", "-") or "-")),
            ]
        )
    )

    story.append(Paragraph("Financial Summary", _SECTION_STYLE))
    story.append(
        _kv_table(
            [
                ("Monthly Income", f"₹{application.get('monthly_income', 0):,.0f}"),
                ("Existing EMIs", f"₹{application.get('existing_emis', 0):,.0f}"),
                ("Requested Loan Amount", f"₹{application.get('requested_loan_amount', 0):,.0f}"),
            ]
        )
    )

    story.append(Paragraph("Credit Bureau (CIBIL) Report", _SECTION_STYLE))
    story.append(
        _kv_table(
            [
                ("CIBIL Score", str(application.get("cibil_score", "-"))),
                ("Active Loans", str(application.get("active_loans", "-"))),
                ("Late Payments", str(application.get("late_payments", "-"))),
                ("Credit Utilization", f"{application.get('credit_utilization', '-')}%"),
                ("Credit History", f"{application.get('credit_history_years', '-')} years"),
            ]
        )
    )

    story.append(Paragraph("Risk Assessment (ML Model)", _SECTION_STYLE))
    story.append(
        _kv_table(
            [
                ("Default Probability", f"{application.get('default_probability', 0):.1%}"
                 if application.get("default_probability") is not None else "-"),
                ("Risk Category", str(application.get("risk_category", "-"))),
            ]
        )
    )

    story.append(Paragraph("Business Rule Decision", _SECTION_STYLE))
    story.append(
        _kv_table(
            [
                ("Rule Triggered", str(application.get("rule_triggered", "-"))),
                ("Approved Loan Amount", f"₹{application.get('approved_loan_amount', 0):,.0f}"),
                ("Interest Rate", f"{application.get('interest_rate', 0):.2%}"
                 if application.get("interest_rate") else "-"),
                ("Manual Review Required", "Yes" if application.get("requires_manual_review") else "No"),
            ]
        )
    )

    story.append(Paragraph("Reasoning", _SECTION_STYLE))
    reasons = application.get("decision_reasons") or []
    for reason in reasons:
        story.append(Paragraph(f"• {reason}", _BODY_STYLE))

    story.append(Paragraph("Recommendation", _SECTION_STYLE))
    if verdict == "Approved":
        recommendation = (
            "Proceed to loan agreement generation and disbursal per standard "
            "onboarding workflow."
        )
    elif verdict == "Approved with Review":
        recommendation = (
            "Route to a human underwriter for final sign-off before disbursal. "
            "Automated approval is conditional."
        )
    else:
        recommendation = (
            "Do not proceed with disbursal. Applicant may reapply after "
            "addressing the factors listed above."
        )
    story.append(Paragraph(recommendation, _BODY_STYLE))

    doc.build(story)
    logger.info("Approval report generated at %s", filepath)
    return filepath
