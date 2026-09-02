"""
src/rag/build_policy_pdf.py
------------------------------
Generates data/bank_policy.pdf — the internal policy document the RAG
Knowledge Assistant (FEATURE 1) is grounded on. Run once:
    python -m src.rag.build_policy_pdf
Replace the generated PDF with your own bank_policy.pdf at any time; the
RAG pipeline re-indexes automatically the next time the FAISS cache is
cleared or force_rebuild=True is passed to build_vectorstore().
"""

from __future__ import annotations

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from config import POLICY_PDF_PATH

_STYLES = getSampleStyleSheet()
_H1 = ParagraphStyle("H1", parent=_STYLES["Heading1"], spaceBefore=16, spaceAfter=8)
_H2 = ParagraphStyle("H2", parent=_STYLES["Heading2"], spaceBefore=12, spaceAfter=6)
_BODY = ParagraphStyle("Body", parent=_STYLES["BodyText"], spaceAfter=6, leading=15)

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1. Eligibility Criteria",
        [
            "Applicants must be between 21 and 65 years of age at the time of loan maturity.",
            "Applicants must be salaried employees, self-employed professionals, or registered "
            "business owners with a minimum of 1 year in their current occupation.",
            "Minimum monthly income requirement is ₹15,000 for salaried applicants and ₹25,000 "
            "for self-employed applicants.",
            "Applicants must hold a valid PAN card and Aadhaar card.",
            "A minimum CIBIL score of 650 is required for automated approval consideration; "
            "applicants below this threshold are routed to manual underwriting or declined "
            "per the credit policy.",
        ],
    ),
    (
        "2. Documents Required",
        [
            "Identity Proof: PAN Card (mandatory) and Aadhaar Card (mandatory).",
            "Income Proof: Last 3 months' salary slips for salaried applicants, or last 2 years' "
            "ITR and audited financials for self-employed applicants.",
            "Bank Statements: Last 6 months' primary bank account statements.",
            "Address Proof: Aadhaar Card, utility bill, or rental agreement not older than 3 months.",
            "Passport-size photograph.",
            "Business proof (GST certificate or registration certificate) for self-employed and "
            "business-owner applicants.",
        ],
    ),
    (
        "3. Processing Fees",
        [
            "A one-time processing fee of 1.5% of the sanctioned loan amount (minimum ₹1,000, "
            "maximum ₹15,000) is charged at disbursal, plus applicable GST.",
            "The processing fee is non-refundable once the loan is disbursed.",
            "No processing fee is charged if an application is rejected before disbursal.",
            "A documentation and stamp duty charge of ₹250 (plus applicable state stamp duty) "
            "applies to every sanctioned loan.",
        ],
    ),
    (
        "4. EMI Policy",
        [
            "EMIs are calculated on a reducing-balance basis over the agreed loan tenure "
            "(12 to 120 months, subject to the sanctioned loan amount and applicant profile).",
            "EMI due date is fixed at loan disbursal and falls on the same date each month; "
            "if that date does not exist in a given month, the EMI is debited on the last "
            "business day of the month.",
            "Auto-debit (NACH mandate) from the applicant's primary bank account is mandatory "
            "for all sanctioned loans.",
            "The maximum permissible EMI-to-income ratio (debt-to-income, or DTI) is 40% of "
            "monthly income, inclusive of all existing EMI obligations.",
        ],
    ),
    (
        "5. Foreclosure and Prepayment Policy",
        [
            "Loans may be foreclosed (fully prepaid) any time after 6 EMIs have been paid.",
            "A foreclosure charge of 2% of the outstanding principal applies for loans "
            "foreclosed within the first 24 months; no foreclosure charge applies after "
            "24 months from disbursal.",
            "Partial prepayment is permitted up to twice per year, capped at 25% of the "
            "outstanding principal per instance, with no partial-prepayment charge.",
            "Foreclosure requests must be submitted at least 7 working days before the "
            "intended settlement date.",
        ],
    ),
    (
        "6. Late Payment Policy",
        [
            "A late payment fee of 2% of the overdue EMI amount (minimum ₹200) is charged if "
            "an EMI is not honored on the due date.",
            "Overdue EMIs accrue additional interest at 2% per month on the overdue amount "
            "until cleared.",
            "Three or more late payments within any rolling 12-month period may result in the "
            "loan being flagged for credit review and can affect the applicant's future "
            "eligibility and CIBIL score.",
            "Applicants facing genuine financial hardship may request a restructuring review "
            "by contacting customer support before an EMI is missed; approval is at the "
            "bank's discretion.",
        ],
    ),
    (
        "7. Interest Rate Policy",
        [
            "Interest rates are personalized based on CIBIL score, income stability, "
            "debt-to-income ratio, and the applicant's assessed default-risk category, "
            "ranging from a minimum of 9.0% to a maximum of 18.0% per annum.",
            "The applicable base rate and risk-based adjustments are reviewed quarterly by "
            "the credit policy committee and may change for new applications; the interest "
            "rate on an already-disbursed loan remains fixed for its tenure.",
        ],
    ),
    (
        "8. Loan Amount Limits",
        [
            "The maximum sanctioned loan amount is capped as a multiple of the applicant's "
            "verified monthly income, determined by the applicant's risk tier under the "
            "bank's credit policy, and is subject to a hard ceiling of ₹40,00,000 per "
            "applicant regardless of income.",
            "Minimum loan amount is ₹50,000.",
        ],
    ),
]


def build_policy_pdf() -> None:
    POLICY_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(POLICY_PDF_PATH),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )
    story = [
        Paragraph("National Trust Bank", _STYLES["Title"]),
        Paragraph("Personal Loan — Policy Handbook", _H1),
        Paragraph(
            "This handbook governs the origination, servicing, and closure of personal "
            "loans issued by National Trust Bank. It is the authoritative source for the "
            "LoanGuard AI Knowledge Assistant. Terms are illustrative and configured for "
            "the LoanGuard Enterprise AI demo environment.",
            _BODY,
        ),
        Spacer(1, 8),
    ]

    for i, (heading, points) in enumerate(SECTIONS):
        story.append(Paragraph(heading, _H2))
        for point in points:
            story.append(Paragraph(f"• {point}", _BODY))
        if i < len(SECTIONS) - 1:
            story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Bank policy PDF written to {POLICY_PDF_PATH}")


if __name__ == "__main__":
    build_policy_pdf()
