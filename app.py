"""
app.py
--------
LoanGuard Enterprise AI — Streamlit entry point.

This file IS the "Home" page. The other five pages (Apply for Loan, AI
Assistant, Upload Documents, Risk Dashboard, Approval Report) live in
pages/ and are picked up automatically by Streamlit's multipage app
convention — the sidebar you see is native Streamlit navigation, styled
by src/utils/ui_theme.py.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import APP_SUBTITLE, APP_TITLE
from src.database.db import application_count, init_db
from src.utils.language import SUPPORTED_LANGUAGES
from src.utils.ui_theme import apply_theme

apply_theme("Home")
init_db()

# --- NEW: Multilingual Voice AI — global language preference, shared across
# every page via st.session_state so a choice made here (or auto-detected
# from voice/text on the interview page) follows the customer everywhere. ---
st.session_state.setdefault("preferred_language", "auto")
with st.sidebar:
    st.markdown("#### 🌐 Language")
    lang_options = ["auto"] + list(SUPPORTED_LANGUAGES.keys())
    st.session_state["preferred_language"] = st.selectbox(
        "Preferred language",
        lang_options,
        format_func=lambda code: "Auto-detect" if code == "auto" else SUPPORTED_LANGUAGES[code],
        index=lang_options.index(st.session_state["preferred_language"]),
        label_visibility="collapsed",
    )
    st.caption("Applies to the AI Assistant and Apply for Loan pages.")

st.markdown('<div class="lg-kicker">Enterprise AI Loan Origination System</div>', unsafe_allow_html=True)
st.title(f"🏦 {APP_TITLE}")
st.markdown(f"##### {APP_SUBTITLE}")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="lg-metric"><div class="lg-kicker">Applications Processed</div>'
        f'<h2>{application_count()}</h2></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="lg-metric"><div class="lg-kicker">Decision Engine</div>'
        '<h2>Rule-Based · Explainable</h2></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="lg-metric"><div class="lg-kicker">Risk Model</div>'
        '<h2>XGBoost</h2></div>',
        unsafe_allow_html=True,
    )

st.markdown("### What this system does")

feature_cols = st.columns(3)
features = [
    ("💬", "AI Policy Assistant", "Answers eligibility, fee, and EMI questions strictly "
     "from the bank's policy document — never guesses."),
    ("🧾", "Conversational Interview", "Collects applicant details naturally, with OCR "
     "auto-fill from uploaded PAN, Aadhaar, and salary slips."),
    ("📡", "CIBIL Bureau Pull", "Retrieves a credit report through a swappable bureau-API "
     "abstraction layer."),
    ("🤖", "ML Risk Scoring", "An XGBoost model estimates default probability and "
     "risk category from the applicant's financial profile."),
    ("⚖️", "Business Rule Engine", "A deterministic, auditable policy engine — not the ML "
     "model — makes every approve/reject decision."),
    ("📄", "Executive Reporting", "Every decision is logged to the database and can be "
     "exported as a formatted PDF approval report."),
    ("🎙️", "Multilingual Voice AI", "Complete the interview and ask policy questions by "
     "speaking or typing in English, Telugu, Hindi, Tamil, or Kannada."),
]
for i, (icon, title, desc) in enumerate(features):
    with feature_cols[i % 3]:
        st.markdown(
            f'<div class="lg-card"><h3>{icon} {title}</h3>'
            f'<p style="color:var(--lg-muted)">{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    '<p style="color:var(--lg-muted)">Use the sidebar to start a new application, '
    'ask the AI policy assistant a question, or open the executive risk dashboard.</p>',
    unsafe_allow_html=True,
)
