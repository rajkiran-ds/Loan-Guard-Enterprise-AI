"""
pages/2_Apply_for_Loan.py
----------------------------
FEATURE 2 — AI Customer Interview, presented as a step-by-step chat rather
than one long form. Fields are pre-filled from st.session_state["ocr_prefill"]
(set by the Upload Documents page) when available.

On the final step this page also runs the rest of the pipeline end-to-end:
FEATURE 4 (mock CIBIL pull) -> FEATURE 5 (XGBoost risk model) -> FEATURE 6
(business rule engine) -> FEATURE 8 (save to database) — then hands the
resulting application id to the Approval Report page.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from src.api.cibil_api import CibilBureauClient
from src.database.db import save_application
from src.decision.explainer import build_explanation, format_reasons_markdown
from src.decision.rule_engine import ApplicantSnapshot, evaluate
from src.ml.model import RiskModel
from src.rag.multilingual_chat import translate_text
from src.utils.language import SUPPORTED_LANGUAGES, detect_language, language_name
from src.utils.ui_theme import apply_theme, badge
from src.utils.validators import (
    validate_aadhaar,
    validate_age,
    validate_dob,
    validate_employment_type,
    validate_income,
    validate_loan_amount,
    validate_name,
    validate_pan,
)
from src.voice.stt import SpeechToText
from src.voice.tts import synthesize

apply_theme("Apply for Loan")
prefill = st.session_state.get("ocr_prefill", {})

st.markdown('<div class="lg-kicker">Feature 2 · Conversational Interview</div>', unsafe_allow_html=True)
st.title("🧾 Apply for a Personal Loan")
st.caption("Answer a few quick questions — fields already extracted from your documents are pre-filled below.")
st.caption("🌐 Voice-first: speak your answers in English, Telugu, Hindi, Tamil, or Kannada, or type as usual.")

# --- NEW: Multilingual Voice AI — voice-first interview -------------------
st.session_state.setdefault("voice_answers", {})
st.session_state.setdefault("detected_lang", "en")

preferred = st.session_state.get("preferred_language", "auto")
active_lang = preferred if preferred in SUPPORTED_LANGUAGES else st.session_state["detected_lang"]

mode_col, listen_col = st.columns([3, 1])
with mode_col:
    interview_mode = st.radio(
        "Interview mode", ["📝 Type my answers", "🎙️ Speak my answers"], horizontal=True,
        label_visibility="collapsed",
    )
with listen_col:
    if st.button(f"🔊 Hear questions ({language_name(active_lang)})", use_container_width=True):
        questions_en = (
            "Please tell me your full name, age, PAN, Aadhaar number, monthly income, "
            "employment type, company name, existing monthly EMIs, years in your current job, "
            "loan purpose, and the loan amount you would like to request."
        )
        spoken_prompt = translate_text(questions_en, active_lang) if active_lang != "en" else questions_en
        audio = synthesize(spoken_prompt, lang=active_lang)
        if audio:
            st.audio(audio, format="audio/mp3", autoplay=True)
        else:
            st.info("Voice playback unavailable in this environment.")

VOICE_FIELDS = [
    ("full_name", "Full Name", str),
    ("pan", "PAN", str),
    ("aadhaar", "Aadhaar", str),
    ("company_name", "Company / Business Name", str),
    ("loan_purpose", "Loan Purpose", str),
    ("age", "Age", "number"),
    ("monthly_income", "Monthly Income", "number"),
    ("existing_emis", "Existing Monthly EMIs", "number"),
    ("employment_years", "Years in Current Job/Business", "number"),
    ("requested_loan_amount", "Requested Loan Amount", "number"),
]


def _extract_number(text: str) -> float | None:
    """Pulls the first number out of a transcribed spoken answer, e.g.
    'sixty thousand' won't parse but '60000' or '₹60,000' will — the
    customer can always correct the value in the field afterward."""
    import re

    match = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


if interview_mode == "🎙️ Speak my answers":
    st.markdown("#### Speak each answer")
    st.caption("Record an answer, review the transcript, then continue to the form below to submit.")
    voice_cols = st.columns(2)
    for i, (field_key, label, kind) in enumerate(VOICE_FIELDS):
        with voice_cols[i % 2]:
            st.markdown(f"**{label}**")
            audio = st.audio_input(f"Speak: {label}", key=f"mic_{field_key}", label_visibility="collapsed")
            if audio is not None:
                with st.spinner("Transcribing..."):
                    transcription = SpeechToText.transcribe(audio.getvalue())
                if not transcription.engine_available:
                    st.warning("Voice engine unavailable — please type this field below instead.")
                elif transcription.text:
                    st.session_state["detected_lang"] = detect_language(transcription.text)
                    if kind == "number":
                        parsed = _extract_number(transcription.text)
                        if parsed is not None:
                            st.session_state["voice_answers"][field_key] = parsed
                            st.success(f"Heard: {transcription.text} → {parsed:g}")
                        else:
                            st.info(f"Heard: {transcription.text} (no number found — enter it below)")
                    else:
                        st.session_state["voice_answers"][field_key] = transcription.text
                        st.success(f"Heard: {transcription.text}")
    st.markdown("---")

voice_answers = st.session_state["voice_answers"]
# --- NEW: Multilingual Voice AI — spoken answers take priority over OCR
# prefill so the form below reflects what the customer just said, while
# still falling back to OCR-extracted values, exactly like before. ---
merged = {**prefill, **voice_answers}
if voice_answers:
    st.caption("🎙️ Fields below are pre-filled from your spoken answers — review and correct as needed.")

with st.form("interview_form"):
    c1, c2 = st.columns(2)
    with c1:
        full_name = st.text_input("Full Name", value=merged.get("full_name", ""))
        age = st.number_input("Age", min_value=18, max_value=100, value=int(merged.get("age", 28)))
        pan = st.text_input("PAN", value=merged.get("pan", ""), max_chars=10, help="Format: AAAAA9999A")
        aadhaar = st.text_input("Aadhaar", value=merged.get("aadhaar", ""), max_chars=14)
        dob = st.date_input("Date of Birth", value=dt.date(1996, 1, 1), min_value=dt.date(1940, 1, 1))
    with c2:
        monthly_income = st.number_input(
            "Monthly Income (₹)", min_value=0.0, value=float(merged.get("monthly_income", 60000)), step=1000.0
        )
        employment_type = st.selectbox(
            "Employment Type", ["Salaried", "Self-Employed", "Business Owner", "Unemployed"]
        )
        company_name = st.text_input("Company / Business Name", value=merged.get("company_name", ""))
        existing_emis = st.number_input(
            "Existing Monthly EMIs (₹)", min_value=0.0, value=float(merged.get("existing_emis", 0.0)), step=500.0
        )
        employment_years = st.number_input(
            "Years in Current Job/Business", min_value=0.0, value=float(merged.get("employment_years", 2.0)), step=0.5
        )

    loan_purpose = st.text_input("Loan Purpose", value=merged.get("loan_purpose", ""))
    requested_loan_amount = st.number_input(
        "Requested Loan Amount (₹)", min_value=0.0,
        value=float(merged.get("requested_loan_amount", 500000.0)), step=10000.0,
    )

    # --- NEW: manual CIBIL override for testing — the mock bureau is
    # deterministic per (PAN, DOB), which makes it hard to test specific
    # rule-engine outcomes on demand. Off by default; when off, behavior
    # is exactly as before (mock bureau pull only).
    override_cibil = st.checkbox("🔧 Manually set CIBIL score & late payments (for testing)")
    oc1, oc2 = st.columns(2)
    with oc1:
        manual_cibil_score = st.number_input(
            "CIBIL Score", min_value=300, max_value=900, value=750, step=1, disabled=not override_cibil
        )
    with oc2:
        manual_late_payments = st.number_input(
            "Late Payments", min_value=0, max_value=20, value=0, step=1, disabled=not override_cibil
        )

    submitted = st.form_submit_button("Submit Application →")

if submitted:
    checks = [
        validate_name(full_name),
        validate_age(age),
        validate_pan(pan),
        validate_aadhaar(aadhaar),
        validate_dob(dob.isoformat()),
        validate_income(monthly_income),
        validate_employment_type(employment_type.lower()),
        validate_loan_amount(requested_loan_amount, monthly_income),
    ]
    errors = [msg for ok, msg in checks if not ok]

    if errors:
        st.error("Please fix the following before submitting:")
        for e in errors:
            st.markdown(f"- {e}")
    else:
        with st.spinner("Pulling CIBIL report..."):
            cibil = CibilBureauClient().fetch_score(pan, dob.isoformat())

        # --- NEW: apply the manual override, if the customer/tester set one.
        # Everything downstream (ML features, rule engine, saved record)
        # already reads from `cibil`, so this is the only place we need to
        # touch — no changes needed anywhere else in the pipeline.
        if override_cibil:
            cibil["cibil_score"] = int(manual_cibil_score)
            cibil["late_payments"] = int(manual_late_payments)

        # Approximate assets/liabilities/loan_term for the ML feature vector using
        # interview inputs — a production system would collect these explicitly.
        ml_features = {
            "cibil_score": cibil["cibil_score"],
            "annual_income": monthly_income * 12,
            "employment_years": employment_years,
            "loan_amount": requested_loan_amount,
            "loan_term_months": 60,
            "residential_assets": 0,
            "commercial_assets": 0,
            "existing_liabilities": existing_emis * 12,
        }
        with st.spinner("Running ML risk model..."):
            risk = RiskModel.predict(ml_features)

        applicant = ApplicantSnapshot(
            monthly_income=monthly_income,
            existing_emis=existing_emis,
            employment_years=employment_years,
            cibil_score=cibil["cibil_score"],
            late_payments=cibil["late_payments"],
            risk_category=risk.risk_category,
            default_probability=risk.default_probability,
        )
        decision = evaluate(applicant)
        explanation = build_explanation(decision, risk)

        record = {
            "full_name": full_name, "age": int(age), "pan": pan.upper(), "aadhaar": aadhaar,
            "monthly_income": monthly_income, "employment_type": employment_type,
            "company_name": company_name, "existing_emis": existing_emis,
            "loan_purpose": loan_purpose, "requested_loan_amount": requested_loan_amount,
            "documents_uploaded": st.session_state.get("documents_uploaded", {}),
            "ocr_confidence": st.session_state.get("ocr_confidence"),
            "cibil_score": cibil["cibil_score"], "active_loans": cibil["active_loans"],
            "late_payments": cibil["late_payments"], "credit_utilization": cibil["credit_utilization"],
            "credit_history_years": cibil["credit_history_years"],
            "default_probability": risk.default_probability, "risk_category": risk.risk_category,
            "decision": decision.outcome, "rule_triggered": decision.rule_triggered,
            "approved_loan_amount": decision.approved_loan_amount, "interest_rate": decision.interest_rate,
            "decision_reasons": decision.reasons, "requires_manual_review": decision.requires_manual_review,
        }
        app_id = save_application(record)
        record["id"] = app_id
        st.session_state["last_application"] = record

        st.markdown("---")
        st.markdown(badge(explanation["verdict"], decision.outcome), unsafe_allow_html=True)
        st.markdown(f"### {explanation['verdict']}")
        st.markdown(
            f'<div class="lg-card">CIBIL Score: <b>{cibil["cibil_score"]}</b> &nbsp;|&nbsp; '
            f'Risk: <b>{risk.risk_category}</b> ({risk.default_probability:.1%} default probability) '
            f'&nbsp;|&nbsp; Approved Amount: <b>₹{decision.approved_loan_amount:,.0f}</b> '
            f'&nbsp;|&nbsp; Rate: <b>{decision.interest_rate:.2%}</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown(format_reasons_markdown(explanation))
        st.info("Application saved. Open **Approval Report** in the sidebar to download the PDF.")

        # --- NEW: Multilingual Voice AI — Feature 6: if the customer's
        # interview language wasn't English, generate the same decision
        # explanation in that language and read it aloud. The English
        # explanation above (and the DB record) are unchanged. ---
        decision_lang = preferred if preferred in SUPPORTED_LANGUAGES else st.session_state["detected_lang"]
        if decision_lang != "en":
            with st.spinner(f"Translating decision to {language_name(decision_lang)}..."):
                explanation_en = f"{explanation['verdict']}. {explanation['summary']}\n\n" + \
                    format_reasons_markdown(explanation)
                explanation_translated = translate_text(explanation_en, decision_lang)
            st.markdown(f"##### 🌐 {language_name(decision_lang)}")
            st.markdown(explanation_translated)
            decision_audio = synthesize(explanation_translated, lang=decision_lang)
        else:
            decision_audio = synthesize(
                f"{explanation['verdict']}. {explanation['summary']}", lang="en"
            )
        if decision_audio:
            st.audio(decision_audio, format="audio/mp3", autoplay=True)