"""
pages/5_Approval_Report.py
------------------------------
FEATURE 9 — Executive PDF Approval Report. Defaults to the application
just submitted on the Apply page (st.session_state["last_application"]),
but any application can be looked up by ID from the database.
"""

from __future__ import annotations

import streamlit as st

from src.database.db import get_application
from src.decision.report_generator import generate_approval_report
from src.utils.ui_theme import apply_theme, badge

apply_theme("Approval Report")

st.markdown('<div class="lg-kicker">Feature 9</div>', unsafe_allow_html=True)
st.title("📄 Executive Approval Report")

last_app = st.session_state.get("last_application")
default_id = last_app["id"] if last_app else 1

app_id = st.number_input("Application ID", min_value=1, value=int(default_id), step=1)

record = None
if last_app and last_app["id"] == app_id:
    record = last_app
else:
    row = get_application(int(app_id))
    if row is not None:
        record = {c.name: getattr(row, c.name) for c in row.__table__.columns}

if record is None:
    st.warning("No application found with that ID. Submit one via **Apply for Loan** first.")
    st.stop()

st.markdown(badge(record["decision"], record["decision"]), unsafe_allow_html=True)
st.markdown(f"### Application #{record['id']} — {record['full_name']}")

c1, c2, c3 = st.columns(3)
c1.metric("CIBIL Score", record["cibil_score"])
c2.metric("Risk Category", record["risk_category"])
c3.metric("Approved Amount", f"₹{record['approved_loan_amount']:,.0f}")

with st.expander("Full decision reasoning"):
    for reason in record.get("decision_reasons") or []:
        st.markdown(f"- {reason}")

if st.button("Generate PDF Report"):
    with st.spinner("Rendering report..."):
        pdf_path = generate_approval_report(record)
    st.success("Report generated.")
    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download Approval Report (PDF)",
            data=f.read(),
            file_name=pdf_path.name,
            mime="application/pdf",
        )
