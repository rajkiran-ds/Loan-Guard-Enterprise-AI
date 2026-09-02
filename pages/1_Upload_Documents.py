"""
pages/1_Upload_Documents.py
------------------------------
FEATURE 3 — OCR document upload. Extracted fields are pushed into
st.session_state["ocr_prefill"] so the "Apply for Loan" interview page
can pre-fill its form. Degrades gracefully to manual entry if PaddleOCR
isn't available in the current environment (see src/ocr/extractor.py).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import UPLOADS_DIR
from src.ocr.extractor import OCRExtractor
from src.utils.ui_theme import apply_theme

apply_theme("Upload Documents")
st.session_state.setdefault("ocr_prefill", {})

st.markdown('<div class="lg-kicker">Feature 3</div>', unsafe_allow_html=True)
st.title("📎 Upload Documents")
st.markdown(
    "Upload PAN, Aadhaar, and a salary slip. Extracted fields will auto-fill the "
    "**Apply for Loan** interview — you can always correct them manually there."
)

DOC_TYPES = {
    "PAN Card": "pan_card",
    "Aadhaar Card": "aadhaar_card",
    "Salary Slip": "salary_slip",
}

extractor = OCRExtractor()
cols = st.columns(3)

for (label, key), col in zip(DOC_TYPES.items(), cols):
    with col:
        st.markdown(f'<div class="lg-card"><h4>{label}</h4></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(f"Upload {label}", type=["png", "jpg", "jpeg"], key=key)
        if uploaded is not None:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = UPLOADS_DIR / f"{key}_{uploaded.name}"
            save_path.write_bytes(uploaded.getbuffer())
            st.image(uploaded, use_container_width=True)

            with st.spinner("Running OCR..."):
                result = extractor.extract(save_path)

            uploaded_docs = st.session_state.get("documents_uploaded", {})
            uploaded_docs[key] = str(save_path)
            st.session_state["documents_uploaded"] = uploaded_docs

            if not result.engine_available:
                st.warning(
                    "OCR engine unavailable in this environment — please fill in the "
                    "matching fields manually on the Apply page."
                )
            elif result.extracted_fields:
                st.session_state["ocr_prefill"].update(result.extracted_fields)
                st.session_state["ocr_confidence"] = result.confidence
                st.success(f"Extracted: {result.extracted_fields}")
            else:
                st.info("Document uploaded — no fields could be auto-parsed; enter them manually.")

st.markdown("---")
if st.session_state["ocr_prefill"]:
    st.markdown("#### Auto-filled fields so far")
    st.json(st.session_state["ocr_prefill"])
else:
    st.caption("No fields extracted yet. Upload a document above, or skip straight to Apply for Loan.")
