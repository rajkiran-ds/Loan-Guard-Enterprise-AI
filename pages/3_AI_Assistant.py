"""
pages/3_AI_Assistant.py
--------------------------
FEATURE 1 — RAG Knowledge Assistant chat UI. Every answer is grounded in
data/bank_policy.pdf via src/rag/qa_chain.py; retrieved source chunks are
shown in an expander so the answer is auditable, not just plausible.
"""

from __future__ import annotations

import streamlit as st

from src.rag.multilingual_chat import ask_multilingual
from src.utils.language import SUPPORTED_LANGUAGES, detect_language, language_name
from src.utils.ui_theme import apply_theme
from src.voice.stt import SpeechToText
from src.voice.tts import synthesize

apply_theme("AI Assistant")

st.markdown('<div class="lg-kicker">Feature 1 · Grounded RAG Assistant</div>', unsafe_allow_html=True)
st.title("💬 Bank Policy Assistant")
st.caption(
    "Answers eligibility, document, fee, EMI, foreclosure, and late-payment questions — "
    "strictly from the bank's policy document. Nothing else."
)
st.caption("🌐 Ask in English, Telugu, Hindi, Tamil, or Kannada — the assistant replies in the same language.")

# --- NEW: Multilingual Voice AI — mode toggle + mic input ---
input_mode = st.radio("Input mode", ["📝 Text", "🎙️ Voice"], horizontal=True, label_visibility="collapsed")
voice_question = None
if input_mode == "🎙️ Voice":
    audio = st.audio_input("Speak your question")
    if audio is not None:
        with st.spinner("Transcribing..."):
            transcription = SpeechToText.transcribe(audio.getvalue())
        if not transcription.engine_available:
            st.warning("Voice engine unavailable in this environment — please use Text mode instead.")
        elif transcription.text:
            voice_question = transcription.text
            st.success(f"Heard: {voice_question}")
        else:
            st.info("Couldn't make out any speech — please try again or switch to Text mode.")

SAMPLE_QUESTIONS = [
    "What is the minimum CIBIL score required?",
    "What documents do I need to apply?",
    "What is the foreclosure charge?",
    "What happens if I miss an EMI payment?",
]

st.session_state.setdefault("chat_history", [])

with st.expander("💡 Try a sample question"):
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("lang_label"):
            st.caption(turn["lang_label"])
        if turn.get("sources"):
            with st.expander("Source policy chunks"):
                for i, chunk in enumerate(turn["sources"], start=1):
                    st.markdown(f"**Chunk {i}:**\n\n{chunk}")
        if turn.get("audio"):
            st.audio(turn["audio"], format="audio/mp3")

user_question = st.chat_input("Ask about eligibility, fees, EMI policy, foreclosure...") or voice_question
if "pending_question" in st.session_state:
    user_question = st.session_state.pop("pending_question")

if user_question:
    # --- NEW: Multilingual Voice AI — detect language, honor the sidebar
    # preference if the user pinned one, otherwise auto-detect per message. ---
    preferred = st.session_state.get("preferred_language", "auto")
    detected_lang = preferred if preferred in SUPPORTED_LANGUAGES else detect_language(user_question)

    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
        st.caption(f"🌐 Detected: {language_name(detected_lang)}")

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking bank policy..."):
                result = ask_multilingual(user_question, lang=detected_lang)
            st.markdown(result.answer)
            lang_label = f"🌐 Replied in {language_name(result.language)}"
            st.caption(lang_label)
            if result.source_chunks:
                with st.expander("Source policy chunks (English, from bank_policy.pdf)"):
                    for i, chunk in enumerate(result.source_chunks, start=1):
                        st.markdown(f"**Chunk {i}:**\n\n{chunk}")

            audio_bytes = synthesize(result.answer, lang=result.language)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.source_chunks,
                    "lang_label": lang_label,
                    "audio": audio_bytes,
                }
            )
        except RuntimeError as exc:
            st.error(str(exc))
