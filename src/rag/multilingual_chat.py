"""
src/rag/multilingual_chat.py
--------------------------------
NEW — Multilingual Voice AI feature.

Extends src/rag/qa_chain.py (unmodified) so the AI Assistant can be asked
a question in Telugu, Hindi, Tamil, Kannada, or English and reply in that
same language — without translating or re-indexing bank_policy.pdf.

The policy document stays English-only in FAISS. Retrieval is unchanged;
only the answer-generation prompt changes, instructing the LLM to reason
over the retrieved English chunks but respond in the target language. This
keeps the existing RAG pipeline (LangChain + FAISS + Groq/Llama 3) exactly
as-is — this module only adds a language-aware prompt on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import GROQ_API_KEY, RAG_TOP_K
from src.rag.qa_chain import PolicyAssistant
from src.utils.language import SUPPORTED_LANGUAGES, detect_language, language_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Localized "not found in policy" refusal — kept static (no LLM call needed)
# so the one guarantee that matters most (never hallucinate) never depends
# on a translation call succeeding.
_NOT_FOUND_TRANSLATIONS = {
    "en": "I couldn't find this in the bank policy.",
    "hi": "मुझे यह जानकारी बैंक की नीति में नहीं मिली।",
    "te": "ఈ సమాచారం బ్యాంక్ పాలసీలో నాకు కనబడలేదు.",
    "ta": "இந்தத் தகவலை வங்கியின் கொள்கையில் என்னால் கண்டறிய முடியவில்லை.",
    "kn": "ಈ ಮಾಹಿತಿ ಬ್ಯಾಂಕ್ ನೀತಿಯಲ್ಲಿ ನನಗೆ ಕಂಡುಬಂದಿಲ್ಲ.",
}

_MULTILINGUAL_SYSTEM_PROMPT = """You are the LoanGuard Enterprise AI policy assistant.

RULES (never break these):
1. Answer ONLY using the "POLICY CONTEXT" provided below (it is in English).
   Do not use outside knowledge about banking, loans, or anything else.
2. Respond ENTIRELY in {language_name} ({language_code}), regardless of what
   language the POLICY CONTEXT is written in. Translate the relevant facts
   into {language_name} yourself — do not answer in English unless
   {language_name} IS English.
3. If the POLICY CONTEXT does not contain enough information to answer,
   respond with EXACTLY this sentence in {language_name} and nothing else:
   "{not_found}"
4. Never invent numbers, fees, dates, or policy terms not literally present
   in the POLICY CONTEXT.

POLICY CONTEXT (English):
{context}
"""


@dataclass
class MultilingualAnswer:
    answer: str
    language: str
    source_chunks: list[str] = field(default_factory=list)
    grounded: bool = True


def ask_multilingual(question: str, lang: str | None = None, top_k: int = RAG_TOP_K) -> MultilingualAnswer:
    """Same retrieval as PolicyAssistant.ask(), but the answer is generated
    in `lang` (auto-detected from `question` if not given)."""
    target_lang = lang or detect_language(question)
    if target_lang not in SUPPORTED_LANGUAGES:
        target_lang = "en"

    if not question or not question.strip():
        return MultilingualAnswer(
            answer=_NOT_FOUND_TRANSLATIONS[target_lang], language=target_lang, grounded=False
        )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )

    # Reuse the existing cached FAISS index and Groq client rather than
    # duplicating their setup/config.
    vectorstore = PolicyAssistant._get_vectorstore()
    docs = vectorstore.similarity_search(question, k=top_k)

    if not docs:
        logger.info("No chunks retrieved for question (lang=%s): %s", target_lang, question)
        return MultilingualAnswer(
            answer=_NOT_FOUND_TRANSLATIONS[target_lang], language=target_lang, grounded=False
        )

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    system_message = _MULTILINGUAL_SYSTEM_PROMPT.format(
        language_name=language_name(target_lang),
        language_code=target_lang,
        not_found=_NOT_FOUND_TRANSLATIONS[target_lang],
        context=context,
    )

    llm = PolicyAssistant._get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]
    )
    answer_text = response.content.strip()
    grounded = _NOT_FOUND_TRANSLATIONS[target_lang] not in answer_text

    logger.info("Multilingual RAG answer grounded=%s lang=%s", grounded, target_lang)
    return MultilingualAnswer(
        answer=answer_text,
        language=target_lang,
        source_chunks=[d.page_content for d in docs],
        grounded=grounded,
    )


def translate_text(text: str, target_lang: str) -> str:
    """Small helper used to render the final loan decision/explanation
    (produced in English by the unchanged rule engine + explainer) in the
    customer's language. Not used for policy Q&A — that path never
    translates, it generates directly in the target language."""
    if target_lang not in SUPPORTED_LANGUAGES or target_lang == "en":
        return text
    if not GROQ_API_KEY or not text.strip():
        return text

    llm = PolicyAssistant._get_llm()
    prompt = (
        f"Translate the following loan-decision explanation into "
        f"{language_name(target_lang)}. Preserve all numbers, currency "
        f"amounts, and percentages exactly. Output ONLY the translation, "
        f"no preamble.\n\n{text}"
    )
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as exc:  # noqa: BLE001 — a translation failure should show English, not crash
        logger.warning("Translation to %s failed (%s) — showing English.", target_lang, exc)
        return text
