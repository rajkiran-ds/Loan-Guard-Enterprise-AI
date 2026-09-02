"""
src/rag/qa_chain.py
----------------------
FEATURE 1 — RAG Knowledge Assistant.

Wires the FAISS retriever (vectorstore.py) to a Groq-hosted Llama 3 model
through a system prompt that hard-forbids answering from anything except
the retrieved bank_policy.pdf chunks. If retrieval comes back empty or the
LLM can't ground an answer in the provided context, we return
RAG_NOT_FOUND_MESSAGE instead of letting the model guess — this is what
"never hallucinates" means operationally, not just as a prompt request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, RAG_NOT_FOUND_MESSAGE, RAG_TOP_K
from src.rag.vectorstore import build_vectorstore
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the LoanGuard Enterprise AI policy assistant.

RULES (never break these):
1. Answer ONLY using the "POLICY CONTEXT" provided below. Do not use outside
   knowledge about banking, loans, or anything else, even if you believe you
   know the answer.
2. If the POLICY CONTEXT does not contain enough information to answer the
   question, respond with EXACTLY this sentence and nothing else:
   "{not_found}"
3. Never invent numbers, fees, dates, or policy terms that are not literally
   present in the POLICY CONTEXT.
4. Keep answers concise and cite the general topic (e.g. "Per the EMI
   policy section...") without fabricating a page number if one isn't
   given.

POLICY CONTEXT:
{context}
"""


@dataclass
class RAGAnswer:
    answer: str
    source_chunks: list[str] = field(default_factory=list)
    grounded: bool = True


class PolicyAssistant:
    """Lazy-loading singleton so Streamlit reruns reuse the same FAISS
    index and LLM client instead of rebuilding them on every interaction."""

    _vectorstore = None
    _llm = None

    @classmethod
    def _get_vectorstore(cls):
        if cls._vectorstore is None:
            cls._vectorstore = build_vectorstore()
        return cls._vectorstore

    @classmethod
    def _get_llm(cls) -> ChatGroq:
        if cls._llm is None:
            if not GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env file "
                    "(see .env.example)."
                )
            cls._llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
        return cls._llm

    @classmethod
    def ask(cls, question: str, top_k: int = RAG_TOP_K) -> RAGAnswer:
        if not question or not question.strip():
            return RAGAnswer(answer=RAG_NOT_FOUND_MESSAGE, grounded=False)

        vectorstore = cls._get_vectorstore()
        docs: list[Document] = vectorstore.similarity_search(question, k=top_k)

        if not docs:
            logger.info("No chunks retrieved for question: %s", question)
            return RAGAnswer(answer=RAG_NOT_FOUND_MESSAGE, grounded=False)

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        system_message = _SYSTEM_PROMPT.format(not_found=RAG_NOT_FOUND_MESSAGE, context=context)

        llm = cls._get_llm()
        response = llm.invoke(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ]
        )
        answer_text = response.content.strip()
        grounded = RAG_NOT_FOUND_MESSAGE not in answer_text

        logger.info("RAG answer grounded=%s for question=%r", grounded, question)
        return RAGAnswer(
            answer=answer_text,
            source_chunks=[d.page_content for d in docs],
            grounded=grounded,
        )
