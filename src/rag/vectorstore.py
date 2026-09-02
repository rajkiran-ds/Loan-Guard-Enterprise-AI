"""
src/rag/vectorstore.py
-------------------------
Builds (or loads a cached) FAISS index over the bank policy chunks using
HuggingFace sentence-transformer embeddings. The index is persisted to
disk at VECTORSTORE_PATH so the app doesn't re-embed the policy document
on every Streamlit rerun.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, VECTORSTORE_PATH
from src.rag.loader import load_policy_chunks
from src.utils.logger import get_logger

logger = get_logger(__name__)

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Load a cached FAISS index if present, otherwise build one from the
    bank policy PDF and persist it."""
    embeddings = get_embeddings()

    if VECTORSTORE_PATH.exists() and not force_rebuild:
        logger.info("Loading cached FAISS index from %s", VECTORSTORE_PATH)
        return FAISS.load_local(
            str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True
        )

    logger.info("Building FAISS index from bank policy PDF...")
    chunks = load_policy_chunks()
    store = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_PATH))
    logger.info("FAISS index built and saved to %s", VECTORSTORE_PATH)
    return store
