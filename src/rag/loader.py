"""
src/rag/loader.py
--------------------
Loads bank_policy.pdf and splits it into overlapping chunks ready for
embedding. Kept separate from vectorstore.py so the chunking strategy can
be tuned (or swapped for a different splitter) without touching FAISS code.
"""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import POLICY_PDF_PATH, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_policy_chunks() -> list[Document]:
    if not POLICY_PDF_PATH.exists():
        raise FileNotFoundError(
            f"Bank policy PDF not found at {POLICY_PDF_PATH}. "
            f"Run `python -m src.rag.build_policy_pdf` or place your own PDF there."
        )

    loader = PyPDFLoader(str(POLICY_PDF_PATH))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    logger.info("Loaded %d pages -> %d chunks from %s", len(pages), len(chunks), POLICY_PDF_PATH)
    return chunks