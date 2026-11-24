"""
vector_store.py
───────────────
Manages a local FAISS vector index:
  - build_index   : create a new index from documents
  - save_index    : persist to disk
  - load_index    : reload from disk
  - add_documents : incrementally add new docs to an existing index
  - search        : semantic similarity search
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embeddings import get_embeddings
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
TOP_K      = int(os.getenv("TOP_K", 5))


def build_index(documents: list[Document]) -> FAISS:
    """Create a FAISS index from a list of LangChain Document objects."""
    if not documents:
        raise ValueError("Cannot build index: document list is empty.")
    print(f"[VectorStore] Building index from {len(documents)} chunks …")
    embeddings = get_embeddings()
    db = FAISS.from_documents(documents, embeddings)
    print("[VectorStore] Index built ✓")
    return db


def save_index(db: FAISS, path: str = INDEX_PATH) -> None:
    """Persist the FAISS index to disk."""
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    print(f"[VectorStore] Index saved → {path}")


def load_index(path: str = INDEX_PATH) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No FAISS index found at '{path}'. Build one first.")
    embeddings = get_embeddings()
    db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"[VectorStore] Index loaded ← {path}")
    return db


def add_documents(db: FAISS, documents: list[Document]) -> FAISS:
    """Add new documents to an existing FAISS index."""
    if not documents:
        return db
    db.add_documents(documents)
    print(f"[VectorStore] Added {len(documents)} chunks to index.")
    return db


def search(db: FAISS, query: str, k: int = TOP_K) -> list[Document]:
    """Return the top-k most semantically similar documents for a query."""
    results = db.similarity_search(query, k=k)
    return results


def index_exists(path: str = INDEX_PATH) -> bool:
    """Check whether a saved FAISS index already exists on disk."""
    return os.path.exists(os.path.join(path, "index.faiss"))
