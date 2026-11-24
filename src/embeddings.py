"""
embeddings.py
─────────────
Wraps a Hugging Face sentence-transformer model as a LangChain
Embeddings object so it can be used directly with FAISS.

Default model: all-MiniLM-L6-v2  (fast, lightweight, 384-dim vectors)
Override via the EMBEDDING_MODEL env variable.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_embeddings_instance = None   # module-level singleton


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return (and cache) the HuggingFace embeddings model.
    The model is downloaded on first call and cached by sentence-transformers.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"[Embeddings] Loading model: {_EMBEDDING_MODEL} …")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # swap to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Embeddings] Model ready ✓")
    return _embeddings_instance
