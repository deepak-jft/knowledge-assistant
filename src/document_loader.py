"""
document_loader.py
──────────────────
Handles ingestion of documents from multiple sources:
  - PDF files (via pypdf)
  - Plain text / DOCX files
  - Web URLs (via requests + BeautifulSoup)
  - Raw text strings

Returns a list of LangChain Document objects ready for chunking.
"""

import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )


# ── PDF ──────────────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and split it into chunks."""
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    return _get_splitter().split_documents(pages)


# ── Text / DOCX ───────────────────────────────────────────────────────────────

def load_text_file(file_path: str) -> list[Document]:
    """Load a .txt file and split it into chunks."""
    loader = TextLoader(file_path, encoding="utf-8")
    docs   = loader.load()
    return _get_splitter().split_documents(docs)


def load_docx_file(file_path: str) -> list[Document]:
    """Load a .docx file and split it into chunks."""
    loader = Docx2txtLoader(file_path)
    docs   = loader.load()
    return _get_splitter().split_documents(docs)


# ── Web URL ───────────────────────────────────────────────────────────────────

def load_url(url: str) -> list[Document]:
    """
    Fetch a web page, extract visible text, and split it into chunks.
    Falls back gracefully on network or parsing errors.
    """
    try:
        headers  = {"User-Agent": "Mozilla/5.0 (Knowledge-Assistant/1.0)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        doc  = Document(page_content=text, metadata={"source": url})
        return _get_splitter().split_documents([doc])

    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch URL '{url}': {exc}") from exc


# ── Raw text ──────────────────────────────────────────────────────────────────

def load_raw_text(text: str, source: str = "manual_input") -> list[Document]:
    """Wrap arbitrary text in a Document and split."""
    doc = Document(page_content=text, metadata={"source": source})
    return _get_splitter().split_documents([doc])


# ── Auto-dispatch ─────────────────────────────────────────────────────────────

def load_document(file_path: str) -> list[Document]:
    """
    Auto-detect file type from extension and load accordingly.
    Supported: .pdf  .txt  .docx
    """
    ext = os.path.splitext(file_path)[-1].lower()
    dispatch = {
        ".pdf":  load_pdf,
        ".txt":  load_text_file,
        ".docx": load_docx_file,
    }
    if ext not in dispatch:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {list(dispatch)}")
    return dispatch[ext](file_path)
