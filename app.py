"""
app.py — Knowledge Assistant
─────────────────────────────
Main Streamlit application entry point.

Run:  streamlit run app.py
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

# ── Lazy imports (heavy models loaded only when needed) ───────────────────────
from src.document_loader import load_document, load_url, load_raw_text
from src.vector_store    import (
    build_index, save_index, load_index,
    add_documents, search, index_exists,
)
from src.llm_chain  import build_qa_chain, ask
from src.summariser import summarise

# Voice input is optional — app works fine without SpeechRecognition/pyaudio
try:
    from src.voice_input import listen_from_microphone, is_microphone_available
except ImportError:
    def is_microphone_available(): return False
    def listen_from_microphone(**_): return ""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.85rem;
        color: #cdd6f4;
    }
    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "db":           None,   # FAISS vector store
        "qa_chain":     None,   # RetrievalQA chain
        "chat_history": [],     # list of {"role": str, "content": str}
        "docs_loaded":  0,      # total chunks indexed
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _index_documents(docs, label: str):
    """Add docs to (or create) the FAISS index and rebuild the QA chain."""
    if not docs:
        st.warning("No content extracted — please check the source.")
        return

    with st.spinner(f"Indexing {len(docs)} chunks from {label} …"):
        if st.session_state.db is None:
            st.session_state.db = build_index(docs)
        else:
            st.session_state.db = add_documents(st.session_state.db, docs)
        save_index(st.session_state.db)
        st.session_state.qa_chain   = build_qa_chain(st.session_state.db)
        st.session_state.docs_loaded += len(docs)

    st.success(f"✅ Indexed **{len(docs)}** chunks from *{label}*")


def _chat(user_message: str):
    """Run a user query through the RAG chain and append to chat history."""
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please load at least one document or URL first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_message})

    with st.spinner("Thinking …"):
        response = ask(st.session_state.qa_chain, user_message)

    answer  = response["answer"]
    sources = response["sources"]

    # Format source snippets
    source_md = ""
    if sources:
        snippets = []
        for doc in sources:
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_info = f" · p.{page+1}" if page != "" else ""
            snippets.append(f"**{src}{page_info}** — {doc.page_content[:200]}…")
        source_md = "\n\n---\n**📎 Sources:**\n" + "\n\n".join(snippets)

    full_response = answer + source_md
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Knowledge Assistant")
    st.caption("Load documents, then ask anything.")
    st.divider()

    # ── Status ────────────────────────────────────────────────────────────────
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Chunks indexed", st.session_state.docs_loaded)
    status_col2.metric(
        "Index on disk",
        "Yes ✅" if index_exists() else "No ❌",
    )
    st.divider()

    # ── Upload files ──────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF, TXT or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("📥 Index Uploaded Files", use_container_width=True):
        for uf in uploaded_files:
            suffix = os.path.splitext(uf.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            docs = load_document(tmp_path)
            os.unlink(tmp_path)
            _index_documents(docs, uf.name)

    st.divider()

    # ── URL ingestion ─────────────────────────────────────────────────────────
    st.markdown("### 🌐 Ingest a Web URL")
    url_input = st.text_input("Paste URL here", placeholder="https://example.com/article")
    if st.button("🔗 Scrape & Index URL", use_container_width=True):
        if url_input.strip():
            try:
                docs = load_url(url_input.strip())
                _index_documents(docs, url_input)
            except RuntimeError as e:
                st.error(str(e))
        else:
            st.warning("Please enter a URL first.")

    st.divider()

    # ── Load existing index ───────────────────────────────────────────────────
    st.markdown("### 💾 Saved Index")
    if st.button("📂 Load Index from Disk", use_container_width=True):
        try:
            with st.spinner("Loading index …"):
                st.session_state.db       = load_index()
                st.session_state.qa_chain = build_qa_chain(st.session_state.db)
            st.success("Index loaded ✅")
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))

    st.divider()

    # ── Summarise ─────────────────────────────────────────────────────────────
    st.markdown("### 📝 Summarise")
    if st.button("✨ Summarise All Loaded Docs", use_container_width=True):
        if st.session_state.db is None:
            st.warning("No documents loaded yet.")
        else:
            with st.spinner("Summarising …"):
                all_docs = st.session_state.db.docstore._dict.values()
                summary  = summarise(list(all_docs))
            st.info(summary)

    st.divider()
    st.caption("Built with LangChain · HuggingFace · FAISS · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🧠 Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask anything about your documents, PDFs, or web pages.</div>', unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 Load a document or URL from the sidebar, then ask me anything!")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# ── Input row ─────────────────────────────────────────────────────────────────
col_text, col_voice, col_clear = st.columns([8, 1, 1])

with col_text:
    user_query = st.chat_input("Ask a question …")

with col_voice:
    mic_available = is_microphone_available()
    if st.button("🎙️", help="Speak your question", disabled=not mic_available):
        with st.spinner("Listening …"):
            spoken = listen_from_microphone()
        if spoken:
            st.toast(f"Heard: {spoken}", icon="🎙️")
            _chat(spoken)
            st.rerun()
        else:
            st.warning("Could not understand audio. Please try again.")

with col_clear:
    if st.button("🗑️", help="Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ── Process text input ────────────────────────────────────────────────────────
if user_query:
    _chat(user_query)
    st.rerun()
