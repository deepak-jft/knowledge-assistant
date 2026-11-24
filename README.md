# 🧠 Knowledge Assistant

> An advanced AI-powered Knowledge Assistant built with Python that lets you upload documents, scrape web content, and have intelligent conversations with your own data — all through a sleek Streamlit interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack & Tools](#tech-stack--tools)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

**Knowledge Assistant** is a Retrieval-Augmented Generation (RAG) application that transforms your documents and web content into an interactive, conversational knowledge base. Powered by LangChain, Hugging Face models, and FAISS vector search, it allows anyone to query large volumes of text using natural language — no manual searching required.

Whether you're a researcher, student, developer, or enterprise user, Knowledge Assistant makes your information instantly accessible through smart semantic search and AI-generated answers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF / Document Upload** | Upload and query PDF files, text documents, and more |
| 🌐 **Web Scraping / URL Ingestion** | Ingest content directly from URLs for instant indexing |
| 💬 **Conversational Q&A** | Ask questions and get precise AI-generated answers from your data |
| 🔎 **Semantic Search** | Deep vector-based search to find the most relevant content |
| 📝 **Document Summarisation** | Automatically summarise long documents in seconds |
| 🎙️ **Voice Input** | Speak your queries for a hands-free experience |
| 🖥️ **Streamlit Web UI** | Clean, interactive browser-based interface |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Web UI                   │
│         (Document upload, Chat, Voice input)        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               LangChain Orchestration                │
│   (Document loading → Chunking → Retrieval → LLM)   │
└──────┬─────────────────────────────┬────────────────┘
       │                             │
┌──────▼──────┐              ┌───────▼────────────────┐
│  FAISS      │              │  Hugging Face Models    │
│  Vector     │              │  (Embeddings + LLM)     │
│  Store      │              │                         │
└─────────────┘              └────────────────────────┘
```

**Flow:**
1. User uploads a document or provides a URL
2. Content is split into chunks by LangChain's text splitter
3. Chunks are embedded using a Hugging Face sentence-transformer model
4. Embeddings are stored in a local **FAISS** vector index
5. On query, the most relevant chunks are retrieved semantically
6. A Hugging Face LLM generates a grounded answer from the retrieved context
7. The response is displayed in the Streamlit chat interface

---

## 🛠️ Tech Stack & Tools

### 🐍 Core Language
| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Primary programming language |

### 🤖 AI / LLM Framework
| Tool | Purpose | Why We Use It |
|---|---|---|
| **[LangChain](https://python.langchain.com/)** | RAG orchestration framework | Chains together document loading, splitting, retrieval, and LLM calls in a clean pipeline |
| **[Hugging Face Transformers](https://huggingface.co/docs/transformers)** | LLM inference | Provides access to powerful open-source LLMs (e.g., FLAN-T5, Mistral, LLaMA) without API costs |
| **[Hugging Face `sentence-transformers`](https://sbert.net/)** | Text embedding | Converts text chunks into dense vector representations for semantic search |
| **[`langchain-huggingface`](https://pypi.org/project/langchain-huggingface/)** | LangChain ↔ HuggingFace bridge | Integrates Hugging Face models cleanly into LangChain pipelines |

### 🗄️ Vector Database
| Tool | Purpose | Why We Use It |
|---|---|---|
| **[FAISS](https://faiss.ai/)** (Facebook AI Similarity Search) | Local vector store | Ultra-fast similarity search on CPU/GPU; no cloud dependency; perfect for local RAG |

### 📄 Document Processing
| Tool | Purpose | Why We Use It |
|---|---|---|
| **[PyPDF2](https://pypdf2.readthedocs.io/)** / **[pypdf](https://pypdf.readthedocs.io/)** | PDF parsing | Extracts text content from PDF files |
| **[LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)** | Multi-format ingestion | Load PDFs, text files, CSVs, HTML and more through a unified interface |
| **[LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)** | Chunking | Splits large documents into overlapping chunks for accurate retrieval |
| **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** / **[Requests](https://requests.readthedocs.io/)** | Web scraping | Fetches and parses HTML from URLs for ingestion |

### 🎙️ Voice Input
| Tool | Purpose | Why We Use It |
|---|---|---|
| **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)** | Speech-to-text | Converts microphone input to text queries using Google Speech API or offline engines |
| **[PyAudio](https://pypi.org/project/PyAudio/)** | Audio stream capture | Captures real-time audio from the microphone |

### 🖥️ User Interface
| Tool | Purpose | Why We Use It |
|---|---|---|
| **[Streamlit](https://streamlit.io/)** | Web application framework | Rapidly builds interactive data apps in pure Python with minimal boilerplate |

### ⚙️ Utilities
| Tool | Purpose |
|---|---|
| **[python-dotenv](https://pypi.org/project/python-dotenv/)** | Manage API keys and environment variables via `.env` file |
| **[NumPy](https://numpy.org/)** | Numerical operations required by FAISS and embedding models |
| **[tqdm](https://tqdm.github.io/)** | Progress bars for document ingestion loops |

---

## 📋 Requirements

- Python **3.9** or higher
- `pip` package manager
- At least **4 GB RAM** (8 GB recommended for larger models)
- Internet connection (for downloading models on first run and web scraping)

> **Optional:** A CUDA-compatible GPU for faster inference with Hugging Face models.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/deepak-jft/knowledge-assistant.git
cd knowledge-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Optional: Only needed if using Hugging Face Inference API (not local models)
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

---

## 🚀 Usage

### Start the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Workflow

1. **Upload a Document** — Use the sidebar to upload a PDF or text file
2. **Or Enter a URL** — Paste any web URL to scrape and index its content
3. **Ask Questions** — Type (or speak) your query in the chat box
4. **Get Answers** — The assistant retrieves relevant chunks and generates a grounded response
5. **Summarise** — Use the "Summarise" button to get a quick summary of any loaded document

---

## 📁 Project Structure

```
knowledge-assistant/
│
├── app.py                  # Streamlit main app entry point
├── requirements.txt        # All Python dependencies
├── .env                    # Environment variables (not committed)
├── .gitignore
├── README.md
│
├── src/
│   ├── document_loader.py  # PDF, URL, and file ingestion logic
│   ├── embeddings.py       # Hugging Face embedding pipeline
│   ├── vector_store.py     # FAISS index creation and retrieval
│   ├── llm_chain.py        # LangChain RAG chain setup
│   ├── summariser.py       # Document summarisation module
│   └── voice_input.py      # Speech-to-text microphone capture
│
├── data/
│   └── faiss_index/        # Persisted FAISS vector index (auto-generated)
│
└── assets/
    └── logo.png            # App logo / branding
```

---

## ⚙️ How It Works

### 1. Document Ingestion
- PDFs and text files are loaded via **LangChain Document Loaders**
- Web pages are fetched using **Requests + BeautifulSoup4**
- Content is split into overlapping chunks (e.g., 500 tokens, 50 token overlap) using `RecursiveCharacterTextSplitter`

### 2. Embedding & Indexing
- Each chunk is converted to a dense vector using a **Hugging Face sentence-transformer** model (e.g., `all-MiniLM-L6-v2`)
- Vectors are stored and indexed locally in **FAISS**

### 3. Query & Retrieval
- The user's question is embedded using the same model
- FAISS performs a **cosine similarity search** to retrieve the top-k most relevant chunks

### 4. Answer Generation
- Retrieved chunks are passed as context to a **Hugging Face LLM** (e.g., `google/flan-t5-large`, `mistralai/Mistral-7B-Instruct-v0.1`)
- The LLM generates a grounded, context-aware answer
- The answer and source chunks are displayed in the **Streamlit** chat UI

### 5. Voice Input
- **SpeechRecognition + PyAudio** captures microphone audio
- Audio is transcribed to text and fed directly into the query pipeline

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to your branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Built with ❤️ using Python, LangChain, Hugging Face, FAISS & Streamlit</strong>
</div>
