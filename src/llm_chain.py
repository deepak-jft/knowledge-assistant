"""
llm_chain.py
────────────
Builds and runs the RAG (Retrieval-Augmented Generation) chain using
modern LangChain LCEL (no deprecated langchain.chains required):
  1. Retrieve top-k relevant chunks from FAISS
  2. Format them as context in a prompt
  3. Pass to a Hugging Face LLM and return the answer

Default LLM: google/flan-t5-large  (instruction-following, runs on CPU)
Override via LLM_MODEL env variable.
"""

import os
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")
TOP_K     = int(os.getenv("TOP_K", 5))

_llm_instance = None   # singleton


# ── Prompt template ───────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI knowledge assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
)


# ── LLM loader ───────────────────────────────────────────────────────────────

def _get_llm() -> HuggingFacePipeline:
    global _llm_instance
    if _llm_instance is None:
        print(f"[LLM] Loading model: {LLM_MODEL} …")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model     = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )
        _llm_instance = HuggingFacePipeline(pipeline=hf_pipeline)
        print("[LLM] Model ready ✓")
    return _llm_instance


def _format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved doc chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ── LCEL RAG chain ────────────────────────────────────────────────────────────

def build_qa_chain(db: FAISS) -> dict:
    """
    Build a LCEL RAG chain over the given FAISS vector store.
    Returns a dict containing the chain and retriever for reuse.
    """
    llm       = _get_llm()
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    # LCEL pipeline: retrieve → format → prompt → llm → parse
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(retriever.invoke(x["question"])))
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def ask(chain_bundle: dict, question: str) -> dict:
    """
    Run a question through the LCEL RAG chain.

    Returns:
        {
            "answer":  str,
            "sources": list[Document]
        }
    """
    chain     = chain_bundle["chain"]
    retriever = chain_bundle["retriever"]

    # Retrieve source docs for citation display
    source_docs = retriever.invoke(question)
    answer      = chain.invoke({"question": question})

    return {
        "answer":  answer.strip() if answer else "No answer generated.",
        "sources": source_docs,
    }
