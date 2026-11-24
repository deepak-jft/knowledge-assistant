"""
summariser.py
─────────────
Summarise an entire document or a list of text chunks using a
modern LCEL map-reduce approach — no langchain.chains needed.
"""

import os
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")
_llm_instance = None


def _get_llm() -> HuggingFacePipeline:
    """Lazy-load the HuggingFace LLM (shared singleton)."""
    global _llm_instance
    if _llm_instance is None:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model     = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        hf_pipe   = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
        )
        _llm_instance = HuggingFacePipeline(pipeline=hf_pipe)
    return _llm_instance


_MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Summarise the following text concisely:\n\n{text}\n\nSummary:",
)

_COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Combine these section summaries into one coherent final summary:\n\n"
        "{text}\n\nFinal Summary:"
    ),
)


def summarise(documents: list[Document]) -> str:
    """
    Summarise a list of Document chunks via LCEL map-reduce:
      1. Map  — summarise each chunk individually
      2. Reduce — combine all chunk summaries into one final summary

    Args:
        documents: list of Document objects

    Returns:
        A single summary string.
    """
    if not documents:
        return "No content to summarise."

    llm    = _get_llm()
    parser = StrOutputParser()

    # Map: summarise each chunk
    map_chain = _MAP_PROMPT | llm | parser
    chunk_summaries = []
    for doc in documents:
        summary = map_chain.invoke({"text": doc.page_content})
        chunk_summaries.append(summary.strip())

    # Reduce: combine all chunk summaries
    combined_text   = "\n\n".join(chunk_summaries)
    combine_chain   = _COMBINE_PROMPT | llm | parser
    final_summary   = combine_chain.invoke({"text": combined_text})
    return final_summary.strip() or "Could not generate summary."
