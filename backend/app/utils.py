# utils.py
import os
import urllib.parse
import logging
from typing import List

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# optional Gemini
import google.generativeai as genai

from config import CHROMA_BASE, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, GOOGLE_API_KEY, LLM_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")

# init embedding model once
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# configure genai (if key present)
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("GenAI configured")
    except Exception as e:
        logger.warning("Could not configure GenAI: %s", e)

def _coll_name(filename: str) -> str:
    return urllib.parse.quote_plus(filename)

def pdf_to_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [p.get_text() for p in doc]
    return "\n".join(pages)

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    # filter too-small
    return [c.strip() for c in chunks if c and len(c.strip()) > 40]

def create_vectorstore_for_file(filename: str, chunks: List[str]):
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_BASE, coll)
    os.makedirs(persist_dir, exist_ok=True)
    docs = [Document(page_content=c) for c in chunks]
    vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    try:
        vs.persist()
    except Exception:
        # some chroma versions auto-persist
        pass
    return vs




import json
import re

def summarize_chunks_with_gemini_json(chunks: List[str], max_chunks: int = 8) -> List[str]:
    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks[:max_chunks])])
    prompt = f"""
You are an expert summarizer. Using ONLY the context below, produce exactly 6 short bullet points that summarize the document.
Return the result as a JSON array of strings ONLY (for example: ["point one", "point two", ...]).
Do NOT include any text outside the JSON array.

Context:
{context}

JSON summary:
"""
    raw = genai_generate(prompt)
    # Attempt to extract JSON array from output
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(1))
            # ensure strings
            return [str(x).strip() for x in arr if x and str(x).strip()]
        except Exception:
            pass
    # Fallback: split lines into bullets if JSON parsing fails
    lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
    return lines[:6]
