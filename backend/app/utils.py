# utils.py
import os
import uuid
import urllib.parse
import logging
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# optional LLM (Gemini) wrapper
import google.generativeai as genai

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, GOOGLE_API_KEY, LLM_MODEL

logger = logging.getLogger("aura_backend")
logging.basicConfig(level=logging.INFO)

# configure genai only if key exists
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        logger.warning("Could not configure genai: %s", e)

# initialize embeddings once (LangChain wrapper)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Ensure chroma persist dir exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


# ---------- PDF -> Text -> Chunks ----------
def pdf_bytes_to_text(file_bytes: bytes) -> str:
    """Extract full text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text())
    full_text = "\n".join(texts)
    return full_text


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Use LangChain's RecursiveCharacterTextSplitter to create semantically-sound chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    # filter empty/micro chunks
    chunks = [c.strip() for c in chunks if c and len(c.strip()) > 20]
    return chunks


# ---------- Chroma vector store management ----------
def _make_collection_name(filename: str) -> str:
    # url-encode to get safe collection name
    return urllib.parse.quote_plus(filename)


def create_or_replace_chroma_collection(filename: str, chunks: List[str]) -> Chroma:
    """
    Create (or replace) a Chroma collection for this filename and upsert embeddings.
    Returns the Chroma vectorstore object.
    """
    collection_name = _make_collection_name(filename)

    # Build Documents list for LangChain/Chroma API
    docs = [Document(page_content=c) for c in chunks]

    # If collection exists, Chroma.from_documents with same persist_directory and collection name will replace content.
    # We'll use a persist_directory per filename to keep things simple and avoid concurrency issues.
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection_name)
    os.makedirs(persist_dir, exist_ok=True)

    # If the persist dir already contains an old store, you can choose to remove it first.
    # Here we just create a new store in that directory (Chroma will manage files).
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    try:
        vectorstore.persist()
    except Exception:
        # newer chroma may auto-persist; ignore
        pass
    return vectorstore


# ---------- Retrieval ----------
def make_retriever_for_file(filename: str, k: int = 4):
    """Return a retriever for that file's chroma collection (k results by default)."""
    collection_name = _make_collection_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection_name)
    # create Chroma object pointed at the collection directory
    vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever


def get_top_k_chunks(retriever, query: str, k: int = 4) -> List[str]:
    """
    Robustly fetch top-k chunks from retriever. Handles different retriever implementations.
    Returns plain list of chunk texts.
    """
    docs = []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)[:k]
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)[:k]
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)[:k]
        else:
            # attempt fallback: call retriever with query attribute
            docs = retriever(query)[:k]
    except Exception:
        # fail gracefully
        logger.exception("Retriever call failed; returning empty list")
        docs = []

    texts = []
    for d in docs:
        txt = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        texts.append(txt)
    return texts


# ---------- Optional simple LLM wrapper (Gemini) ----------
def genai_generate(prompt: str, model_name: str = LLM_MODEL) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return getattr(response, "text", str(response))
