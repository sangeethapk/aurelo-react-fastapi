# main.py
import os
import time
from typing import Optional
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import pdf_bytes_to_text, split_text_into_chunks, create_or_replace_chroma_collection, make_retriever_for_file, get_top_k_chunks, genai_generate
from config import CHROMA_PERSIST_DIR, HOST, PORT, GOOGLE_API_KEY   


app = FastAPI(title="AuraLearn RAG Backend (Chunking + Chroma)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WARNING: open to all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory state so frontend can poll status
# session state keyed by filename: { "status": "processing"|"ready"|"error", "summary": str, "created": timestamp }
SESSIONS = {}


# Upload endpoint (synchronous chunk+embed store; returns "ready" status but no summary)
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    data = await file.read()
    filename = file.filename

    # process synchronously: extract -> chunk -> embed -> store
    try:
        full_text = pdf_bytes_to_text(data)
        chunks = split_text_into_chunks(full_text)
        create_or_replace_chroma_collection(filename, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # Do NOT return a text summary here. Just indicate the file is ready for summary generation.
    return {"status": "ready", "filename": filename, "chunks": len(chunks)}

# Request body for summary endpoint
class SummaryRequest(BaseModel):
    use_llm: Optional[bool] = True   # frontend may override; if False, do extractive fallback
    top_k: Optional[int] = 8        # how many chunks to use for summarization
    # you can add more options later (length, style, etc.)

# Summary endpoint (runs only when user clicks Generate Summary)
@app.post("/summary/{filename}")
def generate_summary(filename: str, req: SummaryRequest):
    # ensure collection exists / was processed
    # note: if you keep an in-memory SESSIONS map, you could check SESSIONS[filename]['status']=='ready'
    # but safer to attempt to create retriever and fetch chunks; return 404 if no collection dir
    try:
        retriever = make_retriever_for_file(filename, k=req.top_k or 8)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found for {filename}: {e}")

    # get top-k chunks
    chunks = get_top_k_chunks(retriever, "summarize the document", k=req.top_k or 8)
    if not chunks:
        raise HTTPException(status_code=500, detail="Could not retrieve document chunks for summarization")

    combined_context = "\n\n".join(chunks)

    # If user asked to use LLM and we have a key, call LLM
    use_llm = bool(req.use_llm)
    # If genai not configured, fallback to extractive
    try:
        if use_llm:
            # genai_generate raises if key missing (or you may check GOOGLE_API_KEY in config)
            prompt = f"""
You are an AI summarizer. Summarize the following textbook content into a concise summary of about 6 bullet points:

Text Context:
{combined_context}

Summary:
"""
            summary_text = genai_generate(prompt)
            return {"summary": summary_text, "method": "llm", "chunks_used": len(chunks)}
    except Exception as e:
        # Log the LLM failure and continue with extractive fallback
        # (in production you may want to bubble this up as an error)
        print(f"LLM summarization failed: {e}")

    # Extractive fallback (simple): return the top N chunks concatenated and trimmed
    # For a slightly nicer extractive summary, take first sentences of each chunk
    def extractive_summary_from_chunks(chunks_list, max_chars=800):
        out = []
        total = 0
        for c in chunks_list:
            snippet = c.strip().replace("\n", " ")
            take = snippet[:400]  # take first 400 chars of each chunk
            out.append(take)
            total += len(take)
            if total > max_chars:
                break
        return "\n\n".join(out)[:max_chars]

    fallback_summary = extractive_summary_from_chunks(chunks, max_chars=1000)
    return {"summary": fallback_summary, "method": "extractive", "chunks_used": len(chunks)}

@app.get("/status/{filename}")
def status(filename: str):
    s = SESSIONS.get(filename)
    if not s:
        return {"status": "not_found"}
    return {"status": s.get("status"), "chunks": s.get("chunks", 0), "summary": s.get("summary"), "error": s.get("error")}

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    use_llm: Optional[bool] = False

@app.post("/qa/{filename}")
def qa(filename: str, req: QARequest):
    # ensure collection is ready
    s = SESSIONS.get(filename)
    if not s or s.get("status") != "ready":
        raise HTTPException(status_code=400, detail="File not ready or not found")

    # build retriever and fetch top-k context chunks
    retriever = make_retriever_for_file(filename, k=req.top_k or 4)
    chunks = get_top_k_chunks(retriever, req.question, k=req.top_k or 4)
    combined_context = "\n\n".join(chunks)

    if req.use_llm:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=400, detail="LLM key not configured")
        prompt = f"""Use ONLY the context below to answer the question. If answer not found, reply 'Answer is not available in the textbook.'

Question: {req.question}

Context:
{combined_context}

Answer:
"""
        try:
            answer = genai_generate(prompt)
            return {"answer": answer, "context": combined_context}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    else:
        # return retrieved context so frontend can decide
        return {"context": combined_context, "chunks": chunks}
