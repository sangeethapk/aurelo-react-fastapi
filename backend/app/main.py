import os
import json
import re
import shutil
import urllib.parse
import logging
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz  # PyMuPDF
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
UPLOAD_DIR = "./uploads"
CHROMA_PERSIST_DIR = "./chroma_store"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logger.info(f"Created {UPLOAD_DIR} directory")

if not os.path.exists(CHROMA_PERSIST_DIR):
    os.makedirs(CHROMA_PERSIST_DIR)
    logger.info(f"Created {CHROMA_PERSIST_DIR} directory")

# Configure Gemini API
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("✓ Google Generative AI configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
else:
    logger.warning("⚠ GOOGLE_API_KEY not found in environment variables")

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info(f"✓ HuggingFace embeddings model loaded: {EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"Failed to load embeddings model: {e}")
    logger.info("Attempting to proceed anyway - embeddings will be initialized on first use")
    embedding_model = None

# FastAPI app
app = FastAPI()

# Helper: Convert filename to URL-safe collection name
def _coll_name(filename: str) -> str:
    """
    Convert filename to URL-safe collection directory name.
    
    Args:
        filename (str): The PDF filename to convert.
    
    Returns:
        str: URL-encoded filename suitable for use as directory name.
    
    Example:
        >>> _coll_name('My Document.pdf')
        'My+Document.pdf'
    """
    return urllib.parse.quote_plus(filename)

# Helper: Delete old collection directory for a filename
def _delete_old_collection(filename: str):
    """
    Remove Chroma collection directory for the given filename.
    
    When a user re-uploads a PDF with the same filename, the old vectorstore
    collection is deleted to ensure clean re-indexing with fresh embeddings.
    
    Args:
        filename (str): The PDF filename whose collection should be deleted.
    
    Returns:
        None
    
    Note:
        Errors during deletion are logged as warnings but don't halt execution.
    """
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
            print(f"Deleted old collection for {filename}: {persist_dir}")
        except Exception as e:
            print(f"Warning: Failed to delete old collection: {e}")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Helper — extract text from PDF
def extract_text_from_pdf(path: str) -> str:
    """
    Extract all text content from a PDF file.
    
    Uses PyMuPDF (fitz) to read the PDF file and extract text from each page.
    
    Args:
        path (str): Absolute path to the PDF file.
    
    Returns:
        str: Concatenated text from all pages, with page breaks preserved.
    
    Raises:
        Exception: If PDF cannot be read or is corrupted.
    
    Example:
        >>> text = extract_text_from_pdf('/path/to/document.pdf')
        >>> len(text) > 0
        True
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ---------------------------------------------------------------------
# Helper — LLM JSON summary
# ---------------------------------------------------------------
def genai_generate(prompt: str) -> str:
    """
    Generate text using Google Gemini 2.0 Flash LLM.
    
    Sends a prompt to the Gemini API and returns the generated response.
    Uses the fast 'gemini-2.0-flash' model for cost efficiency.
    
    Args:
        prompt (str): The prompt/question to send to Gemini.
    
    Returns:
        str: The LLM-generated text response. Returns error message string
             if API call fails.
    
    Note:
        Requires GOOGLE_API_KEY environment variable to be configured.
        Errors are caught and returned as strings to prevent API failures
        from crashing the application.
    
    Example:
        >>> response = genai_generate('Summarize this text: ...')
        >>> isinstance(response, str)
        True
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR: {e}"


def summarize_chunks_with_gemini_json(chunks: List[str], max_chunks: int = 8, num_points: int = 6) -> List[str]:
    """
    Generate beginner-friendly bullet point summary from document chunks using Gemini.
    
    This function asks the Gemini LLM to produce a JSON array of `num_points`
    simple bullet points that summarize the provided document chunks. It will
    use up to `max_chunks` chunks from the provided list as context for the LLM.
    
    Args:
        chunks (List[str]): List of document text chunks to summarize.
        max_chunks (int): Maximum number of chunks to include in the LLM context (default: 8).
        num_points (int): Desired number of bullet points to return (default: 6).
    
    Returns:
        List[str]: List of summary bullet strings (length may be <= `num_points` on fallback).
    
    Processing:
        1. Concatenate the top `max_chunks` chunks into a context string
        2. Prompt Gemini to return a JSON array with exactly `num_points` strings
        3. Extract the JSON array from Gemini's response using regex
        4. If JSON extraction fails, fall back to line-based extraction
    
    Note:
        - Returns empty list if `chunks` is empty
        - The returned list will be at most `num_points` items long when falling back
    
    Example:
        >>> bullets = summarize_chunks_with_gemini_json(chunks, max_chunks=8, num_points=9)
        >>> len(bullets) <= 9
        True
    """
    if not chunks:
        return []

    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks[:max_chunks])])

    prompt = f"""
You are a helpful summarizer who explains complex topics in a way that beginners can easily understand.

Your task: Read the context below and create {num_points} explanatory summary points that describe the key ideas of this document.

Guidelines:
- Write in simple, everyday language (avoid jargon and technical terms)
- Each point should be 2-3 sentences that fully explain the concept
- Focus on the most important information someone should know
- Arrange points in a logical order (from basic to more detailed)
- Each point should be detailed and independent - make sense on its own
- Be specific and include examples or details from the document when relevant

Return ONLY a JSON array of strings - nothing else. NO explanations, NO extra text.
Format: ["detailed point 1", "detailed point 2", ..., "detailed point {num_points}"]

Context:
{context}

JSON summary ({num_points} explanatory points in simple language):
"""

    raw = genai_generate(prompt)

    # Extract JSON array
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(1))
            return [str(x).strip() for x in arr if x and str(x).strip()][:num_points]
        except Exception:
            pass

    # fallback if JSON fails
    lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
    return lines[:num_points]


# ---------------------------------------------------------------------
# ROUTE: Upload PDF + Build Vectorstore
# ---------------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a PDF file with RAG vectorization.
    
    Orchestrates the complete pipeline:
    1. Validates PDF file format
    2. Deletes old vectorstore (if re-uploading same filename)
    3. Extracts text from PDF using PyMuPDF
    4. Chunks text using RecursiveCharacterTextSplitter
    5. Filters small chunks (<40 chars) to reduce noise
    6. Creates per-file Chroma vectorstore with embeddings
    7. Persists vectorstore to disk
    
    Args:
        file (UploadFile): Multipart form-data containing PDF file.
    
    Returns:
        dict: {\"filename\": str, \"status\": \"ready\", \"ready\": bool, \"total_chunks\": int}
    
    Raises:
        HTTPException 400: If file is not a PDF or contains no readable text.
        HTTPException 500: If embedding model is not initialized or processing fails.
    
    Notes:
        - Each PDF gets its own isolated Chroma collection
        - Per-file isolation enables multi-document support
        - Chunk size: 1000 chars, overlap: 200 chars
        - Small chunks (<40 chars) are filtered out
    
    Example:
        POST /upload (form-data with file)
        Response: {\"filename\": \"doc.pdf\", \"status\": \"ready\", \"total_chunks\": 42}
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed.")

    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    # Delete old Chroma collection for this filename (if exists)
    # This ensures clean re-indexing when user uploads same filename again
    _delete_old_collection(file.filename)

    # save file
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # extract text
    text = extract_text_from_pdf(save_path)

    if not text.strip():
        raise HTTPException(400, "PDF contains no readable text.")

    # split text using RecursiveCharacterTextSplitter for better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    
    # Filter out very small chunks (< 40 chars) to avoid noise
    chunks = [c.strip() for c in chunks if c and len(c.strip()) > 40]

    if not chunks:
        raise HTTPException(400, "PDF contains no substantial text after chunking.")

    documents = [Document(page_content=c) for c in chunks]

    # Create per-file Chroma collection
    coll = _coll_name(file.filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    os.makedirs(persist_dir, exist_ok=True)
    
    try:
        vectorstore = Chroma.from_documents(
            documents,
            embedding_model,
            persist_directory=persist_dir
        )
        
        # Persist (some Chroma versions auto-persist, but explicit call ensures compatibility)
        try:
            vectorstore.persist()
        except Exception:
            pass

        logger.info(f"✓ Uploaded and indexed PDF: {file.filename} with {len(chunks)} chunks")

        return {
            "filename": file.filename,
            "status": "ready",
            "ready": True,
            "total_chunks": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise HTTPException(500, f"Error processing PDF: {str(e)}")


# ---------------------------------------------------------------------
# ROUTE: Summary (RAG → Gemini → JSON array)
# ---------------------------------------------------------------
@app.post("/summary")
async def summary(filename: str, use_llm: bool = True, num_points: int = None):
    """
    Generate a 6-point summary of an uploaded PDF using RAG + LLM.
    
    Implements Retrieval-Augmented Generation (RAG) pipeline:
    1. Loads the per-file Chroma vectorstore
    2. Performs semantic similarity search to retrieve top 8 chunks
    3. Optionally sends chunks to Gemini for LLM-powered summarization
    4. Falls back to raw chunk extraction if LLM is disabled or API unavailable
    
    Args:
        filename (str): Name of the uploaded PDF file (URL-encoded in query).
        use_llm (bool): If True, use Gemini LLM for summary. If False, return
                       raw extracted chunks (default: True).
        num_points (int, optional): Desired number of summary bullet points.
                                    If omitted, defaults to 6. The value is
                                    clamped between 3 and 12.
    
    Returns:
        dict: {\"summary\": List[str]} - List of 6 bullet points (or raw chunks).
    
    Raises:
        HTTPException 404: If no vectorstore exists for the filename.
        HTTPException 500: If vectorstore loading or search fails.
        HTTPException 400: If no relevant chunks found in document.
    
    Processing:
        - Query: \"summary of {filename}\" for semantic relevance
        - Retrieval: Top k=8 chunks via cosine similarity
        - LLM: Gemini 2.0 Flash for beginner-friendly summarization
        - Fallback: Returns raw chunks if LLM disabled or API key missing
    
    Note:
        - Each PDF must be uploaded first via /upload endpoint
        - Vectorstore must be successfully created before summary generation
        - Requires GOOGLE_API_KEY for LLM-powered summaries
    
    Example:
        POST /summary?filename=document.pdf&use_llm=true
        Response: {\"summary\": [\"Point 1: ...\", \"Point 2: ...\", ...]}
    """
    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")

    # Load vectorstore for this specific file
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    
    if not os.path.exists(persist_dir):
        raise HTTPException(404, f"No vectorstore found for '{filename}'. Please upload the file first.")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")

    # Decide how many summary points to produce (clamped between 3 and 12)
    desired_points = 6 if (num_points is None) else int(num_points)
    desired_points = max(3, min(12, desired_points))

    # Retrieve a larger set of candidates for summarization; use more
    # candidates when more summary points are requested.
    query = f"summary of {filename}"
    k = max(8, desired_points * 2)
    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(500, f"Error searching document: {str(e)}")

    if not results:
        raise HTTPException(400, "No chunks found for summary. The PDF may have insufficient content.")

    chunks = [doc.page_content for doc in results]

    # If user enabled Gemini LLM
    if use_llm:
        if not GOOGLE_API_KEY:
            logger.warning("LLM requested but GOOGLE_API_KEY not configured. Falling back to extractive summary.")
            summary_items = chunks[:desired_points]
        else:
            summary_items = summarize_chunks_with_gemini_json(chunks, max_chunks=8, num_points=desired_points)
    else:
        # simple fallback: return top retrieved chunks as-is
        summary_items = chunks[:desired_points]

    return {"summary": summary_items}


# ---------------------------------------------------------------------
# ROUTE: Chat (Q&A about the document)
# ---------------------------------------------------------------
def answer_question_with_gemini(question: str, chunks: List[str]) -> str:
    """
    Generate an answer to a user question using document chunks and Gemini LLM.
    
    Takes user question and relevant document chunks, constructs a prompt for
    Gemini, and returns the LLM-generated answer.
    
    Args:
        question (str): User's question about the document.
        chunks (List[str]): Retrieved document chunks relevant to the question.
    
    Returns:
        str: LLM-generated answer to the question, or error message if
             generation fails or no chunks are provided.
    
    Processing:
        1. Returns fallback message if chunks list is empty
        2. Formats top 5 chunks as numbered sections
        3. Constructs prompt instructing Gemini to be factual and simple
        4. Sends to Gemini 2.0 Flash model
        5. Returns response or error message
    
    Notes:
        - Uses only provided chunks; won't hallucinate external knowledge
        - Instructs model to say \"not found\" if answer unavailable in chunks
        - Simple language emphasis for beginner accessibility
        - Errors are caught and returned as user-friendly messages
    
    Example:
        >>> chunks = ['PDF text chunk 1', 'PDF text chunk 2']
        >>> answer = answer_question_with_gemini('What is the main topic?', chunks)
        >>> isinstance(answer, str)
        True
    """
    if not chunks:
        return "I don't have enough information to answer that question."
    
    context = "\n\n".join([f"Section {i+1}:\n{c}" for i, c in enumerate(chunks[:5])])
    
    prompt = f"""
You are a helpful assistant answering questions about a document.

Read the document sections provided and answer the user's question clearly and concisely.
- If the answer is in the document, provide it based on the content
- If you cannot find the answer in the provided sections, say "I couldn't find that information in the document"
- Keep your answer simple and easy to understand
- Be specific and factual

Document sections:
{context}

User's question: {question}

Answer:
"""
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Sorry, I encountered an error processing your question: {str(e)}"


@app.post("/chat")
async def chat(filename: str, chat_request: ChatRequest):
    """
    API endpoint for chatbot Q&A about uploaded PDFs.
    
    Implements a document-grounded chatbot that:
    1. Loads the per-file vectorstore for the PDF
    2. Retrieves top 5 semantically similar chunks
    3. Passes chunks to Gemini LLM for answer generation
    4. Returns LLM-generated answer grounded in document content
    
    Args:
        filename (str): Name of the uploaded PDF (URL query parameter).
        chat_request (ChatRequest): Request body with {"question": "..."}
    
    Returns:
        dict: {"answer": str} - LLM-generated answer or error message.
    
    Raises:
        HTTPException 500: If embedding model not initialized.
        HTTPException 400: If question is empty.
        HTTPException 404: If no vectorstore found for filename.
        HTTPException 500: If vectorstore loading or search fails.
    
    Processing:
        1. Validates question is not empty
        2. Loads per-file Chroma vectorstore
        3. Similarity search: retrieves k=5 most relevant chunks
        4. Calls answer_question_with_gemini() with chunks
        5. Logs Q&A interaction
    
    Notes:
        - Each question is independent; no conversation history maintained
        - Answers are grounded in document chunks only
        - Requires GOOGLE_API_KEY for LLM generation
        - Graceful fallback if API key not configured
    
    Example:
        POST /chat?filename=document.pdf
        Body: {"question": "What is the main topic?"}
        Response: {"answer": "The main topic is..."}
    """
    question = chat_request.question.strip()
    
    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")

    # Validate question
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    # Load vectorstore for this specific file
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    
    if not os.path.exists(persist_dir):
        raise HTTPException(404, f"No vectorstore found for '{filename}'. Please upload the file first.")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")

    # Retrieve relevant chunks for the question
    try:
        results = vectorstore.similarity_search(question, k=5)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(500, f"Error searching document: {str(e)}")

    if not results:
        return {
            "answer": "I couldn't find relevant information in the document to answer that question. Try asking something more specific."
        }

    chunks = [doc.page_content for doc in results]

    # Generate answer using Gemini
    if not GOOGLE_API_KEY:
        logger.warning("Chat requested but GOOGLE_API_KEY not configured.")
        return {
            "answer": "The chatbot feature requires API configuration. Please check the server logs."
        }

    answer = answer_question_with_gemini(question, chunks)
    logger.info(f"✓ Answered question about {filename}: {question[:50]}...")

    return {"answer": answer}


# ---------------------------------------------------------------------
# Helper — MCQ generation
# ---------------------------------------------------------------
def generate_mcqs_with_gemini(chunks: List[str], difficulty: str = "medium", num_questions: int = 5) -> List[dict]:
    """
    Generate multiple-choice questions from document chunks using Gemini LLM.
    
    Creates MCQs at specified difficulty level with 4 options and one correct answer.
    
    Args:
        chunks (List[str]): Document chunks to generate questions from.
        difficulty (str): "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of questions to generate (default: 5).
    
    Returns:
        List[dict]: List of MCQ objects with structure:
                    {
                      "question": str,
                      "options": List[str],
                      "correct_answer": str,
                      "explanation": str
                    }
    
    Processing:
        1. Combines top chunks into context
        2. Sends prompt to Gemini with difficulty specification
        3. Parses JSON response to extract questions
        4. Falls back to empty list if generation fails
    
    Note:
        - Requires GOOGLE_API_KEY to be configured
        - Higher difficulty = more conceptual/analytical questions
        - Lower difficulty = more factual/definition questions
    """
    if not chunks:
        return []
    
    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks[:8])])
    
    difficulty_guidance = {
        "easy": "focus on basic definitions, facts, and key terms directly from the text",
        "medium": "mix of factual recall and understanding of concepts and relationships",
        "hard": "require analysis, synthesis, and application of concepts across multiple chunks"
    }
    
    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])
    
    prompt = f"""
You are an expert educator creating multiple-choice questions for student assessment.

Your task: Create exactly {num_questions} multiple-choice questions based on the provided document chunks.

Difficulty level: {difficulty.upper()}
Focus: {guidance}

For each question:
1. Create a clear, concise question
2. Provide 4 distinct options (labeled A, B, C, D)
3. Exactly one correct answer
4. Include a brief explanation for why the answer is correct

Return ONLY a JSON array with this structure (no other text):
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option C",
    "explanation": "Explanation of why this is correct based on the document."
  }},
  ...
]

Document context:
{context}

JSON MCQ array:
"""

    raw = genai_generate(prompt)
    
    # Extract JSON array
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            questions = json.loads(m.group(1))
            if isinstance(questions, list) and len(questions) > 0:
                # Validate structure
                valid_questions = []
                for q in questions:
                    if all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                        valid_questions.append(q)
                return valid_questions[:num_questions]
        except Exception as e:
            logger.error(f"Failed to parse MCQ JSON: {e}")
    
    return []


# ROUTE: MCQ Generation
@app.post("/mcq")
async def generate_mcq(filename: str, difficulty: str = "medium", num_questions: int = 5):
    """
    API endpoint to generate MCQs from an uploaded PDF.
    
    Implements MCQ generation pipeline:
    1. Loads the per-file vectorstore
    2. Retrieves relevant chunks via similarity search
    3. Sends chunks to Gemini for MCQ generation
    4. Returns structured MCQ objects with answers and explanations
    
    Args:
        filename (str): Name of uploaded PDF (URL query parameter).
        difficulty (str): Question difficulty - "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of MCQs to generate, 1-15 (default: 5).
    
    Returns:
        dict: {
          "questions": List[{
            "question": str,
            "options": List[str],
            "correct_answer": str,
            "explanation": str
          }]
        }
    
    Raises:
        HTTPException 500: If embedding model not initialized.
        HTTPException 404: If no vectorstore found for filename.
        HTTPException 400: If no chunks found or MCQ generation fails.
    
    Processing:
        1. Validates difficulty level and num_questions range
        2. Loads per-file Chroma vectorstore
        3. Similarity search: retrieves k=8-10 relevant chunks
        4. Calls generate_mcqs_with_gemini() with chunks
        5. Returns structured MCQ array
    
    Note:
        - Difficulty levels: easy (basic facts), medium (understanding), hard (analysis)
        - Num_questions clamped between 1 and 15
        - Requires GOOGLE_API_KEY for LLM generation
    
    Example:
        POST /mcq?filename=document.pdf&difficulty=medium&num_questions=5
        Response: {"questions": [{"question": "...", "options": [...], "correct_answer": "...", "explanation": "..."}]}
    """
    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")
    
    # Validate parameters
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"
    
    num_questions = max(1, min(15, int(num_questions)))
    
    # Load vectorstore for this specific file
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    
    if not os.path.exists(persist_dir):
        raise HTTPException(404, f"No vectorstore found for '{filename}'. Please upload the file first.")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")
    
    # Retrieve relevant chunks for MCQ generation
    query = f"key topics, concepts, definitions, and important details"
    k = max(8, num_questions)
    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(500, f"Error searching document: {str(e)}")
    
    if not results:
        raise HTTPException(400, "No content found in document to generate MCQs.")
    
    chunks = [doc.page_content for doc in results]
    
    # Generate MCQs using Gemini
    if not GOOGLE_API_KEY:
        logger.warning("MCQ generation requested but GOOGLE_API_KEY not configured.")
        raise HTTPException(400, "MCQ generation requires API configuration.")
    
    questions = generate_mcqs_with_gemini(chunks, difficulty, num_questions)
    
    if not questions:
        raise HTTPException(500, "Failed to generate MCQs. Please try again.")
    
    logger.info(f"✓ Generated {len(questions)} MCQs for {filename} (difficulty: {difficulty})")
    
    return {"questions": questions}


# HELPER: Fill in the Blanks generation with Gemini
def generate_fill_blanks_with_gemini(chunks: List[str], difficulty: str = "medium", num_questions: int = 5) -> List[dict]:
    """
    Generate fill-in-the-blanks questions from document chunks using Gemini LLM.
    
    Creates questions with single blank spaces to test vocabulary and terminology comprehension.
    
    Args:
        chunks (List[str]): Document chunks to generate questions from.
        difficulty (str): "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of questions to generate (default: 5).
    
    Returns:
        List[dict]: List of fill-in-the-blanks objects with structure:
                    {
                      "question": str (contains single "____" placeholder),
                      "correct_answer": str (the word/phrase that fills the blank)
                    }
    
    Processing:
        1. Combines top chunks into context
        2. Sends prompt to Gemini with difficulty specification
        3. Parses JSON response to extract questions
        4. Falls back to empty list if generation fails
    
    Difficulty levels:
        - easy: Common terms, basic vocabulary from the text
        - medium: Key concepts and important terminology
        - hard: Precise/technical terms, nuanced vocabulary
    
    Note:
        - Requires GOOGLE_API_KEY to be configured
        - Each question has exactly one blank represented as "____"
    """
    if not chunks:
        return []
    
    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks[:8])])
    
    difficulty_guidance = {
        "easy": "use common, basic terms that are frequently mentioned in the document",
        "medium": "use important concepts and key vocabulary from the document",
        "hard": "use precise, technical terms and nuanced vocabulary that require deep understanding"
    }
    
    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])
    
    prompt = f"""
You are an expert educator creating fill-in-the-blanks questions for vocabulary and comprehension assessment.

Your task: Create exactly {num_questions} fill-in-the-blanks questions based on the provided document chunks.

Difficulty level: {difficulty.upper()}
Guideline: {guidance}

For each question:
1. Create a complete sentence with exactly ONE blank space
2. The blank should be represented as "____" in the question
3. Provide the correct word/phrase that fills the blank
4. The blank should test comprehension of key terms, not trivial words
5. Keep questions simple and clear

Return ONLY a JSON array with this structure (no other text):
[
  {{
    "question": "The mitochondria is often called the _____ of the cell.",
    "correct_answer": "powerhouse"
  }},
  {{
    "question": "Photosynthesis occurs in the _____ organelle.",
    "correct_answer": "chloroplast"
  }},
  ...
]

Document context:
{context}

JSON fill-in-the-blanks array:
"""

    raw = genai_generate(prompt)
    
    # Extract JSON array
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            questions = json.loads(m.group(1))
            if isinstance(questions, list) and len(questions) > 0:
                # Validate structure
                valid_questions = []
                for q in questions:
                    if all(k in q for k in ["question", "correct_answer"]):
                        # Ensure question has at least one blank
                        if "_____" in q.get("question", ""):
                            valid_questions.append(q)
                return valid_questions[:num_questions]
        except Exception as e:
            logger.error(f"Failed to parse fill-in-the-blanks JSON: {e}")
    
    return []


# ROUTE: Fill in the Blanks Generation
@app.post("/fill-blanks")
async def generate_fill_blanks(filename: str, difficulty: str = "medium", num_questions: int = 5):
    """
    API endpoint to generate fill-in-the-blanks questions from an uploaded PDF.
    
    Implements fill-in-the-blanks generation pipeline:
    1. Loads the per-file vectorstore
    2. Retrieves relevant chunks via similarity search
    3. Sends chunks to Gemini for question generation
    4. Returns structured fill-in-the-blanks objects with answers
    
    Args:
        filename (str): Name of uploaded PDF (URL query parameter).
        difficulty (str): Question difficulty - "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of questions to generate, 1-10 (default: 5).
    
    Returns:
        dict: {
          "questions": List[{
            "question": str (contains single "____" placeholder),
            "correct_answer": str (the word/phrase that fills the blank)
          }]
        }
    
    Raises:
        HTTPException 500: If embedding model not initialized.
        HTTPException 404: If no vectorstore found for filename.
        HTTPException 400: If no chunks found or generation fails.
    
    Processing:
        1. Validates difficulty level and num_questions range
        2. Loads per-file Chroma vectorstore
        3. Similarity search: retrieves k=8-10 relevant chunks
        4. Calls generate_fill_blanks_with_gemini() with chunks
        5. Returns structured fill-in-the-blanks array
    
    Note:
        - Difficulty levels: easy (common terms), medium (key concepts), hard (precise terminology)
        - Num_questions clamped between 1 and 10
        - Requires GOOGLE_API_KEY for LLM generation
        - Each question contains exactly one blank represented as "____"
    
    Example:
        POST /fill-blanks?filename=document.pdf&difficulty=medium&num_questions=5
        Response: {"questions": [{"question": "The _____ is...", "correct_answer": "term"}]}
    """
    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")
    
    # Validate parameters
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"
    
    num_questions = max(1, min(10, int(num_questions)))
    
    # Load vectorstore for this specific file
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    
    if not os.path.exists(persist_dir):
        raise HTTPException(404, f"No vectorstore found for '{filename}'. Please upload the file first.")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")
    
    # Retrieve relevant chunks for fill-in-the-blanks generation
    query = f"key terms, vocabulary, important phrases, and definitions"
    k = max(8, num_questions)
    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(500, f"Error searching document: {str(e)}")
    
    if not results:
        raise HTTPException(400, "No content found in document to generate fill-in-the-blanks questions.")
    
    chunks = [doc.page_content for doc in results]
    
    # Generate fill-in-the-blanks using Gemini
    if not GOOGLE_API_KEY:
        logger.warning("Fill-in-the-blanks generation requested but GOOGLE_API_KEY not configured.")
        raise HTTPException(400, "Fill-in-the-blanks generation requires API configuration.")
    
    questions = generate_fill_blanks_with_gemini(chunks, difficulty, num_questions)
    
    if not questions:
        raise HTTPException(500, "Failed to generate fill-in-the-blanks questions. Please try again.")
    
    logger.info(f"✓ Generated {len(questions)} fill-in-the-blanks questions for {filename} (difficulty: {difficulty})")
    
    return {"questions": questions}


# HELPER: Short Answer and "Give Reason" question generation with Gemini
def generate_short_answer_with_gemini(chunks: List[str], difficulty: str = "medium", num_questions: int = 5) -> List[dict]:
    """
    Generate short answer and "give reason" questions from document chunks using Gemini LLM.
    
    Creates open-ended questions that require explanatory answers, testing comprehension and analysis.
    
    Args:
        chunks (List[str]): Document chunks to generate questions from.
        difficulty (str): "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of questions to generate (default: 5).
    
    Returns:
        List[dict]: List of short answer question objects with structure:
                    {
                      "question": str,
                      "suggested_answer": str (2-3 sentence reference answer),
                      "hint": str (key points to include in answer)
                    }
    
    Processing:
        1. Combines top chunks into context
        2. Sends prompt to Gemini with difficulty specification
        3. Parses JSON response to extract questions
        4. Falls back to empty list if generation fails
    
    Difficulty levels:
        - easy: Direct recall and simple explanations from the text
        - medium: Understanding concepts and explaining relationships
        - hard: Analysis, synthesis, and application of concepts with reasoning
    
    Note:
        - Requires GOOGLE_API_KEY to be configured
        - Each question includes a suggested answer and key points hint
        - Grading is subjective (not automatic); teacher/student can compare with suggested answer
    """
    if not chunks:
        return []
    
    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks[:8])])
    
    difficulty_guidance = {
        "easy": "ask students to recall facts and provide simple explanations directly from the document",
        "medium": "ask students to explain concepts, relationships, and why something happens based on the document",
        "hard": "ask students to analyze, compare, synthesize information and provide reasoned explanations for complex concepts"
    }
    
    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])
    
    prompt = f"""
You are an expert educator creating short answer and "give reason" questions for comprehensive assessment.

Your task: Create exactly {num_questions} short answer questions based on the provided document chunks.

Difficulty level: {difficulty.upper()}
Focus: {guidance}

For each question:
1. Create a clear, open-ended question that requires a written answer (not a single word)
2. Provide a suggested answer (2-3 sentences) that a good student might write
3. Include 2-3 key points that should be covered in the answer

Question types:
- "What is..." / "Explain..." / "How does..."
- "Why..." / "Give reason for..." (analytical questions)
- "Compare..." / "Discuss..." (synthesis questions)

Return ONLY a JSON array with this structure (no other text):
[
  {{
    "question": "Why is photosynthesis important for life on Earth?",
    "suggested_answer": "Photosynthesis is crucial because it converts sunlight into chemical energy stored in glucose, which feeds most organisms. It also produces oxygen as a byproduct, which is essential for aerobic respiration in most living things.",
    "hint": "Energy conversion, glucose production, oxygen production"
  }},
  {{
    "question": "Explain how the cell membrane controls what enters and exits the cell.",
    "suggested_answer": "The cell membrane uses selective permeability to control substances. It has phospholipid bilayers with embedded proteins that act as channels and carriers. Some molecules pass freely while others require energy (active transport) or carrier proteins.",
    "hint": "Selective permeability, phospholipid bilayer, proteins, active transport"
  }},
  ...
]

Document context:
{context}

JSON short answer questions array:
"""

    raw = genai_generate(prompt)
    
    # Extract JSON array
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            questions = json.loads(m.group(1))
            if isinstance(questions, list) and len(questions) > 0:
                # Validate structure
                valid_questions = []
                for q in questions:
                    if all(k in q for k in ["question", "suggested_answer", "hint"]):
                        valid_questions.append(q)
                return valid_questions[:num_questions]
        except Exception as e:
            logger.error(f"Failed to parse short answer JSON: {e}")
    
    return []


# ROUTE: Short Answer Question Generation
@app.post("/short-answer")
async def generate_short_answer(filename: str, difficulty: str = "medium", num_questions: int = 5):
    """
    API endpoint to generate short answer and "give reason" questions from an uploaded PDF.
    
    Implements short answer question generation pipeline:
    1. Loads the per-file vectorstore
    2. Retrieves relevant chunks via similarity search
    3. Sends chunks to Gemini for question generation
    4. Returns structured question objects with suggested answers and key points
    
    Args:
        filename (str): Name of uploaded PDF (URL query parameter).
        difficulty (str): Question difficulty - "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of questions to generate, 1-10 (default: 5).
    
    Returns:
        dict: {
          "questions": List[{
            "question": str,
            "suggested_answer": str (reference answer),
            "hint": str (key points to include)
          }]
        }
    
    Raises:
        HTTPException 500: If embedding model not initialized.
        HTTPException 404: If no vectorstore found for filename.
        HTTPException 400: If no chunks found or generation fails.
    
    Processing:
        1. Validates difficulty level and num_questions range
        2. Loads per-file Chroma vectorstore
        3. Similarity search: retrieves k=8-10 relevant chunks
        4. Calls generate_short_answer_with_gemini() with chunks
        5. Returns structured short answer array
    
    Note:
        - Difficulty levels: easy (recall), medium (understanding), hard (analysis/reasoning)
        - Num_questions clamped between 1 and 10
        - Requires GOOGLE_API_KEY for LLM generation
        - Grading is subjective; suggested_answer is a reference for comparison
    
    Example:
        POST /short-answer?filename=document.pdf&difficulty=medium&num_questions=5
        Response: {"questions": [{"question": "...", "suggested_answer": "...", "hint": "..."}]}
    """
    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")
    
    # Validate parameters
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"
    
    num_questions = max(1, min(10, int(num_questions)))
    
    # Load vectorstore for this specific file
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    
    if not os.path.exists(persist_dir):
        raise HTTPException(404, f"No vectorstore found for '{filename}'. Please upload the file first.")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")
    
    # Retrieve relevant chunks for short answer generation
    query = f"main concepts, explanations, relationships, and detailed information"
    k = max(8, num_questions)
    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(500, f"Error searching document: {str(e)}")
    
    if not results:
        raise HTTPException(400, "No content found in document to generate short answer questions.")
    
    chunks = [doc.page_content for doc in results]
    
    # Generate short answer questions using Gemini
    if not GOOGLE_API_KEY:
        logger.warning("Short answer generation requested but GOOGLE_API_KEY not configured.")
        raise HTTPException(400, "Short answer generation requires API configuration.")
    
    questions = generate_short_answer_with_gemini(chunks, difficulty, num_questions)
    
    if not questions:
        raise HTTPException(500, "Failed to generate short answer questions. Please try again.")
    
    logger.info(f"✓ Generated {len(questions)} short answer questions for {filename} (difficulty: {difficulty})")
    
    return {"questions": questions}


@app.post("/generate-all")
async def generate_all_content(filename: str):
    """
    Generate all learning content (summary, MCQ, fill-in-the-blanks, short answer) for a file.
    
    This endpoint is called after successful upload to pre-generate all content
    and cache it on the client side for instant retrieval.
    
    Args:
        filename (str): The PDF filename (already uploaded).
    
    Returns:
        dict: {
            "filename": str,
            "summary": List[str],
            "mcq": List[dict],
            "fill_blanks": List[dict],
            "short_answer": List[dict]
        }
    
    Raises:
        HTTPException 400: If file not found or has no chunks.
        HTTPException 500: If generation fails.
    """
    try:
        # Ensure embedding model is available
        if embedding_model is None:
            raise HTTPException(500, "Embedding model not initialized. Check server logs.")

        # Load vectorstore for this specific file
        coll = _coll_name(filename)
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
        
        if not os.path.exists(persist_dir):
            raise HTTPException(400, f"File {filename} not found or has no indexed content.")
        
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model
            )
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")
        
        # Retrieve chunks for generation
        query = f"content from {filename}"
        results = vectorstore.similarity_search(query, k=12)
        
        if not results:
            raise HTTPException(400, "No content found in document.")
        
        chunks = [doc.page_content for doc in results]
        
        # Generate all content
        all_content = {
            "filename": filename,
            "summary": [],
            "mcq": [],
            "fill_blanks": [],
            "short_answer": []
        }
        
        # 1. Generate summary (default: 6 points)
        if GOOGLE_API_KEY:
            try:
                summary_points = summarize_chunks_with_gemini_json(chunks, max_chunks=8, num_points=6)
                all_content["summary"] = summary_points
                logger.info(f"✓ Generated summary for {filename}")
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
        
        # 2. Generate MCQ (medium difficulty, 5 questions)
        if GOOGLE_API_KEY:
            try:
                mcq_questions = generate_mcqs_with_gemini(chunks, difficulty="medium", num_questions=5)
                all_content["mcq"] = mcq_questions
                logger.info(f"✓ Generated MCQ for {filename}")
            except Exception as e:
                logger.warning(f"MCQ generation failed: {e}")
        
        # 3. Generate Fill-in-the-Blanks (medium difficulty, 5 questions)
        if GOOGLE_API_KEY:
            try:
                fib_questions = generate_fill_blanks_with_gemini(chunks, difficulty="medium", num_questions=5)
                all_content["fill_blanks"] = fib_questions
                logger.info(f"✓ Generated fill-in-the-blanks for {filename}")
            except Exception as e:
                logger.warning(f"Fill-in-the-blanks generation failed: {e}")
        
        # 4. Generate Short Answer (medium difficulty, 5 questions)
        if GOOGLE_API_KEY:
            try:
                short_answer_questions = generate_short_answer_with_gemini(chunks, difficulty="medium", num_questions=5)
                all_content["short_answer"] = short_answer_questions
                logger.info(f"✓ Generated short answer questions for {filename}")
            except Exception as e:
                logger.warning(f"Short answer generation failed: {e}")
        
        logger.info(f"✓ Generated all content for {filename}")
        return all_content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating all content: {e}")
        raise HTTPException(500, f"Error generating content: {str(e)}")

