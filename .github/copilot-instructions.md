# Aurelo React-FastAPI Codebase Guidelines

## Project Overview
**Aurelo** is a full-stack document intelligence platform for PDF summarization and analysis. It combines a FastAPI backend (Python) with a React frontend for RAG-based document processing.

### Architecture: Backend → API → Frontend
- **Backend** (`backend/app/`): FastAPI server handling PDF uploads, vectorization, and LLM-powered summarization
- **Frontend** (`frontend/src/`): React+Vite SPA with Bootstrap components for upload UI and summary display
- **Cross-component**: CORS-enabled REST API on `http://localhost:8000`

---

## Technology Stack

### Backend (Python)
- **FastAPI** - REST API framework with CORS middleware
- **LangChain** - Document chunking (`RecursiveCharacterTextSplitter`), embedding abstraction
- **Chroma DB** - Vector embeddings storage (persisted to `./chroma_store/`)
- **HuggingFace Embeddings** - Sentence transformer (`all-MiniLM-L6-v2` default)
- **Google Generative AI** - Gemini API for LLM summarization (env var: `GOOGLE_API_KEY`)
- **PyMuPDF (fitz)** - PDF text extraction

### Frontend (JavaScript)
- **React 19** - UI framework with hooks (useState)
- **Vite 7** - Build tool with HMR
- **React-Bootstrap 2.10** - Pre-styled component library
- **ESLint 9** - Code quality

---

## Critical Data Flow Patterns

### PDF Upload → Vectorization (Backend)
**File**: `backend/app/main.py:upload_file()` and `backend/app/utils.py`

1. PDF received via `POST /upload` (multipart form-data)
2. Text extraction: `PyMuPDF.open(path)` → `page.get_text()`
3. Text chunking: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
4. Filter chunks < 40 chars to avoid noise
5. Generate embeddings using HuggingFace model
6. Store in **per-file Chroma collection** (filename → URL-encoded collection name in `./chroma_store/`)
7. Return `{"status": "ready", "filename": "doc.pdf", "total_chunks": N}`

### Summary Generation (RAG Loop)
**File**: `backend/app/main.py:summary()` and `utils.py:safe_get_top_k()`

1. Frontend POSTs to `POST /summary?filename=doc.pdf` with `{use_llm: bool}`
2. Load Chroma vectorstore for that filename's collection
3. Similarity search: `vectorstore.similarity_search(f"summary of {filename}", k=8)`
4. Extract `page_content` from returned Document objects
5. If `use_llm=true`: Send chunks to **Gemini 1.5 Flash** with strict JSON prompt
6. Parse response regex: `(\[.*\])` to extract JSON array of 6 bullet points
7. Return `{"summary": ["point1", "point2", ...]}`

**Key**: Each uploaded file gets its own Chroma collection to support multiple concurrent PDFs.

---

## Project-Specific Conventions

### Configuration
- **Backend config**: Environment variables loaded in `backend/app/config.py`
  - `GOOGLE_API_KEY` - Gemini API auth (required for LLM features)
  - `EMBEDDING_MODEL` - HuggingFace model name (default: `all-MiniLM-L6-v2`)
  - `CHUNK_SIZE`, `CHUNK_OVERLAP` - Text chunking parameters
  - `CHROMA_BASE` - Vectorstore root directory (default: `./chroma_store`)

- **Frontend API endpoint**: `http://localhost:8000` (hardcoded in components; centralized in `src/api.js`)

### Error Handling Patterns
- **Backend**: HTTPException with status codes (400 for user error, 500 for system)
- **Frontend**: Catch fetch errors → display in red Alert component; show spinner during async operations
- **LLM fallback**: If Gemini JSON parsing fails, return raw chunk text (lines 6)

### UI Component Structure
- **`Upload.jsx`**: Single file input + upload button; manages upload state + status messages
- **`SummaryCards.jsx`**: Stateless horizontal scrollable card layout (flexbox, 320px min-width)
- **`App.jsx`**: Main orchestrator; toggles between Upload and Summary views; manages `filename`, `useLLM`, `summaryItems` state
- **`api.js`**: Centralized fetch wrapper (currently unused; components call fetch directly)

---

## Development Workflows

### Running Backend
```bash
cd backend
# Install: pip install fastapi uvicorn langchain-community python-multipart python-dotenv fitz google-generativeai
# Run: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Frontend
```bash
cd frontend
npm install
npm run dev  # Starts Vite dev server on http://localhost:5173
```

### Build & Deploy
- **Frontend build**: `npm run build` → outputs to `dist/`
- **Frontend preview**: `npm run preview` (serve dist locally before deploy)
- **Linting**: `npm run lint` (ESLint)

---

## Common Workflows & Pitfalls

### Adding a New PDF Processing Feature
1. **Backend**: Add new route in `main.py` or `utils.py`
2. **Chroma queries**: Use `make_retriever_for_file(filename)` to access per-file vectorstore
3. **Chunking**: Use utility `chunk_text()` for consistency (RecursiveCharacterTextSplitter config)
4. **Frontend**: Call new endpoint in component, handle JSON response, update state

### Debugging Vectorstore Issues
- Chroma collections are **persisted per filename** (URL-encoded) in `./chroma_store/`
- If a query returns no results, check: (1) file was uploaded, (2) text extraction succeeded (PDF readable), (3) chunks > 40 chars
- `safe_get_top_k()` includes fallback for retriever API version mismatches

### API Response Inconsistencies
- **Upload endpoint** returns `status` field (check for "ready" string)
- **Summary endpoint** expects `use_llm` boolean in request body
- Frontend expects summary as **array of strings**; if backend returns plain text, frontend has fallback split

### Gemini Prompt Engineering
- Located in `main.py:summarize_chunks_with_gemini_json()`
- **Must** return JSON array only (no extra text) for regex parsing to work
- Falls back to raw chunks if JSON extraction fails
- Uses `genai.GenerativeModel("gemini-1.5-flash")` (flash for cost efficiency)

---

## File Organization Reference

| Path | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI app, upload/summary routes |
| `backend/app/utils.py` | Shared utilities: PDF extraction, chunking, vectorstore management |
| `backend/app/config.py` | Environment variable loading |
| `frontend/src/App.jsx` | Main React component (orchestrator) |
| `frontend/src/api.js` | Fetch wrappers (centralized API calls) |
| `frontend/src/components/` | Upload, SummaryCards, (future: McqPractice) |
| `./chroma_store/` | Persisted Chroma vectorstores (per-file collections) |
| `./uploads/` | Temporary PDF storage |

---

## Tips for Agent Productivity
- **Understand per-file isolation**: Each PDF gets a separate Chroma collection; vectorstore state is not global
- **Watch for API contract changes**: Frontend and backend must agree on JSON shape (e.g., `summary` array vs object)
- **Test locally first**: Run backend on 8000, frontend on 5173; check browser console + backend logs for errors
- **Leverage LangChain abstractions**: Use `as_retriever()`, `Document` objects to maintain consistency with existing code
- **Mind the prompt**: Gemini responses must be JSON-only for the regex parser to work reliably
