from dotenv import load_dotenv
import os

load_dotenv()
CHROMA_BASE = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
