import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL + pgvector config
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Claude API config
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-opus-4-1"
EMBEDDING_DIMENSION = 1536

# RAG config
RETRIEVAL_TOP_K = 3
SIMILARITY_THRESHOLD = 0.7
