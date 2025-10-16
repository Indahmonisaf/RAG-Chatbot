# app/core/config.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class _Settings(BaseModel):
    # Provider: "openai" (kamu pakai OpenAI) atau "gemini"
    PROVIDER: str = os.getenv("PROVIDER", "openai").lower()

    # Embeddings & LLM
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./index")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

# >>> WAJIB: inisialisasi settings <<<
settings = _Settings()
