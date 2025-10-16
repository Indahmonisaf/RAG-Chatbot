# app/vectorstore/chroma_store.py
import os, re
from app.core.config import settings
from langchain_chroma import Chroma  # paket baru

def _embedding_fn():
    if settings.PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    elif settings.PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported provider: {settings.PROVIDER}")

def _collection_name() -> str:
    m = re.sub(r"[^a-zA-Z0-9_]+", "_", settings.EMBEDDING_MODEL)
    return f"docs_{m}"

def get_vectorstore(create_if_missing: bool = True):
    os.makedirs(settings.PERSIST_DIR, exist_ok=True)
    emb = _embedding_fn()
    # Chroma auto-create koleksi saat add/query; create_if_missing disimpan untuk kompatibilitas
    return Chroma(
        persist_directory=settings.PERSIST_DIR,
        collection_name=_collection_name(),
        embedding_function=emb,
    )

def new_vectorstore():
    os.makedirs(settings.PERSIST_DIR, exist_ok=True)
    emb = _embedding_fn()
    # Instance baru; koleksi sama (direset otomatis saat add_documents pertama kalinya)
    return Chroma(
        persist_directory=settings.PERSIST_DIR,
        collection_name=_collection_name(),
        embedding_function=emb,
    )
