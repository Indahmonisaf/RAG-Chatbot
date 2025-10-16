# app/vectorstore/chroma_store.py
import os, re
from app.core.config import settings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

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

def _client():
    os.makedirs(settings.PERSIST_DIR, exist_ok=True)
    # ✅ PersistentClient + telemetry off (paling “nempel” ke Chroma)
    return chromadb.PersistentClient(
        path=settings.PERSIST_DIR,
        settings=Settings(
            anonymized_telemetry=False,   # <— ini yang mematikan telemetry
        ),
    )

def get_vectorstore(create_if_missing: bool = True):
    emb = _embedding_fn()
    client = _client()
    # ✅ Saat pakai client=..., JANGAN lagi pakai persist_directory di Chroma()
    return Chroma(
        client=client,
        collection_name=_collection_name(),
        embedding_function=emb,
    )

def new_vectorstore():
    emb = _embedding_fn()
    client = _client()
    return Chroma(
        client=client,
        collection_name=_collection_name(),
        embedding_function=emb,
    )
