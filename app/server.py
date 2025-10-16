import os, time, faiss, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from .utils import load_document, chunk_text, embedder, save_metadata
from .utils import load_metadata, build_prompt


# Providers
from openai import OpenAI
import ollama

load_dotenv()

STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
INDEX_PATH = STORAGE_DIR / "index.faiss"
META_PATH  = STORAGE_DIR / "metadata.json"

PROVIDER = os.getenv("PROVIDER", "openai").lower()
TOP_K = int(os.getenv("TOP_K", "5"))

# Embeddings model (local SentenceTransformer for retrieval)
EMB = embedder()

# Load FAISS + metadata on startup
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise RuntimeError("Index missing. Run: python -m app.ingest")

INDEX = faiss.read_index(str(INDEX_PATH))
METADATA = load_metadata(META_PATH)

class AskRequest(BaseModel):
    question: str

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

def retrieve(query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict]]:
    q_emb = EMB.encode([query], normalize_embeddings=True).astype("float32")
    D, I = INDEX.search(q_emb, top_k)
    results = []

    CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"
    if not CHUNKS_PATH.exists():
        raise RuntimeError("chunks.jsonl missing. Re-run ingest with latest code.")
    # lazy read once
    if not hasattr(retrieve, "_chunks"):
        retrieve._chunks = [json.loads(line)["text"] for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = retrieve._chunks

    for idx in I[0]:
        if idx < 0 or idx >= len(METADATA): continue
        results.append((chunks[idx], METADATA[idx]))
    return results

def generate_with_openai(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a concise RAG assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def generate_with_ollama(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3")
    resp = ollama.chat(model=model, messages=[{"role":"system","content":"You are a concise RAG assistant."},
                                              {"role":"user","content":prompt}])
    return resp["message"]["content"].strip()

@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    contexts = retrieve(question, TOP_K)
    prompt = build_prompt(question, contexts)

    started = time.time()
    if PROVIDER == "ollama":
        answer = generate_with_ollama(prompt)
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        retriever_name = "FAISS"
    else:
        answer = generate_with_openai(prompt)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        retriever_name = "FAISS"

    # Build sources
    sources = []
    for _, md in contexts:
        src = md.get("source", "unknown")
        loc = md.get("location", "")
        if loc:
            sources.append(f"{src}: {loc}")
        else:
            sources.append(src)

    return {
        "question": question,
        "context_sources": sources,
        "answer": answer,
        "metadata": {
            "model": model_name,
            "retrieval_engine": retriever_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latency_ms": int((time.time() - started) * 1000),
        },
    }

@app.post("/reload")
def reload_index():
    global INDEX, METADATA
    INDEX = faiss.read_index(str(INDEX_PATH))
    METADATA = load_metadata(META_PATH)
    # drop cached chunks so they re-load
    if hasattr(retrieve, "_chunks"):
        delattr(retrieve, "_chunks")
    return {"status": "ok", "message": "Index reloaded"}
