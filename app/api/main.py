from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.core.schema import QuestionRequest, AskResponse, IngestJsonRequest, IngestJsonResponse, SourcesResponse, HealthResponse
from app.core.config import settings
from app.ingest.indexer import build_index_from_dir, build_index_from_payload
from app.vectorstore.chroma_store import get_vectorstore
from app.rag.retriever import retrieve_with_scores
from app.rag.generator import generate_answer
from app.rag.prompt import build_prompt
from datetime import datetime
import time

from time import perf_counter
from typing import Dict
_VS = None
_METRICS: Dict[str, float | int] = {"q_count": 0, "last_latency_ms": 0, "avg_latency_ms": 0}

def _vs():
    global _VS
    if _VS is None:
        _VS = get_vectorstore(create_if_missing=False)
    return _VS

app = FastAPI(title="RAG Chatbot API", version="1.0.0")
@app.get("/", include_in_schema=False)
def root():
    # Langsung arahkan ke Swagger UI
    return RedirectResponse(url="/docs")

# CORS (boleh dibatasi sesuai kebutuhan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    vs_exists = get_vectorstore(create_if_missing=False) is not None
    return HealthResponse(status="ok", vector_index_ready=vs_exists, provider=settings.PROVIDER, model=settings.LLM_MODEL)

@app.get("/sources", response_model=SourcesResponse)
def sources():
    vs = get_vectorstore(create_if_missing=False)
    if vs is None:
        return SourcesResponse(documents=[], vector_store="Chroma", total_chunks=0)
    # Chroma tidak expose daftar dokumen secara langsung;
    # namun kita bisa baca metadatas via collection.get()
    coll = vs._collection
    ids = coll.get(include=["metadatas"]).get("metadatas", [])
    files = {}
    for mlist in ids:
        for md in (mlist if isinstance(mlist, list) else [mlist]):
            if not md: 
                continue
            source = md.get("source", "unknown")
            files[source] = files.get(source, 0) + 1
    docs = [{"filename": f, "chunks": c} for f, c in files.items()]
    total = sum(files.values())
    return SourcesResponse(documents=docs, vector_store="Chroma", total_chunks=total)

@app.post("/ingest-json", response_model=IngestJsonResponse)
def ingest_json(payload: IngestJsonRequest):
    try:
        stats = build_index_from_payload(payload)
        return IngestJsonResponse(status="ok", indexed_files=stats["files"], vector_store="Chroma", total_chunks=stats["total_chunks"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask(req: QuestionRequest):
    vs = _vs()
    if vs is None:
        raise HTTPException(status_code=400, detail="no_index: Upload & build index dulu (gunakan /ingest-json atau CLI).")

    t0 = time.time()
    # Ambil dokumen relevan + skor (0..1)
    docs_scores = retrieve_with_scores(vs, req.question, top_k=req.top_k)
    contexts = [ds[0] for ds in docs_scores]
    scores = [float(ds[1]) for ds in docs_scores] if docs_scores else []
    avg_sim = sum(scores)/len(scores) if scores else 0.0

    # Susun prompt
    prompt = build_prompt(contexts, req.question)

    # Generate jawaban dari LLM
    answer_text = generate_answer(prompt, temperature=req.temperature, max_tokens=req.max_tokens)

    # Siapkan context_sources yang rapi
    sources = []
    for d in contexts:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page")
        line_start = md.get("line_start")
        line_end = md.get("line_end")
        if page is not None:
            label = f"{src}: page {page}"
        elif line_start is not None and line_end is not None:
            label = f"{src}: lines {line_start}–{line_end}"
        else:
            label = f"{src}"
        if label not in sources:
            sources.append(label)

    latency_ms = int((time.time() - t0) * 1000)
    return AskResponse(
        question=req.question,
        context_sources=sources,
        answer=answer_text,
        metadata={
            "model": settings.LLM_MODEL,
            "retrieval_engine": "Chroma",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "latency_ms": latency_ms,
            "top_k": req.top_k,
            "avg_similarity": round(avg_sim, 3),
            "provider": settings.PROVIDER,
        }
    )
