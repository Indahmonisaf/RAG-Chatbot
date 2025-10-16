# app/ingest/indexer.py
from typing import Dict, Any
from app.core.config import settings
from app.core.logger import logger
from app.ingest.loader import load_documents_from_dir, load_documents_from_payload
from app.ingest.cleaner import clean_documents
from app.vectorstore.chroma_store import new_vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def _split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # simulasi line range jika tersedia
    for c in chunks:
        md = c.metadata
        if "line_start" in md and md["line_start"] is not None and md.get("line_end") is None:
            ln = max(1, len(c.page_content.splitlines()))
            md["line_end"] = md["line_start"] + ln - 1
    return chunks

def _collect_stats(chunks: list[Document]):
    files = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        files[src] = files.get(src, 0) + 1
    return [{"filename": k, "chunks": v} for k, v in files.items()]

def build_index_from_dir() -> Dict[str, Any]:
    docs = load_documents_from_dir()
    if not docs:
        raise ValueError("No documents found in ./data. Add PDF/TXT/MD/PNG first or use /ingest-json.")
    docs = clean_documents(docs)
    chunks = _split_docs(docs)
    if not chunks:
        raise ValueError("No chunks could be created. Files may be empty or unreadable (e.g., scanned PDFs without OCR).")
    vs = new_vectorstore()           # Chroma 0.4+ auto-persist
    vs.add_documents(chunks)
    logger.info(f"Indexed chunks: {len(chunks)}")
    return {"files": _collect_stats(chunks), "total_chunks": len(chunks)}

def build_index_from_payload(payload) -> Dict[str, Any]:
    docs, saved = load_documents_from_payload(payload.files)
    # override chunk params jika dikirim
    if payload.options and payload.options.chunk_size:
        settings.CHUNK_SIZE = payload.options.chunk_size  # type: ignore
    if payload.options and payload.options.chunk_overlap:
        settings.CHUNK_OVERLAP = payload.options.chunk_overlap  # type: ignore
    docs = clean_documents(docs)
    chunks = _split_docs(docs)
    if not chunks:
        raise ValueError("Uploaded files produced 0 chunks. Ensure files have extractable text or use OCR.")
    vs = new_vectorstore()
    vs.add_documents(chunks)
    logger.info(f"Indexed chunks: {len(chunks)}")
    return {"files": saved, "total_chunks": len(chunks)}

# CLI
if __name__ == "__main__":
    out = build_index_from_dir()
    print(f"Indexed: {out['total_chunks']} chunks")
