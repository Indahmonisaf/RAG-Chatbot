import os, json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss

from .utils import load_document, chunk_text, embedder, save_metadata

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = STORAGE_DIR / "index.faiss"
META_PATH  = STORAGE_DIR / "metadata.json"
CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"

def main():
    print(f"[INGEST] Reading from {DATA_DIR} ...")
    files = []
    for ext in ("*.txt", "*.md", "*.markdown", "*.pdf", "*.png", "*.jpg", "*.jpeg"):
        files.extend(DATA_DIR.glob(ext))
    if not files:
        print("No documents found in /data. Add some and re-run.")
        return

    model = embedder()
    texts: List[str] = []
    metas: List[Dict] = []

    for f in files:
        raw = load_document(f)
        if not raw.strip():
            print(f"Skipping (empty or unsupported): {f.name}")
            continue
        chunks = chunk_text(raw)
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": f.name, "location": f"chunk {idx+1}"})

    if not texts:
        print("No chunks produced.")
        return

    # --- write chunks.jsonl so server can read chunk texts ---
    with CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for t in texts:
            out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {CHUNKS_PATH}")

    print(f"Embedding {len(texts)} chunks...")
    embs = model.encode(texts, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embs, dtype="float32"))

    faiss.write_index(index, str(INDEX_PATH))
    save_metadata(META_PATH, metas)
    print(f"Saved index to {INDEX_PATH} and metadata to {META_PATH}")

if __name__ == "__main__":
    main()
