# app/rag/retriever.py
from typing import List, Tuple
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# cache model kecil untuk rerank (cepat, lokal)
_CE = None
def _ce():
    global _CE
    if _CE is None:
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CE

def _bm25_from_docs(all_docs: List[Document]):
    tokenized = [d.page_content.split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def retrieve_with_scores(vs, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """
    Hybrid: vector (MMR) + BM25 → union → CrossEncoder rerank → top_k
    """
    # 1) vector MMR
    vec_ret = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": min(top_k, 6), "fetch_k": 25, "lambda_mult": 0.6},
    )
    vec_docs = vec_ret.invoke(query)

    # 2) BM25 keyword
    raw = vs._collection.get(include=["documents", "metadatas"])
    all_docs = [Document(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]
    bm25, tokenized = _bm25_from_docs(all_docs)
    scores = bm25.get_scores(query.split())
    bm_idx = np.argsort(scores)[::-1][:max(top_k, 5)]
    bm_docs = [all_docs[i] for i in bm_idx]

    # 3) union candidates
    seen = set()
    cands: List[Document] = []
    for d in vec_docs + bm_docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:60])
        if key not in seen:
            seen.add(key)
            cands.append(d)

    # 4) rerank dengan CrossEncoder
    ce = _ce()
    pairs = [(query, d.page_content) for d in cands]
    if pairs:
        ce_scores = ce.predict(pairs).tolist()
    else:
        ce_scores = []

    ranked = sorted(zip(cands, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]
    # normalisasi skor 0..1 untuk metadata
    if ranked:
        mx, mn = max(s for _, s in ranked), min(s for _, s in ranked)
        rng = max(1e-6, mx - mn)
        ranked = [(d, (s - mn) / rng) for d, s in ranked]
    return ranked
