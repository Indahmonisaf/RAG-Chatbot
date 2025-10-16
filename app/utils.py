import os, json, re
from typing import List, Tuple, Dict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from markdown import markdown
from bs4 import BeautifulSoup
from PIL import Image

# OCR optional
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
try:
    import pytesseract  # requires system Tesseract
except Exception:
    pytesseract = None
    ENABLE_OCR = False

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text): break
        start = end - overlap
    return [c for c in chunks if c.strip()]

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_md(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    html = markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n")

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
    return "\n".join(out)

def load_png(path: Path) -> str:
    if not ENABLE_OCR or pytesseract is None:
        return ""
    img = Image.open(str(path))
    text = pytesseract.image_to_string(img, lang="eng")
    return text or ""

def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt"]:
        return load_txt(path)
    if suffix in [".md", ".markdown"]:
        return load_md(path)
    if suffix in [".pdf"]:
        return load_pdf(path)
    if suffix in [".png", ".jpg", ".jpeg"]:
        return load_png(path)
    return ""

def embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    # local, no key required
    return SentenceTransformer(model_name)

def save_metadata(meta_path: Path, meta: List[Dict]):
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_metadata(meta_path: Path) -> List[Dict]:
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return []

def build_prompt(question: str, contexts: List[Tuple[str, Dict]]) -> str:
    # contexts: list of (chunk_text, metadata_dict)
    joined = ""
    for i, (ct, md) in enumerate(contexts, 1):
        src = md.get("source", "unknown")
        loc = md.get("location", "")
        joined += f"[{i}] Source: {src} {loc}\n{ct}\n\n"
    return (
        "You are a helpful RAG assistant. Answer concisely using ONLY the context.\n"
        "If the answer isn't present, say you don't know.\n\n"
        f"Context:\n{joined}\n"
        f"User question: {question}\n"
        "Answer:"
    )
