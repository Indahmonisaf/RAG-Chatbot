import os, io, base64
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.core.logger import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from markdown import markdown
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

def _ensure_data_dir():
    os.makedirs(settings.DATA_DIR, exist_ok=True)

def _save_base64(filename: str, b64: str) -> str:
    _ensure_data_dir()
    raw = base64.b64decode(b64)
    path = os.path.join(settings.DATA_DIR, filename)
    with open(path, "wb") as f:
        f.write(raw)
    return path

def _save_text(filename: str, text: str) -> str:
    _ensure_data_dir()
    path = os.path.join(settings.DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def _md_to_text(md_str: str) -> str:
    # Render markdown → HTML → plain text untuk buang markup
    html = markdown(md_str)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n")

def load_documents_from_dir() -> List[Document]:
    """Load semua file dari data dir (pdf/txt/md/png)"""
    _ensure_data_dir()
    docs: List[Document] = []
    for name in os.listdir(settings.DATA_DIR):
        path = os.path.join(settings.DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        if name.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif name.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif name.lower().endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                txt = _md_to_text(f.read())
            docs.append(Document(page_content=txt, metadata={"source": name}))
        elif name.lower().endswith(".png"):
            try:
                img = Image.open(path)
                txt = pytesseract.image_to_string(img)
                docs.append(Document(page_content=txt, metadata={"source": name}))
            except Exception as e:
                logger.warning(f"OCR gagal untuk {name}: {e}")
        else:
            logger.info(f"Skip unsupported: {name}")
    # Tambahkan numbering line untuk txt/md/png
    enhanced = []
    for d in docs:
        meta = dict(d.metadata) if d.metadata else {}
        meta.setdefault("source", meta.get("source") or os.path.basename(meta.get("file_path", "unknown")))
        if "page" not in meta and meta.get("source", "").lower().endswith((".txt", ".md", ".png")):
            # simulasikan line range saat chunking (di indexer)
            meta.setdefault("line_start", 1)
            meta.setdefault("line_end", None)
        enhanced.append(Document(page_content=d.page_content, metadata=meta))
    return enhanced

def load_documents_from_payload(files_payload: List[Dict[str, Any]]) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """Simpan file dari payload (base64/text), lalu load menjadi Documents"""
    saved_info: List[Dict[str, Any]] = []
    for f in files_payload:
        fn = f["filename"]
        mime = f.get("mime", "")
        if f.get("base64"):
            _save_base64(fn, f["base64"])
        elif f.get("text") is not None:
            _save_text(fn, f["text"])
        else:
            raise ValueError(f"{fn}: butuh base64 atau text.")
        saved_info.append({"filename": fn, "mime": mime})

    # Setelah tersimpan → load dari folder
    docs = load_documents_from_dir()
    return docs, saved_info
