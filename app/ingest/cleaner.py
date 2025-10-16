import re
from langchain.schema import Document

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def clean_documents(docs):
    cleaned = []
    for d in docs:
        cleaned.append(Document(page_content=normalize_text(d.page_content), metadata=d.metadata))
    return cleaned
