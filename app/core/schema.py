from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = 5
    temperature: float = 0.2
    max_tokens: int = 512

class AskResponse(BaseModel):
    question: str
    context_sources: List[str]
    answer: str
    metadata: Dict[str, Any]

class IngestFile(BaseModel):
    filename: str
    mime: str
    base64: Optional[str] = None
    text: Optional[str] = None

class IngestOptions(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    ocr: Optional[bool] = True

class IngestJsonRequest(BaseModel):
    files: List[IngestFile]
    options: Optional[IngestOptions] = None

class IngestJsonResponse(BaseModel):
    status: str
    indexed_files: List[Dict[str, Any]]
    vector_store: str
    total_chunks: int

class SourcesResponse(BaseModel):
    documents: List[Dict[str, Any]]
    vector_store: str
    total_chunks: int

class HealthResponse(BaseModel):
    status: str
    vector_index_ready: bool
    provider: str
    model: str
