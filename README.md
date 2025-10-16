# RAG Chatbot 

> A single ask–answer chatbot using **RAG** (Retrieval-Augmented Generation) with **FastAPI** + **OpenAI** + **Chroma**.
> Supports ingestion of **.txt, .pdf, .md, .png** (PNG via **Tesseract OCR**).
> Primary endpoint: `POST /ask`.

Author: Indah Monisa Firdiantika
---

## 1) Install Python & Create a New Environment (named `RAG_chatbot`)

### Windows (recommended)

1. Install **Python 3.10 or 3.11**
   Download from [https://www.python.org/downloads/](https://www.python.org/downloads/) and tick **“Add Python to PATH”**.

2. Create and activate a virtual env named **`RAG_chatbot`**:

   ```powershell
   python -m venv RAG_chatbot
   .\RAG_chatbot\Scripts\Activate.ps1
   python -V
   ```

   You should see the prompt prefixed with `(RAG_chatbot)`.

> In VS Code, set the interpreter to `RAG_chatbot\Scripts\python.exe` (Ctrl+Shift+P → *Python: Select Interpreter*).

---

## 2) Install Requirements

Create a **`.env`** file in the project root with your configuration:

```
# OpenAI credentials & defaults
OPENAI_API_KEY=sk-...your-key...
PROVIDER=openai
EMBEDDINGS_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Paths & chunking
DATA_DIR=./data
PERSIST_DIR=./index
CHUNK_SIZE=800
CHUNK_OVERLAP=120
CHROMA_TELEMETRY_DISABLED=1

# (OCR – set after installing Tesseract, see section 3)
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
# TESSERACT_LANG=eng
```

Then install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Install Tesseract OCR (for `.png` ingestion)

### Using **winget** (quickest)

```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```

Verify:

```powershell
tesseract --version
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

Add to your `.env` (recommended for stability):

```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_LANG=eng     # or ind / eng+ind
```

> If `tesseract` is still not recognized, close and reopen your terminal (or VS Code), or ensure `C:\Program Files\Tesseract-OCR` is on your PATH.

### If `tesseract` is “not recognized” (PATH issue)

Tesseract is installed but your current terminal session doesn’t know where it is.

**Fix for the current session (quick):**

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:Path += ";C:\Program Files\Tesseract-OCR"
```
---

## 4) How to Run the Project

### 4.1. Provide your dataset (choose one)

**A) Put files in `./data` (simplest)**
Copy your documents into `.\data\` (mix allowed: `.txt`, `.pdf`, `.md`, `.png`).
Then **build the vector index**:

```powershell
python -m app.ingest.indexer
```

You should see something like: `Indexed: N chunks`.

---

### 4.2. Start the API

```powershell
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Open **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Quick checks:**

* `GET /health` → basic status & model.
* `GET /sources` → indexed documents and total chunk count.
* `POST /ask` → ask questions over your data.

**Example – `POST /ask` (PowerShell):**

```powershell
$body = '{"question":"can you explain yolov6?","top_k":6,"temperature":0.2,"max_tokens":512}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST -ContentType "application/json" -Body $body
```

**Example response:**

```json
{
  "question": "can you explain yolov6?",
  "context_sources": [
    "Screenshot 2025-10-16 072406.png: lines 1–8",
    "Screenshot 2025-10-16 072349.png: lines 1–10",
    "Screenshot 2025-10-16 072430.png: lines 1–6",
    "Screenshot 2025-10-16 072406.png: lines 1–12",
    "Screenshot 2025-10-16 072349.png: lines 1–12"
  ],
  "answer": "YOLOv6 is a detection model designed with hardware-friendly principles, featuring two scalable parameterizable backbones and necks to accommodate models of different sizes. It includes an efficient decoupled head utilizing a hybrid-channel strategy. YOLOv6 achieves notable performance, hitting 35.9% Average Precision (AP) on the COCO dataset at a throughput of 1234 FPS on an NVIDIA Tesla T4 GPU. The model also offers a faster version, YOLOv6-L-ReLU, which achieves 51.7% AP with a latency of 8.8 ms, outperforming other models like YOLOX-L and PPYOLOE-L in both accuracy and speed. Additionally, YOLOv6 incorporates a self-distillation strategy and various advanced detection techniques to enhance performance.",
  "metadata": {
    "model": "gpt-4o-mini",
    "retrieval_engine": "Chroma",
    "timestamp": "2025-10-16T01:03:10.957011Z",
    "latency_ms": 6434,
    "top_k": 6,
    "avg_similarity": 0.371,
    "provider": "openai"
  }
}
```

---

## 5) How to Delete the Index (and why)

**Command (Windows PowerShell):**

```powershell
Remove-Item -Recurse -Force .\index
```

**Why you might need this:**

* You **changed the embedding model** (`EMBEDDING_MODEL`) which changes vector dimensions; the old index becomes incompatible.
* You want a **clean rebuild** after major dataset changes.

After deleting, rebuild:

```powershell
python -m app.ingest.indexer
```

---

## 6) What Dataset Is Used & What Tools Power the System

### Datasets

* Drop files into `./data` and run the indexer, **or** upload via `POST /ingest-json`.
* Supported:

  * **`.txt`, `.md`** → parsed as text.
  * **`.pdf`** → parsed via `pypdf`/`pdfplumber`. (If it’s a scanned PDF without text, OCR it to PNG or make a searchable PDF first.)
  * **`.png`** → OCR via **Tesseract** (requires section 3 above).
Got it—I’ll give you a short, drop-in subsection you can paste under **“6) What Dataset Is Used & What Tools Power the System”** to explicitly state *your* dataset choice (YOLOv6 paper screenshots).

---

### Dataset used in this submission (my case)

For this submission, I used a **mixed dataset** consisting of:

* **PNG screenshots** from a paper (indexed via **OCR / Tesseract**),
* **One PDF** (`maskrcnn.pdf`) extracted as text,
* **One Markdown** file (`yolov7.md`),
* **One TXT** file (`yolov10.txt`).

This demonstrates the system’s ability to handle **image-only sources** (via OCR) as well as **text-based** documents (PDF/MD/TXT) in the same index.

**Folder example**

```
data/
  maskrcnn.pdf                       # pdf file
  Screenshot 2025-10-16 072349.png   # png file (pages 1–2)
  Screenshot 2025-10-16 072406.png   # png file (pages 3–4)
  Screenshot 2025-10-16 072417.png   # png file (pages 5–6)
  Screenshot 2025-10-16 072430.png   # png file (pages 7–8)
  Screenshot 2025-10-16 072440.png   # png file (pages 9–10)
  Screenshot 2025-10-16 072450.png   # png file (pages 11–12)
  yolov7.md                          # markdown file
  yolov10.txt                        # text file
```

**Notes**

* PNGs are converted to text via Tesseract OCR (`TESSERACT_CMD` and `TESSERACT_LANG` set in `.env`).
* PDF/MD/TXT are parsed directly and chunked using the same parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`).

### Tools / Stack

* **API framework**: **FastAPI** (interactive docs at `/docs`).
* **Vector store**: **Chroma** (via `langchain-chroma`, auto-persist at `./index`).
* **Embeddings**: **OpenAI** `text-embedding-3-small` (or `-large`).
* **LLM (generator)**: **OpenAI** `gpt-4o-mini` by default.
* **RAG flow**:

  1. Ingest → clean → split (RecursiveCharacterTextSplitter)
  2. Embed → store in Chroma
  3. Retrieve (MMR) → prompt → generate answer
* **Quality-of-life endpoints**:

  * `GET /health` – service/model status
  * `GET /sources` – which documents are indexed
  * `POST /ingest-json` – ingest without using the filesystem
  * `POST /reindex` – rebuild from `./data`
  * `GET /metrics` – simple performance metrics (q_count, latency)

---

## 7) Additional Notes & Best Practices

* **Language matching**: Ask in the same language as the documents for best retrieval (English docs → ask in English).
* **For broad questions** (e.g., “explain the architecture”), increase `top_k` to **8–12** so more relevant chunks are considered.
* **Changing embedding models**: Always delete `./index` and re-run the indexer to avoid dimension mismatch errors.
* **Performance**:

  * For demos, run without `--reload` for lower latency.
  * Keep the index on a fast local SSD.
  * Use `temperature: 0.0` for factual/technical questions.
* **Troubleshooting**:

  * `api_key must be set`: ensure `.env` is loaded and `OPENAI_API_KEY` is present.
  * Dimension mismatch error: delete `./index` and rebuild after changing `EMBEDDING_MODEL`.
  * `TesseractNotFoundError`: ensure Tesseract is installed and `TESSERACT_CMD` points to the executable; restart terminal.

---

## Project Structure (brief)

```
app/
  api/main.py                 # FastAPI routes (/ask, /ingest-json, /reindex, /sources, /health, /metrics)
  core/                       # config (.env), schema (pydantic), logger
  ingest/                     # indexer CLI, loaders (txt/pdf/md/png), cleaners
  rag/                        # retriever (MMR), prompt builder, generator (OpenAI)
  vectorstore/chroma_store.py # Chroma setup (persisted in ./index)
data/                         # place your documents here (if using filesystem mode)
index/                        # Chroma persistence (auto-created)
requirements.txt
.env                          # your secrets & config (do not commit)
```

---

## Example Submission Flow

1. Create env:

   ```powershell
   python -m venv RAG_chatbot
   .\RAG_chatbot\Scripts\Activate.ps1
   ```
2. Set up `.env` (with your `OPENAI_API_KEY`).
3. Install deps:

   ```powershell
   pip install -r requirements.txt
   ```
4. Put a few files into `./data` and build the index:

   ```powershell
   python -m app.ingest.indexer
   ```
5. Start API:

   ```powershell
   uvicorn app.api.main:app
   ```
6. Test in **Swagger** (`/docs`) & push to GitHub (exclude `.env`, `index/`, `RAG_chatbot/`).

---
## 🎥 Demo Video

![RAG Chatbot – Demo](./demo.gif)

---
## Future-Work

Fine-tuning (optional): Not included in this submission. For this task, accuracy is driven primarily by Retrieval-Augmented Generation (RAG). Fine-tuning/LoRA could be added later to align style or domain tone once we have a larger, high-quality supervised dataset.
