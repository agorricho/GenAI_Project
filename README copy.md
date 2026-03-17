# MSA 8700 — Milestone 2 Pipeline

Fetches 20 arXiv papers on ESG, chunks each into 10 segments, embeds via local Ollama (`nomic-embed-text`), and upserts 200 vectors into Qdrant Cloud (GCP).

---

## Prerequisites

- Python 3.12 via conda environment `gra-venv`
- Ollama installed locally with `nomic-embed-text` pulled
- `.env` file in `/home/agorricho/MSA8700/` (see Section 2)

---

## Section 1 — Configure Ollama

```bash
# Start the local Ollama server (runs on localhost:11434 by default)
ollama serve

# In a new terminal — pull the embedding model
ollama pull nomic-embed-text

# Verify the model is available
ollama list
```

---

## Section 2 — `.env` File Format

Create a `.env` file in `/home/agorricho/MSA8700/` with the following contents:

```
QDRANT_URL = https://fc2a9938-...qdrant.io
QDRANT_API_KEY = <your-qdrant-jwt>
0LLAMA = <your-ollama-api-key>
```

> **Note:** The `0LLAMA` key name starts with a zero (`0`), not a capital letter O. The Ollama API key is loaded but not used for local connections — it is retained for future cloud migration.

---

## Section 3 — Install Dependencies

```bash
conda activate gra-venv
pip install -r requirementsM2.txt
```

---

## Section 4 — Run the Pipeline

### Option A: Single command (recommended)

```bash
conda activate gra-venv
cd /home/agorricho/MSA8700
python run_pipeline.py
```

`run_pipeline.py` runs both steps in sequence with a 120-second pause between them to allow Ollama to free RAM before embedding begins.

### Option B: Run steps manually

```bash
# Step 1 — Download PDFs from arXiv and chunk text to chunks.json
python download_papers.py

# Step 2 — Embed chunks and upsert vectors to Qdrant (run after Step 1)
python embed_upsert.py
```

> `embed_upsert.py` is crash-safe: it checkpoints progress to `progress.json` after each paper and resumes from the last completed paper on re-run.

---

## Section 5 — Verify 200 Vectors in Qdrant Cloud

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
from qdrant_client import QdrantClient
c = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
print('Vectors stored:', c.get_collection('msa8700_m2').vectors_count)
"
```

Expected output: `Vectors stored: 200`

---

## Architecture

```
arXiv API → 20 PDFs (./papers/)
         → pdfplumber (extract_text)            [download_papers.py]
         → chunk_text (10 segments each)
         → chunks.json (200 entries)
              ↓ 120-second pause (run_pipeline.py)
         → Ollama local (nomic-embed-text, 768-dim)  [embed_upsert.py]
         → Qdrant Cloud GCP (msa8700_m2, 200 vectors)
```

### Pipeline files

| File | Role |
|------|------|
| `run_pipeline.py` | Orchestrator — runs Step 1, waits 120 s, runs Step 2 |
| `download_papers.py` | Step 1 — arXiv fetch, PDF extraction, chunking → `chunks.json` |
| `embed_upsert.py` | Step 2 — Ollama embedding, Qdrant upsert, checkpoint/resume |
| `requirementsM2.txt` | Python dependencies for this pipeline |
| `chunks.json` | Intermediate output: 200 text chunks (generated, not committed) |
| `progress.json` | Checkpoint file for `embed_upsert.py` resume (generated, not committed) |
| `papers/` | Downloaded PDFs (generated, not committed) |

### Configuration (top of each script)

| Setting | File | Default |
|---------|------|---------|
| `TOPIC` | `download_papers.py` | `"ESG Performance impact on Firm Value"` |
| `MAX_PAPERS` | `download_papers.py` | `20` |
| `CHUNKS_PER_PAPER` | `download_papers.py` | `10` |
| `TEXT_LIMIT` (chunk size) | `download_papers.py` | `3000` chars |
| `COLLECTION` | `embed_upsert.py` | `"msa8700_m2"` |
| `EMBED_MODEL` | `embed_upsert.py` | `"nomic-embed-text"` |
| `VECTOR_DIM` | `embed_upsert.py` | `768` |
| `TEXT_LIMIT` (embed size) | `embed_upsert.py` | `1500` chars |
| `OLLAMA_BASE_URL` | `embed_upsert.py` | `http://localhost:11434/api/embeddings` |
