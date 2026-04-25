# DAIS — Project Documentation
## MSA 8700: Building Generative AI Business Solutions
**System:** Document-Driven Agentic Intelligence System (Variation B — Research Advisor)
**Authors:** Alejandro Gorricho
**Last Updated:** 2026-04-25

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Milestone History](#3-milestone-history)
4. [Architecture](#4-architecture)
5. [Environment Setup](#5-environment-setup)
6. [Running the System](#6-running-the-system)
7. [Configuration Reference (.env)](#7-configuration-reference-env)
8. [Debugging Log](#8-debugging-log)
9. [Known Issues & Workarounds](#9-known-issues--workarounds)
10. [Upcoming Milestones](#10-upcoming-milestones)

---

## 1. Project Overview

DAIS ingests ESG academic papers from arXiv and answers research questions with cited, evidence-backed responses using a multi-agent RAG (Retrieval-Augmented Generation) pipeline built with LangGraph.

**Core research question served:** *How does ESG (Environmental, Social, Governance) performance affect firm value?*

The pipeline covers the full lifecycle:
- Fetching and chunking PDFs from arXiv (M2)
- Embedding chunks into Qdrant Cloud (M2)
- Rephrasing, retrieving, extracting, and synthesizing answers via LangGraph agents (M3)
- Evaluating pipeline quality across 5 metrics against a human-authored Q&A dataset (M4)

---

## 2. Repository Structure

```
GenAI_Project/
├── .env                        # Root-level env for M2 scripts (ARC GPU endpoint) — see §7
├── download_papers.py          # M2 Step 1: arXiv fetch + PDF chunking → chunks.json
├── embed_upsert.py             # M2 Step 2: chunk embedding + Qdrant upsert
├── run_pipeline.py             # M2 end-to-end runner (calls both scripts sequentially)
├── chunks.json                 # M2 output: all chunks from 20 papers (~1400 entries)
├── papers/                     # Downloaded arXiv PDFs
├── requirementsM2.txt          # M2 dependencies (root-level scripts)
├── MSA8700_Project.md          # CLAUDE.md — AI assistant project guidance
├── README.md                   # Project README with milestone overview
│
├── Milestone2/                 # M2 corrected submission copy
│   ├── .env                    # Milestone2-specific credentials (Ollama cloud)
│   ├── download_papers copy.py
│   ├── embed_upsert copy.py
│   ├── requirements.txt
│   └── run_pipeline.py
│
└── Milestone3/                 # M3/M4 source of truth — all new development here
    ├── .env                    # Milestone3 credentials (Ollama cloud + Qdrant)
    ├── app_interface.py        # Streamlit UI (5 tabs)
    ├── requirements.txt        # M3/M4 dependencies
    ├── src/
    │   ├── state.py            # ResearchState TypedDict shared across all nodes
    │   ├── pipeline.py         # LangGraph graph builder + run_query() entry point
    │   └── agents/
    │       ├── rephraser.py    # Node 1: NL question → academic search terms
    │       ├── retriever.py    # Node 2: fastembed → Qdrant top-5 lookup
    │       ├── extractor.py    # Node 3: per-chunk relevance filter + finding extraction
    │       └── synthesizer.py  # Node 4: combines findings into cited answer
    └── eval/
        ├── eval_template.csv   # Blank annotator sheet (50 rows)
        ├── csv_to_json.py      # Converts filled CSV → eval_dataset.json
        ├── eval_dataset.json   # 50-item human-authored Q&A test set
        ├── generate_dataset.py # Alternative: LLM-generated 75-item dataset
        ├── metrics.py          # 5 scoring functions + OllamaJudge
        ├── run_eval.py         # Eval runner (resumable, atomic checkpoint per item)
        └── results/
            ├── eval_results.json
            └── eval_results.csv
```

---

## 3. Milestone History

| Milestone | Description | Status | Points |
|-----------|-------------|--------|--------|
| M1 | Weekly assignments — Ollama, Pydantic, web scraping, Qdrant basics | ✅ Complete | — |
| M2 | RAG ingestion pipeline — arXiv PDFs → chunks → Qdrant Cloud | ✅ Complete | — |
| M3 | LangGraph RAG pipeline — Rephraser → Retriever → Extractor → Synthesizer + Streamlit UI | ✅ Complete | — |
| M4 | Evaluation framework — 50-item Q&A dataset, 5 metrics, resumable eval runner | ✅ Complete | — |
| **M5** | Iterative improvements + ablation study | Upcoming | 16 |
| **Final** | Deployed system + report + video demo | Upcoming | 12 |

---

## 4. Architecture

### 4.1 M2 — Ingestion Pipeline

```
arXiv API
  → download_papers.py
      ├── arxiv.Search("ESG Performance impact on Firm Value", max=20)
      ├── pdfplumber PDF text extraction
      ├── RecursiveCharacterTextSplitter (chunk_size=1200, overlap=200)
      │     separators: ["\n\n", "\n", ". ", " "]
      └── chunks.json  (~20–103 chunks per paper, variable)
  → embed_upsert.py
      ├── Ollama /api/embeddings  (mxbai-embed-large, 1024-dim)
      ├── Qdrant Cloud upsert    (collection: msa8700_m3)
      └── progress.json          (crash-safe checkpoint per paper)
```

**Embedding model:** `mxbai-embed-large` — 334M params, BERT family, F16, 1024-dim, MTEB retrieval leader.
**Qdrant collection:** `msa8700_m3`, cosine distance, 1024-dim vectors.

**Payload schema per Qdrant vector:**

| Field | Type | Description |
|-------|------|-------------|
| `title` | str | Paper title |
| `authors` | str | Comma-separated author names |
| `year` | int | Publication year |
| `arxiv_id` | str | arXiv short ID |
| `abstract` | str | Full abstract text |
| `pdf_path` | str | Local path to downloaded PDF |
| `chunk_index` | int | 0-based index of this chunk within the paper |
| `chunk_total` | int | Total chunks for this paper |
| `text` | str | Raw chunk text |

### 4.2 M3 — LangGraph Query Pipeline

```
run_query(question)
    │
    ├── rephraser.py    ChatOllama (temp=0) — compact academic search terms
    ├── retriever.py    fastembed (nomic-ai/nomic-embed-text-v1.5, 768-dim) → Qdrant top-5
    │                   NOTE: populates chunk "text" from payload["abstract"], not raw chunk
    ├── extractor.py    per-chunk relevance filter + finding extraction (ChatOllama, temp=0)
    └── synthesizer.py  combines findings into cited academic answer (ChatOllama, temp=0)
    │
    └── returns {answer: str, citations: list[{title,authors,year}], chunks: list[dict]}
```

**State shared across all nodes:** `ResearchState` TypedDict in `src/state.py`.

**Module-level singletons:** Each agent instantiates `_llm`, `_qdrant`, `_embedder` at import time (not inside the node function). The `.env` walk also happens at module load time.

### 4.3 M3 — Streamlit App (`app_interface.py`)

Five tabs:

| Tab | Content |
|-----|---------|
| Overview | Project description |
| Paper Search | OpenAlex API search (sidebar controls: topic, count) |
| Research Framework | Rule-based placeholder causal extraction table + CSV download |
| Qdrant | Live connection status + collection list |
| Chat | Full pipeline via `run_query()`; conversation history in `st.session_state` |

### 4.4 M4 — Evaluation Framework

**Two-step flow:**

```
Step 1 (one-time):  eval_template.csv  →  csv_to_json.py  →  eval_dataset.json
Step 2 (live):      eval_dataset.json  →  run_eval.py     →  eval_results.json / .csv
```

**Five metrics and weights:**

| Metric | Method | Weight |
|--------|--------|--------|
| Faithfulness | LLM-as-judge (OllamaJudge) | 0.30 |
| Answer Relevancy | LLM-as-judge (OllamaJudge) | 0.25 |
| Semantic Similarity | Embedding cosine (fastembed) | 0.25 |
| Citation Recall | Fuzzy substring match | 0.15 |
| ROUGE-L | Lexical overlap | 0.05 |

**Dataset distribution (50 items):**

| `generation_mode` | Count | `difficulty` | `is_in_corpus` |
|-------------------|-------|-------------|----------------|
| `abstract` | 25 | `broad` | TRUE |
| `chunk` | 20 | `specific` | TRUE |
| `negative` | 5 | `out-of-scope` | FALSE |

**OllamaJudge:** wraps `/api/generate` at temperature=0; strips markdown fences; retries up to 3× on JSON parse failure.

**Checkpoint strategy:** `run_eval.py` writes atomically (`.tmp` rename) after each item — safe to kill and resume at any time.

### 4.5 Critical Embedding Compatibility Note

The M3 retriever (`Milestone3/src/agents/retriever.py`) uses **fastembed `nomic-ai/nomic-embed-text-v1.5` (768-dim)** to query `msa8700_m3`. The M2 ingestion pipeline embeds with **mxbai-embed-large (1024-dim)**. These dimensions are incompatible — Qdrant will reject queries on a dimension mismatch.

**Rule:** the embedding model used at ingest time must also be used at query time, or the collection must be rebuilt. This is an open incompatibility to resolve in M5.

---

## 5. Environment Setup

### 5.1 Conda Environment

```bash
conda activate gra_venv    # Python 3.12 — NOTE: underscore, not hyphen
```

### 5.2 Installing Dependencies

The `pip` / `pip3` commands at `~/.local/bin/` have a **broken shebang** pointing to a Python interpreter path that no longer exists (stale pre-conda install). Always invoke pip through the active Python interpreter:

```bash
# DO NOT use:  pip install ...   (broken shebang — "cannot execute: required file not found")
# USE INSTEAD:
python -m pip install -r requirements.txt
```

Install by milestone:

```bash
# M2 root scripts
conda activate gra_venv
python -m pip install -r /home/agorricho1/AI_Scientist2/GenAI_Project/requirementsM2.txt

# Milestone2/ copy
python -m pip install -r /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone2/requirements.txt

# Milestone3/ (M3 + M4)
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python -m pip install -r requirements.txt
python -m pip install fastembed    # ONNX embedder — one-time ~130 MB download on first run
python -m pip install --upgrade qdrant-client
```

### 5.3 .env File Locations

| Location | Used by | Notes |
|----------|---------|-------|
| `GenAI_Project/.env` | `embed_upsert.py`, `download_papers.py`, `run_pipeline.py` | Points to **ARC GPU** (`https://gpu-01.insight.gsu.edu:11443`) for embeddings. Created 2026-04-25. |
| `Milestone2/.env` | `Milestone2/` scripts | Points to Ollama Cloud (`https://api.ollama.com`) |
| `Milestone3/.env` | All `Milestone3/src/agents/` | Points to Ollama Cloud; used by M3 LangGraph pipeline and M4 eval |

**How `.env` discovery works:** every script calls `_find_env()` which walks up the directory tree from the script's own location (`Path(__file__).resolve().parents`) until it finds a `.env` file. Scripts in `GenAI_Project/` root will find `GenAI_Project/.env`; scripts in `Milestone3/` will find `Milestone3/.env`.

**Never place a `.env` inside `src/`** — it intercepts the walk before reaching the intended parent.

---

## 6. Running the System

All commands assume `conda activate gra_venv` is active.

### M2 — Full ingestion (download + embed)

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project

# End-to-end (recommended)
python run_pipeline.py

# Or step-by-step
python download_papers.py     # arXiv fetch + chunk → chunks.json
python embed_upsert.py        # embed + upsert → Qdrant msa8700_m3
```

`embed_upsert.py` is crash-safe — delete `progress.json` to force a full fresh run.

### M3 — Query pipeline (CLI)

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python -c "from src.pipeline import run_query; r = run_query('How does ESG affect firm performance?'); print(r['answer'])"
```

### M3 — Streamlit UI

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
streamlit run app_interface.py    # opens at http://localhost:8501
```

Use **Tab 5 (Chat)** for the full pipeline.

### M4 — Evaluation

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3

# Step 1 (one-time): fill eval/eval_template.csv, then convert
python eval/csv_to_json.py --validate-only    # dry-run check
python eval/csv_to_json.py                    # write eval_dataset.json

# Step 2: run evaluation (~30–90 min, auto-resumes on crash)
python eval/run_eval.py              # resume from checkpoint
python eval/run_eval.py --no-resume  # force full re-run
```

### Tests

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python -m pytest tests/test_m4_eval.py -v    # 57/57 pass (no live services needed)
```

### Verify Qdrant collection after ingest

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project
python -c "
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(next(p/'.env' for p in Path('.').resolve().parents if (p/'.env').exists()))
from qdrant_client import QdrantClient
c = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
info = c.get_collection('msa8700_m3')
print('Vectors stored:', info.vectors_count)
print('Vector dim:', info.config.params.vectors.size)
"
```

---

## 7. Configuration Reference (.env)

### `GenAI_Project/.env` (M2 root scripts — ARC GPU)

```
QDRANT_URL="https://<cluster>.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY="<qdrant-jwt>"
OLLAMA_BASE_URL="https://gpu-01.insight.gsu.edu:11443"
EMBED_MODEL="mxbai-embed-large"
VECTOR_DIM="1024"
```

Auth for the ARC Ollama endpoint is sourced from the OS environment variable `ARC_OLLAMA_API` (already set on the cluster). It is **not** stored in the `.env` file.

### `Milestone3/.env` (M3/M4 pipeline — Ollama Cloud)

```
QDRANT_URL="https://<cluster>.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY="<qdrant-jwt>"
0LLAMA="<ollama-cloud-api-key>"    # NOTE: leading zero, not capital O
OLLAMA_BASE_URL="https://api.ollama.com"
OLLAMA_EMBED_BASE_URL="https://api.ollama.com"
M3_MODEL=gpt-oss:120b
```

### Key `.env` variables

| Variable | Script | Description |
|----------|--------|-------------|
| `QDRANT_URL` | All | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | All | Qdrant JWT bearer token |
| `OLLAMA_BASE_URL` | `embed_upsert.py`, M3 agents | Base URL for Ollama API; `/api/embeddings` appended automatically by `embed_upsert.py` |
| `EMBED_MODEL` | `embed_upsert.py` | Embedding model name (default: `mxbai-embed-large`) |
| `VECTOR_DIM` | `embed_upsert.py` | Embedding dimension as integer string (default: `"1024"`) |
| `0LLAMA` | M3 agents, Milestone2 | Ollama Cloud API key (leading zero is intentional) |
| `M3_MODEL` | M3 agents | LLM model name for ChatOllama nodes |
| `ARC_OLLAMA_API` | `embed_upsert.py` | ARC GPU bearer token — read from **OS env**, not `.env` |

---

## 8. Debugging Log

### Bug 1 — `embed_upsert.py` 500 Server Error (2026-04-25)

**Symptom:**
```
[3/20] Embedding: The China Trade Shock and the ESG Performances of US firms (103 chunks)
    Embed failed (attempt 1), retrying in 10s: 500 Server Error: Internal Server Error
                                                for url: http://localhost:11434/api/embeddings
    Embed failed (attempt 2), retrying in 20s: 500 Server Error: ...
Traceback: requests.exceptions.HTTPError: 500 Server Error
```

Papers 1 and 2 succeeded (38 and 34 chunks respectively). Paper 3 failed with 103 chunks.

**Root Cause:**
`embed_upsert.py` hardcoded the Ollama URL:
```python
OLLAMA_BASE_URL = "http://localhost:11434/api/embeddings"   # line 56, original
```
The local Windows Ollama server ran out of memory or hit a context limit while processing the 103-chunk paper. The script ignored the `OLLAMA_BASE_URL` key in `.env` entirely.

Additionally, no `.env` existed at `GenAI_Project/` root on the Linux cluster — the `_find_env()` walk-up would fail to find credentials when running from this directory.

**Investigation steps:**
1. Confirmed ARC GPU endpoint is reachable and `ARC_OLLAMA_API` env var is set on the cluster.
2. Queried ARC `/api/tags` to verify available models:
   ```bash
   curl -s -H "Authorization: Bearer $ARC_OLLAMA_API" \
     https://gpu-01.insight.gsu.edu:11443/api/tags
   ```
   **Result:** `mxbai-embed-large:latest` confirmed present (334M params, BERT, F16).
3. Verified the original `.env` files are only in `Milestone2/` and `Milestone3/` subdirectories — not reachable by the root-level scripts' walk-up logic.

**Fix applied to `embed_upsert.py`:**

Before (hardcoded):
```python
OLLAMA_API_KEY = os.getenv("0LLAMA")
...
COLLECTION      = "msa8700_m3"
OLLAMA_BASE_URL = "http://localhost:11434/api/embeddings"   # <-- hardcoded
EMBED_MODEL     = "mxbai-embed-large"
VECTOR_DIM      = 1024
```

After (env-driven):
```python
# Auth: ARC GPU bearer token takes priority; fall back to Ollama cloud key
OLLAMA_API_KEY = os.environ.get("ARC_OLLAMA_API") or os.getenv("0LLAMA")
...
COLLECTION  = "msa8700_m3"
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
VECTOR_DIM  = int(os.getenv("VECTOR_DIM", "1024"))
_ollama_base    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_BASE_URL = f"{_ollama_base}/api/embeddings"
```

**New file created:** `GenAI_Project/.env`
```
OLLAMA_BASE_URL="https://gpu-01.insight.gsu.edu:11443"
EMBED_MODEL="mxbai-embed-large"
VECTOR_DIM="1024"
QDRANT_URL="<same as Milestone3/.env>"
QDRANT_API_KEY="<same as Milestone3/.env>"
```
ARC auth (`ARC_OLLAMA_API`) is consumed directly from the OS environment — not stored in file.

**Smoke test result (post-fix):**
```
Loaded .env from: /home/agorricho1/AI_Scientist2/GenAI_Project/.env
URL:   https://gpu-01.insight.gsu.edu:11443/api/embeddings
Model: mxbai-embed-large
Auth:  ARC_OLLAMA_API
Status: 200
Embedding dim: 1024  first 5: [-0.038, 0.029, -0.034, 0.031, -0.031]
```

### Bug 1 — Follow-up Fix: ARC Endpoint Overload on Large Papers (2026-04-25)

**Context:** After the URL/env fix above, 500 errors persisted on papers with 103+ chunks when running against the ARC GPU endpoint. Root cause: rapid back-to-back chunk embedding requests overwhelm the remote Ollama server under load, and the original 3-retry logic was insufficient to recover.

**Root Cause (refined):**
- No throttle between chunk embed calls — 103 requests fired sequentially with no pause
- Only 3 retries with max backoff of 40s — not enough for ARC to recover under sustained load
- Chunks near the model's context window limit (~8K tokens) may trigger a distinct server-side OOM path

**Fix applied to `embed_upsert.py` (BMAD quick-dev, 2026-04-25):**

Three targeted changes, no structural refactoring:

**1. Increased retry count + 4xx/5xx distinction**

```python
# Before
def embed_with_retry(text: str, max_retries: int = 3) -> list[float]:
    ...
    except requests.exceptions.HTTPError as e:
        if attempt < max_retries - 1:     # retried 4xx AND 5xx identically

# After
def embed_with_retry(text: str, max_retries: int = 5) -> list[float]:
    ...
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        if 0 < status < 500:    # 4xx — auth/not-found: raise immediately, no retry
            raise
        # 5xx and status==0 → retry with backoff: 10s, 20s, 40s, 80s, 160s
```

**2. Inter-chunk throttle (env-configurable)**

```python
CHUNK_DELAY = float(os.getenv("EMBED_CHUNK_DELAY", "0.5"))  # seconds between chunk embeds

# In main loop — after each successful embed:
if CHUNK_DELAY > 0:
    time.sleep(CHUNK_DELAY)
```

**3. Text truncation for oversized chunks (env-configurable)**

```python
MAX_CHUNK_CHARS = max(1, int(os.getenv("MAX_CHUNK_CHARS", "6000")))  # truncate before embedding

# In main loop:
text_to_embed = chunk["text"][:MAX_CHUNK_CHARS]   # embed-only; full text still stored in payload
vector = embed_with_retry(text_to_embed)
```

**`.env` tuning knobs added:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBED_CHUNK_DELAY` | `0.5` | Seconds between chunk embed calls; set to `0` to disable |
| `MAX_CHUNK_CHARS` | `6000` | Max chars sent to embed model; full text still stored in Qdrant payload |

**Adversarial review findings (all resolved):**

| Finding | Resolution |
|---------|-----------|
| `status==0` (no response) was treated as 4xx and not retried | Fixed: `0 < status < 500` — status==0 now retries |
| `MAX_CHUNK_CHARS=0` would produce empty string embed | Fixed: `max(1, int(...))` guard |
| Embeddings computed on truncated text but full text stored in payload (silent divergence) | Documented inline: comment on embed call line |

---

### Bug 2 — `pip install` fails: "cannot execute: required file not found" (2026-04-25)

**Symptom:**
```
(gra_venv) $ pip install -r requirements.txt
bash: /home/agorricho1/.local/bin/pip: cannot execute: required file not found
(gra_venv) $ pip3 install -r requirements.txt
bash: /home/agorricho1/.local/bin/pip3: cannot execute: required file not found
```

**Root Cause:**
The `pip` and `pip3` wrapper scripts at `~/.local/bin/` contain a shebang line (`#!/path/to/python`) pointing to a Python interpreter that was installed before the conda environment and no longer exists at that path. The conda `gra_venv` environment has its own pip at `/home/agorricho1/miniconda3/envs/gra_venv/bin/pip`, but the shell resolves `pip` to the stale `~/.local/bin/pip` first via `$PATH`.

**Fix:** invoke pip through the active Python interpreter, which bypasses the broken wrapper:
```bash
python -m pip install -r requirements.txt
```

This routes through `/home/agorricho1/miniconda3/envs/gra_venv/bin/python` directly, which has a working pip (v26.0.1).

**Packages successfully installed for Milestone2/requirements.txt:**
`langgraph 1.1.9`, `langchain 1.2.15`, `langchain-core 1.3.2`, `langchain-ollama 1.1.0`,
`streamlit 1.56.0`, `langchain-text-splitters 1.1.2`, plus all transitive dependencies.

**Permanent workaround:** always use `python -m pip` inside `gra_venv`:
```bash
# Correct on this cluster:
python -m pip install <package>
python -m pip install -r requirements.txt

# Broken (do not use):
pip install <package>
pip3 install <package>
```

---

## 9. Known Issues & Workarounds

### KI-001 — Embedding Dimension Mismatch (M2 vs M3)

| Layer | Model | Dimension | Collection |
|-------|-------|-----------|------------|
| M2 ingest (`embed_upsert.py`) | `mxbai-embed-large` | **1024** | `msa8700_m3` |
| M3 retriever (`retriever.py`) | `nomic-ai/nomic-embed-text-v1.5` (fastembed) | **768** | `msa8700_m3` |

These are incompatible. The M3 retriever will return zero results or fail if the collection was built with 1024-dim vectors.

**Workaround:** Rebuild `msa8700_m3` using nomic-embed-text at 768-dim, OR update the M3 retriever to use mxbai-embed-large at 1024-dim. Resolution deferred to M5.

### KI-002 — Broken `~/.local/bin/pip`

Documented in Bug 2 above. Always use `python -m pip` on this cluster.

### KI-003 — `embed_upsert.py` CHUNKS_FILE / PROGRESS_FILE use relative paths

`CHUNKS_FILE = "./chunks.json"` and `PROGRESS_FILE = "./progress.json"` resolve relative to the **current working directory**, not the script's location. The script must be run from `GenAI_Project/`:

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project
python embed_upsert.py    # ✅ correct
python GenAI_Project/embed_upsert.py    # ❌ chunks.json not found
```

### KI-004 — `embed_upsert.py` verify block at end uses hardcoded relative `.env`

Lines 181–187 print a verification snippet that calls `load_dotenv('.env')`. This will only find `.env` if run from the `Milestone3/` directory. Ignore or adapt this block when verifying from other directories.

---

## 10. Upcoming Milestones

### M5 — Iterative Improvements + Ablation Study (16 pts)

Candidate improvements:
- **Fix KI-001** — align embedding model between M2 ingest and M3 retriever (either rebuild collection with nomic 768-dim, or swap M3 retriever to mxbai 1024-dim)
- Chunking strategy upgrade (current 1200-char fixed size → semantic/sliding window)
- Retriever top-k tuning and score threshold filtering
- Synthesizer prompt engineering improvements
- Ablation study comparing pipeline variants across M4 metrics

### Final — Deployed System + Report + Video Demo (12 pts)

- Streamlit app deployed and accessible
- Final written report covering design decisions, evaluation results, ablation findings
- Video demo walkthrough

---

*Document maintained by: Alejandro Gorricho + Claude Code (AI assistant)*
*Repository: `/home/agorricho1/AI_Scientist2/GenAI_Project/`*
