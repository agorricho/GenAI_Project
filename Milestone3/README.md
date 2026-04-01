# Milestone 3 — CAIS Research Advisor

A LangGraph-powered RAG pipeline with a Streamlit chat interface for querying academic papers stored in Qdrant Cloud.

## Architecture

```
User query
    │
    ▼
Rephraser  →  converts natural-language question to academic search terms (Ollama LLM)
    │
    ▼
Retriever  →  embeds rephrased query via fastembed (nomic-ai/nomic-embed-text-v1.5, ONNX)
           →  vector searches Qdrant Cloud collection msa8700_m3 (top-5 hits)
    │
    ▼
Extractor  →  per-chunk relevance filter + finding extraction (Ollama LLM)
    │
    ▼
Synthesizer →  combines findings into a cited academic answer (Ollama LLM)
    │
    ▼
Answer + Citations
```

**Key files:**

| File | Role |
|------|------|
| `app_interface.py` | Streamlit UI — 5 tabs including the Chat interface |
| `src/pipeline.py` | `run_query()` entry point — compiles the LangGraph graph |
| `src/state.py` | `ResearchState` TypedDict shared across all nodes |
| `src/agents/rephraser.py` | Node 1 — query rephrasing |
| `src/agents/retriever.py` | Node 2 — fastembed + Qdrant vector search |
| `src/agents/extractor.py` | Node 3 — per-chunk finding extraction |
| `src/agents/synthesizer.py` | Node 4 — answer synthesis + citation list |

---

## Setup

### 1. Environment

```bash
conda activate gra-venv
pip install -r requirements.txt
pip install fastembed          # ONNX embedder (downloaded once on first run)
pip install --upgrade qdrant-client  # ensure version compatibility with Qdrant Cloud
```

### 2. Credentials — `.env` file

Place a `.env` file at the **repo root** (`MSA8700/.env`). Do **not** put a `.env` inside `src/` — all agents walk up the directory tree to find credentials, and a file in `src/` will intercept the walk before reaching the root.

Required keys:

```env
QDRANT_URL="https://<your-cluster>.gcp.cloud.qdrant.io"
QDRANT_API_KEY="<your-qdrant-api-key>"
OLLAMA_BASE_URL="https://api.ollama.com"   # or http://localhost:11434 for local Ollama
0LLAMA="<your-ollama-cloud-api-key>"       # omit if using local Ollama
M3_MODEL=llama3.2                          # or any model available on your Ollama instance
```

> `0LLAMA` is the API key env var name used by all agent files. The leading zero is intentional.

---

## Running the Pipeline (CLI)

Test the full pipeline from the command line:

```bash
cd /mnt/c/Users/alejo/OneDrive/Documents/MSA8700/GenAI_Project/Milestone3 && conda activate gra-venv && python -c "from src.pipeline import run_query; r = run_query('How does ESG affect firm performance?'); print(r['answer'])"
```

Expected terminal output:
```
[Rephraser] Converting query to academic search terms...
[Rephraser] '...' → '...'
[Retriever] Embedding query and searching 'msa8700_m3'...
[Retriever] Retrieved 5 chunks (top score: 0.xxxx)
[Extractor] Extracting findings from 5 chunks...
[Synthesizer] Synthesising N findings into final answer...
[Synthesizer] Answer ready. N citations.
```

---

## Running the Streamlit App

```bash
cd /mnt/c/Users/alejo/OneDrive/Documents/MSA8700/GenAI_Project/Milestone3 && conda activate gra-venv && streamlit run app_interface.py
```

Opens at `http://localhost:8501`.

---

## Streamlit Interface — Tab Guide

### Overview (Tab 1)
Project description and suggested next steps. No interaction required.

### Paper Search (Tab 2)
- Enter a **research topic** in the sidebar and click **Search Arxiv**
- Results are pulled from **OpenAlex** (not arXiv despite the label)
- Browse the paper table; select a row number to inspect title, authors, year, DOI, and abstract

### Research Framework (Tab 3)
- Populates automatically after a Paper Search
- Displays a rule-based extraction table: Citation | Aims | Data | Analysis | Methods | Findings
- Download the table as CSV with the **Download framework CSV** button
- Note: this tab uses placeholder text extraction, not the LLM pipeline. Use the Chat tab for real extraction.

### Qdrant (Tab 4)
- Shows live connection status to your Qdrant Cloud cluster
- Displays collection list if connected
- Shows an example upsert code stub

### Chat (Tab 5) — Main Interface
1. Type a research question in the chat box at the bottom (e.g., *"How does ESG performance affect firm value?"*)
2. The full LangGraph pipeline runs: **Rephraser → Retriever → Extractor → Synthesizer**
3. The answer appears in the chat window with a formatted citation list below it
4. Chat history persists within the session; refresh the page to clear it

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Pipeline error: unauthorized (status code: 401)` | Agent loaded wrong `OLLAMA_BASE_URL` or missing API key | Ensure no `.env` exists inside `src/`; confirm root `.env` has correct `OLLAMA_BASE_URL` and `0LLAMA` |
| `Qdrant not connected` | Wrong URL or missing API key in `.env` | Check `QDRANT_URL` and `QDRANT_API_KEY` in root `.env` |
| `Qdrant client version incompatible` warning | `qdrant-client` package version behind server | `pip install --upgrade qdrant-client` |
| `torch.classes` path warning in Streamlit | Streamlit file watcher scanning PyTorch internals | Already fixed via `.streamlit/config.toml` (`fileWatcherType = "none"`) |
| All chunks return empty text | Retriever reading wrong payload key from Qdrant | `retriever.py` must use `payload.get("abstract", "")` — the key stored by `embed_upsert.py` |
| fastembed downloads model on every run | Model not cached | Model caches automatically after first download in `~/.cache/huggingface/` — subsequent runs are instant |
