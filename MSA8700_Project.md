# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSA 8700 — *Building Generative AI Business Solutions* coursework and final project **DAIS (Document-Driven Agentic Intelligence System), Variation B — Research Advisor**. The system ingests ESG academic papers and answers research questions with cited, evidence-backed responses using a multi-agent RAG pipeline.

---

## Active Directories

Three parallel codebases exist — always confirm which the user is working in:

| Directory | Purpose |
|-----------|---------|
| `GenAI_Project/` | M2 original pipeline root + **active M3/M4 development** (source of truth) |
| `Milestone2_Corrected/` | M2 corrected submission (improved chunking, upgraded embedding model) |
| `Milestone2_Windows/` | Archive copy — do not develop here |

**M3/M4 source of truth:** `GenAI_Project/Milestone3/`

---

## Environment Setup

```bash
conda activate gra_venv          # Python 3.12
cd GenAI_Project/Milestone3
pip install -r requirements.txt
pip install fastembed            # ONNX embedder — downloads model on first run (~500 MB)
pip install --upgrade qdrant-client
```

`.env` must be placed at repo root (`MSA8700/`) or any ancestor of the script. All agent files and `embed_upsert.py` walk up the directory tree (`Path(__file__).resolve().parents`) to find the nearest `.env`. **Never place a `.env` inside `src/`** — it intercepts the walk before credentials are found.

Required `.env` keys:
```
QDRANT_URL=https://<cluster>.gcp.cloud.qdrant.io
QDRANT_API_KEY=<jwt>
OLLAMA_BASE_URL=https://api.ollama.com   # used by agents; embed_upsert.py hardcodes localhost
0LLAMA=<ollama-cloud-api-key>            # zero, not letter O
M3_MODEL=llama3.2
```

> **Note:** `embed_upsert.py` hardcodes `OLLAMA_BASE_URL = "http://localhost:11434/api/embeddings"` and does NOT read it from `.env`. The M3 agents do read `OLLAMA_BASE_URL` from `.env`.

---

## Running Scripts

### M2 — Ingest pipeline

```bash
# GenAI_Project root (nomic-embed-text via Ollama, 768-dim, collection msa8700_m2)
cd GenAI_Project
python run_pipeline.py           # runs download_papers.py → 120s pause → embed_upsert.py

# Milestone2_Corrected (mxbai-embed-large, 1024-dim, collection msa8700_m3)
cd Milestone2_Corrected
python run_pipeline.py
```

Both pipelines are crash-safe: `progress.json` checkpoints the last completed paper index. Delete `progress.json` to force a full fresh run.

### M3 — Query pipeline (CLI)

```bash
cd GenAI_Project/Milestone3
python -c "from src.pipeline import run_query; r = run_query('How does ESG affect firm performance?'); print(r['answer'])"
```

### M3 — Streamlit app

```bash
cd GenAI_Project/Milestone3
streamlit run app_interface.py   # opens at http://localhost:8501
```

### M4 — Evaluation

```bash
cd GenAI_Project/Milestone3
python eval/run_eval.py              # resumes from checkpoint automatically
python eval/run_eval.py --no-resume  # force full re-run
python eval/run_eval.py --dataset path/to/eval_dataset.json --output-dir eval/results
```

Results written to `eval/results/eval_results.json` and `eval_results.csv`.

### Tests

```bash
cd GenAI_Project/Milestone3
python -m pytest tests/test_m4_eval.py -v        # full test suite
python -m pytest tests/test_m4_eval.py::test_name -v  # single test
```

---

## Architecture

### M2 Ingestion Pipeline

```
arXiv API → PDFs (pdfplumber)
         → RecursiveCharacterTextSplitter (1200 chars / 200 overlap)   [Milestone2_Corrected]
           OR fixed 10 equal segments                                    [GenAI_Project]
         → chunks.json
         → Ollama embed → Qdrant Cloud upsert
```

- **GenAI_Project** stores to `msa8700_m2` (nomic-embed-text via Ollama, 768-dim).
- **Milestone2_Corrected** stores to `msa8700_m3` (mxbai-embed-large, 1024-dim).

### M3 LangGraph Pipeline (`GenAI_Project/Milestone3/src/`)

```
run_query(question)
    │
    ▼  src/agents/rephraser.py   — ChatOllama (temp=0): question → compact academic search terms
    ▼  src/agents/retriever.py   — fastembed (nomic-ai/nomic-embed-text-v1.5, ONNX) → Qdrant msa8700_m3 top-5
    ▼  src/agents/extractor.py   — per-chunk relevance filter + finding extraction (ChatOllama, temp=0)
    ▼  src/agents/synthesizer.py — combines findings into cited academic answer (ChatOllama, temp=0)
    │
    └─ returns {answer: str, citations: list[{title,authors,year}], chunks: list[dict]}
```

- `src/state.py` — `ResearchState` TypedDict shared across all 4 nodes
- `src/pipeline.py` — `build_graph()` compiles the LangGraph `StateGraph`; `run_query()` is the public entry point; module-level `app = build_graph()` is imported by the Streamlit app

**Retriever quirk:** `retriever_node` populates chunk `text` from `payload.get("abstract", "")`, not the raw chunk text field. The extractor and synthesizer therefore reason over abstracts, not chunked body text.

Each agent file does its own `.env` walk at **module load time** (not inside the node function). LLM clients (`_llm`, `_qdrant`, `_embedder`) are also instantiated at module load time as module-level singletons.

### Streamlit App (`app_interface.py`)

Five tabs:
1. **Overview** — project description
2. **Paper Search** — OpenAlex API search (sidebar: topic + count controls)
3. **Research Framework** — rule-based placeholder extraction table (CSV download)
4. **Qdrant** — live connection status + collection list
5. **Chat** — wires to `run_query()`; conversation history persists in `st.session_state` as `HumanMessage`/`AIMessage` objects

### M4 Evaluation (`eval/`)

- `eval_dataset.json` — test items: `item_id`, `question`, `reference_answer`, `expected_source_title`, `source_chunk_index`, `generation_mode` (`abstract`/`chunk`/`negative`), `difficulty` (`broad`/`specific`/`out-of-scope`), `esg_category`, `is_in_corpus`
- `metrics.py` — five scoring functions: Faithfulness (0.30), Answer Relevancy (0.25), Semantic Similarity (0.25), Citation Recall (0.15), ROUGE-L (0.05); `compute_composite_score()` applies weights
- `run_eval.py` — atomic checkpoint per item (`eval/results/eval_results.json`); batch semantic similarity at end; generates JSON + CSV report; `sys.path` injection lets `from src.pipeline import run_query` work without installing as a package
- `generate_dataset.py` — one-time script to produce `eval_dataset.json` (75 items: 40 abstract-seeded, 30 chunk-grounded, 5 negative)
- `tests/test_m4_eval.py` — pytest suite

---

## Critical Embedding Compatibility Note

The `GenAI_Project/Milestone3` retriever uses **fastembed `nomic-ai/nomic-embed-text-v1.5`** (768-dim) and queries collection `msa8700_m3`. However, `Milestone2_Corrected/embed_upsert.py` populates `msa8700_m3` with **mxbai-embed-large** (1024-dim). These dimensions are incompatible — Qdrant will reject queries if there is a mismatch.

**Rule:** whichever model built the collection must also be used at query time. If `msa8700_m3` was created with mxbai-embed-large, the retriever must also embed with mxbai, or the collection must be rebuilt with nomic-embed-text.

---

## Key Design Patterns

- **`.env` discovery:** `embed_upsert.py` and every agent file in `src/agents/` walks `Path(__file__).resolve().parents` to find the nearest `.env`. Scripts can be run from any working directory as long as a `.env` exists somewhere in the ancestor path.
- **Checkpoint/resume:** M2 `embed_upsert.py` writes `progress.json` after each paper. M4 `run_eval.py` does an atomic write (write to `.tmp` then rename) after each eval item. Both resume automatically on re-run.
- **Payload schema** (Qdrant, per vector): `title`, `authors`, `year`, `arxiv_id`, `abstract`, `pdf_path`, `chunk_index`, `chunk_total`, `text`
- **`0LLAMA` env key:** the Ollama API key is stored under the key name `0LLAMA` (zero, not capital O) in all `.env` files and agent code.
- **OllamaJudge (M4):** wraps `/api/generate` at temperature=0; strips markdown fences from responses; retries up to 3× on JSON parse failure; used for Faithfulness and Answer Relevancy scoring.

---

## Upcoming Milestones

| M | Focus | Points |
|---|-------|--------|
| M4 | Evaluation framework — 75 test items | 16 |
| M5 | Iterative improvements + ablation study | 16 |
| Final | Deployed system + report + video demo | 12 |
