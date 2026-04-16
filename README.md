# GenAI_Project
Final Project — MSA 8700: Building Generative AI Business Solutions

**DAIS: Document-Driven Agentic Intelligence System** — a Research Advisor RAG agent that ingests academic papers and answers queries using retrieved context.

---

## Milestone Overview

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Weekly assignments — Ollama, Pydantic, web scraping, Qdrant basics | ✅ Complete |
| M2 | RAG ingestion pipeline — arXiv PDFs → chunks → Qdrant Cloud | ✅ Complete |
| M3 | LangGraph RAG pipeline — Rephraser → Retriever → Extractor → Synthesizer + Streamlit UI | ✅ Complete |
| M4 | Evaluation framework — 75-item Q&A dataset, 5 metrics, resumable eval runner | ✅ Complete |

---

## Milestone 2 — Ingestion Pipeline

Fetches arXiv PDFs, chunks them, embeds with `nomic-embed-text`, and upserts to Qdrant Cloud (`msa8700_m3`).

```bash
conda activate gra_venv
pip install -r requirementsM2.txt
ollama pull nomic-embed-text

# Step 1: fetch PDFs
python Milestone2/download_papers.py

# Step 2: embed + upsert (crash-safe — resumes from progress.json)
python Milestone2/embed_upsert.py
```

---

## Milestone 3 — LangGraph RAG Pipeline

Four-node pipeline wired with LangGraph. Answers ESG research questions from the `msa8700_m3` Qdrant collection.

```
Rephraser → Retriever → Extractor → Synthesizer
```

### Setup

```bash
conda activate gra_venv
cd Milestone3
pip install -r requirements.txt
```

`.env` file required at `Milestone3/.env`:

```env
QDRANT_URL="https://<your-cluster>.gcp.cloud.qdrant.io"
QDRANT_API_KEY="<your-qdrant-api-key>"
OLLAMA_BASE_URL="https://api.ollama.com"
0LLAMA="<your-ollama-cloud-api-key>"    # leading zero is intentional
M3_MODEL=gpt-oss:120b
```

### Run (CLI)

```bash
cd Milestone3
python -c "from src.pipeline import run_query; r = run_query('How does ESG affect firm performance?'); print(r['answer'])"
```

### Run (Streamlit UI)

```bash
cd Milestone3
streamlit run app_interface.py
```

Opens at `http://localhost:8501`. Use **Tab 5 (Chat)** for the full pipeline.

---

## Milestone 4 — Evaluation Framework

Systematic evaluation of the M3 pipeline across 5 metrics using a 75-item Q&A dataset.

### What it measures

| Metric | Type | Weight |
|--------|------|--------|
| Faithfulness | LLM-as-judge | 0.30 |
| Answer Relevancy | LLM-as-judge | 0.25 |
| Semantic Similarity | Embedding cosine (fastembed) | 0.25 |
| Citation Recall | Fuzzy substring match | 0.15 |
| ROUGE-L | Lexical overlap | 0.05 |

### File structure

```
Milestone3/
├── eval/
│   ├── generate_dataset.py   # Step 1: build 75-item Q&A dataset from Qdrant
│   ├── metrics.py            # All 5 scoring functions + OllamaJudge
│   ├── run_eval.py           # Step 2: run pipeline + score + report
│   ├── eval_dataset.json     # Generated artifact (committed after Step 1)
│   └── results/
│       ├── eval_results.json # Full per-item scores + summary
│       ├── eval_results.csv  # Tabular view
│       └── eval_summary.txt  # Console snapshot
└── tests/
    └── test_m4_eval.py       # 57 unit tests (all mocked — no live services needed)
```

### Step 1 — Generate dataset (~10–15 min, requires live Qdrant + Ollama)

Scrolls all points from `msa8700_m3` and generates 75 Q&A items using the LLM:
- 40 abstract-seeded (2 per paper)
- 30 chunk-grounded (1 per randomly-sampled paper, seed=42)
- 5 negative / off-topic (tests graceful fallback)

```bash
cd Milestone3
python eval/generate_dataset.py
```

Output: `eval/eval_dataset.json`

### Step 2 — Run evaluation (~30–90 min, auto-resumes on re-run)

Runs `run_query()` for every dataset item, scores all 5 metrics, checkpoints atomically after each item.

```bash
cd Milestone3

# Auto-resume (safe to re-run after crash)
python eval/run_eval.py

# Force full re-run from scratch
python eval/run_eval.py --no-resume
```

Output:
- `eval/results/eval_results.json` — full per-item scores, summary stats, by-category breakdown
- `eval/results/eval_results.csv` — one row per item
- Console summary table printed on completion

### Run unit tests (no live services needed)

```bash
cd Milestone3
pytest tests/test_m4_eval.py -v
# 57/57 tests pass
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Dataset not found` on `run_eval.py` | Run `generate_dataset.py` first |
| LLM judge returns empty scores | Check `OLLAMA_BASE_URL` and `0LLAMA` in `.env`; note leading zero |
| Eval resumes from wrong position | Delete `eval/results/eval_results.json` and re-run, or use `--no-resume` |
| `fastembed` model downloads on first run | One-time ~130 MB download; cached in `~/.cache/` afterwards |
