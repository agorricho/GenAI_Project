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

Systematic evaluation of the M3 pipeline across 5 metrics using a 50-item human-authored Q&A dataset.

### What it measures

| Metric | Type | Weight |
|--------|------|--------|
| Faithfulness | LLM-as-judge | 0.30 |
| Answer Relevancy | LLM-as-judge | 0.25 |
| Semantic Similarity | Embedding cosine (fastembed) | 0.25 |
| Citation Recall | Fuzzy substring match | 0.15 |
| ROUGE-L | Lexical overlap | 0.05 |

### How it works — two-step flow

```
Step 1 — Build the dataset (human-authored, one-time):
  eval_template.csv  →  [csv_to_json.py]  →  eval_dataset.json
  (fill in 50 rows)                           (test questions + reference answers)

Step 2 — Run the evaluation (requires live Qdrant + Ollama):
  eval_dataset.json  →  [run_eval.py]  →  eval/results/eval_results.json
                                          eval/results/eval_results.csv
                                          (console summary table)
```

`run_eval.py` runs every question through the live DAIS pipeline (Rephraser → Retriever → Extractor → Synthesizer), scores all 5 metrics per item, and writes human-readable results. It checkpoints atomically after each item so a crash never loses progress.

### File structure

```
Milestone3/
├── eval/
│   ├── eval_template.csv     # Step 1a: fill this — 50-row annotator sheet
│   ├── csv_to_json.py        # Step 1b: convert filled CSV → eval_dataset.json
│   ├── eval_dataset.json     # Step 1 output; consumed by run_eval.py
│   ├── generate_dataset.py   # Alternative: LLM-generated 75-item dataset (requires live services)
│   ├── metrics.py            # All 5 scoring functions + OllamaJudge
│   ├── run_eval.py           # Step 2: run pipeline + score + report
│   └── results/              # Created by run_eval.py
│       ├── eval_results.json # Full per-item scores + summary stats + by-category breakdown
│       └── eval_results.csv  # One row per item — open in Excel or Sheets
└── tests/
    └── test_m4_eval.py       # 57 unit tests (all mocked — no live services needed)
```

### Step 1a — Fill the validation set CSV

Open `eval/eval_template.csv` in Excel, Google Sheets, or any text editor. Fill 50 rows — one research question per row — drawing answers directly from paper abstracts in the `msa8700_m3` Qdrant collection.

**Target distribution:**

| `generation_mode` | Count | `difficulty` | `is_in_corpus` |
|-------------------|-------|-------------|----------------|
| `abstract` | 25 | `broad` | `TRUE` |
| `chunk` | 20 | `specific` | `TRUE` |
| `negative` | 5 | `out-of-scope` | `FALSE` |

**Column reference:**

| Column | Allowed values | Notes |
|--------|---------------|-------|
| `question` | free text | Natural-language research question answerable from the corpus |
| `reference_answer` | free text (1–3 sentences) | Ground-truth answer written from the abstract — no external knowledge |
| `expected_source_title` | exact paper title, or `N/A` | Copy verbatim from Qdrant; use `N/A` for negative items |
| `generation_mode` | `abstract` / `chunk` / `negative` | See distribution table above |
| `difficulty` | `broad` / `specific` / `out-of-scope` | Matches generation_mode 1:1 |
| `esg_category` | `environmental` / `social` / `governance` / `general` / `off_topic` | Topic of the question |
| `is_in_corpus` | `TRUE` / `FALSE` | `FALSE` for negative items only |

To browse papers in the corpus, scroll the Qdrant collection from the terminal:

```python
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv("Milestone3/.env")
c = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
results, _ = c.scroll("msa8700_m3", limit=100, with_payload=True)
for r in results:
    print(r.payload.get("title"), "—", r.payload.get("abstract", "")[:120])
```

### Step 1b — Convert CSV → eval_dataset.json

```bash
cd Milestone3

# Validate only (no file written — use to check for errors before committing)
python eval/csv_to_json.py --validate-only

# Convert and write eval_dataset.json
python eval/csv_to_json.py
```

The converter validates every row (blank fields, invalid enum values, casing issues) and exits with a specific error message if anything is wrong. Casing is normalised automatically — `Abstract` and `abstract` both work.

### Step 2 — Run evaluation (~30–90 min, auto-resumes on re-run)

Requires live Qdrant Cloud + Ollama. Runs `run_query()` for every item in `eval_dataset.json`, scores all 5 metrics, checkpoints atomically after each item.

```bash
cd Milestone3

# Auto-resume (safe to re-run after crash)
python eval/run_eval.py

# Force full re-run from scratch
python eval/run_eval.py --no-resume
```

**Output:**
- `eval/results/eval_results.json` — full per-item scores, summary stats (mean ± std), by-category breakdown
- `eval/results/eval_results.csv` — one row per item; open in Excel or Sheets to filter and sort by metric
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
| `CSV not found` on `csv_to_json.py` | Run from `Milestone3/`; default input is `eval/eval_template.csv` |
| `Validation N error(s) found` | Converter prints row number + column name — fix those cells and re-run |
| `Dataset not found` on `run_eval.py` | Run `csv_to_json.py` first to produce `eval_dataset.json` |
| LLM judge returns empty scores | Check `OLLAMA_BASE_URL` and `0LLAMA` in `.env`; note leading zero |
| Eval resumes from wrong position | Delete `eval/results/eval_results.json` and re-run, or use `--no-resume` |
| `fastembed` model downloads on first run | One-time ~130 MB download; cached in `~/.cache/` afterwards |
