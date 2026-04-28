# M4 Implementation Plan — Validation Dataset & Evaluation

**Project:** DAIS — MSA 8700 Final Project
**Last Updated:** 2026-04-28
**Status:** Pipeline running — embed phase in progress

---

## Phase 1 — Collect 50 Papers (Pipeline)

### What ran

| Step | Script | Status |
|------|--------|--------|
| PDF download + chunking | `download_papers.py` | ✅ Complete — 49 PDFs in `papers/` |
| Embedding + Qdrant upsert | `embed_upsert.py` | 🔄 In progress — 41/~49 papers embedded |

### Collection details

| Item | Value |
|------|-------|
| Qdrant collection | `msa8700_m4` |
| Embedding model | `nomic-embed-text` (768-dim, ARC GPU) |
| Vector distance | Cosine |
| Checkpoint file | `progress.json` → `last_completed_paper_index` |

### Monitor progress

```bash
tail -f /home/agorricho1/AI_Scientist2/GenAI_Project/pipeline_m4.log
cat /home/agorricho1/AI_Scientist2/GenAI_Project/progress.json
```

### If embedding crashes mid-run (safe to resume)

```bash
source /home/agorricho1/miniconda3/etc/profile.d/conda.sh && conda activate gra_venv
cd /home/agorricho1/AI_Scientist2/GenAI_Project
nohup python -u run_pipeline.py > pipeline_m4.log 2>&1 &
```

`embed_upsert.py` reads `progress.json` and skips already-completed papers automatically.

---

## Phase 2 — Deduplicate chunks.json → validation_chunks.json

### What the script does

`build_validation_chunks.py` (standalone, discard after use):

1. Reads `chunks.json` (all chunks from all papers, ~1300–3000 entries)
2. Groups chunks by `arxiv_id`
3. Keeps **one entry per unique paper** (first chunk seen for that `arxiv_id`)
4. Outputs `validation_chunks.json` with these fields per paper:

```json
{
  "paper_index": 0,
  "title":       "Full paper title",
  "authors":     "Author A, Author B",
  "year":        2023,
  "arxiv_id":    "2301.12345",
  "arxiv_url":   "https://arxiv.org/abs/2301.12345",
  "abstract":    "Full abstract text..."
}
```

### Run it (after pipeline completes)

```bash
source /home/agorricho1/miniconda3/etc/profile.d/conda.sh && conda activate gra_venv
cd /home/agorricho1/AI_Scientist2/GenAI_Project
python build_validation_chunks.py
```

Output: `GenAI_Project/validation_chunks.json` — one entry per paper, sorted by `paper_index`.

---

## Phase 3 — Build the M4 Validation CSV (Team Task)

### What your team does

Open `validation_chunks.json` alongside `Milestone3/eval/eval_template.csv`.

For each paper in `validation_chunks.json`, copy the abstract and write **2–3 rows** in `eval_template.csv`:

| Column | How to fill |
|--------|-------------|
| `question` | Write a causal research question answerable from the abstract (e.g., "What is the effect of ESG scores on firm financial performance?") |
| `reference_answer` | 1–3 sentence answer drawn directly from the abstract — no external knowledge |
| `expected_source_title` | Copy the `title` field from `validation_chunks.json` exactly |
| `generation_mode` | `abstract` (for broad questions) or `chunk` (for specific detail questions) |
| `difficulty` | `broad` for abstract-level questions, `specific` for detail questions |
| `esg_category` | `environmental` / `social` / `governance` / `general` / `off_topic` |
| `is_in_corpus` | `TRUE` for all real paper rows |

**Target distribution (50 rows total):**

| `generation_mode` | Count | `difficulty` | `is_in_corpus` |
|-------------------|-------|-------------|----------------|
| `abstract` | 25 | `broad` | TRUE |
| `chunk` | 20 | `specific` | TRUE |
| `negative` | 5 | `out-of-scope` | FALSE |

For the **5 negative rows**: write questions on topics NOT in the corpus (e.g., "How does interest rate policy affect housing prices?"). Set `expected_source_title = N/A`, `is_in_corpus = FALSE`, `esg_category = off_topic`.

### Causal relationship focus

Each question should target a **cause → effect** relationship from the paper:
- Factor (X): ESG score, board diversity, carbon disclosure, etc.
- Effect (Y): firm value, stock returns, ROA, cost of capital, etc.

Example from a strong abstract:
> "We find that higher ESG scores are associated with a significant reduction in cost of capital (β = −0.43, p < 0.01), controlling for firm size and leverage."

→ `question`: "Does ESG performance affect cost of capital?"
→ `reference_answer`: "Higher ESG scores are significantly associated with lower cost of capital, with a coefficient of −0.43 (p < 0.01), controlling for firm size and leverage."

---

## Phase 4 — Convert CSV → JSON and Run Evaluation

### Step 4a: Convert

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3

# Validate first (no file written)
python eval/csv_to_json.py --validate-only

# Write eval_dataset.json
python eval/csv_to_json.py
```

### Step 4b: Run evaluation

Requires live Qdrant Cloud (`msa8700_m4`) + Ollama (ARC GPU or Ollama Cloud).

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python eval/run_eval.py
```

Safe to kill and resume — checkpoints atomically after each item.

### Step 4c: Verify unit tests pass (no live services needed)

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python -m pytest tests/test_m4_eval.py -v
# Expected: 57/57 pass
```

---

## Guardrails (do not violate)

- **Do not modify** any existing script inside `GenAI_Project/` beyond the authorized changes already made to `download_papers.py` and `embed_upsert.py`
- `build_validation_chunks.py` is a **standalone one-off tool** — discard after `validation_chunks.json` is confirmed correct
- The M4 `eval_template.csv` format is fixed — do not add or remove columns
- `reference_answer` must come from the abstract text only — no LLM-generated answers, no external sources
- `expected_source_title` must match the Qdrant payload title exactly (copy from `validation_chunks.json`)

---

## File Map

| File | Role | Delete after? |
|------|------|--------------|
| `chunks.json` | All paper chunks (pipeline output) | No — used by embed_upsert.py |
| `validation_chunks.json` | 50 unique abstracts for team reference | After eval_template.csv is filled |
| `build_validation_chunks.py` | Dedup script | After validation_chunks.json confirmed |
| `pipeline_m4.log` | Pipeline run log | After pipeline complete |
| `progress.json` | Embedding checkpoint | After msa8700_m4 fully loaded |
| `Milestone3/eval/eval_template.csv` | Team fills this in | No — M4 submission artifact |
| `Milestone3/eval/eval_dataset.json` | csv_to_json.py output | No — M4 submission artifact |
