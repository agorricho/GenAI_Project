# M4 Validation Dataset — Team Instructions

**Project:** DAIS — MSA 8700 Final Project  
**Last Updated:** 2026-04-28  
**Who this is for:** Everyone filling in `eval_template.csv`

---

## What you are building

A 50-row evaluation dataset that will be used to automatically score our RAG system. Each row is a question, a correct reference answer, and metadata. The evaluation runner asks the pipeline each question and measures how well its answer matches yours.

**Your job:** Write questions and reference answers drawn from real paper abstracts. The pipeline's job is to find and answer those questions from the indexed corpus.

---

## Files you will work with

| File | Location | Purpose |
|------|----------|---------|
| `validation_chunks.json` | `GenAI_Project/` | 49 paper abstracts — your source material |
| `eval_template.csv` | `GenAI_Project/Milestone3/eval/` | The spreadsheet you fill in |

Open both files side by side. For each paper in `validation_chunks.json`, write 1–2 rows in the CSV.

---

## Step 1 — Open the two files

```
GenAI_Project/validation_chunks.json         ← reference, read-only
GenAI_Project/Milestone3/eval/eval_template.csv  ← fill this in
```

Each entry in `validation_chunks.json` looks like this:

```json
{
  "paper_index": 0,
  "title":    "ESG-FTSE: A corpus of news articles with ESG relevance labels and use cases",
  "authors":  "Mariya Pavlova, Bernard Casey, Miaosen Wang",
  "year":     2024,
  "arxiv_id": "2405.20218v1",
  "arxiv_url": "https://arxiv.org/abs/2405.20218v1",
  "abstract": "We present ESG-FTSE, the first corpus comprised of news articles with ESG relevance annotations..."
}
```

---

## Step 2 — Understand the target distribution (50 rows total)

| `generation_mode` | `difficulty` | `is_in_corpus` | Count | What it means |
|-------------------|-------------|----------------|-------|---------------|
| `abstract`        | `broad`     | `TRUE`         | 25    | Broad causal question answerable from the whole abstract |
| `chunk`           | `specific`  | `TRUE`         | 20    | Specific detail question requiring a particular fact or number |
| `negative`        | `out-of-scope` | `FALSE`     | 5     | Question on a topic NOT in the corpus at all |

**Total: 50 rows.**

Roughly 1 row per paper. You do not need to cover every single paper — focus on the ones with the richest abstracts.

---

## Step 3 — Fill in each column

### Column reference

| Column | Allowed values | Description |
|--------|---------------|-------------|
| `question` | Any text | A causal research question answerable from the abstract |
| `reference_answer` | Any text | 1–3 sentences drawn **only** from the abstract text |
| `expected_source_title` | Exact title string | Copy the `title` field from `validation_chunks.json` exactly |
| `generation_mode` | `abstract` \| `chunk` \| `negative` | See distribution table above |
| `difficulty` | `broad` \| `specific` \| `out-of-scope` | Matches `generation_mode` |
| `esg_category` | `environmental` \| `social` \| `governance` \| `general` \| `off_topic` | Topic of the paper |
| `is_in_corpus` | `TRUE` \| `FALSE` | `TRUE` for all real paper rows; `FALSE` for negative rows only |

### Rules

- `reference_answer` must come **only from the abstract** — no external knowledge, no LLM-generated answers.
- `expected_source_title` must be copied **exactly** from `validation_chunks.json` — do not paraphrase.
- `is_in_corpus` is `FALSE` **only** for the 5 negative rows.
- Each question should target a **cause → effect** relationship: Factor (X) → Effect on (Y).

---

## Step 4 — Examples

### Example A — `abstract` / `broad`

**Abstract excerpt:**
> "We find that higher ESG scores are associated with a significant reduction in cost of capital (β = −0.43, p < 0.01), controlling for firm size and leverage."

| Column | Value |
|--------|-------|
| `question` | Does ESG performance affect cost of capital? |
| `reference_answer` | Higher ESG scores are significantly associated with lower cost of capital, with a coefficient of −0.43 (p < 0.01), controlling for firm size and leverage. |
| `expected_source_title` | *(paste exact title)* |
| `generation_mode` | `abstract` |
| `difficulty` | `broad` |
| `esg_category` | `governance` |
| `is_in_corpus` | `TRUE` |

---

### Example B — `chunk` / `specific`

**Abstract excerpt:**
> "The paper applies the Stock and Yogo (2005) rule of thumb requiring a first-stage F-statistic greater than 10 to confirm that the instrumental variable is not weak."

| Column | Value |
|--------|-------|
| `question` | What F-statistic threshold does the paper use to validate instrument strength? |
| `reference_answer` | The paper applies the Stock and Yogo (2005) rule requiring a first-stage F-statistic greater than 10 to confirm the instrument is not weak. |
| `expected_source_title` | *(paste exact title)* |
| `generation_mode` | `chunk` |
| `difficulty` | `specific` |
| `esg_category` | `governance` |
| `is_in_corpus` | `TRUE` |

---

### Example C — `negative` / `out-of-scope`

| Column | Value |
|--------|-------|
| `question` | How does interest rate policy affect housing prices? |
| `reference_answer` | This question is not related to ESG or firm performance research and is not covered by the corpus. |
| `expected_source_title` | `N/A` |
| `generation_mode` | `negative` |
| `difficulty` | `out-of-scope` |
| `esg_category` | `off_topic` |
| `is_in_corpus` | `FALSE` |

---

## Step 5 — Save and validate the CSV

Once the CSV is filled in, run the validator to catch any typos or missing fields **before** converting.

```bash
source /home/agorricho1/miniconda3/etc/profile.d/conda.sh && conda activate gra_venv
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3

# Validate only — no file written
python eval/csv_to_json.py --validate-only
```

The validator checks:
- All 7 columns are present and non-empty
- `generation_mode`, `difficulty`, and `esg_category` have valid values
- `is_in_corpus` is exactly `TRUE` or `FALSE`
- Warns if row count is not 50

If there are errors, fix them in the CSV and re-run. The script prints which row and which field failed.

---

## Step 6 — Convert CSV to JSON

When validation passes with no errors:

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python eval/csv_to_json.py
```

This writes `Milestone3/eval/eval_dataset.json`. It also prints a summary:

```
  Total items : 50
  abstract    : 25
  chunk       : 20
  negative    : 5
```

Check that the counts match the target distribution.

---

## Step 7 — Run the evaluation

Requires: Qdrant Cloud (`msa8700_m4` collection loaded) and Ollama accessible at ARC or locally.

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python eval/run_eval.py
```

The runner is **safe to interrupt and resume** — it checkpoints after every item. To resume after a crash:

```bash
python eval/run_eval.py           # auto-resumes from checkpoint
```

To force a full re-run from scratch:

```bash
python eval/run_eval.py --no-resume
```

### What the runner outputs

After all 50 items are scored:

```
  Results saved to eval/results/
```

Two files are written:
- `eval/results/eval_results.json` — full details per item
- `eval/results/eval_results.csv` — spreadsheet-friendly summary

The console prints a summary table showing mean scores across 5 metrics:

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Does the pipeline answer stick to the retrieved chunks? |
| Answer Relevancy | Does the answer address the question asked? |
| Semantic Similarity | How close is the pipeline answer to your reference answer? |
| Citation Recall | Did the pipeline cite the expected paper? |
| ROUGE-L | Lexical overlap between pipeline answer and reference |
| **COMPOSITE** | Weighted combination of all 5 |

---

## Step 8 — Verify unit tests pass (optional sanity check)

No live services needed for this step:

```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python -m pytest tests/test_m4_eval.py -v
# Expected: 57/57 pass
```

---

## Common mistakes to avoid

| Mistake | Why it matters |
|---------|---------------|
| Paraphrasing the `expected_source_title` | The evaluator does exact string matching — the title must be copied verbatim |
| Writing `reference_answer` from memory or LLM | Evaluation integrity — answers must come from the abstract only |
| Using `true` / `false` (lowercase) for `is_in_corpus` | `csv_to_json.py` accepts `TRUE`/`FALSE` (case-insensitive), but be consistent |
| Fewer than 5 negative rows | The evaluation needs out-of-corpus examples to test retrieval rejection |
| Empty trailing rows in the CSV | The converter filters blank rows automatically — no action needed |
