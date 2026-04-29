# Milestone 4 — Eval Dataset Generator: Implementation Plan

**Script:** `GenAI_Project/generate_eval_dataset.py`
**Status:** Ready — no API key required. Claude Code generates content directly in-session.

---

## What Was Built

`generate_eval_dataset.py` — a CSV builder and validator with **no external API dependency**. The script handles file I/O, merging, and validation. Content generation (questions, reference answers, esg categories) is done directly by Claude Code reading the abstracts in-session and writing a `generated_rows.json` file, which the script merges into the CSV.

---

## How It Works (No API Key Required)

Claude Code acts as the LLM judge directly in the terminal session:

1. Claude Code reads all 49 abstracts from `validation_chunks.json`
2. Generates 47 rows of content inline (questions, reference answers, esg categories)
3. Writes the generated rows to `generated_rows.json`
4. Script merges with the 3 existing rows and writes the complete `eval_template.csv`
5. Script auto-runs `csv_to_json.py --validate-only` to confirm correctness

---

## How to Run (Next Session)

```bash
conda activate gra_venv
cd /home/agorricho1/AI_Scientist2/GenAI_Project
```

**Step 1 — Validate current state (3 existing rows):**
```bash
python generate_eval_dataset.py --validate-only
```

**Step 2 — Claude Code generates `generated_rows.json` directly in-session** (47 rows covering 24 abstract/broad + 19 chunk/specific + 4 negative)

**Step 3 — Merge and write CSV:**
```bash
python generate_eval_dataset.py --from-json generated_rows.json
```

**Step 4 — Convert CSV to JSON:**
```bash
cd /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3
python eval/csv_to_json.py
```

**Step 5 — Run the full evaluation:**
```bash
python eval/run_eval.py
```

---

## Script Flags

| Flag | Description |
|------|-------------|
| `--validate-only` | Validate current `eval_template.csv` without writing |
| `--from-json PATH` | Merge pre-generated JSON rows into `eval_template.csv` |

---

## Target Distribution (50 rows total)

| `generation_mode` | `difficulty` | `is_in_corpus` | Target | Already in CSV | To add |
|---|---|---|---|---|---|
| `abstract` | `broad` | `TRUE` | 25 | 1 | 24 |
| `chunk` | `specific` | `TRUE` | 20 | 1 | 19 |
| `negative` | `out-of-scope` | `FALSE` | 5 | 1 | 4 |

**Paper index assignments:**
- Papers `0–23` → `abstract/broad` rows (24 papers)
- Papers `24–42` → `chunk/specific` rows (19 papers)
- Papers `43–48` → not used for paper rows
- 4 invented off-topic questions → `negative` rows

---

## Column Responsibilities

| Column | How filled |
|--------|-----------|
| `question` | Claude Code generates from abstract |
| `reference_answer` | Claude Code — near-verbatim from abstract only, no external knowledge |
| `expected_source_title` | Copied exactly from `paper['title']` — never paraphrased |
| `generation_mode` | Hard-assigned by paper index range |
| `difficulty` | Derived from `generation_mode` — no LLM |
| `esg_category` | Claude Code classification: `environmental\|social\|governance\|general` |
| `is_in_corpus` | Hard-coded `TRUE`/`FALSE` by row type |

---

## generated_rows.json Format

The JSON file Claude Code produces must be a list of dicts:

```json
[
  {
    "question": "...",
    "reference_answer": "...",
    "expected_source_title": "exact title from validation_chunks.json",
    "generation_mode": "abstract",
    "difficulty": "broad",
    "esg_category": "governance",
    "is_in_corpus": "TRUE"
  },
  ...
]
```

Allowed values:
- `generation_mode`: `abstract` | `chunk` | `negative`
- `difficulty`: `broad` | `specific` | `out-of-scope`
- `esg_category`: `environmental` | `social` | `governance` | `general` | `off_topic`
- `is_in_corpus`: `TRUE` | `FALSE`

---

## Key Design Decisions

1. **No API key** — Claude Code generates content directly; script is a pure formatter/validator.
2. **Fixed paper index boundaries** — abstract always `0–23`, chunk always `24–42`; re-running is safe and idempotent.
3. **`expected_source_title` never touches LLM** — always copied verbatim from JSON; evaluator uses exact string matching.
4. **`reference_answer` from abstract only** — enforced by instruction to Claude Code at generation time.
5. **Pre-write validation** — script validates all rows before writing; exits with error code if any field is invalid.
6. **Auto-validator** — script runs `csv_to_json.py --validate-only` automatically after writing.

---

## Validation Rules (enforced by script)

- All 7 columns must be non-empty
- `generation_mode` must be one of: `abstract`, `chunk`, `negative`
- `difficulty` must be one of: `broad`, `specific`, `out-of-scope`
- `esg_category` must be one of: `environmental`, `social`, `governance`, `general`, `off_topic`
- `is_in_corpus` must be exactly `TRUE` or `FALSE`
- `off_topic` esg_category is only valid for `negative` rows
- Target: 50 rows total (warning if count differs)
