"""
csv_to_json.py — Convert a human-authored CSV into eval_dataset.json

Usage (from Milestone3/):
    python eval/csv_to_json.py                     # validate + write eval_dataset.json
    python eval/csv_to_json.py --validate-only     # validate only, no file written
    python eval/csv_to_json.py --input path/to/file.csv
    python eval/csv_to_json.py --output path/to/output.json

Expected CSV columns (in any order):
    question, reference_answer, expected_source_title,
    generation_mode, difficulty, esg_category, is_in_corpus

item_id and source_chunk_index are auto-assigned — do not include in the CSV.

Target: 50 rows (25 abstract, 20 chunk, 5 negative)
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_COLUMNS = {
    "question",
    "reference_answer",
    "expected_source_title",
    "generation_mode",
    "difficulty",
    "esg_category",
    "is_in_corpus",
}

ALLOWED_GENERATION_MODES = {"abstract", "chunk", "negative"}
ALLOWED_DIFFICULTIES      = {"broad", "specific", "out-of-scope"}
ALLOWED_ESG_CATEGORIES    = {"environmental", "social", "governance", "general", "off_topic"}

TARGET_TOTAL = 50


def _parse_bool(value: str) -> bool:
    """Accept TRUE/FALSE (case-insensitive). Raise ValueError on anything else."""
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValueError(f"Expected TRUE or FALSE, got: {value!r}")


def load_and_validate(csv_path: Path) -> list[dict]:
    """
    Read CSV, validate schema and field values, return clean item list.
    Raises SystemExit on any validation error.
    """
    if not csv_path.exists():
        print(f"[Error] CSV not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("[Error] CSV appears to be empty.")
            sys.exit(1)

        # Strip whitespace from header names to avoid KeyError on row access
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        actual_cols = set(reader.fieldnames)
        missing = REQUIRED_COLUMNS - actual_cols
        if missing:
            print(f"[Error] Missing required columns: {sorted(missing)}")
            sys.exit(1)

        rows = list(reader)

    # Filter out completely empty rows (template spacer rows)
    rows = [r for r in rows if any(v.strip() for v in r.values())]

    errors: list[str] = []
    items: list[dict] = []

    for i, row in enumerate(rows, start=2):  # row 2 = first data row (row 1 = header)
        question       = row["question"].strip()
        reference      = row["reference_answer"].strip()
        source_title   = row["expected_source_title"].strip()
        # Normalise to lowercase so annotators can use any casing
        gen_mode       = row["generation_mode"].strip().lower()
        difficulty     = row["difficulty"].strip().lower()
        esg_category   = row["esg_category"].strip().lower()
        is_in_corpus_s = row["is_in_corpus"].strip()

        row_errors: list[str] = []

        # Required non-empty checks
        for field, value in [
            ("question", question),
            ("reference_answer", reference),
            ("expected_source_title", source_title),
            ("generation_mode", gen_mode),
            ("difficulty", difficulty),
            ("esg_category", esg_category),
            ("is_in_corpus", is_in_corpus_s),
        ]:
            if not value:
                row_errors.append(f"Row {i}: '{field}' is blank")

        # Allowed-value checks
        if gen_mode and gen_mode not in ALLOWED_GENERATION_MODES:
            row_errors.append(
                f"Row {i}: invalid generation_mode={gen_mode!r}. "
                f"Allowed: {sorted(ALLOWED_GENERATION_MODES)}"
            )

        if difficulty and difficulty not in ALLOWED_DIFFICULTIES:
            row_errors.append(
                f"Row {i}: invalid difficulty={difficulty!r}. "
                f"Allowed: {sorted(ALLOWED_DIFFICULTIES)}"
            )

        if esg_category and esg_category not in ALLOWED_ESG_CATEGORIES:
            row_errors.append(
                f"Row {i}: invalid esg_category={esg_category!r}. "
                f"Allowed: {sorted(ALLOWED_ESG_CATEGORIES)}"
            )

        try:
            is_in_corpus = _parse_bool(is_in_corpus_s) if is_in_corpus_s else None
        except ValueError as e:
            row_errors.append(f"Row {i}: is_in_corpus — {e}")
            is_in_corpus = None

        errors.extend(row_errors)

        if not row_errors:
            items.append({
                "question":               question,
                "reference_answer":       reference,
                "expected_source_title":  source_title,
                "generation_mode":        gen_mode,
                "difficulty":             difficulty,
                "esg_category":           esg_category,
                "is_in_corpus":           is_in_corpus,
            })

    if errors:
        print(f"[Validation] {len(errors)} error(s) found:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)

    return items


def assign_ids(items: list[dict]) -> list[dict]:
    """Auto-assign item_id and source_chunk_index to each item."""
    for idx, item in enumerate(items, start=1):
        item["item_id"]            = f"item_{idx:03d}"
        item["source_chunk_index"] = 0
    return items


def print_summary(items: list[dict]) -> None:
    modes = Counter(i["generation_mode"] for i in items)
    cats  = Counter(i["esg_category"] for i in items)

    print(f"\n  Total items : {len(items)}")
    print(f"  abstract    : {modes.get('abstract', 0)}")
    print(f"  chunk       : {modes.get('chunk', 0)}")
    print(f"  negative    : {modes.get('negative', 0)}")
    print(f"\n  by category : {dict(cats)}")

    if len(items) != TARGET_TOTAL:
        print(f"\n  [Warning] Expected {TARGET_TOTAL} items, got {len(items)}.")
        print(f"           Fill the remaining rows in the CSV template.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert human-authored CSV → eval_dataset.json for M4 evaluation"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).parent / "eval_template.csv",
        help="Path to the filled CSV (default: eval/eval_template.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent / "eval_dataset.json",
        help="Output path for eval_dataset.json (default: eval/eval_dataset.json)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the CSV without writing the JSON output",
    )
    args = parser.parse_args()

    print(f"[csv_to_json] Reading: {args.input}")
    items = load_and_validate(args.input)
    items = assign_ids(items)

    if args.validate_only:
        print(f"[csv_to_json] Validation OK — {len(items)} rows")
        print_summary(items)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    print(f"[csv_to_json] Wrote {len(items)} items → {args.output}")
    print_summary(items)


if __name__ == "__main__":
    main()
