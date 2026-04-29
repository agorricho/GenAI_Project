#!/usr/bin/env python3
"""
generate_eval_dataset.py
Fills eval_template.csv with generated eval rows for Milestone 4.

TWO MODES:
  --from-json PATH   Accept pre-generated rows from a JSON file and write to CSV.
                     Claude Code generates this JSON directly in the session.
  --validate-only    Just validate the current eval_template.csv (no writes).

Usage:
    # Validate current CSV
    python generate_eval_dataset.py --validate-only

    # Write from pre-generated JSON
    python generate_eval_dataset.py --from-json generated_rows.json

Generated JSON format (list of row dicts):
[
  {
    "question": "...",
    "reference_answer": "...",
    "expected_source_title": "...",
    "generation_mode": "abstract|chunk|negative",
    "difficulty": "broad|specific|out-of-scope",
    "esg_category": "environmental|social|governance|general|off_topic",
    "is_in_corpus": "TRUE|FALSE"
  },
  ...
]
"""

from __future__ import annotations

import json
import csv
import sys
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
CHUNKS_PATH = PROJECT_DIR / "validation_chunks.json"
TEMPLATE_PATH = PROJECT_DIR / "Milestone3" / "eval" / "eval_template.csv"

FIELDNAMES = [
    "question",
    "reference_answer",
    "expected_source_title",
    "generation_mode",
    "difficulty",
    "esg_category",
    "is_in_corpus",
]

VALID_GENERATION_MODES = {"abstract", "chunk", "negative"}
VALID_DIFFICULTIES = {"broad", "specific", "out-of-scope"}
VALID_ESG_CATEGORIES = {"environmental", "social", "governance", "general", "off_topic"}
VALID_IS_IN_CORPUS = {"TRUE", "FALSE"}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_existing_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(v.strip() for v in row.values()):
                rows.append(dict(row))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_rows(rows: list[dict]) -> list[str]:
    errors = []
    for i, row in enumerate(rows, 1):
        for col in FIELDNAMES:
            if not row.get(col, "").strip():
                errors.append(f"Row {i}: empty field '{col}'")
        gm = row.get("generation_mode", "")
        if gm and gm not in VALID_GENERATION_MODES:
            errors.append(f"Row {i}: invalid generation_mode '{gm}'")
        diff = row.get("difficulty", "")
        if diff and diff not in VALID_DIFFICULTIES:
            errors.append(f"Row {i}: invalid difficulty '{diff}'")
        cat = row.get("esg_category", "")
        if cat and cat not in VALID_ESG_CATEGORIES:
            errors.append(f"Row {i}: invalid esg_category '{cat}'")
        iic = row.get("is_in_corpus", "")
        if iic and iic not in VALID_IS_IN_CORPUS:
            errors.append(f"Row {i}: invalid is_in_corpus '{iic}' (must be TRUE or FALSE)")
    return errors


def print_distribution(rows: list[dict]) -> None:
    n_abstract = sum(1 for r in rows if r.get("generation_mode") == "abstract")
    n_chunk = sum(1 for r in rows if r.get("generation_mode") == "chunk")
    n_negative = sum(1 for r in rows if r.get("generation_mode") == "negative")
    print(f"  Total rows:        {len(rows)}")
    print(f"  abstract/broad:    {n_abstract}  (target: 25)")
    print(f"  chunk/specific:    {n_chunk}  (target: 20)")
    print(f"  negative:          {n_negative}  (target: 5)")


def run_csv_to_json_validator() -> int:
    validator = TEMPLATE_PATH.parent / "csv_to_json.py"
    if not validator.exists():
        print(f"Validator not found at {validator} — skipping.")
        return 0
    result = subprocess.run(
        [sys.executable, str(validator), "--validate-only"],
        cwd=str(TEMPLATE_PATH.parent.parent),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 4 eval CSV builder — no API key required"
    )
    parser.add_argument(
        "--from-json",
        metavar="PATH",
        help="JSON file containing pre-generated row dicts to merge into eval_template.csv",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate current eval_template.csv without writing",
    )
    args = parser.parse_args()

    if args.validate_only:
        rows = read_existing_rows(TEMPLATE_PATH)
        print(f"Validating {TEMPLATE_PATH.name} ({len(rows)} rows)...")
        errors = validate_rows(rows)
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")
            sys.exit(1)
        else:
            print("Internal validation: OK")
            print_distribution(rows)
            print("\nRunning csv_to_json.py validator...")
            rc = run_csv_to_json_validator()
            sys.exit(rc)

    if args.from_json:
        json_path = Path(args.from_json)
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
            sys.exit(1)

        new_rows = json.loads(json_path.read_text())
        print(f"Loaded {len(new_rows)} new rows from {json_path.name}")

        existing = read_existing_rows(TEMPLATE_PATH)
        print(f"Preserving {len(existing)} existing rows from eval_template.csv")

        all_rows = existing + new_rows

        # Pre-write validation
        errors = validate_rows(all_rows)
        if errors:
            print("Validation errors — CSV not written:")
            for e in errors:
                print(f"  {e}")
            sys.exit(1)

        write_csv(TEMPLATE_PATH, all_rows)
        print(f"\nWrote {len(all_rows)} rows to {TEMPLATE_PATH}")
        print_distribution(all_rows)

        print("\nRunning csv_to_json.py validator...")
        rc = run_csv_to_json_validator()
        if rc == 0:
            print("Validation passed.")
        else:
            print("Validation errors found — fix and re-run.")
        sys.exit(rc)

    parser.print_help()


if __name__ == "__main__":
    main()
