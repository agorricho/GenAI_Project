#!/usr/bin/env python3
"""
build_validation_chunks.py — standalone, one-off utility. Discard after use.

Reads chunks.json and writes validation_chunks.json with one entry per
unique paper (deduped by arxiv_id), sorted by paper_index.

Output fields per entry:
  paper_index, title, authors, year, arxiv_id, arxiv_url, abstract

Run from GenAI_Project/:
    conda activate gra_venv
    python build_validation_chunks.py
"""
import json
from pathlib import Path

CHUNKS_FILE = Path(__file__).parent / "chunks.json"
OUTPUT_FILE = Path(__file__).parent / "validation_chunks.json"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"chunks.json not found at {CHUNKS_FILE}")

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = json.load(f)

    seen: dict[str, dict] = {}
    for chunk in chunks:
        arxiv_id = (chunk.get("arxiv_id") or "").strip()
        if not arxiv_id or arxiv_id in seen:
            continue
        seen[arxiv_id] = {
            "paper_index": chunk.get("paper_index"),
            "title": (chunk.get("title") or "").strip(),
            "authors": (chunk.get("authors") or "").strip(),
            "year": chunk.get("year"),
            "arxiv_id": arxiv_id,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "abstract": (chunk.get("abstract") or "").strip(),
        }

    papers = sorted(seen.values(), key=lambda p: p.get("paper_index") or 0)

    required = {"paper_index", "title", "authors", "year", "arxiv_id", "arxiv_url", "abstract"}
    for p in papers:
        missing = [k for k in required if p.get(k) is None]
        if missing:
            raise ValueError(f"Paper {p.get('arxiv_id')} missing fields: {missing}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\nDone — {len(papers)} unique papers written to {OUTPUT_FILE}\n")
    for i, p in enumerate(papers, 1):
        title_short = p["title"][:68] + ".." if len(p["title"]) > 70 else p["title"]
        print(f"  [{i:02d}] ({p['year']}) {title_short}")
        print(f"       {p['arxiv_url']}")
    print()


if __name__ == "__main__":
    main()
