"""
download_papers.py — Milestone 2, Step 1 of 2
MSA 8700 Final Project (DAIS, Variation B — Research Advisor)

What this script does:
  1. Search arXiv for papers on TOPIC (skips download if PDF already exists)
  2. Extract full raw text from each PDF with pdfplumber
  3. Chunk each paper into CHUNKS_PER_PAPER equal text segments
  4. Save all chunks to ./chunks.json

Output: chunks.json — 200 entries (20 papers × 10 chunks each)
No Ollama or Qdrant calls — safe to re-run at any time.
"""

import os
import json
import arxiv
import pdfplumber

# ── Configuration ─────────────────────────────────────────────────────────────
TOPIC            = "ESG Performance impact on Firm Value"
MAX_PAPERS       = 20
CHUNKS_PER_PAPER = 10
TEXT_LIMIT       = 3000   # chars per chunk sent to embed (reduced from 8000)
PAPERS_DIR       = "./papers"
CHUNKS_FILE      = "./chunks.json"

os.makedirs(PAPERS_DIR, exist_ok=True)


# ── Step 1: Fetch papers from arXiv (idempotent — skip if PDF exists) ─────────
print(f"Searching arXiv for: '{TOPIC}' (max {MAX_PAPERS} papers)...")

search = arxiv.Search(
    query=TOPIC,
    max_results=MAX_PAPERS,
    sort_by=arxiv.SortCriterion.Relevance
)

papers = []
for result in search.results():
    safe_id  = result.get_short_id().replace("/", "_")
    pdf_path = os.path.join(PAPERS_DIR, f"{safe_id}.pdf")

    if os.path.exists(pdf_path):
        print(f"  [skip] Already downloaded: {result.title[:70]}")
    else:
        print(f"  [download] {result.title[:70]}")
        result.download_pdf(filename=pdf_path)

    papers.append({
        "title":    result.title,
        "authors":  ", ".join(a.name for a in result.authors),
        "year":     result.published.year,
        "pdf_path": pdf_path,
    })

print(f"\nReady: {len(papers)} papers in {PAPERS_DIR}/")


# ── Step 2: PDF text extraction ───────────────────────────────────────────────
def extract_text(pdf_path):
    """Extract full raw text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


# ── Step 3: Chunk text into n equal segments ──────────────────────────────────
def chunk_text(text, n=CHUNKS_PER_PAPER):
    """Split text into n roughly equal character segments."""
    if not text:
        return [""] * n
    size = max(len(text) // n, 1)
    chunks = [text[i:i+size] for i in range(0, len(text), size)]
    if len(chunks) < n:
        chunks += [""] * (n - len(chunks))
    return chunks[:n]


# ── Step 4: Build chunk list and save to chunks.json ─────────────────────────
all_chunks = []
for paper_index, paper in enumerate(papers):
    print(f"[{paper_index + 1}/{len(papers)}] Extracting: {paper['title'][:60]}")
    text   = extract_text(paper["pdf_path"])
    chunks = chunk_text(text)

    for chunk_index, chunk_text_val in enumerate(chunks):
        all_chunks.append({
            "paper_index": paper_index,
            "chunk_index": chunk_index,
            "title":       paper["title"],
            "authors":     paper["authors"],
            "year":        paper["year"],
            "pdf_path":    paper["pdf_path"],
            "text":        chunk_text_val[:TEXT_LIMIT],
        })

with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(all_chunks)} chunks saved to {CHUNKS_FILE}")
print(f"Expected: {MAX_PAPERS} papers × {CHUNKS_PER_PAPER} chunks = "
      f"{MAX_PAPERS * CHUNKS_PER_PAPER} entries")
print(f"\nNext step: python embed_upsert.py")
