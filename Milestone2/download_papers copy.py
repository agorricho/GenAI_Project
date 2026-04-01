"""
download_papers.py — Milestone 2, Step 1 of 2
MSA 8700 Final Project (DAIS, Variation B — Research Advisor)

What this script does:
  1. Search arXiv for papers on TOPIC (skips download if PDF already exists)
  2. Extract full raw text from each PDF with pdfplumber
  3. Chunk each paper using RecursiveCharacterTextSplitter (sentence/paragraph-aware, with overlap)
  4. Save all chunks to ./chunks.json

Output: chunks.json — variable number of entries (~20-60 chunks per paper)
No Ollama or Qdrant calls — safe to re-run at any time.
"""

import os
import json
import arxiv
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────────────────────────────
TOPIC       = "ESG Performance impact on Firm Value"
MAX_PAPERS  = 20
PAPERS_DIR  = "./papers"
CHUNKS_FILE = "./chunks.json"

# Chunking parameters — tuned for mxbai-embed-large (512-token context window)
# 1200 chars ≈ 300 tokens: fits 2-3 academic paragraphs, well within model context
# 200-char overlap ≈ 50 tokens: carries one paragraph's conclusion into the next chunk
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200

os.makedirs(PAPERS_DIR, exist_ok=True)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


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
        "arxiv_id": result.get_short_id(),
        "abstract": result.summary.replace("\n", " ").strip(),
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


# ── Step 3: Chunk text with RecursiveCharacterTextSplitter ────────────────────
def chunk_text(text):
    """
    Split text into semantically coherent segments using paragraph/sentence boundaries.
    Returns a list of chunk strings (variable length — content-driven, not fixed count).
    """
    if not text:
        return []
    return _splitter.split_text(text)


# ── Step 4: Build chunk list and save to chunks.json ─────────────────────────
all_chunks = []
for paper_index, paper in enumerate(papers):
    print(f"[{paper_index + 1}/{len(papers)}] Extracting: {paper['title'][:60]}")
    text   = extract_text(paper["pdf_path"])
    chunks = chunk_text(text)
    chunk_total = len(chunks)

    for chunk_index, chunk_text_val in enumerate(chunks):
        all_chunks.append({
            "paper_index": paper_index,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "title":       paper["title"],
            "authors":     paper["authors"],
            "year":        paper["year"],
            "arxiv_id":    paper["arxiv_id"],
            "abstract":    paper["abstract"],
            "pdf_path":    paper["pdf_path"],
            "text":        chunk_text_val,
        })

    print(f"  → {chunk_total} chunks")

with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(all_chunks)} chunks saved to {CHUNKS_FILE}")
print(f"Papers processed: {len(papers)} | Avg chunks/paper: {len(all_chunks) // max(len(papers), 1)}")
print(f"\nNext step: python embed_upsert.py")
