"""
embed_upsert.py — Milestone 2, Step 2 of 2
MSA 8700 Final Project (DAIS, Variation B — Research Advisor)

What this script does:
  1. Load chunks.json produced by download_papers.py
  2. Embed each chunk via Ollama Cloud API (nomic-embed-text, 768-dim)
  3. Upsert vectors + metadata to Qdrant Cloud, one paper at a time
  4. Checkpoint progress to progress.json after each paper

Re-run safely after a crash — resumes from last completed paper.
"""

import os
import json
import time
import uuid
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

# ── Load credentials from .env ────────────────────────────────────────────────
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

def _load_env_fallback(env_path):
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, val = parts
                        val = val.strip('"').strip("'")
                        if not os.environ.get(key):
                            os.environ[key] = val
    except FileNotFoundError:
        raise FileNotFoundError(f".env file not found at: {env_path}")

_load_env_fallback(ENV_PATH)

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_API_KEY = os.getenv("0LLAMA")   # NOTE: key name starts with zero in .env

if not QDRANT_URL or not QDRANT_API_KEY or not OLLAMA_API_KEY:
    raise ValueError(
        f"Missing credentials. Check .env at {ENV_PATH}\n"
        f"  QDRANT_URL     = {'set' if QDRANT_URL else 'MISSING'}\n"
        f"  QDRANT_API_KEY = {'set' if QDRANT_API_KEY else 'MISSING'}\n"
        f"  0LLAMA         = {'set' if OLLAMA_API_KEY else 'MISSING'}"
    )

# ── Configuration ─────────────────────────────────────────────────────────────
COLLECTION      = "msa8700_m2"
OLLAMA_BASE_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL     = "nomic-embed-text"
VECTOR_DIM      = 768
TEXT_LIMIT      = 1500   # chars per embed call (reduced further for low-RAM WSL2)
CHUNKS_FILE     = "./chunks.json"
PROGRESS_FILE   = "./progress.json"


# ── Load chunks ───────────────────────────────────────────────────────────────
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        f"{CHUNKS_FILE} not found. Run download_papers.py first."
    )

with open(CHUNKS_FILE, encoding="utf-8") as f:
    all_chunks = json.load(f)

print(f"Loaded {len(all_chunks)} chunks from {CHUNKS_FILE}")

# Group chunks by paper_index for per-paper processing
papers_map: dict[int, list[dict]] = {}
for chunk in all_chunks:
    idx = chunk["paper_index"]
    papers_map.setdefault(idx, []).append(chunk)

paper_indices = sorted(papers_map.keys())
print(f"Papers to process: {len(paper_indices)}")


# ── Checkpoint: determine resume point ───────────────────────────────────────
resume_index = -1
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        resume_index = json.load(f).get("last_completed_paper_index", -1)
    print(f"Checkpoint found — resuming from paper index {resume_index + 1}")
else:
    print("No checkpoint found — starting fresh run")


# ── Qdrant setup ──────────────────────────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if resume_index == -1:
    # Fresh run: recreate collection
    if client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' already exists — recreating...")
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION}' created.\n")
else:
    print(f"Resuming — keeping existing collection '{COLLECTION}'.\n")


# ── Embed with retry + sleep ──────────────────────────────────────────────────
def embed_with_retry(text: str, max_retries: int = 3) -> list[float]:
    """Embed text via Ollama with exponential backoff on HTTP 500."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_BASE_URL,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
                json={"model": EMBED_MODEL, "prompt": text[:TEXT_LIMIT], "keep_alive": "5m"}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)   # 10s, 20s, 40s
                print(f"    Embed failed (attempt {attempt + 1}), "
                      f"retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def unload_model() -> None:
    """Force Ollama to evict the embedding model from RAM immediately."""
    try:
        requests.post(
            OLLAMA_BASE_URL,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
            json={"model": EMBED_MODEL, "prompt": "", "keep_alive": 0},
            timeout=10,
        )
        time.sleep(3)   # give OS time to reclaim RAM before next paper loads model
    except Exception:
        pass  # Non-fatal: next paper's embed_with_retry will reload the model


# ── Main loop: embed per paper, upsert, checkpoint ───────────────────────────
total_upserted = 0

for paper_index in paper_indices:
    if paper_index <= resume_index:
        print(f"  [skip] Paper index {paper_index} (already completed)")
        total_upserted += len(papers_map[paper_index])
        continue

    chunks = papers_map[paper_index]
    title  = chunks[0]["title"]
    print(f"[{paper_index + 1}/{len(paper_indices)}] Embedding: {title[:60]}")

    points = []
    for chunk in chunks:
        vector = embed_with_retry(chunk["text"])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "title":       chunk["title"],
                "authors":     chunk["authors"],
                "year":        chunk["year"],
                "pdf_path":    chunk["pdf_path"],
                "chunk_index": chunk["chunk_index"],
            }
        ))

    unload_model()   # free RAM before moving to next paper

    client.upsert(collection_name=COLLECTION, points=points)
    total_upserted += len(points)
    print(f"  → {len(points)} vectors upserted  (total so far: {total_upserted})")

    # Save checkpoint
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_completed_paper_index": paper_index}, f)

print(f"\nDone. {total_upserted} vectors stored in Qdrant collection '{COLLECTION}'.")
print(f"\nVerify:")
print(f'  python -c "')
print(f"  import os; from dotenv import load_dotenv; load_dotenv('.env')")
print(f"  from qdrant_client import QdrantClient")
print(f"  c = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))")
print(f"  print('Vectors stored:', c.get_collection('{COLLECTION}').vectors_count)")
print(f'  "')
