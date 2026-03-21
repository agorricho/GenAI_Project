"""
Node 2 — Retriever

Input:  state["rephrased_query"]
Output: state["retrieved_chunks"]  (top-5 Qdrant hits)

Embeds the rephrased query via the same Ollama/nomic-embed-text call used in
Milestone 2, then vector-searches the msa8700_m2 Qdrant collection.

Each chunk dict: {text, title, authors, year, score}
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# ── Load credentials ──────────────────────────────────────────────────────────
for _p in Path(__file__).resolve().parents:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
OLLAMA_API_KEY  = os.getenv("0LLAMA", "")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434") + "/api/embeddings"
EMBED_MODEL     = "nomic-embed-text"
COLLECTION      = "msa8700_m2"
TOP_K           = 5
TEXT_LIMIT      = 1500

# ── Qdrant client ─────────────────────────────────────────────────────────────
_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _embed(text: str) -> list[float]:
    """Embed text via Ollama (mirrors M2 embed_with_retry logic)."""
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    resp = requests.post(
        OLLAMA_EMBED_URL,
        headers=headers,
        json={"model": EMBED_MODEL, "prompt": text[:TEXT_LIMIT], "keep_alive": "5m"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ── Node function ─────────────────────────────────────────────────────────────
def retriever_node(state: dict) -> dict:
    query = state["rephrased_query"]
    print(f"[Retriever] Embedding query and searching '{COLLECTION}'...")

    vector = _embed(query)
    hits = _qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    chunks = []
    for hit in hits:
        payload = hit.payload or {}
        chunks.append({
            "text":    payload.get("text", ""),
            "title":   payload.get("title", "Unknown"),
            "authors": payload.get("authors", ""),
            "year":    payload.get("year", ""),
            "score":   round(hit.score, 4),
        })

    print(f"[Retriever] Retrieved {len(chunks)} chunks (top score: {chunks[0]['score'] if chunks else 'n/a'})")
    return {"retrieved_chunks": chunks}
