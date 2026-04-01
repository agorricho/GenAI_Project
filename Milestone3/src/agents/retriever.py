"""
Node 2 — Retriever

Input:  state["rephrased_query"]
Output: state["retrieved_chunks"]  (top-5 Qdrant hits)

Embeds the rephrased query via fastembed (nomic-embed-text ONNX, same vectors
as Milestone 2), then vector-searches the msa8700_m3 Qdrant collection.

Each chunk dict: {text, title, authors, year, score}
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# ── Load credentials ──────────────────────────────────────────────────────────
for _p in Path(__file__).resolve().parents:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "msa8700_m3"
TOP_K          = 5

# ── Qdrant client ─────────────────────────────────────────────────────────────
_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── Embedding model (ONNX, compatible with nomic-embed-text vectors in Qdrant) ─
_embedder = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")


def _embed(text: str) -> list[float]:
    """Embed text via fastembed (nomic-embed-text, ONNX — same vectors as Ollama)."""
    return list(next(iter(_embedder.embed([text]))))


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
            "text":    payload.get("abstract", ""),
            "title":   payload.get("title", "Unknown"),
            "authors": payload.get("authors", ""),
            "year":    payload.get("year", ""),
            "score":   round(hit.score, 4),
        })

    print(f"[Retriever] Retrieved {len(chunks)} chunks (top score: {chunks[0]['score'] if chunks else 'n/a'})")
    return {"retrieved_chunks": chunks}
