"""
generate_dataset.py — M4 Step 1: produce eval_dataset.json

75 Q&A items sourced from Qdrant msa8700_m3 collection:
  - Mode A (abstract-seeded): 40 items — 2 per paper from abstract
  - Mode B (chunk-grounded):  30 items — 1 per randomly-sampled paper (seed=42)
  - Negative:                  5 items — off-topic / out-of-scope questions

Run from Milestone3/:
    python eval/generate_dataset.py
"""

import json
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# ── Load .env ────────────────────────────────────────────────────────────────
for _p in Path(__file__).resolve().parents:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "msa8700_m3"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY  = os.getenv("0LLAMA", "")
MODEL           = os.getenv("M3_MODEL", "llama3.2")

OUTPUT_PATH = Path(__file__).parent / "eval_dataset.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_qdrant_papers() -> list[dict]:
    """Scroll all points from Qdrant and return as paper dicts."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    papers = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in result:
            p = point.payload or {}
            papers.append({
                "title":       p.get("title", "Unknown"),
                "authors":     p.get("authors", ""),
                "year":        p.get("year", ""),
                "abstract":    p.get("abstract", ""),
                "chunk_index": p.get("chunk_index", 0),
                "point_id":    str(point.id),
            })
        if next_offset is None:
            break
        offset = next_offset
    print(f"[Dataset] Loaded {len(papers)} points from Qdrant")
    return papers


def _parse_json_from_response(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _call_ollama_json(prompt: str, model: str, base_url: str, api_key: str, max_retries: int = 3) -> dict:
    """POST to Ollama /api/generate, parse JSON response. Retry on parse failure."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return _parse_json_from_response(raw)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"  [Dataset] JSON parse failed (attempt {attempt+1}), retrying...")
                time.sleep(2)
            else:
                raise RuntimeError(f"Failed to parse JSON after {max_retries} attempts: {e}") from e
    return {}


# ── Generation modes ─────────────────────────────────────────────────────────

def generate_abstract_items(papers: list[dict], n_per_paper: int, model: str, base_url: str, api_key: str) -> list[dict]:
    """Mode A: 2 questions per paper from abstract (40 items total)."""
    items = []
    item_num = 1

    # Use unique papers (dedupe by title)
    seen_titles: set[str] = set()
    unique_papers = []
    for p in papers:
        if p["title"] not in seen_titles and p["abstract"]:
            seen_titles.add(p["title"])
            unique_papers.append(p)

    for paper in unique_papers:
        if len(items) >= 40:
            break
        abstract = paper["abstract"][:1500]  # truncate for context window
        prompt = f"""You are creating evaluation questions for a RAG system.

Paper title: {paper['title']}
Abstract: {abstract}

Generate exactly {n_per_paper} research questions that can be answered using this abstract.
For each question, also write a reference answer (1-3 sentences) based solely on the abstract.

Respond with ONLY a JSON array:
[
  {{"question": "...", "reference_answer": "..."}},
  {{"question": "...", "reference_answer": "..."}}
]"""

        try:
            result = _call_ollama_json(prompt, model, base_url, api_key)
            if isinstance(result, list):
                qa_pairs = result
            elif isinstance(result, dict) and "questions" in result:
                qa_pairs = result["questions"]
            else:
                print(f"  [Dataset] Unexpected response format for '{paper['title'][:40]}'")
                continue

            for qa in qa_pairs[:n_per_paper]:
                items.append({
                    "item_id": f"item_{item_num:03d}",
                    "question": qa.get("question", ""),
                    "reference_answer": qa.get("reference_answer", ""),
                    "expected_source_title": paper["title"],
                    "source_chunk_index": paper.get("chunk_index", 0),
                    "generation_mode": "abstract",
                    "difficulty": "broad",
                    "esg_category": "",
                    "is_in_corpus": True,
                })
                item_num += 1
                if len(items) >= 40:
                    break
            print(f"  [Dataset] Abstract item {len(items)}/40 — '{paper['title'][:50]}'")
        except Exception as e:
            print(f"  [Dataset] Error on '{paper['title'][:40]}': {e}")

    return items


def generate_chunk_items(papers: list[dict], n_total: int, model: str, base_url: str, api_key: str) -> list[dict]:
    """Mode B: 1 targeted question per sampled paper (30 items, seed=42)."""
    rng = random.Random(42)

    seen_titles: set[str] = set()
    unique_papers = [p for p in papers if p["abstract"]]
    deduped: list[dict] = []
    for p in unique_papers:
        if p["title"] not in seen_titles:
            seen_titles.add(p["title"])
            deduped.append(p)

    sampled = rng.sample(deduped, min(n_total, len(deduped)))
    items = []

    for i, paper in enumerate(sampled):
        abstract = paper["abstract"][:1500]
        prompt = f"""You are creating a precise evaluation question for a RAG retrieval test.

Paper title: {paper['title']}
Abstract: {abstract}

Generate exactly 1 specific question that targets a precise fact, number, or finding from this abstract.
Write a reference answer (1-2 sentences) based solely on the abstract.

Respond with ONLY a JSON object:
{{"question": "...", "reference_answer": "..."}}"""

        try:
            result = _call_ollama_json(prompt, model, base_url, api_key)
            items.append({
                "item_id": f"item_{100 + i + 1:03d}",
                "question": result.get("question", ""),
                "reference_answer": result.get("reference_answer", ""),
                "expected_source_title": paper["title"],
                "source_chunk_index": paper.get("chunk_index", 0),
                "generation_mode": "chunk",
                "difficulty": "specific",
                "esg_category": "",
                "is_in_corpus": True,
            })
            print(f"  [Dataset] Chunk item {len(items)}/30 — '{paper['title'][:50]}'")
        except Exception as e:
            print(f"  [Dataset] Error on chunk item '{paper['title'][:40]}': {e}")

    return items


def generate_negative_items(papers: list[dict], model: str, base_url: str, api_key: str) -> list[dict]:
    """5 off-topic questions that should NOT be answered by the corpus."""
    # Try to find non-ESG papers in corpus
    non_esg_keywords = {"algorithm", "neural", "genetic", "scheduling", "protein",
                        "image", "audio", "vision", "robotics", "npv", "codesign"}
    off_topic = [
        p for p in papers
        if any(kw in (p["title"] + p["abstract"]).lower() for kw in non_esg_keywords)
    ]

    # Hardcoded fallback questions if corpus is homogeneous
    hardcoded = [
        {
            "question": "What is the optimal algorithm for solving the travelling salesman problem?",
            "reference_answer": "This topic is not covered in the research corpus.",
            "expected_source_title": "N/A",
        },
        {
            "question": "How do convolutional neural networks process image data?",
            "reference_answer": "This topic is not covered in the research corpus.",
            "expected_source_title": "N/A",
        },
        {
            "question": "What is the melting point of iron?",
            "reference_answer": "This topic is not covered in the research corpus.",
            "expected_source_title": "N/A",
        },
        {
            "question": "How does protein folding relate to Alzheimer's disease?",
            "reference_answer": "This topic is not covered in the research corpus.",
            "expected_source_title": "N/A",
        },
        {
            "question": "What genetic mutation causes cystic fibrosis?",
            "reference_answer": "This topic is not covered in the research corpus.",
            "expected_source_title": "N/A",
        },
    ]

    items = []
    sources = off_topic[:5] if len(off_topic) >= 5 else []

    if sources:
        # Generate off-topic questions from genuinely off-topic papers
        for i, paper in enumerate(sources[:5]):
            prompt = f"""Generate 1 question about this paper that is completely unrelated to ESG (Environmental, Social, Governance) research.
Paper title: {paper['title']}
The question should be about the paper's non-ESG topic, making it off-topic for an ESG research advisor.
Respond with ONLY: {{"question": "...", "reference_answer": "Not covered in the ESG corpus."}}"""
            try:
                result = _call_ollama_json(prompt, model, base_url, api_key)
                items.append({
                    "item_id": f"item_{131 + i:03d}",
                    "question": result.get("question", hardcoded[i]["question"]),
                    "reference_answer": "Not covered in the ESG research corpus.",
                    "expected_source_title": paper["title"],
                    "source_chunk_index": 0,
                    "generation_mode": "negative",
                    "difficulty": "out-of-scope",
                    "esg_category": "off_topic",
                    "is_in_corpus": False,
                })
            except Exception:
                items.append({**hardcoded[i], "item_id": f"item_{131 + i:03d}",
                               "source_chunk_index": 0, "generation_mode": "negative",
                               "difficulty": "out-of-scope", "esg_category": "off_topic",
                               "is_in_corpus": False})
    else:
        # Corpus is ESG-homogeneous — use hardcoded off-topic items
        print("  [Dataset] Corpus appears homogeneous — using hardcoded negative items")
        for i, hc in enumerate(hardcoded):
            items.append({
                "item_id": f"item_{131 + i:03d}",
                "question": hc["question"],
                "reference_answer": hc["reference_answer"],
                "expected_source_title": hc["expected_source_title"],
                "source_chunk_index": 0,
                "generation_mode": "negative",
                "difficulty": "out-of-scope",
                "esg_category": "off_topic",
                "is_in_corpus": False,
            })

    return items[:5]


def assign_categories(items: list[dict]) -> list[dict]:
    """Assign esg_category via keyword heuristic on question text."""
    env_kw  = {"environment", "emission", "carbon", "climate", "green", "renewable",
                "pollution", "waste", "energy", "biodiversity"}
    soc_kw  = {"social", "labor", "employee", "diversity", "community", "human rights",
                "workforce", "safety", "health", "gender"}
    gov_kw  = {"governance", "board", "ceo", "executive", "audit", "transparency",
                "shareholder", "accountability", "corruption", "compliance"}

    for item in items:
        if item["esg_category"]:  # already assigned (e.g. off_topic)
            continue
        q = item["question"].lower()
        if any(kw in q for kw in env_kw):
            item["esg_category"] = "environmental"
        elif any(kw in q for kw in soc_kw):
            item["esg_category"] = "social"
        elif any(kw in q for kw in gov_kw):
            item["esg_category"] = "governance"
        else:
            item["esg_category"] = "general"
    return items


def save_dataset(items: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    # Summary
    from collections import Counter
    modes = Counter(i["generation_mode"] for i in items)
    print(f"\n[Dataset] Saved {len(items)} items to {output_path}")
    print(f"  abstract: {modes.get('abstract', 0)}")
    print(f"  chunk:    {modes.get('chunk', 0)}")
    print(f"  negative: {modes.get('negative', 0)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("[Dataset] Starting dataset generation...")
    papers = load_qdrant_papers()

    print("[Dataset] Generating abstract-seeded items (40)...")
    abstract_items = generate_abstract_items(papers, n_per_paper=2,
                                              model=MODEL, base_url=OLLAMA_BASE_URL,
                                              api_key=OLLAMA_API_KEY)

    print("[Dataset] Generating chunk-grounded items (30)...")
    chunk_items = generate_chunk_items(papers, n_total=30,
                                        model=MODEL, base_url=OLLAMA_BASE_URL,
                                        api_key=OLLAMA_API_KEY)

    print("[Dataset] Generating negative items (5)...")
    negative_items = generate_negative_items(papers, model=MODEL,
                                              base_url=OLLAMA_BASE_URL,
                                              api_key=OLLAMA_API_KEY)

    all_items = abstract_items[:40] + chunk_items[:30] + negative_items[:5]

    # Re-number item_ids sequentially
    for idx, item in enumerate(all_items):
        item["item_id"] = f"item_{idx + 1:03d}"

    all_items = assign_categories(all_items)
    save_dataset(all_items, OUTPUT_PATH)


if __name__ == "__main__":
    main()
