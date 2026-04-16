"""
metrics.py — M4 scoring functions

Five metrics:
  Faithfulness        (weight 0.30) — LLM-as-judge
  Answer Relevancy    (weight 0.25) — LLM-as-judge
  Semantic Similarity (weight 0.25) — fastembed cosine
  Citation Recall     (weight 0.15) — fuzzy substring match
  ROUGE-L             (weight 0.05) — lexical overlap

Public API:
  OllamaJudge
  score_faithfulness(answer, chunks, llm_client)
  score_answer_relevancy(question, answer, llm_client)
  score_semantic_similarity(answer, reference_answer, embed_model)
  score_citation_recall(citations, expected_source_title, is_in_corpus)
  score_rouge_l(answer, reference_answer)
  compute_composite_score(scores)
  batch_semantic_similarity(pairs, embed_model)
"""

import json
import math
import re
import time
from typing import Any

import requests
from rouge_score import rouge_scorer

# Metric weights
WEIGHTS = {
    "faithfulness":         0.30,
    "answer_relevancy":     0.25,
    "semantic_similarity":  0.25,
    "citation_recall":      0.15,
    "rouge_l":              0.05,
}


# ── OllamaJudge ───────────────────────────────────────────────────────────────

class OllamaJudge:
    """HTTP judge that calls Ollama /api/generate at temperature=0."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.api_key  = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text

    def judge(self, prompt: str, max_retries: int = 3) -> dict:
        """Call the LLM and parse JSON. Retries on parse failure."""
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0},
        }
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers=self._headers(),
                    timeout=120,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                cleaned = self._strip_fences(raw)
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    # Return safe defaults on final failure
                    return {}
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Ollama request failed: {e}") from e
        return {}


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_faithfulness(answer: str, chunks: list[dict], llm_client: OllamaJudge) -> dict:
    """
    Check whether every factual claim in the answer is supported by the source chunks.
    Returns {faithfulness_score, supported_claims, total_claims, reasoning}
    """
    if not answer or not chunks:
        return {"faithfulness_score": 0.0, "supported_claims": 0, "total_claims": 0, "reasoning": "no input"}

    top_chunks = chunks[:3]
    chunks_text = "\n\n".join(
        f"[{i+1}] {c.get('text', c.get('abstract', ''))[:500]}"
        for i, c in enumerate(top_chunks)
    )

    prompt = f"""You are an evaluator checking whether an AI answer is faithful to its source documents.

Source chunks:
{chunks_text}

Answer to evaluate:
{answer}

For each factual claim in the answer, check if it is directly supported by the source chunks.
Respond with ONLY a JSON object:
{{"supported_claims": <int>, "total_claims": <int>, "faithfulness_score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    result = llm_client.judge(prompt)
    score = float(result.get("faithfulness_score", 0.0))
    score = max(0.0, min(1.0, score))
    return {
        "faithfulness_score":  score,
        "supported_claims":    int(result.get("supported_claims", 0)),
        "total_claims":        int(result.get("total_claims", 0)),
        "reasoning":           result.get("reasoning", ""),
    }


def score_answer_relevancy(question: str, answer: str, llm_client: OllamaJudge) -> dict:
    """
    Rate how well the answer addresses the question.
    Returns {relevancy_score, reasoning}
    """
    if not question or not answer:
        return {"relevancy_score": 0.0, "reasoning": "no input"}

    prompt = f"""You are evaluating whether an answer addresses the given question.

Question: {question}
Answer: {answer}

Rate from 0.0 (completely unrelated) to 1.0 (fully and directly addresses the question).
Respond with ONLY: {{"relevancy_score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    result = llm_client.judge(prompt)
    score = float(result.get("relevancy_score", 0.0))
    score = max(0.0, min(1.0, score))
    return {
        "relevancy_score": score,
        "reasoning":       result.get("reasoning", ""),
    }


def score_semantic_similarity(answer: str, reference_answer: str, embed_model: Any) -> dict:
    """
    Cosine similarity between answer and reference using fastembed.
    Returns {semantic_similarity_score}
    """
    if not answer or not reference_answer:
        return {"semantic_similarity_score": 0.0}

    vecs = list(embed_model.embed([answer, reference_answer]))
    a, b = vecs[0], vecs[1]

    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if mag_a == 0 or mag_b == 0:
        return {"semantic_similarity_score": 0.0}

    score = max(0.0, min(1.0, dot / (mag_a * mag_b)))
    return {"semantic_similarity_score": round(score, 4)}


def score_citation_recall(citations: list[dict], expected_source_title: str, is_in_corpus: bool) -> dict:
    """
    Check if the expected paper title appears in pipeline citations.
    For off-topic items (is_in_corpus=False): score=1.0 if NOT cited.
    Returns {citation_recall_score}
    """
    if not expected_source_title or expected_source_title == "N/A":
        # Negative items with no expected title — score 1.0 if citations are empty/irrelevant
        return {"citation_recall_score": 1.0 if not citations else 0.5}

    cited_titles = [c.get("title", "").lower() for c in (citations or [])]
    expected_lower = expected_source_title.lower()

    found = any(expected_lower in ct or ct in expected_lower for ct in cited_titles)

    if is_in_corpus:
        score = 1.0 if found else 0.0
    else:
        # Off-topic: reward NOT citing unrelated sources
        score = 0.0 if found else 1.0

    return {"citation_recall_score": score}


def score_rouge_l(answer: str, reference_answer: str) -> dict:
    """
    ROUGE-L (F-measure) between answer and reference.
    Returns {rouge_l_score}
    """
    if not answer or not reference_answer:
        return {"rouge_l_score": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference_answer, answer)
    return {"rouge_l_score": round(scores["rougeL"].fmeasure, 4)}


def compute_composite_score(scores: dict) -> float:
    """
    Weighted composite:
      Faithfulness×0.30 + Relevancy×0.25 + SemanticSim×0.25 + CitationRecall×0.15 + ROUGE-L×0.05
    """
    composite = (
        scores.get("faithfulness_score",        0.0) * WEIGHTS["faithfulness"] +
        scores.get("relevancy_score",            0.0) * WEIGHTS["answer_relevancy"] +
        scores.get("semantic_similarity_score",  0.0) * WEIGHTS["semantic_similarity"] +
        scores.get("citation_recall_score",      0.0) * WEIGHTS["citation_recall"] +
        scores.get("rouge_l_score",              0.0) * WEIGHTS["rouge_l"]
    )
    return round(max(0.0, min(1.0, composite)), 4)


def batch_semantic_similarity(pairs: list[tuple[str, str]], embed_model: Any) -> list[float]:
    """
    Compute cosine similarity for all (answer, reference) pairs in 2 batch embed calls.
    """
    if not pairs:
        return []

    answers    = [p[0] for p in pairs]
    references = [p[1] for p in pairs]

    all_texts = answers + references
    all_vecs  = list(embed_model.embed(all_texts))

    n = len(pairs)
    answer_vecs    = all_vecs[:n]
    reference_vecs = all_vecs[n:]

    results = []
    for a, b in zip(answer_vecs, reference_vecs):
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            results.append(0.0)
        else:
            results.append(round(max(0.0, min(1.0, dot / (mag_a * mag_b))), 4))
    return results
