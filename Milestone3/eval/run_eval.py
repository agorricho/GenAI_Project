"""
run_eval.py — M4 Step 2: evaluate the M3 RAG pipeline

Usage (from Milestone3/):
    python eval/run_eval.py              # auto-resumes on re-run
    python eval/run_eval.py --no-resume  # force full re-run from scratch

Flow:
    load_dataset → init OllamaJudge + FastEmbed →
    for each item (skip if checkpointed):
        run_query → score 5 metrics → atomic checkpoint
    → batch semantic similarity update →
    → generate_report (JSON + CSV + console table)
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Path injection so `from src.pipeline import run_query` works ──────────────
_MILESTONE3_DIR = Path(__file__).resolve().parent.parent
if str(_MILESTONE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MILESTONE3_DIR))

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
for _p in Path(__file__).resolve().parents:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY  = os.getenv("0LLAMA", "")
MODEL           = os.getenv("M3_MODEL", "llama3.2")

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR  = Path(__file__).parent / "results"

from eval.metrics import (
    OllamaJudge,
    batch_semantic_similarity,
    compute_composite_score,
    score_answer_relevancy,
    score_citation_recall,
    score_faithfulness,
    score_rouge_l,
    score_semantic_similarity,
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_dataset(dataset_path: Path) -> list[dict]:
    with open(dataset_path) as f:
        return json.load(f)


def load_checkpoint(results_path: Path) -> dict[str, dict]:
    """Return dict of item_id → result for already-scored items."""
    if not results_path.exists():
        return {}
    try:
        with open(results_path) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("items", [])
        return {item["item_id"]: item for item in items if "scores" in item}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_checkpoint(results: list[dict], results_path: Path) -> None:
    """Atomic write: write to .tmp then rename."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = results_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.replace(results_path)


# ── Single-item evaluation ────────────────────────────────────────────────────

def run_single_item(item: dict, judge: OllamaJudge, embed_model) -> dict:
    """Run pipeline + score all 5 metrics for one dataset item."""
    from src.pipeline import run_query  # imported here to avoid top-level side effects

    result = {
        "item_id":            item["item_id"],
        "question":           item["question"],
        "reference_answer":   item["reference_answer"],
        "expected_source":    item.get("expected_source_title", ""),
        "generation_mode":    item.get("generation_mode", ""),
        "esg_category":       item.get("esg_category", ""),
        "is_in_corpus":       item.get("is_in_corpus", True),
        "pipeline_answer":    "",
        "pipeline_citations": [],
        "pipeline_chunks":    [],
        "scores":             {},
        "faithfulness_reasoning": "",
        "relevancy_reasoning":    "",
        "pipeline_error":     None,
        "duration_seconds":   0.0,
    }

    t0 = time.time()
    try:
        pipeline_out = run_query(item["question"])
        result["pipeline_answer"]    = pipeline_out.get("answer", "")
        result["pipeline_citations"] = pipeline_out.get("citations", [])
        result["pipeline_chunks"]    = pipeline_out.get("chunks", [])
    except Exception as e:
        result["pipeline_error"] = str(e)
        result["duration_seconds"] = round(time.time() - t0, 2)
        return result

    answer     = result["pipeline_answer"]
    chunks     = result["pipeline_chunks"]
    citations  = result["pipeline_citations"]
    reference  = item["reference_answer"]
    question   = item["question"]
    is_corpus  = item.get("is_in_corpus", True)
    exp_title  = item.get("expected_source_title", "")

    scores = {}

    # 1. Faithfulness
    faith = score_faithfulness(answer, chunks, judge)
    scores["faithfulness_score"]  = faith["faithfulness_score"]
    result["faithfulness_reasoning"] = faith.get("reasoning", "")

    # 2. Answer Relevancy
    rel = score_answer_relevancy(question, answer, judge)
    scores["relevancy_score"] = rel["relevancy_score"]
    result["relevancy_reasoning"] = rel.get("reasoning", "")
    time.sleep(1)  # rate-limit between LLM judge calls

    # 3. Semantic Similarity (per-item; batch update happens after full loop)
    sem = score_semantic_similarity(answer, reference, embed_model)
    scores["semantic_similarity_score"] = sem["semantic_similarity_score"]

    # 4. Citation Recall
    cit = score_citation_recall(citations, exp_title, is_corpus)
    scores["citation_recall_score"] = cit["citation_recall_score"]

    # 5. ROUGE-L
    rouge = score_rouge_l(answer, reference)
    scores["rouge_l_score"] = rouge["rouge_l_score"]

    # Composite
    scores["composite"] = compute_composite_score(scores)

    result["scores"] = scores
    result["duration_seconds"] = round(time.time() - t0, 2)
    return result


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    dataset_path: Path,
    results_dir: Path,
    judge: OllamaJudge,
    embed_model,
    resume: bool = True,
) -> list[dict]:
    dataset       = load_dataset(dataset_path)
    checkpoint_path = results_dir / "eval_results.json"
    checkpoint    = load_checkpoint(checkpoint_path) if resume else {}

    results: list[dict] = list(checkpoint.values())
    done_ids = set(checkpoint.keys())

    pending = [item for item in dataset if item["item_id"] not in done_ids]
    print(f"[Eval] {len(done_ids)} already scored — {len(pending)} remaining")

    for i, item in enumerate(pending):
        print(f"[Eval] Item {i+1}/{len(pending)}: {item['item_id']} — {item['question'][:60]}")
        result = run_single_item(item, judge, embed_model)
        results.append(result)
        save_checkpoint(results, checkpoint_path)
        if result.get("pipeline_error"):
            print(f"  ⚠ Error: {result['pipeline_error']}")
        else:
            s = result["scores"]
            print(f"  ✓ composite={s.get('composite', 0):.3f} "
                  f"faith={s.get('faithfulness_score', 0):.3f} "
                  f"rel={s.get('relevancy_score', 0):.3f}")

    # Batch semantic similarity update (replaces per-item estimates with batch computation)
    valid = [r for r in results if not r.get("pipeline_error") and r["pipeline_answer"]]
    pairs = [(r["pipeline_answer"], r["reference_answer"]) for r in valid]
    if pairs:
        print(f"[Eval] Batch embedding {len(pairs)} pairs...")
        batch_scores = batch_semantic_similarity(pairs, embed_model)
        for r, bs in zip(valid, batch_scores):
            r["scores"]["semantic_similarity_score"] = bs
            r["scores"]["composite"] = compute_composite_score(r["scores"])
        save_checkpoint(results, checkpoint_path)

    return results


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(results: list[dict], output_dir: Path, model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid   = [r for r in results if not r.get("pipeline_error") and r.get("scores")]
    errored = [r for r in results if r.get("pipeline_error")]

    def _mean(vals):
        return round(statistics.mean(vals), 4) if vals else 0.0

    def _std(vals):
        return round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0

    metric_keys = ["faithfulness_score", "relevancy_score", "semantic_similarity_score",
                   "citation_recall_score", "rouge_l_score", "composite"]

    summary = {}
    for k in metric_keys:
        vals = [r["scores"][k] for r in valid if k in r["scores"]]
        summary[k] = {"mean": _mean(vals), "std": _std(vals),
                      "min": min(vals) if vals else 0.0,
                      "max": max(vals) if vals else 0.0}

    # By category
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        cat = r.get("esg_category", "unknown")
        by_cat[cat].append(r["scores"].get("composite", 0.0))
    by_category = {cat: {"n": len(v), "composite_mean": _mean(v)} for cat, v in by_cat.items()}

    # By difficulty
    by_diff: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        # difficulty is in dataset; find matching item
        diff = "unknown"
        for item in results:
            if item["item_id"] == r["item_id"]:
                diff = item.get("generation_mode", "unknown")
                break
        by_diff[diff].append(r["scores"].get("composite", 0.0))

    # Full JSON
    report = {
        "metadata": {
            "run_timestamp": datetime.now().isoformat(),
            "model": model,
            "n_items": len(results),
            "n_errors": len(errored),
            "n_scored": len(valid),
        },
        "summary": summary,
        "by_category": by_category,
        "items": results,
    }
    json_path = output_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # CSV
    csv_path = output_dir / "eval_results.csv"
    fieldnames = ["item_id", "question", "generation_mode", "esg_category", "is_in_corpus",
                  "faithfulness_score", "relevancy_score", "semantic_similarity_score",
                  "citation_recall_score", "rouge_l_score", "composite",
                  "pipeline_error", "duration_seconds"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {**r, **r.get("scores", {})}
            writer.writerow(row)

    # Console summary
    _print_summary(summary, by_category, model, len(valid), len(results))
    print(f"\n  Results saved to {output_dir}/")


def _print_summary(summary: dict, by_category: dict, model: str, n_scored: int, n_total: int) -> None:
    label_map = {
        "faithfulness_score":         "Faithfulness",
        "relevancy_score":            "Answer Relevancy",
        "semantic_similarity_score":  "Semantic Similarity",
        "citation_recall_score":      "Citation Recall",
        "rouge_l_score":              "ROUGE-L",
        "composite":                  "COMPOSITE (weighted)",
    }
    print("\n" + "=" * 62)
    print("  M4 EVALUATION SUMMARY — Research Advisor RAG System")
    print("=" * 62)
    print(f"  Model: {model:<20} Items: {n_scored}/{n_total}")
    print()
    print("  METRIC SCORES (mean ± std)")
    print("  ┌─────────────────────────┬────────┬───────┐")
    for k, label in label_map.items():
        s = summary.get(k, {"mean": 0.0, "std": 0.0})
        print(f"  │ {label:<23} │ {s['mean']:.4f} │ {s['std']:.4f} │")
    print("  └─────────────────────────┴────────┴───────┘")
    print()
    print("  BY CATEGORY: " + "  ".join(
        f"{cat}={v['composite_mean']:.2f}" for cat, v in sorted(by_category.items())
    ))
    print("=" * 62)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M4 Evaluation Runner")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force full re-run (ignore checkpoint)")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                        help="Path to eval_dataset.json")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR),
                        help="Directory for results output")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    results_dir  = Path(args.output_dir)

    if not dataset_path.exists():
        print(f"[Eval] Dataset not found: {dataset_path}")
        print("       Run: python eval/generate_dataset.py")
        sys.exit(1)

    print(f"[Eval] Initialising OllamaJudge (model={MODEL})...")
    judge = OllamaJudge(base_url=OLLAMA_BASE_URL, model=MODEL, api_key=OLLAMA_API_KEY)

    print("[Eval] Initialising FastEmbed (nomic-embed-text)...")
    from fastembed import TextEmbedding
    embed_model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")

    results = run_evaluation(
        dataset_path=dataset_path,
        results_dir=results_dir,
        judge=judge,
        embed_model=embed_model,
        resume=not args.no_resume,
    )

    generate_report(results, results_dir, MODEL)


if __name__ == "__main__":
    main()
