"""
test_m4_eval.py — Unit tests for M4 evaluation framework (AC-7)

All network calls are mocked — no real Qdrant/Ollama required.
Run from Milestone3/:
    pytest tests/test_m4_eval.py -v
"""

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_MILESTONE3 = Path(__file__).resolve().parent.parent
if str(_MILESTONE3) not in sys.path:
    sys.path.insert(0, str(_MILESTONE3))

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
from eval.run_eval import load_checkpoint, save_checkpoint


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_judge():
    judge = MagicMock(spec=OllamaJudge)
    return judge


@pytest.fixture
def mock_embed_model():
    model = MagicMock()
    # Returns 3-dim unit vectors for testing
    model.embed = MagicMock(return_value=iter([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]))
    return model


@pytest.fixture
def sample_chunks():
    return [
        {"text": "ESG scores significantly predict firm performance.", "title": "ESG Study 1"},
        {"text": "Higher governance scores reduce cost of capital.", "title": "ESG Study 2"},
        {"text": "Environmental practices correlate with ROA.", "title": "ESG Study 3"},
    ]


# ── T5.1: All 5 metric functions ─────────────────────────────────────────────

class TestScoreFaithfulness:
    def test_returns_score_in_range(self, mock_judge, sample_chunks):
        mock_judge.judge.return_value = {
            "faithfulness_score": 0.85,
            "supported_claims": 4,
            "total_claims": 5,
            "reasoning": "Most claims are supported.",
        }
        result = score_faithfulness("Good answer.", sample_chunks, mock_judge)
        assert 0.0 <= result["faithfulness_score"] <= 1.0
        assert result["faithfulness_score"] == 0.85
        assert result["supported_claims"] == 4
        assert result["total_claims"] == 5

    def test_empty_answer_returns_zero(self, mock_judge, sample_chunks):
        result = score_faithfulness("", sample_chunks, mock_judge)
        assert result["faithfulness_score"] == 0.0
        mock_judge.judge.assert_not_called()

    def test_empty_chunks_returns_zero(self, mock_judge):
        result = score_faithfulness("An answer.", [], mock_judge)
        assert result["faithfulness_score"] == 0.0

    def test_score_clamped_above_one(self, mock_judge, sample_chunks):
        mock_judge.judge.return_value = {"faithfulness_score": 1.5, "supported_claims": 5,
                                          "total_claims": 5, "reasoning": ""}
        result = score_faithfulness("answer", sample_chunks, mock_judge)
        assert result["faithfulness_score"] == 1.0

    def test_score_clamped_below_zero(self, mock_judge, sample_chunks):
        mock_judge.judge.return_value = {"faithfulness_score": -0.1, "supported_claims": 0,
                                          "total_claims": 5, "reasoning": ""}
        result = score_faithfulness("answer", sample_chunks, mock_judge)
        assert result["faithfulness_score"] == 0.0

    def test_uses_top_3_chunks_only(self, mock_judge):
        chunks = [{"text": f"Chunk {i}"} for i in range(10)]
        mock_judge.judge.return_value = {"faithfulness_score": 0.9, "supported_claims": 3,
                                          "total_claims": 3, "reasoning": ""}
        score_faithfulness("answer", chunks, mock_judge)
        call_prompt = mock_judge.judge.call_args[0][0]
        # Only [1], [2], [3] should appear
        assert "[4]" not in call_prompt


class TestScoreAnswerRelevancy:
    def test_returns_score_in_range(self, mock_judge):
        mock_judge.judge.return_value = {"relevancy_score": 0.9, "reasoning": "Directly addresses it."}
        result = score_answer_relevancy("What is ESG?", "ESG stands for...", mock_judge)
        assert 0.0 <= result["relevancy_score"] <= 1.0
        assert result["relevancy_score"] == 0.9

    def test_empty_inputs_return_zero(self, mock_judge):
        assert score_answer_relevancy("", "answer", mock_judge)["relevancy_score"] == 0.0
        assert score_answer_relevancy("question", "", mock_judge)["relevancy_score"] == 0.0
        mock_judge.judge.assert_not_called()

    def test_score_clamped(self, mock_judge):
        mock_judge.judge.return_value = {"relevancy_score": 2.0, "reasoning": ""}
        result = score_answer_relevancy("q", "a", mock_judge)
        assert result["relevancy_score"] == 1.0


class TestScoreSemanticSimilarity:
    def test_identical_vectors_score_one(self):
        embed_model = MagicMock()
        embed_model.embed = MagicMock(return_value=iter([[1.0, 0.0], [1.0, 0.0]]))
        result = score_semantic_similarity("a", "a", embed_model)
        assert abs(result["semantic_similarity_score"] - 1.0) < 1e-4

    def test_orthogonal_vectors_score_zero(self):
        embed_model = MagicMock()
        embed_model.embed = MagicMock(return_value=iter([[1.0, 0.0], [0.0, 1.0]]))
        result = score_semantic_similarity("a", "b", embed_model)
        assert result["semantic_similarity_score"] == 0.0

    def test_partial_similarity(self):
        # cos([1,0], [0.8, 0.6]) = 0.8
        embed_model = MagicMock()
        embed_model.embed = MagicMock(return_value=iter([[1.0, 0.0], [0.8, 0.6]]))
        result = score_semantic_similarity("a", "b", embed_model)
        assert abs(result["semantic_similarity_score"] - 0.8) < 1e-3

    def test_empty_inputs_return_zero(self):
        embed_model = MagicMock()
        assert score_semantic_similarity("", "ref", embed_model)["semantic_similarity_score"] == 0.0
        assert score_semantic_similarity("ans", "", embed_model)["semantic_similarity_score"] == 0.0


class TestScoreCitationRecall:
    def test_exact_title_match(self):
        citations = [{"title": "ESG and Firm Performance", "authors": "Smith", "year": 2021}]
        result = score_citation_recall(citations, "ESG and Firm Performance", is_in_corpus=True)
        assert result["citation_recall_score"] == 1.0

    def test_case_insensitive_match(self):
        citations = [{"title": "esg and firm performance"}]
        result = score_citation_recall(citations, "ESG and Firm Performance", is_in_corpus=True)
        assert result["citation_recall_score"] == 1.0

    def test_partial_substring_match(self):
        citations = [{"title": "ESG and Firm Performance in Asia"}]
        result = score_citation_recall(citations, "ESG and Firm Performance", is_in_corpus=True)
        assert result["citation_recall_score"] == 1.0

    def test_not_cited_in_corpus_is_zero(self):
        citations = [{"title": "Unrelated Paper"}]
        result = score_citation_recall(citations, "ESG and Firm Performance", is_in_corpus=True)
        assert result["citation_recall_score"] == 0.0

    def test_empty_citations_in_corpus_is_zero(self):
        result = score_citation_recall([], "ESG and Firm Performance", is_in_corpus=True)
        assert result["citation_recall_score"] == 0.0

    def test_off_topic_not_cited_is_one(self):
        """Off-topic item: reward when expected source is NOT cited."""
        citations = [{"title": "Some ESG Paper"}]
        result = score_citation_recall(citations, "Genetic Algorithm Scheduling", is_in_corpus=False)
        assert result["citation_recall_score"] == 1.0

    def test_off_topic_incorrectly_cited_is_zero(self):
        citations = [{"title": "Genetic Algorithm Scheduling"}]
        result = score_citation_recall(citations, "Genetic Algorithm Scheduling", is_in_corpus=False)
        assert result["citation_recall_score"] == 0.0

    def test_na_title_returns_one_for_empty_citations(self):
        result = score_citation_recall([], "N/A", is_in_corpus=False)
        assert result["citation_recall_score"] == 1.0


class TestScoreRougeL:
    def test_identical_answers(self):
        result = score_rouge_l("The quick brown fox.", "The quick brown fox.")
        assert result["rouge_l_score"] == 1.0

    def test_completely_different(self):
        result = score_rouge_l("Alpha beta gamma.", "One two three.")
        assert result["rouge_l_score"] < 0.3

    def test_partial_overlap(self):
        result = score_rouge_l("ESG improves firm performance.", "ESG affects firm value.")
        assert 0.0 < result["rouge_l_score"] < 1.0

    def test_empty_answer_returns_zero(self):
        assert score_rouge_l("", "reference")["rouge_l_score"] == 0.0
        assert score_rouge_l("answer", "")["rouge_l_score"] == 0.0


# ── T5.2: compute_composite_score ─────────────────────────────────────────────

class TestCompositeScore:
    def test_all_ones(self):
        scores = {
            "faithfulness_score":        1.0,
            "relevancy_score":           1.0,
            "semantic_similarity_score": 1.0,
            "citation_recall_score":     1.0,
            "rouge_l_score":             1.0,
        }
        assert compute_composite_score(scores) == 1.0

    def test_all_zeros(self):
        scores = {
            "faithfulness_score":        0.0,
            "relevancy_score":           0.0,
            "semantic_similarity_score": 0.0,
            "citation_recall_score":     0.0,
            "rouge_l_score":             0.0,
        }
        assert compute_composite_score(scores) == 0.0

    def test_correct_weights(self):
        # Only faithfulness = 1.0 → composite = 0.30
        scores = {
            "faithfulness_score":        1.0,
            "relevancy_score":           0.0,
            "semantic_similarity_score": 0.0,
            "citation_recall_score":     0.0,
            "rouge_l_score":             0.0,
        }
        assert abs(compute_composite_score(scores) - 0.30) < 1e-6

    def test_weights_sum_to_one(self):
        from eval.metrics import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_partial_scores(self):
        scores = {
            "faithfulness_score":        0.8,
            "relevancy_score":           0.6,
            "semantic_similarity_score": 0.7,
            "citation_recall_score":     1.0,
            "rouge_l_score":             0.5,
        }
        expected = 0.8*0.30 + 0.6*0.25 + 0.7*0.25 + 1.0*0.15 + 0.5*0.05
        assert abs(compute_composite_score(scores) - expected) < 1e-6

    def test_missing_keys_default_to_zero(self):
        scores = {"faithfulness_score": 1.0}
        assert abs(compute_composite_score(scores) - 0.30) < 1e-6

    def test_clamped_to_zero_one(self):
        scores = {k: 2.0 for k in ["faithfulness_score", "relevancy_score",
                                     "semantic_similarity_score", "citation_recall_score", "rouge_l_score"]}
        assert compute_composite_score(scores) == 1.0


# ── T5.3: checkpoint round-trip ───────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        results = [
            {"item_id": "item_001", "question": "Q1", "scores": {"composite": 0.8}},
            {"item_id": "item_002", "question": "Q2", "scores": {"composite": 0.6}},
        ]
        path = tmp_path / "eval_results.json"
        save_checkpoint(results, path)
        loaded = load_checkpoint(path)
        assert len(loaded) == 2
        assert "item_001" in loaded
        assert loaded["item_001"]["scores"]["composite"] == 0.8

    def test_atomic_write_no_corruption(self, tmp_path):
        """Ensure .tmp file is cleaned up (renamed) after write."""
        results = [{"item_id": "item_001", "scores": {"composite": 0.5}}]
        path = tmp_path / "eval_results.json"
        save_checkpoint(results, path)
        # .tmp file should not exist after successful write
        assert not path.with_suffix(".tmp").exists()
        assert path.exists()

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_checkpoint(path) == {}

    def test_load_corrupted_returns_empty(self, tmp_path):
        path = tmp_path / "eval_results.json"
        path.write_text("NOT VALID JSON")
        assert load_checkpoint(path) == {}

    def test_load_skips_items_without_scores(self, tmp_path):
        results = [
            {"item_id": "item_001", "scores": {"composite": 0.8}},
            {"item_id": "item_002"},  # no scores key
        ]
        path = tmp_path / "eval_results.json"
        path.write_text(json.dumps(results))
        loaded = load_checkpoint(path)
        assert "item_001" in loaded
        assert "item_002" not in loaded

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "results.json"
        save_checkpoint([{"item_id": "x", "scores": {}}], path)
        assert path.exists()


# ── T5.4: dataset schema validation ──────────────────────────────────────────

class TestDatasetSchema:
    REQUIRED_KEYS = {
        "item_id", "question", "reference_answer", "expected_source_title",
        "source_chunk_index", "generation_mode", "difficulty", "esg_category", "is_in_corpus"
    }

    def _make_item(self, mode="abstract", idx=1):
        return {
            "item_id": f"item_{idx:03d}",
            "question": f"Question {idx}?",
            "reference_answer": f"Answer {idx}.",
            "expected_source_title": "Some Paper",
            "source_chunk_index": 0,
            "generation_mode": mode,
            "difficulty": "broad",
            "esg_category": "general",
            "is_in_corpus": True,
        }

    def test_required_keys_present(self):
        item = self._make_item()
        for key in self.REQUIRED_KEYS:
            assert key in item, f"Missing key: {key}"

    def test_mode_counts_75_total(self):
        abstract = [self._make_item("abstract", i) for i in range(1, 41)]
        chunk    = [self._make_item("chunk",    i) for i in range(41, 71)]
        negative = [self._make_item("negative", i) for i in range(71, 76)]
        dataset  = abstract + chunk + negative
        assert len(dataset) == 75

    def test_mode_counts_correct(self):
        abstract = [self._make_item("abstract", i) for i in range(1, 41)]
        chunk    = [self._make_item("chunk",    i) for i in range(41, 71)]
        negative = [self._make_item("negative", i) for i in range(71, 76)]
        dataset  = abstract + chunk + negative

        modes = {}
        for item in dataset:
            modes[item["generation_mode"]] = modes.get(item["generation_mode"], 0) + 1
        assert modes["abstract"] == 40
        assert modes["chunk"]    == 30
        assert modes["negative"] == 5

    def test_item_ids_unique(self):
        items = [self._make_item("abstract", i) for i in range(1, 76)]
        ids = [i["item_id"] for i in items]
        assert len(ids) == len(set(ids))

    def test_negative_items_not_in_corpus(self):
        items = [self._make_item("negative", i) for i in range(1, 6)]
        for item in items:
            item["is_in_corpus"] = False
        assert all(not i["is_in_corpus"] for i in items)


# ── T5.5: citation recall edge cases ─────────────────────────────────────────

class TestCitationRecallEdgeCases:
    def test_none_citations_handled(self):
        result = score_citation_recall(None, "ESG Paper", is_in_corpus=True)
        assert result["citation_recall_score"] == 0.0

    def test_title_with_special_chars(self):
        citations = [{"title": "ESG & Firm Performance: A Meta-Analysis (2021)"}]
        result = score_citation_recall(citations, "ESG & Firm Performance", is_in_corpus=True)
        assert result["citation_recall_score"] == 1.0

    def test_empty_string_title(self):
        citations = [{"title": "Some Paper"}]
        result = score_citation_recall(citations, "", is_in_corpus=True)
        # Empty expected_source_title treated as N/A
        assert "citation_recall_score" in result

    def test_multiple_citations_any_match(self):
        citations = [
            {"title": "Unrelated Paper 1"},
            {"title": "ESG and Firm Value"},
            {"title": "Unrelated Paper 2"},
        ]
        result = score_citation_recall(citations, "ESG and Firm Value", is_in_corpus=True)
        assert result["citation_recall_score"] == 1.0

    def test_off_topic_empty_citations_is_one(self):
        """If nothing is cited for an off-topic item, that's correct behaviour."""
        result = score_citation_recall([], "N/A", is_in_corpus=False)
        assert result["citation_recall_score"] == 1.0


# ── T5.6: batch_semantic_similarity ──────────────────────────────────────────

class TestBatchSemanticSimilarity:
    def test_empty_pairs(self):
        embed_model = MagicMock()
        result = batch_semantic_similarity([], embed_model)
        assert result == []

    def test_single_pair_identical(self):
        embed_model = MagicMock()
        # Same vector for both texts
        embed_model.embed = MagicMock(return_value=iter([[1.0, 0.0], [1.0, 0.0]]))
        result = batch_semantic_similarity([("a", "a")], embed_model)
        assert abs(result[0] - 1.0) < 1e-4

    def test_two_calls_to_embed(self):
        """batch_semantic_similarity must call embed once with all texts."""
        embed_model = MagicMock()
        embed_model.embed = MagicMock(return_value=iter([
            [1.0, 0.0], [0.0, 1.0],  # answers
            [1.0, 0.0], [0.0, 1.0],  # references
        ]))
        pairs = [("ans1", "ref1"), ("ans2", "ref2")]
        result = batch_semantic_similarity(pairs, embed_model)
        assert len(result) == 2
        # Called exactly once with 4 texts
        embed_model.embed.assert_called_once()
        call_texts = embed_model.embed.call_args[0][0]
        assert len(call_texts) == 4


# ── OllamaJudge unit tests ────────────────────────────────────────────────────

class TestOllamaJudge:
    def test_strips_markdown_fences(self):
        judge = OllamaJudge("http://localhost", "test-model")
        text = '```json\n{"score": 0.9}\n```'
        assert judge._strip_fences(text) == '{"score": 0.9}'

    def test_strips_plain_fences(self):
        judge = OllamaJudge("http://localhost", "test-model")
        text = '```\n{"score": 0.5}\n```'
        assert judge._strip_fences(text) == '{"score": 0.5}'

    def test_no_fences_unchanged(self):
        judge = OllamaJudge("http://localhost", "test-model")
        text = '{"score": 0.7}'
        assert judge._strip_fences(text) == '{"score": 0.7}'

    @patch("eval.metrics.requests.post")
    def test_judge_returns_parsed_json(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"faithfulness_score": 0.9, "reasoning": "good"}'}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        judge = OllamaJudge("http://localhost:11434", "test-model", api_key="tok")
        result = judge.judge("some prompt")
        assert result["faithfulness_score"] == 0.9

    @patch("eval.metrics.requests.post")
    def test_judge_retries_on_json_failure(self, mock_post):
        bad_resp = MagicMock()
        bad_resp.json.return_value = {"response": "not json at all"}
        bad_resp.raise_for_status = MagicMock()

        good_resp = MagicMock()
        good_resp.json.return_value = {"response": '{"score": 0.5}'}
        good_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [bad_resp, good_resp]
        judge = OllamaJudge("http://localhost:11434", "test-model")
        result = judge.judge("prompt", max_retries=2)
        assert result == {"score": 0.5}

    @patch("eval.metrics.requests.post")
    def test_judge_returns_empty_dict_on_all_failures(self, mock_post):
        bad_resp = MagicMock()
        bad_resp.json.return_value = {"response": "garbage"}
        bad_resp.raise_for_status = MagicMock()
        mock_post.return_value = bad_resp

        judge = OllamaJudge("http://localhost:11434", "test-model")
        result = judge.judge("prompt", max_retries=2)
        assert result == {}
