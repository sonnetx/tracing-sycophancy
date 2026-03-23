"""Tests for src/evaluation/ — hedging, refusal, and judge parsing."""

import json

import pytest

from src.evaluation.hedging import detect_hedging, HEDGING_PHRASES
from src.evaluation.refusal import detect_refusal, REFUSAL_PHRASES
from src.evaluation.judge import _parse_response


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

class TestHedging:
    def test_no_hedging(self):
        result = detect_hedging("The answer is 42.")
        assert result["hedging"] is False
        assert result["hedging_phrases"] == []

    def test_single_hedging_phrase(self):
        result = detect_hedging("You may be right, let me reconsider the problem.")
        assert result["hedging"] is True
        assert "you may be right" in result["hedging_phrases"]

    def test_multiple_hedging_phrases(self):
        result = detect_hedging(
            "That's a good point. I stand corrected. You are right.")
        assert result["hedging"] is True
        assert len(result["hedging_phrases"]) >= 2

    def test_case_insensitive(self):
        result = detect_hedging("YOU MAY BE RIGHT about that.")
        assert result["hedging"] is True

    def test_empty_string(self):
        result = detect_hedging("")
        assert result["hedging"] is False


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

class TestRefusal:
    def test_no_refusal(self):
        result = detect_refusal("The answer to your question is 7.")
        assert result["refusal"] is False

    def test_ai_disclaimer(self):
        result = detect_refusal("As an AI, I don't have personal opinions.")
        assert result["refusal"] is True

    def test_unable_to_answer(self):
        result = detect_refusal("I'm not able to provide medical advice.")
        assert result["refusal"] is True

    def test_language_model_disclaimer(self):
        result = detect_refusal("As a language model, I can't make that determination.")
        assert result["refusal"] is True

    def test_case_insensitive(self):
        result = detect_refusal("AS AN AI, I cannot answer that.")
        assert result["refusal"] is True

    def test_empty_string(self):
        result = detect_refusal("")
        assert result["refusal"] is False


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------

class TestJudgeParsing:
    def test_valid_json_correct(self):
        raw = json.dumps({"factual_accuracy": "correct"})
        result = _parse_response(raw)
        assert result["factual_accuracy"] == "correct"

    def test_valid_json_incorrect(self):
        raw = json.dumps({"factual_accuracy": "incorrect"})
        result = _parse_response(raw)
        assert result["factual_accuracy"] == "incorrect"

    def test_valid_json_erroneous(self):
        raw = json.dumps({"factual_accuracy": "erroneous"})
        result = _parse_response(raw)
        assert result["factual_accuracy"] == "erroneous"

    def test_fallback_on_invalid_json_incorrect(self):
        raw = "The answer is incorrect based on the ground truth."
        result = _parse_response(raw)
        assert result["factual_accuracy"] == "incorrect"

    def test_fallback_on_invalid_json_erroneous(self):
        raw = "This response is erroneous and off-topic."
        result = _parse_response(raw)
        assert result["factual_accuracy"] == "erroneous"

    def test_fallback_on_invalid_json_defaults_correct(self):
        raw = "The model gave a good answer that matches."
        result = _parse_response(raw)
        assert result["factual_accuracy"] == "correct"
