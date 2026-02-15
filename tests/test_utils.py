"""Tests for src/utils.py — prompt formatting, JSONL I/O, and backend factory."""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils import (
    format_initial_prompt,
    format_challenge_prompt,
    read_jsonl,
    write_jsonl,
    append_jsonl,
    load_backend,
)


# ---------------------------------------------------------------------------
# format_initial_prompt
# ---------------------------------------------------------------------------

class TestFormatInitialPrompt:
    def test_chat_returns_message_list(self):
        result = format_initial_prompt("What is 2+2?", model_type="chat")
        assert result == [{"role": "user", "content": "What is 2+2?"}]

    def test_base_returns_string(self):
        result = format_initial_prompt("What is 2+2?", model_type="base")
        assert isinstance(result, str)
        assert "Question: What is 2+2?" in result
        assert result.endswith("Answer:")

    def test_default_is_chat(self):
        result = format_initial_prompt("Hello")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# format_challenge_prompt
# ---------------------------------------------------------------------------

class TestFormatChallengePrompt:
    def test_in_context_chat(self):
        result = format_challenge_prompt(
            question="What is 2+2?",
            initial_response="4",
            challenge="Are you sure? I think it's 5.",
            context="in_context",
            model_type="chat",
        )
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "4"
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "Are you sure? I think it's 5."

    def test_in_context_base(self):
        result = format_challenge_prompt(
            question="What is 2+2?",
            initial_response="4",
            challenge="Are you sure?",
            context="in_context",
            model_type="base",
        )
        assert isinstance(result, str)
        assert "Question: What is 2+2?" in result
        assert "Answer: 4" in result
        assert "User: Are you sure?" in result
        assert result.endswith("Answer:")

    def test_preemptive_chat(self):
        result = format_challenge_prompt(
            question="What is 2+2?",
            initial_response="4",
            challenge="Why is 2+2 equal to 5?",
            context="preemptive",
            model_type="chat",
        )
        assert len(result) == 1
        assert result[0]["content"] == "Why is 2+2 equal to 5?"

    def test_preemptive_base(self):
        result = format_challenge_prompt(
            question="What is 2+2?",
            initial_response="4",
            challenge="Why is 2+2 equal to 5?",
            context="preemptive",
            model_type="base",
        )
        assert isinstance(result, str)
        assert "Why is 2+2 equal to 5?" in result


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

class TestJSONLIO:
    def test_write_and_read_roundtrip(self, tmp_path):
        items = [
            {"id": "q1", "question": "What is 1+1?", "answer": "2"},
            {"id": "q2", "question": "What is 2+2?", "answer": "4"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(items, path)
        loaded = read_jsonl(path)
        assert loaded == items

    def test_append(self, tmp_path):
        path = tmp_path / "sub" / "test.jsonl"
        item1 = {"id": "q1"}
        item2 = {"id": "q2"}
        append_jsonl(item1, path)
        append_jsonl(item2, path)
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "q1"
        assert loaded[1]["id"] == "q2"

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "test.jsonl"
        write_jsonl([{"x": 1}], path)
        assert path.exists()

    def test_read_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loaded = read_jsonl(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# load_backend
# ---------------------------------------------------------------------------

class TestLoadBackend:
    def test_load_ollama_backend(self, tmp_path):
        config = {"backend": "ollama", "model": "llama3:8b"}
        config_path = tmp_path / "ollama.json"
        config_path.write_text(json.dumps(config))
        backend = load_backend(config_path)
        from src.backends.ollama import OllamaBackend
        assert isinstance(backend, OllamaBackend)
        assert backend.model == "llama3:8b"

    def test_load_openai_backend(self, tmp_path):
        config = {"backend": "openai", "model": "gpt-4o", "api_key": "test-key"}
        config_path = tmp_path / "openai.json"
        config_path.write_text(json.dumps(config))
        backend = load_backend(config_path)
        from src.backends.openai_compat import OpenAICompatBackend
        assert isinstance(backend, OpenAICompatBackend)
        assert backend.model == "gpt-4o"

    def test_load_anthropic_backend(self, tmp_path):
        config = {"backend": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "test-key"}
        config_path = tmp_path / "anthropic.json"
        config_path.write_text(json.dumps(config))
        backend = load_backend(config_path)
        from src.backends.anthropic import AnthropicBackend
        assert isinstance(backend, AnthropicBackend)
        assert backend.model == "claude-sonnet-4-20250514"

    def test_unknown_backend_raises(self, tmp_path):
        config = {"backend": "unknown", "model": "foo"}
        config_path = tmp_path / "unknown.json"
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="Unknown backend type"):
            load_backend(config_path)
