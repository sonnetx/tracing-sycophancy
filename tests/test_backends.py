"""Tests for src/backends/ — ModelBackend interface and concrete backends."""

import pytest

from src.backends.base import ModelBackend


# ---------------------------------------------------------------------------
# A concrete stub for testing the base class logic
# ---------------------------------------------------------------------------

class ChatOnlyBackend(ModelBackend):
    """Stub that only supports chat."""

    def chat(self, messages, **kwargs):
        return f"chat:{messages[0]['content']}"

    def complete(self, prompt, **kwargs):
        raise RuntimeError("No completion support")

    @property
    def supports_chat(self):
        return True


class CompletionOnlyBackend(ModelBackend):
    """Stub that only supports completion."""

    def chat(self, messages, **kwargs):
        raise RuntimeError("No chat support")

    def complete(self, prompt, **kwargs):
        return f"complete:{prompt}"

    @property
    def supports_completion(self):
        return True


class DualBackend(ModelBackend):
    """Stub that supports both modes."""

    def chat(self, messages, **kwargs):
        return f"chat:{messages[0]['content']}"

    def complete(self, prompt, **kwargs):
        return f"complete:{prompt}"

    @property
    def supports_chat(self):
        return True

    @property
    def supports_completion(self):
        return True


# ---------------------------------------------------------------------------
# ModelBackend.infer() dispatcher tests
# ---------------------------------------------------------------------------

class TestInfer:
    def test_infer_chat(self):
        backend = DualBackend()
        result = backend.infer(messages=[{"role": "user", "content": "hello"}])
        assert result == "chat:hello"

    def test_infer_complete(self):
        backend = DualBackend()
        result = backend.infer(prompt="hello")
        assert result == "complete:hello"

    def test_infer_both_raises(self):
        backend = DualBackend()
        with pytest.raises(ValueError, match="not both"):
            backend.infer(messages=[{"role": "user", "content": "a"}], prompt="b")

    def test_infer_neither_raises(self):
        backend = DualBackend()
        with pytest.raises(ValueError, match="Must provide"):
            backend.infer()

    def test_infer_chat_unsupported(self):
        backend = CompletionOnlyBackend()
        with pytest.raises(RuntimeError, match="does not support chat"):
            backend.infer(messages=[{"role": "user", "content": "a"}])

    def test_infer_completion_unsupported(self):
        backend = ChatOnlyBackend()
        with pytest.raises(RuntimeError, match="does not support completion"):
            backend.infer(prompt="a")


# ---------------------------------------------------------------------------
# Concrete backend property checks (no network calls)
# ---------------------------------------------------------------------------

class TestOllamaBackendProperties:
    def test_supports_both(self):
        from src.backends.ollama import OllamaBackend
        backend = OllamaBackend(model="test")
        assert backend.supports_chat is True
        assert backend.supports_completion is True
        assert backend.model == "test"


class TestOpenAICompatBackendProperties:
    def test_supports_both(self):
        from src.backends.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(model="gpt-4o", api_key="test")
        assert backend.supports_chat is True
        assert backend.supports_completion is True
        assert backend.model == "gpt-4o"

    def test_custom_base_url(self):
        from src.backends.openai_compat import OpenAICompatBackend
        backend = OpenAICompatBackend(
            model="llama", api_key="unused", base_url="http://localhost:8000/v1")
        assert backend.model == "llama"


class TestAnthropicBackendProperties:
    def test_chat_only(self):
        from src.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(model="claude-sonnet-4-20250514", api_key="test")
        assert backend.supports_chat is True
        assert backend.supports_completion is False

    def test_complete_raises(self):
        from src.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(model="claude-sonnet-4-20250514", api_key="test")
        with pytest.raises(RuntimeError, match="does not support"):
            backend.complete("hello")
