import os
import time

from anthropic import Anthropic

from src.backends.base import ModelBackend


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic Claude models. Chat-only (primarily for LLM judge)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None,
                 max_retries: int = 3, retry_delay: float = 5.0, **default_kwargs):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_kwargs = {"max_tokens": 2000, **default_kwargs}
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=key)

    def chat(self, messages: list[dict], **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        system_msg = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered.append(msg)

        create_kwargs = {"model": self.model, "messages": filtered, **merged}
        if system_msg:
            create_kwargs["system"] = system_msg

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**create_kwargs)
                return response.content[0].text
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    print(f"Rate limit hit, retrying in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    raise

    def complete(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("Anthropic Claude does not support raw text completion.")

    @property
    def supports_chat(self) -> bool:
        return True
