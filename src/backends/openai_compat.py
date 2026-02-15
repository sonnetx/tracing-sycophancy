import os
import time

from openai import OpenAI

from src.backends.base import ModelBackend


class OpenAICompatBackend(ModelBackend):
    """Backend for OpenAI-compatible APIs (OpenAI, vLLM, Together AI, etc.)."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 3, retry_delay: float = 5.0, **default_kwargs):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_kwargs = default_kwargs
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key, base_url=base_url)

    def chat(self, messages: list[dict], **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **merged)
                return response.choices[0].message.content
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    print(f"Rate limit hit, retrying in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    raise

    def complete(self, prompt: str, **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        merged.pop("response_format", None)
        for attempt in range(self.max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model, prompt=prompt, **merged)
                return response.choices[0].text
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    print(f"Rate limit hit, retrying in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    raise

    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_completion(self) -> bool:
        return True
