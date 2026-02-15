import ollama

from src.backends.base import ModelBackend


class OllamaBackend(ModelBackend):
    """Backend for Ollama models, supporting both chat and text completion."""

    def __init__(self, model: str, host: str | None = None, **default_kwargs):
        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.default_kwargs = default_kwargs

    def chat(self, messages: list[dict], **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        response = self.client.chat(model=self.model, messages=messages, **merged)
        return response["message"]["content"]

    def complete(self, prompt: str, **kwargs) -> str:
        merged = {**self.default_kwargs, **kwargs}
        response = self.client.generate(model=self.model, prompt=prompt, **merged)
        return response["response"]

    @property
    def supports_chat(self) -> bool:
        return True

    @property
    def supports_completion(self) -> bool:
        return True
