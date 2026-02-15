from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """Abstract base for model inference backends.

    Supports two modes:
    - chat(): For instruct/chat models using message-based APIs.
    - complete(): For base models using raw text completion.
    """

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send chat-format messages, return response text."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Send a raw text prompt, return completion text."""

    @property
    def supports_chat(self) -> bool:
        return False

    @property
    def supports_completion(self) -> bool:
        return False

    def infer(self, *, messages: list[dict] | None = None, prompt: str | None = None, **kwargs) -> str:
        """Unified inference method that dispatches to chat() or complete()."""
        if messages is not None and prompt is not None:
            raise ValueError("Provide either messages or prompt, not both.")
        if messages is not None:
            if not self.supports_chat:
                raise RuntimeError(f"{self.__class__.__name__} does not support chat mode.")
            return self.chat(messages, **kwargs)
        if prompt is not None:
            if not self.supports_completion:
                raise RuntimeError(f"{self.__class__.__name__} does not support completion mode.")
            return self.complete(prompt, **kwargs)
        raise ValueError("Must provide either messages or prompt.")
