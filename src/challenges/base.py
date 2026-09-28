from abc import ABC, abstractmethod

from src.backends.base import ModelBackend


class ChallengeGenerator(ABC):
    """Abstract base for generating sycophancy challenge prompts."""

    @abstractmethod
    def generate(self, item: dict, backend: ModelBackend | None = None) -> dict:
        """Add challenges to a preprocessed question item."""
