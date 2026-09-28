import random
from abc import ABC, abstractmethod

from src.utils import write_jsonl


class DatasetPreprocessor(ABC):
    """Abstract base for converting raw datasets to standardized JSONL format."""

    dataset_type: str  # "factual", "opinion", "implicit"
    dataset_name: str  # "computational", "medical_advice", etc.

    @abstractmethod
    def load_raw(self, raw_path: str) -> list[dict]:
        """Load and parse raw data into a list of question dicts."""

    def sample(self, data: list[dict], n: int, seed: int = 42) -> list[dict]:
        rng = random.Random(seed)
        if n >= len(data):
            return data
        return rng.sample(data, n)

    def to_jsonl(self, data: list[dict], output_path: str) -> None:
        items = []
        for item in data:
            item.setdefault("dataset_type", self.dataset_type)
            item.setdefault("dataset_name", self.dataset_name)
            item.setdefault("challenges", [])
            items.append(item)
        write_jsonl(items, output_path)

    def preprocess(self, raw_path: str, output_path: str, sample_size: int | None = None, seed: int = 42) -> None:
        data = self.load_raw(raw_path)
        if sample_size is not None:
            data = self.sample(data, sample_size, seed)
        self.to_jsonl(data, output_path)
        print(f"Wrote {len(data)} items to {output_path}")
