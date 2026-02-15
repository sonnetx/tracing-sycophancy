from src.datasets.base import DatasetPreprocessor


class OpinionPreprocessor(DatasetPreprocessor):
    """Placeholder preprocessor for opinion sycophancy datasets."""

    dataset_type = "opinion"
    dataset_name = "opinion"

    def load_raw(self, raw_path: str) -> list[dict]:
        raise NotImplementedError(
            "Opinion sycophancy dataset preprocessing is not yet implemented."
        )
