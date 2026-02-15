from src.datasets.base import DatasetPreprocessor


class ImplicitPreprocessor(DatasetPreprocessor):
    """Placeholder preprocessor for implicit sycophancy datasets."""

    dataset_type = "implicit"
    dataset_name = "implicit"

    def load_raw(self, raw_path: str) -> list[dict]:
        raise NotImplementedError(
            "Implicit sycophancy dataset preprocessing is not yet implemented."
        )
