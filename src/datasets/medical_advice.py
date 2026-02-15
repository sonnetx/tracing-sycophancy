import pandas as pd

from src.datasets.base import DatasetPreprocessor


class MedicalAdvicePreprocessor(DatasetPreprocessor):
    """Preprocessor for medical advice Q&A dataset (CSV with Question, Answer, qtype)."""

    dataset_type = "factual"
    dataset_name = "medical_advice"

    def load_raw(self, raw_path: str) -> list[dict]:
        df = pd.read_csv(raw_path)
        items = []
        for idx, row in df.iterrows():
            items.append({
                "id": f"med_{row.qtype}_{idx}",
                "question": str(row.Question),
                "correct_answer": str(row.Answer),
                "category": str(row.qtype),
                "subcategory": None,
            })
        return items
