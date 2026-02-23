import pandas as pd

from src.datasets.base import DatasetPreprocessor


class MedicalAdvicePreprocessor(DatasetPreprocessor):
    """Preprocessor for medical advice Q&A dataset (CSV with Question, Answer, qtype)."""

    dataset_type = "factual"
    dataset_name = "medical_advice"

    def load_raw(self, raw_path: str) -> list[dict]:
        import os
        # raw_path may be the directory; find the CSV inside it
        if os.path.isdir(raw_path):
            csv_path = os.path.join(raw_path, "CSV", "formatted_selected_lines_train.csv")
        else:
            csv_path = raw_path
        df = pd.read_csv(csv_path)
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
