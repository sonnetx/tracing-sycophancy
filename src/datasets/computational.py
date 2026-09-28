import os
import re

from src.datasets.base import DatasetPreprocessor


class ComputationalPreprocessor(DatasetPreprocessor):
    """Preprocessor for computational (math) problems from AMPS dataset."""

    dataset_type = "factual"
    dataset_name = "computational"

    def load_raw(self, raw_path: str) -> list[dict]:
        items = []
        for root, _dirs, files in os.walk(raw_path):
            for fname in files:
                if not fname.endswith(".txt"):
                    continue
                file_path = os.path.join(root, fname)
                try:
                    question, answer = self._extract_qa(file_path)
                except ValueError:
                    continue

                parts = file_path.replace("\\", "/").split("/")
                file_id = os.path.splitext(fname)[0]
                sub_category = parts[-3] if len(parts) >= 3 else "unknown"
                subcategory = parts[-2] if len(parts) >= 2 else "unknown"

                items.append({
                    "id": f"comp_{sub_category}_{subcategory}_{file_id}",
                    "question": question,
                    "correct_answer": answer,
                    "category": sub_category,
                    "subcategory": subcategory,
                })
        return items

    @staticmethod
    def _extract_qa(file_path: str) -> tuple[str, str]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        question_match = re.search(r"Problem:\s*(.*?)\s*Answer:", content, re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.*?)$", content, re.DOTALL)
        if question_match and answer_match:
            return question_match.group(1).strip(), answer_match.group(1).strip()
        raise ValueError(f"Could not extract Q&A from {file_path}")
