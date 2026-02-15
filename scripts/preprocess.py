#!/usr/bin/env python3
"""Step 1: Preprocess raw datasets into standardized JSONL format.

Usage:
    python scripts/preprocess.py --dataset computational --raw-path data/raw/computational/amps --output data/processed/computational.jsonl --sample-size 500
    python scripts/preprocess.py --dataset medical_advice --raw-path data/raw/medical_advice/formatted_selected_lines_train.csv --output data/processed/medical_advice.jsonl
"""

import argparse
import sys


PREPROCESSORS = {
    "computational": "src.datasets.computational.ComputationalPreprocessor",
    "medical_advice": "src.datasets.medical_advice.MedicalAdvicePreprocessor",
    "opinion": "src.datasets.opinion.OpinionPreprocessor",
    "implicit": "src.datasets.implicit.ImplicitPreprocessor",
}


def get_preprocessor(dataset_name: str):
    if dataset_name not in PREPROCESSORS:
        print(f"Unknown dataset: {dataset_name}. Available: {list(PREPROCESSORS.keys())}")
        sys.exit(1)

    module_path, class_name = PREPROCESSORS[dataset_name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw datasets to JSONL")
    parser.add_argument("--dataset", required=True, choices=list(PREPROCESSORS.keys()),
                        help="Dataset to preprocess")
    parser.add_argument("--raw-path", required=True,
                        help="Path to raw data (directory for computational, CSV for medical_advice)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of items to randomly sample (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    preprocessor = get_preprocessor(args.dataset)
    preprocessor.preprocess(args.raw_path, args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
