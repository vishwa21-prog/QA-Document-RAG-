#!/usr/bin/env python
"""
CLI helper to run the evaluation harness and pretty-print the results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --dataset data/eval/qa_dataset.json --top-k 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation import run_evaluation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--dataset", default="data/eval/qa_dataset.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--verbose", action="store_true", help="Print per-question results.")
    args = parser.parse_args()

    results = run_evaluation(args.dataset, top_k=args.top_k)

    print("=== Evaluation summary ===")
    for k, v in results["summary"].items():
        print(f"{k}: {v}")

    if args.verbose:
        print("\n=== Per-question detail ===")
        print(json.dumps(results["per_question"], indent=2))


if __name__ == "__main__":
    main()
