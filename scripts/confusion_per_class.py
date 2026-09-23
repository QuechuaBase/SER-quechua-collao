"""Derive per-class F1 from an aggregated confusion matrix."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        table = list(csv.reader(handle))
    labels = table[0][1:]
    if [row[0] for row in table[1:]] != labels:
        raise ValueError("Row and column labels must match")
    matrix = [[int(value) for value in row[1:]] for row in table[1:]]
    if any(len(row) != len(labels) or any(v < 0 for v in row) for row in matrix):
        raise ValueError("Expected a square matrix of nonnegative counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "f1", "aggregation"])
        for i, label in enumerate(labels):
            denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
            score = 2 * matrix[i][i] / denominator if denominator else 0.0
            writer.writerow([label, score, "aggregated_confusion_matrix"])


if __name__ == "__main__":
    main()
