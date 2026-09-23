"""Check published rounding, fold aggregation, and pooled per-class metrics."""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    expected = {
        "": [(0.667, .089), (.541, .142), (.695, .063), (.634, .091),
             (.449, .083), (.450, .085), (.422, .097)],
        "egemaps_": [(.595, .117), (.219, .075), (.587, .104), (.467, .077),
                     (.287, .071), (.289, .074), (.244, .080)],
        "ser_only_": [(.608, .057), (.393, .175), (.596, .072), (.533, .079),
                      (.361, .059), (.363, .062), (.317, .071)],
        "vanilla_xlsr/": [(.673, .050), (.506, .131), (.677, .035), (.619, .057),
                          (.378, .084), (.378, .087), (.344, .093)],
    }
    metrics = ["ccc_arousal", "ccc_valence", "ccc_dominance", "mean_ccc",
               "accuracy", "balanced_accuracy", "macro_f1"]
    for prefix, targets in expected.items():
        summaries = {}
        for task in ("vad", "emotion"):
            path = ROOT / "results" / f"{prefix}{task}_cv_summary.csv"
            rows = read(path)
            summaries.update({row["metric"]: row for row in rows})
            folds = read(path.with_name(path.stem + "_folds.csv"))
            assert sorted(int(row["fold"]) for row in folds) == list(range(1, 7))
            for row in rows:
                values = [float(fold[row["metric"]]) for fold in folds]
                assert abs(statistics.mean(values) - float(row["mean"])) < 1e-6
                assert abs(statistics.stdev(values) - float(row["std"])) < 1e-6
        for metric, target in zip(metrics, targets):
            actual = tuple(round(float(summaries[metric][key]), 3) for key in ("mean", "std"))
            assert actual == target, (prefix, metric, actual, target)
    for task, metric, target in [
        ("vad", "mean_ccc", [.637, .529, .579, .622, .688, .656]),
        ("emotion", "macro_f1", [.374, .311, .248, .337, .512, .284]),
    ]:
        rows = read(ROOT / "results/vanilla_xlsr" / f"{task}_cv_summary_folds.csv")
        assert [round(float(row[metric]), 3) for row in rows] == target
    path = ROOT / "results/vanilla_xlsr/emotion_cv_confusion_matrix.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    matrix = [[int(v) for v in row[1:]] for row in rows[1:]]
    expected_f1 = [.654, .296, .432, .432, .244, .477, .349, .215, .236]
    for i, expected_value in enumerate(expected_f1):
        score = 2 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix))
        assert round(score, 3) == expected_value
    print("All four systems, sample standard deviations, vanilla folds and pooled per-class F1 verified.")


if __name__ == "__main__":
    main()
