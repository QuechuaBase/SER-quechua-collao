"""Plot four-system fold results without hardcoded scientific values."""

import argparse
import csv
import math
from pathlib import Path


def read_folds(path, metric):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sorted(int(row["fold"]) for row in rows) != list(range(1, 7)):
        raise ValueError(f"Expected exactly folds 1 through 6: {path}")
    values = [float(row[metric]) for row in sorted(rows, key=lambda r: int(r["fold"]))]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite values: {path}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [
        ("ASR-SER", "", "#0072B2", "-"),
        ("Vanilla XLS-R", "vanilla_xlsr/", "#D55E00", "--"),
        ("SER-only", "ser_only_", "#009E73", "-."),
        ("eGeMAPS", "egemaps_", "#CC79A7", ":"),
    ]
    angles = [2 * math.pi * i / 6 for i in range(6)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, metric, title in [
        ("vad", "mean_ccc", "VAD: mean CCC per fold"),
        ("emotion", "macro_f1", "Categorical SER: macro-F1 per fold"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": "polar"})
        for label, prefix, color, style in systems:
            values = read_folds(args.results_dir / f"{prefix}{task}_cv_summary_folds.csv", metric)
            ax.plot(angles + angles[:1], values + values[:1], label=label,
                    color=color, linestyle=style, marker="o")
        ax.set_xticks(angles, [f"Fold {i}" for i in range(1, 7)])
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1))
        fig.savefig(args.output_dir / f"{task}_fold_radar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
