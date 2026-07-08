#!/usr/bin/env python3
"""Train classical eGeMAPS SVR/SVC baselines on one actor-disjoint fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.egemaps_metrics import (
    EMOTION_CLASSES,
    VAD_TARGETS,
    classification_metrics,
    vad_metrics,
)
from src.egemaps_utils import load_fold_data


REGRESSION_GRID = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", 0.01, 0.001],
    "model__epsilon": [0.05, 0.1, 0.2],
    "model__kernel": ["rbf"],
}
CLASSIFICATION_GRID = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", 0.01, 0.001],
    "model__kernel": ["rbf"],
}


def json_ready(value: Any) -> Any:
    """Convert numpy/path values recursively for JSON output."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def fixed_macro_f1_scorer(
    estimator: Pipeline,
    features: pd.DataFrame,
    targets: pd.Series,
) -> float:
    """Score all nine canonical emotions, including absent inner-fold classes."""
    predictions = estimator.predict(features)
    return float(
        f1_score(
            targets,
            predictions,
            labels=EMOTION_CLASSES,
            average="macro",
            zero_division=0,
        )
    )


def fit_estimator(
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    tune: str,
    scoring: Any,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any], float | None]:
    """Fit defaults or tune exclusively with leave-one-training-actor-out CV."""
    if tune == "none":
        pipeline.fit(x_train, y_train)
        selected_params = {key: pipeline.get_params()[key] for key in grid}
        return pipeline, selected_params, None

    unique_groups = sorted(groups.astype(str).unique().tolist())
    if len(unique_groups) < 2:
        raise ValueError("Inner actor CV requires at least two training actors")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=scoring,
        cv=LeaveOneGroupOut(),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x_train, y_train, groups=groups.astype(str))
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Emotion confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train StandardScaler + SVR/SVC using eGeMAPS features."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("vad_regression", "emotion_classification"),
    )
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--features_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--tune",
        choices=("none", "inner_actor_cv"),
        default="inner_actor_cv",
    )
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    fold_dir = args.fold_dir.expanduser().resolve()
    features_csv = args.features_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train, valid, feature_columns = load_fold_data(
        fold_dir, features_csv, args.task
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_actors = sorted(train["actor"].astype(str).unique().tolist())
    valid_actors = sorted(valid["actor"].astype(str).unique().tolist())
    config = {
        **vars(args),
        "fold_dir": str(fold_dir),
        "features_csv": str(features_csv),
        "output_dir": str(output_dir),
        "feature_count": len(feature_columns),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_actors": train_actors,
        "valid_actors": valid_actors,
        "actor_leakage_verified": True,
        "scaler_fit_scope": "training split inside each outer/inner fold",
        "uses_asr": False,
        "uses_neural_models": False,
        "emotion_classes": EMOTION_CLASSES,
        "vad_target_order": VAD_TARGETS,
    }
    save_json(output_dir / "config.json", config)

    x_train = train[feature_columns]
    x_valid = valid[feature_columns]
    groups = train["actor"]
    prediction_table: dict[str, Any] = {
        "utterance_id": valid["utterance_id"].astype(str).tolist(),
        "actor": valid["actor"].astype(str).tolist(),
    }
    if "audio_path" in valid.columns:
        prediction_table["audio_path"] = valid["audio_path"].astype(str).tolist()

    if args.task == "vad_regression":
        predictions = np.zeros((len(valid), len(VAD_TARGETS)), dtype=np.float64)
        best_params: dict[str, Any] = {}
        for index, target in enumerate(VAD_TARGETS):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            )
            estimator, params, inner_score = fit_estimator(
                pipeline=pipeline,
                grid=REGRESSION_GRID,
                x_train=x_train,
                y_train=train[target],
                groups=groups,
                tune=args.tune,
                scoring="neg_mean_squared_error",
                n_jobs=args.n_jobs,
            )
            predictions[:, index] = estimator.predict(x_valid)
            joblib.dump(estimator, output_dir / f"model_{target}.joblib")
            best_params[target] = {
                "parameters": params,
                "inner_cv_best_neg_mse": inner_score,
            }
            prediction_table[f"true_{target}"] = valid[target].to_numpy()
            prediction_table[f"pred_{target}"] = predictions[:, index]

        metrics = vad_metrics(valid[VAD_TARGETS].to_numpy(), predictions)
        save_json(output_dir / "best_params.json", best_params)
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        estimator, params, inner_score = fit_estimator(
            pipeline=pipeline,
            grid=CLASSIFICATION_GRID,
            x_train=x_train,
            y_train=train["emotion"],
            groups=groups,
            tune=args.tune,
            scoring=fixed_macro_f1_scorer,
            n_jobs=args.n_jobs,
        )
        predictions = estimator.predict(x_valid)
        joblib.dump(estimator, output_dir / "model.joblib")
        metrics = classification_metrics(valid["emotion"], predictions)
        save_json(
            output_dir / "best_params.json",
            {
                "parameters": params,
                "inner_cv_best_macro_f1": inner_score,
            },
        )
        prediction_table["true_emotion"] = valid["emotion"].tolist()
        prediction_table["pred_emotion"] = predictions.tolist()
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        pd.DataFrame(
            matrix,
            index=EMOTION_CLASSES,
            columns=EMOTION_CLASSES,
        ).to_csv(output_dir / "confusion_matrix.csv", index_label="true\\pred")
        plot_confusion_matrix(
            matrix,
            EMOTION_CLASSES,
            output_dir / "confusion_matrix.png",
        )

    pd.DataFrame(prediction_table).to_csv(
        output_dir / "valid_predictions.csv",
        index=False,
    )
    save_json(output_dir / "metrics.json", metrics)
    print(f"Task: {args.task}")
    print(f"Train actors: {train_actors}")
    print(f"Validation actors: {valid_actors}")
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
