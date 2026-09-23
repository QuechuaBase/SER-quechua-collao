#!/usr/bin/env python3
"""Count trainable parameters for comparable SER baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ser_models import build_ser_model
from src.ser_only_models import LogMelSERModel


def count_parameters(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Save trainable parameter counts.")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--pooling", default="attentive", choices=("mean", "mean_std", "attentive"))
    parser.add_argument("--ser_only_hidden_dim", type=int, default=256)
    args = parser.parse_args()

    systems: dict[str, dict[str, object]] = {}
    for name in ("vanilla_xlsr_frozen", "puno_quechua_asr_ser_frozen"):
        systems[name] = {
            "encoder_trainable_parameters": 0,
            "encoder_frozen": True,
            "embedding_dim": args.input_dim,
            "pooling": args.pooling,
            "tasks": {
                task: count_parameters(build_ser_model(task, args.input_dim, args.pooling))
                for task in ("vad_regression", "emotion_classification")
            },
        }

    systems["ser_only_neural"] = {
        "uses_encoder_embeddings": False,
        "pooling": args.pooling,
        "hidden_dim": args.ser_only_hidden_dim,
        "tasks": {
            task: count_parameters(
                LogMelSERModel(
                    task=task,
                    hidden_dim=args.ser_only_hidden_dim,
                    pooling=args.pooling if args.pooling != "mean_std" else "attentive",
                )
            )
            for task in ("vad_regression", "emotion_classification")
        },
    }

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(systems, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Parameter counts: {output_path}")


if __name__ == "__main__":
    main()
