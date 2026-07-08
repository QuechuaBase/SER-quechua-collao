#!/usr/bin/env python3
"""Inspect checkpoint structure without running inference or training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def human_size(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def compact(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in list(value.items())[:30]}
    if hasattr(value, "keys"):
        try:
            return {str(k): compact(value[k]) for k in list(value.keys())[:30]}
        except Exception:
            pass
    return str(value)[:500]


def inspect_pt(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect .pt/.bin checkpoints.") from exc

    load_kwargs = {"map_location": "meta"}
    try:
        checkpoint = torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(path, **load_kwargs)
    if not isinstance(checkpoint, dict):
        return "PyTorch serialized object (unknown model format)", {
            "python_type": type(checkpoint).__name__
        }

    keys = [str(key) for key in checkpoint.keys()]
    metadata: dict[str, Any] = {"top_level_keys": keys}
    for key in ("cfg", "args", "extra_state", "optimizer_history"):
        if key in checkpoint:
            metadata[key] = compact(checkpoint[key])

    model_state = checkpoint.get("model")
    if isinstance(model_state, dict):
        state_keys = list(model_state.keys())
        metadata["model_tensor_count"] = len(state_keys)
        metadata["model_key_examples"] = state_keys[:20]
        if any(key.startswith("w2v_encoder.") for key in state_keys):
            return "Fairseq wav2vec2/XLS-R CTC checkpoint", metadata
        return "Fairseq-like checkpoint with a model state dictionary", metadata
    if "state_dict" in checkpoint:
        return "Generic PyTorch/Lightning checkpoint", metadata
    return "Unknown PyTorch checkpoint", metadata


def inspect_directory(path: Path) -> tuple[str, dict[str, Any]]:
    names = {item.name for item in path.iterdir() if item.is_file()}
    transformer_markers = {
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "preprocessor_config.json",
    }
    metadata = {"files": sorted(names)}
    if "config.json" in names and names.intersection(transformer_markers - {"config.json"}):
        return "Transformers model directory", metadata
    if "checkpoint_best.pt" in names:
        return "Directory containing a probable Fairseq checkpoint", metadata
    return "Unknown model directory", metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an ASR checkpoint's format and lightweight metadata."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    args = parser.parse_args()
    path = args.checkpoint_path.expanduser().resolve()
    if not path.exists():
        parser.error(f"Checkpoint does not exist: {path}")

    try:
        if path.is_dir():
            model_type, metadata = inspect_directory(path)
            print(f"Path: {path}")
        else:
            size = path.stat().st_size
            print(f"Path: {path}")
            print(f"Size: {human_size(size)} ({size:,} bytes)")
            if path.suffix.lower() in {".pt", ".bin"}:
                model_type, metadata = inspect_pt(path)
            elif path.suffix.lower() == ".safetensors":
                model_type = "Transformers/safetensors weights"
                metadata = {}
            else:
                model_type = "Unknown file format"
                metadata = {}
        print(f"Detected type: {model_type}")
        print("Metadata:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
        print(f"\nConclusion: {model_type}. No training or inference was executed.")
    except Exception as exc:
        print(f"ERROR: checkpoint inspection failed: {exc}", file=sys.stderr)
        print(
            "The file may require the same Python/Fairseq versions used to create it.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
