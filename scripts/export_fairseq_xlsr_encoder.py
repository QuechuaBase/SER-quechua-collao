#!/usr/bin/env python3
"""Export the acoustic wav2vec2/XLS-R module from a Fairseq CTC checkpoint."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAIRSEQ_DATA = PROJECT_ROOT / "resources" / "qxp_v2"


def select_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return requested


def jsonable_config(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=False)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): jsonable_config(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def find_encoder(model: Any) -> tuple[Any, str, list[str]]:
    candidates = [
        ("model.w2v_encoder.w2v_model", lambda: model.w2v_encoder.w2v_model),
        ("model.w2v_encoder", lambda: model.w2v_encoder),
        ("model.encoder.w2v_model", lambda: model.encoder.w2v_model),
    ]
    errors: list[str] = []
    for name, getter in candidates:
        try:
            encoder = getter()
            return encoder, name, errors
        except AttributeError:
            errors.append(f"Missing {name}")
    raise RuntimeError(
        "Could not locate a wav2vec2/XLS-R acoustic module. " + "; ".join(errors)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Fairseq wav2vec2/XLS-R acoustic encoder without its CTC head."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--fairseq_data_dir",
        type=Path,
        default=DEFAULT_FAIRSEQ_DATA,
        help="Directory containing dict.ltr.txt for the Fairseq CTC task.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_dir = args.fairseq_data_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint file does not exist: {checkpoint_path}")
    if not (data_dir / "dict.ltr.txt").is_file():
        parser.error(f"Fairseq dictionary not found: {data_dir / 'dict.ltr.txt'}")

    try:
        import torch
        from fairseq import checkpoint_utils

        device = select_device(args.device, torch)
        print(f"Loading Fairseq checkpoint on {device}: {checkpoint_path}")
        models, saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint_path)],
            arg_overrides={"data": str(data_dir)},
        )
        if not models:
            raise RuntimeError("Fairseq returned no models.")
        full_model = models[0].eval().to(device)
        encoder, extracted_from, warnings = find_encoder(full_model)
        if extracted_from != "model.w2v_encoder.w2v_model":
            warnings.append(
                "The innermost acoustic module was not found; the export may include "
                "an adapter or projection associated with CTC."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        encoder = encoder.cpu().eval()
        torch.save(encoder, output_dir / "encoder.pt")

        encoder_cfg = {
            "model_type": "fairseq_wav2vec2_xlsr",
            "sample_rate": 16000,
            "extracted_from": extracted_from,
            "fairseq_config": jsonable_config(saved_cfg),
        }
        (output_dir / "encoder_config.json").write_text(
            json.dumps(encoder_cfg, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "original_checkpoint": str(checkpoint_path),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "detected_model_type": "Fairseq wav2vec2/XLS-R CTC",
            "extracted_from": extracted_from,
            "ctc_head_removed": extracted_from == "model.w2v_encoder.w2v_model",
            "device_used_for_loading": device,
            "fairseq_data_dir": str(data_dir),
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "warnings": warnings,
            "security_note": (
                "encoder.pt is a Python pickle-backed PyTorch module. Load only "
                "artifacts generated from trusted checkpoints."
            ),
        }
        (output_dir / "export_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Encoder saved to: {output_dir / 'encoder.pt'}")
        print(f"Configuration saved to: {output_dir / 'encoder_config.json'}")
        print(f"Metadata saved to: {output_dir / 'export_metadata.json'}")
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    except ImportError as exc:
        print(
            "ERROR: Fairseq/PyTorch is not installed correctly. Use Python 3.10 "
            "and install the versions documented in README.md.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"ERROR: could not export the encoder: {exc}", file=sys.stderr)
        print(
            "Check Fairseq compatibility and verify --fairseq_data_dir contains "
            "dict.ltr.txt.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
