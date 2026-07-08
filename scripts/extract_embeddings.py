#!/usr/bin/env python3
"""Extract embeddings from one short audio file for a lightweight smoke test."""

from __future__ import annotations

import argparse
import sys
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


def load_audio(path: Path, target_rate: int = 16000):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        audio = resample_poly(
            audio, target_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    return audio


def load_exported_encoder(path: Path, device: str, torch: Any):
    try:
        encoder = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        encoder = torch.load(path, map_location=device)
    if not hasattr(encoder, "eval"):
        raise RuntimeError("The exported file does not contain a PyTorch module.")
    return encoder.eval().to(device)


def load_checkpoint_encoder(
    checkpoint_path: Path, data_dir: Path, device: str
):
    from fairseq import checkpoint_utils

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint_path)],
        arg_overrides={"data": str(data_dir)},
    )
    if not models:
        raise RuntimeError("Fairseq returned no models.")
    model = models[0].eval().to(device)
    try:
        return model.w2v_encoder.w2v_model.eval(), model
    except AttributeError as exc:
        raise RuntimeError(
            "Could not find model.w2v_encoder.w2v_model in the checkpoint."
        ) from exc


def forward_encoder(encoder: Any, source: Any, padding_mask: Any):
    attempts = [
        lambda: encoder(
            source=source,
            padding_mask=padding_mask,
            mask=False,
            features_only=True,
        ),
        lambda: encoder(
            source=source,
            padding_mask=padding_mask,
            mask=False,
        ),
        lambda: encoder(source, padding_mask=padding_mask, mask=False),
    ]
    errors = []
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict):
                for key in ("x", "encoder_out", "features"):
                    if key in result:
                        hidden = result[key]
                        if key == "encoder_out" and hidden.ndim == 3:
                            hidden = hidden.transpose(0, 1)
                        return hidden
            if isinstance(result, tuple):
                return result[0]
            if hasattr(result, "last_hidden_state"):
                return result.last_hidden_state
            return result
        except (TypeError, KeyError) as exc:
            errors.append(str(exc))
    raise RuntimeError("Encoder forward failed: " + " | ".join(errors))


def apply_pooling(hidden: Any, mode: str):
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    if mode == "none":
        return hidden
    if mode == "mean":
        return hidden.mean(dim=1)
    if mode == "max":
        return hidden.max(dim=1).values
    raise ValueError(f"Unsupported pooling: {mode}")


def save_output(tensor: Any, path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        torch.save(tensor, path)
    elif suffix == ".npy":
        import numpy as np

        np.save(path, tensor.numpy())
    else:
        raise RuntimeError("--output_path must end in .pt or .npy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from one short audio file, not a batch pipeline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint_path", type=Path)
    source.add_argument("--encoder_path", type=Path)
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--pooling", choices=("mean", "max", "none"), default="mean"
    )
    parser.add_argument(
        "--fairseq_data_dir",
        type=Path,
        default=DEFAULT_FAIRSEQ_DATA,
        help="Directory containing dict.ltr.txt when loading a full checkpoint.",
    )
    args = parser.parse_args()

    audio_path = args.audio_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"Audio file does not exist: {audio_path}")

    try:
        import torch

        device = select_device(args.device, torch)
        audio = load_audio(audio_path)
        source_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        padding_mask = torch.zeros_like(source_tensor, dtype=torch.bool)
        owner = None
        if args.encoder_path:
            encoder_path = args.encoder_path.expanduser().resolve()
            if not encoder_path.is_file():
                parser.error(f"Encoder file does not exist: {encoder_path}")
            encoder = load_exported_encoder(encoder_path, device, torch)
        else:
            checkpoint_path = args.checkpoint_path.expanduser().resolve()
            data_dir = args.fairseq_data_dir.expanduser().resolve()
            if not checkpoint_path.is_file():
                parser.error(f"Checkpoint file does not exist: {checkpoint_path}")
            if not (data_dir / "dict.ltr.txt").is_file():
                parser.error(f"Fairseq dictionary not found in: {data_dir}")
            encoder, owner = load_checkpoint_encoder(
                checkpoint_path, data_dir, device
            )

        with torch.inference_mode():
            hidden = forward_encoder(encoder, source_tensor, padding_mask)
            embedding = apply_pooling(hidden, args.pooling).detach().cpu()
        save_output(embedding, output_path, torch)
        print(f"Device: {device}")
        print(f"Audio samples at 16 kHz: {len(audio):,}")
        print(f"Embedding shape: {tuple(embedding.shape)}")
        print(f"Saved to: {output_path}")
        del owner
    except ImportError as exc:
        print(
            "ERROR: missing dependency. Install requirements in a Python 3.10 environment.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"ERROR: embedding extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
