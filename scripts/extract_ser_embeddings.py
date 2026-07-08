#!/usr/bin/env python3
"""Extract frozen frame-level XLS-R representations for the SER corpus."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import crop_or_pad, load_audio_mono


def select_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def load_encoder(checkpoint: Path, data_dir: Path, device: str):
    from fairseq import checkpoint_utils

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint)], arg_overrides={"data": str(data_dir)}
    )
    if not models:
        raise RuntimeError("Fairseq returned no model")
    owner = models[0].eval().to(device)
    if not hasattr(owner, "w2v_encoder") or not hasattr(owner.w2v_encoder, "w2v_model"):
        raise RuntimeError("Checkpoint lacks model.w2v_encoder.w2v_model")
    encoder = owner.w2v_encoder.w2v_model.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, owner


def forward_encoder(
    encoder: Any, source: Any, padding_mask: Any, valid_samples: int
):
    result = encoder(
        source=source,
        padding_mask=padding_mask,
        mask=False,
        features_only=True,
    )
    hidden = result["x"]
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected [B,T,D], got {tuple(hidden.shape)}")
    output_padding = result.get("padding_mask")
    if output_padding is not None:
        valid_frames = int((~output_padding[0]).sum().item())
    else:
        valid_ratio = valid_samples / source.shape[1]
        valid_frames = max(1, min(hidden.shape[1], round(hidden.shape[1] * valid_ratio)))
    return hidden[:, :valid_frames]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one frozen frame-level embedding file per SER utterance."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pooling", choices=("none",), default="none")
    parser.add_argument("--max_duration_seconds", type=float, default=12.0)
    parser.add_argument(
        "--fairseq_data_dir",
        type=Path,
        default=PROJECT_ROOT / "resources" / "qxp_v2",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint_path.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_dir = args.fairseq_data_dir.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    if not (data_dir / "dict.ltr.txt").is_file():
        parser.error(f"Fairseq dictionary not found in {data_dir}")
    if args.max_duration_seconds <= 0:
        parser.error("--max_duration_seconds must be positive")

    try:
        import torch

        device = select_device(args.device, torch)
        encoder, owner = load_encoder(checkpoint, data_dir, device)
        manifest = pd.read_csv(manifest_path)
        required = {
            "audio_path", "utterance_id", "actor", "emotion",
            "valence", "arousal", "dominance", "duration",
        }
        if missing := required - set(manifest.columns):
            parser.error(f"Manifest missing columns: {sorted(missing)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        max_samples = int(round(args.max_duration_seconds * 16000))
        metadata_rows = []
        for row in manifest.to_dict("records"):
            audio_path = Path(str(row["audio_path"])).expanduser()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            embedding_path = output_dir / f"{row['utterance_id']}.pt"
            if embedding_path.exists() and not args.overwrite:
                metadata_rows.append({**row, "embedding_path": str(embedding_path.resolve())})
                continue
            audio, _ = load_audio_mono(audio_path, 16000)
            audio, valid_samples = crop_or_pad(audio, max_samples)
            source = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            padding_mask = torch.ones_like(source, dtype=torch.bool)
            padding_mask[:, :valid_samples] = False
            with torch.inference_mode():
                hidden = forward_encoder(
                    encoder, source, padding_mask, valid_samples
                )[0].cpu()
            payload = {
                "hidden_states": hidden,
                "utterance_id": str(row["utterance_id"]),
                "audio_path": str(audio_path.resolve()),
                "duration": float(row["duration"]),
                "actor": str(row["actor"]),
                "emotion": str(row["emotion"]),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
            }
            torch.save(payload, embedding_path)
            metadata_rows.append({**payload, "embedding_path": str(embedding_path.resolve())})
            metadata_rows[-1].pop("hidden_states")
            print(f"{row['utterance_id']}: {tuple(hidden.shape)}")

        fields = [
            "embedding_path", "audio_path", "utterance_id", "actor", "emotion",
            "valence", "arousal", "dominance", "duration",
        ]
        with (output_dir / "embeddings_metadata.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [{field: row[field] for field in fields} for row in metadata_rows]
            )
        print(f"Metadata: {output_dir / 'embeddings_metadata.csv'}")
        del owner
    except Exception as exc:
        print(f"ERROR: embedding extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
