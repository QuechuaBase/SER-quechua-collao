from pathlib import Path
import argparse
import json
from datetime import datetime
import torch
import fairseq
from fairseq import checkpoint_utils


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint_path)]
    )

    model = models[0]
    model.eval()

    # Preferimos CPU para exportar, porque no necesitamos ejecutar inferencia.
    model = model.to(device)

    if not hasattr(model, "w2v_encoder"):
        raise RuntimeError("The loaded model does not have attribute 'w2v_encoder'.")

    w2v_encoder = model.w2v_encoder

    if hasattr(w2v_encoder, "w2v_model"):
        acoustic_encoder = w2v_encoder.w2v_model
        selected_component = "model.w2v_encoder.w2v_model"
    else:
        acoustic_encoder = w2v_encoder
        selected_component = "model.w2v_encoder"

    state_dict = acoustic_encoder.state_dict()

    encoder_path = output_dir / "encoder_state_dict.pt"
    torch.save(
        {
            "state_dict": state_dict,
            "selected_component": selected_component,
            "source_checkpoint": str(checkpoint_path),
            "exported_at": datetime.now().isoformat(),
        },
        encoder_path,
    )

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "selected_component": selected_component,
        "output_file": str(encoder_path),
        "num_tensors": len(state_dict),
        "device_used_for_export": device,
        "note": "Only tensor state_dict was saved. The full Fairseq object was not pickled.",
    }

    with open(output_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved encoder state_dict to: {encoder_path}")
    print(f"Saved metadata to: {output_dir / 'export_metadata.json'}")
    print("Export completed successfully.")


if __name__ == "__main__":
    main()
