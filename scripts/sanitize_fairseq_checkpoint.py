from pathlib import Path
import argparse
import copy
import torch
import fairseq  # needed so torch can unpickle fairseq objects
from omegaconf import OmegaConf


def cfg_to_plain(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
    return copy.deepcopy(cfg)


def remove_path(obj, dotted_path):
    parts = dotted_path.split(".")
    cur = obj

    for part in parts[:-1]:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False

    last = parts[-1]
    if isinstance(cur, dict) and last in cur:
        del cur[last]
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[
            "task.rebuild_batches",
        ],
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"Loading: {src}")
    state = torch.load(src, map_location="cpu")

    if "cfg" not in state:
        raise RuntimeError("Checkpoint does not contain a cfg key.")

    cfg_plain = cfg_to_plain(state["cfg"])

    removed = []
    missing = []

    for key in args.remove:
        if remove_path(cfg_plain, key):
            removed.append(key)
        else:
            missing.append(key)

    state["cfg"] = OmegaConf.create(cfg_plain)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)

    print(f"Saved sanitized checkpoint: {dst}")
    print(f"Removed keys: {removed}")
    print(f"Missing keys: {missing}")


if __name__ == "__main__":
    main()
