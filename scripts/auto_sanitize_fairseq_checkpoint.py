from pathlib import Path
import argparse
import copy
import inspect
import torch
import fairseq
from omegaconf import OmegaConf


def cfg_to_plain(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
    return copy.deepcopy(cfg)


def dataclass_field_names(cls):
    if cls is None:
        return set()
    return set(getattr(cls, "__dataclass_fields__", {}).keys())


def find_model_config_class(model_name):
    try:
        from fairseq.models import MODEL_DATACLASS_REGISTRY
        cls = MODEL_DATACLASS_REGISTRY.get(model_name)
        if cls is not None:
            return cls
    except Exception:
        pass

    try:
        import fairseq.models.wav2vec.wav2vec2_asr as wav2vec2_asr
        import fairseq.models.wav2vec.wav2vec2 as wav2vec2

        modules = [wav2vec2_asr, wav2vec2]

        for module in modules:
            candidates = []
            for name, obj in vars(module).items():
                if inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__"):
                    fields = dataclass_field_names(obj)

                    if model_name == "wav2vec_ctc":
                        if "w2v_path" in fields or "freeze_finetune_updates" in fields:
                            candidates.append((name, obj, fields))

                    if model_name == "wav2vec2":
                        if "encoder_layers" in fields and "conv_feature_layers" in fields:
                            candidates.append((name, obj, fields))

            if candidates:
                print(f"Detected config candidates for {model_name}:")
                for name, _, fields in candidates:
                    print(f"  - {name}: {len(fields)} fields")
                return candidates[0][1]

    except Exception as e:
        print(f"Could not inspect wav2vec modules: {e}")

    return None


def get_generation_config_fields():
    try:
        from fairseq.dataclass.configs import GenerationConfig
        fields = dataclass_field_names(GenerationConfig)
        fields.add("_name")
        return fields
    except Exception:
        return {
            "_name",
            "beam",
            "nbest",
            "max_len_a",
            "max_len_b",
            "min_len",
            "match_source_len",
            "unnormalized",
            "no_early_stop",
            "no_beamable_mm",
            "lenpen",
            "unkpen",
            "replace_unk",
            "sacrebleu",
            "score_reference",
            "prefix_size",
            "no_repeat_ngram_size",
            "sampling",
            "sampling_topk",
            "sampling_topp",
            "constraints",
            "temperature",
            "diverse_beam_groups",
            "diverse_beam_strength",
            "diversity_rate",
            "print_alignment",
        }


def filter_dict_keys(d, allowed):
    removed = []
    if not isinstance(d, dict):
        return removed

    for key in list(d.keys()):
        if key not in allowed:
            removed.append(key)
            del d[key]

    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"Loading: {src}")
    state = torch.load(src, map_location="cpu")

    if "cfg" not in state:
        raise RuntimeError("Checkpoint does not contain cfg.")

    cfg = cfg_to_plain(state["cfg"])

    from fairseq.tasks.audio_finetuning import AudioFinetuningConfig
    from fairseq.tasks.audio_pretraining import AudioPretrainingConfig

    allowed_finetune_task = dataclass_field_names(AudioFinetuningConfig)
    allowed_pretrain_task = dataclass_field_names(AudioPretrainingConfig)
    allowed_finetune_model = dataclass_field_names(find_model_config_class("wav2vec_ctc"))
    allowed_pretrain_model = dataclass_field_names(find_model_config_class("wav2vec2"))
    allowed_generation = get_generation_config_fields()

    for allowed in [
        allowed_finetune_task,
        allowed_pretrain_task,
        allowed_finetune_model,
        allowed_pretrain_model,
        allowed_generation,
    ]:
        allowed.add("_name")

    removed = {}

    if "task" in cfg and isinstance(cfg["task"], dict):
        removed["task"] = filter_dict_keys(cfg["task"], allowed_finetune_task)

        if "eval_wer_config" in cfg["task"] and isinstance(cfg["task"]["eval_wer_config"], dict):
            removed["task.eval_wer_config"] = filter_dict_keys(
                cfg["task"]["eval_wer_config"],
                allowed_generation,
            )

    if "model" in cfg and isinstance(cfg["model"], dict):
        removed["model"] = filter_dict_keys(cfg["model"], allowed_finetune_model)

        w2v_args = cfg["model"].get("w2v_args", None)

        if isinstance(w2v_args, dict):
            # Fairseq only needs the nested task/model configs to rebuild the acoustic model.
            for top_key in list(w2v_args.keys()):
                if top_key not in {"task", "model"}:
                    removed.setdefault("model.w2v_args", []).append(top_key)
                    del w2v_args[top_key]

            if "task" in w2v_args and isinstance(w2v_args["task"], dict):
                removed["model.w2v_args.task"] = filter_dict_keys(
                    w2v_args["task"],
                    allowed_pretrain_task,
                )

            if "model" in w2v_args and isinstance(w2v_args["model"], dict):
                removed["model.w2v_args.model"] = filter_dict_keys(
                    w2v_args["model"],
                    allowed_pretrain_model,
                )

    if "generation" in cfg and isinstance(cfg["generation"], dict):
        removed["generation"] = filter_dict_keys(cfg["generation"], allowed_generation)

    state["cfg"] = OmegaConf.create(cfg)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)

    print(f"Saved sanitized checkpoint: {dst}")
    for section, keys in removed.items():
        print(f"Removed {section} keys: {keys}")


if __name__ == "__main__":
    main()
