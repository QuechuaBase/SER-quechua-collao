# ASR-to-SER Transfer Learning for Quechua Collao

Research code for experiments accompanying a paper on transferring frozen
Puno Quechua ASR representations to Quechua Collao speech emotion recognition
(SER).

## Overview

This repository accompanies "ASR-to-SER Transfer Learning: Towards Speech
Emotion Recognition for Quechua Collao". It evaluates whether a Puno
Quechua-adapted XLS-R ASR encoder transfers to Quechua Collao SER.
The frozen vanilla XLS-R control tests gains beyond generic multilingual
representations. It does not independently separate SSL pretraining, ASR
supervision, and Puno Quechua-specific adaptation.

## Systems

The repository supports four experiment families:

- ASR encoder + SER: frozen XLS-R/Fairseq ASR encoder features followed by SER
  heads.
- SER-only log-Mel baseline: neural baseline trained directly on emotional
  speech audio.
- eGeMAPS baseline: classical acoustic features with scikit-learn models.
- Vanilla XLS-R 300M frozen: `facebook/wav2vec2-xls-r-300m`, using exactly
  the same attentive pooling, task heads, six folds, and training settings as
  ASR-SER. VAD uses a CCC objective; categorical SER uses class-weighted
  cross-entropy and selects models by macro-F1.

## Main Results

Mean and sample standard deviation over six actor-partition folds:

| System | Mean CCC | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---|---|---|---|
| eGeMAPS | 0.467 ± 0.077 | 0.287 ± 0.071 | 0.289 ± 0.074 | 0.244 ± 0.080 |
| SER-only | 0.533 ± 0.079 | 0.361 ± 0.059 | 0.363 ± 0.062 | 0.317 ± 0.071 |
| Vanilla XLS-R | 0.619 ± 0.057 | 0.378 ± 0.084 | 0.378 ± 0.087 | 0.344 ± 0.093 |
| ASR-SER | 0.634 ± 0.091 | 0.449 ± 0.083 | 0.450 ± 0.085 | 0.422 ± 0.097 |

Generic XLS-R substantially improves over eGeMAPS and SER-only. ASR-SER
further improves mean CCC from 0.619 to 0.634 and macro-F1 from 0.344 to
0.422 (approximately 2.5% and 22.5% relative gains using unrounded CSV means).
Vanilla XLS-R is slightly stronger on arousal CCC: 0.673 versus 0.667.
ASR-SER does not win every metric, and these differences do not establish
that Puno Quechua adaptation alone caused the improvement.

See [the vanilla reproduction guide](docs/vanilla_xlsr_baseline.md) for
extraction, validation, six-fold training, summaries, and parameter counting.

The recommended ASR checkpoint is:

```text
QuechuaBase/xls-r-cpt-qxp-silver
```

This is a Fairseq checkpoint (`checkpoint_best.pt`), not a Hugging Face
Transformers package. The expected acoustic encoder path inside the checkpoint
is:

```text
model.w2v_encoder.w2v_model
```

No raw datasets, ASR weights, exported encoders, embeddings, trained
classifiers, or private corpora are included.

## Repository Structure

```text
README.md
LICENSE
CITATION.cff
requirements.txt
requirements_egemaps.txt
scripts/                  command-line entry points
src/                      reusable dataset, model, metric, and audio utilities
resources/qxp_v2/         small Fairseq dictionary required for CTC task loading
data/                     local data placeholders only
artifacts/                local generated artifact placeholders only
results/                  lightweight result summaries and final aggregate outputs
docs/                     release notes and audit documentation
```

Generated files under `data/` and `artifacts/` are intentionally ignored by
Git. Only `.gitkeep` placeholders should be committed there.

## Installation

Use Python 3.10. Fairseq 0.12.2 is old, so `pip<24.1` may be needed in fresh
environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<24.1" setuptools wheel
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip<24.1" setuptools wheel
python -m pip install -r requirements.txt
```

For CUDA, install the matching `torch` and `torchaudio` wheels from the PyTorch
instructions for your machine, then install the remaining requirements.

The eGeMAPS baseline also needs:

```bash
python -m pip install -r requirements_egemaps.txt
```

## Data Availability

This repository does not distribute private or third-party speech corpora.
Researchers must obtain all data from the original sources and place them in
local paths passed through command-line arguments.

Expected local layout:

```text
data/raw/scs/              Common Voice Scripted Speech for qxp
data/raw/sps/              Common Voice Spontaneous Speech for qxp
data/raw/add_data/         out-of-domain ASR data, if available
data/raw/silver/           silver transcription tables, if available
data/raw/quechua_collao_corpus/
data/processed/manifests/
data/processed/folds/
```

The paper used Common Voice Scripted Speech 25.0 (`SCS-25`), Common Voice
Spontaneous Speech 3.0 (`SPS-3`), `Add_data` of approximately 0.27 hours, and
silver spontaneous transcriptions for the V+S ASR condition. If those exact
Mozilla versions are no longer available, use Scripted Speech 26.0 and
Spontaneous Speech 4.0 as operational substitutes and report the difference.

The Quechua Collao emotional corpus must be obtained from its original source.
Pass its local root with `--corpus_root`; do not hardcode machine-specific
paths.

## Smoke Test

Run this lightweight check from the repository root. It does not download data,
load checkpoints, train models, or inspect private corpora.

```bash
python scripts/smoke_test.py
```

You can also check individual CLIs:

```bash
python scripts/build_ser_manifest.py --help
python scripts/train_ser.py --help
python scripts/train_ser_only.py --help
python scripts/train_egemaps_baseline.py --help
```

## Prepare ASR Manifests

Create local data directories:

```bash
python scripts/prepare_dataset_dirs.py --data_root data
```

Inspect an extracted Mozilla/Common Voice directory:

```bash
python scripts/inspect_mozilla_dataset.py \
  --dataset_dir data/raw/scs
```

Prepare normalized ASR manifests:

```bash
python scripts/prepare_qxp_manifests.py \
  --dataset_dir data/raw/scs data/raw/sps data/raw/add_data data/raw/silver \
  --output_dir data/processed/manifests
```

These scripts do not download datasets or convert audio.

## ASR Checkpoint and Encoder Export

Download the public checkpoint only when explicitly needed:

```bash
python scripts/download_asr_weights.py \
  --repo_id QuechuaBase/xls-r-cpt-qxp-silver \
  --output_dir artifacts/checkpoints/xls-r-cpt-qxp-silver
```

Inspect it:

```bash
python scripts/inspect_checkpoint.py \
  --checkpoint_path artifacts/checkpoints/xls-r-cpt-qxp-silver/checkpoint_best.pt
```

If Fairseq configuration incompatibilities appear, sanitize a local copy:

```bash
python scripts/auto_sanitize_fairseq_checkpoint.py \
  --src artifacts/checkpoints/xls-r-cpt-qxp-silver/checkpoint_best.pt \
  --dst artifacts/checkpoints/xls-r-cpt-qxp-silver/checkpoint_best_auto_sanitized.pt
```

Export only the encoder state dictionary:

```bash
python scripts/export_fairseq_xlsr_encoder_state_dict.py \
  --checkpoint_path artifacts/checkpoints/xls-r-cpt-qxp-silver/checkpoint_best_auto_sanitized.pt \
  --output_dir artifacts/encoders/xls-r-cpt-qxp-silver \
  --device cpu
```

Use CPU if the available GPU is not supported by the current PyTorch build.

## Prepare SER Manifest and Folds

Build the merged Quechua Collao SER manifest:

```bash
python scripts/build_ser_manifest.py \
  --corpus_root data/raw/quechua_collao_corpus \
  --output_path data/processed/ser_manifest.csv
```

Create the six leave-one-actor-out folds:

```bash
python scripts/create_actor_folds.py \
  --manifest data/processed/ser_manifest.csv \
  --output_dir data/processed/folds
```

The fold script verifies that train and validation actors are disjoint.

The corpus reports seven recorded individuals. Actress 7 completed/replaced
the incomplete recording set of actress 5; their recordings form one corpus
actor partition for this protocol, not one individual. Released metadata uses
six partitions (`a1` through `a6`), yielding six leave-one-actor-partition-out
folds. Repeat each single-fold command below for folds 1 through 6 before
summarizing.

## ASR Encoder + SER Experiments

Extract frame-level frozen ASR embeddings once and share them across folds:

```bash
python scripts/extract_ser_embeddings.py \
  --checkpoint_path artifacts/checkpoints/xls-r-cpt-qxp-silver/checkpoint_best_auto_sanitized.pt \
  --manifest data/processed/ser_manifest.csv \
  --output_dir artifacts/embeddings/qxp_asr_ser \
  --device cpu \
  --pooling none \
  --max_duration_seconds 12
```

Train VAD regression for one fold:

```bash
python scripts/train_ser.py \
  --task vad_regression \
  --fold_dir data/processed/folds/fold_1 \
  --embeddings_metadata artifacts/embeddings/qxp_asr_ser/embeddings_metadata.csv \
  --output_dir artifacts/classifiers/vad/fold_1 \
  --pooling attentive \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --device cuda
```

Train categorical emotion classification for one fold:

```bash
python scripts/train_ser.py \
  --task emotion_classification \
  --fold_dir data/processed/folds/fold_1 \
  --embeddings_metadata artifacts/embeddings/qxp_asr_ser/embeddings_metadata.csv \
  --output_dir artifacts/classifiers/emotion/fold_1 \
  --pooling attentive \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --device cuda \
  --use_class_weights
```

Evaluate and summarize:

```bash
python scripts/evaluate_ser.py \
  --model_path artifacts/classifiers/emotion/fold_1/best_model.pt \
  --fold_dir data/processed/folds/fold_1 \
  --embeddings_metadata artifacts/embeddings/qxp_asr_ser/embeddings_metadata.csv \
  --output_dir artifacts/classifiers/emotion/fold_1 \
  --device cuda
```

```bash
python scripts/summarize_cv_results.py \
  --results_dir artifacts/classifiers/vad \
  --task vad_regression \
  --output_path results/vad_cv_summary.csv
```

```bash
python scripts/summarize_cv_results.py \
  --results_dir artifacts/classifiers/emotion \
  --task emotion_classification \
  --output_path results/emotion_cv_summary.csv
```

## SER-Only Log-Mel Baseline

This baseline uses only the emotional corpus audio and labels. It does not use
ASR checkpoints, XLS-R, wav2vec2, Whisper, or ASR embeddings.

```bash
python scripts/train_ser_only.py \
  --task vad_regression \
  --fold_dir data/processed/folds/fold_1 \
  --output_dir artifacts/classifiers_ser_only/vad/fold_1 \
  --pooling attentive \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --device cuda \
  --max_duration_seconds 12
```

```bash
python scripts/train_ser_only.py \
  --task emotion_classification \
  --fold_dir data/processed/folds/fold_1 \
  --output_dir artifacts/classifiers_ser_only/emotion/fold_1 \
  --pooling attentive \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --device cuda \
  --max_duration_seconds 12 \
  --use_class_weights
```

Summaries:

```bash
python scripts/summarize_ser_only_cv_results.py \
  --results_dir artifacts/classifiers_ser_only/vad \
  --task vad_regression \
  --output_path results/ser_only_vad_cv_summary.csv
```

```bash
python scripts/summarize_ser_only_cv_results.py \
  --results_dir artifacts/classifiers_ser_only/emotion \
  --task emotion_classification \
  --output_path results/ser_only_emotion_cv_summary.csv
```

## eGeMAPS Baseline

Extract eGeMAPSv02 functionals once:

```bash
python scripts/extract_egemaps_features.py \
  --manifest data/processed/ser_manifest.csv \
  --output_path artifacts/features/egemaps/egemaps_features.csv \
  --feature_set eGeMAPSv02
```

Train VAD with `StandardScaler + SVR`:

```bash
python scripts/train_egemaps_baseline.py \
  --task vad_regression \
  --fold_dir data/processed/folds/fold_1 \
  --features_csv artifacts/features/egemaps/egemaps_features.csv \
  --output_dir artifacts/classifiers_egemaps/vad/fold_1 \
  --tune inner_actor_cv
```

Train emotion classification with `StandardScaler + SVC`:

```bash
python scripts/train_egemaps_baseline.py \
  --task emotion_classification \
  --fold_dir data/processed/folds/fold_1 \
  --features_csv artifacts/features/egemaps/egemaps_features.csv \
  --output_dir artifacts/classifiers_egemaps/emotion/fold_1 \
  --tune inner_actor_cv
```

Summaries:

```bash
python scripts/summarize_egemaps_cv_results.py \
  --results_dir artifacts/classifiers_egemaps/vad \
  --task vad_regression \
  --output_path results/egemaps_vad_cv_summary.csv
```

```bash
python scripts/summarize_egemaps_cv_results.py \
  --results_dir artifacts/classifiers_egemaps/emotion \
  --task emotion_classification \
  --output_path results/egemaps_emotion_cv_summary.csv
```

## Vanilla XLSR Baseline

Extract Embeddings

Smoke test on a few utterances:

```bash
python scripts/extract_vanilla_xlsr_embeddings.py \
  --manifest data/processed/ser_manifest.csv \
  --output_dir artifacts/embeddings/vanilla_xlsr_300m_smoke \
  --device auto \
  --max_duration_seconds 12 \
  --max_utterances 4 \
  --revision fdca614bc5b1534b850bccd3fabca0489ea723d7 \
  --overwrite
```

Full extraction:

```bash
python scripts/extract_vanilla_xlsr_embeddings.py \
  --manifest data/processed/ser_manifest.csv \
  --output_dir artifacts/embeddings/vanilla_xlsr_300m \
  --device cuda \
  --max_duration_seconds 12 \
  --revision fdca614bc5b1534b850bccd3fabca0489ea723d7
```

Validate shapes, labels, and actor-disjoint folds:

```bash
python scripts/check_ser_embedding_protocol.py \
  --embeddings_metadata artifacts/embeddings/vanilla_xlsr_300m/embeddings_metadata.csv \
  --folds_dir data/processed/folds \
  --expected_dim 1024 \
  --output_path results/vanilla_xlsr/protocol_check.json
```

Train VAD:

```bash
for i in 1 2 3 4 5 6; do
  python scripts/train_ser.py \
    --task vad_regression \
    --fold_dir data/processed/folds/fold_${i} \
    --embeddings_metadata artifacts/embeddings/vanilla_xlsr_300m/embeddings_metadata.csv \
    --output_dir artifacts/classifiers_vanilla_xlsr/vad/fold_${i} \
    --pooling attentive \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda
done
```

Train Emotion:

```bash
for i in 1 2 3 4 5 6; do
  python scripts/train_ser.py \
    --task emotion_classification \
    --fold_dir data/processed/folds/fold_${i} \
    --embeddings_metadata artifacts/embeddings/vanilla_xlsr_300m/embeddings_metadata.csv \
    --output_dir artifacts/classifiers_vanilla_xlsr/emotion/fold_${i} \
    --pooling attentive \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda \
    --use_class_weights
done
```

Summaries:

```bash
python scripts/summarize_cv_results.py \
  --results_dir artifacts/classifiers_vanilla_xlsr/vad \
  --task vad_regression \
  --output_path results/vanilla_xlsr/vad_cv_summary.csv
  ```

```bash
python scripts/summarize_cv_results.py \
  --results_dir artifacts/classifiers_vanilla_xlsr/emotion \
  --task emotion_classification \
  --output_path results/vanilla_xlsr/emotion_cv_summary.csv
```

## Results and Figures

The `results/` directory contains lightweight aggregate CSV summaries and the
available final confusion-matrix image. Regenerate those files with the summary
commands above after rerunning the fold experiments.

Existing ASR-SER summaries retain their unprefixed names in `results/`;
eGeMAPS and SER-only retain their filename prefixes. Vanilla results live in
`results/vanilla_xlsr/`. Existing results have not been changed.

Regenerate the four-system fold radar plots and aggregated-confusion per-class
F1 table:

```bash
python scripts/plot_ser_folds.py --results_dir results --output_dir figures
python scripts/confusion_per_class.py --input results/vanilla_xlsr/emotion_cv_confusion_matrix.csv --output results/vanilla_xlsr/emotion_per_class.csv
```

The VAD radar uses mean CCC and the categorical radar uses macro-F1 per fold.
Per-class scores from the pooled confusion matrix are distinct from
fold-averaged metrics and must not replace the reported macro-F1.
No manuscript or workflow-diagram source is included in this repository.

## Expected Outputs

Local generated outputs include:

```text
artifacts/checkpoints/                  downloaded ASR checkpoints
artifacts/encoders/                     exported ASR encoder files
artifacts/embeddings/                   frozen ASR frame-level embeddings
artifacts/classifiers/                  ASR encoder + SER models and logs
artifacts/classifiers_ser_only/         log-Mel SER baseline models and logs
artifacts/features/                     eGeMAPS features
artifacts/classifiers_egemaps/          eGeMAPS models and logs
results/                               lightweight aggregate summaries
```

Do not commit checkpoints, embeddings, features, trained models, raw data, or
private corpus files.

## Citation

If you use this code, cite the accompanying paper. The repository includes
`CITATION.cff` with placeholder metadata that should be completed before the
public GitHub release.

## License

Code is provided under the Apache License 2.0; see `LICENSE`.
Third-party datasets and upstream model weights retain their own terms.

## Contact

Maintainer: `Joaquin Sanchez`

Email: `joaquin.sanchezs@pucp.edu.pe`

Repository URL: `https://github.com/QuechuaBase/SER-quechua-collao`
