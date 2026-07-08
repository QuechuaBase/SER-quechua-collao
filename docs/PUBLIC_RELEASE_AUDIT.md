# Public Release Audit

Date: 2026-07-08

## Scope

Audited the repository for a public academic GitHub release accompanying an
ASR-to-SER transfer learning paper for Quechua Collao speech emotion
recognition.

## Contents Found

- Source code: `src/`
- CLI scripts: `scripts/`
- Small versioned resource: `resources/qxp_v2/dict.ltr.txt`
- Local data placeholders: `data/raw/.gitkeep`, `data/processed/.gitkeep`
- Local artifact placeholders: `artifacts/checkpoints/.gitkeep`,
  `artifacts/encoders/.gitkeep`, `artifacts/embeddings/.gitkeep`
- Lightweight result summaries and one confusion-matrix image: `results/`
- No notebooks or LaTeX paper source were present at audit time.

## Sensitive Information Review

Searched text files for credentials, tokens, private URLs, absolute local
paths, usernames, reviewer/anonymization markers, confidential wording, and
hardcoded dataset paths.

Findings handled:

- Replaced one example absolute corpus path in `README.md` with the relative
  example `data/raw/quechua_collao_corpus`.
- Moved lightweight result summaries from ignored `artifacts/results/` to
  versionable `results/`.
- Removed generated Python bytecode caches from the working tree.

No API keys, tokens, passwords, private URLs, raw private data, model
checkpoints, embeddings, or trained model weights were found.

## Publication Cautions

- `CITATION.cff` contains placeholder author/paper/repository metadata.
- `LICENSE` is intentionally conservative and should be replaced with the
  selected public license before upload.
- `AGENTS.md` is useful for agent context but includes internal workflow
  instructions. Decide whether it should be public or moved to private project
  notes before release.
- The results in `results/` are lightweight summaries, but confirm they match
  the final paper tables before upload.

## Excluded by `.gitignore`

- Raw data under `data/raw/`
- Processed manifests/folds under `data/processed/`
- Checkpoints, encoders, embeddings, features, trained classifiers, and local
  experiment outputs under `artifacts/`
- Serialized tensors/models: `*.pt`, `*.bin`, `*.safetensors`, `*.joblib`,
  `*.pkl`
- Python caches, virtual environments, logs, local caches, editor folders, and
  LaTeX auxiliary files
