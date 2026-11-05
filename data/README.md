# Data

This folder holds the local data used by the speech emotion recognition (SER)
experiments. Nothing in it is committed except `.gitkeep` placeholders: no
audio, labels, manifests, or folds are distributed with this repository.
Obtain each corpus from its original source and place it in the layout below.

Two kinds of data are involved, at different stages:

| Stage | Data | Used for | Needed to run SER? |
|---|---|---|---|
| Upstream | Puno Quechua Common Voice (`qxp`) | Training the ASR encoder | No, the trained checkpoint is public |
| Downstream | Quechua Collao emotional corpus | SER manifest, folds, training, evaluation | Yes |

---

## 1. Quechua Collao emotional corpus

All SER systems in this repository (ASR-SER, vanilla XLS-R, SER-only, and
eGeMAPS) are trained and evaluated on the Quechua Collao emotional speech
corpus of Paccotacya-Yanque et al. (2022), *A speech corpus of Quechua Collao
for automatic dimensional emotion recognition*, Scientific Data 9, 778,
[doi:10.1038/s41597-022-01855-9](https://doi.org/10.1038/s41597-022-01855-9).

| | |
|---|---|
| Speech type | Scripted, acted emotional speech |
| Speakers | 7 native Quechua Collao speakers (4 women, 3 men, aged 24–49) |
| Varieties | Puno (3 speakers), Cusco (3), Apurímac (1) |
| Clips | 12,420 |
| Duration | 15 h 15 min 15 s |
| Categorical labels | `angry`, `bored`, `calm`, `excited`, `fear`, `happy`, `neutral`, `sad`, `sleepy` |
| Dimensional labels | Valence, arousal, dominance (VAD) |

Download the corpus from the source linked in that article and follow its
terms of use. Do not commit any part of it to this repository.

### Expected raw layout

Place the extracted corpus at `data/raw/quechua_collao_corpus/` (or anywhere,
passing the path through `--corpus_root`). `scripts/build_ser_manifest.py`
expects:

```text
quechua_collao_corpus/
├── Data/
│   ├── Data.xlsx                  categorical metadata, sheet "map"
│   └── ...                        audio files
└── Labels/
    └── Labels/
        ├── Labels_clean.csv       dimensional labels (preferred)
        └── Labels.csv             fallback if Labels_clean.csv is absent
```

| File | Required columns |
|---|---|
| `Data/Data.xlsx`, sheet `map` | `Audio`, `Emotion`, `Actor`, `File`, `Duration (s)` |
| `Labels/Labels/Labels_clean.csv` | `Audio`, `Valence`, `Arousal`, `Dominance` |

The two tables are joined on `Audio`, ignoring case and file extension. Audio
is looked up through the `File` and `Audio` values, first relative to the
corpus root and to `Data/`, `Audio/`, or `Audios/`, then by file name anywhere
under the corpus root (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`). Audio is
downmixed to mono and resampled to 16 kHz when loaded, so no conversion step is
needed.

### Label normalisation

Emotion labels are lower-cased and mapped to the nine names above, so both
noun and adjective forms are accepted (for example `anger` → `angry`,
`fearful` → `fear`, `happiness` → `happy`). Any other label stops the manifest
build with an error.

### Actor partitions

The corpus reports seven recorded individuals, but actress 7 completed the
unfinished recording set of actress 5, and the released metadata groups their
recordings into one partition. The `Actor` column therefore has six values,
`a1` to `a6`. Each partition is one cross-validation fold. A partition is not
always one person.

---

## 2. Generated files

Running the manifest and fold scripts (see the main [README](../README.md))
writes the following files under `data/processed/`:

```text
data/processed/
├── ser_manifest.csv
├── ser_manifest_summary.json
└── folds/
    ├── folds_summary.json
    ├── fold_1/
    │   ├── train.csv
    │   └── valid.csv
    ├── ...
    └── fold_6/
```

**`ser_manifest.csv`**: one row per utterance with audio present.

| Column | Description |
|---|---|
| `audio_path` | Absolute path to the audio file on the local machine |
| `utterance_id` | Identifier derived from the `Audio` value |
| `actor` | Actor partition, `a1` to `a6` |
| `emotion` | Normalised categorical label (one of nine) |
| `valence`, `arousal`, `dominance` | Dimensional labels |
| `duration` | Clip duration in seconds |

**`ser_manifest_summary.json`** reports the total clips and hours, the clip
counts per actor and per emotion, missing values per column, and any rows whose
audio could not be found. Check `missing_audio_count` before continuing.

**`folds/fold_k/{train,valid}.csv`** have the same columns as the manifest.
Folds follow leave-one-actor-partition-out cross-validation: actors are sorted,
and fold *k* holds out the *k*-th partition (`fold_1` holds out `a1`, and so
on up to `fold_6` and `a6`). The other five partitions form the training set.
The script fails if any actor appears on both sides of a fold.

**`folds/folds_summary.json`** records the held-out actor and the training and
validation row counts for each fold.

Because `audio_path` is absolute, the generated files are specific to one
machine. Regenerate them rather than copying them between machines.

---

## 3. Puno Quechua Common Voice data (optional)

The ASR encoder reused here (`QuechuaBase/xls-r-cpt-qxp-silver`) was
continued-pretrained and fine-tuned on Puno Quechua (ISO 639-3: `qxp`) speech
collected through Mozilla Common Voice and released under CC0-1.0 on the
[Mozilla Data Collective](https://mozilladatacollective.com). This checkpoint
comes from the first evaluation round, which used Common Voice Scripted Speech
25.0 (`SCS-25`) and Spontaneous Speech 3.0 (`SPS-3`), together with silver
spontaneous transcriptions and a 0.27-hour out-of-domain evaluation set. None
of this data is needed to reproduce the SER results, because the checkpoint is
public. It matters only if you rebuild the ASR side, in which case place it
under `data/raw/scs/`, `data/raw/sps/`, `data/raw/silver/`, and
`data/raw/add_data/`. Release history, splits, file formats, and download
steps are documented in the
[ASR repository's data README](https://github.com/QuechuaBase/asr-puno-quechua/tree/main/data).

---

## 4. Licence and ethics

- **Quechua Collao emotional corpus**: governed by the terms of its original
  release. Obtain it from the source and do not redistribute it.
- **Common Voice `qxp` audio and metadata**: CC0-1.0. Users of the Mozilla Data
  Collective accept a data usage agreement that forbids attempting to
  re-identify contributors and forbids redistributing the datasets.
- Do not commit audio, labels, manifests, folds, or any derived file containing
  speaker identifiers to this repository.
