# CNN Image Classifier — DeFungi 64×64 Benchmark

This project compares two custom convolutional networks and two lightweight
pre-trained architectures on five classes from the UCI DeFungi dataset. All
experiments use 64×64 RGB inputs and the same deterministic, class-aware split.

> **Canonical result status:** Stage 2 retraining is complete. The authoritative
> leakage-free metrics, checkpoints, histories, plots, and generated tables are
> under `results_clean_split/`. The pre-correction `results/` tree is retained
> only as historical evidence and must not be used as a final result source.

## Environment setup

Clone this branch and create an isolated Python environment:

```bash
git clone https://github.com/NavigatorPrime/CNN-Image-Classifier.git
cd CNN-Image-Classifier
git checkout EN3150-A03-Final-Corrected-Report-v3
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\activate` on Windows, then install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root. Paths below are relative and portable.

## Dataset preparation and validation

The committed package contains 9,114 images from classes `H1`, `H2`, `H3`,
`H5`, and `H6`:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| H1 | 3,083 | 661 | 660 | 4,404 |
| H2 | 1,634 | 350 | 350 | 2,334 |
| H3 | 573 | 123 | 123 | 819 |
| H5 | 573 | 123 | 122 | 818 |
| H6 | 517 | 111 | 111 | 739 |
| **Total** | **6,380** | **1,368** | **1,366** | **9,114** |

Validate the ZIP before extraction. The command fails for missing or extra
files, manifest inconsistencies, duplicate filenames, non-RGB images, wrong
dimensions, or raw/decoded content shared across splits:

```bash
python data_preparation.py --verify-package
```

Extract and revalidate the exact committed split:

```bash
python data_preparation.py
python data_preparation.py --verify-only
```

The final validator line must be `CROSS-SPLIT DUPLICATES = 0`. Rebuilding the
manifest and package is an intentional maintenance operation, not part of a
normal reproduction run:

```bash
python data_preparation.py --rebuild-clean-split --yes-rebuild
```

The algorithm uses seed 42, hashes both raw bytes and decoded RGB pixels, and
assigns each identical-content group wholly to one split. See
`docs/DATASET_SPLIT_AUDIT.md` for the before/after audit.

The committed Stage 2 result audit can be reproduced with:

```bash
python scripts/validate_clean_results.py \
  --results-root results_clean_split
```

It validates dataset hashes and leakage, all 20-epoch histories, checkpoint
selection, checkpoint sizes, confusion-matrix totals, classification metrics,
optimizer controls, CSV/JSON agreement, and MAC/FLOP consistency. See
`docs/STAGE2_CLEAN_RETRAINING.md` for exact run provenance and results.

## Leakage-free retraining workflow

Training scripts create unique timestamped directories under `runs/` by
default. The explicit paths below make one coherent run easier to assemble.
They refuse to write into a non-empty directory unless `--overwrite` is given,
so the historical `results/` tree cannot be replaced by a casual rerun.

The checked-in canonical run used `results_clean_split/` as its explicit root.
Use a new path for independent reproductions unless you intentionally pass
`--overwrite`.

### Model A — canonical 20 epochs

```bash
python model_a.py --epochs 20 --output-dir runs/leakage_free/model_a
```

### Model B — 20 epochs

```bash
python model_b.py --epochs 20 --output-dir runs/leakage_free/model_b
```

### Optimizer comparison

This controlled Model B experiment compares standard SGD, SGD with momentum,
and Adam from identical initial weights:

```bash
python optimizer_comparison.py --epochs 20 --output-dir runs/leakage_free/optimizer_comparison
```

### MobileNetV2 and ShuffleNetV2

Both commands use the same SOTA output root. A single-model run merges its
result into the existing summary instead of replacing the other model’s entry:

```bash
python sota_models.py --model mobilenet_v2 --epochs 20 --output-dir runs/leakage_free
python sota_models.py --model shufflenet_v2 --epochs 20 --output-dir runs/leakage_free
```

## Profiling

Profile the four architectures with a `1×3×64×64` input:

```bash
python scripts/profile_models.py --output-dir runs/leakage_free/computational_cost
```

The profiler counts Conv2d and Linear multiply-accumulate operations only and
reports FLOPs as `2 × MACs`. These calculations are not device latency or peak
runtime-memory measurements.

## Final artifact generation

The checked-in clean run can be regenerated directly from its actual JSON,
CSV, text, checkpoint, and profiler artifacts:

```bash
python scripts/generate_final_artifacts.py \
  --results-root results_clean_split \
  --output-root results_clean_split/final_artifacts \
  --overwrite
python scripts/generate_architecture_tables.py \
  --output-dir results_clean_split/final_artifacts/tables
python scripts/generate_report_data.py
```

The generator contains no fixed performance values. Its default destination is
a unique timestamped directory; an explicit non-empty output directory
requires `--overwrite`.

## Report generation

The report sources read the generated result macros and canonical clean-split
figures directly. Compile each report twice so cross-references settle:

```bash
cd reports
pdflatex Draco_A03_EN3150.tex
pdflatex Draco_A03_EN3150.tex
pdflatex Section_6_Final_Comparison.tex
pdflatex Section_6_Final_Comparison.tex
cd ..
```

Review `docs/SUBMISSION_CHECKLIST.md` before submission. The assignment’s
external dataset information sheet is outside this repository; update and
submit it separately rather than treating this repository change as completion.

## Key project files

- `data_preparation.py`: deterministic packaging, extraction, and strict validation
- `split_manifest.csv`: authoritative split membership
- `model_a.py`, `model_b.py`: custom models
- `optimizer_comparison.py`: controlled optimizer experiment
- `sota_models.py`: MobileNetV2 and ShuffleNetV2 fine-tuning
- `scripts/profile_models.py`: consistent architecture profiling
- `scripts/generate_final_artifacts.py`: artifact-derived tables and figures
- `scripts/generate_architecture_tables.py`: implementation-derived architecture tables
- `scripts/generate_report_data.py`: artifact-derived LaTeX report values
- `Model_B_Design.md`, `SOTA_Evaluation.md`: architecture documentation

Dataset source: [UCI DeFungi (dataset 773)](https://archive.ics.uci.edu/dataset/773/defungi).
