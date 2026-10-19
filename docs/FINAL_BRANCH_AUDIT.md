# EN3150 Assignment 03 — Final Branch Audit

> **Superseded audit:** This document records an earlier branch review and its
> dataset conclusions are no longer valid. See `DATASET_SPLIT_AUDIT.md` for the
> Stage 1 leakage finding and corrected split. All recorded model results below
> are historical pending retraining.

**Repository:** `NavigatorPrime/CNN-Image-Classifier`  
**Audited branch:** `Final`  
**Audited commit:** `3e335db30d025e03ee89ebe7233fa23973e065f1`  
**Audit mode:** Read-only. No branch, code, results, or checkpoints were modified.

## Executive conclusion

The `Final` branch contains a valid DeFungi dataset split, working implementations of Model A, Model B, MobileNetV2, and ShuffleNetV2, 20-epoch experimental artifacts for all four models, and internally consistent test-set metrics. However, it is **not submission-ready yet**.

The largest assignment gap is **Task 3 (optimizer comparison)**: no SGD vs. SGD+Momentum vs. selected-optimizer experiment, curves, table, or result artifact exists on the `Final` branch. This task must be added or a previously completed matching experiment must be imported and verified.

Several documentation/numerical issues also need correction without retraining: Model A's documented parameter count is wrong, Model B's total-parameter reduction percentage is wrong, the published MAC counts are not consistent with the model definitions under a standard Conv/Linear MAC convention, Model B's reported “best validation loss” is inconsistent with its saved history, and some deployment/latency claims are unsupported.

The existing Model A, Model B, MobileNetV2, and ShuffleNetV2 checkpoints/results do **not** need to be retrained based on this audit.

---

## 1. Assignment requirement audit

| Requirement | Status | Evidence | Problem | Severity | Exact action needed |
|---|---|---|---|---|---|
| Dataset is not CIFAR-10 and is suitable for low-resolution image classification | Correct | `README.md`, `data_preparation.py`; DeFungi UCI dataset | None | CORRECT | Keep DeFungi |
| Images are maximum 64×64 | Correct | Dataset ZIP inspected: all 9,114 images are exactly 64×64 RGB | None | CORRECT | Do not change dataset resolution |
| Train/validation/test split is 70/15/15 | Correct | ZIP + `split_manifest.csv`: 6,377 / 1,364 / 1,373 = 69.97% / 14.97% / 15.06% | Integer rounding is expected | CORRECT | Keep current split |
| Five DeFungi classes | Correct | H1, H2, H3, H5, H6 in dataset and all reports | None | CORRECT | Keep class mapping |
| Same split for all models | Correct for existing runs | All scripts load `dataset/train`, `dataset/val`, `dataset/test`; all four reports use identical test supports 662/351/124/124/112 | Model A uses a hard-coded absolute dataset path, creating a future reproducibility risk | IMPORTANT | Make Model A resolve dataset relative to repository, as Model B/SOTA do |
| No data leakage / accidental re-splitting | Correct for packaged dataset | All model scripts consume pre-created folders; no model script re-splits data | Raw `data_preparation.py` can regenerate membership and does not explicitly validate against the manifest | MINOR | Verify or regenerate manifest deterministically when preparing raw data |
| Split manifest has correct counts / duplicates | Correct | 9,114 rows; 6,377 train, 1,364 val, 1,373 test; no duplicate class+filename rows | None | CORRECT | Preserve manifest |
| Dataset preparation reproducibility | Partial | `RANDOM_SEED=42` in `data_preparation.py` | `discover_images()` does not explicitly sort the discovered file list; raw regeneration could depend on filesystem traversal order | IMPORTANT | Sort image lists before seeded shuffle; write/verify manifest |
| Dataset ZIP portability | Incorrect | `dataset_64x64.zip` exists in repo | `data_preparation.py` uses hard-coded `D:\...` as `BASE_DIR`, so clean clones may not find the ZIP beside the script | IMPORTANT | Replace with `Path(__file__).resolve().parent` |
| Model A uses interleaved standard Conv2D + MaxPool | Correct | `model_a.py`: four Conv-BN-ReLU-MaxPool blocks | None | CORRECT | Keep architecture |
| Model A output dimensions | Correct | 64→32→16→8→4 spatial path; flatten 256×4×4=4096 | None | CORRECT | Keep |
| Model A exact trainable parameters | Code/result correct; design doc wrong | Code/result: **2,553,349** | `Model_A_Design.md` states **2,552,677**; classifier subtotal is also wrong | IMPORTANT | Correct total to 2,553,349 and classifier subtotal to 2,163,973 |
| Model A activation justification | Partial | ReLU is used throughout | No clear hardware-aware justification comparable to Model B | IMPORTANT | Add concise ReLU hardware rationale; no retraining |
| Model B uses depthwise separable convolution | Correct | `model_b.py`: depthwise grouped 3×3 + pointwise 1×1 blocks | None | CORRECT | Keep |
| Model B ≤100,000 trainable parameters | Correct | Exact code-derived count: **37,957** | None | CORRECT | Keep |
| Model B parameter calculation | Correct | Layer-by-layer total = 37,957 | None | CORRECT | Keep exact total |
| Model B activation choice | Correct implementation | ReLU6 after stem/depthwise/pointwise | Documentation overstates some hardware behavior (“prevents saturation”, “optimal”) | MINOR | Reword conservatively: bounded activation can aid quantization; do not claim universal optimality or exact cycle counts without sources |
| Model B total parameter reduction vs Model A | Incorrect documentation | 2,553,349 → 37,957 | `Model_B_Design.md` states **87.0% reduction**; correct reduction is **98.51%**, or ~67.27× fewer parameters | IMPORTANT | Correct percentage/factor |
| Model A vs B MAC values | Not reliable as written | Docs claim 44.97M vs 8.34M | Under a standard Conv2D/Linear multiply-accumulate convention, the architectures give about **62.33M** (A) and **7.26M** (B), excluding BN/activations/pooling. Existing numbers use an undocumented/inconsistent convention | IMPORTANT | Recompute all models with one documented MAC convention and label as calculated, not measured |
| Optimizer chosen and LR stated | Partial | Adam lr=0.001 appears in Model A/B code | Assignment requires explicit comparison vs SGD and SGD+Momentum; that comparison is absent from `Final` | CRITICAL | Add/verify Task 3 experiment and discussion |
| Standard SGD comparison | Missing | No script/result/plot found in `Final`; repository search found no optimizer-comparison artifact | Required by assignment | CRITICAL | Run or import a matching verified Model B SGD experiment |
| SGD + Momentum comparison | Missing | Same as above | Required by assignment | CRITICAL | Run or import a matching verified Model B momentum experiment and explain momentum |
| Both custom models trained ≥20 epochs | Correct | Model A and Model B histories each contain 20 epochs | Model A default CLI is 30, but saved experiment is 20; assignment says at least 20 | CORRECT | Keep saved 20-epoch results |
| Training/validation loss curves for Model A/B | Correct | `results/model_a/loss_curves.png`, `results/model_b/loss_curves.png` | None | CORRECT | Reuse |
| Custom-model test accuracy | Correct / measured | A 68.9002%; B 70.4297% | None | CORRECT | Reuse |
| Custom-model precision/recall | Correct / measured | Classification reports present | None | CORRECT | Reuse |
| Custom-model confusion matrices | Present | PNG artifacts present | Not independently reconstructed from stored raw predictions because predictions are not saved | CORRECT | Reuse; optionally save predictions in future |
| Model A vs B comparison table | Partial | README + `Model_B_Design.md` include comparison information | Parameter-reduction value is wrong; units are inconsistent; table should explicitly include avg epoch time and requested KB size | IMPORTANT | Build one verified Task 4 table |
| Model A avg epoch time | Calculable | 588.8 s / 20 = **29.44 s/epoch** | Not stored in training history | MINOR | Report as calculated from recorded total time |
| Model B avg epoch time | Measured/calculated | 434.3 s / 20 = **21.71 s/epoch** | None | CORRECT | Reuse |
| Two pretrained lightweight SOTA models | Correct | MobileNetV2 + ShuffleNetV2 1.0x with torchvision pretrained weights | None | CORRECT | Keep |
| SOTA fine-tuning uses same dataset folders | Correct | `sota_models.py` loads same train/val/test folders | No model freezing, so all parameters are fine-tuned | CORRECT | Document full fine-tuning |
| SOTA models trained 20 epochs | Correct | histories contain 20 epochs | None | CORRECT | Reuse |
| MobileNetV2 params | Correct | **2,230,277** after replacing classifier with 5-class head | None | CORRECT | Reuse |
| ShuffleNetV2 params | Correct | **1,258,729** after replacing FC with 5-class head | None | CORRECT | Reuse |
| SOTA model sizes | Correct values, mislabeled unit convention | Checkpoint sizes: MobileNet 9,164,555 B = 8.740 MiB; ShuffleNet 5,207,139 B = 4.966 MiB | Code labels 1024²-based values as “MB” instead of MiB | MINOR | Either relabel as MiB or convert to decimal MB consistently |
| SOTA test metrics | Correct / measured | MobileNet 86.31%, loss 0.4092, macro F1 0.8805; ShuffleNet 83.76%, loss 0.4167, macro F1 0.8590 | None identified | CORRECT | Reuse |
| Final Model B vs SOTA accuracy comparison | Available | Existing verified accuracies support discussion | None | CORRECT | Reuse |
| Final memory-footprint comparison | Available with caveats | Checkpoint sizes + params available | Checkpoint size is not runtime RAM/peak activation memory | IMPORTANT | Distinguish disk/weight memory from runtime memory |
| Final computational-cost comparison | Incomplete | A/B MAC claims exist; SOTA MAC counts absent | Assignment asks heavy focus on computational cost; training time is not inference cost | IMPORTANT | Calculate consistent MACs for all three/four models and optionally benchmark inference |
| Final inference latency | Unverified | No latency benchmark artifact in `Final` | `SOTA_Evaluation.md` claims “sub-50ms inference latency” without measurement | IMPORTANT | Remove claim or add a controlled latency benchmark |
| Edge deployment claims | Unsupported/overstated | `SOTA_Evaluation.md` says Model B can require “zero external RAM” and implies MCU fit from file size | Weight file size alone does not prove peak RAM/flash/operator feasibility | IMPORTANT | Qualify claims; report peak activation/runtime memory only if measured |
| “<100 KB memory” deployment conclusion | Incorrect/contradictory | Model B checkpoint itself is ~180 KiB | Cannot call it a <100 KB-memory solution without quantization/compression evidence | IMPORTANT | Replace with “sub-100k-parameter model”; do not equate parameter count with KB |
| Model B “best validation loss” in summary | Incorrect | Saved history minimum val loss = **0.6596** at epoch 20; checkpoint chosen by best val accuracy has val loss ≈0.6719 at epoch 16 | README/SOTA table lists 0.7029 | IMPORTANT | Correct the value/label; specify whether reporting global min loss or checkpoint-selected loss |
| Model A “best validation loss” | Correct | min history ≈0.7354 | None | CORRECT | Reuse |
| Source/result unit consistency | Needs correction | Model B report uses KB via /1024; SOTA uses “MB” via /1024² | KB/MB vs KiB/MiB mixed | MINOR | Standardize units |
| Absolute paths | Incorrect | `model_a.py` and `data_preparation.py` hard-code `D:\Sem 5\...` | Not portable; can fail on Linux/other PCs | IMPORTANT | Resolve paths relative to repo |
| Device handling | Correct | CUDA if available else CPU | None | CORRECT | Keep |
| Checkpoint loading | Correct | Best checkpoints loaded before test evaluation | A/B choose best validation accuracy; SOTA choose best validation loss—different selection criteria | MINOR | Document selection criterion; no mandatory retrain |
| Output directories | Mostly correct | Model B/SOTA create repo-relative outputs; Model A creates under hard-coded base | Model A portability issue | IMPORTANT | Make repo-relative |
| Random seeds for training | Missing | No explicit torch/numpy/random seed in training scripts | Exact reruns are not deterministic | IMPORTANT | Add a shared seed utility for future runs; do not invalidate existing measured results |
| Dependency specification | Missing | No `requirements.txt` on `Final` root | Fresh-environment reproducibility is weaker | IMPORTANT | Add `requirements.txt` |
| README freshness | Partial | Current benchmark table is useful | Collaboration section still describes “teammate build Model B” although Model B is already integrated | MINOR | Update README for final submission |
| Git commit history | Good | Multiple implementation/training/merge commits from Oct 8–18 | None | CORRECT | Preserve history; include repo/profile link in report |
| Final report PDF | Missing from `Final` | No report/LaTeX files in branch root | Assignment grades from PDF and requires names/index numbers | CRITICAL | Stage 4 must create final report PDF |
| Group details in final report | Missing because report is absent | User-provided group: Draco; four members known | Must appear in PDF | CRITICAL | Add group name/description and members to title page |
| GitHub/profile link in report | Missing because report is absent | Repository exists with regular commits | Assignment explicitly requires GitHub/SVN profile link | IMPORTANT | Put repository/profile link in final PDF |
| Interpretation/discussion | Partial | Markdown docs include some discussion | Several unsupported hardware claims and Task 3 is absent | IMPORTANT | Rewrite discussion using verified evidence |
| Dataset information sheet update | Unverified | Assignment asks to update dataset information “in this sheet” | The external sheet is not part of the repository | IMPORTANT | Confirm team has updated the required sheet separately |

---

## 2. Dataset verification details

The uploaded repository dataset archive was independently inspected:

- Total images: **9,114**
- Image dimensions: **all 9,114 are 64×64**
- Image mode: **all RGB**
- Train: **6,377**
- Validation: **1,364**
- Test: **1,373**

Per-class split:

| Split | H1 | H2 | H3 | H5 | H6 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,082 | 1,633 | 573 | 572 | 517 | 6,377 |
| Validation | 660 | 350 | 122 | 122 | 110 | 1,364 |
| Test | 662 | 351 | 124 | 124 | 112 | 1,373 |

`split_manifest.csv` contains exactly 9,114 rows with the same split/class counts and no duplicate class+filename rows.

The four saved classification reports have exactly the same test support counts, strongly supporting that the existing runs used the common packaged test split.

---

## 3. Numerical verification

### 3.1 Parameter counts

**Model A**

Code-derived exact trainable parameter total:

- Feature extractor (Conv + BatchNorm): **389,376**
- Classifier: **2,163,973**
- **Total: 2,553,349**

Therefore `model_a.py`, `README.md`, and the classification report are correct; `Model_A_Design.md` is incorrect where it states 2,552,677.

**Model B**

Code-derived exact trainable parameter total:

- Stem: 464
- Blocks 1–6: 752 + 1,440 + 2,528 + 4,928 + 9,152 + 18,048
- Classifier: 645
- **Total: 37,957**

This is **62.04% below the 100,000-parameter ceiling** and uses **98.51% fewer parameters than Model A** (about **67.27× fewer**).

The `Model_B_Design.md` row stating “87.0% parameter reduction” is incorrect.

### 3.2 Checkpoint sizes

| Model | File bytes | Binary size |
|---|---:|---:|
| Model A | 10,228,869 B | 9.755 MiB |
| Model B | 184,481 B | 180.157 KiB / 0.176 MiB |
| MobileNetV2 | 9,164,555 B | 8.740 MiB |
| ShuffleNetV2 | 5,207,139 B | 4.966 MiB |

Model B's checkpoint is approximately **98.20% smaller** than Model A's actual checkpoint.

### 3.3 Existing measured performance

| Model | Test accuracy | Test loss | Macro precision | Macro recall | Macro F1 | Avg epoch time |
|---|---:|---:|---:|---:|---:|---:|
| Model A | 68.9002% | 0.7680 | 0.7249 | 0.6379 | 0.6597 | 29.44 s* |
| Model B | 70.4297% | 0.7132 | 0.7113 | 0.6824 | 0.6927 | 21.71 s |
| MobileNetV2 | 86.31% | 0.4092 | ~0.89 | ~0.87 | 0.8805 | 40.90 s |
| ShuffleNetV2 | 83.76% | 0.4167 | ~0.87 | ~0.85 | 0.8590 | 20.59 s |

`*` Model A average is calculated from recorded 588.8 s / 20 epochs.

Accuracy gains over Model B:
- MobileNetV2: **+15.88 percentage points**
- ShuffleNetV2: **+13.33 percentage points**

### 3.4 MAC claims

Current docs report:
- Model A: 44.97 M MACs
- Model B: 8.34 M MACs

These values should **not be used in the final report until recomputed consistently**.

Using a straightforward Conv2D + Linear MAC convention on the implemented 64×64 architectures (excluding BN, activations, pooling and data movement) gives approximately:
- Model A: **62.33 M MACs**
- Model B: **7.26 M MACs**

Because the repository does not document how 44.97 M / 8.34 M were obtained, the existing values are classified as **UNVERIFIED/INCONSISTENT**, not measured facts.

Stage 2/3 should use one profiler/counting convention for Model A, Model B, MobileNetV2 and ShuffleNetV2 and document what is included.

---

## 4. Evidence classification

### MEASURED
These come from saved experimental artifacts:
- test accuracy and loss
- precision/recall/F1 classification reports
- epoch histories
- total training times / recorded epoch times
- checkpoint byte sizes
- 20-epoch completion

### CALCULATED
These can be reproduced from code/artifacts:
- exact trainable parameter counts
- average Model A epoch time from total/20
- split percentages
- parameter/storage reduction percentages
- accuracy differences in percentage points

### ESTIMATED / THEORETICAL
These require a clearly stated convention:
- MAC counts
- FLOP counts
- theoretical edge efficiency
- FP32 weight-memory estimate

### UNVERIFIED
Do not present these as measured facts in the final report unless Stage 2/3 measures them:
- sub-50 ms inference latency
- peak runtime RAM / activation memory
- power consumption
- “zero external RAM”
- guaranteed ESP32/STM32 fit
- optimizer superiority, because the required Task 3 comparison is absent from `Final`

---

# A. Correct items that must NOT be changed

1. DeFungi dataset choice.
2. 64×64 RGB dataset archive.
3. Existing 6,377 / 1,364 / 1,373 split.
4. Five classes H1/H2/H3/H5/H6.
5. Existing split manifest.
6. Model A architecture.
7. Model B architecture and 37,957-parameter count.
8. Existing 20-epoch Model A results.
9. Existing 20-epoch Model B results.
10. MobileNetV2 architecture/head replacement and existing result.
11. ShuffleNetV2 architecture/head replacement and existing result.
12. Existing four checkpoints unless a genuine corruption is discovered later.
13. Existing classification reports and training histories.
14. Git history on the `Final` branch.

---

# B. Corrections that do NOT require retraining

1. Make `data_preparation.py` repository-relative and deterministic.
2. Make `model_a.py` repository-relative.
3. Add shared reproducibility seeds for future runs.
4. Add `requirements.txt`.
5. Correct `Model_A_Design.md` parameter total and classifier subtotal.
6. Correct Model B total parameter-reduction percentage to 98.51%.
7. Remove/recompute undocumented MAC numbers.
8. Correct Model B validation-loss summary.
9. Standardize KB/KiB and MB/MiB units.
10. Remove unsupported “sub-50 ms inference” claim unless measured.
11. Remove/qualify “zero external RAM” and MCU-fit claims.
12. Fix the “<100 KB memory” wording; use “sub-100k parameters”.
13. Add Model A hardware-aware ReLU justification.
14. Update stale README collaboration language.
15. Build verified Task 4 and Task 5 comparison tables.
16. Add consistent computational-cost profiling for all models (no training needed).
17. Create the final report and required final comparison figures.

---

# C. Corrections that DO require rerunning an experiment

## Required unless matching artifacts already exist elsewhere and can be verified

**Task 3 optimizer comparison**

The `Final` branch has no evidence for:
- standard SGD
- SGD with Momentum
- comparison against selected Adam configuration
- optimizer convergence plot/table
- momentum discussion based on experiment

A Model B optimizer experiment must therefore be run **unless the team already has matching artifacts from the exact same Model B and exact same dataset split**, in which case Stage 2 should import and verify them instead of retraining.

No evidence from this audit requires retraining Model A, Model B's main Task 4 run, MobileNetV2, or ShuffleNetV2.

## Optional benchmark, not required training

A controlled inference-latency benchmark may be run if the final report wants measured latency. It is not model retraining.

---

# D. Missing graphs/tables/report material

### Missing / required
1. Task 3 SGD vs SGD+Momentum vs selected optimizer convergence figure.
2. Task 3 optimizer comparison result table/discussion.
3. One corrected Model A vs Model B Task 4 table with exact requested fields.
4. Consistent computational-cost table for Model B vs both SOTA models.
5. Final accuracy–memory–computation trade-off figures.
6. Complete assignment report PDF.
7. Standalone Section 6 version if desired for the assigned group contribution.
8. Title page with:
   - Group: **Draco**
   - Description: **Dealing with AI from the core**
   - GUNARATHNA K.T.M.B. — 230203G
   - JAYAWARDANA K.P.P. — 230294H
   - JITHMAL P.G.I. — 230306B
   - NIMADITH W.N.A.C. — 230442L
9. GitHub/profile link in the report.
10. Proper bibliography entries for MobileNetV2, ShuffleNetV2, DeFungi/UCI, and any other technical claims used.

### Already present and reusable
- Model A loss/training curves and confusion matrix.
- Model B loss/training curves and confusion matrix.
- MobileNetV2 loss/training curves and confusion matrix.
- ShuffleNetV2 loss/training curves and confusion matrix.
- Per-model classification reports.
- Training histories.
- SOTA summary JSON/table.

---

# E. Exact files Stage 2 should touch

### Modify
- `data_preparation.py`
- `model_a.py`
- `model_b.py` only for reproducibility/comments if needed; architecture should remain unchanged
- `Model_A_Design.md`
- `Model_B_Design.md`
- `sota_models.py` only for reproducibility/benchmark support if needed; architecture should remain unchanged
- `SOTA_Evaluation.md`
- `README.md`

### Add
- `requirements.txt`
- `optimizer_comparison.py` (or equivalently named Task 3 runner)
- `results/optimizer_comparison/` artifacts
- a computational-cost profiling script, e.g. `scripts/profile_models.py`
- verified CSV/JSON comparison tables

### Do NOT replace
- `results/model_a/model_a_best.pth`
- `results/model_b/model_b_best.pth`
- `results/mobilenet_v2/mobilenet_v2_best.pth`
- `results/shufflenet_v2/shufflenet_v2_best.pth`
- existing valid training histories/classification reports, unless later evidence proves corruption

### Stage 3/4 rather than Stage 2
- final combined figures
- LaTeX report source
- final report PDF
- standalone Section 6 PDF

---

## Final Stage-1 verdict

**Status: NOT YET SUBMISSION-READY**

The core four-model experiment is usable and should be preserved. The main experimental gap is Task 3. Most other issues are documentation, portability, numerical consistency, computational-cost reporting, and final-report preparation and can be fixed without repeating the expensive training runs.
