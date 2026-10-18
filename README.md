# CNN Image Classifier - DeFungi Low-Resolution Benchmark

Custom Convolutional Neural Network architectures for low-resource sensor image classification on the **DeFungi** microscopic fungal dataset (UCI Machine Learning Repository).

---

## Reproduce the Identical Split

The repository preserves and verifies one exact train, validation, and test split.

### 1. Clone the Repository
Clone the repository and select this corrected branch:
```bash
git clone https://github.com/NavigatorPrime/CNN-Image-Classifier.git
cd CNN-Image-Classifier
git checkout EN3150-A03-Final-Corrected-Report-v2
```

### 2. Extract the Pre-Processed Dataset (1 Second)
The repository contains the pre-downscaled ($64 \times 64$) dataset package (`dataset_64x64.zip`, only ~10MB) and the split manifest (`split_manifest.csv`).

To extract the dataset:
```bash
python data_preparation.py
```
This automatically unpacks the exact structured dataset:
```
dataset/
├── train/   (6,377 images across 5 classes - 70%)
├── val/     (1,364 images across 5 classes - 15%)
└── test/    (1,373 images across 5 classes - 15%)
```

---

## 📊 Dataset Specifications
- **Source:** [UCI Machine Learning Repository - DeFungi (ID 773)](https://archive.ics.uci.edu/dataset/773/defungi)
- **Input Resolution:** $64 \times 64$ RGB pixels (downscaled to emulate low-resource sensor environments)
- **Classes:** 5 fungal classes (`H1`, `H2`, `H3`, `H5`, `H6`)
- **Total Images:** 9,114
  - **Train (70%):** 6,377 images
  - **Val (15%):** 1,364 images
  - **Test (15%):** 1,373 images
- **Manifest:** [split_manifest.csv](split_manifest.csv) tracks every individual file's assigned split.

---

## 🏆 Current Benchmark Results

| Architecture | Model Paradigm | Total Params | Checkpoint Size (MiB) | Min Val Loss | Test Accuracy (%) | Test Loss | Macro F1 | Training Time (20 eps) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Standard Conv2D + MaxPool | 2,553,349 | 9.755 | 0.7354 | 68.90% | 0.7680 | 0.6597 | 588.8s (29.4s/ep) |
| **Model B** | Custom Sub-100k Edge CNN | 37,957 | 0.176 | 0.6596 | 70.43% | 0.7132 | 0.6927 | 434.3s (21.7s/ep) |
| **ShuffleNetV2 (1.0x)** | Channel Split & Shuffle (SOTA) | 1,258,729 | 4.966 | 0.3836 | **83.76%** | 0.4167 | 0.8590 | **412.0s (20.6s/ep)** |
| **MobileNetV2** | Inverted Residuals (SOTA) | 2,230,277 | 8.740 | **0.3293** | **86.31%** | **0.4092** | **0.8805** | 818.4s (40.9s/ep) |

The Model B checkpoint was selected by validation accuracy at epoch 16 (validation loss 0.6719); its run-wide minimum validation loss was 0.6596 at epoch 20. Checkpoint sizes are serialized files, not runtime RAM, and training time is not inference latency.

### Task 3: Optimizer comparison

Run the controlled Model B comparison on the verified split:

```bash
python optimizer_comparison.py
```

The script compares standard SGD, SGD with momentum, and Adam for 20 epochs from identical initial weights. It writes the required CSV, JSON, curve, metadata, and optional checkpoints under `results/optimizer_comparison/`.

### Computational cost

```bash
python scripts/profile_models.py
```

The profiler uses a `1×3×64×64` input and counts Conv2d/Linear MACs only. Batch normalization, activations, pooling, residual additions, concatenation, channel shuffle, softmax, and data movement are excluded; FLOPs are reported as `2 × MACs`. The calculated counts are not latency measurements.

---

### SOTA Deliverables & Artifacts (Task 5)
- **Detailed Documentation:** [SOTA_Evaluation.md](SOTA_Evaluation.md)
- **Runner Script:** [sota_models.py](sota_models.py)
- **MobileNetV2 Artifacts:** [results/mobilenet_v2/](results/mobilenet_v2/) (`mobilenet_v2_best.pth`, `loss_curves.png`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`)
- **ShuffleNetV2 Artifacts:** [results/shufflenet_v2/](results/shufflenet_v2/) (`shufflenet_v2_best.pth`, `loss_curves.png`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`)


---

## Final Corrected Assignment Deliverables

The corrected assignment package adds the missing controlled optimizer comparison, consistent computational profiling, final figures/tables, and compiled reports.

### Reports
- `reports/Draco_A03_EN3150.pdf`
- `reports/Draco_A03_EN3150.tex`
- `reports/Section_6_Final_Comparison.pdf`
- `reports/Section_6_Final_Comparison.tex`

### Final figures
See `figures/task1/`, `figures/task3/`, `figures/task4/`, `figures/task5/`, and `figures/task6/`.

### Final tables
See `tables/` for CSV and LaTeX versions of the optimizer, custom-model, SOTA, and final comparison tables.

### Audit and submission notes
See `docs/FINAL_BRANCH_AUDIT.md`, `docs/STAGE3_4_SUMMARY.md`, and `docs/SUBMISSION_CHECKLIST.md`.

The report does not claim unmeasured device inference latency or peak runtime RAM. MAC/FLOP values are calculated consistently from Conv2d and Linear layers for a 3x64x64 input.
