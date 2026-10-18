# CNN Image Classifier - DeFungi Low-Resolution Benchmark

Custom Convolutional Neural Network architectures for low-resource sensor image classification on the **DeFungi** microscopic fungal dataset (UCI Machine Learning Repository).

---

## 👥 How to Collaborate & Share the Identical Split

To ensure both you and your teammate evaluate models on the **100% exact same train, validation, and test sets**:

### 1. Clone the Repository
Your teammate should run:
```bash
git clone https://github.com/NavigatorPrime/CNN-Image-Classifier.git
cd CNN-Image-Classifier
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

## 🛠️ How Your Teammate Can Build Another Model (e.g. Model B)

To work in parallel without merge conflicts, your teammate should:

### Step 1: Create a Separate Git Branch
```bash
git checkout -b model-b
```

### Step 2: Build the New Model (`model_b.py`)
They can reuse the shared data loader pipeline directly:
```python
import torch
import torch.nn as nn
from model_a import get_data_loaders, DATASET_DIR

# Load the identical split in one line
train_loader, val_loader, test_loader, class_names, n_train, n_val, n_test = get_data_loaders(
    DATASET_DIR, batch_size=32
)

# Define Model B (e.g., Dilated convolutions, Residual connections, or Depthwise Separable)
class ModelB(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Custom Model B layers here...
        pass
```

### Step 3: Push Their Work to GitHub
```bash
git add model_b.py
git commit -m "Implement and evaluate Model B"
git push origin model-b
```
Then create a Pull Request on GitHub to merge into `main`.

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

| Architecture | Model Paradigm | Total Params | Model Size (MB) | Best Val Loss | Test Accuracy (%) | Test Loss | Macro F1 | Training Time (20 eps) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Standard Conv2D + MaxPool | 2,553,349 | 9.75 MB | 0.7354 | 68.90% | 0.7680 | 0.6597 | 588.8s (29.4s/ep) |
| **Model B** | Custom Sub-100k Edge CNN | 37,957 | 0.18 MB (180 KB) | 0.7029 | 70.43% | 0.7132 | 0.6927 | 434.3s (21.7s/ep) |
| **ShuffleNetV2 (1.0x)** | Channel Split & Shuffle (SOTA) | 1,258,729 | 4.97 MB | 0.3836 | **83.76%** | 0.4167 | 0.8590 | **412.0s (20.6s/ep)** |
| **MobileNetV2** | Inverted Residuals (SOTA) | 2,230,277 | 8.74 MB | **0.3293** | **86.31%** | **0.4092** | **0.8805** | 818.4s (40.9s/ep) |

---

### SOTA Deliverables & Artifacts (Task 5)
- **Detailed Documentation:** [SOTA_Evaluation.md](SOTA_Evaluation.md)
- **Runner Script:** [sota_models.py](sota_models.py)
- **MobileNetV2 Artifacts:** [results/mobilenet_v2/](results/mobilenet_v2/) (`mobilenet_v2_best.pth`, `loss_curves.png`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`)
- **ShuffleNetV2 Artifacts:** [results/shufflenet_v2/](results/shufflenet_v2/) (`shufflenet_v2_best.pth`, `loss_curves.png`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`)

