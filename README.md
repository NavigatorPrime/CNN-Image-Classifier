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

## 🏆 Current Results

### Model A: Standard Interleaved Conv2D + MaxPooling
- **Parameters:** 2,553,349 (~2.55M)
- **Training Time:** 588.8s (~9.8 min on CPU, 20 epochs)
- **Best Validation Accuracy:** **69.06%**
- **Test Accuracy:** **68.90%**
- **Test Loss:** **0.7680**
- **Documentation:** See [Model_A_Design.md](Model_A_Design.md)
- **Plots:** [results/model_a/](results/model_a/) contains `loss_curves.png`, `training_curves.png`, and `confusion_matrix.png`.

### Model B: Lightweight Edge CNN (Depthwise Separable Convolutions)
- **Parameters:** **37,957** (< 100,000 budget, **87.0% parameter reduction**)
- **Training Time:** **434.3s** (~7.2 min on CPU, 20 epochs, **26% faster than Model A**)
- **Best Validation Accuracy:** **73.24%** (**+4.18% over Model A**)
- **Test Accuracy:** **70.43%** (**+1.53% over Model A**)
- **Test Loss:** **0.7132** (Lower loss than Model A)
- **Model Size on Disk:** **180.16 KB** vs 10.2 MB for Model A (**98.2% reduction**)
- **Documentation:** See [Model_B_Design.md](Model_B_Design.md)
- **Plots:** [results/model_b/](results/model_b/) contains `loss_curves.png`, `training_curves.png`, and `confusion_matrix.png`.

