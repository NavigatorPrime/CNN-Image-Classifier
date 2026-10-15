# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

> **Branch:** `Final-Code`  
> **Dataset:** DeFungi (UCI ML Repository, ~3,600 microscopic fungi images, 5 classes)  
> **Framework:** PyTorch 2.x · Python 3.10+

---

## 1. Project Overview & Branch Integration Summary

This repository contains the complete, integrated, and debugged codebase for EN3150 Assignment 03. The `Final-Code` branch synthesises contributions from three source branches — **none of which were modified**:

| Source Branch | Original File | Integrated Into | Role |
|:---|:---|:---|:---|
| `main` | `data_preparation.py` | `pipeline.py` → `prepare_dataset()` | Task 1 — DeFungi download, 64×64 resize, 70/15/15 split |
| `main` | `model_a.py` | `pipeline.py` → `ModelA_StandardCNN` | Task 2 Model A, Task 3 optimizer comparison |
| `model-b` | `model_b.py` | `pipeline.py` → `ModelB_LightweightCNN` | Task 2 Model B (<100k params), Task 4 |
| `SOTA` | `sota_models.py` | `pipeline.py` → `task5_sota_fine_tuning()` | Task 5 — MobileNetV2 & ShuffleNetV2 fine-tuning |

### Bugs Fixed & Improvements Made

1. **Unified path resolution** — `BASE_DIR` / `DATASET_DIR` now resolves relative to the script file, so the pipeline runs from any working directory without manual path edits.
2. **Task 3 optimizer comparison** — moved to a shared, reusable `task3_optimizer_comparison()` function; no duplicate training loops.
3. **All figures at 300 DPI** — previously 150 DPI; now meets assignment's high-resolution requirement.
4. **Model size + epoch time for Model A** — the original `model_a.py` did not record these; now measured and included in comparison tables.
5. **Inference latency (ms/image)** — added for all four models; required for Task 5 & 6 rubric items.
6. **Normalised + raw confusion matrices** — both are plotted side-by-side on one figure per model pair.
7. **Final Master Comparison Table (Task 6)** — generated as both `.csv` and `.md`; was entirely missing from source branches.
8. **`assert trainable_params < 100_000`** — enforced at runtime for Model B.
9. **`dataset_64x64.zip` fast-path** — if the pre-packaged repo zip is present, extraction takes ~2 s instead of downloading the 149 MB raw archive.
10. **Windows compatibility** — `num_workers=0` default, UTF-8 `sys.stdout` reconfiguration, `matplotlib.use("Agg")` backend.

---

## 2. Setup & Execution Instructions

### 2.1 Prerequisites
- Python 3.10 or 3.11
- (Optional but recommended) NVIDIA GPU with CUDA 11.8+

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Dataset

The repository ships `dataset_64x64.zip` (~10 MB) — a pre-packaged, already-split 64×64 version of the DeFungi dataset. The pipeline extracts it automatically on first run.

If you prefer to download the raw dataset from scratch (~149 MB):
```bash
python pipeline.py   # Will download from UCI ML Repository automatically
```

### 2.4 Run the Full Pipeline (Tasks 1–6)

```bash
# Full run — all tasks, 20 epochs each
python pipeline.py

# Skip download step (if dataset/ already exists)
python pipeline.py --skip-data-prep

# Custom epoch count
python pipeline.py --epochs 30

# Run a single task
python pipeline.py --task 4          # Only Task 4 (custom model training)
python pipeline.py --task 5          # Only Task 5 (SOTA fine-tuning)
python pipeline.py --skip-task3      # Skip optimizer comparison to save time
```

### 2.5 Run Individual Scripts (as in source branches)

```bash
python data_preparation.py    # Task 1 — dataset download & split
python model_a.py             # Task 2/4 — Model A training
```

---

## 3. Output Directory Map

After a full run, the following files are generated:

```
outputs/
├── figures/
│   ├── dataset_samples_and_distribution.png   ← Task 1
│   ├── optimizer_comparison_curves.png         ← Task 3
│   ├── custom_models_loss_acc_curves.png       ← Task 4
│   ├── custom_models_confusion_matrices.png    ← Task 4
│   ├── sota_models_loss_acc_curves.png         ← Task 5
│   ├── sota_models_confusion_matrices.png      ← Task 5
│   └── tradeoff_analysis_plot.png              ← Task 6
│
├── tables/
│   ├── model_a_best_report.txt                 ← Task 4 classification report
│   ├── model_b_best_report.txt                 ← Task 4 classification report
│   ├── mobilenet_v2_best_report.txt            ← Task 5 classification report
│   ├── shufflenet_v2_best_report.txt           ← Task 5 classification report
│   ├── custom_model_comparison.csv             ← Task 4 Model A vs B table
│   ├── custom_model_comparison.md              ← Task 4 Model A vs B table
│   ├── final_master_comparison.csv             ← Task 6 master table
│   └── final_master_comparison.md              ← Task 6 master table
│
└── checkpoints/
    ├── model_a_best.pth                        ← Best Model A weights
    ├── model_b_best.pth                        ← Best Model B weights
    ├── mobilenet_v2_best.pth                   ← Best MobileNetV2 weights
    └── shufflenet_v2_best.pth                  ← Best ShuffleNetV2 weights
```

---

## 4. Step-by-Step Guide to Completing the Final Report

### Section 1 — Data Preparation

**Dataset Stats (to copy into the report sheet):**
- **Dataset Name:** DeFungi  
- **Source:** <https://archive.ics.uci.edu/dataset/773/defungi>  
- **Description:** Microscopic images of fungi causing superficial infections (yeasts, moulds, dermatophytes)  
- **Number of Classes:** 5  
- **Image Resolution:** 64×64 pixels (downscaled from original)  
- **Colour Channels:** 3 (RGB)  
- **Random Seed:** 42  

**Preprocessing Steps:**
1. All images resized to 64×64 using high-quality Lanczos resampling.
2. RGB conversion (grayscale images promoted to 3-channel).
3. Normalised with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
4. Training set augmented with random horizontal/vertical flips, ±15° rotation, and colour-jitter (brightness/contrast/saturation ±0.2).
5. Deterministic split: 70% Train / 15% Validation / 15% Test, `random.seed(42)`, shuffle before split — zero data leakage.

**Insert Figure:** `outputs/figures/dataset_samples_and_distribution.png`

---

### Section 2 — Custom Architecture Design (20 Marks)

**Insert from code output:**
- Model A layer-by-layer architecture (printed by `print_architecture_summary`)
- Model B layer-by-layer architecture

#### Parameter Count Formulas

**Standard Conv2D:**
$$\text{Params}_{\text{Standard}} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$
*(+1 for bias; set to 0 when `bias=False` and BatchNorm is used)*

**Depthwise Separable Conv2D:**
$$\text{Params}_{\text{DW}} = K_h \times K_w \times C_{\text{in}} \qquad \text{(no cross-channel mixing)}$$
$$\text{Params}_{\text{PW}} = 1 \times 1 \times C_{\text{in}} \times C_{\text{out}}$$
$$\text{Params}_{\text{DWSep}} = (K_h \times K_w \times C_{\text{in}}) + (C_{\text{in}} \times C_{\text{out}})$$

**Example — Block 3 of Model B (DWSep, 32→64, 3×3):**
- DW params = 3×3×32 = 288  
- PW params = 32×64 = 2,048  
- Total     = 2,336  
- Standard equivalent = 3×3×32×64 = 18,432 → **8× reduction**

#### Hardware-Aware Activation Justification

| Activation | Formula | Hardware Impact |
|:---|:---|:---|
| **ReLU** | max(0, x) | Single comparison op; maps to INT8 with threshold at 0; no LUT |
| **ReLU6** | min(max(0, x), 6) | Two comparisons; bounded [0,6] maps exactly to INT8 [0,255] with scale 1/42.5 |
| Sigmoid | 1/(1+e⁻ˣ) | Requires floating-point exponential unit (LUT on MCU, 10–100× slower) |
| GELU/Swish | x·σ(x) | Requires multiply + sigmoid LUT; unsupported in CMSIS-NN |

Model A uses **ReLU** (standard, well-supported, fast clamp). Model B uses **ReLU6** for tighter quantisation-friendly bounded range.

---

### Section 3 — Optimizer Selection & Tuning (15 Marks)

**Insert Figure:** `outputs/figures/optimizer_comparison_curves.png`

**Chosen Optimizer:** Adam (`lr = 1e-3`, `β₁ = 0.9`, `β₂ = 0.999`)

**Theoretical Explanation:**

- **SGD (momentum=0.0):** Pure gradient descent. Each step uses only the current mini-batch gradient. Susceptible to high-frequency oscillations in narrow valleys (ravines) and gets trapped near saddle points where gradient vanishes.

- **SGD with momentum (β=0.9):** The velocity term $v_t = \beta v_{t-1} + \nabla L_t$ acts as an **exponentially-weighted moving average** (EWMA) of past gradients, dampening transverse oscillations and accelerating progress along consistent gradient directions. Formally:

$$v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta L \qquad \theta \leftarrow \theta - \alpha v_t$$

- **Adam:** Combines adaptive per-parameter learning rates (from RMSprop) with momentum. Maintains first moment (mean) and second moment (uncentred variance) estimates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$$
$$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

This normalises steps by the scale of recent gradients, making Adam robust to sparse gradients and greatly reducing the need for learning-rate tuning — explaining its empirically faster convergence on this task.

---

### Section 4 — Custom Model Training & Evaluation (25 Marks)

**Insert Figures:**
- `outputs/figures/custom_models_loss_acc_curves.png`
- `outputs/figures/custom_models_confusion_matrices.png`

**Insert Tables:**
- `outputs/tables/custom_model_comparison.csv` (or `.md`)

**Key Trade-off Discussion:**

| Aspect | Model A | Model B | Change |
|:---|:---|:---|:---|
| Params | ~2.4 M | ~37,957 | **~63× fewer** |
| MACs (M) | ~45 M | ~8.3 M | **~81% fewer** |
| Size (KB) | Large | Small | ~50-100× smaller |
| Training speed | Slower | Faster (fewer ops) | Faster per epoch |
| Accuracy | Higher | Slightly lower | Small gap (~2-5%) |

The trade-off: depthwise separable convolutions sacrifice some representational capacity (each channel filtered independently), but achieve dramatically lower parameter count and MAC cost — essential for MCU deployment where flash memory is limited to 256 KB–1 MB and there is no FPU.

---

### Section 5 — Lightweight SOTA Fine-Tuning & Evaluation (20 Marks)

**SOTA Architectures:**

| Architecture | Key Innovation | Pre-train | Strategy |
|:---|:---|:---|:---|
| **MobileNetV2** | Inverted Residual Bottlenecks + Linear Bottleneck layers | ImageNet | Replace final 1000-class head with 5-class `nn.Linear`; fine-tune all layers |
| **ShuffleNetV2** | Channel Split + Channel Shuffle (no depthwise conv at all) | ImageNet | Replace `model.fc` with 5-class head; fine-tune all layers |

**Transfer Learning Strategy:**
1. Load pretrained ImageNet weights.
2. Replace the classification head with a randomly-initialised `nn.Linear(in_features, 5)`.
3. Fine-tune **all** parameters (no backbone freezing) — justified because DeFungi microscopy images are very different from ImageNet, requiring full feature adaptation.
4. Use a lower LR (`5e-4`) vs custom models (`1e-3`) to avoid destroying pretrained features.

**Insert Figures:**
- `outputs/figures/sota_models_loss_acc_curves.png`
- `outputs/figures/sota_models_confusion_matrices.png`

---

### Section 6 — Final Comparison & Trade-off Discussion (20 Marks)

**Insert Figure:** `outputs/figures/tradeoff_analysis_plot.png`  
**Insert Table:** `outputs/tables/final_master_comparison.csv` (or `.md`)

#### Structured Comparison: Model B vs SOTA

**Accuracy:**
- SOTA models (MobileNetV2, ShuffleNetV2) typically achieve 3–8% higher test accuracy due to:
  - Pre-training on 1.28M ImageNet images providing rich, transferable feature representations.
  - Deeper architectures with residual/skip connections reducing gradient vanishing.
- Model B achieves competitive accuracy given its 37k parameter budget.

**Memory Footprint (Flash / RAM):**

| Constraint | Model B | MobileNetV2 | Notes |
|:---|:---|:---|:---|
| STM32F4 (1MB flash) | ✔ Fits | ✘ Too large (14MB) | Model B deployable |
| ESP32 (4MB flash) | ✔ Fits | ✘ May not fit | Model B preferred |
| Raspberry Pi (no limit) | ✔ | ✔ | SOTA preferred for accuracy |

**Computational Cost:**
- Model B: ~8.3M MACs → feasible for STM32H7 at ~500 MFLOPS (~17ms/frame)
- MobileNetV2: ~300M MACs at standard resolution (larger even at 64×64) → too slow on Cortex-M4

**Overfitting / Domain-Shift:**
- SOTA models pretrained on natural images may suffer domain shift on microscopy data, but this is mitigated by full fine-tuning and augmentation.
- Model B trained from scratch on DeFungi is domain-adapted from epoch 1 — no domain-shift penalty.

**Power Consumption (Edge Deployment):**
- Fewer MACs → fewer memory accesses → lower dynamic power.
- Model B is the clear winner for battery-powered IoT sensors and edge microscopy devices.

**Conclusion:**
> For **Raspberry Pi / edge server** deployment where accuracy is paramount → use SOTA fine-tuned MobileNetV2.  
> For **STM32 / ESP32 / microcontroller** deployment with strict flash/RAM/power constraints → use Model B.

---

## 5. Licence

This codebase was developed for academic purposes as part of EN3150 Pattern Recognition.
