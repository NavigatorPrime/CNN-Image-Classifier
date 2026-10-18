# Model A: Standard CNN Architecture Design

## 1. Overview & Architectural Philosophy
**Model A** is a classic hierarchical deep convolutional neural network designed specifically for low-resolution image classification (emulating low-resource sensor environments with images constrained to **$64 \times 64$ RGB pixels**).

As required by the assignment specification:
> **Model A (Standard CNN):** Interleave standard 2D convolutional layers and max-pooling layers.

The network systematically alternates between spatial feature extraction (using $3 \times 3$ standard 2D convolutions with unit padding) and spatial resolution reduction (using $2 \times 2$ max-pooling with stride 2). This interleaving structure enables:
1. **Hierarchical Representation Learning:** Early layers detect primitive visual features (edges, color boundaries, microscopic gradients), while deeper layers synthesize high-level mycological structures (spore morphology, fungal hyphae septation, and colony texture patterns).
2. **Translation Invariance:** Max-pooling introduces local spatial translation and perturbation invariance, ensuring fungal structures can be recognized regardless of exact positioning in the microscopic crop.
3. **Controlled Parameter Complexity:** Progressively doubling filter channels ($32 \to 64 \to 128 \to 256$) while halving spatial dimensions ($64 \to 32 \to 16 \to 8 \to 4$) maintains a balanced computational budget suitable for constrained environments.

---

## 2. Layer-by-Layer Structural Specifications

| Stage | Layer Name | Layer Type | Specifications (Kernel, Stride, Padding) | Output Tensor Shape $(C \times H \times W)$ | Param Count | Activation & Regularization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `Input` | Input Image | $64 \times 64 \times 3$ RGB | $3 \times 64 \times 64$ | 0 | - |
| **Block 1** | `conv1` | Conv2D | $32$ filters, $3 \times 3$, stride=1, pad=1 | $32 \times 64 \times 64$ | 896 | BatchNorm2d + ReLU |
| | `pool1` | MaxPool2D | $2 \times 2$, stride=2 | $32 \times 32 \times 32$ | 0 | - |
| **Block 2** | `conv2` | Conv2D | $64$ filters, $3 \times 3$, stride=1, pad=1 | $64 \times 32 \times 32$ | 18,496 | BatchNorm2d + ReLU |
| | `pool2` | MaxPool2D | $2 \times 2$, stride=2 | $64 \times 16 \times 16$ | 0 | - |
| **Block 3** | `conv3` | Conv2D | $128$ filters, $3 \times 3$, stride=1, pad=1 | $128 \times 16 \times 16$ | 73,856 | BatchNorm2d + ReLU |
| | `pool3` | MaxPool2D | $2 \times 2$, stride=2 | $128 \times 8 \times 8$ | 0 | - |
| **Block 4** | `conv4` | Conv2D | $256$ filters, $3 \times 3$, stride=1, pad=1 | $256 \times 8 \times 8$ | 295,168 | BatchNorm2d + ReLU |
| | `pool4` | MaxPool2D | $2 \times 2$, stride=2 | $256 \times 4 \times 4$ | 0 | - |
| **Dense** | `flatten` | Flatten | Flatten $256 \times 4 \times 4$ | $4096$ | 0 | - |
| | `fc1` | Linear | $4096 \to 512$ | $512$ | 2,097,664 | ReLU + Dropout ($p=0.5$) |
| | `fc2` | Linear | $512 \to 128$ | $128$ | 65,664 | ReLU + Dropout ($p=0.3$) |
| | `fc_out` | Linear | $128 \to 5$ | $5$ | 645 | Linear Logits (Cross-Entropy) |

---

## 3. Mathematical Feature Map Calculations

For each convolution and pooling operation, output spatial dimensions are determined by:
$$H_{out} = \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$$

Where:
- $K$: Kernel size
- $S$: Stride
- $P$: Zero-padding

### Step-by-Step Transformations:
1. **Block 1 Conv2D:**
   $$H_{out} = \left\lfloor \frac{64 - 3 + 2(1)}{1} \right\rfloor + 1 = 64 \quad \Longrightarrow \quad 32 \times 64 \times 64$$
   **Block 1 MaxPool2D:**
   $$H_{out} = \left\lfloor \frac{64 - 2 + 0}{2} \right\rfloor + 1 = 32 \quad \Longrightarrow \quad 32 \times 32 \times 32$$

2. **Block 2 Conv2D:**
   $$H_{out} = \left\lfloor \frac{32 - 3 + 2(1)}{1} \right\rfloor + 1 = 32 \quad \Longrightarrow \quad 64 \times 32 \times 32$$
   **Block 2 MaxPool2D:**
   $$H_{out} = \left\lfloor \frac{32 - 2 + 0}{2} \right\rfloor + 1 = 16 \quad \Longrightarrow \quad 64 \times 16 \times 16$$

3. **Block 3 Conv2D:**
   $$H_{out} = \left\lfloor \frac{16 - 3 + 2(1)}{1} \right\rfloor + 1 = 16 \quad \Longrightarrow \quad 128 \times 16 \times 16$$
   **Block 3 MaxPool2D:**
   $$H_{out} = \left\lfloor \frac{16 - 2 + 0}{2} \right\rfloor + 1 = 8 \quad \Longrightarrow \quad 128 \times 8 \times 8$$

4. **Block 4 Conv2D:**
   $$H_{out} = \left\lfloor \frac{8 - 3 + 2(1)}{1} \right\rfloor + 1 = 8 \quad \Longrightarrow \quad 256 \times 8 \times 8$$
   **Block 4 MaxPool2D:**
   $$H_{out} = \left\lfloor \frac{8 - 2 + 0}{2} \right\rfloor + 1 = 4 \quad \Longrightarrow \quad 256 \times 4 \times 4$$

5. **Classification Bottleneck:**
   $$256 \text{ channels} \times 4 \times 4 = 4,096 \text{ flattened features}$$

---

## 4. Parameter Count Analysis

### Convolutional Layers:
- **`conv1`:** $(3 \times 3 \times 3) \times 32 + 32 = 896$ weights & biases
- **`bn1`:** $32 \times 2 = 64$ affine parameters ($\gamma, \beta$)
- **`conv2`:** $(3 \times 3 \times 32) \times 64 + 64 = 18,496$
- **`bn2`:** $64 \times 2 = 128$
- **`conv3`:** $(3 \times 3 \times 64) \times 128 + 128 = 73,856$
- **`bn3`:** $128 \times 2 = 256$
- **`conv4`:** $(3 \times 3 \times 128) \times 256 + 256 = 295,168$
- **`bn4`:** $256 \times 2 = 512$

### Fully Connected Layers:
- **`fc1`:** $4,096 \times 512 + 512 = 2,097,664$
- **`fc2`:** $512 \times 128 + 128 = 65,664$
- **`fc_out`:** $128 \times 5 + 5 = 645$

### Summary:
- **Total Parameters:** $2,553,349$ (~2.55M)
- **Trainable Parameters:** $2,553,349$
- **Feature Extractor Parameters:** $389,376$ (~15.3%)
- **Classifier Parameters:** $2,163,973$ (~84.7%)

### Hardware-aware ReLU choice

Model A uses ReLU after every hidden convolutional and fully connected layer, exactly as implemented in `model_a.py`. ReLU is a piecewise-linear maximum operation, so it avoids the exponentials or divisions required by sigmoid- or tanh-like activations. This makes it broadly efficient on CPUs and common inference accelerators without claiming a device-specific latency. The output layer remains linear because `CrossEntropyLoss` consumes logits directly.

---

## 5. Training & Regularization Strategy
- **Loss Function:** Categorical Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, initial learning rate $\eta = 10^{-3}$)
- **Learning Rate Scheduler:** `ReduceLROnPlateau` reducing learning rate by a factor of 0.5 if validation loss stagnates for 5 epochs.
- **Regularization:**
  - **Batch Normalization:** Stabilizes internal covariate shift after every convolution.
  - **Dropout:** Progressive dropout ($p=0.5$ on `fc1`, $p=0.3$ on `fc2`) prevents co-adaptation of hidden units.
  - **Data Augmentation:** Random horizontal & vertical flips, $\pm 15^\circ$ rotations, and color jitter to combat class imbalance and overfitting on micro-fungi images.

---

## 6. How to Run Model A

```bash
# Run training on DeFungi dataset (default: 30 epochs, batch size 32)
python model_a.py

# Custom training configuration (e.g. 15 epochs)
python model_a.py --epochs 15 --batch-size 32 --lr 0.001
```

All trained checkpoints, loss/accuracy curves (`training_curves.png`), loss curves (`loss_curves.png`), confusion matrix (`confusion_matrix.png`), and classification reports are automatically saved under `results/model_a/`.

---

## 7. Experimental Results (20 Epochs Training Run)

### Overall Performance:
- **Training Time:** 588.8 seconds (~9.8 minutes on CPU)
- **Best Validation Accuracy:** **69.06%** (achieved at epoch 20)
- **Test Set Accuracy:** **68.90%**
- **Test Set Cross-Entropy Loss:** **0.7680**

### Epoch-by-Epoch Convergence Table:

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1045 | 55.73% | 0.9488 | 59.16% | 0.001 |
| 2 | 1.0027 | 59.20% | 0.8723 | 63.05% | 0.001 |
| 3 | 0.9747 | 60.94% | 0.9401 | 60.70% | 0.001 |
| 4 | 0.9368 | 62.05% | 0.8731 | 64.44% | 0.001 |
| 5 | 0.9336 | 62.24% | 0.8678 | 62.54% | 0.001 |
| 6 | 0.9139 | 62.21% | 0.8266 | 64.52% | 0.001 |
| 7 | 0.9085 | 62.87% | 0.8379 | 65.54% | 0.001 |
| 8 | 0.8951 | 63.37% | 0.7999 | 65.76% | 0.001 |
| 9 | 0.8824 | 63.68% | 0.8249 | 66.20% | 0.001 |
| 10 | 0.8714 | 63.73% | 0.7899 | 67.30% | 0.001 |
| 11 | 0.8607 | 64.65% | 0.8190 | 67.96% | 0.001 |
| 12 | 0.8498 | 64.64% | 0.7715 | 67.08% | 0.001 |
| 13 | 0.8508 | 65.47% | 0.7848 | 66.94% | 0.001 |
| 14 | 0.8431 | 65.23% | 0.7746 | 68.04% | 0.001 |
| 15 | 0.8310 | 66.11% | 0.7761 | 66.79% | 0.001 |
| 16 | 0.8276 | 65.83% | 0.7563 | 66.57% | 0.001 |
| 17 | 0.8146 | 66.39% | 0.7611 | 67.96% | 0.001 |
| 18 | 0.8200 | 65.92% | 0.7369 | 68.62% | 0.001 |
| 19 | 0.8033 | 66.80% | 0.7354 | 68.18% | 0.001 |
| 20 | 0.8043 | 67.27% | 0.7565 | 69.06% | 0.001 |

### Per-Class Test Set Performance Breakdown:

| Class | Precision | Recall | F1-Score | Test Samples (Support) |
| :---: | :---: | :---: | :---: | :---: |
| **H1** | 0.6805 | 0.9426 | 0.7904 | 662 |
| **H2** | 0.5190 | 0.2336 | 0.3222 | 351 |
| **H3** | 0.7312 | 0.5484 | 0.6267 | 124 |
| **H5** | 0.9111 | 0.6613 | 0.7664 | 124 |
| **H6** | 0.7826 | 0.8036 | 0.7930 | 112 |
| **Accuracy** | | | **0.6890** | **1373** |
| **Macro Avg** | 0.7249 | 0.6379 | 0.6597 | 1373 |
| **Weighted Avg** | 0.6729 | 0.6890 | 0.6539 | 1373 |

