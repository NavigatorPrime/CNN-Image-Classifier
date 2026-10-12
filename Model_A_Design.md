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
- **Total Parameters:** $\approx 2,552,677$ (~2.55M)
- **Trainable Parameters:** $2,552,677$
- **Feature Extractor Parameters:** $389,376$ (~15.3%)
- **Classifier Parameters:** $2,163,301$ (~84.7%)

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

All trained checkpoints, loss/accuracy curves (`training_curves.png`), confusion matrix (`confusion_matrix.png`), and classification reports are automatically saved under `results/model_a/`.
