# Model B: Lightweight Edge CNN Architecture Design

## 1. Overview & Architectural Philosophy
**Model B** is an ultra-lightweight Convolutional Neural Network specifically engineered for edge computing and resource-constrained environments (e.g., ARM Cortex-M microcontrollers, Raspberry Pi, or edge sensor nodes). It performs low-resolution classification on the **DeFungi** microscopic fungal dataset ($64 \times 64 \times 3$ RGB pixels).

As mandated by Task 2 of the assignment:
> **Model B (Lightweight CNN):** Replace the standard convolutions with Depthwise Separable Convolutions to drastically reduce multiply-accumulate (MAC) operations. Your total trainable parameter count for Model B must not exceed 100,000.

### Core Architectural Decisions:
1. **Depthwise Separable Factorization:** By decoupling standard 2D convolutions into channel-wise spatial filtering (depthwise convolution) and cross-channel feature recombination ($1 \times 1$ pointwise convolution), the network reduces multiply-accumulate operations and weight counts by **over 80%**.
2. **Stem Layer:** A single standard $3 \times 3$ convolution at the input preserves multi-channel spatial relationships from raw RGB channels before factorized operations commence.
3. **Global Average Pooling (GAP) Classifier:** Instead of massive flattened dense layers (which accounted for ~84.7% of Model A's 2.55M parameters), Model B collapses each $8 \times 8$ feature map into a single average value ($128 \times 1 \times 1$). This eliminates hundreds of thousands of parameters, directly keeping the network within the sub-100k budget.
4. **Hardware-Aware Activation Functions:** Uses **$\text{ReLU6}$**, a bounded piecewise-linear activation that avoids the exponentials used by sigmoid- or tanh-like functions. Its bounded range can simplify quantization calibration, although quantized accuracy must be measured after conversion.

---

## 2. Layer-by-Layer Structural Specifications

| Stage | Layer Name | Layer Type | Specifications (Kernel, Stride, Padding) | Output Tensor Shape $(C \times H \times W)$ | Param Count | Activation & Regularization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `Input` | Input Image | $64 \times 64 \times 3$ RGB | $3 \times 64 \times 64$ | 0 | - |
| **Stem** | `stem_conv` | Conv2D | $16$ filters, $3 \times 3$, stride=1, pad=1 | $16 \times 64 \times 64$ | 432 | BatchNorm2d + ReLU6 |
| **Block 1** | `b1_dw` | Depthwise Conv2D | $16$ filters, $3 \times 3$, stride=2, pad=1, groups=16 | $16 \times 32 \times 32$ | 144 | BatchNorm2d + ReLU6 |
| | `b1_pw` | Pointwise Conv2D | $32$ filters, $1 \times 1$, stride=1, pad=0 | $32 \times 32 \times 32$ | 512 | BatchNorm2d + ReLU6 |
| **Block 2** | `b2_dw` | Depthwise Conv2D | $32$ filters, $3 \times 3$, stride=1, pad=1, groups=32 | $32 \times 32 \times 32$ | 288 | BatchNorm2d + ReLU6 |
| | `b2_pw` | Pointwise Conv2D | $32$ filters, $1 \times 1$, stride=1, pad=0 | $32 \times 32 \times 32$ | 1,024 | BatchNorm2d + ReLU6 |
| **Block 3** | `b3_dw` | Depthwise Conv2D | $32$ filters, $3 \times 3$, stride=2, pad=1, groups=32 | $32 \times 16 \times 16$ | 288 | BatchNorm2d + ReLU6 |
| | `b3_pw` | Pointwise Conv2D | $64$ filters, $1 \times 1$, stride=1, pad=0 | $64 \times 16 \times 16$ | 2,048 | BatchNorm2d + ReLU6 |
| **Block 4** | `b4_dw` | Depthwise Conv2D | $64$ filters, $3 \times 3$, stride=1, pad=1, groups=64 | $64 \times 16 \times 16$ | 576 | BatchNorm2d + ReLU6 |
| | `b4_pw` | Pointwise Conv2D | $64$ filters, $1 \times 1$, stride=1, pad=0 | $64 \times 16 \times 16$ | 4,096 | BatchNorm2d + ReLU6 |
| **Block 5** | `b5_dw` | Depthwise Conv2D | $64$ filters, $3 \times 3$, stride=2, pad=1, groups=64 | $64 \times 8 \times 8$ | 576 | BatchNorm2d + ReLU6 |
| | `b5_pw` | Pointwise Conv2D | $128$ filters, $1 \times 1$, stride=1, pad=0 | $128 \times 8 \times 8$ | 8,192 | BatchNorm2d + ReLU6 |
| **Block 6** | `b6_dw` | Depthwise Conv2D | $128$ filters, $3 \times 3$, stride=1, pad=1, groups=128 | $128 \times 8 \times 8$ | 1,152 | BatchNorm2d + ReLU6 |
| | `b6_pw` | Pointwise Conv2D | $128$ filters, $1 \times 1$, stride=1, pad=0 | $128 \times 8 \times 8$ | 16,384 | BatchNorm2d + ReLU6 |
| **Head** | `gap` | AdaptiveAvgPool2d | Target $(1, 1)$ | $128 \times 1 \times 1$ | 0 | Parameter-free |
| | `flatten` | Flatten | Flatten to 1D | $128$ | 0 | Dropout ($p=0.2$) |
| | `classifier`| Linear | $128 \to 5$ | $5$ | 645 | Linear Logits |

---

## 3. Mathematical Calculations & MAC Complexity

### 3.1 Depthwise Separable Factorization
For a standard convolution transforming $M$ input channels into $N$ output channels with kernel $D_K \times D_K$ and output feature map size $D_G \times D_G$:
$$\text{MACs}_{\text{std}} = D_K \times D_K \times M \times N \times D_G \times D_G$$
$$\text{Params}_{\text{std}} = D_K \times D_K \times M \times N$$

In Depthwise Separable Convolution:
1. **Depthwise Step:** Spatial filtering per channel ($M$ groups):
   $$\text{MACs}_{\text{dw}} = D_K \times D_K \times M \times D_G \times D_G$$
   $$\text{Params}_{\text{dw}} = D_K \times D_K \times M$$
2. **Pointwise Step:** $1 \times 1$ channel projection:
   $$\text{MACs}_{\text{pw}} = 1 \times 1 \times M \times N \times D_G \times D_G$$
   $$\text{Params}_{\text{pw}} = M \times N$$

### 3.2 Theoretical Cost Reduction
$$\text{Ratio} = \frac{\text{MACs}_{\text{sep}}}{\text{MACs}_{\text{std}}} = \frac{D_K^2 \cdot M + M \cdot N}{D_K^2 \cdot M \cdot N} = \frac{1}{N} + \frac{1}{D_K^2}$$
For a standard $3 \times 3$ kernel ($D_K = 3$):
$$\text{Ratio} = \frac{1}{9} + \frac{1}{N} \approx 11.1\% \text{ to } 12.5\%$$
This yields an immediate **$\approx 88\%$ reduction in computation and weights per block** compared to standard 2D convolutions.

---

## 4. Parameter Count Verification

### Exact Trainable Parameters by Layer:
- **`stem`:** Conv2d ($3 \times 3 \times 3 \times 16 = 432$) + BN ($16 \times 2 = 32$) = **464**
- **`block1`:** DW ($3 \times 3 \times 16 = 144$) + BN ($32$) + PW ($1 \times 1 \times 16 \times 32 = 512$) + BN ($64$) = **752**
- **`block2`:** DW ($3 \times 3 \times 32 = 288$) + BN ($64$) + PW ($1 \times 1 \times 32 \times 32 = 1,024$) + BN ($64$) = **1,440**
- **`block3`:** DW ($3 \times 3 \times 32 = 288$) + BN ($64$) + PW ($1 \times 1 \times 32 \times 64 = 2,048$) + BN ($128$) = **2,528**
- **`block4`:** DW ($3 \times 3 \times 64 = 576$) + BN ($128$) + PW ($1 \times 1 \times 64 \times 64 = 4,096$) + BN ($128$) = **4,928**
- **`block5`:** DW ($3 \times 3 \times 64 = 576$) + BN ($128$) + PW ($1 \times 1 \times 64 \times 128 = 8,192$) + BN ($256$) = **9,152**
- **`block6`:** DW ($3 \times 3 \times 128 = 1,152$) + BN ($256$) + PW ($1 \times 1 \times 128 \times 128 = 16,384$) + BN ($256$) = **18,048**
- **`classifier`:** Linear ($128 \times 5 + 5 = 645$) = **645**

$$\mathbf{\text{Total Trainable Parameters} = 37,957}$$
$$\mathbf{\text{Assignment Upper Limit} = 100,000}$$
$$\mathbf{\text{Margin Below Ceiling} = 62.04\% \quad (\text{PASSED})}$$

---

## 5. Hardware-Aware Justification for Chosen Activation Functions

As required by Task 2:
> *"Provide hardware-aware justifications for your chosen activation functions (e.g., avoiding computationally expensive exponentials)."*

### 5.1 Piecewise-linear activation cost

Sigmoid and tanh require exponential evaluation, while $\text{ReLU6}(x)=\min(\max(0,x),6)$ needs only comparisons and clamping. Those operations are typically easier for embedded CPU and accelerator runtimes to implement efficiently. Exact cycle counts depend on the processor, compiler, tensor layout, and inference library, so no fixed latency is claimed here.

### 5.2 Bounded range and quantization

ReLU6 bounds activations to $[0,6]$, which can reduce the effect of large positive outliers when choosing an INT8 scale. This is a useful design property, not a guarantee of quantized accuracy or MCU compatibility; post-training quantization and target-device measurements would still be required.

---

## 6. Training & Regularization Strategy
- **Loss Function:** Categorical Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, learning rate $\eta = 10^{-3}$, weight decay $= 10^{-4}$)
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (halves learning rate when validation loss plateaus for 4 epochs)
- **Regularization:**
  - **Batch Normalization:** Applied after each depthwise and pointwise convolution.
  - **Dropout ($p=0.2$):** Regularizes the pooled 128-dimensional embedding before the final classification head.
  - **Data Augmentation:** Random horizontal/vertical flips, rotations ($\pm 15^\circ$), and color jitter to ensure robust generalization on microscopic images.

---

## 7. How to Run Model B

```bash
# Run training on DeFungi dataset (20 epochs)
python model_b.py

# Custom training configuration
python model_b.py --epochs 20 --batch-size 32 --lr 0.001
```

All trained checkpoints, loss curves (`loss_curves.png`), training curves (`training_curves.png`), confusion matrix (`confusion_matrix.png`), and classification reports are automatically saved under `results/model_b/`.

---

## 8. Comparative Analysis: Model A vs. Model B (Task 4 Preview)

| Metric | Model A (Standard CNN) | Model B (Lightweight CNN) | Advantage / Trade-off |
| :--- | :---: | :---: | :--- |
| **Convolution Type** | Standard 2D Conv | **Depthwise Separable Conv** | Decoupled spatial & channel ops |
| **Classifier Type** | 3 Dense Layers ($4096 \to 512 \to 128 \to 5$) | **Global Average Pooling + Linear** | Massive parameter savings |
| **Total Parameters** | 2,553,349 (~2.55M) | **37,957** (~38k) | **98.51% parameter reduction** (~67.27× fewer) |
| **Assignment Limit (<100k)** | ❌ Exceeds limit | ** Meets constraint** | **62% safety margin** |
| **Checkpoint size on disk** | 9.755 MiB (`.pth`) | **0.176 MiB (180.2 KiB)** | **98.20% smaller checkpoint** |
| **Conv/Linear MACs** | 62.325376 M | **7.262848 M** | **88.35% fewer counted MACs** |
| **Deployment implication** | Larger storage and compute demand | Much smaller storage and MAC count | Target-specific RAM and latency profiling still required |

MACs above use the repository profiler's single convention for a `3×64×64` input: convolution weights and linear multiply-accumulates are counted; batch normalization, activation, pooling, residual, concatenation, shuffle, softmax, and data movement are excluded. FLOPs are reported as $2\times$ MACs. These calculated MACs are not measured latency.

---

## 9. Experimental Results (20 Epochs Training Run)

### Overall Performance:
- **Training Time:** 434.3 seconds (~7.2 minutes on CPU, 21.7s per epoch — **26% faster than Model A**)
- **Best Validation Accuracy:** **73.24%** (achieved at epoch 16, **+4.18% over Model A**)
- **Minimum Validation Loss:** **0.6596** (epoch 20); the validation-accuracy-selected checkpoint at epoch 16 had validation loss **0.6719**
- **Test Set Accuracy:** **70.43%** (**+1.53% over Model A**)
- **Test Set Cross-Entropy Loss:** **0.7132** (improved over Model A's 0.7680)
- **Checkpoint Size on Disk:** **180.2 KiB** (`.pth`) vs. 9.755 MiB for Model A (**98.20% storage reduction**)

### Epoch-by-Epoch Convergence Table:

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.0668 | 56.56% | 0.9125 | 61.73% | 0.001 |
| 2 | 0.9544 | 61.00% | 0.8741 | 63.20% | 0.001 |
| 3 | 0.9333 | 61.58% | 0.8877 | 64.22% | 0.001 |
| 4 | 0.9045 | 62.38% | 0.8105 | 67.67% | 0.001 |
| 5 | 0.8696 | 63.70% | 0.8093 | 67.01% | 0.001 |
| 6 | 0.8490 | 64.42% | 0.7621 | 68.84% | 0.001 |
| 7 | 0.8404 | 64.73% | 0.7785 | 67.74% | 0.001 |
| 8 | 0.8253 | 65.75% | 0.7415 | 68.40% | 0.001 |
| 9 | 0.8221 | 65.36% | 0.7426 | 69.28% | 0.001 |
| 10 | 0.8038 | 65.69% | 0.7198 | 69.79% | 0.001 |
| 11 | 0.7872 | 67.29% | 0.7157 | 68.48% | 0.001 |
| 12 | 0.7841 | 66.69% | 0.7393 | 69.57% | 0.001 |
| 13 | 0.7813 | 66.90% | 0.7020 | 70.75% | 0.001 |
| 14 | 0.7786 | 66.94% | 0.7137 | 69.72% | 0.001 |
| 15 | 0.7585 | 68.62% | 0.7157 | 68.04% | 0.001 |
| 16 | 0.7542 | 69.31% | 0.6719 | 73.24% | 0.001 |
| 17 | 0.7464 | 68.53% | 0.6847 | 70.82% | 0.001 |
| 18 | 0.7490 | 68.83% | 0.6790 | 72.29% | 0.001 |
| 19 | 0.7377 | 68.81% | 0.6698 | 71.63% | 0.001 |
| 20 | 0.7460 | 69.08% | 0.6596 | 72.87% | 0.001 |

### Per-Class Test Performance:
```
              precision    recall  f1-score   support

          H1     0.7434    0.8444    0.7907       662
          H2     0.5329    0.4387    0.4813       351
          H3     0.6989    0.5242    0.5991       124
          H5     0.7934    0.7742    0.7837       124
          H6     0.7881    0.8304    0.8087       112

    accuracy                         0.7043      1373
   macro avg     0.7113    0.6824    0.6927      1373
weighted avg     0.6937    0.7043    0.6951      1373
```
