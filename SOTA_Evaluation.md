# Task 5: Lightweight State-of-the-Art (SOTA) Fine-Tuning & Evaluation

---

## 1. Overview & Context

This benchmark fulfills **Task 5 (SOTA Fine-Tuning & Evaluation)** of Assignment 03 (*Resource-Constrained CNN for Edge Image Classification*).

In edge computing scenarios, deploying heavyweight vision backbones (e.g., standard ResNet-50 or VGG-16) is impractical due to constrained SRAM, thermal limits, and limited battery budgets. The industry standard approach relies on **parameter-efficient, hardware-optimized lightweight convolutional architectures**.

To evaluate performance against custom models (Model A and Model B), two pre-trained mobile architectures were selected, fine-tuned, and evaluated on the **DeFungi** microscopic fungal dataset downscaled to **$64 \times 64$ pixels**:
1. **MobileNetV2** (Google, 2018)
2. **ShuffleNetV2 1.0x** (Megvii / Face++, 2018)

---

## 2. Architectural Analysis & Hardware Justifications

### Architecture 1: MobileNetV2
* **Reference:** Sandler et al., *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"* (CVPR 2018).
* **Core Innovation:** **Inverted Residual Blocks with Linear Bottlenecks**.
* **Structural Pipeline:**
  1. **$1 \times 1$ Pointwise Expansion:** Expands low-dimensional channel representation into a higher-dimensional space (expansion factor $t=6$) to preserve information manifold complexity.
  2. **$3 \times 3$ Depthwise Convolution:** Performs lightweight spatial filtering independently per channel.
  3. **$1 \times 1$ Linear Bottleneck Projection:** Projects expanded features back to a compact representation without a non-linear activation (ReLU is omitted in the bottleneck to prevent destroying negative feature activations in low-dimensional subspaces).
  4. **Residual Shortcut:** Connects bottlenecks when input and output dimensions match ($stride=1$), mitigating gradient degradation during fine-tuning.

### Architecture 2: ShuffleNetV2 1.0x
* **Reference:** Ma et al., *"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"* (ECCV 2018).
* **Core Innovation:** **Hardware-Friendly Architecture driven by 4 Direct Metric Principles**:
  1. *Guideline 1 (Equal Channel Width):* Minimizes Memory Access Cost (MAC) when input and output channels are balanced ($c_1 = c_2$).
  2. *Guideline 2 (Group Convolution Cost):* Excessive group convolutions increase MAC; ShuffleNetV2 uses channel split instead.
  3. *Guideline 3 (Network Fragmentation):* Avoids fragmented multi-branch paths that reduce parallelism on GPU/DSP runtimes.
  4. *Guideline 4 (Element-wise Operations):* Minimizes element-wise operations (such as additions and activations).
* **Structural Pipeline:**
  - **Channel Split:** At each block entrance, the $c$ feature channels are split into $c'$ and $c - c'$.
  - One branch remains identity; the other branch undergoes $1\times 1$ Conv $\to$ Depthwise $3\times 3$ Conv $\to$ $1\times 1$ Conv.
  - Branches are concatenated (not added, saving element-wise addition overhead), followed by a **Channel Shuffle** operation that enables cross-group information flow across channels.

---

## 3. Hardware-Aware Edge Justifications

| Hardware Factor | Custom Model B (Sub-100k) | MobileNetV2 | ShuffleNetV2 (1.0x) |
| :--- | :--- | :--- | :--- |
| **Convolution Type** | Depthwise Separable | Inverted Residual + Depthwise | Channel Split + Shuffle + Depthwise |
| **Activation Function** | ReLU6 | ReLU6 | ReLU |
| **Checkpoint Size** | 0.176 MiB | 8.740 MiB | 4.966 MiB |
| **Conv/Linear MACs** | 7.262848 M | 24.454912 M | 11.750752 M |
| **Non-Linearity Design** | Traditional ReLU across all layers | Linear bottleneck prevents manifold collapse | ReLU on single branch only, reducing memory write operations |

> **Activation Function Hardware Rationale:**
> ReLU and ReLU6 use piecewise-linear comparison/clamping operations and avoid exponential evaluation. This generally makes them practical choices for edge inference, but latency and quantized accuracy remain dependent on the target processor, compiler, runtime, and quantization procedure.

---

## 4. Experimental Setup & Protocol

- **Dataset:** UCI DeFungi (Dataset ID 773), 5 classes (`H1`, `H2`, `H3`, `H5`, `H6`).
- **Input Resolution:** $64 \times 64 \times 3$ RGB.
- **Fixed Split (Manifest-Verified):**
  - **Training:** 6,377 images (70%)
  - **Validation:** 1,364 images (15%)
  - **Testing:** 1,373 images (15%)
- **Data Augmentation:** Random Horizontal Flip ($p=0.5$), Random Vertical Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), Color Jitter (brightness/contrast/saturation $\pm 0.2$).
- **Normalization:** ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** Adam ($\text{lr} = 5 \times 10^{-4}$, weight decay $= 10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=3).
- **Epochs:** 20 epochs per model.

---

## 5. Benchmark Results & Artifacts

### 5.1 Full Comparative Benchmark Table (Models A, B vs. Lightweight SOTA)

| Architecture | Model Paradigm | Total Parameters | Trainable Parameters | Checkpoint Size (MiB) | Minimum Val Loss | Test Accuracy (%) | Test Loss | Macro F1-Score | Avg Training Time / Epoch (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Standard Conv2D + MaxPool | 2,553,349 | 2,553,349 | 9.755 | 0.7354 | 68.90% | 0.7680 | 0.6597 | 29.44s |
| **Model B** | Custom Edge (Depthwise Separable) | 37,957 | 37,957 | 0.176 | 0.6596 | 70.43% | 0.7132 | 0.6927 | 21.71s |
| **ShuffleNetV2 (1.0x)** | Channel Split & Shuffle (Pre-trained) | 1,258,729 | 1,258,729 | **4.966** | 0.3836 | **83.76%** | 0.4167 | **0.8590** | **20.59s** |
| **MobileNetV2** | Inverted Residuals (Pre-trained) | 2,230,277 | 2,230,277 | **8.740** | **0.3293** | **86.31%** | **0.4092** | **0.8805** | 40.90s |

The Model B checkpoint was selected by highest validation accuracy (epoch 16, validation loss 0.6719); 0.6596 is the run's minimum validation loss at epoch 20. Checkpoint size is serialized file size, not runtime RAM. The epoch timings are training measurements and must not be interpreted as inference latency.

---

### 5.2 Detailed Class-by-Class Performance Breakdown

#### MobileNetV2 (Test Accuracy: 86.31%, Macro F1: 0.8805)
```text
              precision    recall  f1-score   support
          H1       0.87      0.90      0.88       662
          H2       0.78      0.77      0.77       351
          H3       0.88      0.85      0.87       124
          H5       0.97      0.89      0.93       124
          H6       0.97      0.93      0.95       112

    accuracy                           0.86      1373
   macro avg       0.89      0.87      0.88      1373
weighted avg       0.86      0.86      0.86      1373
```

#### ShuffleNetV2 1.0x (Test Accuracy: 83.76%, Macro F1: 0.8590)
```text
              precision    recall  f1-score   support
          H1       0.83      0.90      0.86       662
          H2       0.77      0.66      0.71       351
          H3       0.80      0.91      0.85       124
          H5       0.96      0.89      0.92       124
          H6       0.98      0.91      0.94       112

    accuracy                           0.84      1373
   macro avg       0.87      0.85      0.86      1373
weighted avg       0.84      0.84      0.84      1373
```

---

## 6. Generated Visualizations & Artifacts

- **MobileNetV2 (`results/mobilenet_v2/`):**
  - Checkpoint: [`mobilenet_v2_best.pth`](results/mobilenet_v2/mobilenet_v2_best.pth) (8.740 MiB)
  - Loss Curves: [`loss_curves.png`](results/mobilenet_v2/loss_curves.png)
  - Training & Accuracy Curves: [`training_curves.png`](results/mobilenet_v2/training_curves.png)
  - Confusion Matrix: [`confusion_matrix.png`](results/mobilenet_v2/confusion_matrix.png)
  - Classification Report: [`classification_report.txt`](results/mobilenet_v2/classification_report.txt)
  - History Log: [`training_history.json`](results/mobilenet_v2/training_history.json)

- **ShuffleNetV2 (`results/shufflenet_v2/`):**
  - Checkpoint: [`shufflenet_v2_best.pth`](results/shufflenet_v2/shufflenet_v2_best.pth) (4.966 MiB)
  - Loss Curves: [`loss_curves.png`](results/shufflenet_v2/loss_curves.png)
  - Training & Accuracy Curves: [`training_curves.png`](results/shufflenet_v2/training_curves.png)
  - Confusion Matrix: [`confusion_matrix.png`](results/shufflenet_v2/confusion_matrix.png)
  - Classification Report: [`classification_report.txt`](results/shufflenet_v2/classification_report.txt)
  - History Log: [`training_history.json`](results/shufflenet_v2/training_history.json)

---

## 7. Final Comparison & Trade-off Discussion (Task 6 Alignment)

### 7.1 Accuracy vs. Memory Footprint vs. Computational Cost Triad

1. **Accuracy Superiority (+15.88% jump):**
   - **MobileNetV2** achieves **86.31%** test accuracy and **ShuffleNetV2** achieves **83.76%**, compared to Model B's **70.43%** and Model A's **68.90%**.
   - *Why?* Pre-trained transfer learning equips SOTA models with generic low-level edge/texture feature extractors pre-trained on 1.28 million ImageNet images. These rich visual representations dramatically accelerate convergence on low-resolution microscopic cellular patterns, preventing overfitting on small training samples.

2. **Checkpoint Size & Edge Deployability:**
   - **Custom Model B has the smallest checkpoint:** 37,957 parameters and **0.176 MiB** on disk, compared with 4.966 MiB for ShuffleNetV2 and 8.740 MiB for MobileNetV2.
   - Serialized checkpoint size does not equal peak runtime RAM. Activations, framework overhead, tensor precision, and workspace allocations must be measured on the intended deployment stack before claiming that a model fits a particular MCU or requires no external memory.

3. **Training Time and Computational Cost:**
   - **ShuffleNetV2 had the lowest measured training time per epoch:** 20.59 seconds/epoch, compared with 21.71 seconds for Model B. This is a training measurement on the recorded environment, not an inference-latency benchmark.
   - Under the common profiler convention, Model B, ShuffleNetV2, and MobileNetV2 require 7.262848 M, 11.750752 M, and 24.454912 M Conv/Linear MACs, respectively. MAC counts are analytical and do not capture memory traffic or runtime-specific kernel efficiency.

4. **Conclusion for Edge Practitioners:**
   - **Tight model-budget scenarios:** Model B meets the assignment's sub-100k parameter constraint and achieves 70.43% test accuracy with the smallest checkpoint and MAC count.
   - **Accuracy-oriented edge scenarios:** ShuffleNetV2 and MobileNetV2 provide higher accuracy, but deployment suitability requires target-device measurements of latency and peak RAM. No sub-50 ms or MCU-fit guarantee is made from training time, checkpoint size, or MACs alone.

### 7.2 Common MAC convention

`scripts/profile_models.py` profiles every model with a `1×3×64×64` input. It counts convolution and linear multiply-accumulates only, excludes batch normalization, activations, pooling, residual additions, concatenation, channel shuffle, softmax, and data movement, and reports FLOPs as $2\times$ MACs. These values are calculated operation counts, not measured latency.
