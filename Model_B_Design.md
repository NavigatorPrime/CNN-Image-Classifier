# Model B - Sub-100k Depthwise-Separable CNN

This document describes the implemented `ModelB` architecture and its canonical
leakage-free 20-epoch result. Numeric results are sourced from
`results_clean_split/model_b/`; the implementation-derived table is in
`results_clean_split/final_artifacts/tables/model_b_architecture.csv`.

## Architecture

The model has a 3x3 standard-convolution stem followed by six
depthwise-separable blocks. Each block applies a 3x3 depthwise convolution
(`groups = input channels`), BatchNorm and ReLU6, then a 1x1 pointwise
convolution, BatchNorm and ReLU6. Stride-2 blocks reduce the spatial resolution.
Global average pooling produces one value per final channel; dropout precedes a
linear five-class classifier. The classifier output is linear logits before
`CrossEntropyLoss`.

| Stage | Channels | Stride | Output | Trainable parameters |
|---|---:|---:|---:|---:|
| Stem | 3->16 | 1 | 16x64x64 | 464 |
| DS block 1 | 16->32 | 2 | 32x32x32 | 752 |
| DS block 2 | 32->32 | 1 | 32x32x32 | 1,440 |
| DS block 3 | 32->64 | 2 | 64x16x16 | 2,528 |
| DS block 4 | 64->64 | 1 | 64x16x16 | 4,928 |
| DS block 5 | 64->128 | 2 | 128x8x8 | 9,152 |
| DS block 6 | 128->128 | 1 | 128x8x8 | 18,048 |
| GAP + dropout | 128 | - | 128 | 0 |
| Classifier | 128->5 | - | 5 logits | 645 |
| **Total** | | | | **37,957** |

For input channels `M`, output channels `N`, and a `KxK` kernel, the implemented
bias-free depthwise convolution contributes `M x K^2` parameters and the
bias-free pointwise convolution contributes `M x N`. Each BatchNorm contributes
two trainable values per output channel. The stem convolution is also
bias-free; the final classifier includes bias and contributes `128x5+5=645`.
This accounting covers both 32-output-channel blocks and all other blocks.

Thus **Model B trainable parameters = 37,957**, which is below the assignment
limit of 100,000.

ReLU6 performs a lower threshold and upper clamp. Like ReLU, it is
piecewise-linear and avoids exponential evaluation. Its bounded output can be
convenient for some quantization pipelines, but no MCU latency, RAM use, or
deployment feasibility was measured here.

## Canonical training command

```bash
python model_b.py --epochs 20 --output-dir runs/leakage_free/model_b
```

## Corrected clean-split result

| Metric | Value |
|---|---:|
| Epochs | 20 |
| Best checkpoint epoch | 20 |
| Test accuracy | 70.57% |
| Test loss | 0.6755 |
| Macro precision | 0.7416 |
| Macro recall | 0.6606 |
| Macro F1 | 0.6800 |
| Trainable parameters | 37,957 |
| Checkpoint size | 0.176 MiB (179.8 KiB) |
| Total training time | 253.80 s |
| Average time per epoch | 12.69 s |
| Conv/Linear MACs | 7.263 M |
| FLOPs (2 per MAC) | 14.526 M |

Against Model A on the same split, Model B is 2.93 percentage points more
accurate while using 98.51% fewer parameters and 88.35% fewer counted MACs.
This makes it the more attractive storage-and-compute candidate in this
experiment, but target-specific runtime RAM and latency still require direct
measurement.
