# Model A - Conventional CNN Baseline

This document describes the implemented `ModelA` architecture and its canonical
leakage-free 20-epoch result. Numeric results are sourced from
`results_clean_split/model_a/`; the complete generated architecture table is in
`results_clean_split/final_artifacts/tables/model_a_architecture.csv`.

## Architecture

Model A uses four standard 3x3 convolution stages. Each convolution is
followed by BatchNorm, ReLU, and 2x2 max pooling. The fourth block produces a
256x4x4 tensor. This is flattened and
passed through `4096 -> 512 -> 128 -> 5` linear layers, with ReLU and dropout
after the first two linear layers. The final layer emits five linear logits for
`CrossEntropyLoss`.

| Stage | Operation | Output | Trainable parameters |
|---|---|---:|---:|
| 1 | Conv 3->32, BN, ReLU, pool | 32x32x32 | 960 |
| 2 | Conv 32->64, BN, ReLU, pool | 64x16x16 | 18,624 |
| 3 | Conv 64->128, BN, ReLU, pool | 128x8x8 | 74,112 |
| 4 | Conv 128->256, BN, ReLU, pool | 256x4x4 | 295,680 |
| Classifier 1 | Linear 4096->512, ReLU, dropout | 512 | 2,097,664 |
| Classifier 2 | Linear 512->128, ReLU, dropout | 128 | 65,664 |
| Output | Linear 128->5 logits | 5 | 645 |
| **Total** | | | **2,553,349** |

For a biased convolution, parameters are
`C_out x (C_in x K_h x K_w + 1)`. BatchNorm contributes two trainable values
per output channel. A linear layer contributes `N_in x N_out + N_out`.

ReLU is a simple piecewise-linear threshold. It avoids the exponential
operations of sigmoid and tanh, but this architectural observation is not a
device-latency measurement.

## Canonical training command

```bash
python model_a.py --epochs 20 --output-dir runs/leakage_free/model_a
```

The training script refuses to overwrite a non-empty directory unless
`--overwrite` is explicitly supplied.

## Corrected clean-split result

| Metric | Value |
|---|---:|
| Epochs | 20 |
| Best checkpoint epoch | 11 |
| Test accuracy | 67.64% |
| Test loss | 0.8360 |
| Macro precision | 0.6820 |
| Macro recall | 0.6360 |
| Macro F1 | 0.6254 |
| Trainable parameters | 2,553,349 |
| Checkpoint size | 9.755 MiB |
| Total training time | 346.43 s |
| Average time per epoch | 17.32 s |
| Conv/Linear MACs | 62.325 M |
| FLOPs (2 per MAC) | 124.651 M |

The checkpoint, history, confusion matrix, predictions, and classification
report are under `results_clean_split/model_a/`. The MAC convention uses a
`1x3x64x64` input and excludes BatchNorm, activations, pooling, and data
movement. Checkpoint size and calculated MACs do not establish runtime RAM or
inference latency on a target device.
