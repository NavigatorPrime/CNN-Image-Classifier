# Stage 2 Leakage-Free Retraining Record

Stage 2 retrained all seven required experiments on the corrected common split.
`results_clean_split/` is the canonical result tree. The older `results/` tree
is intentionally preserved as historical contaminated-split evidence.

## Dataset identity

- Classes: `H1`, `H2`, `H3`, `H5`, `H6`
- Image format: 64x64 RGB
- Seed: 42
- Train/validation/test: 6,380 / 1,368 / 1,366
- Manifest SHA-256: `7725325230ba965cefc254ac0a6fd641f423718bfb2ef489e3e852b16da38898`
- Package SHA-256: `9681be856dda4db013e0ff25611b79b589d8a882f5e5718777e407a46b6b1f84`
- Raw-content duplicate groups: 59; cross-split: 0
- Decoded-pixel duplicate groups: 59; cross-split: 0

## Canonical model results

| Model | Parameters | Test accuracy | Test loss | Macro precision | Macro recall | Macro F1 | Checkpoint | Total time | Time/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A | 2,553,349 | 67.6428% | 0.835983 | 0.681969 | 0.635960 | 0.625374 | 9.7546 MiB | 346.43 s | 17.3190 s |
| Model B | 37,957 | 70.5710% | 0.675528 | 0.741611 | 0.660633 | 0.680046 | 0.1755 MiB | 253.80 s | 12.6886 s |
| MobileNetV2 | 2,230,277 | 86.0900% | 0.3581 | 0.8824 | 0.8825 | 0.8822 | 8.7396 MiB | 753.41 s | 37.66 s |
| ShuffleNetV2 1.0x | 1,258,729 | 83.4600% | 0.3968 | 0.8705 | 0.8519 | 0.8606 | 4.9655 MiB | 482.69 s | 24.13 s |

Model A and Model B selected the highest-validation-accuracy checkpoint.
MobileNetV2 and ShuffleNetV2 selected the lowest-validation-loss checkpoint.
The test set was evaluated only after the selected checkpoint was restored.

## Controlled Model B optimizer comparison

All three trials used the same initial weights, seed, batch size, split,
augmentation, loss, weight decay, scheduler method, and highest-validation-
accuracy checkpoint rule. Their common initial-state SHA-256 was
`0542a27459ffc53ee21ce37271c7fc13cc6fbcff08e68a6f684a5948f6472730`.

| Optimizer | LR | Momentum | Best val accuracy | Best epoch | Final train loss | Final val loss | Test loss | Test accuracy | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard SGD | 0.010 | 0.0 | 66.0088% | 18 | 0.896086 | 0.838369 | 0.835791 | 65.6662% | 0.670471 | 0.571852 | 0.584383 |
| SGD + Momentum | 0.010 | 0.9 | 70.5409% | 18 | 0.793831 | 0.716528 | 0.725504 | 68.8141% | 0.672336 | 0.659399 | 0.656298 |
| Adam | 0.001 | n/a | 73.2456% | 20 | 0.751047 | 0.667148 | 0.675528 | 70.5710% | 0.741611 | 0.660633 | 0.680046 |

## Computational-cost convention

Profiles use an input of `1 x 3 x 64 x 64`, count only Conv2d and Linear MACs,
and report FLOPs as exactly `2 x MACs`.

| Model | MACs | FLOPs |
|---|---:|---:|
| Model A | 62.325376 M | 124.650752 M |
| Model B | 7.262848 M | 14.525696 M |
| MobileNetV2 | 24.454912 M | 48.909824 M |
| ShuffleNetV2 1.0x | 11.750752 M | 23.501504 M |

## Artifact locations

- Model results: `results_clean_split/model_a/`, `model_b/`,
  `mobilenet_v2/`, and `shufflenet_v2/`
- Optimizer results: `results_clean_split/optimizer_comparison/`
- Profiles: `results_clean_split/computational_cost/`
- Generated final tables and figures: `results_clean_split/final_artifacts/`
- Machine-readable audit: `results_clean_split/validation_report.json`

Run the complete audit from the repository root:

```bash
python scripts/validate_clean_results.py \
  --results-root results_clean_split
```

The required terminal status is:

```text
CROSS-SPLIT DUPLICATES = 0
STAGE 2 ARTIFACT VALIDATION = PASS
```
