# Stage 3-4 Clean-Split Completion Summary

All values in this summary come from the corrected leakage-free artifacts in
`results_clean_split/`. The superseded outputs in `results/` are historical and
must not be used for final conclusions.

## Stage 3 - Final figures and tables

`scripts/generate_final_artifacts.py` loads the saved histories,
classification reports, checkpoints, profiler output, and optimizer summaries.
It generated the canonical Task 1, 3, 4, 5, and 6 figures under
`results_clean_split/final_artifacts/figures/` and matching CSV, JSON, and LaTeX
tables under `results_clean_split/final_artifacts/tables/`.

The architecture tables are derived from the model implementations by
`scripts/generate_architecture_tables.py`; report metrics and ratios are loaded
and calculated by `scripts/generate_report_data.py`.

No latency chart is presented because there is no controlled device-level
inference-latency measurement. Training time is not used as an inference proxy.

## Stage 4 - Reports

The following PDFs were compiled from the canonical clean-split sources and
rendered page by page for visual inspection:

- `reports/Draco_A03_EN3150.pdf`
- `reports/Section_6_Final_Comparison.pdf`

The inspection found no clipping, overlap, table overflow, unreadable plot
labels, broken equations or references, unintended blank pages, or stale final
metrics.

## Corrected headline metrics

| Model | Test accuracy | Test loss | Macro F1 | Parameters | Checkpoint | MACs |
|---|---:|---:|---:|---:|---:|---:|
| Model A | 67.64% | 0.8360 | 0.6254 | 2,553,349 | 9.755 MiB | 62.33 M |
| Model B | 70.57% | 0.6755 | 0.6800 | 37,957 | 0.176 MiB | 7.26 M |
| MobileNetV2 | 86.09% | 0.3581 | 0.8822 | 2,230,277 | 8.740 MiB | 24.45 M |
| ShuffleNetV2 1.0x | 83.46% | 0.3968 | 0.8606 | 1,258,729 | 4.966 MiB | 11.75 M |

The controlled optimizer comparison selected Adam: 70.57% test accuracy and
0.6800 macro F1, compared with 68.81% / 0.6563 for SGD with momentum and
65.67% / 0.5844 for standard SGD. All three runs used the same clean split,
initial weights, augmentation, loss, batch size, and checkpoint rule for 20
epochs.

## Remaining external action

The assignment dataset information sheet is external to this repository and
must be updated and submitted separately. Publishing this corrected branch, if
required, must not merge it into another branch.
