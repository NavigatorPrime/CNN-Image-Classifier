# Stage 3-4 Completion Summary

## Stage 3 - Final figures and tables
Generated from the verified Stage-2 results without retraining any model.

### Figures
- Task 1: representative 64x64 samples and class distribution
- Task 3: controlled SGD / SGD+Momentum / Adam convergence comparison
- Task 4: custom-model training curves and both confusion matrices
- Task 5: SOTA fine-tuning curves and both confusion matrices
- Task 6: accuracy, parameter count, checkpoint size, MAC comparison, accuracy-vs-size Pareto plot, and accuracy-vs-MAC Pareto plot

### Tables
- `tables/optimizer_comparison.csv/.tex`
- `tables/model_a_vs_b.csv/.tex`
- `tables/sota_comparison.csv/.tex`
- `tables/final_model_comparison.csv/.tex`
- `tables/derived_tradeoffs.json`

No latency chart is generated because Stage 2 contains no valid controlled device-level inference-latency measurement. Training time is not used as an inference proxy.

## Stage 4 - Report
Generated and compiled:
- `reports/Draco_A03_EN3150.tex`
- `reports/Draco_A03_EN3150.pdf`
- `reports/Section_6_Final_Comparison.tex`
- `reports/Section_6_Final_Comparison.pdf`

Both PDFs were rendered to images and visually checked for clipping and major layout issues.

## Final headline metrics
| Model | Test Accuracy | Macro F1 | Parameters | Checkpoint Size | MACs |
|---|---:|---:|---:|---:|---:|
| Model B | 70.43% | 0.6927 | 37,957 | 0.176 MiB | 7.26 M |
| ShuffleNetV2 1.0x | 83.76% | 0.8590 | 1,258,729 | 4.966 MiB | 11.75 M |
| MobileNetV2 | 86.31% | 0.8805 | 2,230,277 | 8.740 MiB | 24.45 M |

Task 3 controlled optimizer results select Adam: 70.14% test accuracy and 0.6737 macro F1, compared with 67.23% / 0.6356 for SGD+Momentum and 65.91% / 0.5770 for standard SGD.

## Remaining action
The corrected branch still needs to be published to GitHub by a connection/account with branch creation and push permission. A final independent audit can then be run if desired.
