# Task 3 — Model B Optimizer Comparison

This directory contains the required controlled comparison of standard SGD, SGD with momentum, and Adam on the repository's verified DeFungi split.

## Controlled configuration

- Architecture: unchanged `ModelB_LightweightCNN` (37,957 trainable parameters)
- Split: committed `split_manifest.csv` (6,377 train / 1,364 validation / 1,373 test)
- Input and augmentation: the same Model B `64×64` preprocessing and training augmentation for all runs
- Initial weights and seed: identical state dictionary; seed 42 reset before each run
- Epochs and batch size: 20 epochs, batch size 32
- Loss: cross-entropy
- Weight decay: `1e-4` for all optimizers
- Scheduler: `ReduceLROnPlateau(mode="min", factor=0.5, patience=4)` for all optimizers
- Checkpoint rule: highest validation accuracy; test set evaluated once using that checkpoint
- Standard SGD: learning rate `0.01`, momentum `0.0`
- SGD + Momentum: learning rate `0.01`, momentum `0.9`
- Adam: learning rate `0.001`

## Results

| Optimizer | Best Epoch | Best Validation Accuracy | Final Validation Loss | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard SGD | 19 | 67.01% | 0.8255 | 65.91% | 0.6572 | 0.5713 | 0.5770 |
| SGD + Momentum | 12 | 70.09% | 0.7896 | 67.23% | 0.6776 | 0.6225 | 0.6356 |
| Adam (selected) | 18 | **72.29%** | **0.7504** | **70.14%** | **0.7256** | **0.6436** | **0.6737** |

## Measured effect of momentum

Momentum accelerated optimization relative to standard SGD under the controlled settings. It reached 66.28% validation accuracy by epoch 2 and 70.09% by epoch 12, while standard SGD first reached its lower peak of 67.01% at epoch 19. At epoch 20, momentum also had lower training loss (0.7779 versus 0.8759) and lower validation loss (0.7896 versus 0.8255). Its validation curve was not monotonic, so the measured conclusion is faster and better convergence in this run—not guaranteed improvement at every epoch. Adam produced the best validation and held-out test results of the three.

`optimizer_comparison.json` contains full epoch histories, environment metadata, class-level metrics, and the exact configuration. `optimizer_comparison.csv` is the compact summary; `optimizer_curves.png` plots training loss, validation loss, and validation accuracy.
