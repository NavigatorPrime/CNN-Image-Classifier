# Task 5 - Lightweight Pretrained Model Evaluation

This canonical evaluation uses only the leakage-free artifacts in
`results_clean_split/`. MobileNetV2 and ShuffleNetV2 1.0x were initialized from
torchvision's ImageNet default weights, adapted to five output classes, and
fine-tuned for 20 epochs on the same 6,380/1,368/1,366 split used by both custom
models. No identical raw or decoded image content crosses the splits.

## Architectures and setup

MobileNetV2 uses inverted residual blocks, depthwise convolution, and linear
bottlenecks. ShuffleNetV2 uses channel splitting, a transformed branch,
concatenation, and channel shuffle. Their final classifiers were replaced with
five-output linear layers.

Both used 64x64 RGB input, ImageNet normalization, the committed augmentation
pipeline, Adam with learning rate 0.0005 and weight decay 0.0001, and
`ReduceLROnPlateau` with factor 0.5 and patience 3. The checkpoints evaluated
for both pretrained models were selected by lowest validation loss.

## Corrected clean-split results

| Model | Test loss | Test accuracy | Macro precision | Macro recall | Macro F1 | Parameters | Checkpoint | Total time | Time/epoch | MACs | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.3581 | 86.09% | 0.8824 | 0.8825 | 0.8822 | 2,230,277 | 8.740 MiB | 753.41 s | 37.66 s | 24.455 M | 48.910 M |
| ShuffleNetV2 1.0x | 0.3968 | 83.46% | 0.8705 | 0.8519 | 0.8606 | 1,258,729 | 4.966 MiB | 482.69 s | 24.13 s | 11.751 M | 23.502 M |

The complete histories, checkpoints, confusion matrices, predictions, and
classification reports are in `results_clean_split/mobilenet_v2/` and
`results_clean_split/shufflenet_v2/`.

## Comparison with Model B

| Comparison relative to Model B | MobileNetV2 | ShuffleNetV2 1.0x |
|---|---:|---:|
| Accuracy gain | 15.52 pp | 12.88 pp |
| Parameter ratio | 58.76x | 33.16x |
| Checkpoint-size ratio | 49.79x | 28.29x |
| MAC ratio | 3.37x | 1.62x |

The pretrained models provide substantially higher accuracy, but the result
cannot be attributed to pretraining alone: model capacity and architecture also
differ. Model B remains much smaller and cheaper by the stated operation-count
convention. Checkpoint storage, parameter count, and calculated MACs are
distinct from peak runtime RAM and measured latency; no ESP32/STM32 deployment
or sub-50 ms inference claim is made.

## Common profiling convention

The profiler uses input `1x3x64x64`, counts Conv2d and Linear MACs, excludes
BatchNorm, activations, pooling, residual additions, concatenation, channel
shuffle, softmax, and data movement, and reports FLOPs as twice the MAC count.
