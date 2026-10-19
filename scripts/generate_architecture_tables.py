"""Generate Model A and Model B architecture tables from their implementations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_a import ModelA_StandardCNN
from model_b import ModelB_LightweightCNN


def direct_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def model_a_rows(model: ModelA_StandardCNN) -> list[dict]:
    rows = []
    spatial = 64
    in_channels = 3
    for block_index in range(4):
        offset = block_index * 4
        conv = model.features[offset]
        batch_norm = model.features[offset + 1]
        activation = model.features[offset + 2]
        pool = model.features[offset + 3]
        spatial //= int(pool.stride)
        params = direct_parameters(conv) + direct_parameters(batch_norm)
        rows.append(
            {
                "Stage": f"Block {block_index + 1}",
                "Layer sequence": "Conv2D -> BatchNorm -> ReLU -> MaxPool",
                "Input channels/features": in_channels,
                "Output channels/filters": conv.out_channels,
                "Kernel": f"{conv.kernel_size[0]}x{conv.kernel_size[1]}; pool 2x2",
                "Stride": f"{conv.stride[0]}; pool {pool.stride}",
                "Padding": conv.padding[0],
                "Output": f"{conv.out_channels}x{spatial}x{spatial}",
                "Trainable parameters": params,
                "Parameter detail": (
                    f"Conv {direct_parameters(conv):,} + BN {direct_parameters(batch_norm):,}; "
                    f"{type(activation).__name__}/pool 0"
                ),
            }
        )
        in_channels = conv.out_channels

    fc1, fc2, fc3 = model.classifier[1], model.classifier[4], model.classifier[7]
    rows.extend(
        [
            {
                "Stage": "Flatten",
                "Layer sequence": "Flatten",
                "Input channels/features": "256x4x4",
                "Output channels/filters": 4096,
                "Kernel": "-", "Stride": "-", "Padding": "-",
                "Output": "4096", "Trainable parameters": 0,
                "Parameter detail": "No trainable parameters",
            },
            {
                "Stage": "FC1",
                "Layer sequence": "Linear -> ReLU -> Dropout(0.5)",
                "Input channels/features": fc1.in_features,
                "Output channels/filters": fc1.out_features,
                "Kernel": "-", "Stride": "-", "Padding": "-",
                "Output": str(fc1.out_features),
                "Trainable parameters": direct_parameters(fc1),
                "Parameter detail": f"{fc1.in_features}x{fc1.out_features}+{fc1.out_features}",
            },
            {
                "Stage": "FC2",
                "Layer sequence": "Linear -> ReLU -> Dropout(0.3)",
                "Input channels/features": fc2.in_features,
                "Output channels/filters": fc2.out_features,
                "Kernel": "-", "Stride": "-", "Padding": "-",
                "Output": str(fc2.out_features),
                "Trainable parameters": direct_parameters(fc2),
                "Parameter detail": f"{fc2.in_features}x{fc2.out_features}+{fc2.out_features}",
            },
            {
                "Stage": "Output",
                "Layer sequence": "Linear logits",
                "Input channels/features": fc3.in_features,
                "Output channels/filters": fc3.out_features,
                "Kernel": "-", "Stride": "-", "Padding": "-",
                "Output": str(fc3.out_features),
                "Trainable parameters": direct_parameters(fc3),
                "Parameter detail": f"{fc3.in_features}x{fc3.out_features}+{fc3.out_features}",
            },
        ]
    )
    return rows


def model_b_rows(model: ModelB_LightweightCNN) -> list[dict]:
    stem_conv, stem_bn = model.stem[0], model.stem[1]
    rows = [
        {
            "Stage": "Stem",
            "Layer sequence": "Conv2D -> BatchNorm -> ReLU6",
            "Input channels": stem_conv.in_channels,
            "Depthwise channels": "-",
            "Groups": stem_conv.groups,
            "Kernel": "3x3",
            "Stride": stem_conv.stride[0],
            "Padding": stem_conv.padding[0],
            "Pointwise output": stem_conv.out_channels,
            "Output": "16x64x64",
            "Trainable parameters": direct_parameters(stem_conv) + direct_parameters(stem_bn),
            "Parameter detail": "Conv 432 + BN 32",
        }
    ]
    blocks = [model.block1, model.block2, model.block3, model.block4, model.block5, model.block6]
    spatial = 64
    for index, block in enumerate(blocks, start=1):
        depthwise, pointwise = block.depthwise, block.pointwise
        spatial = (spatial + 2 * depthwise.padding[0] - depthwise.kernel_size[0]) // depthwise.stride[0] + 1
        dw_conv = direct_parameters(depthwise)
        dw_bn = direct_parameters(block.bn_dw)
        pw_conv = direct_parameters(pointwise)
        pw_bn = direct_parameters(block.bn_pw)
        rows.append(
            {
                "Stage": f"DS Block {index}",
                "Layer sequence": "DW Conv -> BN -> ReLU6; PW Conv -> BN -> ReLU6",
                "Input channels": depthwise.in_channels,
                "Depthwise channels": depthwise.out_channels,
                "Groups": depthwise.groups,
                "Kernel": "3x3 DW; 1x1 PW",
                "Stride": depthwise.stride[0],
                "Padding": depthwise.padding[0],
                "Pointwise output": pointwise.out_channels,
                "Output": f"{pointwise.out_channels}x{spatial}x{spatial}",
                "Trainable parameters": dw_conv + dw_bn + pw_conv + pw_bn,
                "Parameter detail": (
                    f"DW {dw_conv:,} + BN {dw_bn:,} + PW {pw_conv:,} + BN {pw_bn:,}"
                ),
            }
        )
    classifier = model.classifier
    rows.extend(
        [
            {
                "Stage": "GAP",
                "Layer sequence": "AdaptiveAvgPool2D(1x1)",
                "Input channels": 128, "Depthwise channels": "-", "Groups": "-",
                "Kernel": "global", "Stride": "-", "Padding": "-",
                "Pointwise output": 128, "Output": "128x1x1",
                "Trainable parameters": 0, "Parameter detail": "No trainable parameters",
            },
            {
                "Stage": "Dropout",
                "Layer sequence": "Dropout(0.2)",
                "Input channels": 128, "Depthwise channels": "-", "Groups": "-",
                "Kernel": "-", "Stride": "-", "Padding": "-",
                "Pointwise output": 128, "Output": "128",
                "Trainable parameters": 0, "Parameter detail": "No trainable parameters",
            },
            {
                "Stage": "Classifier",
                "Layer sequence": "Linear logits",
                "Input channels": classifier.in_features,
                "Depthwise channels": "-", "Groups": "-",
                "Kernel": "-", "Stride": "-", "Padding": "-",
                "Pointwise output": classifier.out_features,
                "Output": str(classifier.out_features),
                "Trainable parameters": direct_parameters(classifier),
                "Parameter detail": f"{classifier.in_features}x{classifier.out_features}+{classifier.out_features}",
            },
        ]
    )
    return rows


def write_table(rows: list[dict], stem: str, output: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{stem}.csv", index=False)
    (output / f"{stem}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (output / f"{stem}.tex").write_text(
        frame.to_latex(index=False, escape=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results_clean_split" / "final_artifacts" / "tables",
    )
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    model_a = ModelA_StandardCNN()
    model_b = ModelB_LightweightCNN()
    a_rows = model_a_rows(model_a)
    b_rows = model_b_rows(model_b)
    assert sum(row["Trainable parameters"] for row in a_rows) == 2_553_349
    assert sum(row["Trainable parameters"] for row in b_rows) == 37_957
    write_table(a_rows, "model_a_architecture", output)
    write_table(b_rows, "model_b_architecture", output)
    print(f"Generated architecture tables in {output}")


if __name__ == "__main__":
    main()
