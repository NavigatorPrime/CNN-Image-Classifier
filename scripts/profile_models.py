"""Calculate consistent Conv2d/Linear MACs for the four assignment models.

Convention:
- input shape: 1 x 3 x 64 x 64
- Conv2d MACs: Hout * Wout * Cout * (Cin/groups) * Kh * Kw
- Linear MACs: output elements * input features
- BatchNorm, activation, pooling, residual addition, concatenation, shuffle,
  softmax and data movement are excluded
- one MAC is reported as two FLOPs

These are architecture calculations, not measured latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from model_a import ModelA_StandardCNN
from model_b import ModelB_LightweightCNN
from reproducibility import prepare_run_directory
from sota_models import build_mobilenet_v2, build_shufflenet_v2


INPUT_SHAPE = (1, 3, 64, 64)


def profile_model(model: nn.Module) -> dict:
    layer_rows = []
    handles = []
    module_names = {module: name for name, module in model.named_modules()}

    def hook(module: nn.Module, inputs, output: torch.Tensor) -> None:
        if isinstance(module, nn.Conv2d):
            macs = (
                output.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            macs = output.numel() * module.in_features
        else:
            return
        layer_rows.append(
            {
                "layer": module_names[module],
                "type": type(module).__name__,
                "output_shape": list(output.shape),
                "macs": int(macs),
            }
        )

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(INPUT_SHAPE))
    model.train(was_training)
    for handle in handles:
        handle.remove()

    macs = sum(row["macs"] for row in layer_rows)
    return {
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(output.shape),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "macs": macs,
        "flops_2_per_mac": 2 * macs,
        "layers": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="Output directory; defaults to a unique runs/computational_cost directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty explicit output directory",
    )
    args = parser.parse_args()
    results_dir = prepare_run_directory(
        REPO_DIR, "computational_cost", args.output_dir, args.overwrite
    )
    models = {
        "Model A": ModelA_StandardCNN(),
        "Model B": ModelB_LightweightCNN(),
        "MobileNetV2": build_mobilenet_v2(pretrained=False),
        "ShuffleNetV2 1.0x": build_shufflenet_v2(pretrained=False),
    }
    expected_parameters = {
        "Model A": 2_553_349,
        "Model B": 37_957,
        "MobileNetV2": 2_230_277,
        "ShuffleNetV2 1.0x": 1_258_729,
    }

    profiles = {}
    for name, model in models.items():
        profiles[name] = profile_model(model)
        assert profiles[name]["total_parameters"] == expected_parameters[name]
    assert profiles["Model B"]["trainable_parameters"] <= 100_000

    payload = {
        "status": "calculated; not latency",
        "convention": {
            "input": "1x3x64x64",
            "included": "Conv2d and Linear multiply-accumulates",
            "excluded": (
                "BatchNorm, activations, pooling, residual additions, concatenation, "
                "channel shuffle, softmax and data movement"
            ),
            "flops": "2 FLOPs per MAC",
        },
        "models": profiles,
    }
    (results_dir / "model_profiles.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fields = [
        "model",
        "total_parameters",
        "trainable_parameters",
        "macs",
        "million_macs",
        "flops_2_per_mac",
        "million_flops",
    ]
    with (results_dir / "model_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, profile in profiles.items():
            writer.writerow(
                {
                    "model": name,
                    "total_parameters": profile["total_parameters"],
                    "trainable_parameters": profile["trainable_parameters"],
                    "macs": profile["macs"],
                    "million_macs": profile["macs"] / 1_000_000,
                    "flops_2_per_mac": profile["flops_2_per_mac"],
                    "million_flops": profile["flops_2_per_mac"] / 1_000_000,
                }
            )

    for name, profile in profiles.items():
        print(
            f"{name:20} {profile['total_parameters']:>10,} params | "
            f"{profile['macs'] / 1_000_000:>9.6f} M MACs | "
            f"{profile['flops_2_per_mac'] / 1_000_000:>9.6f} M FLOPs"
        )
    print(f"Profiles saved to: {results_dir}")


if __name__ == "__main__":
    main()
