"""Generate LaTeX result macros from canonical clean-split artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def exact_metrics(summary: dict) -> dict[str, float]:
    report = summary["classification_report"]
    macro = report["macro avg"]
    return {
        "accuracy": report["accuracy"] * 100,
        "test_loss": summary["test_loss"],
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1-score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results_clean_split")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "generated_results.tex")
    args = parser.parse_args()
    results = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    output = args.output if args.output.is_absolute() else ROOT / args.output

    summaries = {
        "ModelA": load_json(results / "model_a" / "training_history.json"),
        "ModelB": load_json(results / "model_b" / "training_history.json"),
        "Mobile": load_json(results / "mobilenet_v2" / "training_history.json"),
        "Shuffle": load_json(results / "shufflenet_v2" / "training_history.json"),
    }
    metrics = {name: exact_metrics(summary) for name, summary in summaries.items()}
    profiles = pd.read_csv(results / "computational_cost" / "model_profiles.csv").set_index("model")
    profile_names = {
        "ModelA": "Model A", "ModelB": "Model B",
        "Mobile": "MobileNetV2", "Shuffle": "ShuffleNetV2 1.0x",
    }
    checkpoint_names = {
        "ModelA": "model_a/model_a_best.pth",
        "ModelB": "model_b/model_b_best.pth",
        "Mobile": "mobilenet_v2/mobilenet_v2_best.pth",
        "Shuffle": "shufflenet_v2/shufflenet_v2_best.pth",
    }
    sizes_mib = {
        name: (results / relative).stat().st_size / (1024**2)
        for name, relative in checkpoint_names.items()
    }

    manifest = pd.read_csv(ROOT / "split_manifest.csv")
    split_counts = manifest["split"].value_counts().to_dict()
    class_counts = manifest.groupby(["class", "split"]).size().to_dict()

    optimizer = load_json(results / "optimizer_comparison" / "optimizer_comparison.json")["results"]
    optimizers = {item["optimizer_id"]: item for item in optimizer}

    commands: dict[str, str] = {}

    def set_command(name: str, value: object) -> None:
        commands[name] = str(value)

    set_command("DatasetTotal", f"{len(manifest):,}")
    set_command("DatasetTrain", f"{split_counts['train']:,}")
    set_command("DatasetValidation", f"{split_counts['val']:,}")
    set_command("DatasetTest", f"{split_counts['test']:,}")
    set_command("DatasetTrainPct", f"{100 * split_counts['train'] / len(manifest):.2f}")
    set_command("DatasetValidationPct", f"{100 * split_counts['val'] / len(manifest):.2f}")
    set_command("DatasetTestPct", f"{100 * split_counts['test'] / len(manifest):.2f}")
    class_command_names = {
        "H1": "HOne", "H2": "HTwo", "H3": "HThree",
        "H5": "HFive", "H6": "HSix",
    }
    for class_name, safe in class_command_names.items():
        set_command(f"{safe}Train", f"{class_counts[(class_name, 'train')]:,}")
        set_command(f"{safe}Validation", f"{class_counts[(class_name, 'val')]:,}")
        set_command(f"{safe}Test", f"{class_counts[(class_name, 'test')]:,}")

    for name, summary in summaries.items():
        profile = profiles.loc[profile_names[name]]
        item = metrics[name]
        set_command(f"{name}Params", f"{int(profile['total_parameters']):,}")
        set_command(f"{name}Accuracy", f"{item['accuracy']:.2f}")
        set_command(f"{name}TestLoss", f"{item['test_loss']:.4f}")
        set_command(f"{name}Precision", f"{item['precision']:.4f}")
        set_command(f"{name}Recall", f"{item['recall']:.4f}")
        set_command(f"{name}FOne", f"{item['f1']:.4f}")
        set_command(f"{name}SizeMiB", f"{sizes_mib[name]:.3f}")
        set_command(f"{name}SizeKiB", f"{sizes_mib[name] * 1024:.1f}")
        set_command(f"{name}TotalTime", f"{summary['total_training_time_seconds']:.2f}")
        set_command(f"{name}EpochTime", f"{summary['avg_epoch_time_seconds']:.2f}")
        set_command(f"{name}BestEpoch", summary["best_epoch"])
        set_command(f"{name}MACs", f"{profile['million_macs']:.3f}")
        set_command(f"{name}FLOPs", f"{profile['million_flops']:.3f}")
        set_command(
            f"{name}HTwoRecall", f"{summary['classification_report']['H2']['recall']:.4f}"
        )

    optimizer_commands = {
        "sgd": "SGD", "sgd_momentum": "Momentum", "adam": "Adam"
    }
    for key, prefix in optimizer_commands.items():
        item = optimizers[key]
        set_command(f"{prefix}LearningRate", f"{item['learning_rate']:.3f}")
        set_command(f"{prefix}Momentum", "--" if item["momentum"] is None else f"{item['momentum']:.1f}")
        set_command(f"{prefix}BestVal", f"{item['best_validation_accuracy'] * 100:.2f}")
        set_command(f"{prefix}BestEpoch", item["best_epoch"])
        set_command(f"{prefix}FinalTrainLoss", f"{item['final_train_loss']:.4f}")
        set_command(f"{prefix}FinalValLoss", f"{item['final_validation_loss']:.4f}")
        set_command(f"{prefix}TestLoss", f"{item['test_loss']:.4f}")
        set_command(f"{prefix}TestAccuracy", f"{item['test_accuracy'] * 100:.2f}")
        set_command(f"{prefix}Precision", f"{item['macro_precision']:.4f}")
        set_command(f"{prefix}Recall", f"{item['macro_recall']:.4f}")
        set_command(f"{prefix}FOne", f"{item['macro_f1']:.4f}")
        set_command(
            f"{prefix}EpochTenVal",
            f"{item['history']['validation_accuracy'][9] * 100:.2f}",
        )

    params = {name: int(profiles.loc[profile_names[name], "total_parameters"]) for name in summaries}
    macs = {name: float(profiles.loc[profile_names[name], "million_macs"]) for name in summaries}
    set_command("ModelBAccuracyGainVsA", f"{metrics['ModelB']['accuracy'] - metrics['ModelA']['accuracy']:.2f}")
    set_command("ModelBParamReductionVsA", f"{100 * (1 - params['ModelB'] / params['ModelA']):.2f}")
    set_command("ModelBParamFactorVsA", f"{params['ModelA'] / params['ModelB']:.2f}")
    set_command("ModelBMACReductionVsA", f"{100 * (1 - macs['ModelB'] / macs['ModelA']):.2f}")
    for name in ["Mobile", "Shuffle"]:
        set_command(
            f"{name}AccuracyGainVsB",
            f"{metrics[name]['accuracy'] - metrics['ModelB']['accuracy']:.2f}",
        )
        set_command(f"{name}ParamFactorVsB", f"{params[name] / params['ModelB']:.2f}")
        set_command(f"{name}SizeFactorVsB", f"{sizes_mib[name] / sizes_mib['ModelB']:.2f}")
        set_command(f"{name}MACFactorVsB", f"{macs[name] / macs['ModelB']:.2f}")

    lines = [
        "% Generated from results_clean_split by scripts/generate_report_data.py.",
        "% Do not edit metric values manually.",
    ]
    lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in commands.items()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {len(commands)} LaTeX macros in {output}")


if __name__ == "__main__":
    main()
