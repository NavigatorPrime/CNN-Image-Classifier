"""Generate final figures and tables from recorded experiment artifacts.

No performance number is embedded in this script. Metrics, timings, model
sizes, parameter counts, and computational profiles are loaded from the
selected results tree and the checkpoint files it contains.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = {
    "Model A": "model_a",
    "Model B": "model_b",
    "MobileNetV2": "mobilenet_v2",
    "ShuffleNetV2 1.0x": "shufflenet_v2",
}
CHECKPOINTS = {
    "Model A": "model_a_best.pth",
    "Model B": "model_b_best.pth",
    "MobileNetV2": "mobilenet_v2_best.pth",
    "ShuffleNetV2 1.0x": "shufflenet_v2_best.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Results tree containing the completed experiment artifacts",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Destination root; defaults to a unique generated/final_artifacts run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty explicit output directory",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def prepare_output_root(path: Path | None, overwrite: bool) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "generated" / "final_artifacts" / stamp
    else:
        output = resolve_path(path)
        if output.exists() and any(output.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}. Choose another "
                "--output-root or pass --overwrite explicitly."
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_report(path: Path) -> dict[str, float | int | None]:
    """Parse metrics and timing from a saved classification report."""
    if not path.is_file():
        raise FileNotFoundError(f"Required report not found: {path}")
    report_text = path.read_text(encoding="utf-8")

    def number(pattern: str) -> float | None:
        match = re.search(pattern, report_text, flags=re.IGNORECASE)
        return float(match.group(1).replace(",", "")) if match else None

    metrics: dict[str, float | int | None] = {
        "test_accuracy": number(r"Test Accuracy:\s*([0-9.,]+)%"),
        "test_loss": number(r"Test Loss:\s*([0-9.,]+)"),
        "total_parameters": number(r"Total Parameters:\s*([0-9,]+)"),
        "training_time_seconds": number(r"Training Time:\s*([0-9.,]+)s"),
        "average_epoch_seconds": number(r"Training Time:.*?\(([0-9.,]+)s/epoch\)"),
    }
    macro = re.search(
        r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", report_text
    )
    if macro:
        metrics.update(
            macro_precision=float(macro.group(1)),
            macro_recall=float(macro.group(2)),
            macro_f1=float(macro.group(3)),
        )
    else:
        metrics.update(macro_precision=None, macro_recall=None, macro_f1=None)
    return metrics


def require_number(value: object, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing required numeric result: {label}")
    return float(value)


def metric_from(summary: dict, report: dict, key: str, label: str) -> float:
    return require_number(summary.get(key, report.get(key)), f"{label}.{key}")


def history_from(payload: dict) -> dict:
    history = payload.get("history", payload)
    required = {"train_loss", "val_loss", "train_acc", "val_acc"}
    if not required.issubset(history):
        raise ValueError(f"Training history is missing: {sorted(required - set(history))}")
    lengths = {len(history[key]) for key in required}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("Training-history series must have one equal non-zero length")
    return history


def average_epoch_seconds(summary: dict, report: dict, history: dict, label: str) -> float:
    direct = summary.get("avg_epoch_time_seconds") or report.get("average_epoch_seconds")
    if direct is not None:
        return float(direct)
    epoch_times = history.get("epoch_times")
    if epoch_times:
        return float(np.mean(epoch_times))
    total = summary.get("total_training_time_seconds") or report.get("training_time_seconds")
    if total is not None:
        return float(total) / len(history["train_loss"])
    raise ValueError(f"No recorded training time for {label}")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def df_to_tex(frame: pd.DataFrame, path: Path, float_format: str | None = None) -> None:
    path.write_text(
        frame.to_latex(index=False, escape=True, float_format=float_format),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    results = resolve_path(args.results_root)
    output = prepare_output_root(args.output_root, args.overwrite)
    figures = output / "figures"
    tables = output / "tables"
    for directory in [
        figures / "task1",
        figures / "task3",
        figures / "task4",
        figures / "task5",
        figures / "task6",
        tables,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    opt = pd.read_csv(results / "optimizer_comparison" / "optimizer_comparison.csv")
    profiles = pd.read_csv(results / "computational_cost" / "model_profiles.csv")
    opt_json = load_json(results / "optimizer_comparison" / "optimizer_comparison.json")
    sota = load_json(results / "sota_benchmark_summary.json")
    manifest = pd.read_csv(ROOT / "split_manifest.csv")

    summaries: dict[str, dict] = {
        "Model A": load_json(results / "model_a" / "training_history.json"),
        "Model B": load_json(results / "model_b" / "training_history.json"),
        "MobileNetV2": sota.get("mobilenet_v2", {}),
        "ShuffleNetV2 1.0x": sota.get("shufflenet_v2", {}),
    }
    reports = {
        label: parse_report(results / key / "classification_report.txt")
        for label, key in MODEL_KEYS.items()
    }
    histories = {label: history_from(payload) for label, payload in summaries.items()}

    metrics: dict[str, dict[str, float]] = {}
    for label in MODEL_KEYS:
        summary, report = summaries[label], reports[label]
        report_dict = summary.get("classification_report", {})
        macro = report_dict.get("macro avg", {})
        # Prefer exact metrics reconstructed from saved predictions when they
        # are available. SOTA headline summaries intentionally round to two or
        # four decimals, while classification_report retains the exact values.
        metrics[label] = {
            "test_accuracy": float(report_dict["accuracy"] * 100)
            if "accuracy" in report_dict
            else metric_from(summary, report, "test_accuracy", label),
            "test_loss": metric_from(summary, report, "test_loss", label),
            "macro_precision": float(macro["precision"])
            if "precision" in macro
            else metric_from(summary, report, "macro_precision", label),
            "macro_recall": float(macro["recall"])
            if "recall" in macro
            else metric_from(summary, report, "macro_recall", label),
            "macro_f1": float(macro["f1-score"])
            if "f1-score" in macro
            else metric_from(summary, report, "macro_f1", label),
        }

    checkpoint_paths = {
        label: results / MODEL_KEYS[label] / filename
        for label, filename in CHECKPOINTS.items()
    }
    missing_checkpoints = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing checkpoints: {missing_checkpoints}")
    size_bytes = {label: path.stat().st_size for label, path in checkpoint_paths.items()}
    size_mib = {label: size / (1024**2) for label, size in size_bytes.items()}
    size_kib = {label: size / 1024 for label, size in size_bytes.items()}
    times = {
        label: average_epoch_seconds(summaries[label], reports[label], histories[label], label)
        for label in MODEL_KEYS
    }
    total_times = {
        label: require_number(
            summaries[label].get("total_training_time_seconds"),
            f"{label}.total_training_time_seconds",
        )
        for label in MODEL_KEYS
    }

    profile_by_model = profiles.set_index("model")
    missing_profiles = set(MODEL_KEYS) - set(profile_by_model.index)
    if missing_profiles:
        raise ValueError(f"Missing computational profiles: {sorted(missing_profiles)}")

    counts = (
        manifest.groupby(["class", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"])
    )
    axis = counts.plot(kind="bar", figsize=(9, 5))
    axis.set_title("DeFungi Class Distribution by Split")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of images")
    axis.grid(axis="y", alpha=0.25)
    savefig(figures / "task1" / "class_distribution.png")

    with zipfile.ZipFile(ROOT / "dataset_64x64.zip") as archive:
        names = archive.namelist()
        samples = []
        for class_name in ["H1", "H2", "H3", "H5", "H6"]:
            candidates = sorted(
                name
                for name in names
                if f"/train/{class_name}/" in name
                and name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            samples.append((class_name, candidates[0]))
        figure, axes = plt.subplots(1, 5, figsize=(12, 2.8))
        for axis, (class_name, name) in zip(axes, samples):
            with archive.open(name) as stream:
                with Image.open(stream) as image:
                    axis.imshow(image.convert("RGB"))
            axis.set_title(class_name)
            axis.axis("off")
        figure.suptitle("Representative 64x64 DeFungi Images", y=1.03)
        savefig(figures / "task1" / "representative_samples.png")

    optimizer_results = {row["optimizer"]: row for row in opt_json["results"]}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, row in optimizer_results.items():
        epochs = np.arange(1, len(row["history"]["train_loss"]) + 1)
        axes[0].plot(epochs, row["history"]["train_loss"], marker="o", markersize=2.5, label=name)
        axes[1].plot(epochs, row["history"]["validation_loss"], marker="o", markersize=2.5, label=name)
        axes[2].plot(epochs, np.array(row["history"]["validation_accuracy"]) * 100, marker="o", markersize=2.5, label=name)
    for axis, title, ylabel in zip(
        axes,
        ["Training Loss", "Validation Loss", "Validation Accuracy"],
        ["Cross-entropy loss", "Cross-entropy loss", "Accuracy (%)"],
    ):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("Task 3: Controlled Optimizer Comparison on Model B")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figures / "task3" / "optimizer_comparison_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    def plot_single_history(label: str, destination: Path, title: str) -> None:
        history = histories[label]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
        axes[0].plot(epochs, history["train_loss"], marker="o", markersize=2.5, label="Train")
        axes[0].plot(epochs, history["val_loss"], marker="o", markersize=2.5, label="Validation")
        axes[1].plot(epochs, np.asarray(history["train_acc"]) * 100, marker="o", markersize=2.5, label="Train")
        axes[1].plot(epochs, np.asarray(history["val_acc"]) * 100, marker="o", markersize=2.5, label="Validation")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[1].set_ylabel("Accuracy (%)")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.set_xticks([1, 5, 10, 15, 20])
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)

    plot_single_history(
        "Model A", figures / "task4" / "model_a_training_curves.png",
        "Model A: Training and Validation Curves",
    )
    plot_single_history(
        "Model B", figures / "task4" / "model_b_training_curves.png",
        "Model B: Training and Validation Curves",
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, label in enumerate(["Model A", "Model B"]):
        history = histories[label]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0, column].plot(epochs, history["train_loss"], label=f"{label} train")
        axes[0, column].plot(epochs, history["val_loss"], label=f"{label} validation")
        axes[1, column].plot(epochs, np.array(history["train_acc"]) * 100, label=f"{label} train")
        axes[1, column].plot(epochs, np.array(history["val_acc"]) * 100, label=f"{label} validation")
        axes[0, column].set_title(f"{label} Loss")
        axes[1, column].set_title(f"{label} Accuracy")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    figure.suptitle("Task 4: Custom Model Training and Validation Curves")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures / "task4" / "custom_model_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    plot_single_history(
        "MobileNetV2", figures / "task5" / "mobilenet_v2_training_curves.png",
        "MobileNetV2: Training and Validation Curves",
    )
    plot_single_history(
        "ShuffleNetV2 1.0x", figures / "task5" / "shufflenet_v2_training_curves.png",
        "ShuffleNetV2 1.0x: Training and Validation Curves",
    )

    for label, destination in [
        ("Model A", "model_a_confusion_matrix.png"),
        ("Model B", "model_b_confusion_matrix.png"),
    ]:
        shutil.copy2(results / MODEL_KEYS[label] / "confusion_matrix.png", figures / "task4" / destination)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, label in enumerate(["MobileNetV2", "ShuffleNetV2 1.0x"]):
        history = histories[label]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0, column].plot(epochs, history["train_loss"], label="Train")
        axes[0, column].plot(epochs, history["val_loss"], label="Validation")
        axes[1, column].plot(epochs, np.array(history["train_acc"]) * 100, label="Train")
        axes[1, column].plot(epochs, np.array(history["val_acc"]) * 100, label="Validation")
        axes[0, column].set_title(f"{label} Loss")
        axes[1, column].set_title(f"{label} Accuracy")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    figure.suptitle("Task 5: Fine-Tuning Curves for Lightweight Pre-Trained Models")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures / "task5" / "sota_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    for label, destination in [
        ("MobileNetV2", "mobilenet_v2_confusion_matrix.png"),
        ("ShuffleNetV2 1.0x", "shufflenet_v2_confusion_matrix.png"),
    ]:
        shutil.copy2(results / MODEL_KEYS[label] / "confusion_matrix.png", figures / "task5" / destination)

    opt_table = opt.copy()
    for column in ["best_validation_accuracy", "final_validation_accuracy", "test_accuracy"]:
        if column in opt_table:
            opt_table[column] *= 100
    opt_table = opt_table.rename(
        columns={
            "optimizer": "Optimizer",
            "learning_rate": "Learning Rate",
            "momentum": "Momentum",
            "best_epoch": "Best Epoch",
            "best_validation_accuracy": "Best Val Accuracy (%)",
            "test_accuracy": "Test Accuracy (%)",
            "macro_precision": "Macro Precision",
            "macro_recall": "Macro Recall",
            "macro_f1": "Macro F1",
            "average_epoch_seconds": "Avg Epoch Time (s)",
        }
    )
    opt_columns = [
        "Optimizer",
        "Learning Rate",
        "Momentum",
        "Best Epoch",
        "Best Val Accuracy (%)",
        "final_train_loss",
        "final_validation_loss",
        "Test Accuracy (%)",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Avg Epoch Time (s)",
    ]
    opt_table = opt_table[opt_columns]
    opt_table = opt_table.rename(
        columns={
            "final_train_loss": "Final Train Loss",
            "final_validation_loss": "Final Validation Loss",
        }
    )
    opt_table.to_csv(tables / "optimizer_comparison.csv", index=False, float_format="%.4f")
    df_to_tex(opt_table, tables / "optimizer_comparison.tex", "%.4f")
    (tables / "optimizer_comparison.json").write_text(
        opt_table.to_json(orient="records", indent=2), encoding="utf-8"
    )

    params = {label: int(profile_by_model.loc[label, "total_parameters"]) for label in MODEL_KEYS}
    custom = pd.DataFrame(
        [
            {"Metric": "Total parameters", "Model A": f"{params['Model A']:,}", "Model B": f"{params['Model B']:,}"},
            {"Metric": "Actual checkpoint size", "Model A": f"{size_kib['Model A']:.1f} KiB", "Model B": f"{size_kib['Model B']:.1f} KiB"},
            {"Metric": "FP32 parameter storage", "Model A": f"{params['Model A'] * 4 / 1024:.1f} KiB", "Model B": f"{params['Model B'] * 4 / 1024:.1f} KiB"},
            {"Metric": "Average training time / epoch", "Model A": f"{times['Model A']:.2f} s", "Model B": f"{times['Model B']:.2f} s"},
            {"Metric": "Total training time", "Model A": f"{total_times['Model A']:.2f} s", "Model B": f"{total_times['Model B']:.2f} s"},
            {"Metric": "Test loss", "Model A": f"{metrics['Model A']['test_loss']:.4f}", "Model B": f"{metrics['Model B']['test_loss']:.4f}"},
            {"Metric": "Test accuracy", "Model A": f"{metrics['Model A']['test_accuracy']:.2f}%", "Model B": f"{metrics['Model B']['test_accuracy']:.2f}%"},
            {"Metric": "Macro precision", "Model A": f"{metrics['Model A']['macro_precision']:.4f}", "Model B": f"{metrics['Model B']['macro_precision']:.4f}"},
            {"Metric": "Macro recall", "Model A": f"{metrics['Model A']['macro_recall']:.4f}", "Model B": f"{metrics['Model B']['macro_recall']:.4f}"},
            {"Metric": "Macro F1", "Model A": f"{metrics['Model A']['macro_f1']:.4f}", "Model B": f"{metrics['Model B']['macro_f1']:.4f}"},
            {"Metric": "Calculated MACs (Conv/Linear only)", "Model A": f"{profile_by_model.loc['Model A', 'million_macs']:.2f} M", "Model B": f"{profile_by_model.loc['Model B', 'million_macs']:.2f} M"},
            {"Metric": "Calculated FLOPs (2 per MAC)", "Model A": f"{profile_by_model.loc['Model A', 'million_flops']:.2f} M", "Model B": f"{profile_by_model.loc['Model B', 'million_flops']:.2f} M"},
        ]
    )
    custom.to_csv(tables / "model_a_vs_b.csv", index=False)
    df_to_tex(custom, tables / "model_a_vs_b.tex")
    (tables / "model_a_vs_b.json").write_text(
        custom.to_json(orient="records", indent=2), encoding="utf-8"
    )

    sota_rows = []
    for label in ["MobileNetV2", "ShuffleNetV2 1.0x"]:
        sota_rows.append(
            {
                "Model": label,
                "Parameters": params[label],
                "Checkpoint Size (MiB)": size_mib[label],
                "Test Accuracy (%)": metrics[label]["test_accuracy"],
                "Test Loss": metrics[label]["test_loss"],
                "Macro Precision": metrics[label]["macro_precision"],
                "Macro Recall": metrics[label]["macro_recall"],
                "Macro F1": metrics[label]["macro_f1"],
                "Total Training Time (s)": total_times[label],
                "Avg Epoch Time (s)": times[label],
                "MACs (M)": float(profile_by_model.loc[label, "million_macs"]),
                "FLOPs (M; 2/MAC)": float(profile_by_model.loc[label, "million_flops"]),
            }
        )
    sota_table = pd.DataFrame(sota_rows)
    sota_table.to_csv(tables / "sota_comparison.csv", index=False, float_format="%.4f")
    df_to_tex(sota_table, tables / "sota_comparison.tex", "%.4f")
    (tables / "sota_comparison.json").write_text(
        sota_table.to_json(orient="records", indent=2), encoding="utf-8"
    )

    final_rows = []
    for label in ["Model A", "Model B", "ShuffleNetV2 1.0x", "MobileNetV2"]:
        profile = profile_by_model.loc[label]
        final_rows.append(
            {
                "Model": label,
                "Test Accuracy (%)": metrics[label]["test_accuracy"],
                "Test Loss": metrics[label]["test_loss"],
                "Macro Precision": metrics[label]["macro_precision"],
                "Macro Recall": metrics[label]["macro_recall"],
                "Macro F1": metrics[label]["macro_f1"],
                "Total Parameters": params[label],
                "Checkpoint Size (MiB)": size_mib[label],
                "FP32 Weights (MiB)": params[label] * 4 / (1024**2),
                "MACs (M)": float(profile["million_macs"]),
                "FLOPs (M; 2/MAC)": float(profile["million_flops"]),
                "Avg Epoch Time (s)": times[label],
                "Total Training Time (s)": total_times[label],
            }
        )
    final = pd.DataFrame(final_rows)
    final.to_csv(tables / "final_model_comparison.csv", index=False, float_format="%.4f")
    df_to_tex(final, tables / "final_model_comparison.tex", "%.4f")
    (tables / "final_model_comparison.json").write_text(
        final.to_json(orient="records", indent=2), encoding="utf-8"
    )
    model_b_vs_sota = final[
        final["Model"].isin(["Model B", "ShuffleNetV2 1.0x", "MobileNetV2"])
    ].reset_index(drop=True)
    model_b_vs_sota.to_csv(
        tables / "model_b_vs_sota_comparison.csv", index=False, float_format="%.4f"
    )
    df_to_tex(
        model_b_vs_sota, tables / "model_b_vs_sota_comparison.tex", "%.4f"
    )
    (tables / "model_b_vs_sota_comparison.json").write_text(
        model_b_vs_sota.to_json(orient="records", indent=2), encoding="utf-8"
    )

    main_models = ["Model B", "ShuffleNetV2 1.0x", "MobileNetV2"]
    chart_labels = ["Model B", "ShuffleNetV2\n1.0x", "MobileNetV2"]
    chart_specs = [
        ("test_accuracy_comparison.png", "Final Test Accuracy Comparison", "Test accuracy (%)", [metrics[label]["test_accuracy"] for label in main_models], False),
        ("parameter_count_comparison.png", "Parameter Count Comparison", "Trainable parameters (log scale)", [params[label] for label in main_models], True),
        ("model_size_comparison.png", "Checkpoint Size Comparison", "Checkpoint size (MiB, log scale)", [size_mib[label] for label in main_models], True),
        ("mac_comparison.png", "Calculated Computational Cost", "MACs per inference (millions)", [float(profile_by_model.loc[label, "million_macs"]) for label in main_models], False),
    ]
    for filename, title, ylabel, values, log_scale in chart_specs:
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(chart_labels, values, width=0.62)
        if log_scale:
            plt.yscale("log")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", alpha=0.2)
        if not log_scale:
            plt.ylim(0, max(values) * 1.16)
        for index, value in enumerate(values):
            label = f"{value:,.2f}" if value >= 1 else f"{value:.3f}"
            plt.text(index, value * (1.16 if log_scale else 1.025), label, ha="center", fontsize=9)
        savefig(figures / "task6" / filename)

    all_models = list(MODEL_KEYS)
    for filename, xlabel, x_values in [
        ("accuracy_vs_model_size.png", "Checkpoint size (MiB, log scale)", [size_mib[label] for label in all_models]),
        ("accuracy_vs_macs.png", "MACs per inference (millions)", [float(profile_by_model.loc[label, "million_macs"]) for label in all_models]),
    ]:
        y_values = [metrics[label]["test_accuracy"] for label in all_models]
        plt.figure(figsize=(8, 5.3))
        plt.scatter(x_values, y_values, s=70)
        for x_value, y_value, label in zip(x_values, y_values, all_models):
            offsets = {
                "Model A": (7, -14),
                "Model B": (7, -2),
                "MobileNetV2": (7, 5),
                "ShuffleNetV2 1.0x": (7, 5),
            }
            plt.annotate(
                label, (x_value, y_value), xytext=offsets[label],
                textcoords="offset points", fontsize=9,
            )
        frontier, best = [], -1.0
        for x_value, y_value, label in sorted(zip(x_values, y_values, all_models)):
            if y_value > best:
                frontier.append((x_value, y_value, label))
                best = y_value
        plt.plot([row[0] for row in frontier], [row[1] for row in frontier], "--", linewidth=1.5, label="Pareto frontier")
        if "size" in filename:
            plt.xscale("log")
        plt.xlabel(xlabel)
        plt.ylabel("Test accuracy (%)")
        plt.ylim(min(y_values) - 3.5, max(y_values) + 4.0)
        plt.title(f"Accuracy vs {('Checkpoint Size' if 'size' in filename else 'Calculated MACs')}")
        plt.grid(alpha=0.2)
        plt.legend()
        savefig(figures / "task6" / filename)

    by_name = final.set_index("Model")
    model_b = by_name.loc["Model B"]
    shuffle = by_name.loc["ShuffleNetV2 1.0x"]
    mobile = by_name.loc["MobileNetV2"]
    derived = {
        "model_b_param_reduction_vs_a_pct": 100 * (1 - params["Model B"] / params["Model A"]),
        "model_b_param_factor_vs_a": params["Model A"] / params["Model B"],
        "mobilenet_accuracy_gain_pp": mobile["Test Accuracy (%)"] - model_b["Test Accuracy (%)"],
        "shufflenet_accuracy_gain_pp": shuffle["Test Accuracy (%)"] - model_b["Test Accuracy (%)"],
        "mobilenet_param_factor_vs_b": mobile["Total Parameters"] / model_b["Total Parameters"],
        "shufflenet_param_factor_vs_b": shuffle["Total Parameters"] / model_b["Total Parameters"],
        "mobilenet_mac_factor_vs_b": mobile["MACs (M)"] / model_b["MACs (M)"],
        "shufflenet_mac_factor_vs_b": shuffle["MACs (M)"] / model_b["MACs (M)"],
        "mobilenet_size_factor_vs_b": mobile["Checkpoint Size (MiB)"] / model_b["Checkpoint Size (MiB)"],
        "shufflenet_size_factor_vs_b": shuffle["Checkpoint Size (MiB)"] / model_b["Checkpoint Size (MiB)"],
    }
    (tables / "derived_tradeoffs.json").write_text(json.dumps(derived, indent=2), encoding="utf-8")
    print(f"Generated figures and tables from {results} in {output}")


if __name__ == "__main__":
    main()
