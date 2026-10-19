"""Validate the canonical leakage-free Stage 2 experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = ["H1", "H2", "H3", "H5", "H6"]
EXPECTED_SPLITS = {"train": 6380, "validation": 1368, "test": 1366}
MODEL_DIRS = {
    "Model A": "model_a",
    "Model B": "model_b",
    "MobileNetV2": "mobilenet_v2",
    "ShuffleNetV2 1.0x": "shufflenet_v2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def assert_report_matches_confusion(summary: dict, label: str) -> None:
    matrix = np.asarray(summary["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (5, 5) or int(matrix.sum()) != EXPECTED_SPLITS["test"]:
        raise AssertionError(f"{label}: invalid confusion matrix shape or total")

    report = summary["classification_report"]
    supports = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precisions = np.divide(
        diagonal, predicted, out=np.zeros(5, dtype=float), where=predicted != 0
    )
    recalls = diagonal / supports
    f1s = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(5, dtype=float),
        where=(precisions + recalls) != 0,
    )
    for index, class_name in enumerate(EXPECTED_CLASSES):
        row = report[class_name]
        if not close(row["precision"], precisions[index]):
            raise AssertionError(f"{label}: {class_name} precision mismatch")
        if not close(row["recall"], recalls[index]):
            raise AssertionError(f"{label}: {class_name} recall mismatch")
        if not close(row["f1-score"], f1s[index]):
            raise AssertionError(f"{label}: {class_name} F1 mismatch")
        if int(row["support"]) != int(supports[index]):
            raise AssertionError(f"{label}: {class_name} support mismatch")

    accuracy = float(diagonal.sum() / matrix.sum())
    macro_precision = float(precisions.mean())
    macro_recall = float(recalls.mean())
    macro_f1 = float(f1s.mean())
    if not close(report["accuracy"], accuracy):
        raise AssertionError(f"{label}: report accuracy mismatch")
    for key, expected in [
        ("precision", macro_precision),
        ("recall", macro_recall),
        ("f1-score", macro_f1),
    ]:
        if not close(report["macro avg"][key], expected):
            raise AssertionError(f"{label}: macro {key} mismatch")

    # Custom models store full precision; SOTA summaries intentionally store
    # four-decimal metrics and two-decimal percentage accuracy.
    summary_tolerance = 5e-5 if label.startswith(("Mobile", "Shuffle")) else 1e-9
    accuracy_tolerance = 0.005 if label.startswith(("Mobile", "Shuffle")) else 1e-9
    if not close(summary["test_accuracy"], accuracy * 100, accuracy_tolerance):
        raise AssertionError(f"{label}: stored test accuracy mismatch")
    for key, expected in [
        ("macro_precision", macro_precision),
        ("macro_recall", macro_recall),
        ("macro_f1", macro_f1),
    ]:
        if not close(summary[key], expected, summary_tolerance):
            raise AssertionError(f"{label}: stored {key} mismatch")


def assert_text_report_matches(summary: dict, path: Path, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    macro_tolerance = 0.00005 if label.startswith(("Mobile", "Shuffle")) else 0.0000005

    def value(pattern: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise AssertionError(f"{label}: missing text-report field {pattern}")
        return float(match.group(1).replace(",", ""))

    checks = [
        (value(r"Test Accuracy:\s*([0-9.,]+)%"), summary["test_accuracy"], 0.005),
        (value(r"Test Loss:\s*([0-9.,]+)"), summary["test_loss"], 0.00005),
        (value(r"Total Parameters:\s*([0-9,]+)"), summary["total_parameters"], 0.0),
        (value(r"Best Epoch:\s*([0-9]+)"), summary["best_epoch"], 0.0),
        (value(r"Macro Precision:\s*([0-9.]+)"), summary["macro_precision"], macro_tolerance),
        (value(r"Macro Recall:\s*([0-9.]+)"), summary["macro_recall"], macro_tolerance),
        (value(r"Macro F1:\s*([0-9.]+)"), summary["macro_f1"], macro_tolerance),
        (
            value(r"Training Time:\s*([0-9.]+)s"),
            summary["total_training_time_seconds"],
            0.005,
        ),
        (
            value(r"Training Time:.*?\(([0-9.]+)s/epoch\)"),
            summary["avg_epoch_time_seconds"],
            0.005,
        ),
    ]
    for actual, expected, tolerance in checks:
        if not close(actual, expected, tolerance):
            raise AssertionError(f"{label}: JSON/text report mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results_clean_split")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    results_root = args.results_root
    if not results_root.is_absolute():
        results_root = (ROOT / results_root).resolve()

    # Import after establishing ROOT so the repository module is always used.
    import sys

    sys.path.insert(0, str(ROOT))
    from data_preparation import verify_packaged_dataset

    dataset_audit = verify_packaged_dataset()
    if dataset_audit["split_counts"] != {"train": 6380, "val": 1368, "test": 1366}:
        raise AssertionError("Unexpected packaged split counts")
    if dataset_audit["raw_cross_split_groups"] or dataset_audit["pixel_cross_split_groups"]:
        raise AssertionError("Cross-split duplicate content detected")

    manifest_sha = sha256(ROOT / "split_manifest.csv")
    package_sha = sha256(ROOT / "dataset_64x64.zip")
    summaries: dict[str, dict] = {}
    for label, directory in MODEL_DIRS.items():
        summary = load_json(results_root / directory / "training_history.json")
        summaries[label] = summary
        if summary["epochs"] != 20:
            raise AssertionError(f"{label}: epochs is not 20")
        if summary["seed"] != 42 or summary["batch_size"] != 32:
            raise AssertionError(f"{label}: inconsistent seed or batch size")
        if summary["samples"] != EXPECTED_SPLITS:
            raise AssertionError(f"{label}: inconsistent sample counts")
        provenance = summary["dataset"]
        if provenance["manifest_sha256"] != manifest_sha:
            raise AssertionError(f"{label}: manifest checksum mismatch")
        if provenance["package_sha256"] != package_sha:
            raise AssertionError(f"{label}: package checksum mismatch")
        if provenance["cross_split_duplicate_content"] != 0:
            raise AssertionError(f"{label}: nonzero duplicate provenance")
        history = summary["history"]
        for key in ["train_loss", "val_loss", "train_acc", "val_acc", "epoch_times"]:
            if len(history[key]) != 20:
                raise AssertionError(f"{label}: {key} does not contain 20 epochs")

        if summary["checkpoint_selection"] == "highest validation accuracy":
            expected_epoch = int(np.argmax(history["val_acc"])) + 1
        elif summary["checkpoint_selection"] == "lowest validation loss":
            expected_epoch = int(np.argmin(history["val_loss"])) + 1
        else:
            raise AssertionError(f"{label}: unknown checkpoint rule")
        if summary["best_epoch"] != expected_epoch:
            raise AssertionError(f"{label}: selected checkpoint epoch mismatch")

        checkpoint = results_root / directory / f"{directory}_best.pth"
        if not checkpoint.is_file():
            raise AssertionError(f"{label}: checkpoint is missing")
        actual_mib = checkpoint.stat().st_size / (1024**2)
        size_tolerance = 0.0005 if label.startswith(("Mobile", "Shuffle")) else 1e-9
        if not close(summary["checkpoint_size_mib"], actual_mib, size_tolerance):
            raise AssertionError(f"{label}: checkpoint size mismatch")

        csv_matrix = np.loadtxt(
            results_root / directory / "confusion_matrix.csv", delimiter=",", dtype=int
        )
        if not np.array_equal(csv_matrix, np.asarray(summary["confusion_matrix"])):
            raise AssertionError(f"{label}: confusion CSV/JSON mismatch")
        assert_report_matches_confusion(summary, label)
        assert_text_report_matches(
            summary, results_root / directory / "classification_report.txt", label
        )

    if summaries["Model B"]["trainable_parameters"] > 100_000:
        raise AssertionError("Model B exceeds 100,000 trainable parameters")

    optimizer_payload = load_json(
        results_root / "optimizer_comparison" / "optimizer_comparison.json"
    )
    optimizer_results = optimizer_payload["results"]
    if len(optimizer_results) != 3:
        raise AssertionError("Optimizer comparison does not contain three runs")
    initial_hashes = {item["initial_state_sha256"] for item in optimizer_results}
    if len(initial_hashes) != 1:
        raise AssertionError("Optimizer runs did not use identical initial weights")
    for item in optimizer_results:
        name = item["optimizer"]
        if item["epochs"] != 20 or item["seed"] != 42 or item["batch_size"] != 32:
            raise AssertionError(f"{name}: inconsistent controlled setting")
        if item["samples"] != EXPECTED_SPLITS:
            raise AssertionError(f"{name}: inconsistent sample counts")
        validation = item["history"]["validation_accuracy"]
        if any(len(item["history"][key]) != 20 for key in item["history"]):
            raise AssertionError(f"{name}: incomplete history")
        if item["best_epoch"] != int(np.argmax(validation)) + 1:
            raise AssertionError(f"{name}: selected checkpoint epoch mismatch")
        if not close(item["best_validation_accuracy"], max(validation)):
            raise AssertionError(f"{name}: best validation accuracy mismatch")
        if not close(item["test_accuracy"], item["classification_report"]["accuracy"]):
            raise AssertionError(f"{name}: test accuracy/report mismatch")
        for result_key, report_key in [
            ("macro_precision", "precision"),
            ("macro_recall", "recall"),
            ("macro_f1", "f1-score"),
        ]:
            if not close(item[result_key], item["classification_report"]["macro avg"][report_key]):
                raise AssertionError(f"{name}: {result_key}/report mismatch")
        checkpoint = results_root / "optimizer_comparison" / f"{item['optimizer_id']}_best.pth"
        if not checkpoint.is_file():
            raise AssertionError(f"{name}: checkpoint is missing")
        progress = load_json(
            results_root / "optimizer_comparison" / f"{item['optimizer_id']}_progress.json"
        )
        if progress["completed_epochs"] != 20 or progress["history"] != item["history"]:
            raise AssertionError(f"{name}: progress/summary JSON mismatch")

    with (results_root / "optimizer_comparison" / "optimizer_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 3:
        raise AssertionError("Optimizer CSV does not contain three rows")
    for csv_row, item in zip(csv_rows, optimizer_results):
        for key in [
            "learning_rate", "momentum", "best_epoch", "best_validation_accuracy",
            "final_train_loss", "final_validation_loss", "test_loss", "test_accuracy",
            "macro_precision", "macro_recall", "macro_f1",
        ]:
            csv_value = csv_row[key]
            json_value = item[key]
            if json_value is None:
                if csv_value != "":
                    raise AssertionError(f"{item['optimizer']}: CSV {key} mismatch")
            elif not close(float(csv_value), float(json_value)):
                raise AssertionError(f"{item['optimizer']}: CSV {key} mismatch")

    profiles = {}
    with (results_root / "computational_cost" / "model_profiles.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            profiles[row["model"]] = row
    for label, summary in summaries.items():
        profile = profiles[label]
        if int(profile["total_parameters"]) != summary["total_parameters"]:
            raise AssertionError(f"{label}: profile parameter count mismatch")
        if int(profile["flops_2_per_mac"]) != 2 * int(profile["macs"]):
            raise AssertionError(f"{label}: FLOPs is not 2 x MACs")
    profile_json = load_json(results_root / "computational_cost" / "model_profiles.json")
    for label, profile in profiles.items():
        json_profile = profile_json["models"][label]
        for csv_key, json_key in [
            ("total_parameters", "total_parameters"),
            ("trainable_parameters", "trainable_parameters"),
            ("macs", "macs"),
            ("flops_2_per_mac", "flops_2_per_mac"),
        ]:
            if int(profile[csv_key]) != int(json_profile[json_key]):
                raise AssertionError(f"{label}: profile CSV/JSON mismatch")

    sota_combined = load_json(results_root / "sota_benchmark_summary.json")
    for label, key in [("MobileNetV2", "mobilenet_v2"), ("ShuffleNetV2 1.0x", "shufflenet_v2")]:
        if sota_combined[key] != summaries[label]:
            raise AssertionError(f"{label}: combined/per-model SOTA summary mismatch")

    report = {
        "status": "PASS",
        "dataset": dataset_audit,
        "manifest_sha256": manifest_sha,
        "package_sha256": package_sha,
        "models_validated": list(MODEL_DIRS),
        "optimizer_runs_validated": [item["optimizer"] for item in optimizer_results],
        "shared_optimizer_initial_state_sha256": next(iter(initial_hashes)),
        "cross_split_duplicates": 0,
    }
    if args.write_report:
        with (results_root / "validation_report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    print("CROSS-SPLIT DUPLICATES = 0")
    print("STAGE 2 ARTIFACT VALIDATION = PASS")


if __name__ == "__main__":
    main()
