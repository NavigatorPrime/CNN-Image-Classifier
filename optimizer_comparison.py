"""Task 3: controlled Model B optimizer comparison on the packaged split.

The three runs use the same initial weights, split, data transforms, batch
size, loss, weight decay, scheduler, checkpoint rule and random seed. Only the
optimizer algorithm and documented learning rate/momentum differ.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from data_preparation import verify_prepared_dataset
from model_b import (
    DATASET_DIR,
    ModelB_LightweightCNN,
    evaluate,
    get_data_loaders,
    get_predictions,
    train_one_epoch,
)
from reproducibility import (
    DEFAULT_SEED,
    dataset_provenance,
    prepare_run_directory,
    seed_everything,
)


REPO_DIR = Path(__file__).resolve().parent
NUM_CLASSES = 5
DEFAULT_EPOCHS = 20
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4

OPTIMIZER_CONFIGS = {
    "sgd": {
        "display_name": "Standard SGD",
        "learning_rate": 0.01,
        "momentum": 0.0,
    },
    "sgd_momentum": {
        "display_name": "SGD + Momentum",
        "learning_rate": 0.01,
        "momentum": 0.9,
    },
    "adam": {
        "display_name": "Adam (selected)",
        "learning_rate": 0.001,
        "momentum": None,
    },
}


def make_optimizer(name: str, parameters):
    config = OPTIMIZER_CONFIGS[name]
    if name == "adam":
        return optim.Adam(
            parameters,
            lr=config["learning_rate"],
            weight_decay=WEIGHT_DECAY,
        )
    return optim.SGD(
        parameters,
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=WEIGHT_DECAY,
    )


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_optimizer(
    optimizer_name: str,
    initial_state: dict[str, torch.Tensor],
    epochs: int,
    device: torch.device,
    num_workers: int,
    results_dir: Path,
) -> dict:
    seed_everything(DEFAULT_SEED)
    train_loader, val_loader, test_loader, class_names, n_train, n_val, n_test = (
        get_data_loaders(
            DATASET_DIR,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
        )
    )

    model = ModelB_LightweightCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 37_957
    assert parameter_count <= 100_000

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(optimizer_name, model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }
    best_validation_accuracy = -1.0
    best_epoch = 0
    checkpoint_path = results_dir / f"{optimizer_name}_best.pth"
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(validation_loss)
        epoch_seconds = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["epoch_seconds"].append(epoch_seconds)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"[{optimizer_name:12}] {epoch:02d}/{epochs:02d} | "
            f"train loss {train_loss:.4f} | val loss {validation_loss:.4f} | "
            f"val acc {validation_accuracy:.2%} | {epoch_seconds:.1f}s",
            flush=True,
        )

        progress = {
            "optimizer": optimizer_name,
            "completed_epochs": epoch,
            "history": history,
        }
        (results_dir / f"{optimizer_name}_progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )

    elapsed = time.perf_counter() - start
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
    )
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    y_true, y_pred = get_predictions(model, test_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    macro = report["macro avg"]
    config = OPTIMIZER_CONFIGS[optimizer_name]
    return {
        "optimizer_id": optimizer_name,
        "optimizer": config["display_name"],
        "learning_rate": config["learning_rate"],
        "momentum": config["momentum"],
        "weight_decay": WEIGHT_DECAY,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "seed": DEFAULT_SEED,
        "samples": {"train": n_train, "validation": n_val, "test": n_test},
        "parameter_count": parameter_count,
        "initial_state_sha256": state_dict_sha256(initial_state),
        "checkpoint_selection": "highest validation accuracy",
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 4,
        },
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1-score"],
        "total_training_seconds": elapsed,
        "average_epoch_seconds": float(np.mean(history["epoch_seconds"])),
        "history": history,
        "classification_report": report,
        "checkpoint": checkpoint_path.as_posix(),
    }


def save_summary(results: list[dict], device: torch.device, results_dir: Path) -> None:
    payload = {
        "experiment": "Task 3 Model B optimizer comparison",
        "method": (
            "Same initial state_dict, packaged split, transforms, loss, batch size, "
            "weight decay, scheduler, seed and validation-accuracy checkpoint rule."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset": dataset_provenance(REPO_DIR),
        "results": results,
    }
    (results_dir / "optimizer_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fields = [
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
        "best_epoch",
        "best_validation_accuracy",
        "final_train_loss",
        "final_train_accuracy",
        "final_validation_loss",
        "final_validation_accuracy",
        "test_loss",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "average_epoch_seconds",
        "total_training_seconds",
    ]
    with (results_dir / "optimizer_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def plot_curves(results: list[dict], results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("train_loss", "Training loss", False),
        ("validation_loss", "Validation loss", False),
        ("validation_accuracy", "Validation accuracy (%)", True),
    ]
    for result in results:
        epochs = range(1, result["epochs"] + 1)
        for axis, (key, ylabel, percentage) in zip(axes, panels):
            values = np.asarray(result["history"][key])
            if percentage:
                values = values * 100
            axis.plot(epochs, values, marker="o", markersize=3, label=result["optimizer"])
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Task 3: Model B optimizer comparison")
    fig.tight_layout()
    fig.savefig(results_dir / "optimizer_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", help="Run output directory; defaults to a unique runs/optimizer_comparison directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    results_dir = prepare_run_directory(
        REPO_DIR, "optimizer_comparison", args.output_dir, args.overwrite
    )
    verify_prepared_dataset()
    torch.set_num_threads(args.threads)
    seed_everything(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = ModelB_LightweightCNN(num_classes=NUM_CLASSES)
    initial_state = copy.deepcopy(base_model.state_dict())
    assert sum(p.numel() for p in base_model.parameters()) == 37_957

    results = [
        run_optimizer(
            name,
            initial_state,
            args.epochs,
            device,
            args.num_workers,
            results_dir,
        )
        for name in OPTIMIZER_CONFIGS
    ]
    save_summary(results, device, results_dir)
    plot_curves(results, results_dir)
    print(f"Saved Task 3 artifacts to {results_dir}")


if __name__ == "__main__":
    main()
