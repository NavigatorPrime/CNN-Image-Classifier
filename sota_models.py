"""
Task 5: Lightweight State-of-the-Art (SOTA) Fine-Tuning & Evaluation
=====================================================================
Fine-tunes and evaluates two industry-standard lightweight pre-trained
architectures designed for mobile/edge computing on the DeFungi dataset:
  1. MobileNetV2 (Inverted Residuals + Linear Bottlenecks)
  2. ShuffleNetV2 1.0x (Channel Split + Channel Shuffle)

Dataset: DeFungi (5 classes, 64x64 RGB images)
Splits: Train (70%), Val (15%), Test (15%) - Identical to Models A & B
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ensure UTF-8 stdout on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from reproducibility import DEFAULT_SEED, make_generator, seed_everything, seed_worker

# ============================================================
# Configuration & Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

IMG_SIZE = 64
NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set PyTorch thread count for high-performance CPU execution
torch.set_num_threads(min(8, os.cpu_count() or 8))


# ============================================================
# Data Loading with Pretrained Normalization
# ============================================================
def get_sota_data_loaders(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=0):
    """
    Data loaders applying standard ImageNet normalization required by
    pre-trained SOTA backbones.
    """
    pin_mem = torch.cuda.is_available()

    # ImageNet stats expected by pre-trained torchvision weights
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    train_dataset = datasets.ImageFolder(root=str(dataset_dir / "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=str(dataset_dir / "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=str(dataset_dir / "test"),  transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        worker_init_fn=seed_worker, generator=make_generator(DEFAULT_SEED),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker,
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names, len(train_dataset), len(val_dataset), len(test_dataset)


# ============================================================
# Model Builders
# ============================================================
def build_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True):
    """
    MobileNetV2: Lightweight CNN using Inverted Residual blocks with
    Linear Bottlenecks and Depthwise Separable convolutions.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace 1000-class classifier head with 5-class head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_shufflenet_v2(num_classes=NUM_CLASSES, pretrained=True):
    """
    ShuffleNetV2 1.0x: Hardware-friendly lightweight CNN utilizing
    Channel Split and Channel Shuffle operations to minimize MAC.
    """
    weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    model = models.shufflenet_v2_x1_0(weights=weights)

    # Replace 1000-class classification head with 5-class head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# Training & Validation Loops
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for b_idx, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if b_idx % 30 == 0 or b_idx == total_batches:
            batch_loss = running_loss / total
            batch_acc = correct / total
            print(f"  [Batch {b_idx:03d}/{total_batches:03d}] Loss: {batch_loss:.4f} | Acc: {batch_acc*100:.2f}%", flush=True)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def full_test_evaluation(model, loader, device, class_names):
    """Gather predictions for confusion matrix, precision, recall, f1."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)

    return cm, report_dict, report_text, all_labels, all_preds


# ============================================================
# Plotting Helpers
# ============================================================
def plot_curves(history, output_dir, model_name):
    output_dir = Path(output_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss curves only (required specifically in prompt)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, history["val_loss"], 'r-s', label='Validation Loss', linewidth=2)
    plt.title(f'{model_name} - Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (CrossEntropy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()

    # 2. Dual Training Curves (Loss and Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r-s', label='Val Loss')
    ax1.set_title(f'{model_name} Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Acc')
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], 'r-s', label='Val Acc')
    ax2.set_title(f'{model_name} Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir, model_name):
    output_dir = Path(output_dir)
    plt.figure(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title(f'{model_name} Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


# ============================================================
# Training Runner
# ============================================================
def train_and_evaluate_model(model_name, model_id, model, train_loader, val_loader, test_loader, class_names, num_epochs=NUM_EPOCHS):
    out_dir = RESULTS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  TRAINING & EVALUATION: {model_name} ({model_id})")
    print("=" * 70)

    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_times": []}
    best_val_loss = float('inf')
    best_model_path = out_dir / f"{model_id}_best.pth"

    total_start = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        t_epoch = time.time() - t0

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(t_epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            mark = " [BEST]"
        else:
            mark = ""

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({t_epoch:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%{mark}")

    total_time = time.time() - total_start
    avg_epoch_time = float(np.mean(history["epoch_times"]))

    # Load best checkpoint for final test evaluation
    print(f"\n[INFO] Loading best model checkpoint from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    cm, report_dict, report_text, _, _ = full_test_evaluation(model, test_loader, DEVICE, class_names)

    # Binary file size in MiB. This is storage, not peak runtime RAM.
    model_size_mib = os.path.getsize(best_model_path) / (1024 * 1024)

    print("\n" + "-" * 70)
    print(f"  {model_name} TEST SET PERFORMANCE")
    print("-" * 70)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Checkpoint:    {model_size_mib:.2f} MiB")
    print(f"Total Params:  {total_params:,}")
    print(f"Avg Epoch Time:{avg_epoch_time:.2f}s (Total: {total_time:.1f}s)")
    print("\nClassification Report:\n" + report_text)

    # Save artifacts
    plot_curves(history, out_dir, model_name)
    plot_confusion_matrix(cm, class_names, out_dir, model_name)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Checkpoint Size on Disk: {model_size_mib:.2f} MiB\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report_text)

    results_summary = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_size_mib": round(model_size_mib, 3),
        "test_accuracy": round(test_acc * 100, 2),
        "test_loss": round(test_loss, 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        "avg_epoch_time_seconds": round(avg_epoch_time, 2),
        "total_training_time_seconds": round(total_time, 2),
        "best_val_loss": round(best_val_loss, 4),
        "history": history
    }

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


# ============================================================
# Main Execution Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Task 5: Fine-tune MobileNetV2 and ShuffleNetV2")
    parser.add_argument("--model", choices=["all", "mobilenet_v2", "shufflenet_v2"], default="all",
                        help="Which SOTA model to train and evaluate")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    seed_everything(DEFAULT_SEED)

    print("=" * 70)
    print("  Task 5: Lightweight SOTA Benchmark on DeFungi (64x64)")
    print("=" * 70)
    print(f"[INFO] Using PyTorch Device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names, n_train, n_val, n_test = get_sota_data_loaders(
        batch_size=args.batch_size
    )

    print(f"[INFO] Data loaded successfully:")
    print(f"       Train images: {n_train} | Val images: {n_val} | Test images: {n_test}")
    print(f"       Classes ({len(class_names)}): {class_names}")

    all_summaries = {}

    # 1. MobileNetV2
    if args.model in ["all", "mobilenet_v2"]:
        mb_model = build_mobilenet_v2(num_classes=len(class_names), pretrained=True)
        all_summaries["mobilenet_v2"] = train_and_evaluate_model(
            "MobileNetV2", "mobilenet_v2", mb_model, train_loader, val_loader, test_loader, class_names, num_epochs=args.epochs
        )

    # 2. ShuffleNetV2
    if args.model in ["all", "shufflenet_v2"]:
        sf_model = build_shufflenet_v2(num_classes=len(class_names), pretrained=True)
        all_summaries["shufflenet_v2"] = train_and_evaluate_model(
            "ShuffleNetV2", "shufflenet_v2", sf_model, train_loader, val_loader, test_loader, class_names, num_epochs=args.epochs
        )

    # Save overall SOTA summary
    summary_path = RESULTS_DIR / "sota_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Generate Markdown Summary Table
    table_path = RESULTS_DIR / "sota_summary_table.md"
    with open(table_path, "w") as f:
        f.write("# Task 5: Lightweight SOTA Benchmark Results\n\n")
        f.write("| Architecture | Parameters | Checkpoint Size (MiB) | Test Accuracy (%) | Test Loss | Macro F1 | Avg Time/Epoch (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for k, v in all_summaries.items():
            f.write(f"| **{v['model_name']}** | {v['total_parameters']:,} | {v['checkpoint_size_mib']} MiB | **{v['test_accuracy']}%** | {v['test_loss']} | {v['macro_f1']} | {v['avg_epoch_time_seconds']}s |\n")

    print("\n" + "=" * 70)
    print("  TASK 5 SOTA BENCHMARK COMPLETE!")
    print(f"  Results saved to: {summary_path}")
    print(f"  Summary table:    {table_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
