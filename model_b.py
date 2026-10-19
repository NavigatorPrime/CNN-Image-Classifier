"""
Model B: Lightweight Edge CNN (Depthwise Separable Convolutions)
=================================================================
Custom lightweight CNN architecture replacing standard convolutions with
Depthwise Separable Convolutions to drastically reduce multiply-accumulate
(MAC) operations and memory footprint on resource-constrained devices.

Assignment Constraint: Total trainable parameter count MUST NOT exceed 100,000.
Achieved: 37,957 parameters (< 38% of limit).

Dataset: DeFungi (5 classes, 64x64 RGB images)
Architecture: Stem Conv -> 6x Depthwise Separable Blocks -> Global Avg Pool -> Linear Head
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from data_preparation import verify_prepared_dataset
from reproducibility import (
    DEFAULT_SEED,
    dataset_provenance,
    make_generator,
    prepare_run_directory,
    seed_everything,
    seed_worker,
)

# ============================================================
# Configuration & Directory Resolution
# ============================================================
REPO_DIR = Path(__file__).resolve().parent

BASE_DIR = REPO_DIR
DATASET_DIR = BASE_DIR / "dataset"

# Hyperparameters
IMG_SIZE = 64
NUM_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Dataset directory: {DATASET_DIR}")

# ============================================================
# Data Loading & Transforms (Identical to Model A)
# ============================================================
def get_data_loaders(dataset_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=0):
    """Create train, validation, and test DataLoaders."""
    pin_mem = torch.cuda.is_available()
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
# Model B Building Blocks: Depthwise Separable Convolution
# ============================================================
class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise Separable Convolution Block:
    1. Depthwise Conv: 3x3 spatial filter per input channel (groups = in_channels).
       BN + ReLU6 activation.
    2. Pointwise Conv: 1x1 cross-channel linear combination.
       BN + ReLU6 activation.

    Hardware-Aware Activation Justification:
    - Avoids computationally expensive transcendental exponentials (unlike Sigmoid, Tanh, GELU, Swish).
    - ReLU6 bounds positive activations at 6.0. This bounded range can aid
      quantization, but quantized accuracy and target-runtime efficiency must
      be measured for the chosen deployment.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableBlock, self).__init__()
        
        # 1. Depthwise Convolution (groups = in_channels)
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU6(inplace=True)
        
        # 2. Pointwise Convolution (1x1 conv)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial filtering
        x = self.act_dw(self.bn_dw(self.depthwise(x)))
        # Channel combination
        x = self.act_pw(self.bn_pw(self.pointwise(x)))
        return x


# ============================================================
# Model B: Lightweight Edge CNN Architecture
# ============================================================
class ModelB_LightweightCNN(nn.Module):
    """
    Model B: Ultra-Lightweight CNN designed for edge image classification.
    
    Constraint: Trainable parameter count < 100,000.
    Achieved:   37,957 parameters (62% margin below limit).
    Calculated cost: 7.262848 M Conv2d/Linear MACs for a 3x64x64 input.
    BatchNorm, activations and pooling are excluded; this is not latency.
    
    Topology:
        Input: 3 x 64 x 64
        Stem:   Standard Conv2D(3->16, 3x3, stride=1) -> BN -> ReLU6 (64x64x16)
        Block 1: SepConv(16->32, stride=2) (32x32x32)
        Block 2: SepConv(32->32, stride=1) (32x32x32)
        Block 3: SepConv(32->64, stride=2) (16x16x64)
        Block 4: SepConv(64->64, stride=1) (16x16x64)
        Block 5: SepConv(64->128, stride=2) (8x8x128)
        Block 6: SepConv(128->128, stride=1) (8x8x128)
        Head:   Global Average Pooling (128) -> Dropout(0.2) -> Linear(128->5)
    """
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.2):
        super(ModelB_LightweightCNN, self).__init__()
        
        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Depthwise Separable Blocks
        self.block1 = DepthwiseSeparableBlock(16, 32, stride=2)   # 64x64 -> 32x32
        self.block2 = DepthwiseSeparableBlock(32, 32, stride=1)   # 32x32 -> 32x32
        self.block3 = DepthwiseSeparableBlock(32, 64, stride=2)   # 32x32 -> 16x16
        self.block4 = DepthwiseSeparableBlock(64, 64, stride=1)   # 16x16 -> 16x16
        self.block5 = DepthwiseSeparableBlock(64, 128, stride=2)  # 16x16 -> 8x8
        self.block6 = DepthwiseSeparableBlock(128, 128, stride=1) # 8x8 -> 8x8
        
        # Edge-Friendly Classifier Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# ============================================================
# Training & Evaluation Functions
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
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
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def get_predictions(model, loader, device):
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
    
    return np.array(all_labels), np.array(all_preds)


# ============================================================
# Plotting Functions
# ============================================================
def plot_loss_curves(history, save_path):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=4, label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=4, label='Validation Loss')
    plt.title('Model B (Lightweight CNN) - Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Loss curve saved to {save_path}", flush=True)


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', markersize=3, label='Val Loss')
    ax1.set_title('Model B - Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-o', markersize=3, label='Val Accuracy')
    ax2.set_title('Model B - Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Combined curves saved to {save_path}", flush=True)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Greens', values_format='d')
    ax.set_title('Model B - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


# ============================================================
# Model Summary & Parameter Verification
# ============================================================
def print_model_summary(model):
    print("\n" + "=" * 65)
    print("MODEL B: LIGHTWEIGHT CNN (DEPTHWISE SEPARABLE CONVOLUTION)")
    print("=" * 65)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    status = "PASSED (< 100,000)" if trainable_params < 100000 else "FAILED (>= 100,000)"
    print(f"Assignment Limit:     {status}")
    print("=" * 65)
    return total_params, trainable_params


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model B: Lightweight Edge CNN on DeFungi")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs (default: 20 per Task 4)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--output-dir", help="Run output directory; defaults to a unique runs/model_b directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    args = parser.parse_args()

    results_dir = prepare_run_directory(
        REPO_DIR, "model_b", args.output_dir, args.overwrite
    )
    print(f"[INFO] Results directory: {results_dir}")

    seed_everything(DEFAULT_SEED)
    verify_prepared_dataset(DATASET_DIR)

    # Load data
    train_loader, val_loader, test_loader, class_names, n_train, n_val, n_test = get_data_loaders(
        DATASET_DIR, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Training samples:   {n_train}")
    print(f"[INFO] Validation samples: {n_val}")
    print(f"[INFO] Test samples:       {n_test}")

    # Initialize model
    model = ModelB_LightweightCNN(num_classes=NUM_CLASSES).to(DEVICE)
    total_params, trainable_params = print_model_summary(model)
    assert trainable_params < 100000, f"Error: Model B parameter count ({trainable_params}) exceeds 100,000 limit!"

    # Loss function and optimizer (Adam as tuned in Task 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'epoch_times': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'LR':>10}")
    print("-" * 72)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(time.time() - epoch_start)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.2%} | {val_loss:>10.4f} | {val_acc:>9.2%} | {current_lr:>10.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), str(results_dir / "model_b_best.pth"))
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.2%} at epoch {best_epoch}")
    
    # Test evaluation
    print("\n[INFO] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(str(results_dir / "model_b_best.pth"), weights_only=True))
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[RESULT] Test Loss: {test_loss:.4f}")
    print(f"[RESULT] Test Accuracy: {test_acc:.2%}")
    
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    macro = report_dict["macro avg"]
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[RESULT] Classification Report:\n{report}")
    
    # Save classification report
    checkpoint_size_kib = os.path.getsize(str(results_dir / "model_b_best.pth")) / 1024
    checkpoint_size_mib = checkpoint_size_kib / 1024
    average_epoch_time = float(np.mean(history["epoch_times"]))
    with open(str(results_dir / "classification_report.txt"), 'w') as f:
        f.write("Model B: Lightweight Edge CNN - Classification Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
        f.write(f"\nTest Accuracy: {test_acc:.4%}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Checkpoint Size on Disk: {checkpoint_size_kib:.2f} KiB\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.6f}\n")
        f.write(f"Macro Precision: {macro['precision']:.6f}\n")
        f.write(f"Macro Recall: {macro['recall']:.6f}\n")
        f.write(f"Macro F1: {macro['f1-score']:.6f}\n")
        f.write(f"Training Time: {total_time:.2f}s ({average_epoch_time:.2f}s/epoch)\n")
    
    # Save plots
    plot_loss_curves(history, str(results_dir / "loss_curves.png"))
    plot_training_curves(history, str(results_dir / "training_curves.png"))
    plot_confusion_matrix(y_true, y_pred, class_names, str(results_dir / "confusion_matrix.png"))
    np.savetxt(results_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    
    # Save history json
    checkpoint_path = results_dir / "model_b_best.pth"
    summary = {
        "model_name": "Model B",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": DEFAULT_SEED,
        "samples": {"train": n_train, "validation": n_val, "test": n_test},
        "dataset": dataset_provenance(REPO_DIR),
        "checkpoint_selection": "highest validation accuracy",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_acc,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_size_mib": checkpoint_size_mib,
        "test_loss": test_loss,
        "test_accuracy": test_acc * 100,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1-score"],
        "total_training_time_seconds": total_time,
        "avg_epoch_time_seconds": average_epoch_time,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "history": history,
    }
    with open(str(results_dir / "training_history.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 65)
    print("MODEL B: LIGHTWEIGHT CNN - FINAL SUMMARY")
    print("=" * 65)
    print(f"Architecture:     Stem + 6x Depthwise Separable Blocks + GAP + Linear")
    print(f"Parameters:       {total_params:,} (< 100,000 constraint)")
    print(f"Checkpoint Size:  {checkpoint_size_kib:.2f} KiB")
    print(f"Best Val Acc:     {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Test Accuracy:    {test_acc:.2%}")
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Training Time:    {total_time:.1f}s")
    print(f"Results saved to: {results_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
