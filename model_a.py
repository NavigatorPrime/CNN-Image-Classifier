"""
Model A: Standard CNN (Conv2D + MaxPooling)
=============================================
Custom CNN architecture that interleaves standard 2D convolutional
layers and max-pooling layers for DeFungi image classification.

Dataset: DeFungi (5 classes, 64x64 RGB images)
Architecture: Conv2D-MaxPool blocks → Fully Connected layers
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

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"d:\Sem 5\Pattern Recognition\CNN Assignment")
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results" / "model_a"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
IMG_SIZE = 64
NUM_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# ============================================================
# Data Loading & Transforms
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

    # Load datasets
    train_dataset = datasets.ImageFolder(root=str(dataset_dir / "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=str(dataset_dir / "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=str(dataset_dir / "test"),  transform=eval_transform)

    # Data loaders (num_workers=0 avoids Windows multiprocessing spawn issues)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names, len(train_dataset), len(val_dataset), len(test_dataset)



# ============================================================
# Model A: Standard CNN Architecture
# ============================================================
class ModelA_StandardCNN(nn.Module):
    """
    Standard CNN with interleaved Conv2D and MaxPooling layers.
    
    Architecture:
        Input: 3 x 64 x 64
        
        Block 1: Conv2D(3->32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2x2)
                 Output: 32 x 32 x 32
        
        Block 2: Conv2D(32->64, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2x2)
                 Output: 64 x 16 x 16
        
        Block 3: Conv2D(64->128, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2x2)
                 Output: 128 x 8 x 8
        
        Block 4: Conv2D(128->256, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2x2)
                 Output: 256 x 4 x 4
        
        Classifier:
            Flatten -> 256*4*4 = 4096
            FC(4096 -> 512) -> ReLU -> Dropout(0.5)
            FC(512 -> 128)  -> ReLU -> Dropout(0.3)
            FC(128 -> 5)    (5 classes)
    """
    
    def __init__(self, num_classes=5):
        super(ModelA_StandardCNN, self).__init__()
        
        # Feature Extraction: Conv2D + MaxPool blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 64x64 -> 32x32
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 32x32 -> 16x16
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 16x16 -> 8x8
            
            # Block 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 8x8 -> 4x4
        )
        
        # Classifier: Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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
    """Get all predictions and true labels for evaluation."""
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
def plot_training_curves(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', markersize=3, label='Val Loss')
    ax1.set_title('Model A - Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-o', markersize=3, label='Val Accuracy')
    ax2.set_title('Model A - Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Model A - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


# ============================================================
# Model Summary
# ============================================================
def print_model_summary(model):
    """Print model architecture and parameter count."""
    print("\n" + "=" * 60)
    print("MODEL A: Standard CNN Architecture")
    print("=" * 60)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    return total_params, trainable_params


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model A: Standard CNN on DeFungi")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows)")
    args = parser.parse_args()

    # Load data
    train_loader, val_loader, test_loader, class_names, n_train, n_val, n_test = get_data_loaders(
        DATASET_DIR, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Training samples:   {n_train}")
    print(f"[INFO] Validation samples: {n_val}")
    print(f"[INFO] Test samples:       {n_test}")

    # Initialize model
    model = ModelA_StandardCNN(num_classes=NUM_CLASSES).to(DEVICE)
    total_params, trainable_params = print_model_summary(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'LR':>10}")
    print("-" * 72)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.2%} | {val_loss:>10.4f} | {val_acc:>9.2%} | {current_lr:>10.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), str(RESULTS_DIR / "model_a_best.pth"))
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.2%} at epoch {best_epoch}")
    
    # --------------------------------------------------------
    # Evaluation on Test Set
    # --------------------------------------------------------
    print("\n[INFO] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(str(RESULTS_DIR / "model_a_best.pth"), weights_only=True))
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[RESULT] Test Loss: {test_loss:.4f}")
    print(f"[RESULT] Test Accuracy: {test_acc:.2%}")
    
    # Get predictions for detailed metrics
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\n[RESULT] Classification Report:\n{report}")
    
    # Save classification report
    with open(str(RESULTS_DIR / "classification_report.txt"), 'w') as f:
        f.write("Model A: Standard CNN - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write(f"\nTest Accuracy: {test_acc:.4%}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Time: {total_time:.1f}s\n")
    
    # --------------------------------------------------------
    # Save Plots
    # --------------------------------------------------------
    plot_training_curves(history, str(RESULTS_DIR / "training_curves.png"))
    plot_confusion_matrix(y_true, y_pred, class_names, str(RESULTS_DIR / "confusion_matrix.png"))
    
    # Save training history as JSON
    with open(str(RESULTS_DIR / "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL A: STANDARD CNN - FINAL SUMMARY")
    print("=" * 60)
    print(f"Architecture:     4 Conv2D-MaxPool blocks + 3 FC layers")
    print(f"Parameters:       {total_params:,}")
    print(f"Best Val Acc:     {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Test Accuracy:    {test_acc:.2%}")
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Training Time:    {total_time:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
