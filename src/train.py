"""
Module: src/train.py
Description: Production-quality, unified training module for Plant Leaf Disease Classification.
Supports Day 7 (baseline: unbalanced loss & sampler) and Day 8 (balanced: weighted loss & sampler).
Features:
- CLI Configuration (Mode, Epochs, Batch Size, LR, Patience, etc.)
- Automatic Mixed Precision (AMP) optimized for NVIDIA RTX 3050 GPUs
- WeightedRandomSampler and Weighted CrossEntropyLoss loaders toggles
- Early Stopping on validation loss
- Comprehensive training metrics: Confusion Matrix, Classification Report, Per-class Recall
- TensorBoard logging & training curve plotting
- Seamless checkpoint resuming
"""

import os
import argparse
import copy
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
)

# Import model definition and dataset variables
try:
    from src.model_generator import LeafDiseaseCNN, NUM_CLASSES
    from src.DataLoader import (
        train_dataset,
        val_dataset,
        sample_weights,
        class_weight_tensor,
        CLASS_NAMES,
    )
except ImportError:
    # Handle if run from within src directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.model_generator import LeafDiseaseCNN, NUM_CLASSES
    from src.DataLoader import (
        train_dataset,
        val_dataset,
        sample_weights,
        class_weight_tensor,
        CLASS_NAMES,
    )

# Try importing TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Leaf Disease CNN Model - Day 7 Baseline vs Day 8 Balanced"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="day8",
        choices=["day7", "day8"],
        help="Training mode: 'day7' for baseline (unbalanced), 'day8' for balanced training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience epochs before early stopping."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training if a mode-specific checkpoint exists."
    )
    parser.add_argument(
        "--no-tb",
        action="store_true",
        help="Disable TensorBoard logging."
    )
    return parser.parse_args()


def get_data_loaders(mode, batch_size):
    """
    Sets up the PyTorch DataLoaders.
    Based on the mode, it either uses WeightedRandomSampler (day8) or standard shuffling (day7).
    """
    print(f"\n[Data Setup] Configuring dataloaders for Mode: {mode.upper()}...")
    
    if mode == "day8":
        # Weighted sampler for handling imbalanced dataset classes (Day 8 configuration)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print("--> WeightedRandomSampler active.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,  # 0 is recommended for stable multi-processing on Windows
            pin_memory=True
        )
    else:
        # Standard shuffle loader (Day 7 baseline configuration)
        print("--> Standard random shuffling active (no sampler).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    # Validation loader remains identical for fair benchmark comparison
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"--> Training batches  : {len(train_loader)}")
    print(f"--> Validation batches: {len(val_loader)}")
    return train_loader, val_loader


def get_loss_criterion(mode, device):
    """
    Returns the Loss criterion. Day 8 uses class-frequency weights, Day 7 uses unweighted loss.
    """
    if mode == "day8":
        print("--> Using Weighted CrossEntropyLoss (imbalance correcting).")
        # Align class weights length (3) with the model's output classes (16)
        full_weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32)
        full_weight_tensor[:len(class_weight_tensor)] = class_weight_tensor
        return nn.CrossEntropyLoss(weight=full_weight_tensor.to(device))
    else:
        print("--> Using Standard (Unweighted) CrossEntropyLoss.")
        return nn.CrossEntropyLoss()



def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP) for RTX GPUs.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="       Training",
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # PyTorch Autocast optimizes forward-pass variables to FP16 (RTX 3050 speedup)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if torch.isnan(loss):
            raise ValueError("Training generated NaN loss. Aborting training.")

        # Scales the loss, computes gradients, updates weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update tqdm live postfix stats
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}", 
            acc=f"{(correct/total)*100:.1f}%"
        )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = correct / len(loader.dataset)
    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluates the model on validation data. Returns loss, accuracy, predictions and labels.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(
        loader,
        desc="     Evaluating",
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_accuracy = correct / total
    return val_loss, val_accuracy, all_predictions, all_labels


def save_checkpoint(path, epoch, model, optimizer, scaler, best_weights, best_loss, best_acc, best_ep, wait, train_hist, val_hist):
    """
    Saves a full state checkpoint to resume training later.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_weights": best_weights,
        "best_val_loss": best_loss,
        "best_val_accuracy": best_acc,
        "best_epoch": best_ep,
        "wait": wait,
        "train_losses": train_hist["loss"],
        "train_accs": train_hist["acc"],
        "val_losses": val_hist["loss"],
        "val_accs": val_hist["acc"]
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    """
    Loads training state from a checkpoint.
    """
    print(f"--> Loading checkpoint from '{path}'...")
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_val_loss"]
    best_acc = checkpoint["best_val_accuracy"]
    best_ep = checkpoint["best_epoch"]
    wait = checkpoint["wait"]
    
    train_hist = {"loss": checkpoint.get("train_losses", []), "acc": checkpoint.get("train_accs", [])}
    val_hist = {"loss": checkpoint.get("val_losses", []), "acc": checkpoint.get("val_accs", [])}
    best_weights = checkpoint["best_weights"]
    
    print(f"--> Successfully loaded checkpoint. Resuming from Epoch {start_epoch}")
    return start_epoch, best_weights, best_loss, best_acc, best_ep, wait, train_hist, val_hist


def save_evaluation_reports(mode, labels, predictions):
    """
    Computes per-class recall, classification report, and confusion matrix, and saves them.
    Saves outputs in reports/ directory and duplicate CSVs to root workspace for ease of access.
    """
    os.makedirs("reports", exist_ok=True)
    
    # Restrict metrics calculations to the classes actually active in this dataset (len = 3)
    active_labels = list(range(len(CLASS_NAMES)))
    
    # Calculate Per-Class Recall
    recall = recall_score(
        labels,
        predictions,
        average=None,
        labels=active_labels,
        zero_division=0
    )
    
    df_recall = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Recall": recall
    })
    
    # Save mode-specific recalls in reports/
    reports_recall_path = f"reports/recall_{mode}.csv"
    df_recall.to_csv(reports_recall_path, index=False)
    print(f"[Export] Recall report saved -> {reports_recall_path}")
    
    # Export to root workspace paths as requested: recall_before.csv (day7) or recall_after.csv (day8)
    root_recall_filename = "recall_before.csv" if mode == "day7" else "recall_after.csv"
    df_recall.to_csv(root_recall_filename, index=False)
    print(f"[Export] Root recall report saved -> {root_recall_filename}")
    
    # Save Classification Report
    class_report_str = classification_report(
        labels,
        predictions,
        labels=active_labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    report_path = f"reports/classification_report_{mode}.txt"
    # Also save to generic reports/classification_report.txt for compatibility
    for path in [report_path, "reports/classification_report.txt"]:
        with open(path, "w") as file:
            file.write(class_report_str)
    print(f"[Export] Classification report saved -> {report_path}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(labels, predictions, labels=active_labels)
    df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = f"reports/confusion_matrix_{mode}.csv"
    # Also save to generic reports/confusion_matrix.csv for compatibility
    for path in [cm_path, "reports/confusion_matrix.csv"]:
        df_cm.to_csv(path)
    print(f"[Export] Confusion matrix saved -> {cm_path}")



def plot_training_curves(mode, train_hist, val_hist):
    """
    Saves visual loss and accuracy graphs.
    """
    epochs_range = range(1, len(train_hist["loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_hist["loss"], "o-", label="Train Loss")
    plt.plot(epochs_range, val_hist["loss"], "s-", label="Val Loss")
    plt.title(f"Loss Curves ({mode.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_hist["acc"], "o-", label="Train Accuracy")
    plt.plot(epochs_range, val_hist["acc"], "s-", label="Val Accuracy")
    plt.title(f"Accuracy Curves ({mode.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    curves_mode_path = f"reports/training_curves_{mode}.png"
    # Also save to reports/training_curves.png for compatibility
    for path in [curves_mode_path, "reports/training_curves.png"]:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[Export] Training curves saved -> {curves_mode_path}")


def gpu_memory_info(device):
    """
    Logs allocated GPU VRAM details.
    """
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"       GPU Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Leaf Disease classification - {args.mode.upper()} Pipeline")
    print(f"Running Device : {device}")
    if device.type == "cuda":
        print(f"GPU Model      : {torch.cuda.get_device_name(0)}")
        # Check if RTX 3050 is active and note optimizations
        if "3050" in torch.cuda.get_device_name(0):
            print("--> RTX 3050 GPU detected: optimizing AMP (FP16) scalability and worker load.")
    print("=" * 70)

    # Load DataLoaders
    train_loader, val_loader = get_data_loaders(args.mode, args.batch_size)

    # Model definition
    model = LeafDiseaseCNN(num_classes=NUM_CLASSES).to(device)

    # Optimizer, Criterion, and APM Scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = get_loss_criterion(args.mode, device)
    
    # GradScaler controls gradient scaling for float16 mixed precision training (RTX performance booster)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Config Checkpoint paths
    checkpoint_path = f"models/checkpoint_{args.mode}.pth"
    best_model_path = f"models/leaf_cnn_best_{args.mode}.pth"
    # Legacy path support
    legacy_best_path = "models/leaf_cnn_best.pth"

    # Training state variables
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    wait = 0
    
    train_hist = {"loss": [], "acc": []}
    val_hist = {"loss": [], "acc": []}
    best_weights = copy.deepcopy(model.state_dict())

    # Resume code path
    if args.resume and os.path.exists(checkpoint_path):
        (
            start_epoch, 
            best_weights, 
            best_val_loss, 
            best_val_accuracy, 
            best_epoch, 
            wait, 
            train_hist, 
            val_hist
        ) = load_checkpoint(checkpoint_path, model, optimizer, scaler, device)
    else:
        print("\n[Init] Starting fresh training sessions.")

    # Setup TensorBoard
    tb_writer = None
    if TENSORBOARD_AVAILABLE and not args.no_tb:
        log_dir = f"runs/leaf_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_writer = SummaryWriter(log_dir=log_dir)
        print(f"--> TensorBoard logging active: '{log_dir}'")
    else:
        status_reason = "disabled via --no-tb" if args.no_tb else "tensorboard package not installed"
        print(f"--> TensorBoard Logging: INACTIVE ({status_reason})")

    # Logging text file config
    log_file_path = f"reports/training_log_{args.mode}.txt"
    # Append to cumulative log log file
    log_file = open("reports/training_log.txt", "a")
    log_file.write(f"\n==================================================\n")
    log_file.write(f"Mode: {args.mode.upper()} Training Started at {datetime.now()}\n")
    log_file.write(f"Parameters: LR={args.lr}, BS={args.batch_size}, Epochs={args.epochs}, Patience={args.patience}\n")
    log_file.write(f"==================================================\n")

    print("\n" + "-" * 50)
    print(f"Hyperparameters: Epochs={args.epochs} | LR={args.lr} | BS={args.batch_size} | Patience={args.patience}")
    print("-" * 50)

    # ========================== Training Loop ==========================
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Save metrics history
        train_hist["loss"].append(train_loss)
        train_hist["acc"].append(train_acc)
        val_hist["loss"].append(val_loss)
        val_hist["acc"].append(val_acc)

        print(f"       Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"       Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc*100:.2f}%")
        gpu_memory_info(device)

        # TensorBoard Logging
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
            tb_writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Local Cumulative TXT Log
        log_file.write(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
        )
        log_file.flush()

        # Checkpoint Saving & Early Stopping Decision
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch
            wait = 0
            best_weights = copy.deepcopy(model.state_dict())
            
            # Save the best weights
            torch.save(best_weights, best_model_path)
            # Duplicate save to general best path for downstream evaluations
            torch.save(best_weights, legacy_best_path)
            print(f"       >>> Validation loss improved. Saved Best weights to '{best_model_path}'")
        else:
            wait += 1
            print(f"       >>> Loss Did Not Improve. Patience wait counts: ({wait}/{args.patience})")

        # Save latest resume checkpoint
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, scaler, best_weights,
            best_val_loss, best_val_accuracy, best_epoch, wait, train_hist, val_hist
        )

        if wait >= args.patience:
            print(f"\n[Early Stop] Stopped training as loss did not improve for {args.patience} epochs.")
            log_file.write(f"Early Stopping triggered at Epoch {epoch}\n")
            break

    # ========================== Final Evaluation ==========================
    print("\n" + "=" * 70)
    print("Training phase finished. Starting final evaluation...")
    print("=" * 70)
    
    # Close log writer and log file
    if tb_writer:
        tb_writer.close()
    
    # Restore the best validation weights
    print(f"--> Restoring Best Model Weights from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(best_weights)
    
    # Final Validation run to compute complete statistics
    final_loss, final_acc, all_preds, all_gts = validate(
        model, val_loader, criterion, device
    )

    print(f"\nResults on Best Weights:")
    print(f"Loss: {final_loss:.4f} | Accuracy: {final_acc*100:.2f}%")

    # Generate Reports
    save_evaluation_reports(args.mode, all_gts, all_preds)
    
    # Plot Curves
    plot_training_curves(args.mode, train_hist, val_hist)
    
    # Write summary statistics to logs
    log_file.write(f"\n---------------- TRAINING SUMMARY ----------------\n")
    log_file.write(f"Best Validation Loss     : {best_val_loss:.6f}\n")
    log_file.write(f"Best Validation Accuracy : {best_val_accuracy*100:.2f}%\n")
    log_file.write(f"Best Epoch               : {best_epoch}\n")
    log_file.write(f"--------------------------------------------------\n")
    log_file.close()

    print("\nTraining completed successfully! Saved all checkpoints, logs, and curves.")
    print("=" * 70)


if __name__ == "__main__":
    main()