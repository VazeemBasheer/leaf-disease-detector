import os
import sys
import time
import json
import copy
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
)
from torchvision import models

# Import dataset variables
try:
    from src.DataLoader import (
        train_loader,
        val_loader,
        CLASS_NAMES,
        class_weight_tensor,
    )
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.DataLoader import (
        train_loader,
        val_loader,
        CLASS_NAMES,
        class_weight_tensor,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on Leaf Disease Dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Total number of epochs (Phase 1 + Phase 2).")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Number of epochs for backbone frozen Phase 1.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience during Phase 2.")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for fully connected head.")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for backbone (layer4).")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"ResNet18 Fine-Tuning Pipeline Started")
    print(f"Execution Device : {device}")
    if device.type == "cuda":
        print(f"GPU Hardware Model: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 2. Model Setup
    print("[Model Setup] Initializing Pretrained ResNet18...")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Replace classifier head (fc)
    num_classes = len(CLASS_NAMES)
    in_features = backbone.fc.in_features
    # Replaced head has requires_grad = True by default
    backbone.fc = nn.Linear(in_features, num_classes)
    backbone = backbone.to(device)

    print(f"Classifier head (fc) replaced with Linear({in_features} -> {num_classes})")
    print(f"Target plant categories: {CLASS_NAMES}")

    # Save class_names.json as requested
    class_names_path = "models/class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(CLASS_NAMES, f)
    print(f"[Export] Saved class name mappings -> '{class_names_path}'")

    # 3. Training Components
    # Loss criterion uses inverse family weights
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Initial optimizer for Phase 1: Only head parameters trainable
    head_params = list(backbone.fc.parameters())
    optimizer = optim.Adam([{"params": head_params, "lr": args.lr_head}])

    # 4. Training State
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    best_weights = copy.deepcopy(backbone.state_dict())

    train_hist = {"loss": [], "acc": []}
    val_hist = {"loss": [], "acc": []}

    total_start_time = time.time()

    # Log text file config
    log_path = "reports/training_log.txt"
    log_file = open(log_path, "a")
    log_file.write(f"\n==================================================\n")
    log_file.write(f"ResNet18 Fine-Tuning Started at {datetime.now()}\n")
    log_file.write(f"Parameters: Epochs={args.epochs}, Phase1={args.phase1_epochs}, LR_head={args.lr_head}, LR_backbone={args.lr_backbone}\n")
    log_file.write(f"Active Hardware : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})\n")
    log_file.write(f"==================================================\n")

    # ========================== Training Loop ==========================
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Step into Phase 2: Unfreeze layer4 backbone
        if epoch == args.phase1_epochs + 1:
            print("\n" + "#" * 60)
            print(">>> STEP INTO PHASE 2: Unfreezing layer4 backbone...")
            for param in backbone.layer4.parameters():
                param.requires_grad = True

            # Reconfigure optimizer parameter groups
            optimizer = optim.Adam([
                {"params": backbone.fc.parameters(), "lr": args.lr_head},
                {"params": backbone.layer4.parameters(), "lr": args.lr_backbone}
            ])
            print(">>> Re-initialized optimizer with head and layer4 parameter groups.")
            print("#" * 60 + "\n")

        phase_title = "PHASE 1 (Frozen Backbone)" if epoch <= args.phase1_epochs else "PHASE 2 (Unfrozen layer4)"
        print(f"\nEpoch {epoch}/{args.epochs} | {phase_title}")

        # Training Epoch
        backbone.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc="Batch Training",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Auto Mixed Precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = backbone(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Validation Epoch
        backbone.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = backbone(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_gts.extend(labels.cpu().tolist())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start_time

        # Track history
        train_hist["loss"].append(epoch_train_loss)
        train_hist["acc"].append(epoch_train_acc)
        val_hist["loss"].append(epoch_val_loss)
        val_hist["acc"].append(epoch_val_acc)

        print(f"       Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"       Val Loss  : {epoch_val_loss:.4f} | Val Acc  : {epoch_val_acc*100:.2f}%")
        print(f"       Epoch Time: {epoch_time:.2f}s")

        # Logging output to cumulative txt file
        log_file.write(
            f"Epoch {epoch:02d} ({phase_title[:8]}) | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Time: {epoch_time:.2f}s\n"
        )
        log_file.flush()

        # Checkpoint Saving based on accuracy
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(backbone.state_dict())

            # Save best checkpoint with custom metadata
            checkpoint_data = {
                "state_dict": best_weights,
                "epoch": best_epoch,
                "val_acc": best_val_accuracy,
                "class_list": CLASS_NAMES,
                "class_to_idx": {name: idx for idx, name in enumerate(CLASS_NAMES)},
                "model_description": "ResNet18 Fine-Tuned for Leaf Disease Classification"
            }
            torch.save(checkpoint_data, "models/resnet18_leaf_best.pth")
            print(f"       >>> Validation accuracy improved. Saved best model to models/resnet18_leaf_best.pth")

        # Early stopping check (based on loss improvement in Phase 2)
        if epoch > args.phase1_epochs:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                wait = 0
            else:
                wait += 1
                print(f"       >>> Validation loss did not improve. Patience: ({wait}/{args.patience})")
                if wait >= args.patience:
                    print(f"\n[Early Stopping] Phase 2 training stopped early at Epoch {epoch}")
                    log_file.write(f"Early Stopping triggered at Epoch {epoch}\n")
                    break

    total_training_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"Training completed successfully! Total Time: {total_training_time:.2f}s")
    print("=" * 70)

    # 5. Final Evaluation Reports
    # Restore the best validation weights
    print(f"--> Restoring Best Model Weights from Epoch {best_epoch} (Val Accuracy: {best_val_accuracy*100:.2f}%)")
    backbone.load_state_dict(best_weights)

    backbone.eval()
    final_preds, final_gts = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = backbone(images)
            _, preds = outputs.max(1)
            final_preds.extend(preds.cpu().tolist())
            final_gts.extend(labels.cpu().tolist())

    # Generate Confusion Matrix, classification report, and recall CSV
    active_labels = list(range(len(CLASS_NAMES)))
    
    # Save Classification Report
    resnet_report = classification_report(
        final_gts,
        final_preds,
        labels=active_labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    with open("reports/classification_report_resnet.txt", "w") as f:
        f.write(resnet_report)
    print("[Export] Saved ResNet classification report -> reports/classification_report_resnet.txt")

    # Save Confusion Matrix
    cm = confusion_matrix(final_gts, final_preds, labels=active_labels)
    df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_cm.to_csv("reports/confusion_matrix_resnet.csv")
    print("[Export] Saved ResNet confusion matrix -> reports/confusion_matrix_resnet.csv")

    # Plot Curves
    epochs_range = range(1, len(train_hist["loss"]) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_hist["loss"], "o-", label="Train Loss")
    plt.plot(epochs_range, val_hist["loss"], "s-", label="Val Loss")
    plt.title("ResNet18 Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_hist["acc"], "o-", label="Train Accuracy")
    plt.plot(epochs_range, val_hist["acc"], "s-", label="Val Accuracy")
    plt.title("ResNet18 Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reports/training_curves_resnet.png", dpi=300)
    plt.close()
    print("[Export] Saved ResNet training curves -> reports/training_curves_resnet.png")

    # Compute comparison metrics
    # Load Day 8 baseline CNN scores
    scratch_acc = 98.80
    scratch_recall_apple = 0.9843
    scratch_recall_pepper = 0.9898
    scratch_recall_tomato = 0.9884

    # Calculate ResNet18 recalls
    recall = recall_score(final_gts, final_preds, average=None, labels=active_labels, zero_division=0)
    resnet_accuracy = (np.array(final_preds) == np.array(final_gts)).mean() * 100

    compare_report = f"""======================================================================
COMPARATIVE EVALUATION: SCRATCH CNN VS FINE-TUNED RESNET18
======================================================================
Generated on: {datetime.now()}
Hardware Used: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})
Total Training Time: {total_training_time:.2f} seconds

----------------------------------------------------------------------
OVERALL RESULTS COMPARISON
----------------------------------------------------------------------
Model                 | Val Accuracy
----------------------+-------------
Scratch CNN (Day 8)   | {scratch_acc:.2f}%
ResNet18 Fine-Tuned   | {resnet_accuracy:.2f}%
Accuracy Delta        | {resnet_accuracy - scratch_acc:+.2f}%

----------------------------------------------------------------------
CROP-LEVEL RECALL RECOVERY COMPARISON
----------------------------------------------------------------------
Crop Family  | Scratch Recall | ResNet18 Recall | Target Delta
-------------+----------------+-----------------+-------------
Apple        | {scratch_recall_apple:.4f}         | {recall[0]:.4f}          | {recall[0] - scratch_recall_apple:+.4f}
Pepper       | {scratch_recall_pepper:.4f}         | {recall[1]:.4f}          | {recall[1] - scratch_recall_pepper:+.4f}
Tomato       | {scratch_recall_tomato:.4f}         | {recall[2]:.4f}          | {recall[2] - scratch_recall_tomato:+.4f}
======================================================================
"""
    with open("reports/resnet18_vs_scratch_cnn.txt", "w") as f:
        f.write(compare_report)
    print("[Export] Saved comparative evaluation report -> reports/resnet18_vs_scratch_cnn.txt")
    print(compare_report)

    # Write summary statistics to log file
    log_file.write(f"\n---------------- TRAINING SUMMARY ----------------\n")
    log_file.write(f"Total training time      : {total_training_time:.2f} seconds\n")
    log_file.write(f"Best Validation Accuracy : {best_val_accuracy*100:.2f}%\n")
    log_file.write(f"Best Epoch               : {best_epoch}\n")
    log_file.write(f"--------------------------------------------------\n")
    log_file.close()

if __name__ == "__main__":
    main()
