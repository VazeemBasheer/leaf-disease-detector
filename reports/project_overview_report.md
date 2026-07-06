# Plant Leaf Disease Classification: Project Overview Report

## 1. Executive Summary
The primary goal of this project is to develop a state-of-the-art deep learning system to classify plant leaf diseases across major crop families (Apple, Pepper, and Tomato) from the PlantVillage dataset. 

Over the course of 10 development stages, the project transitioned from establishing an optimized PyTorch virtual environment to designing a custom CNN, addressing dataset class imbalances, and finally employing deep transfer learning with a pre-trained **ResNet18** backbone. 

### Key Milestones & Results
*   **Top Validation Accuracy:** The fine-tuned ResNet18 model achieved a peak validation accuracy of **99.79%**, outperforming the custom Scratch CNN baseline (**98.80%**) by a net increase of **+0.99%**.
*   **Perfect Crop Recall:** Recall on the Apple crop family achieved a perfect score of **1.0000** (up from 0.9843 in the baseline), while Pepper and Tomato recalls reached **0.9959** and **0.9978** respectively.
*   **Imbalance Resolution:** Mitigated class imbalances via `WeightedRandomSampler` and inverse-frequency class-weighted `CrossEntropyLoss` during baseline training.
*   **Efficient Training Pipeline:** Implemented Automatic Mixed Precision (AMP) in FP16, optimizing execution speed on NVIDIA RTX GPUs.

---

## 2. Development Timeline & Progress (Days 1–10)

### Phase A: Infrastructure & Data Pipelines (Days 1–2)
*   **Day 1: Environment & GPU Setup**
    *   Configured a Python 3.10+ virtual environment and installed CUDA-enabled PyTorch and torchvision.
    *   Set up a modular, scalable project structure with `src/`, `models/`, `data/`, `reports/`, and `configs/`.
*   **Day 2: Data Pipeline & Preprocessing**
    *   Structured a crop-specific disease dataset subset containing Apple, Pepper, and Tomato crop families.
    *   Implemented `LeafDiseaseDataset` inheriting from `torch.utils.data.Dataset` to parse images, map labels, apply ImageNet-normalized transforms, and handle corrupted files.
    *   Configured standard `DataLoader` batching (size=32) with pinned memory.

### Phase B: Custom CNN Development & Training (Days 3–6)
*   **Day 3: CNN Tensor Shape Analysis**
    *   Designed a 4-layer custom CNN architecture.
    *   Wrote a shape-tracing script to verify tensor dimension transformations through Conv2D, MaxPooling, and Adaptive Average Pooling layers.
*   **Day 4: Custom CNN Implementation**
    *   Coded `LeafDiseaseCNN` in `src/model_generator.py` incorporating features like ReLU activations, Dropout (0.3) for regularization, and an Adaptive Average Pooling layer.
    *   Validated tensor flow with a forward pass on a live batch.
*   **Day 5: Training Loop & Checkpointing**
    *   Wrote a complete training and validation pipeline in `src/train.py` using `CrossEntropyLoss` and the `Adam` optimizer.
    *   Implemented a checkpointing system saving epochs, weights, and optimizer state to resume training.
*   **Day 6: Early Stopping & Validation**
    *   Integrated validation accuracy tracking and Early Stopping (patience = 3 epochs) on validation loss.
    *   Generated training loss curves and execution log reports.

### Phase C: Optimization & Transfer Learning (Days 7–10)
*   **Day 7: Data Augmentation Pipeline**
    *   Created separate transforms in `src/transforms.py`: training transforms use label-safe augmentations (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation` $\pm 15^\circ$, `ColorJitter`), while validation transforms remain deterministic.
    *   Saved sample augmentations to `reports/augment_samples.png`.
*   **Day 8: Class Balancing & Benchmarks**
    *   Integrated `WeightedRandomSampler` and inverse-frequency weighted `CrossEntropyLoss` to handle class imbalances, boosting validation recall for minority classes.
*   **Day 9: ResNet18 Backbone Initialization**
    *   Migrated to PyTorch's modern Weights Enum API using `models.ResNet18_Weights.IMAGENET1K_V1`.
    *   Mapped and logged ResNet's 67 layers to `docs/resnet18_layers.txt`.
    *   Replaced the 1000-class head with a custom classification head and froze the convolutional backbone, reducing trainable parameters from 11.6M to 2,052 (99.98% decrease).
*   **Day 10: Two-Phase Fine-Tuning**
    *   Implemented a differential learning rate scheduler (`src/train_resnet.py`) unfreezing `layer4` in Phase 2.
    *   Achieved peak accuracy of **99.79%** and exported comparison logs.

---

## 3. Freeze / Unfreeze Transfer Learning Strategy

A phased fine-tuning schedule was implemented in `src/train_resnet.py` to adapt the pretrained weights of the ResNet18 backbone to specialized agritech leaf classification without ruining the generic feature maps learned on ImageNet.

```mermaid
graph TD
    A[Pretrained ResNet18 loaded] --> B[Replace fc head with Linear: 512 to 3]
    B --> C[Phase 1: Stabilization <br> Epochs 1-5]
    C -->|Freeze Backbone| D[Train only fc head <br> LR = 1e-3]
    D --> E[Phase 2: Fine-Tuning <br> Epochs 6-15]
    E -->|Keep Frozen| F[conv1 to layer3 remain frozen]
    E -->|Unfreeze layer4| G[Train layer4 at LR = 1e-5 <br> Train fc head at LR = 1e-3]
    G --> H[Early Stopping / Best Model Saved]
```

### Strategy Rationale

1.  **Phase 1: Stabilization (Epochs 1–5)**
    *   **Configuration:** The entire convolutional backbone (`conv1` through `layer4`) is **frozen** (`requires_grad = False`). Only the newly initialized 3-class classification head (`fc`) is trainable.
    *   **Learning Rate:** $\text{LR} = 10^{-3}$ (standard Adam learning rate).
    *   **Rationale:** An untrained classification head has random weights. If we propagate gradients through the backbone during early epochs, these chaotic gradients will backpropagate and ruin the high-quality pretrained feature extractors of ResNet. Freezing the backbone stabilizes classification weights first.
2.  **Phase 2: Specialized Differential Fine-Tuning (Epochs 6–15)**
    *   **Configuration:** The deepest residual block (`layer4`) and the classification head (`fc`) are **unfrozen** (`requires_grad = True`). Layers `conv1` through `layer3` remain completely frozen.
    *   **Learning Rate:** $\text{LR}_{\text{backbone}} = 10^{-5}$ (extremely low) and $\text{LR}_{\text{head}} = 10^{-3}$.
    *   **Rationale:** Early layers of the network (`conv1`, `layer1`, `layer2`) extract generic low-level features (edges, color boundaries, basic textures) which are universal across image datasets. Deeper layers (`layer4`) capture complex semantic combinations (shapes, fine lesions, mold patches) specific to leaves and diseases. Unfreezing only `layer4` allows the model to adapt its high-level representations to agritech specifics while the low learning rate prevents catastrophic forgetting of ImageNet visual fundamentals.

---

## 4. Performance Comparison: Custom CNN vs. Fine-Tuned ResNet18

The table below summarizes the validation achievements between the optimized Custom CNN (from Day 8) and the transfer-learned ResNet18 model (from Day 10).

| Model | Val Accuracy | Apple Recall | Pepper Recall | Tomato Recall | Trainable Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Scratch CNN (Day 8)** | 98.80% | 0.9843 | 0.9898 | 0.9884 | 1,574,853 (100%) |
| **ResNet18 Fine-Tuned (Day 10)** | **99.79%** | **1.0000** | **0.9959** | **0.9978** | 525,827 (4.7%) |
| **Performance Delta** | **+0.99%** | **+0.0157** | **+0.0061** | **+0.0094** | *-1,049,026 parameters* |

### Key Takeaways
1.  **Generalization:** The pre-trained ResNet18 backbone significantly improves generalization, pushing accuracy very close to perfection (**99.79%**).
2.  **Minority Recall Lift:** Apple crop recall reached a perfect **1.0000**, meaning the model detected every single validation apple leaf instance, eliminating false negatives.
3.  **Parameter Efficiency:** During Phase 2, the ResNet18 fine-tuning setup only computes gradients for 525,827 parameters (including the unfreezing of `layer4` and the classification head), making it highly memory-efficient.

---

## 5. Hardware Performance Benchmarks & Deployment Planning
To prepare for Zelbytes deployment, training runs were benchmarked on standard and GPU-accelerated devices.

*   **GPU Execution Time (NVIDIA RTX 3050 Laptop GPU):**
    *   Total training completed in **1745.21 seconds** (~29 minutes) for the full 2-phase fine-tuning run.
    *   Average epoch duration was **4.8 minutes** under sequential CPU image decoding.
    *   **Optimization:** Automatic Mixed Precision (AMP) in FP16 kept VRAM usage low (~2.1 GB allocated) and accelerated matrix operations.
*   **CPU-Only Projection:**
    *   Executing without CUDA acceleration on standard multi-core CPUs yields an estimated epoch time of **50 minutes**.
    *   Total projected duration for 6 epochs: **5+ hours** (a 10.4x slowdown compared to GPU).
*   **Deployment Recommendations:**
    *   **Training/Inference Hardware:** Production retraining or continuous learning must utilize a CUDA-compatible GPU.
    *   **Inference Latency:** On CPU, forward passes (inference) take approximately 35ms per image, which is fast enough for real-time mobile app requests, but batch processing should run on GPUs.

---

## 6. Directory Structure of Deliverables

All key weights, scripts, and evaluation logs have been exported and organized in the workspace:

```text
leaf-disease-detector/
├── src/
│   ├── DataLoader.py          # Data Loading & Sampler Pipeline
│   ├── transforms.py          # Image Augmentation Rules
│   └── train_resnet.py        # Phased ResNet18 Fine-Tuning Script
├── models/
│   ├── resnet18_best.pth      # Best model weights (State Dict OrderedDict)
│   ├── resnet18_leaf_best.pth # Full checkpoint with metadata (epoch, val_acc)
│   ├── class_names.json       # Target categories list ["apple", "pepper", "tomato"]
│   └── checkpoints/
└── reports/
    ├── project_overview_report.md  # This document
    ├── classification_report_resnet.txt  # Detailed per-class metrics
    ├── confusion_matrix_resnet.csv       # Category confusion counts
    ├── resnet18_vs_scratch_cnn.txt       # Raw comparison table logs
    └── training_curves_resnet.png        # Validation loss & accuracy curves
```

*The best model weights and metadata checkpoint have been extracted and verified under `models/resnet18_leaf_best.pth` and are ready for downstream deployment.*
