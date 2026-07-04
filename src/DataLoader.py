from pathlib import Path
from collections import Counter
import csv

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    Subset
)

from src.transforms import (
    train_transform,
    val_transform,
)

# ==========================================================
# Configuration
# ==========================================================

DATASET_ROOT = Path("data/processed")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0      # Windows
VAL_SPLIT = 0.20
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

# ==========================================================
# Dataset
# ==========================================================


class LeafDiseaseDataset(Dataset):

    def __init__(self, root, transform=None):

        self.root = Path(root)
        self.transform = transform

        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        class_dirs = sorted(
            [
                d
                for d in self.root.iterdir()
                if d.is_dir()
            ]
        )

        for class_index, class_dir in enumerate(class_dirs):

            self.classes.append(class_dir.name)

            self.class_to_idx[class_dir.name] = class_index

            for image_path in class_dir.rglob("*"):

                if image_path.suffix not in IMAGE_EXTENSIONS:
                    continue

                try:

                    Image.open(image_path).verify()

                    self.samples.append(
                        (
                            image_path,
                            class_index
                        )
                    )

                except Exception:

                    print(
                        f"Broken image skipped: {image_path}"
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# ==========================================================
# Full Dataset (No Transform)
# ==========================================================

full_dataset = LeafDiseaseDataset(
    DATASET_ROOT,
    transform=None
)

print(f"\nTotal Images : {len(full_dataset)}")
print(f"Total Classes: {len(full_dataset.classes)}")
# ==========================================================
# Reproducible Train / Validation Split
# ==========================================================

total_images = len(full_dataset)

indices = torch.randperm(
    total_images,
    generator=torch.Generator().manual_seed(RANDOM_SEED)
).tolist()

split = int(total_images * (1 - VAL_SPLIT))

train_indices = indices[:split]
val_indices = indices[split:]

# ==========================================================
# Create Separate Dataset Objects
# ==========================================================

train_dataset = LeafDiseaseDataset(
    DATASET_ROOT,
    transform=train_transform
)

val_dataset = LeafDiseaseDataset(
    DATASET_ROOT,
    transform=val_transform
)

# ==========================================================
# Create Independent Subsets
# ==========================================================

train_dataset = Subset(
    train_dataset,
    train_indices
)

val_dataset = Subset(
    val_dataset,
    val_indices
)

print(f"\nTraining Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")

# ==========================================================
# Compute Class Distribution (Training Only)
# ==========================================================

train_labels = []

for idx in train_indices:

    _, label = full_dataset.samples[idx]

    train_labels.append(label)

class_counts = Counter(train_labels)

print("\nTraining Class Distribution")
print("-" * 60)

for class_name, class_index in full_dataset.class_to_idx.items():

    print(
        f"{class_name:45}"
        f"{class_counts[class_index]}"
    )

# ==========================================================
# Compute Inverse Frequency Weights
# ==========================================================

num_classes = len(full_dataset.classes)

total_train = len(train_labels)

class_weights = {}

for class_index in range(num_classes):

    count = class_counts[class_index]

    class_weights[class_index] = (
        total_train /
        (num_classes * count)
    )

# ==========================================================
# Export reports/class_balance.csv
# ==========================================================

csv_path = REPORT_DIR / "class_balance.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Class Index",
            "Class Name",
            "Training Images",
            "Weight"
        ]
    )

    for class_name, class_index in full_dataset.class_to_idx.items():

        writer.writerow(
            [
                class_index,
                class_name,
                class_counts[class_index],
                round(class_weights[class_index], 6)
            ]
        )

print(f"\nClass balance report saved to:\n{csv_path}")
# ==========================================================
# Sample Weights for WeightedRandomSampler
# ==========================================================

sample_weights = []

for index in train_indices:

    _, label = full_dataset.samples[index]

    sample_weights.append(
        class_weights[label]
    )

sample_weights = torch.DoubleTensor(sample_weights)

# ==========================================================
# Weighted Random Sampler
# ==========================================================

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ==========================================================
# Weighted CrossEntropy Tensor
# ==========================================================

class_weight_tensor = torch.tensor(
    [
        class_weights[i]
        for i in range(num_classes)
    ],
    dtype=torch.float32
)

print("\nTrain Batches      :", len(train_loader))
print("Validation Batches :", len(val_loader))

print("\nWeightedRandomSampler Enabled")
print("Weighted CrossEntropy Enabled")

# ==========================================================
# Verify One Batch
# ==========================================================

images, labels = next(iter(train_loader))

print("\nBatch Shape :", images.shape)
print("Labels      :", labels[:8])
# ==========================================================
# Visualize Augmentations
# ==========================================================

print("\nGenerating augmentation preview...")

# Use the first image from the training subset
sample_image_path, sample_label = train_dataset.dataset.samples[
    train_dataset.indices[0]
]

original_image = Image.open(sample_image_path).convert("RGB")

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6)
)

for ax in axes.flatten():

    augmented = train_transform(original_image)

    # Undo normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    augmented = augmented * std + mean
    augmented = augmented.clamp(0, 1)

    ax.imshow(
        augmented.permute(1, 2, 0)
    )

    ax.axis("off")

plt.tight_layout()

augment_path = REPORT_DIR / "augment_samples.png"

plt.savefig(
    augment_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Augmentation samples saved to:\n{augment_path}")

# ==========================================================
# Export Variables
# ==========================================================

CLASS_NAMES = full_dataset.classes

CLASS_COUNTS = {
    CLASS_NAMES[i]: class_counts[i]
    for i in range(num_classes)
}

CLASS_WEIGHTS = {
    CLASS_NAMES[i]: class_weights[i]
    for i in range(num_classes)
}

NUM_CLASSES = num_classes

# ==========================================================
# Module Exports
# ==========================================================

__all__ = [
    "train_loader",
    "val_loader",
    "train_dataset",
    "val_dataset",
    "CLASS_NAMES",
    "CLASS_COUNTS",
    "CLASS_WEIGHTS",
    "class_weight_tensor",
    "NUM_CLASSES",
]