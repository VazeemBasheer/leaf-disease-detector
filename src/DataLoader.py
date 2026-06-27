from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ----------------------------------------
# Configuration
# ----------------------------------------

DATASET_ROOT = Path("data/processed")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0      # Windows
VAL_SPLIT = 0.2

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------------------
# Dataset
# ----------------------------------------

class LeafDiseaseDataset(Dataset):

    def __init__(self, root, transform=None):

        self.root = Path(root)
        self.transform = transform

        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # Find disease folders recursively
        class_dirs = sorted([
            d for d in self.root.rglob("*")
            if d.is_dir() and any(d.glob("*.JPG"))
               or any(d.glob("*.jpg"))
               or any(d.glob("*.png"))
               or any(d.glob("*.jpeg"))
        ])

        for idx, class_dir in enumerate(class_dirs):

            class_name = class_dir.name

            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):

                for img_path in class_dir.glob(ext):

                    try:
                        Image.open(img_path).verify()
                        self.samples.append((img_path, idx))
                    except Exception:
                        print(f"Broken image skipped: {img_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------------------
# Create dataset
# ----------------------------------------

dataset = LeafDiseaseDataset(DATASET_ROOT, transform)

print(f"\nTotal Images : {len(dataset)}")
print(f"Total Classes: {len(dataset.classes)}\n")

# ----------------------------------------
# Class distribution
# ----------------------------------------

counts = Counter()

for _, label in dataset.samples:
    counts[label] += 1

print("Class Distribution")
print("-" * 40)

for class_name, idx in dataset.class_to_idx.items():
    print(f"{class_name:45} {counts[idx]}")

# ----------------------------------------
# Train / Validation Split
# ----------------------------------------

train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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

print("\nTrain batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# ----------------------------------------
# Verify one batch
# ----------------------------------------

images, labels = next(iter(train_loader))

print("\nBatch Shape :", images.shape)
print("Labels      :", labels[:8])

# Expected:
# torch.Size([32, 3, 224, 224])

# ----------------------------------------
# Visualize first image
# ----------------------------------------

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

img = images[0] * std + mean
img = img.clamp(0,1)

plt.imshow(img.permute(1,2,0))
plt.title(dataset.classes[labels[0]])
plt.axis("off")
plt.show()