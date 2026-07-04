from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.transforms import train_transform

# ----------------------------------------
# Find one sample image
# ----------------------------------------

DATASET_ROOT = Path("data/processed")

extensions = ("*.jpg", "*.JPG", "*.png", "*.jpeg")

image_path = None

for ext in extensions:
    files = list(DATASET_ROOT.rglob(ext))
    if files:
        image_path = files[0]
        break

if image_path is None:
    raise FileNotFoundError("No image found.")

print("Using image:")
print(image_path)

# ----------------------------------------
# Original image
# ----------------------------------------

original = Image.open(image_path).convert("RGB")

# ----------------------------------------
# Plot
# ----------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Original
axes[0, 0].imshow(original)
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

# Remaining 8 images
positions = [
    (0,1),(0,2),
    (1,0),(1,1),(1,2),
    (2,0),(2,1),(2,2)
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

for i, (r, c) in enumerate(positions):

    tensor = train_transform(original)

    # Undo normalization
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)

    tensor = tensor.clamp(0, 1)

    axes[r, c].imshow(tensor.permute(1, 2, 0))
    axes[r, c].set_title(f"Augmented {i+1}")
    axes[r, c].axis("off")

plt.tight_layout()

Path("reports").mkdir(exist_ok=True)

plt.savefig(
    "reports/augment_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("reports/augment_comparison.png")