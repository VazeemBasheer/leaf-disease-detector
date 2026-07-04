import os
import torch
import torch.nn as nn

# --------------------------------------------------
# Constants
# --------------------------------------------------

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Pepper_Bacterial_spot",
    "Pepper_healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

NUM_CLASSES = len(CLASS_NAMES)  # 16
NUM_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------
# CNN Model
# --------------------------------------------------

class LeafDiseaseCNN(nn.Module):
    """
    Simple Convolutional Neural Network for
    Plant Leaf Disease Classification.

    Input:
        (N, 3, 224, 224)

    Output:
        (N, NUM_CLASSES)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def estimate_memory(model):

    params = count_parameters(model)

    bytes_used = params * 4        # float32

    mb = bytes_used / (1024 ** 2)

    return mb


# --------------------------------------------------
# Demo
# --------------------------------------------------

if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)

    model = LeafDiseaseCNN()

    print(model)

    # Save architecture

    with open("models/architecture.txt", "w") as f:
        f.write(str(model))

    print("\nArchitecture saved to models/architecture.txt")

    # ----------------------------------------------

    # Train mode

    model.train()

    print("\nModel Mode:", "TRAIN")

    dummy = torch.randn(8, 3, 224, 224)

    logits = model(dummy)

    print("Dummy Output Shape :", logits.shape)

    assert logits.shape == (8, NUM_CLASSES)

    # ----------------------------------------------

    # Evaluation mode

    model.eval()

    print("Model Mode:", "EVAL")

    with torch.no_grad():

        logits = model(dummy)

    print("Eval Output Shape :", logits.shape)

    assert logits.shape == (8, NUM_CLASSES)

    # ----------------------------------------------

    total_params = count_parameters(model)

    print(f"\nTrainable Parameters : {total_params:,}")

    print(
        f"Estimated Memory     : {estimate_memory(model):.2f} MB"
    )

    print("\nAll tests passed.")