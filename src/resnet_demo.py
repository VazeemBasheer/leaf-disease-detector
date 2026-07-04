import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import warnings

class Logger(object):
    """
    Logger that duplicates stdout writes to a specified file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    os.makedirs("docs", exist_ok=True)
    log_path = os.path.join("docs", "resnet18_demo_output.log")
    
    # Set sys.stdout to Logger to mirror output to doc file
    sys.stdout = Logger(log_path)

    # 1. Trigger deprecated pretrained=True warning
    print("\n=== STEP 1: Testing Deprecated API (pretrained=True) ===")
    try:
        # Load models using deprecated API
        # By default this triggers a UserWarning in torchvision
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = models.resnet18(pretrained=True)
            for warn in w:
                print(f"[UserWarning caught]: {warn.message}")
    except Exception as e:
        print(f"Could not load with pretrained=True (expected in some strict environments): {e}")

    # 2. Load backbone using weights enum API
    print("\n=== STEP 2: Loading ResNet18 with Weights Enum ===")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    print("Pretrained ResNet18 model loaded successfully.")

    # 3. Save module layers hierarchy list
    print("\n=== STEP 3: Saving Layer Names to docs/resnet18_layers.txt ===")
    layer_names = []
    for name, module in backbone.named_modules():
        if name: # skip empty string which is the root model itself
            layer_names.append(f"{name} ({module.__class__.__name__})")
            
    layers_file_path = os.path.join("docs", "resnet18_layers.txt")
    with open(layers_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(layer_names))
    print(f"Saved {len(layer_names)} layer names to '{layers_file_path}'")

    # 4. Run frozen-backbone forward pass on leaf image before any training
    print("\n=== STEP 4: Running Pretrained Inference on Leaf Image ===")
    img_path = "data/processed/apple/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG"
    if not os.path.exists(img_path):
        print(f"Could not find leaf image at '{img_path}'. Using dummy image tensor.")
        input_batch = torch.randn(1, 3, 224, 224)
    else:
        # Load leaf image using PIL
        img = Image.open(img_path).convert("RGB")
        # Apply standard ImageNet normalization
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # creating a batch [1, 3, 224, 224]
        print(f"Successfully loaded and preprocessed leaf image to shape: {input_batch.shape}")

    # Forward pass on pretrained backbone
    backbone.eval()
    with torch.no_grad():
        output = backbone(input_batch)
    
    # Calculate top-1 ImageNet prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    imagenet_categories = weights.meta["categories"]
    predicted_class = imagenet_categories[top1_catid.item()]
    print(f"Top-1 Predicted ImageNet Category: '{predicted_class}' with probability: {top1_prob.item():.4f}")

    # 5. Load and count parameters BEFORE replacing the head
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params_before = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nBefore Head Replacement:")
    print(f"--> Total parameters: {total_params:,}")
    print(f"--> Trainable parameters: {trainable_params_before:,}")

    # 6. Freeze backbone parameters
    print("\n=== STEP 5: Freezing Backbone Parameters ===")
    for param in backbone.parameters():
        param.requires_grad = False

    # 7. Replace the head (fully connected layer)
    print("\n=== STEP 6: Replacing Classifier Head for Transfer Learning ===")
    num_classes = 4 # apple, pepper, tomato, healthy/other
    in_features = backbone.fc.in_features
    # Replaced head has requires_grad = True by default
    backbone.fc = nn.Linear(in_features, num_classes)
    print(f"Classifier head (fc) replaced with Linear({in_features} -> {num_classes})")

    # Double check parameter requires_grad configuration
    for name, param in backbone.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 8. Count parameters AFTER replacing classification head and freezing backbone
    total_params_after = sum(p.numel() for p in backbone.parameters())
    trainable_params_after = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nAfter Head Replacement & Freezing:")
    print(f"--> Total parameters: {total_params_after:,}")
    print(f"--> Trainable parameters: {trainable_params_after:,} (Only 'fc.weight' and 'fc.bias')")

    # 9. Verify forward pass on replaced head model
    with torch.no_grad():
        output_custom = backbone(input_batch)
    print(f"\n=== STEP 7: Verifying Custom Replaced Head Forward Pass ===")
    print(f"Output shape of leaf batch through fine-tuned head: {output_custom.shape}")
    print(f"Raw output logits: {output_custom[0].tolist()}")
    print("Verification completed successfully!\n")
    
    # Restore stdout
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()
