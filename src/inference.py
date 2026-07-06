import os
import sys
import json
import torch
from PIL import Image
from torchvision import models

# Append parent directory to solve local package resolution
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.transforms import val_transform

def load_model(weights_path):
    """
    Dynamically loads the proper model architecture and classifier head from a checkpoint.
    Supports ResNet18, MobileNetV2, and LeafDiseaseCNN.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint file not found at '{weights_path}'")
        
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Resolve state_dict dictionary key
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Inspect state dict to instantiate the correct model architecture
    if 'fc.weight' in state_dict:
        # ResNet18
        num_classes = state_dict['fc.weight'].shape[0]
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
    elif 'classifier.1.weight' in state_dict:
        # MobileNetV2
        num_classes = state_dict['classifier.1.weight'].shape[0]
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(state_dict)
    elif 'classifier.2.weight' in state_dict:
        # LeafDiseaseCNN (our custom CNN)
        num_classes = state_dict['classifier.2.weight'].shape[0]
        from src.model_generator import LeafDiseaseCNN
        model = LeafDiseaseCNN(num_classes=num_classes)
        model.load_state_dict(state_dict)
    else:
        # Fallback to ResNet18 if we cannot determine
        num_classes = 3
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)

    model.eval()
    return model, num_classes, checkpoint

def get_class_names(checkpoint, num_classes):
    """
    Attempts to retrieve class names from the checkpoint metadata or local config files.
    """
    # 1. From checkpoint dictionary metadata
    if isinstance(checkpoint, dict) and 'class_list' in checkpoint:
        return checkpoint['class_list']
    if isinstance(checkpoint, dict) and 'class list' in checkpoint:
        return checkpoint['class list']

    # 2. From inference_config.json if the number of classes matches
    config_path = "models/inference_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if config.get("num_classes") == num_classes and "classes" in config:
                    return config["classes"]
        except Exception:
            pass

    # 3. From class_names.json if length matches
    names_path = "models/class_names.json"
    if os.path.exists(names_path):
        try:
            with open(names_path, "r", encoding="utf-8") as f:
                names = json.load(f)
                if len(names) == num_classes:
                    return names
        except Exception:
            pass

    # 4. Global fallback imports
    if num_classes == 16:
        try:
            from src.model_generator import CLASS_NAMES
            return CLASS_NAMES
        except ImportError:
            pass
    elif num_classes == 3:
        return ["apple", "pepper", "tomato"]

    return [f"class_{i}" for i in range(num_classes)]

def get_operating_threshold(num_classes):
    """
    Loads Operating threshold from config. Defaults to 0.35 for 16-class model, otherwise 0.50.
    """
    config_path = "models/inference_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if config.get("num_classes") == num_classes:
                    return config.get("operating_threshold", 0.35 if num_classes == 16 else 0.50)
        except Exception:
            pass
    return 0.35 if num_classes == 16 else 0.50

def predict_image(img, model, class_names, threshold):
    """
    Takes a PIL Image object, processes it through val_transform, runs prediction.
    """
    tensor = val_transform(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
    conf, idx = probs.max(0)
    label = class_names[idx]
    
    # Calculate is_diseased based on healthy classes in class names list
    healthy_indices = [i for i, name in enumerate(class_names) if "healthy" in name.lower()]
    
    if healthy_indices:
        prob_healthy = sum(probs[i].item() for i in healthy_indices)
        is_diseased = float(1.0 - prob_healthy) > threshold
    else:
        is_diseased = False
        
    result = {
        "predicted_class": label,
        "confidence": float(conf.item()),
        "is_diseased": bool(is_diseased),
        "probabilities": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
    }
    return result
