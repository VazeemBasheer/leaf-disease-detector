#!/usr/bin/env python3
import os
import sys
import json
import argparse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import (
    load_model,
    get_class_names,
    get_operating_threshold,
    predict_image
)

def main():
    parser = argparse.ArgumentParser(description="Leaf disease classification and inference CLI")
    parser.add_argument("--image", required=True, help="Path to input image file.")
    parser.add_argument("--model", default="models/resnet18_leaf_best.pth", 
                        help="Path to trained model .pth checkpoint.")
    args = parser.parse_args()

    image_path = args.image

    # 1. Exit code non-zero (1) on missing file
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Load model and verify architecture
    try:
        model, num_classes, checkpoint = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Error: Failed to load model weights. Details: {e}", file=sys.stderr)
        sys.exit(4)

    class_names = get_class_names(checkpoint, num_classes)
    threshold = get_operating_threshold(num_classes)

    # 3. Exit code non-zero (2) on corrupt or invalid image
    try:
        img = Image.open(image_path).convert("RGB")
        result = predict_image(img, model, class_names, threshold)
    except Exception as e:
        print(f"Error: Failed to process or load image (corrupt or invalid format). Details: {e}", file=sys.stderr)
        sys.exit(2)

    # Print outputs in formatted JSON to standard output
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
