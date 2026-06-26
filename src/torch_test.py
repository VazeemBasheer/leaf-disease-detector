import torch
import torchvision

print(f"PyTorch {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Quick tensor smoke test
x = torch.randn(2, 3, 224, 224)  # batch of 2 RGB leaf patches
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print(f"Tensor device: {x.device}")
