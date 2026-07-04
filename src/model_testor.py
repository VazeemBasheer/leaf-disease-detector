import torch

from model_generator import LeafDiseaseCNN, NUM_CLASSES

from DataLoader import train_loader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_generator = LeafDiseaseCNN().to(device)

model_generator.eval()

images, labels = next(iter(train_loader))

images = images.to(device)

with torch.no_grad():

    outputs = model_generator(images)

print("Input Shape :", images.shape)

print("Output Shape:", outputs.shape)

assert outputs.shape[0] == images.shape[0]

assert outputs.shape[1] == NUM_CLASSES

print("Real batch forward pass successful.")