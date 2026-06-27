import torch
import torch.nn as nn

# --------------------------------------------------
# Shape tracing function
# --------------------------------------------------

def trace_shapes(x, layers):
    print(f"{'Layer':<25} {'Output Shape'}")
    print("-" * 50)

    for name, layer in layers:
        x = layer(x)
        print(f"{name:<25} {tuple(x.shape)}")

    return x


# --------------------------------------------------
# Dummy input image
# --------------------------------------------------

x = torch.randn(1, 3, 224, 224)

print("Input Shape")
print(tuple(x.shape))
print()

# --------------------------------------------------
# CNN Layers
# --------------------------------------------------

layers = [

    ("Conv2d (3→32)", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)),
    ("ReLU", nn.ReLU()),
    ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2)),

    ("Conv2d (32→64)", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
    ("ReLU", nn.ReLU()),
    ("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2)),

    ("Conv2d (64→128)", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
    ("ReLU", nn.ReLU()),

    ("AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d((1, 1))),

    ("Flatten", nn.Flatten()),

    ("Linear (128→4)", nn.Linear(128, 4))
]

# --------------------------------------------------
# Trace tensor shapes
# --------------------------------------------------

trace_shapes(x, layers)

# --------------------------------------------------
# Hand Calculated Shapes
# --------------------------------------------------

print("\nExpected Shape Calculations")
print("-" * 50)

print("Input                 : (1, 3, 224, 224)")
print("Conv1 (3x3,p=1,s=1)   : (1, 32, 224, 224)")
print("Pool1 (2x2,s=2)       : (1, 32, 112, 112)")
print("Conv2 (3x3,p=1,s=1)   : (1, 64, 112, 112)")
print("Pool2 (2x2,s=2)       : (1, 64, 56, 56)")
print("Conv3 (3x3,p=1,s=1)   : (1, 128, 56, 56)")
print("AdaptiveAvgPool       : (1, 128, 1, 1)")
print("Flatten              : (1, 128)")
print("Linear               : (1, 4)")