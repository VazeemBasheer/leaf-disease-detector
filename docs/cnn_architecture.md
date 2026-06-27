\# CNN Architecture and Shape Analysis



\## Objective



The objective of this task is to understand how Convolutional Neural Networks (CNNs) process images by tracing tensor dimensions through each layer, calculating output shapes manually, and designing a simple CNN architecture for plant leaf disease classification.



\---



\# 1. How Conv2D Filters Detect Local Patterns



A Convolutional Neural Network uses \*\*Conv2D filters (kernels)\*\* to detect small local patterns in an image. Each filter slides across the image and extracts useful features.



For plant leaf disease detection:



\* \*\*Early convolution layers\*\* learn simple features such as:



&#x20; \* Leaf edges

&#x20; \* Leaf margins

&#x20; \* Veins

&#x20; \* Texture



\* \*\*Middle convolution layers\*\* detect more complex patterns such as:



&#x20; \* Brown spots

&#x20; \* Yellow lesions

&#x20; \* Mold patches

&#x20; \* Leaf discoloration



\* \*\*Deeper convolution layers\*\* combine these features to recognize specific diseases and distinguish healthy leaves from infected ones.



\---



\# 2. CNN Architecture



```text

Input Image (224 × 224 × 3)

&#x20;           │

&#x20;           ▼

Conv2D (3 → 32, 3×3, Padding=1)

&#x20;           │

&#x20;          ReLU

&#x20;           │

&#x20;      MaxPool2D (2×2)

&#x20;           │

&#x20;           ▼

Conv2D (32 → 64, 3×3, Padding=1)

&#x20;           │

&#x20;          ReLU

&#x20;           │

&#x20;      MaxPool2D (2×2)

&#x20;           │

&#x20;           ▼

Conv2D (64 → 128, 3×3, Padding=1)

&#x20;           │

&#x20;          ReLU

&#x20;           │

AdaptiveAvgPool2D (1×1)

&#x20;           │

&#x20;        Flatten

&#x20;           │

Linear Layer (128 → 4)

&#x20;           │

&#x20;           ▼

Output Classes

```



\---



\# 3. Layer-by-Layer Shape Trace



| Layer             | Output Shape      |

| ----------------- | ----------------- |

| Input             | (1, 3, 224, 224)  |

| Conv2D (3→32)     | (1, 32, 224, 224) |

| ReLU              | (1, 32, 224, 224) |

| MaxPool2D         | (1, 32, 112, 112) |

| Conv2D (32→64)    | (1, 64, 112, 112) |

| ReLU              | (1, 64, 112, 112) |

| MaxPool2D         | (1, 64, 56, 56)   |

| Conv2D (64→128)   | (1, 128, 56, 56)  |

| ReLU              | (1, 128, 56, 56)  |

| AdaptiveAvgPool2D | (1, 128, 1, 1)    |

| Flatten           | (1, 128)          |

| Linear (128→4)    | (1, 4)            |



The tensor shapes printed by the PyTorch script match the manually calculated dimensions.



\---



\# 4. Manual Shape Calculations



The output size of a convolution layer is calculated using:



\[

\\text{Output Size} = \\frac{W - K + 2P}{S} + 1

]



Where:



\* \*\*W\*\* = Input width/height

\* \*\*K\*\* = Kernel size

\* \*\*P\*\* = Padding

\* \*\*S\*\* = Stride



\### Conv1



Input:



```

224 × 224

```



Parameters:



\* Kernel = 3

\* Padding = 1

\* Stride = 1



Calculation:



```

(224 − 3 + 2×1)/1 + 1 = 224

```



Output:



```

(1, 32, 224, 224)

```



\---



\### Pool1



Pooling layer:



```

Kernel = 2

Stride = 2

```



Calculation:



```

224 / 2 = 112

```



Output:



```

(1, 32, 112, 112)

```



\---



\### Conv2



```

112 → 112

```



Output:



```

(1, 64, 112, 112)

```



\---



\### Pool2



```

112 / 2 = 56

```



Output:



```

(1, 64, 56, 56)

```



\---



\### Conv3



```

56 → 56

```



Output:



```

(1, 128, 56, 56)

```



\---



\### Adaptive Average Pooling



```

56 × 56

&#x20;     ↓

1 × 1

```



Output:



```

(1, 128, 1, 1)

```



\---



\### Flatten



```

128 × 1 × 1

&#x20;       ↓

128

```



Output:



```

(1, 128)

```



\---



\### Fully Connected Layer



```

128 → 4

```



Output:



```

(1, 4)

```



\---



\# 5. Why AdaptiveAvgPool2D Instead of Flatten?



Using a normal Flatten layer directly after the convolution layers would produce a very large feature vector.



Example:



```

56 × 56 × 128

=

401,408 features

```



A fully connected layer with over 400,000 input features would require a large number of trainable parameters, increasing memory usage and the risk of overfitting.



`AdaptiveAvgPool2D((1,1))` reduces every feature map to a single value, producing:



```

128 × 1 × 1

```



which becomes:



```

128 features

```



\### Advantages



\* Reduces the number of trainable parameters.

\* Lowers memory consumption.

\* Helps prevent overfitting.

\* Allows the network to accept different input image sizes.

\* Simplifies the classifier.



\---



\# 6. Shape Verification



| Layer             | Manual Calculation | PyTorch Output    

| ----------------- | ------------------ | ----------------- 

| Conv1             | (1, 32, 224, 224)  | (1, 32, 224, 224) 

| Pool1             | (1, 32, 112, 112)  | (1, 32, 112, 112) 

| Conv2             | (1, 64, 112, 112)  | (1, 64, 112, 112) 

| Pool2             | (1, 64, 56, 56)    | (1, 64, 56, 56)   

| Conv3             | (1, 128, 56, 56)   | (1, 128, 56, 56)  

| AdaptiveAvgPool2D | (1, 128, 1, 1)     | (1, 128, 1, 1)    

| Flatten           | (1, 128)           | (1, 128)          

| Linear            | (1, 4)             | (1, 4)           



The manually calculated tensor dimensions exactly match the output produced by the PyTorch implementation.



\---



\# 7. Glossary



| Term                | Definition                                                                                      |

| ------------------- | ----------------------------------------------------------------------------------------------- |

| \*\*Kernel (Filter)\*\* | A small matrix that scans an image to detect local features such as edges, spots, and textures. |

| \*\*Stride\*\*          | The number of pixels the filter moves during convolution.                                       |

| \*\*Padding\*\*         | Extra pixels added around the image border to preserve spatial dimensions after convolution.    |

| \*\*Feature Map\*\*     | The output generated by a convolution layer after applying filters to the input image.          |



\---



\# Conclusion



A simple CNN architecture for four-class plant leaf disease classification was designed and analyzed. Manual output shape calculations were verified against the PyTorch implementation, confirming correct tensor dimensions throughout the network. The use of `AdaptiveAvgPool2D` significantly reduces model complexity while maintaining important learned features before classification.



