# leaf-disease-detector

leaf-disease-detector is a cnn enabled project which helps to identify plant diseases

\####PyTorch Environment \& GPU Setup -day1



\### Objective



The objective of Day 1 was to establish a reproducible development environment for the Plant Leaf Disease Detector project. This included configuring Python, creating an isolated virtual environment, installing the required deep learning libraries, verifying GPU support, and initializing the project repository with a standard machine learning folder structure.



\---



\## Tasks Completed



\### Python Environment



\* Installed \*\*Python 3.10+\*\*.

\* Verified the installation using:



&#x20; python --version



\### Virtual Environment



\* Created a dedicated virtual environment using:



&#x20; python -m venv venv



\* Activated the virtual environment successfully.



\### Package Installation



Installed the required dependencies:



\* PyTorch

\* torchvision

\* Pillow

\* matplotlib



Verified successful installation by importing the libraries in Python.



\### GPU Verification



Executed a PyTorch test script to verify hardware acceleration.



Verified:



\* PyTorch version

\* torchvision version

\* CUDA availability

\* GPU device name (when available)



The project is configured to automatically fall back to CPU execution if CUDA is unavailable.



\### Git Repository Initialization



Initialized a Git repository for version control.



Created a project `.gitignore` to exclude unnecessary files and directories, including:



\* `venv/`

\* `data/`

\* `\\\*.pth`

\* `\\\_\\\_pycache\\\_\\\_/`

\* `.vscode/`





\## Environment Information



| Component           | Status      |

| ------------------- | ----------- |

| Python 3.10+        | Installed   |

| Virtual Environment | Created     |

| PyTorch             | Installed   |

| torchvision         | Installed   |

| Pillow              | Installed   |

| matplotlib          | Installed   |

| Git Repository      | Initialized |

| CUDA Support        | Verified    |

| GPU/CPU Execution   | Working     |





\## Learning Outcomes



\* Learned how to create and manage isolated Python virtual environments.

\* Installed and configured PyTorch for deep learning development.

\* Verified CUDA/GPU support with automatic CPU fallback.

\* Initialized a Git repository using best practices.

\* Organized the project using a scalable machine learning directory structure.

\* Configured `.gitignore` to prevent large files, datasets, virtual environments, and model weights from being committed.



\## Next Steps



\* Load and inspect the plant leaf image dataset.

\* Perform image preprocessing using Pillow and torchvision.

\* Build the initial data loading pipeline.

\* Prepare the dataset for model training.
# Day 2 – PlantVillage Dataset Preparation and Data Loading Pipeline



\## Objective



Prepare a PlantVillage-style dataset for deep learning and implement a custom PyTorch data pipeline using `Dataset` and `DataLoader`.



\## Tasks Completed



\### 1. Dataset Organization



\* Cloned the PlantVillage dataset repository.

\* Created a custom dataset subset containing:



&#x20; \* Apple

&#x20; \* Pepper (Bell Pepper)

&#x20; \* Tomato

\* Organized the dataset into crop-wise folders while preserving disease-specific class folders.



Example directory structure:



```text

data/

└── processed/

&#x20;   ├── Apple/

&#x20;   │   ├── Apple\_\_\_Apple\_scab/

&#x20;   │   ├── Apple\_\_\_Black\_rot/

&#x20;   │   ├── Apple\_\_\_Cedar\_apple\_rust/

&#x20;   │   └── Apple\_\_\_healthy/

&#x20;   ├── Pepper/

&#x20;   │   ├── Pepper,\_bell\_\_\_Bacterial\_spot/

&#x20;   │   └── Pepper,\_bell\_\_\_healthy/

&#x20;   └── Tomato/

&#x20;       ├── Tomato\_\_\_Bacterial\_spot/

&#x20;       ├── Tomato\_\_\_Early\_blight/

&#x20;       ├── Tomato\_\_\_healthy/

&#x20;       ├── Tomato\_\_\_Late\_blight/

&#x20;       ├── Tomato\_\_\_Leaf\_Mold/

&#x20;       ├── Tomato\_\_\_Septoria\_leaf\_spot/

&#x20;       ├── Tomato\_\_\_Spider\_mites Two-spotted\_spider\_mite/

&#x20;       ├── Tomato\_\_\_Target\_Spot/

&#x20;       ├── Tomato\_\_\_Tomato\_mosaic\_virus/

&#x20;       └── Tomato\_\_\_Tomato\_Yellow\_Leaf\_Curl\_Virus/

```



\## Custom Dataset



Implemented a custom `LeafDiseaseDataset` class by inheriting from `torch.utils.data.Dataset`.



Features:



\* Automatically scans dataset folders.

\* Maps disease classes to numeric labels.

\* Loads RGB images using Pillow.

\* Applies preprocessing transforms.

\* Returns `(image\_tensor, label)` pairs.

\* Handles corrupted images by skipping and logging them.



\## Image Preprocessing



Applied the following preprocessing pipeline:



\* Resize images to \*\*224 × 224\*\*

\* Convert images to tensors

\* Normalize using ImageNet mean and standard deviation



\## DataLoader



Created both training and validation `DataLoader` objects with:



\* Batch size: \*\*32\*\*

\* Shuffle enabled for training

\* Validation loader without shuffling

\* `num\_workers = 0` (Windows compatible)

\* `pin\_memory = True`



\## Dataset Verification



Verified the data pipeline by:



\* Printing the total number of images

\* Counting images in each disease class

\* Splitting the dataset into training and validation sets

\* Confirming batch tensor shape:



```python

torch.Size(\[32, 3, 224, 224])

```



\## Visualization



Successfully visualized a batch of images after denormalizing the tensors using Matplotlib to verify:



\* Correct image loading

\* Proper normalization and denormalization

\* Correct label mapping

\* Expected tensor dimensions



\## Learning Outcomes



\* Understood the role of PyTorch `Dataset` and `DataLoader`.

\* Learned how image datasets are structured for deep learning.

\* Implemented an efficient image loading pipeline.

\* Verified data integrity before model training.

\* Gained experience with batching, shuffling, normalization, and visualization.



\## Next Steps



\* Build a Convolutional Neural Network (CNN) or use a pretrained model.

\* Implement the training and validation loops.

\* Evaluate model performance using accuracy and loss metrics.

\* Save the trained model for inference.


# Day 3 – CNN Architecture and Tensor Shape Analysis



\## Objective



Understand the fundamental building blocks of Convolutional Neural Networks (CNNs), analyze how tensor dimensions change through each layer, and design a simple CNN architecture for four-class plant leaf disease classification.



\---



\## Tasks Completed



\### CNN Fundamentals



\- Studied how Conv2D filters detect local image features.

\- Understood how early convolution layers detect:

&#x20; - Leaf edges

&#x20; - Veins

&#x20; - Leaf margins

\- Learned how deeper layers identify disease-specific features such as:

&#x20; - Lesions

&#x20; - Brown spots

&#x20; - Mold patches

&#x20; - Leaf discoloration



\### CNN Architecture Design



Designed a simple CNN consisting of:



\- Conv2D (3 → 32)

\- ReLU

\- MaxPool2D

\- Conv2D (32 → 64)

\- ReLU

\- MaxPool2D

\- Conv2D (64 → 128)

\- ReLU

\- AdaptiveAvgPool2D

\- Flatten

\- Fully Connected Layer (128 → 4)



\### Tensor Shape Analysis



Implemented a shape-tracing script to print tensor dimensions after each layer.



Verified the following tensor shapes:



| Layer | Output Shape |

|--------|--------------|

| Input | (1, 3, 224, 224) |

| Conv2D | (1, 32, 224, 224) |

| MaxPool2D | (1, 32, 112, 112) |

| Conv2D | (1, 64, 112, 112) |

| MaxPool2D | (1, 64, 56, 56) |

| Conv2D | (1, 128, 56, 56) |

| AdaptiveAvgPool2D | (1, 128, 1, 1) |

| Flatten | (1, 128) |

| Linear | (1, 4) |



\### Manual Shape Verification



Calculated output dimensions manually using the convolution formula:



\\\[

Output = \\frac{W - K + 2P}{S} + 1

\\]



Verified that all manually calculated tensor dimensions matched the output generated by the PyTorch implementation.



\### Adaptive Average Pooling



Learned why `AdaptiveAvgPool2D` is preferred over directly flattening large feature maps:



\- Reduces the number of trainable parameters.

\- Decreases memory usage.

\- Helps prevent overfitting.

\- Supports variable input image sizes.

\- Simplifies the classifier.



\### Documentation



Created:



\- `docs/cnn\_architecture.md`

\- CNN architecture diagram

\- Tensor shape trace

\- Manual shape calculations

\- CNN terminology glossary



\---



\## Learning Outcomes



\- Understood how convolution layers extract hierarchical image features.

\- Learned how padding, stride, and kernel size affect output dimensions.

\- Performed manual tensor shape calculations for CNN layers.

\- Verified tensor shapes using a PyTorch shape-tracing script.

\- Understood the purpose and advantages of Adaptive Average Pooling.

\- Designed a simple CNN architecture for four-class plant disease classification.



\---



\## Next Steps



\- Implement the complete CNN model using `torch.nn.Module`.

\- Build the forward propagation function.

\- Define the loss function and optimizer.

\- Train the CNN using the prepared PlantVillage dataset.

\- Evaluate model performance on the validation dataset.

