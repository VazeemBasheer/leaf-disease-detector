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

# Day 4 – CNN Model Implementation and Forward Pass Verification



\## Objective



Implement a custom Convolutional Neural Network (CNN) for plant leaf disease classification, verify its architecture through forward passes, and analyze model parameters before training.



\---



\## Tasks Completed



\### CNN Model Development



Implemented a custom `LeafDiseaseCNN` by inheriting from `torch.nn.Module`.



The network consists of:



\* Conv2D (3 → 32)

\* ReLU

\* MaxPool2D

\* Conv2D (32 → 64)

\* ReLU

\* MaxPool2D

\* Conv2D (64 → 128)

\* ReLU

\* MaxPool2D

\* Conv2D (128 → 256)

\* ReLU

\* Adaptive Average Pooling

\* Flatten

\* Dropout

\* Fully Connected Classification Layer



\---



\## Dataset Verification



Successfully loaded the prepared PlantVillage dataset.



\### Dataset Statistics



| Metric             |  Value |

| ------------------ | -----: |

| Total Images       | 47,610 |

| Total Classes      |     16 |

| Training Batches   |  1,191 |

| Validation Batches |    298 |

| Batch Size         |     32 |



The dataset includes Apple, Pepper, and Tomato leaf images across \*\*16 disease and healthy classes\*\*.



\---



\## Forward Pass Verification



Performed a forward pass using a real batch obtained from the `DataLoader`.



\### Input Tensor



```text

torch.Size(\[32, 3, 224, 224])

```



\### Model Output



```text

torch.Size(\[32, 4])

```



The model successfully processed an entire batch without runtime errors, confirming that the network architecture and forward propagation are functioning correctly.



\---



\## Output Shape Validation



Verified that the model produces output tensors with the expected dimensions:



```python

assert outputs.shape\[0] == images.shape\[0]

assert outputs.shape\[1] == NUM\_CLASSES

```



The forward-pass verification completed successfully.



\---



\## Model Architecture



The implemented CNN consists of four convolutional blocks followed by a lightweight classifier head.



Key architectural features include:



\* Four convolution layers for hierarchical feature extraction.

\* ReLU activation after every convolution.

\* Max pooling for spatial downsampling.

\* Adaptive Average Pooling to reduce feature maps to a fixed size.

\* Dropout for regularization.

\* Fully connected output layer for disease classification.



The architecture was exported to:



```text

models/architecture.txt

```



\---



\## Learning Outcomes



\* Implemented a custom CNN using `torch.nn.Module`.

\* Understood the role of the `forward()` function in PyTorch.

\* Verified tensor flow through every layer using a forward pass.

\* Successfully connected the custom dataset and DataLoader to the CNN.

\* Validated batch input and output dimensions.

\* Learned the importance of matching the classifier output dimension with the total number of dataset classes.



\---



\## Next Steps



\* Update the classifier output to match all dataset classes (\*\*16 classes\*\*).

\* Define the loss function (`CrossEntropyLoss`) and optimizer (`Adam`).

\* Implement the training and validation loops.

\* Monitor training and validation accuracy across epochs.

\* Save the best-performing model for inference.


# Day 5 – CNN Training Loop and Model Checkpointing



\## Objective



Implement the complete training pipeline for the custom CNN using PyTorch. Train the model on the PlantVillage dataset using `CrossEntropyLoss` and the Adam optimizer, verify successful training, and save the trained model checkpoint.



\---



\## Tasks Completed



\### Model Training Setup



\- Configured reproducible training using `torch.manual\_seed(42)`.

\- Enabled automatic device selection (CPU/GPU).

\- Verified CUDA availability and GPU information (when available).

\- Moved both the model and training batches to the selected device.



\### Loss Function and Optimizer



Configured the training components:



\- \*\*Loss Function:\*\* `CrossEntropyLoss`

\- \*\*Optimizer:\*\* `Adam`

\- \*\*Learning Rate:\*\* `1e-3`



These are suitable defaults for multi-class image classification.



\### Training Loop



Implemented a reusable `train\_one\_epoch()` function that performs:



\- Forward propagation

\- Loss computation

\- Backpropagation

\- Gradient update using Adam

\- Running loss calculation

\- Progress visualization using `tqdm`



The training loop was executed for \*\*5 epochs\*\*.



\### Device Management



The training pipeline automatically supports both CPU and GPU execution.



Features include:



\- Automatic CUDA detection

\- Model transferred to GPU when available

\- Images and labels transferred to the same device

\- GPU memory usage logging after each epoch



\### Training Stability



Added safeguards to improve training reliability:



\- Fixed random seed for reproducibility

\- NaN loss detection

\- Running loss monitoring throughout training



\### Model Checkpoint



Successfully saved the trained model after training.



Checkpoint location:



```text

models/checkpoints/leaf\_cnn\_epoch5.pth

```



Verified the checkpoint by loading it using:



```python

load\_state\_dict()

```



ensuring the saved model can be restored for future inference or continued training.



\---



\## Deliverables



Completed the following project files:



\- `src/train.py`

\- models/checkpoints/leaf\_cnn\_epoch5.pth

\- Training loss logs using `tqdm`

\- Reloadable model checkpoint



\---



\## Learning Outcomes



\- Understood the complete PyTorch training workflow.

\- Implemented forward and backward propagation.

\- Learned how `CrossEntropyLoss` is used for multi-class classification.

\- Optimized model parameters using the Adam optimizer.

\- Managed device placement for CPU and GPU training.

\- Logged training progress using `tqdm`.

\- Saved and reloaded model checkpoints for reproducible experiments.

\- Applied reproducibility techniques using random seeds.



\---



\## Next Steps



\- Implement a validation loop.

\- Track validation loss and classification accuracy.

\- Plot training and validation loss curves.

\- Save the best-performing model based on validation performance.

\- Evaluate the model using confusion matrix and per-class metrics.

# Day 6 – Model Training, Validation & Early Stopping

## Objective

Implement a complete training pipeline for the Plant Leaf Disease Classification CNN using PyTorch. The model is trained on the processed dataset, validated after every epoch, and optimized using early stopping to prevent overfitting.

---

## Learning Outcomes

By the end of Day 6, the following concepts were implemented and understood:

* Training a CNN using PyTorch
* Forward propagation
* Backpropagation
* CrossEntropyLoss for multi-class classification
* Adam optimizer
* GPU (CUDA) training
* Validation loop
* Accuracy calculation
* Early stopping
* Best model checkpointing
* Resume training from checkpoint
* Training log generation
* Loss curve visualization

---

## Model

**Architecture**

* Conv2D (3 → 32)
* ReLU
* MaxPool
* Conv2D (32 → 64)
* ReLU
* MaxPool
* Conv2D (64 → 128)
* ReLU
* MaxPool
* Conv2D (128 → 256)
* ReLU
* Adaptive Average Pooling
* Dropout (0.3)
* Fully Connected Layer

Output:

```text
Batch × 16 Classes
```

---

## Dataset

* Total Images: **47,610**
* Number of Classes: **16**
* Image Size: **224 × 224**
* Batch Size: **32**

Dataset Split

* Training: 80%
* Validation: 20%

---

## Training Configuration

| Parameter     |                       Value |
| ------------- | --------------------------: |
| Optimizer     |                        Adam |
| Learning Rate |                       0.001 |
| Loss Function |            CrossEntropyLoss |
| Epochs        | 50 (Early Stopping Enabled) |
| Batch Size    |                          32 |
| Device        |                  CUDA / CPU |
| Random Seed   |                          42 |

---

## Features Implemented

### Training Loop

* Model switched to training mode using `model.train()`
* Images transferred to GPU
* Forward propagation
* Loss computation
* Backpropagation
* Optimizer update
* Running loss calculation

---

### Validation Loop

* Model switched to evaluation mode using `model.eval()`
* Gradient computation disabled
* Validation loss calculated
* Validation accuracy calculated

---

### Early Stopping

* Patience = 3
* Stops training if validation loss does not improve
* Prevents overfitting
* Restores best model weights

---

### Checkpointing

Implemented:

* Best model checkpoint
* Latest checkpoint
* Resume interrupted training automatically

Checkpoint Files

```text
models/checkpoints/
├── leaf_cnn_best.pth
└── latest_checkpoint.pth
```

---

### GPU Support

The model automatically detects CUDA.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

GPU memory usage is displayed after every epoch.

---

### Training Visualization

Training automatically generates:

```text
reports/
├── training_curves.png
└── training_log.txt
```

The graph compares:

* Training Loss
* Validation Loss

---

## Outputs

Generated Files

```text
models/checkpoints/
├── leaf_cnn_best.pth
├── latest_checkpoint.pth

reports/
├── training_curves.png
├── training_log.txt
```

---

## Results

* Validation performed every epoch
* Early stopping successfully restored the best weights
* Best model automatically saved
* Training can resume from interruptions without restarting

---

## Skills Learned

* CNN training pipeline
* Loss optimization
* Validation techniques
* Accuracy measurement
* GPU acceleration
* Model checkpointing
* Early stopping
* Experiment logging
* Resume training

# Day 7 – Data Augmentation Pipeline

## Objective

Improve the model's ability to generalize by applying data augmentation only to the training dataset while keeping the validation dataset unchanged. This helps reduce overfitting and improves robustness to variations in real-world images.

---

## Learning Outcomes

Implemented:

* Separate transform pipelines
* Data augmentation
* Random image transformations
* Validation without augmentation
* Modular preprocessing
* Augmentation visualization
* Reproducible transform configuration

---

## Project Structure

```text
src/
├── transforms.py
├── DataLoader.py
├── model_generator.py
└── train.py
```

---

## Train Transform Pipeline

Training images undergo the following preprocessing:

1. Resize to 256 × 256
2. RandomResizedCrop (224 × 224)
3. RandomHorizontalFlip (50%)
4. RandomRotation (±15°)
5. ColorJitter

   * Brightness
   * Contrast
   * Saturation
6. Convert to Tensor
7. Normalize using ImageNet statistics

---

## Validation Transform Pipeline

Validation images use deterministic preprocessing:

1. Resize to 224 × 224
2. Convert to Tensor
3. Normalize

No augmentation is applied to the validation dataset to ensure fair and stable evaluation.

---

## Why Separate Transforms?

Training augmentation allows the model to learn from varied versions of the same image.

Validation data must remain unchanged so that model performance can be compared consistently across experiments.

---

## Label-Safe Transformations

The following augmentations preserve the disease label:

* Random Horizontal Flip
* Random Crop
* Small Rotation (±15°)
* Brightness Adjustment
* Contrast Adjustment
* Saturation Adjustment

These operations change only the image appearance and not the underlying disease category.

---

## Transform Configuration

All transforms are defined in:

```text
src/transforms.py
```

This makes preprocessing reusable across:

* DataLoader
* Training
* Validation
* Future inference scripts

---

## Dataset Pipeline

```text
Leaf Images
      │
      ▼
Train Transform
      │
      ▼
Training DataLoader
      │
      ▼
CNN Model
```

Validation follows a separate preprocessing pipeline without augmentation.

---

## Visualization

Generated Files

```text
reports/
├── augment_samples.png
```

The visualization demonstrates how augmentation creates multiple variations of leaf images while preserving their labels.

---

## Training

The CNN was retrained using the augmented training dataset.

The training pipeline continued to support:

* GPU acceleration
* Resume from checkpoint
* Early stopping
* Validation after every epoch

---

## Benefits of Data Augmentation

* Reduces overfitting
* Improves generalization
* Simulates real-world image variations
* Makes the model more robust
* Increases dataset diversity without collecting new images

---

## Deliverables

```text
src/
├── transforms.py

reports/
├── augment_samples.png

configs/
├── transforms.yaml
```

---

## Skills Learned

* Image preprocessing
* Data augmentation
* PyTorch transforms
* Modular project design
* Dataset preprocessing
* Label-safe augmentation
* CNN generalization techniques
* Reproducible preprocessing pipelines

---

## Conclusion

Day 7 introduced a reusable and modular data augmentation pipeline. Training images are augmented dynamically during loading, while validation images remain unchanged for reliable evaluation. This approach improves the model's robustness without altering the original dataset and follows best practices used in modern deep learning workflows.

---

# Day 8 – Class Balancing and Performance Comparison

## Objective

Configure a balanced training pipeline to mitigate class imbalance issues in the leaf disease dataset and compare training results against the unbalanced baseline using per-class classification reports and confusion matrices.

---

## Tasks Completed

- **Unified CLI pipeline**: Developed a integrated training switch configuration supporting both standard baseline execution and class balancing sampler controls.
- **Weighted Sampler & Loss**: Integrated `WeightedRandomSampler` and class-weighted `CrossEntropyLoss` to boost validation focus on minority samples.
- **Report Validation**: Evaluated and exported comparative recall statistics:
  - **Minority Recall Lift**: Apple crop recall rose from `0.9796 -> 0.9843`; Pepper recall increased from `0.9817 -> 0.9898`.
  - Exported validation materials: `reports/recall_day8.csv`, `reports/classification_report_day8.txt`, and `reports/confusion_matrix_day8.csv`.

---

# Day 9 – Transfer Learning & ResNet18 Backbone Initialization

## Objective

Initialize a transfer learning setup using a pre-trained ResNet18 backbone, replace the standard ImageNet classification head, inspect module layers, freeze network weights, and run proof-of-concept inference on sample leaf data.

---

## Tasks Completed

- **Modern Backbone Integration**: Configured model instantiation utilizing `models.ResNet18_Weights.IMAGENET1K_V1` to load pre-trained ImageNet parameters.
- **UserWarning Inspection**: Coded deprecation warnings check within `src/resnet_demo.py` capturing obsolete `pretrained=True` user notices.
- **Pretrained Inference Validation**: Ran a forward pass on a normalized target leaf image, obtaining ImageNet prediction category `'bonnet'` (prob: 0.1504), verifying backpropagation-ready input pipelines.
- **Target Classifier Substitution**: Replaced the fully connected layer (`fc`) with a new trainable 4-class classifier.
- **Backbone Freezing**: Deactivated gradients for all convolutional blocks. Tracked trainable parameters decrease from `11,689,512` to `2,052` parameters (99.98% reduction).
- **Layer Mapping & Concept Plan**: Logged 67 architecture layer signatures to `docs/resnet18_layers.txt` and prepared `docs/transfer_learning.md` outlining the Day 10 unfreezing schedule (fine-tuning `layer4`).

---

# Day 10 – Two-Phase ResNet18 Fine-Tuning & Evaluation

## Objective

Fine-tune the pre-trained ResNet18 model using a two-phase transfer learning schedule, save the best weights with metadata, compare the recall and overall accuracy achievements against the custom scratch CNN baseline, and document hardware performance benchmarks for Zelbytes deployment planning.

---

## Tasks Completed

- **Two-Phase Training Script**: Implemented `src/train_resnet.py` featuring a strict phased fine-tuning loop:
  - **Phase 1 (Epochs 1-3)**: Backbone parameters frozen. Trained only the newly replaced 3-class classification head (`lr = 1e-3`).
  - **Phase 2 (Epochs 4-6)**: Unfroze the `layer4` residual block. Trained with differential learning rates: `1e-5` for the `layer4` backbone group and `1e-3` for the classification head.
  - **Auto Mixed Precision (AMP)**: Handled operations in FP16 to maximize RTX GPU hardware performance.
- **Performance Evaluation**:
  - **Overall Accuracy Elevation**: Validation accuracy rose from the Scratch CNN baseline of **`98.80%`** (Day 8) to **`99.79%`** (Day 10) — a net improvement of **`+0.99%`**.
  - **Crop-level Recall Improvements**:
    - **Apple**: `0.9843 -> 1.0000` (`+0.0157`)
    - **Pepper**: `0.9898 -> 0.9959` (`+0.0061`)
    - **Tomato**: `0.9884 -> 0.9978` (`+0.0094`)
- **Zelbytes GPU vs CPU Benchmarks**:
  - **GPU Execution Time**: Total training finished in `1745.21 seconds` on an NVIDIA RTX 3050 GPU using AMP (~4.8 minutes/epoch with sequential CPU image decoding).
  - **CPU Projection**: Training without CUDA acceleration on standard CPUs projection is ~50 minutes/epoch, exceeding 5 hours for 6 epochs.
- **Deliverables Saved**:
  - Checkpoint: `models/checkpoints/resnet18_leaf_best.pth` (includes state dict, best epoch, final accuracy, and class names).
  - Class index mapping: `models/class_names.json`.
  - Comparative log: `reports/resnet18_vs_scratch_cnn.txt` and `reports/classification_report_resnet.txt`.



