# Plant Disease Leaf Classifier: Edge Handoff Benchmark Report

This document compares **ResNet18** and **MobileNetV2** models to select the optimum network architecture for the Zelbytes platform rollout. The evaluation encompasses overall validation accuracy, parameter count, file size, CPU single-image inference latency, and GPU batch inference throughput.

## Model Performance & Throughput Comparison Table

| Architecture | Val Accuracy | File Size (MB) | Trainable Params | CPU Latency (BS=1) | CPU Latency (Quant, BS=1) | GPU Batch Latency (BS=32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 99.79% | 42.72 MB | 11,178,051 | 16.32 ms | 15.74 ms | 35.15 ms (1.098 ms/img) |
| **MobileNetV2** | 99.58% | 8.74 MB | 2,227,715 | 10.54 ms | 10.01 ms | 44.82 ms (1.401 ms/img) |
| **Tradeoff (Delta)** | **-0.21%** | **-33.98 MB** | **-8,950,336** | **-5.78 ms** | **-5.73 ms** | **--9.67 ms** |

> [!NOTE]
> CPU latencies are measured on CPU (Intel/AMD platform) as average of 100 runs.
> Quantized (INT8) profile is dynamically quantized applying dynamic 8-bit weights quantization on `nn.Linear` layers (`torch.quantization.quantize_dynamic`).
> GPU Batch Latencies are measured on NVIDIA GeForce RTX 3050 Laptop GPU.

---

## Deployment Recommendation Analysis

### Server vs. Edge Deployment Profile

*   **Edge deployment (Camera Swarms / Polyhouse Edge Nodes)**:
    We strongly recommend deploying the **MobileNetV2 Classifier** or its **Quantized MobileNetV2 variant** for power-constrained edge terminals. 
    1. **Size Reduction**: MobileNetV2 uses depthwise separable convolutions yielding a **4.9x reduction** in memory size (8.74 MB vs 42.72 MB). This represents a direct reduction in disk space and allows loading weights fully to SRAM.
    2. **Speed Enhancement**: MobileNetV2 delivers a **1.5x CPU speedup** under single image batch (BS=1) inference (10.54 ms vs 16.32 ms). Applying dynamic INT8 quantization further lowers compute costs.
    3. **Accuracy Tradeoff**: The validation accuracy drops by a mere **0.21%** (from 99.79% to 99.58%), which is a negligible cost compared to the immense latency and weight compression benefits.

*   **Server deployment (Zelbytes Cloud / Host Retraining)**:
    For centralized server-side environments where high throughput is critical and hardware acceleration is available (GPUs):
    1. **GPU Batch Throughput**: Under server workloads with large batch sizes (BS=32), **ResNet18** leverages standard convolutions which have higher arithmetic intensity. This allows the GPU tensor cores to maximize parallelism, yielding highly efficient batch times.
    2. **No Compromise**: Since server memory and power are not constraint factors, we recommend serving the **ResNet18** model to leverage the maximum classification boundary accuracy (99.79%) and perfect Apple recall performance.
