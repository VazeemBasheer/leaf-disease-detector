# Day 18 – Dockerization Fundamentals for Machine Learning Deployments

## 1. Value of Docker in Agritech Edge Deployments

In agritech edge systems (such as the Zelbytes polyhouse gateways), running machine learning inference requires strict environment reproducibility:
- **PyTorch & Python Wheel Alignment**: Deep learning libraries depend heavily on compiled C bindings, system-level libraries (like OpenBlas, glibc), and specific Python version bindings. Even micro-version differences can result in slightly modified inference results, memory leaks, or runtime segmentation faults.
- **Immutable Shipments**: Packing the FastAPI code, dependencies, configurations, and weights inside an immutable container ensures that what is validated by developers will execute identically on the polyhouse edge node without dependencies drifting over time or conflicting with pre-existing libraries on the host gateway.
- **Isolation**: Keeping ML modules inside containers protects the gateway's core operating system, letting other gateway applications co-exist and run without Python version collisions.

---

## 2. Base Image Comparison: Slim vs. Full vs. Alpine

When choosing a base Docker image to bundle the container, image size and system compatibility are critical parameters:

| Image Identifier | Virtual Size (approx) | System Toolchain | Pros & Cons for ML Deployments |
| :--- | :--- | :--- | :--- |
| **`python:3.11` (Full)** | **~1.0 GB** | Full `gcc`/`g++` compiler, `git`, build utilities, headers, libc | **Pros**: Building custom C/CUDA operator extensions is automatic. <br>**Cons**: Unacceptably large footprint. Slows edge gateway cellular deployment. Increased security vulnerabilities. |
| **`python:3.11-slim`** | **~120 MB** | Stripped down. Contains only runtime Python interpreter and crucial utilities | **Pros**: Optimized lightweight deployment. Smaller download times, reduced attack surface. Standard standard target for CPU ML pipelines. <br>**Cons**: Native extension compilation fails unless build tools are installed via `apt-get` on demand. |
| **`python:3.11-alpine`** | **~50 MB** | Stripped down. Uses lightweight `musl-libc` instead of `glibc` | **Pros**: Smallest possible size. <br>**Cons**: Highly problematic for python machine learning. Libraries like `PyTorch` and `NumPy` publish binary wheels compiled for `glibc`. Building on Alpine necessitates hours of C code compilation, often triggering compilation exceptions. |

---

## 3. Architecture Diagram: Request Routing through Client -> Container -> Model

The diagram below reflects client interaction with the containerized FastAPI service on the Zelbytes polyhouse edge host:

```
                               +-------------------------------------------------+
                               | Zelbytes Polyhouse Edge Gateway Host            |
                               | (Docker Daemon Container Runtime)               |
                               |                                                 |
+--------------+               |   +-----------------------------------------+   |
| Client       |  GET /health  |   | FastAPI Container                       |   |
| (Edge Sensor | ------------> |   |                                         |   |
| or Frontend) |  POST /predict|   |   +---------------------------------+   |   |
+--------------+ <------------ |   |   | uvicorn HTTP Server (Port 8000) |   |   |
   (JSON /     |               |   |   +-----------------+---------------+   |   |
   Metadata    |               |   |                     |                   |   |
   Response)   |               |   |   +-----------------v---------------+   |   |
                               |   |   | PyTorch Inference Engine        |   |   |
                               |   |   +-----------------+---------------+   |   |
                               |   |                     |                   |   |
                               |   +---------------------|-------------------+   |
                               |                         |                       |
                               |   +---------------------v-------------------+   |
                               |   | Checkpoint Model Weights (Volume Mount) |   |
                               |   | /app/models/resnet18_leaf_best.pth      |   |
                               |   +-----------------------------------------+   |
                               +-------------------------------------------------+
```
To minimize image sizes in production, the raw weights file should ideally be mounted as a read-only Docker Volume instead of baked directly into the image layer, keeping image footprints under 200MB.

---

## 4. Minimal Dockerfile Blueprint

Our minimal `Dockerfile` is mapped below:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/resnet18_leaf_best.pth
ENV CLASS_NAMES_PATH=/app/models/class_names.json

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
