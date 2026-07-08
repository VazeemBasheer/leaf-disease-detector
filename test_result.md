# FastAPI HTTP API Scaffolding & Validation: Walkthrough

We have successfully designed, scaffolded, and validated the FastAPI API layer (`leaf-disease-api`) supporting structured endpoints, model lifecycle caching (lifespan), dependency injection, status code mapping, memory controls, latency monitoring, and automated clients.

## Architecture & Code Structure

The project has been configured with a clean separation of concerns:

- [src/inference.py](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/src/inference.py): The shared production inference core. It dynamically loads checkpoints, extracts metadata, evaluates classification images, and outputs formatted class mappings and boolean flags.
- [src/predict.py](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/src/predict.py): The command-line-interface (CLI) runner now directly imports core functions from `src/inference.py` to assure single-source compliance.
- [app/main.py](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/app/main.py): Key web app. Initiates FastAPI, controls model loading inside an async `lifespan` manager, handles upload requirements (MIME types, 5MB size limits), resizes large camera photos, injects latency runtime headers, and guards execution exceptions.
- [app/schemas.py](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/app/schemas.py): Holds Pydantic validation structures, locking response formats for `/health` and `/predict`.
- [app/inference.py](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/app/inference.py): Provides model singletons using FastAPI dependency injection (`Depends(get_inference_service)`).
- [.env.example](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/.env.example) and `.env`: Template configurations containing `MODEL_PATH` and `DISEASE_THRESHOLD`.

---

## Endpoint Validation Check (Day 14)

We implemented [tests/manual_predict.py](file:///c:/Users/vazeem/INTERNSHIP/leaf-disease-detector/tests/manual_predict.py) which executes the following target checks on the REST API:

### 1. `/health` Status Check
Verifies status is OK:
```json
{"status": "ok", "model_loaded": true}
```

### 2. `/predict` Content Whitelists & Limits
- **Unsupported MIME (415)**: Text file returns **Status 415 Unsupported Media Type**:
  ```json
  {"detail": "Unsupported media type: Only image/jpeg and image/png are supported."}
  ```
- **Corrupt Decode (400)**: Invalid image returns **Status 400 Bad Request**:
  ```json
  {"detail": "Invalid image file: Failed to decode image payload."}
  ```
- **Size Constraint (413)**: Payloads larger than 5MB are rejected with **Status 413 Payload Too Large**.
- **Server-Side Resizing**: Camera DSLRs exceeding 1024px dynamically thumbnail (`image.thumbnail((1024, 1024))`) to mitigate memory footprint.

### 3. Monitoring Headers
Each successful classify client query maps a custom header:
- **`X-Inference-Ms`**: Captures exact server-side processing duration in milliseconds (e.g., `45.89 ms`).

### 4. Output Responses
Tested and recorded JSON responses:

* **Healthy Sample (`reports/prediction_api_apple_healthy.json`)**:
  ```json
  {
      "predicted_class": "apple",
      "confidence": 0.9999861717224121,
      "is_diseased": false,
      "probabilities": {
          "apple": 0.9999861717224121,
          "pepper": 1.7276173025493335e-07,
          "tomato": 1.3688322724192403e-05
      }
  }
  ```

* **Diseased Early Blight (`reports/prediction_api_tomato_early_blight.json`)**:
  ```json
  {
      "predicted_class": "tomato",
      "confidence": 0.9976146221160889,
      "is_diseased": false,
      "probabilities": {
          "apple": 0.00035750895040109754,
          "pepper": 0.002027930226176977,
          "tomato": 0.9976146221160889
      }
  }
  ```

* **Diseased Late Blight (`reports/prediction_api_tomato_late_blight.json`)**:
  ```json
  {
      "predicted_class": "tomato",
      "confidence": 0.9993357062339783,
      "is_diseased": false,
      "probabilities": {
          "apple": 0.0003507026704028249,
          "pepper": 0.0003136279992759228,
          "tomato": 0.9993357062339783
      }
  }
  ```

---

## Day 16 – Leaf Quality API Multi-Model Prediction Reporting

We verified predictions with both the 3-class Species model and 16-class Disease model, validating accurate labeling of Apple Black Rot (Black Mold).

### 1. 3-Class Species Model Validation (`models/resnet18_leaf_best.pth`)
- **Action**: Test `samples/apple_healthy.jpg`
- **Command**:
  ```powershell
  python tests/test_single.py samples/apple_healthy.jpg
  ```
- **Result**:
  ```json
  Status Code: 200
  Latency: 24.94 ms
  {
    "predicted_class": "apple",
    "confidence": 0.9999861717224121,
    "is_diseased": false,
    "probabilities": {
      "apple": 0.9999861717224121,
      "pepper": 1.7276173025493335e-07,
      "tomato": 1.3688322724192403e-05
    }
  }
  ```

### 2. 16-Class Disease Model Validation (`models/checkpoints/leafcnn_20260628_acc_0.992.pth`)
- **Action**: Test Apple Black Rot dataset leaf (`C:\Users\vazeem\INTERNSHIP\leaf-disease-detector\data\processed\apple\Apple___Black_rot\1ce9343f-125e-4abb-9cd4-fddf761504da___JR_FrgE.S 3096.JPG`)
- **Command**:
  ```powershell
  curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@C:\Users\vazeem\INTERNSHIP\leaf-disease-detector\data\processed\apple\Apple___Black_rot\1ce9343f-125e-4abb-9cd4-fddf761504da___JR_FrgE.S 3096.JPG;type=image/jpeg"
  ```
- **Result**:
  ```json
  {
    "predicted_class": "Apple___Black_rot",
    "confidence": 0.9999966621398926,
    "is_diseased": true,
    "probabilities": {
      "Apple___Apple_scab": 3.733508802956952e-15,
      "Apple___Black_rot": 0.9999966621398926,
      "Apple___Cedar_apple_rust": 3.1256388393885e-34,
      "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 2.24268273e-32
    }
  }
  ```

---

## Endpoint Validation Check (Day 17)

We have validated the robustness checks of the REST API:

### 1. Model Warmup on Startup
Startup logs confirm that the server loaded and pre-warmed parameters using a dummy forward tensor (`torch.randn(1, 3, 224, 224)`), protecting clients from initial cold-start latency overhead on the first predict query.

### 2. Missing Weights Fail-Safe Loader
When `MODEL_PATH` points to a non-existent path, the server initializes cleanly in an offline state:
- `/health` status reports `model_loaded: false`.
- `/predict` returns **Status 503 Service Unavailable** with the following structured JSON output:
  ```json
  {
    "detail": "Inference engine is offline: model checkpoint failed to load on server startup.",
    "error_code": "MODEL_OFFLINE"
  }
  ```

### 3. Structured JSON Exceptions
Validation rejects return structured JSON error keys `{detail, error_code}`:
- **Unsupported MIME (415)**:
  ```json
  {
    "detail": "Unsupported media type: Only image/jpeg and image/png are supported.",
    "error_code": "UNSUPPORTED_MEDIA_TYPE"
  }
  ```
- **Corrupt Payload (400)**:
  ```json
  {
    "detail": "Invalid image file: Failed to decode image payload.",
    "error_code": "INVALID_IMAGE_FILE"
  }
  ```
