import os
# Hot-reloaded model configs to 16 classes
import sys
import io
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, Response, Request
from fastapi.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv

# Ensure root directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app.schemas import HealthResponse, PredictionResponse, ErrorResponse
from app.inference import get_inference_service, InferenceService

# Load environment configs
load_dotenv()

# Configure logging to stdout for Docker log aggregation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("leaf_api")

class APIException(Exception):
    """Custom API Exception for structured error outputs."""
    def __init__(self, status_code: int, detail: str, error_code: str):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    State manager handling the startup and shutdown lifecycle events.
    Caches model singletons, labels, and thresholds.
    """
    model_path = os.getenv("MODEL_PATH", "models/resnet18_leaf_best.pth")
    threshold = float(os.getenv("DISEASE_THRESHOLD", "0.35"))
    
    logger.info("Starting API. Validating model path configuration: %s", model_path)
    
    app.state.model = None
    app.state.class_names = []
    app.state.threshold = threshold
    app.state.model_loaded = False
    
    if not os.path.exists(model_path):
        logger.error("Configuration Error: Model file does not exist at path: %s", model_path)
    else:
        try:
            import torch
            from src.inference import load_model, get_class_names
            
            logger.info("Loading model weights from %s...", model_path)
            model, num_classes, checkpoint = load_model(model_path)
            class_names = get_class_names(checkpoint, num_classes)
            
            # Warm up model with dummy tensor to prevent cold-start latency spike
            logger.info("Performing model warmup with dummy tensor...")
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(dummy_input)
                
            app.state.model = model
            app.state.class_names = class_names
            app.state.model_loaded = True
            logger.info("Model loaded and warmed up successfully. Class count: %d", num_classes)
        except Exception as e:
            logger.exception("Model initialization failed on startup: %s", e)

    yield
    logger.info("Shutting down API service.")

app = FastAPI(
    title="Zelbytes Leaf Disease API",
    description="Autonomous quality inspection — visual disease detection",
    version="1.0.0",
    lifespan=lifespan,
)

@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handles structured API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors to avoid leaking stack traces to clients."""
    logger.exception("Unhandled server exception occurred: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error_code": "INTERNAL_SERVER_ERROR"}
    )

def verify_model_loaded():
    """Dependency verifying that the model loaded successfully during startup."""
    model_loaded = getattr(app.state, "model_loaded", False)
    if not model_loaded:
        raise APIException(
            status_code=503,
            detail="Inference engine is offline: model checkpoint failed to load on server startup.",
            error_code="MODEL_OFFLINE"
        )

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Reports API service health status and loaded model availability.
    """
    model_loaded = getattr(app.state, "model_loaded", False)
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
    )

@app.post(
    "/predict", 
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image payload"},
        413: {"model": ErrorResponse, "description": "Payload exceeds 5MB limit"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        503: {"model": ErrorResponse, "description": "Inference engine offline"}
    }
)
async def predict_endpoint(
    response: Response,
    file: UploadFile = File(...),
    _=Depends(verify_model_loaded),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Accepts an uploaded image file, processes it, and returns the classification output.
    Enforces MIME type limits, size constraints, and server-side resizing.
    """
    # 1. Validate MIME type
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise APIException(
            status_code=415, 
            detail="Unsupported media type: Only image/jpeg and image/png are supported.",
            error_code="UNSUPPORTED_MEDIA_TYPE"
        )

    # 2. Start latency tracking
    start_time = time.perf_counter()

    # 3. Read image bytes and validate file size (Limit to 5MB)
    try:
        raw_data = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file payload: %s", e)
        raise APIException(
            status_code=400,
            detail="Failed to read upload payload.",
            error_code="BAD_REQUEST"
        )

    logger.info("Received prediction request for file '%s' (size: %d bytes)", file.filename, len(raw_data))

    if len(raw_data) > 5 * 1024 * 1024:
        raise APIException(
            status_code=413, 
            detail="Payload Too Large: image exceeds the maximum 5MB limit.",
            error_code="PAYLOAD_TOO_LARGE"
        )

    # 4. Decrypt / decode image
    try:
        image = Image.open(io.BytesIO(raw_data)).convert("RGB")
    except Exception:
        raise APIException(
            status_code=400, 
            detail="Invalid image file: Failed to decode image payload.",
            error_code="INVALID_IMAGE_FILE"
        )

    # 5. Server-side resizing of large DSLRs
    width, height = image.size
    if width > 1024 or height > 1024:
        logger.info("Resizing input image from %dx%d down to max 1024px boundary", width, height)
        image.thumbnail((1024, 1024))

    # 6. Execute inference and handle exceptions cleanly
    try:
        result = service.predict(image)
    except Exception as e:
        logger.exception("Inference error occurred during model predict pass: %s", e)
        raise APIException(
            status_code=500, 
            detail="Inference pipeline execution error.",
            error_code="INFERENCE_EXECUTION_ERROR"
        )

    # 7. Add Latency Headers
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Inference-Ms"] = f"{duration_ms:.2f}"

    return PredictionResponse(**result)
