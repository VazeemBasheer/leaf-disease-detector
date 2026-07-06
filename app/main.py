import os
import sys
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from PIL import Image
from dotenv import load_dotenv

# Ensure root directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app.schemas import HealthResponse, PredictionResponse
from app.inference import get_inference_service, InferenceService

# Load environment configs
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    State manager handling the startup and shutdown lifecycle events.
    Caches model singletons, labels, and thresholds.
    """
    # Startup configurations
    model_path = os.getenv("MODEL_PATH", "models/resnet18_leaf_best.pth")
    threshold = float(os.getenv("DISEASE_THRESHOLD", "0.35"))
    
    print(f"Starting API. Loading model from {model_path} with threshold {threshold}...")
    try:
        from src.inference import load_model, get_class_names
        model, num_classes, checkpoint = load_model(model_path)
        class_names = get_class_names(checkpoint, num_classes)
        
        app.state.model = model
        app.state.class_names = class_names
        app.state.threshold = threshold
        app.state.model_loaded = True
        print(f"Model successfully loaded. Classification head shape: {num_classes} classes.")
    except Exception as e:
        print(f"Error loading model singleton during startup: {e}")
        app.state.model = None
        app.state.class_names = []
        app.state.threshold = threshold
        app.state.model_loaded = False
        
    yield
    # Shutdown logic (cleanup if required)
    print("Shutting down API service.")

app = FastAPI(
    title="Zelbytes Leaf Disease API",
    description="Autonomous quality inspection — visual disease detection",
    version="1.0.0",
    lifespan=lifespan,
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

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Accepts an uploaded image file, processes it, and returns the classification output.
    """
    if service.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Inference engine is offline: model checkpoint failed to load on server startup."
        )

    # 1. Load image
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Provided file is not a valid image format. Details: {e}"
        )

    # 2. Run prediction
    try:
        result = service.predict(image)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error executing model prediction pipeline. Details: {e}"
        )
