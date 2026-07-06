import os
import sys
from fastapi import Request
from PIL import Image

# Ensure project root is in path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.inference import predict_image

class InferenceService:
    def __init__(self, model, class_names, threshold):
        self.model = model
        self.class_names = class_names
        self.threshold = threshold

    def predict(self, image: Image.Image):
        """
        Executes leaf disease evaluation on the provided PIL Image.
        """
        if self.model is None:
            raise RuntimeError("Primary model singleton is not loaded on this node.")
        return predict_image(image, self.model, self.class_names, self.threshold)

def get_inference_service(request: Request) -> InferenceService:
    """
    Dependency injection function returning a configured InferenceService wrapper.
    Resolves loaded model configuration from the FastAPI application state.
    """
    app_state = request.app.state
    model = getattr(app_state, "model", None)
    class_names = getattr(app_state, "class_names", [])
    threshold = getattr(app_state, "threshold", 0.35)
    return InferenceService(model, class_names, threshold)
