from pydantic import BaseModel
from typing import Dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_diseased: bool
    probabilities: Dict[str, float]
