from pydantic import BaseModel, Field
from typing import Optional, List

class MetadataLayer(BaseModel):
    status: str  # "verified_ai", "verified_human", "not_found"
    provider: Optional[str] = None
    description: str

class ForensicsLayer(BaseModel):
    status: str  # "detected", "not_detected", "in_progress"
    probability: float
    signals: List[str]

class DetectionLayers(BaseModel):
    layer1_metadata: MetadataLayer
    layer2_forensics: ForensicsLayer

class DetectionResponse(BaseModel):
    summary: str
    confidence_score: float
    layers: DetectionLayers
