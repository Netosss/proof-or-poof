from pydantic import BaseModel
from typing import Optional

class DetectionResponse(BaseModel):
    is_ai: Optional[bool]
    provider: Optional[str]
    method: str
    confidence: float



