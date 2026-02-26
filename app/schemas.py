from pydantic import BaseModel
from typing import Optional, List

class EvidenceItem(BaseModel):
    layer: str      # e.g., "Metadata Check", "Technical Analysis", "Deep Forensics"
    status: str     # "passed" (Green), "warning" (Yellow), "flagged" (Red), "info" (Neutral)
    label: str      # Short title e.g., "Digital Signature"
    detail: str     # User-facing explanation e.g., "Valid camera metadata found."

class DetectionResponse(BaseModel):
    summary: str
    confidence_score: float
    new_balance: Optional[int] = None
    is_short_circuited: bool = False
    evidence_chain: List[EvidenceItem]
    short_id: Optional[str] = None

class ShareRequest(BaseModel):
    short_id: str

class ShareResponse(BaseModel):
    report_id: str
