"""
Pydantic models for the enterprise S2S API.

Response envelopes follow a Stripe-style shape so partner SDKs can parse
machine-readable error codes without relying on free-form `detail` strings.
"""


from pydantic import BaseModel, Field

from app.schemas.detection import DetectionResponse


class EnterpriseError(BaseModel):
    type: str = Field(description="Error category, e.g. 'invalid_request_error', 'authentication_error'")
    code: str = Field(description="Machine-readable code, e.g. 'insufficient_credits'")
    message: str = Field(description="Human-readable explanation")
    request_id: str | None = Field(default=None, description="Correlates with the X-Request-ID header")


class EnterpriseErrorEnvelope(BaseModel):
    error: EnterpriseError


class EnterpriseAnalyzeResponse(BaseModel):
    data: DetectionResponse
    request_id: str
    credits_remaining: int
    idempotent_replay: bool = False
