from pydantic import BaseModel


class ShareRequest(BaseModel):
    short_id: str


class ShareResponse(BaseModel):
    report_id: str
