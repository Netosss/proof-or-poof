from pydantic import BaseModel
from app.config import settings


class RechargeRequest(BaseModel):
    device_id: str
    amount: int = settings.default_recharge_amount
    secret_key: str
