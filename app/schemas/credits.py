from pydantic import BaseModel, Field
from app.config import settings


class RechargeRequest(BaseModel):
    device_id: str
    # Bounded deliberately. `amount: int` accepted any value, so the recharge
    # secret was an unlimited mint, and a NEGATIVE amount debited the wallet
    # instead — neither is a legitimate operator action.
    amount: int = Field(
        default=settings.default_recharge_amount,
        ge=1,
        le=settings.max_recharge_amount,
    )
    secret_key: str
