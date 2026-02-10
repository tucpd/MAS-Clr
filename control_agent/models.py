from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Schema cho Air Conditioner
class AirConditioner(BaseModel):
    status: int = Field(..., ge=0, le=1)
    set_temperature: Optional[float] = Field(None, ge=20, le=28)
    fan_speed: Optional[str] = Field(None, regex="^(low|med|high)$")

# Schema cho Device Commands (Validation)
class DeviceCommands(BaseModel):
    lights_pairs: List[Dict[str, int]] = Field(..., min_items=2, max_items=2)  # [{"status": 0 or 1}]
    fans_pairs: List[Dict[str, int]] = Field(..., min_items=2, max_items=2)    # [{"speed": 0/25/50/99}]
    air_conditioner: AirConditioner
    projector: int = Field(..., ge=0, le=1)
    reasoning: str = Field(..., min_length=1)  # Lý do, cho debug