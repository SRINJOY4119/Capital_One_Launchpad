from pydantic import BaseModel, Field
from typing import Optional, Any

class PestPredictionRequest(BaseModel):
    query: str = Field(..., description="Pest prediction query")
    image_path: Optional[str] = Field(None, description="Path to pest image for identification")

class PestPredictionResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent processed the query successfully")
    response: Optional[str] = Field(None, description="Agent's pest prediction response")
    error: Optional[str] = Field(None, description="Error message if processing failed")
