from pydantic import BaseModel, Field
from typing import Optional

class CropDiseaseDetectionResponse(BaseModel):
    success: bool = Field(..., description="Whether disease detection was successful")
    diseases: Optional[str] = Field(None, description="Diagnosis and recommendations for crop disease")
    image_path: Optional[str] = Field(None, description="Path to the uploaded image")
    error: Optional[str] = Field(None, description="Error message if detection failed")
