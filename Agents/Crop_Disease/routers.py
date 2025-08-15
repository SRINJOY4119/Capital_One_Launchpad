from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from .agent import CropDiseaseAgent
from .schemas import CropDiseaseDetectionResponse

router = APIRouter(prefix="/api/v1/cropdisease", tags=["CropDiseaseDetection"])

_agent_instance = None

def get_agent() -> CropDiseaseAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CropDiseaseAgent()
    return _agent_instance

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/detect", response_model=CropDiseaseDetectionResponse)
async def detect_disease(image: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(image.filename)[-1]
        temp_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        agent = get_agent()
        result = agent.analyze_disease(temp_path)
        disease_name = result.disease_name
        disease_probability = result.disease_probability
        symptoms = result.symptoms
        treatments = result.Treatments
        prevention_tips = result.prevention_tips
        return CropDiseaseDetectionResponse(
            success=True,
            diseases=disease_name,
            disease_probabilities=disease_probability,
            symptoms=symptoms,
            Treatments=treatments,
            prevention_tips=prevention_tips,
            image_path=temp_path
        )
    except Exception as e:
        return CropDiseaseDetectionResponse(
            success=False,
            diseases=None,
            disease_probabilities=None,
            symptoms=None,
            Treatments=None,
            prevention_tips=None,
            image_path=None,
            error=f"Failed to detect crop disease: {str(e)}"
        )
