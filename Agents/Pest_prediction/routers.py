from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
from .agent import PestPredictionAgent
from .schemas import PestPredictionRequest, PestPredictionResponse
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pest", tags=["PestPrediction"])

_agent_instance = None

def get_agent() -> PestPredictionAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = PestPredictionAgent()
            logger.info("Pest prediction agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/predict", response_model=PestPredictionResponse)
def predict_pest(query: str = Form(...), image: UploadFile = File(None)):
    try:
        image_path = None
        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(image.file, tmp)
                image_path = tmp.name
        agent = get_agent()
        result = agent.respond(query, image_path)
        return PestPredictionResponse(
            success=True,
            response=result
        )
    except Exception as e:
        logger.error(f"Pest prediction error: {str(e)}")
        return PestPredictionResponse(
            success=False,
            response=None,
            error=f"Failed to predict pest: {str(e)}"
        )
