from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from .agent import MultiLingualAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agriculture", tags=["Agriculture"])

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    success: bool
    response: str
    error: str = None

_agent_instance = None

def get_agent() -> MultiLingualAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = MultiLingualAgent()
            logger.info("Multilingual agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/respond", response_model=QueryResponse)
async def respond_query(request: QueryRequest):
    try:
        agent = get_agent()
        response_content = agent.respond(request.query)
        
        return QueryResponse(
            success=True,
            response=response_content
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        return QueryResponse(
            success=False,
            response="",
            error=f"Failed to process query: {str(e)}"
        )
         