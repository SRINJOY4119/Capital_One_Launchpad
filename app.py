from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import os
import tempfile
import time
import json
from pathlib import Path

from Agents.Multi_Lingual.routers import router as multilingual_router
from Agents.Risk_Management.routers import router as risk_router
from Agents.Web_Scrapping.routers import router as webscrap_router
from Agents.Credit_Policy_Market.routers import router as creditpolicy_router
from Agents.Weather_forcast.routers import router as weather_router
from Agents.Crop_Disease.routers import router as crop_disease_router
from Agents.Market_Price.routers import router as market_price_router
from Agents.Image_Analysis.routers import router as image_analysis_router
from Agents.Pest_prediction.routers import router as pest_prediction_router
from Agents.Crop_Recommender.routers import router as crop_recommender_router
from Agents.Crop_Yield.routers import router as crop_yield_router
from Agents.Location_Information.routers import router as location_information_router
from Agents.News.routers import router as news_router
from Agents.Fertilizer_Recommender.routers import router as fertilizer_recommender_router
from Deep_Research.routers import router as deep_research_router
from Tools.tool_apis_router import router as tool_apis_router

from workflow import run_workflow


app = FastAPI(
    title="Agricultural Agent API",
    description="API for multilingual agricultural assistance, analytics, and multi-agent workflow processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowRequestNormalQuery(BaseModel):
    query: str = Field(..., description="The agricultural query to process")

class WorkflowResponse(BaseModel):
    answer: str
    agent_responses: Dict[str, Any]
    routed_agents: List[str]
    processing_time: Optional[float] = None

app.include_router(multilingual_router)
app.include_router(risk_router)
app.include_router(webscrap_router)
app.include_router(creditpolicy_router)
app.include_router(weather_router)
app.include_router(crop_disease_router)
app.include_router(market_price_router)
app.include_router(image_analysis_router)
app.include_router(pest_prediction_router)
app.include_router(crop_recommender_router)
app.include_router(location_information_router)
app.include_router(news_router)
app.include_router(fertilizer_recommender_router)
app.include_router(deep_research_router)
app.include_router(crop_yield_router)
app.include_router(tool_apis_router)

def serialize_agent_responses(responses: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for agent_name, response in responses.items():
        try:
            json.dumps(response)
            serialized[agent_name] = response
        except (TypeError, ValueError):
            if hasattr(response, '__dict__'):
                serialized[agent_name] = response.__dict__
            elif hasattr(response, 'dict'):
                try:
                    serialized[agent_name] = response.dict()
                except:
                    serialized[agent_name] = str(response)
            else:
                serialized[agent_name] = str(response)
    return serialized

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Agricultural AI API",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Agricultural AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/api/v1/workflow/process", response_model=WorkflowResponse, tags=["Multi-Agent Workflow"])
async def process_workflow_query(request: WorkflowRequestNormalQuery):
    try:
        start_time = time.time()
        result = run_workflow(
            query=request.query,
            image_path=None
        )
        end_time = time.time()
        processing_time = end_time - start_time

        result["processing_time"] = processing_time
        result["agent_responses"] = serialize_agent_responses(result.get("agent_responses", {}))

        return WorkflowResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/workflow/process-with-image", tags=["Multi-Agent Workflow"])
async def process_workflow_with_image(
    query: str,
    image: Optional[UploadFile] = File(None)
):
    temp_file_path = None
    try:
        image_path = None

        if image:
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            file_extension = Path(image.filename).suffix.lower()

            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
                )

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    content = await image.read()
                    tmp.write(content)
                    temp_file_path = tmp.name
                    image_path = temp_file_path
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving uploaded image: {str(e)}")

        start_time = time.time()
        result = run_workflow(
            query=query,
            image_path=image_path
        )
        end_time = time.time()
        processing_time = end_time - start_time

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_err:
                print(f"Warning: Could not clean up temp file: {cleanup_err}")

        result["processing_time"] = processing_time
        result["agent_responses"] = serialize_agent_responses(result.get("agent_responses", {}))

        return JSONResponse(content=result)

    except ValueError as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Agricultural Agent API",
        "version": "1.0.0",
        "description": "Comprehensive agricultural assistance with multi-agent workflow processing",
        "endpoints": {
            "agriculture_query": "/api/v1/agriculture/respond",
            "risk_analysis": "/api/v1/risk/analyze",
            "web_scrap": "/api/v1/webscrap/scrape",
            "credit_policy": "/api/v1/creditpolicy/analyze",
            "weather_forecast": "/api/v1/weather/forecast",
            "pest_prediction": "/api/v1/pest/predict",
            "multi_agent_workflow": "/api/v1/workflow/process",
            "workflow_with_image": "/api/v1/workflow/process-with-image",
            "available_agents": "/api/v1/workflow/agents"
        },
        "features": [
            "Multi-agent agricultural assistance",
            "Parallel agent execution",
            "Image analysis capabilities",
            "Multi-language support",
            "Real-time data integration",
            "Intelligent response synthesis"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "agriculture-agent-api",
        "workflow_status": "ready"
    }

@app.get("/api-info")
async def api_info():
    return {
        "api_version": "1.0.0",
        "workflow_features": {
            "multi_agent_processing": "Routes queries to relevant specialized agents",
            "parallel_execution": "Runs multiple agents concurrently for efficiency",
            "response_synthesis": "Combines agent responses into comprehensive answers",
            "image_processing": "Handles image-based agricultural queries",
            "intelligent_routing": "Automatically selects appropriate agents for each query"
        },
        "supported_agents": [
            "Crop Recommendation", "Weather Forecasting", "Location Agriculture",
            "Agricultural News", "Credit Policy & Market", "Crop Disease Detection", 
            "Image Analysis", "Market Pricing", "Multi-language Translation",
            "Pest Prediction", "Risk Management", "Web Scraping", "Crop Yield Estimation",
            "Fertilizer Recommendation"
        ],
        "workflow_process": {
            "step_1": "Router analyzes query and selects relevant agents",
            "step_2": "Selected agents process query in parallel",
            "step_3": "Synthesizer combines all agent responses",
            "step_4": "Return comprehensive final answer"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )