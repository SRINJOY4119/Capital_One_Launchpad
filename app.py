from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
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
from Agents.Location_Information.routers import router as location_information_router
from Agents.News.routers import router as news_router
from Tools.tool_apis_router import router as tool_apis_router

app = FastAPI(
    title="Agricultural Agent API",
    description="API for multilingual agricultural assistance and analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(tool_apis_router)

@app.get("/")
async def root():
    return {
        "message": "Agricultural Agent API",
        "version": "1.0.0",
        "endpoints": {
            "agriculture_query": "/api/v1/agriculture/respond",
            "risk_analysis": "/api/v1/risk/analyze",
            "web_scrap": "/api/v1/webscrap/scrape",
            "credit_policy": "/api/v1/creditpolicy/analyze",
            "weather_forecast": "/api/v1/weather/forecast",
            "pest_prediction": "/api/v1/pest/predict"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agriculture-agent-api"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
