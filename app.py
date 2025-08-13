from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Agents.Multi_Lingual.routers import router

app = FastAPI(
    title="Agricultural Agent API",
    description="API for multilingual agricultural assistance with code-switching support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "Multilingual Agricultural Agent API",
        "version": "1.0.0",
        "endpoints": {
            "agriculture_query": "/api/v1/agriculture/respond"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "multilingual-agriculture-api"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
