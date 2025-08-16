import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from pydantic import BaseModel
from agno.tools.google_maps import GoogleMapTools
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.fetchWeatherForecast import  get_google_weather_forecast
from Tools.getCropRecommendation import get_consensus_prediction, get_crop_recommendation, get_all_model_predictions

load_dotenv()

class CropRecommendation(BaseModel):
    crop_names : list[str]
    confidence_scores : list[float]
    justifications : list[str]

class CropRecommenderAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[TavilyTools(), GoogleMapTools(), get_google_weather_forecast, get_all_model_predictions, get_consensus_prediction, get_crop_recommendation],
            instructions="""
You are an expert crop recommendation agent for Indian agriculture. For any query, analyze soil nutrients, climate, rainfall, and location context. Use all available models and tools to provide consensus and model-specific crop recommendations.

OUTPUT REQUIREMENTS:
- Provide a structured output.
- List the top 3 recommended crops, each with:
    - Crop name
    - Confidence score
    - Detailed justification (soil, climate, rainfall, market, and other relevant factors)
- Include a comparison across models if available.
- Give actionable advice for farmers.
- Use the weather tools to fetch real-time weather data and use them properly.
- Incorporate local market trends and prices into recommendations and potential risk factors.
- You have to incorporate weather and market related statistics into your analysis.
- If location or soil data is missing, use TavilyTools or Google Maps to fetch it.
- Present results with clear headings, tables, and bullet points.
- Do not mention tool calling in your response.
- Justifications must be detailed and descriptive
""",
            show_tool_calls=True,
            markdown=True, 
            response_model=CropRecommendation,
        )

    def respond(self, prompt: str) -> str:
        return self.agent.run(prompt).content

if __name__ == "__main__":
    agent = CropRecommenderAgent()
    prompt = "Recommend the best crops for Kharif season in Nashik, Maharashtra with soil and rainfall details."
    print(agent.respond(prompt))


