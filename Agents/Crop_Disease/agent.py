import os
import sys
from agno.agent import Agent
from agno.media import Image
from pathlib import Path
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.crop_disease_detection import detect_crop_disease
from agno.tools.tavily import TavilyTools
load_dotenv()


class CropDiseaseOutput(BaseModel):
    disease_name: Optional[list[str]] = []
    disease_probability: Optional[list[float]] = []
    symptoms: Optional[list[str]] = []
    Treatments: Optional[list[str]] = []
    prevention_tips: Optional[list[str]] = []

class CropDiseaseAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            debug_mode=False,
            show_tool_calls=True,
            add_history_to_messages=True,
            num_history_responses=5,
            tools=[detect_crop_disease, TavilyTools()],
            response_model=CropDiseaseOutput,
            instructions="""
You are an advanced crop disease analysis agent. Your task is to analyze crop images for disease symptoms and provide a clear diagnosis and actionable recommendations.

IMPORTANT: You MUST return a valid JSON response that matches the CropDiseaseOutput schema exactly.

If image is provided, analyze it and provide:
- disease_name: List of top 3 disease names (strings)
- disease_probability: List of probabilities as decimals (e.g., 0.85 for 85%)
- symptoms: List of observed symptoms
- Treatments: List of treatment recommendations  
- prevention_tips: List of prevention tips (max 10 words each)

If no image is provided, set disease_name and disease_probability to empty lists, but still provide symptoms, treatments, and prevention_tips based on the query context.


PROMPTING STRATEGY:
- When an image is provided, first attempt to identify the disease using your own analysis.
- If the crop_disease_detection tool is called and returns a result, present the top 3 most probable diseases with their probabilities.
- Use TavilyTools to search for current weather and disease outbreak information for the location or crop mentioned.
- Always justify your diagnosis and recommendations with reference to visible symptoms, crop type, agricultural context, and current weather.
- Include actionable steps for disease management, prevention, and follow-up monitoring.
"""
        )

    def analyze_disease(self, query: str, image_path=None):
        if image_path and os.path.exists(image_path):
            print("Image path exists")
            image = Image(filepath=Path(image_path))
            prompt = f"Analyze this crop image for disease symptoms and provide diagnosis with structured output: {query}"
            result = self.agent.run(prompt, images=[image])
        else:
            prompt = f"No image provided. Analyze based on context only. Set disease_name and disease_probability to empty lists, but provide symptoms, treatments, and prevention tips for: {query}"
            result = self.agent.run(prompt)
        
        # Check if result has content attribute
        if hasattr(result, 'content'):
            response = result.content
        else:
            response = result
            
        print(f"Agent CropDiseaseDetectionAgent Response: {response}")
        
        # Validate the response has the required fields
        if (hasattr(response, "disease_name") and hasattr(response, "disease_probability") and 
            hasattr(response, "symptoms") and hasattr(response, "Treatments") and 
            hasattr(response, "prevention_tips")):
            
            # Check if all critical fields are None/empty
            if (not response.disease_name and not response.disease_probability and 
                not response.symptoms and not response.Treatments and not response.prevention_tips):
                print("Warning: All outputs are empty. Creating default response.")
                # Create a default response
                response = CropDiseaseOutput(
                    disease_name=["Unable to determine"],
                    disease_probability=[0.0],
                    symptoms=["Image analysis failed"],
                    Treatments=["Consult local agricultural expert"],
                    prevention_tips=["Regular crop monitoring recommended"]
                )
        
        return response

if __name__ == "__main__":
    agent = CropDiseaseAgent()
    result = agent.analyze_disease(query="describe the diseases", image_path="../../Images/Crop/crop_disease.jpg")
    print("Disease names:", result.disease_name)
    print("Disease probabilities:", result.disease_probability)
    print("Symptoms:", result.symptoms)
    print("Treatments:", result.Treatments)
    print("Prevention tips:", result.prevention_tips)