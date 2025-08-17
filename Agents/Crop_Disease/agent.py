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
    disease_name: Optional[list[str]]
    disease_probability: Optional[list[float]]
    symptoms: Optional[list[str]]
    Treatments: Optional[list[str]]
    prevention_tips: Optional[list[str]]

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

If image is not provided then no need to provide disease name and disease probability, provide other things.

PROMPTING STRATEGY:
- When an image is provided, first attempt to identify the disease using your own analysis.
- If the crop_disease_detection tool is called and returns a result, present the top 3 most probable diseases with their probabilities.
- Use TavilyTools to search for current weather and disease outbreak information for the location or crop mentioned. Use weather information to inform your diagnosis and recommendations.
- If you are not satisfied with the model's output, provide your own best diagnosis, justification, and recommendations based on the image, weather, and context.
- Always justify your diagnosis and recommendations with reference to visible symptoms, crop type, agricultural context, and current weather.
- Include actionable steps for disease management, prevention, and follow-up monitoring.
- Use clear, concise language suitable for farmers and agronomists.

OUTPUT REQUIREMENTS:
- List exactly the top 3 disease names or healthy status.
- Provide the probability for each disease (as a percentage).
- Give symptoms for each disease.
- Give actionable recommendations for treatment for each disease.
- Provide max 10 word prevention tips for each disease, including cultural, biological, and chemical methods.
- Advice for monitoring and next steps.
- If you are not satisfied with the model's output, clearly state your own diagnosis and reasoning.
"""
        )

    def analyze_disease(self, query: str, image_path=None):
        prompt = f"Analyze this crop image for disease symptoms and provide diagnosis, justification, and recommendations : {query}"
        if image_path and os.path.exists(image_path):
            from agno.media import Image
            image = Image(filepath=Path(image_path))
            result = self.agent.run(prompt, images=[image]).content
            return result
        elif not image_path:
            prompt = (
                "No image provided. Analyze the crop disease based on context only. "
                f"Do not provide disease name or probability, only give symptoms, treatments, prevention tips, and monitoring advice for this {query}"
            )
            result = self.agent.run(prompt).content
            return result
        elif (
            hasattr(result, "disease_name") and hasattr(result, "disease_probability")
            and hasattr(result, "symptoms") and hasattr(result, "Treatments")
            and hasattr(result, "prevention_tips")
            and not result.disease_name and not result.disease_probability
            and not result.symptoms and not result.Treatments and not result.prevention_tips
        ):
            print("Agent CropDiseaseDetectionAgent Response: All outputs are empty. Likely an error in detection or input.")
        return result

if __name__ == "__main__":
    agent = CropDiseaseAgent()
    result = agent.analyze_disease(query = "describe the diseases", image_path="../../Images/Crop/crop_disease.jpg")
    print(result.disease_name)
    print(result.disease_probability)
    print(result.symptoms)
    print(result.Treatments)
    print(result.prevention_tips)
