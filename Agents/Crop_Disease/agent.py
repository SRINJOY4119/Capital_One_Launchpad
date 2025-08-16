import os
import sys
from agno.agent import Agent
from agno.media import Image
from pathlib import Path
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.crop_disease_detection import detect_crop_disease
from agno.tools.tavily import TavilyTools
load_dotenv()


class CropDiseaseOutput(BaseModel):
    disease_name: list[str]
    disease_probability: list[float]
    symptoms: list[str]
    Treatments: list[str]
    prevention_tips: list[str]

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

PROMPTING STRATEGY:
- When an image is provided, first attempt to identify the disease using your own analysis.
- If the crop_disease_detection tool is called and returns a result, present the top 3 most probable diseases with their probabilities.
- If you are not satisfied with the model's output, provide your own best diagnosis, justification, and recommendations based on the image and context.
- Always justify your diagnosis and recommendations with reference to visible symptoms, crop type, and agricultural context.
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

    def analyze_disease(self, image_path):
        image = Image(filepath=Path(image_path))
        prompt = "Analyze this crop image for disease symptoms and provide diagnosis, justification, and recommendations."
        result = self.agent.run(prompt, images=[image]).content
        return result

if __name__ == "__main__":
    agent = CropDiseaseAgent()
    result = agent.analyze_disease("../../Images/Crop/crop_disease.jpg")
    print(result.disease_name)
    print(result.disease_probability)
    print(result.symptoms)
    print(result.Treatments)
    print(result.prevention_tips)
