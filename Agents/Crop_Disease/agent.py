import os
import sys
from agno.agent import Agent
from agno.media import Image
from pathlib import Path
from agno.models.google import Gemini
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.crop_disease_detection import detect_crop_disease
from agno.tools.tavily import TavilyTools
load_dotenv()

class CropDiseaseAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            debug_mode=False,
            show_tool_calls=True,
            tools=[detect_crop_disease, TavilyTools()],
            instructions="""
You are an advanced crop disease analysis agent. Your task is to analyze crop images for disease symptoms and provide a clear diagnosis and actionable recommendations.

PROMPTING STRATEGY:
- When an image is provided, first attempt to identify the disease using your own analysis.
- If the crop_disease_detection tool is called and returns a result, present that diagnosis and recommendation.
- If you do not use the tool, provide your own best diagnosis and advice based on the image and context.
- Always justify your diagnosis and recommendations with reference to visible symptoms, crop type, and agricultural context.
- Include actionable steps for disease management, prevention, and follow-up monitoring.
- Use clear, concise language suitable for farmers and agronomists.

OUTPUT REQUIREMENTS:
- Disease name or healthy status.
- Justification for diagnosis based on image features.
- Actionable recommendations for treatment and prevention.
- Advice for monitoring and next steps.
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
    print(result)
