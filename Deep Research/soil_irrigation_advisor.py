import os
from typing import List, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv
import pandas as pd

class FarmingResource(BaseModel):
    title: str = Field(..., description="Title of the agricultural resource")
    description: str = Field(..., description="Brief description of the resource")
    url: str = Field(..., description="Direct link to the resource")
    resource_type: str = Field(..., description="Type of resource (Research/Guide/Tool/Practice)")

class LocalRecommendation(BaseModel):
    crop: str = Field(..., description="Crop type")
    soil_type: str = Field(..., description="Soil type")
    npk_ratio: str = Field(..., description="Recommended NPK ratio")
    irrigation: str = Field(..., description="Irrigation guidelines")

class AgriResources(BaseModel):
    soil_guides: List[FarmingResource] = Field(default_factory=list)
    irrigation_methods: List[FarmingResource] = Field(default_factory=list)
    fertilizer_info: List[FarmingResource] = Field(default_factory=list)
    sustainable_practices: List[FarmingResource] = Field(default_factory=list)
    local_recommendations: List[LocalRecommendation] = Field(default_factory=list)

    @classmethod
    def parse_response(cls, response_content: dict) -> 'AgriResources':
        try:
            # Map the response fields to our model fields
            field_mapping = {
                'soilguides': 'soil_guides',
                'irrigationmethods': 'irrigation_methods',
                'fertilizerinfo': 'fertilizer_info',
                'sustainablepractices': 'sustainable_practices',
                'localrecommendations': 'local_recommendations'
            }

            parsed_data = {}
            for api_field, model_field in field_mapping.items():
                if api_field in response_content:
                    resources = []
                    for item in response_content[api_field]:
                        if isinstance(item, dict):
                            if model_field != 'local_recommendations':
                                resources.append(FarmingResource(**item))
                            else:
                                resources.append(LocalRecommendation(**item))
                    parsed_data[model_field] = resources
                else:
                    parsed_data[model_field] = []

            return cls(**parsed_data)
        except Exception as e:
            print(f"Warning: Error parsing response - {str(e)}")
            return cls()  # Return empty instance if parsing fails

class FarmAdvisor:
    def __init__(self):
        load_dotenv()
        self.agent = Agent(
            name="Agricultural Expert",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[GoogleSearchTools(), DuckDuckGoTools(), TavilyTools()],
            response_model=AgriResources,
            use_json_mode=True,
            show_tool_calls=True,
            instructions=self._get_instructions(),
            markdown=True
        )
        self._load_reference_data()

    def _get_instructions(self) -> str:
        return """
        You are an agricultural expert specialized in finding resources for:
        1. 🌱 Soil Management: pH balancing, organic matter, soil structure
        2. 💧 Irrigation Methods: water requirements, timing, techniques
        3. 🌿 Fertilizer Application: NPK ratios, timing, organic/inorganic
        4. ♻️ Sustainable Practices: conservation, efficiency, environmental care

        Ensure all resources are:
        - Research-based and scientifically validated
        - From agricultural extension services or universities
        - Current within last 5 years when possible
        - Practical for farmers to implement
        """

    def _load_reference_data(self):
        try:
            self.crop_data = pd.read_csv('../Dataset/crop_recommendation.csv')
            self.fertilizer_data = pd.read_csv('../Dataset/Fertilizer_recommendation.csv')
        except Exception as e:
            print(f"Warning: Could not load reference data - {str(e)}")
            self.crop_data = pd.DataFrame()
            self.fertilizer_data = pd.DataFrame()

    def find_farming_resources(self, crop_type: str, soil_type: str, region: Optional[str] = None) -> AgriResources:
        query = f"""
        Find agricultural resources for:
        Crop: {crop_type}
        Soil Type: {soil_type}
        {f'Region: {region}' if region else ''}
        
        Please format response as JSON with these exact fields:
        {{
            "soilguides": [{{
                "title": "string",
                "description": "string",
                "url": "string",
                "resource_type": "string"
            }}],
            "irrigationmethods": [...],
            "fertilizerinfo": [...],
            "sustainablepractices": [...]
        }}
        """
        
        try:
            response = self.agent.run(query)
            print(f"\nTokens Used - Input: {response.metrics['input_tokens'][0]}, "
                  f"Output: {response.metrics['output_tokens'][0]}")
            
            # Parse the response content
            if isinstance(response.content, str):
                import json
                content = json.loads(response.content)
            else:
                content = response.content
                
            # Convert to AgriResources model
            resources = AgriResources.parse_response(content)
            return resources
            
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            return AgriResources()

def main():
    advisor = FarmAdvisor()
    
    print("Agricultural Resource Finder")
    print("-" * 30)
    crop = input("Enter crop type: ").strip()
    soil = input("Enter soil type: ").strip()
    region = input("Enter region (optional): ").strip() or None

    try:
        print("\nFinding agricultural resources...")
        resources = advisor.find_farming_resources(crop, soil, region)
        
        sections = {
            "SOIL MANAGEMENT GUIDES": resources.soil_guides,
            "IRRIGATION METHODS": resources.irrigation_methods,
            "FERTILIZER INFORMATION": resources.fertilizer_info,
            "SUSTAINABLE PRACTICES": resources.sustainable_practices
        }

        for title, items in sections.items():
            if items:
                print(f"\n=== {title} ===")
                for res in items:
                    print(f"\n- {res.title}")
                    print(f"  Type: {res.resource_type}")
                    print(f"  Info: {res.description}")
                    print(f"  Link: {res.url}")

        if resources.local_recommendations:
            print("\n=== LOCAL RECOMMENDATIONS ===")
            for rec in resources.local_recommendations:
                print(f"\nCrop: {rec.crop}")
                print(f"Soil Type: {rec.soil_type}")
                print(f"NPK Ratio: {rec.npk_ratio}")
                print(f"Irrigation: {rec.irrigation}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again with different search terms.")

if __name__ == "__main__":
    main()