import os
import logging
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SimpleAgriculturalReport(BaseModel):
    """Simplified agricultural report structure following structured output pattern"""
    
    # Core 10 sections
    problem_diagnosis: str = Field(..., description="Problem identification and diagnosis - ALWAYS fill this section")
    immediate_actions: str = Field(default="None", description="Immediate actions needed") 
    input_requirements: str = Field(default="None", description="Required inputs and specifications")
    cost_analysis: str = Field(default="None", description="Cost breakdown and financial analysis")
    weather_timing: str = Field(default="None", description="Weather and timing recommendations")
    soil_management: str = Field(default="None", description="Soil health and management")
    water_management: str = Field(default="None", description="Irrigation and water management")
    government_schemes: str = Field(default="None", description="Government schemes and subsidies")
    post_harvest: str = Field(default="None", description="Post-harvest and storage guidelines")
    safety_protection: str = Field(default="None", description="Safety and protection protocols")
    
    # Always required
    summary: str = Field(..., description="Executive summary of the report")
    key_recommendations: List[str] = Field(..., description="Top key recommendations")


class AgriculturalReportAgent:
    def __init__(self):
        """Initialize the Agricultural Report Agent with the Gemini model."""
        
        self.agent = Agent(
            name="Agricultural Report Generator",
            model=Gemini(id="gemini-2.0-flash-exp"),
            description="You are an expert Agricultural Consultant AI specialized in Indian farming conditions, crops, and agricultural practices. Generate comprehensive agricultural reports with intelligent section filling based on query relevance.",
            show_tool_calls=False,
            add_datetime_to_instructions=True,
            markdown=True,
            response_model=SimpleAgriculturalReport,  # Ensures structured output
        )

    def generate_agricultural_report(self, query, location="India", crop_type="general"):
        """Generates agricultural report with only relevant sections filled."""
        
        enhanced_query = f"""
FARMER'S QUERY: {query}
LOCATION: {location}
CROP TYPE: {crop_type}

Generate a comprehensive agricultural report with exactly 10 sections:
1. problem_diagnosis (always filled)
2. immediate_actions
3. input_requirements
4. cost_analysis
5. weather_timing
6. soil_management
7. water_management
8. government_schemes
9. post_harvest
10. safety_protection

Rules:
- Fill 'problem_diagnosis' with detailed analysis of the situation
- Fill other sections only if directly relevant, otherwise use exactly "None"
- Provide actionable, detailed content for relevant sections
- Use Indian agricultural context and practices
- Always fill 'summary' and 'key_recommendations'
"""

        try:
            response = self.agent.run(enhanced_query)
            return response.content
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return SimpleAgriculturalReport(
                problem_diagnosis=f"Unable to analyze the query: {query}. Error occurred during processing.",
                immediate_actions="None", 
                input_requirements="None",
                cost_analysis="None",
                weather_timing="None",
                soil_management="None",
                water_management="None",
                government_schemes="None",
                post_harvest="None",
                safety_protection="None",
                summary=f"Error generating report for query: {query}. Error: {e}",
                key_recommendations=[f"Please retry your query or contact support. Error: {str(e)[:100]}"]
            )


def print_report(report: SimpleAgriculturalReport):
    """Nicely print the report in terminal."""
    print("\n🌾 AGRICULTURAL REPORT 🌾")
    print("=" * 80)
    print(f"\n📋 EXECUTIVE SUMMARY:\n{report.summary}")
    
    print("\n🎯 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(report.key_recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n📊 REPORT SECTIONS:")
    sections = [
        ("Problem Diagnosis", report.problem_diagnosis),
        ("Immediate Actions", report.immediate_actions),
        ("Input Requirements", report.input_requirements),
        ("Cost Analysis", report.cost_analysis),
        ("Weather & Timing", report.weather_timing),
        ("Soil Management", report.soil_management),
        ("Water Management", report.water_management),
        ("Government Schemes", report.government_schemes),
        ("Post-Harvest", report.post_harvest),
        ("Safety & Protection", report.safety_protection),
    ]
    
    for name, content in sections:
        print(f"\n➡️ {name}:")
        print(f"{content}")


if __name__ == "__main__":
    agricultural_agent = AgriculturalReportAgent()
    
    query = "My wheat leaves have yellow spots"
    report = agricultural_agent.generate_agricultural_report(query)
    
    # Print in clean terminal format
    print_report(report)
