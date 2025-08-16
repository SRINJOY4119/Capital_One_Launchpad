import os
import re
import json
from textwrap import dedent
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import logging
from rich.pretty import pprint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ActionItem(BaseModel):
    """Individual action item with timeline"""
    action: str = Field(..., description="Specific action to take")
    timeline: str = Field(..., description="When to perform this action")
    priority: str = Field(..., description="Priority level: High/Medium/Low")


class CostBreakdown(BaseModel):
    """Cost breakdown for inputs and treatments"""
    item: str = Field(..., description="Input/treatment name")
    quantity: str = Field(..., description="Required quantity")
    unit_price: float = Field(..., description="Price per unit")
    total_cost: float = Field(..., description="Total cost for this item")


class WeatherRecommendation(BaseModel):
    """Weather-based timing recommendation"""
    activity: str = Field(..., description="Agricultural activity")
    optimal_conditions: str = Field(..., description="Ideal weather conditions")
    timing: str = Field(..., description="Best timing for the activity")


class GovernmentScheme(BaseModel):
    """Government scheme information"""
    scheme_name: str = Field(..., description="Name of the scheme")
    eligibility: str = Field(..., description="Eligibility criteria")
    benefit: str = Field(..., description="Benefits provided")
    application_process: str = Field(..., description="How to apply")


class SimpleAgriculturalReport(BaseModel):
    """Simplified agricultural report structure following structured output pattern"""
    
    # Core sections as simple strings or lists
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
        
        # Agent that uses structured outputs - following the exact movie script pattern!
        self.agent = Agent(
            name="Agricultural Report Generator",
            model=Gemini(id="gemini-2.0-flash-exp"),
            description="You are an expert Agricultural Consultant AI specialized in Indian farming conditions, crops, and agricultural practices. Generate comprehensive agricultural reports with intelligent section filling based on query relevance.",
            show_tool_calls=False,
            add_datetime_to_instructions=True,
            markdown=True,
            response_model=SimpleAgriculturalReport,  # This ensures structured output
        )
        
        # Section triggers for query analysis
        self.section_triggers = {
            'A': ['disease', 'pest', 'symptoms', 'problem', 'issue', 'dying', 'wilting', 'spots', 'yellowing', 'infection', 'damage', 'diagnosis'],
            'B': ['what to do', 'how to treat', 'steps', 'solution', 'fix', 'help', 'urgent', 'emergency', 'immediate', 'action', 'treatment'],
            'C': ['fertilizer', 'pesticide', 'seeds', 'inputs', 'nutrients', 'spray', 'application', 'dosage', 'quantity', 'urea', 'dap'],
            'D': ['profitable', 'cost', 'price', 'money', 'investment', 'roi', 'economics', 'market', 'budget', 'profit', 'loss', 'financial'],
            'E': ['when', 'timing', 'season', 'weather', 'plant', 'harvest', 'schedule', 'monsoon', 'temperature', 'rainfall', 'sowing'],
            'F': ['soil', 'ph', 'nutrients', 'organic matter', 'soil test', 'fertility', 'soil health', 'amendments', 'lime', 'nitrogen'],
            'G': ['water', 'irrigation', 'watering', 'drought', 'flood', 'moisture', 'drip', 'sprinkler', 'water schedule', 'pump'],
            'H': ['subsidy', 'government', 'scheme', 'loan', 'support', 'pm-kisan', 'insurance', 'msp', 'policy', 'pradhan mantri'],
            'I': ['storage', 'harvest', 'post-harvest', 'drying', 'preservation', 'packaging', 'quality', 'losses', 'grading', 'warehouse'],
            'J': ['safety', 'protection', 'precautions', 'toxicity', 'organic', 'sustainable', 'ipm', 'integrated pest management', 'bio']
        }

    def analyze_query_sections(self, query):
        """Analyze query to determine which sections are relevant."""
        query_lower = query.lower()
        relevant_sections = set()
        
        # Section A (Problem Diagnosis) is ALWAYS relevant
        relevant_sections.add('A')
        
        # Primary section detection
        for section, triggers in self.section_triggers.items():
            for trigger in triggers:
                if trigger in query_lower:
                    relevant_sections.add(section)
        
        # Smart auto-inclusion of related sections based on query type
        self._auto_include_related_sections(query_lower, relevant_sections)
        
        return list(relevant_sections)

    def _auto_include_related_sections(self, query_lower, relevant_sections):
        """Automatically include related sections based on query context"""
        
        # If disease/pest/problem symptoms are mentioned, automatically include treatment actions and inputs
        disease_symptoms = ['yellow', 'spot', 'disease', 'pest', 'infection', 'dying', 'wilting', 'brown', 'black', 'rust', 'blight', 'rot', 'fungus', 'insect', 'damage', 'symptoms']
        if any(symptom in query_lower for symptom in disease_symptoms):
            relevant_sections.add('A')  # Problem diagnosis (already added but ensuring)
            relevant_sections.add('B')  # Immediate actions
            relevant_sections.add('C')  # Input requirements (pesticides/fungicides)
            relevant_sections.add('J')  # Safety & protection
        
        # If crop cultivation/planning is mentioned, include timing and soil management
        cultivation_terms = ['cultivation', 'growing', 'plant', 'sow', 'seed', 'farming', 'crop']
        if any(term in query_lower for term in cultivation_terms):
            relevant_sections.add('E')  # Weather & timing
            relevant_sections.add('F')  # Soil management
        
        # If specific inputs are mentioned, include cost analysis
        input_terms = ['fertilizer', 'pesticide', 'seeds', 'urea', 'dap', 'spray', 'treatment']
        if any(term in query_lower for term in input_terms):
            relevant_sections.add('C')  # Input requirements
            relevant_sections.add('D')  # Cost analysis
        
        # If irrigation/water issues are mentioned, include related sections
        water_terms = ['water', 'irrigation', 'drought', 'flood', 'moisture', 'watering']
        if any(term in query_lower for term in water_terms):
            relevant_sections.add('G')  # Water management
            relevant_sections.add('F')  # Soil management (related to water retention)
        
        # If harvest/storage is mentioned
        harvest_terms = ['harvest', 'storage', 'post-harvest', 'storing', 'preservation']
        if any(term in query_lower for term in harvest_terms):
            relevant_sections.add('I')  # Post-harvest & storage
            relevant_sections.add('J')  # Safety & protection
        
        # If cost/economics/profit is mentioned
        economic_terms = ['cost', 'price', 'profit', 'loss', 'money', 'investment', 'budget', 'roi']
        if any(term in query_lower for term in economic_terms):
            relevant_sections.add('D')  # Cost analysis
            relevant_sections.add('C')  # Input requirements (for costing)
        
        # If government schemes/subsidies mentioned
        scheme_terms = ['subsidy', 'government', 'scheme', 'loan', 'support', 'pm-kisan', 'insurance']
        if any(term in query_lower for term in scheme_terms):
            relevant_sections.add('H')  # Government schemes

    def generate_agricultural_report(self, query, location="India", crop_type="general"):
        """Generates agricultural report with only relevant sections filled."""
        
        relevant_sections = self.analyze_query_sections(query)
        logger.info(f"Relevant sections for query: {relevant_sections}")
        
        enhanced_query = f"""
FARMER'S QUERY: {query}
LOCATION: {location}
CROP TYPE: {crop_type}

Generate a comprehensive 10-section agricultural report. Follow these rules:

1. ALWAYS fill 'problem_diagnosis' with detailed analysis of the situation
2. Fill other sections ONLY if directly relevant to the query
3. For irrelevant sections, return exactly "None" as the value
4. Provide actionable, detailed content for relevant sections
5. Use Indian agricultural context and practices

RELEVANT SECTIONS DETECTED: {', '.join(relevant_sections)}

Structure your response according to the SimpleAgriculturalReport model with:
- problem_diagnosis: Always filled with situation analysis
- Other sections: Fill only if relevant, otherwise "None"
- summary: Executive summary of the report
- key_recommendations: List of top recommendations
"""
        
        try:
            response = self.agent.run(enhanced_query)
            return response.content
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return self._create_fallback_report(query, str(e))

    def _create_fallback_report(self, query, error_msg):
        """Create a fallback report when the main generation fails."""
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
            summary=f"Error generating report for query: {query}. Error: {error_msg}",
            key_recommendations=[f"Please retry your query or contact support. Error: {error_msg[:100]}"]
        )

    def generate_crop_specific_report(self, crop, issue, location="India"):
        """Generate crop-specific agricultural report."""
        query = f"My {crop} crop in {location} has {issue}"
        return self.generate_agricultural_report(query, location, crop)

    def generate_seasonal_report(self, season, crop, location="India"):
        """Generate seasonal agricultural planning report."""
        query = f"{season} season planning for {crop} cultivation in {location}"
        return self.generate_agricultural_report(query, location, crop)

    def generate_financial_planning_report(self, crop, area, location="India"):
        """Generate financial planning and cost analysis report."""
        query = f"Cost analysis and financial planning for {crop} cultivation in {area} area at {location}"
        return self.generate_agricultural_report(query, location, crop)

    def print_response(self, query, location="India", crop_type="general"):
        """Print response exactly like the movie script example."""
        print(f"\n🌾 Generating Agricultural Report for: {query}")
        print("="*80)
        
        try:
            report = self.generate_agricultural_report(query, location, crop_type)
            
            # Pretty print the structured output (like the movie example)
            pprint(report.dict())
            
            # Also print a formatted summary
            self._print_formatted_summary(report)
            
        except Exception as e:
            print(f"❌ Error: {e}")

    def _print_formatted_summary(self, report):
        """Print a nicely formatted summary of the report."""
        print("\n" + "="*80)
        print("🌾 AGRICULTURAL REPORT SUMMARY 🌾")
        print("="*80)
        
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print(f"   {report.summary}")
        
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(report.key_recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Show which sections have content vs None
        print(f"\n📊 REPORT SECTIONS STATUS:")
        sections = [
            ("A", "Problem Diagnosis", report.problem_diagnosis),
            ("B", "Immediate Actions", report.immediate_actions),
            ("C", "Input Requirements", report.input_requirements),
            ("D", "Cost Analysis", report.cost_analysis),
            ("E", "Weather & Timing", report.weather_timing),
            ("F", "Soil Management", report.soil_management),
            ("G", "Water Management", report.water_management),
            ("H", "Government Schemes", report.government_schemes),
            ("I", "Post-Harvest", report.post_harvest),
            ("J", "Safety & Protection", report.safety_protection),
        ]
        
        for code, name, content in sections:
            status = "✅ Filled" if content != "None" else "❌ None"
            print(f"   Section {code} - {name}: {status}")
        
        print("\n" + "="*80)


def interactive_report_generation():
    """Interactive interface for agricultural report generation"""
    
    agricultural_agent = AgriculturalReportAgent()
    
    print("🌾 Agricultural Report Generation System")
    print("="*60)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Generate custom report")
        print("2. Crop-specific problem report")
        print("3. Seasonal planning report")
        print("4. Financial planning report")
        print("5. View example reports")
        print("6. Test structured output")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            custom_report_generation(agricultural_agent)
        elif choice == "2":
            crop_problem_report(agricultural_agent)
        elif choice == "3":
            seasonal_planning_report(agricultural_agent)
        elif choice == "4":
            financial_planning_report(agricultural_agent)
        elif choice == "5":
            show_example_reports(agricultural_agent)
        elif choice == "6":
            test_structured_output(agricultural_agent)
        elif choice == "7":
            print("Thank you for using the Agricultural Report System! 🚜")
            break
        else:
            print("❌ Invalid choice! Please enter 1-7.")


def test_structured_output(agent):
    """Test structured output exactly like the movie script example."""
    print("\n🧪 Testing Structured Output")
    print("-" * 50)
    
    # Test cases like your movie script example
    test_cases = [
        "My wheat leaves have yellow spots",
        "When to plant rice in Punjab?",
        "Cost of tomato cultivation per acre",
        "How to manage irrigation for cotton crop?"
    ]
    
    for query in test_cases:
        print(f"\n{'='*60}")
        agent.print_response(query)
        print(f"{'='*60}\n")


def custom_report_generation(agent):
    """Handle custom report generation"""
    print("\n📝 Custom Report Generation")
    print("-" * 30)
    
    query = input("Enter your agricultural query: ").strip()
    if not query:
        print("❌ No query entered!")
        return
    
    location = input("Location (default: India): ").strip() or "India"
    crop_type = input("Crop type (default: general): ").strip() or "general"
    
    print("\n🔄 Generating agricultural report...")
    try:
        agent.print_response(query, location, crop_type)
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def crop_problem_report(agent):
    """Handle crop-specific problem reports"""
    print("\n🌿 Crop Problem Report")
    print("-" * 25)
    
    crop = input("Crop name: ").strip()
    issue = input("Problem/Issue: ").strip()
    location = input("Location (default: India): ").strip() or "India"
    
    if not crop or not issue:
        print("❌ Crop name and issue are required!")
        return
    
    print("\n🔄 Generating crop problem report...")
    try:
        report = agent.generate_crop_specific_report(crop, issue, location)
        agent._print_formatted_summary(report)
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def seasonal_planning_report(agent):
    """Handle seasonal planning reports"""
    print("\n🌤️ Seasonal Planning Report")
    print("-" * 30)
    
    season = input("Season (e.g., rabi, kharif, summer): ").strip()
    crop = input("Crop name: ").strip()
    location = input("Location (default: India): ").strip() or "India"
    
    if not season or not crop:
        print("❌ Season and crop name are required!")
        return
    
    print("\n🔄 Generating seasonal planning report...")
    try:
        report = agent.generate_seasonal_report(season, crop, location)
        agent._print_formatted_summary(report)
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def financial_planning_report(agent):
    """Handle financial planning reports"""
    print("\n💰 Financial Planning Report")
    print("-" * 30)
    
    crop = input("Crop name: ").strip()
    area = input("Area (e.g., 5 acres, 2 hectares): ").strip()
    location = input("Location (default: India): ").strip() or "India"
    
    if not crop or not area:
        print("❌ Crop name and area are required!")
        return
    
    print("\n🔄 Generating financial planning report...")
    try:
        report = agent.generate_financial_planning_report(crop, area, location)
        agent._print_formatted_summary(report)
    except Exception as e:
        print(f"❌ Error generating report: {e}")


def show_example_reports(agent):
    """Show example reports"""
    print("\n📚 Example Reports")
    print("=" * 30)
    
    examples = [
        ("Disease Management", "My wheat crop has rust disease"),
        ("Timing Query", "When to plant rice in West Bengal?"),
        ("Cost Analysis", "What is the cost of tomato cultivation per acre?"),
        ("Water Management", "How to manage irrigation for cotton crop?"),
        ("Government Schemes", "What subsidies are available for organic farming?")
    ]
    
    for i, (title, query) in enumerate(examples, 1):
        print(f"\n{i}. {title}")
        print(f"   Query: {query}")
    
    choice = input("\nEnter example number to generate (1-5): ").strip()
    
    try:
        example_idx = int(choice) - 1
        if 0 <= example_idx < len(examples):
            title, query = examples[example_idx]
            print(f"\n🔄 Generating example report: {title}")
            agent.print_response(query)
        else:
            print("❌ Invalid example number!")
    except ValueError:
        print("❌ Please enter a valid number!")
    except Exception as e:
        print(f"❌ Error generating example report: {e}")


if __name__ == "__main__":
    # Test structured output like the movie script example
    agricultural_agent = AgriculturalReportAgent()
    
    print("🌾 Agricultural Report System - Structured Output Demo")
    print("="*70)
    
    # Test structured output exactly like your movie script example
    agricultural_agent.print_response("My wheat leaves have yellow spots")
    
    print("\n" + "="*70 + "\n")
    
    agricultural_agent.print_response("When to plant rice in Punjab?")
    
    print("\n" + "="*70 + "\n")
    
    agricultural_agent.print_response("Cost of tomato cultivation per acre")
    
    # Uncomment to run interactive mode
    # interactive_report_generation()