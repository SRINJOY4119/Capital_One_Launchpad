import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agno.agent import Agent
from agno.models.google import Gemini
from Tools.pest_prediction import PestPredictor
from dotenv import load_dotenv
import json
from agno.media import Image
from pathlib import Path
from datetime import datetime

load_dotenv()

class PestPredictionAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.pest_predictor = PestPredictor()
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            show_tool_calls=True,
            tools=[
                self.predict_pest_from_image,
                self.get_pest_treatment_advice,
                self.analyze_pest_trends,
                self.generate_pest_report
            ],
            instructions="""You are an expert agricultural pest management specialist with advanced AI-powered pest identification capabilities.

Your primary functions:
1. **Pest Identification**: Use image analysis to identify pests with confidence scores when images are provided
2. **Treatment Recommendations**: Provide specific, actionable treatment advice for identified pests
3. **Risk Assessment**: Evaluate threat levels and urgency of pest problems
4. **Prevention Strategies**: Suggest preventive measures and monitoring techniques
5. **Trend Analysis**: Track pest patterns and provide insights

PEST KNOWLEDGE BASE:
- Aphids: Small, soft-bodied insects that feed on plant sap. Treatment: insecticidal soap, neem oil, beneficial insects
- Army Worm: Caterpillars that feed on grasses and crops. Treatment: Bt spray, beneficial nematodes, crop rotation
- Beetle: Various species affecting different crops. Treatment: pyrethrin sprays, row covers, beneficial predators
- Bollworm: Cotton and tomato pest. Treatment: Bt cotton varieties, pheromone traps, biological control
- Earthworm: Generally beneficial but can indicate soil issues. Management: soil drainage, organic matter balance
- Grasshopper: Chewing insects affecting various crops. Treatment: barrier methods, biological control, targeted spraying
- Mites: Tiny arachnids causing leaf damage. Treatment: miticides, predatory mites, proper irrigation
- Mosquito: Disease vectors, nuisance pests. Treatment: eliminate standing water, biological larvicides
- Sawfly: Wasp-like insects with caterpillar larvae. Treatment: hand removal, Bt spray, beneficial insects
- Stem Borer: Internal feeders in plant stems. Treatment: resistant varieties, biological control, proper timing

RESPONSE GUIDELINES:
- Use predict_pest_from_image when an image is available for analysis
- For general questions about pest treatment, use get_pest_treatment_advice
- Always provide confidence levels for pest identification when using images
- Include immediate action steps and long-term management strategies
- Consider organic and conventional treatment options
- Assess economic impact and treatment cost-effectiveness
- Recommend monitoring and follow-up procedures
- Provide seasonal timing recommendations when relevant

Use the available tools to analyze images and provide comprehensive pest management advice."""
        )
        self.current_image_path = None
    
    def predict_pest_from_image(self, image_path: str = None) -> str:
        """Predict pest type from the current image being analyzed"""
        try:
            # Use the current image path if no specific path provided
            path_to_use = image_path or self.current_image_path
            
            if not path_to_use:
                return "No image path provided for pest identification."
            
            pest, confidence = self.pest_predictor.predict_single_image(path_to_use)
            
            result = {
                "predicted_pest": pest,
                "confidence": f"{confidence:.1%}",
                "reliability": "High" if confidence >= 0.7 else "Low",
                "image_analyzed": os.path.basename(path_to_use),
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def get_pest_treatment_advice(self, pest_name: str, crop_type: str = "general", severity: str = "moderate") -> str:
        """Get specific treatment recommendations for identified pests"""
        treatment_database = {
            "aphids": {
                "immediate": "Apply insecticidal soap or neem oil spray every 3-5 days",
                "biological": "Release ladybugs (50-100 per plant) or lacewings",
                "chemical": "Use systemic insecticides (imidacloprid) if severe infestation",
                "prevention": "Monitor regularly, avoid over-fertilizing with nitrogen, encourage beneficial insects"
            },
            "army_worm": {
                "immediate": "Apply Bt (Bacillus thuringiensis) spray in evening when larvae are active",
                "biological": "Use beneficial nematodes (Steinernema carpocapsae) in soil",
                "chemical": "Spinosad or chlorpyrifos for severe cases (follow label instructions)",
                "prevention": "Crop rotation, maintain field hygiene, destroy stubble"
            },
            "beetle": {
                "immediate": "Hand removal for small populations, apply pyrethrin spray",
                "biological": "Encourage beneficial predators like ground beetles and birds",
                "chemical": "Carbaryl or malathion for heavy infestations",
                "prevention": "Row covers during peak activity, crop rotation, trap crops"
            },
            "bollworm": {
                "immediate": "Monitor with pheromone traps, apply Bt spray every 7-10 days",
                "biological": "Release Trichogramma wasps (50,000-100,000 per hectare)",
                "chemical": "Selective insecticides (emamectin benzoate) timed with egg laying",
                "prevention": "Use Bt cotton varieties, destroy crop residues, refugia management"
            },
            "grasshopper": {
                "immediate": "Physical barriers around young plants, targeted spraying of nymphs",
                "biological": "Apply Nosema locustae for long-term population control",
                "chemical": "Acephate or malathion for severe outbreaks",
                "prevention": "Habitat modification, eliminate egg-laying sites, tillage in fall"
            },
            "mites": {
                "immediate": "Increase humidity around plants, apply miticide spray",
                "biological": "Release predatory mites (Phytoseiulus persimilis)",
                "chemical": "Abamectin or bifenthrin for severe infestations",
                "prevention": "Proper irrigation, avoid dust accumulation, quarantine new plants"
            }
        }
        
        pest_lower = pest_name.lower()
        if pest_lower in treatment_database:
            advice = treatment_database[pest_lower]
            advice["severity_assessment"] = severity
            advice["crop_specific_notes"] = f"Treatment adjusted for {crop_type} cultivation"
            advice["economic_threshold"] = "Monitor population levels before treatment"
            return json.dumps(advice, indent=2)
        else:
            return f"Treatment information not available for {pest_name}. General recommendation: Consult with local agricultural extension services for proper identification and treatment options."
    
    def analyze_pest_trends(self) -> str:
        """Analyze pest prediction trends and patterns"""
        if not hasattr(self.pest_predictor, 'prediction_history') or not self.pest_predictor.prediction_history:
            return "No prediction history available for trend analysis. Start analyzing images to build trend data."
        
        history = self.pest_predictor.prediction_history
        pest_frequency = {}
        confidence_trends = []
        
        for record in history:
            pest = record['predicted_pest']
            pest_frequency[pest] = pest_frequency.get(pest, 0) + 1
            confidence_trends.append(record['confidence'])
        
        analysis = {
            "total_predictions": len(history),
            "most_common_pest": max(pest_frequency.items(), key=lambda x: x[1]) if pest_frequency else None,
            "average_confidence": sum(confidence_trends) / len(confidence_trends) if confidence_trends else 0,
            "pest_distribution": pest_frequency,
            "recommendations": self._generate_trend_recommendations(pest_frequency)
        }
        
        return json.dumps(analysis, indent=2)
    
    def generate_pest_report(self) -> str:
        """Generate a comprehensive pest management report"""
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "total_predictions": len(getattr(self.pest_predictor, 'prediction_history', [])),
            "recommendations": [
                "Implement regular monitoring schedule (weekly inspections)",
                "Maintain prediction confidence above 70% for reliable identification",
                "Document all pest sightings for trend analysis",
                "Consider integrated pest management (IPM) approaches",
                "Use economic thresholds before applying treatments"
            ],
            "best_practices": [
                "Early detection is key to effective pest management",
                "Combine multiple identification methods when possible",
                "Keep detailed records of pest occurrences and treatments",
                "Rotate between different control methods to prevent resistance"
            ]
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_trend_recommendations(self, pest_frequency):
        """Generate recommendations based on pest frequency patterns"""
        if not pest_frequency:
            return ["No data available for recommendations"]
        
        recommendations = []
        most_common = max(pest_frequency.items(), key=lambda x: x[1])
        
        if most_common[1] > 3:
            recommendations.append(f"High frequency of {most_common[0]} detected. Consider targeted prevention strategies.")
        
        if len(pest_frequency) > 5:
            recommendations.append("Multiple pest types detected. Implement broad-spectrum IPM approach.")
        
        recommendations.append("Regular monitoring and early detection crucial for effective pest management.")
        
        return recommendations
    
    def respond(self, query, image_path=None):
        """Process user queries about pest identification and management"""
        # Store the current image path for tool access
        self.current_image_path = image_path
        
        if image_path and os.path.exists(image_path):
            # Process with image
            image = Image(filepath=Path(image_path))
            response = self.agent.run(query, images=[image]).content
        else:
            # Process without image
            response = self.agent.run(query).content
        
        return response

def test_pest_prediction_agent():
    agent = PestPredictionAgent()
    
    test_cases = [
        {
            "query": "Analyze this image for pest identification",
            "image_path": "../Dataset/pest/test/aphids/jpg_0.jpg"
        },
        {
            "query": "What treatment do you recommend for aphids on tomato plants?",
            "image_path": None
        },
        {
            "query": "How should I handle a moderate bollworm infestation in cotton?",
            "image_path": None
        }
    ]
    
    print("🐛 Pest Prediction Agent Test")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Query: {test_case['query']}")
        if test_case['image_path']:
            print(f"   Image: {test_case['image_path']}")
        print("-" * 30)
        
        try:
            response = agent.respond(test_case['query'], test_case['image_path'])
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_pest_prediction_agent()
