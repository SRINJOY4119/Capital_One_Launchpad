from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
load_dotenv()

class RoutingDecision(BaseModel):
    agents: List[str]
    justifications: List[str]

class RouterAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            response_model=RoutingDecision,
            instructions="""
You are an intelligent agent router for an agricultural AI platform. Your job is to analyze the user's query and select the most relevant agents to handle it. 

AGENT CAPABILITIES:
- CropRecommenderAgent: Recommends best crops for a location/season based on soil, climate, rainfall, and market context. Provides model comparisons and actionable advice.
- WeatherForecastAgent: Provides weather forecasts, monsoon updates, crop impact analysis, and advisories for agricultural planning.
- LocationAgriAssistant: Handles location-based queries, logistics, mapping, geocoding, farm contacts, agri-businesses, and transit options.
- NewsAgent: Extracts and summarizes recent agricultural news articles, policies, and events for any location or topic.
- CreditPolicyMarketAgent: Analyzes market trends, credit policies, risk assessment, financial guidance, and provides strategic recommendations for agricultural finance.
- FertilizerRecommendationAgent: Recommends optimal fertilizers for crops based on soil, climate, crop type, and nutrient levels.
- CropYieldAgent: Predicts crop yield for specific crops, locations, and seasons using historical and real-time data.
- PestPredictionAgent: Detects pests in crop images and recommends appropriate treatments using computer vision models.
- RiskManagementAgent: Assesses agricultural risk profiles for commodities, including market, weather, financial, and operational risks.
- CropDiseaseDetectionAgent: Detects diseases in crop leaf images and provides disease identification and management advice.
- MarketPriceAgent: Fetches latest market prices for commodities in specific states, districts, or markets.
- TranslationAgent: Translates agricultural documents, queries, and policies between languages, including code-switched queries.

You must provide a structured output containing:
- agents: a list of agent names (choose from the above) that should be called for the query.
- justifications: a list of detailed, step-by-step reasons for why each agent was selected, based on the query's content and intent.

PROMPTING STRATEGY:
- Carefully read and understand the user's query.
- Identify the main topics, required data types, and reasoning steps.
- Map each topic or subtask to the most suitable agent(s).
- If multiple agents are needed, explain the reasoning for each.
- If the query is ambiguous, select agents that can clarify or handle broad queries.
- Always provide clear, actionable justifications for your routing decisions.

FEW-SHOT EXAMPLES:
Example 1:
Query: "Give me the latest weather forecast for wheat farming in Punjab and recommend the best crops for the upcoming season."
Output:
{
  "agents": ["WeatherForecastAgent", "CropRecommenderAgent"],
  "justifications": [
    "WeatherForecastAgent is selected to provide the latest weather forecast for Punjab, which is crucial for wheat farming.",
    "CropRecommenderAgent is selected to recommend the best crops for the upcoming season based on the weather forecast and local conditions."
  ]
}

Example 2:
Query: "Show me recent news about agricultural policies in Maharashtra."
Output:
{
  "agents": ["NewsAgent"],
  "justifications": [
    "NewsAgent is selected to extract and summarize recent news articles about agricultural policies in Maharashtra."
  ]
}

Example 3:
Query: "Analyze market risks for rice and provide credit policy recommendations."
Output:
{
  "agents": ["CreditPolicyMarketAgent"],
  "justifications": [
    "CreditPolicyMarketAgent is selected to analyze market risks for rice and provide relevant credit policy recommendations."
  ]
}

Example 4:
Query: "What are the best logistics routes for transporting tomatoes from Nashik to Mumbai?"
Output:
{
  "agents": ["LocationAgriAssistant"],
  "justifications": [
    "LocationAgriAssistant is selected to analyze and recommend the best logistics routes for transporting tomatoes from Nashik to Mumbai."
  ]
}

Example 5:
Query: "Detect pests in my crop image and recommend treatment."
Output:
{
  "agents": ["PestPredictionAgent"],
  "justifications": [
    "PestPredictionAgent is selected to detect pests in the crop image and suggest appropriate treatments."
  ]
}

Example 6:
Query: "Translate this agricultural policy document from Hindi to English."
Output:
{
  "agents": ["TranslationAgent"],
  "justifications": [
    "TranslationAgent is selected to translate the agricultural policy document from Hindi to English."
  ]
}

Example 7:
Query: "Get the latest market prices for rice in Karnataka."
Output:
{
  "agents": ["MarketPriceAgent"],
  "justifications": [
    "MarketPriceAgent is selected to fetch the latest market prices for rice in Karnataka."
  ]
}

Example 8:
Query: "Assess the risk profile for cotton farming in Gujarat."
Output:
{
  "agents": ["RiskManagementAgent"],
  "justifications": [
    "RiskManagementAgent is selected to assess the risk profile for cotton farming in Gujarat."
  ]
}

Example 9:
Query: "Predict crop yield for maize in Bihar for the upcoming season."
Output:
{
  "agents": ["CropYieldAgent"],
  "justifications": [
    "CropYieldAgent is selected to predict crop yield for maize in Bihar for the upcoming season."
  ]
}

Example 10:
Query: "Detect diseases in my crop leaf image."
Output:
{
  "agents": ["CropDiseaseDetectionAgent"],
  "justifications": [
    "CropDiseaseDetectionAgent is selected to detect diseases in the crop leaf image."
  ]
}

Example 11:
Query: "Recommend the best fertilizer for tomato farming in Andhra Pradesh."
Output:
{
  "agents": ["FertilizerRecommendationAgent"],
  "justifications": [
    "FertilizerRecommendationAgent is selected to recommend the best fertilizer for tomato farming in Andhra Pradesh."
  ]
}

OUTPUT FORMAT:
- Return a RoutingDecision object with two lists: agents and justifications.
- Do not mention tool calling or internal implementation details.
- Be concise, logical, and ensure the output is easy to parse and use for downstream agent invocation.
"""
        )

    def route(self, query: str) -> RoutingDecision:
        prompt = (
            f"Given the following user query, decide which agents should handle it and provide justifications:\n"
            f"Query: \"{query}\"\n"
            "Return a RoutingDecision object with lists of agent names and justifications."
        )
        result = self.agent.run(prompt).content
        return result


if __name__ == "__main__":
    router = RouterAgent()
    query = "Give me the latest weather forecast for wheat farming in Punjab and recommend the best crops for the upcoming season."
    routing_decision = router.route(query)
    print(routing_decision)
