import os
import sys
from typing import Dict, Any, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END, START

from Agents.Router import RouterAgent
from Agents.Crop_Recommender.agent import CropRecommenderAgent
from Agents.Weather_forcast.agent import WeatherForecastAgent
from Agents.Location_Information.agent import LocationAgriAssistant
from Agents.News.agent import NewsAgent
from Agents.Credit_Policy_Market.agent import CreditPolicyMarketAgent
from Agents.synthesizer_agent import SynthesizerAgent
from Agents.Crop_Disease.agent import CropDiseaseAgent
from Agents.Image_Analysis.agent import ImageAgent
from Agents.Market_Price.agent import MarketPriceAgent
from Agents.Multi_Lingual.agent import MultiLanguageTranslator
from Agents.Pest_prediction.agent import PestPredictionAgent
from Agents.Risk_Management.agent import AgriculturalRiskAnalysisAgent
from Agents.Web_Scrapping.agent import AgriculturalWebScrappingAgent
from Agents.Crop_Yield.agent import CropYieldAssistant
from Agents.Fertilizer_Recommender.agent import FertilizerRecommendationAgent
from utils.Internet_checker import InternetChecker
from utils.hf_model import HFModel

internet_checker = InternetChecker()
base_model_dir = "./models/Qwen1.5-Base"
adapter_dir = "./models/Qwen_1.5_Finetuned"
hf_model = None

if not internet_checker.is_connected():
    print("Offline mode detected. Using HF Model for inference.")
    hf_model = HFModel(base_model_dir, adapter_dir)
else: 
    print("Online mode detected")

# Initialize all agents
crop_recommender_agent = CropRecommenderAgent()
weather_forecast_agent = WeatherForecastAgent()
location_agri_assistant = LocationAgriAssistant()
news_agent = NewsAgent()
crop_yield_assistant = CropYieldAssistant()
credit_policy_market_agent = CreditPolicyMarketAgent()
synthesizer_agent = SynthesizerAgent()
crop_disease_agent = CropDiseaseAgent()
image_analysis_agent = ImageAgent()
market_price_agent = MarketPriceAgent()
multi_language_translator_agent = MultiLanguageTranslator()
pest_prediction_agent = PestPredictionAgent()
risk_management_agent = AgriculturalRiskAnalysisAgent()
web_scraping_agent = AgriculturalWebScrappingAgent()
fertilizer_recommender_agent = FertilizerRecommendationAgent()

class WorkflowState(TypedDict):
    query: str
    image_path: str
    router_result: Dict[str, Any]
    agent_responses: Dict[str, Any]
    synthesized_result: str

def run_router_agent(query: str, image_path: str = None) -> Dict[str, Any]:
    router = RouterAgent()
    if image_path:
        query_with_image = f"{query} [IMAGE_PROVIDED]"
        routing_decision = router.route(query_with_image)
    else:
        routing_decision = router.route(query)
    
    if hasattr(routing_decision, 'agents'):
        agents = routing_decision.agents
    elif isinstance(routing_decision, dict):
        agents = routing_decision.get('agents', [])
    else:
        agents = []
    return {"agents": agents, "routing_decision": routing_decision}

def call_agent(agent_name: str, query: str, image_path: str = None) -> Any:
    if agent_name == "CropRecommenderAgent":
        return crop_recommender_agent.respond(query)
    elif agent_name == "WeatherForecastAgent":
        return weather_forecast_agent.get_weather_analysis(query)
    elif agent_name == "LocationAgriAssistant":
        return location_agri_assistant.respond(query)
    elif agent_name == "NewsAgent":
        return news_agent.get_agri_news(query)
    elif agent_name == "CreditPolicyMarketAgent":
        return credit_policy_market_agent.respond_to_query(query)
    elif agent_name == "CropDiseaseDetectionAgent":
        return crop_disease_agent.analyze_disease(query=query, image_path=image_path)
    elif agent_name == "ImageAnalysisAgent":
        return image_analysis_agent.describe_image(image_path)
    elif agent_name == "MarketPriceAgent":
        return market_price_agent.chat(query)
    elif agent_name == "MultiLanguageTranslatorAgent":
        return multi_language_translator_agent.translate_robust(query)
    elif agent_name == "PestPredictionAgent":
        return pest_prediction_agent.respond(query)
    elif agent_name == "RiskManagementAgent":
        return risk_management_agent.assess(query)
    elif agent_name == "WebScrapingAgent":
        return web_scraping_agent.scrape(query)
    elif agent_name == "CropYieldAgent":
        return crop_yield_assistant.respond(query)
    elif agent_name == "FertilizerRecommenderAgent":
        return fertilizer_recommender_agent.recommend_fertilizer(query)
    else:
        return f"No implementation for agent: {agent_name}"

def call_agent_simple(agent_name: str, query: str, image_path: str = None) -> Dict[str, Any]:
    try:
        agent_response = call_agent(agent_name, query, image_path)
        return {
            "agent_name": agent_name,
            "response": agent_response,
        }
    except Exception as e:
        return {
            "agent_name": agent_name,
            "response": f"Error: {str(e)}",
        }

def router_node(state: WorkflowState):
    router_result = run_router_agent(state["query"], state.get("image_path"))
    return {
        "router_result": router_result
    }

def agent_calls_node(state: WorkflowState):
    agent_responses = {}
    agents = state["router_result"].get("agents", [])
    
    print(f"Routing to agents: {agents}")
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(call_agent_simple, agent, state["query"], state.get("image_path")): agent 
            for agent in agents
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                agent_responses[agent_name] = result["response"]
                print(f"Agent {agent_name} completed successfully")
                
            except Exception as e:
                agent_responses[agent_name] = f"Error: {str(e)}"
                print(f"Agent {agent_name} Error: {str(e)}")
    
    return {
        "agent_responses": agent_responses
    }

def synthesize_node(state: WorkflowState):
    all_responses = []
    
    for agent_name, response in state["agent_responses"].items():
        all_responses.append(response)
    
    print("Synthesizing responses from all agents...")
    synthesized_result = synthesizer_agent.synthesize(all_responses)
    
    return {
        "synthesized_result": synthesized_result
    }

def build_workflow_graph():
    graph = StateGraph(WorkflowState)
    
    graph.add_node("router", router_node)
    graph.add_node("agent_calls", agent_calls_node)
    graph.add_node("synthesize", synthesize_node)
    
    graph.add_edge(START, "router")
    graph.add_edge("router", "agent_calls")
    graph.add_edge("agent_calls", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph

workflow_graph = build_workflow_graph()
compiled_graph = workflow_graph.compile()

def run_workflow(query: str, image_path: str = None) -> Dict[str, Any]:
    if not internet_checker.is_connected() and hf_model:
        hf_response = hf_model.infer(query)
        return {
            "answer": hf_response,
            "mode": "offline"
        }
    
    state = WorkflowState(
        query=query,
        image_path=image_path or "",
        router_result={},
        agent_responses={},
        synthesized_result=""
    )
    
    final_state = compiled_graph.invoke(state)
    
    return {
        "answer": final_state["synthesized_result"],
        "agent_responses": final_state["agent_responses"],
        "routed_agents": final_state["router_result"].get("agents", [])
    }

if __name__ == "__main__":
    import time
    
    questions = [
        "Estimate crop yield for wheat in Punjab in winter of 2025",
        "How can farmers manage pest outbreaks in cotton fields?",
        "What is the market price trend for wheat in India?",
        "How to prevent fungal diseases in tomato crops?",
    ]
    
    image_queries = [
        ("Analyze this crop disease", "Images/Crop/crop_disease.jpg"),
        ("Check for pests in this image", "Images/Pests/jpg_0.jpg"),
    ]
    
    query_type = input("Test type (text/image): ").strip().lower()
    
    if query_type == "image":
        test_queries = image_queries
    else:
        test_queries = [(q, None) for q in questions]
    
    for idx, (user_query, image_path) in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}: {user_query}")
        if image_path:
            print(f"Image Path: {image_path}")
        print('='*80)
        
        start_time = time.time()
        result = run_workflow(user_query, image_path)
        end_time = time.time()
        
        print(f"\nRouted to agents: {result['routed_agents']}")
        print(f"\nFinal Answer: {result['answer']}")
        print(f"\nProcessing Time: {end_time - start_time:.2f}s")
        
        