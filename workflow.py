import os
import sys
from typing import Dict, Any, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END, START

from RAG.workflow import Workflow
from RAG.parallel_rag_main import ParallelRAGSystem
from Agents.Router import RouterAgent
from Agents.Crop_Recommender.agent import CropRecommenderAgent
from Agents.Weather_forcast.agent import WeatherForecastAgent
from Agents.Location_Information.agent import LocationAgriAssistant
from Agents.News.agent import NewsAgent
from Agents.Credit_Policy_Market.agent import CreditPolicyMarketAgent
from Agents.answer_grader import AnswerGraderAgent
from Agents.synthesizer_agent import SynthesizerAgent
from Agents.Crop_Disease.agent import CropDiseaseAgent
from Agents.fact_checker.fscorer import LikertScorer
from Agents.Image_Analysis.agent import ImageAgent
from Agents.Market_Price.agent import MarketPriceAgent
from Agents.Multi_Lingual.agent import MultiLanguageTranslator
from Agents.Pest_prediction.agent import PestPredictionAgent
from Agents.Risk_Management.agent import AgriculturalRiskAnalysisAgent
from Agents.Web_Scrapping.agent import AgriculturalWebScrappingAgent
from Agents.Query_rewriter import QueryRewriterAgent

crop_recommender_agent = CropRecommenderAgent()
weather_forecast_agent = WeatherForecastAgent()
location_agri_assistant = LocationAgriAssistant()
news_agent = NewsAgent()
credit_policy_market_agent = CreditPolicyMarketAgent()
answer_grader_agent = AnswerGraderAgent()
synthesizer_agent = SynthesizerAgent()
crop_disease_agent = CropDiseaseAgent()
fact_checker_agent = LikertScorer()
image_analysis_agent = ImageAgent()
market_price_agent = MarketPriceAgent()
multi_language_translator_agent = MultiLanguageTranslator()
pest_prediction_agent = PestPredictionAgent()
risk_management_agent = AgriculturalRiskAnalysisAgent()
web_scraping_agent = AgriculturalWebScrappingAgent()
query_rewriter_agent = QueryRewriterAgent()

class MainWorkflowState(TypedDict):
    query: str
    initial_mode: str
    current_mode: str
    rag_response: str
    router_result: Dict[str, Any]
    agent_responses: Dict[str, Any]
    fact_checked_responses: Dict[str, Any]
    synthesized_result: str
    grading: Any
    answer_grade: Any
    rewritten_query: str
    generation: str
    extractions: str
    documents: List[str]
    has_switched_mode: bool

def run_adaptive_rag(query: str) -> str:
    rag_system = ParallelRAGSystem(model="gemini-2.0-flash", k=3)
    result = rag_system.process_query(query)
    return result.get("synthesized_answer", "")

def run_router_agent(query: str) -> Dict[str, Any]:
    router = RouterAgent()
    routing_decision = router.route(query)
    if hasattr(routing_decision, 'agents'):
        agents = routing_decision.agents
    elif isinstance(routing_decision, dict):
        agents = routing_decision.get('agents', [])
    else:
        agents = []
    return {"agents": agents, "routing_decision": routing_decision}

def call_agent(agent_name: str, query: str) -> Any:
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
    elif agent_name == "AnswerGraderAgent":
        return answer_grader_agent.grade(query.get("question", ""), query.get("answer", ""))
    elif agent_name == "SynthesizerAgent":
        return synthesizer_agent.synthesize(query.get("responses", []))
    elif agent_name == "CropDiseaseDetectionAgent":
        return crop_disease_agent.detect(query)
    elif agent_name == "FactCheckerAgent":
        return fact_checker_agent.score(query)
    elif agent_name == "ImageAnalysisAgent":
        return image_analysis_agent.analyze(query)
    elif agent_name == "MarketPriceAgent":
        return market_price_agent.get_price(query)
    elif agent_name == "MultiLanguageTranslatorAgent":
        return multi_language_translator_agent.translate_robust(query)
    elif agent_name == "PestPredictionAgent":
        return pest_prediction_agent.detect(query)
    elif agent_name == "RiskManagementAgent":
        return risk_management_agent.assess(query)
    elif agent_name == "WebScrapingAgent":
        return web_scraping_agent.scrape(query)
    else:
        return f"No implementation for agent: {agent_name}"

def fact_check_response(response: str, query: str) -> Dict[str, Any]:
    """Fact checking focused ONLY on factual accuracy"""
    try:
        fact_check_result, tokens = fact_checker_agent.score_text(response, query)
        return {
            "original_response": response,
            "fact_check_result": fact_check_result,
            "tokens_used": tokens,
            "is_factually_accurate": fact_check_result.score >= 4,  # Only check facts
            "factual_confidence_score": fact_check_result.score
        }
    except Exception as e:
        return {
            "original_response": response,
            "fact_check_result": None,
            "tokens_used": 0,
            "is_factually_accurate": False,
            "factual_confidence_score": 0,
            "error": str(e)
        }

def call_agent_with_fact_check(agent_name: str, query: str) -> Dict[str, Any]:
    """Call agent and fact-check ONLY the factual accuracy of its response"""
    try:
        # Get agent response
        agent_response = call_agent(agent_name, query)
        
        # Fact-check ONLY for factual accuracy
        fact_check_result = fact_check_response(str(agent_response), query)
        
        return {
            "agent_name": agent_name,
            "response": agent_response,
            "fact_check": fact_check_result,
            "is_factually_reliable": fact_check_result.get("is_factually_accurate", False),
            "factual_confidence": fact_check_result.get("factual_confidence_score", 0)
        }
    except Exception as e:
        return {
            "agent_name": agent_name,
            "response": f"Error: {str(e)}",
            "fact_check": {
                "is_factually_accurate": False,
                "factual_confidence_score": 0,
                "error": str(e)
            },
            "is_factually_reliable": False,
            "factual_confidence": 0
        }

def grade_answer(question: str, answer: str) -> Dict[str, Any]:
    """Grade the synthesized answer for completeness and relevance (separate from fact checking)"""
    try:
        grade_result = answer_grader_agent.grade(question, answer)
        return {
            "grade": grade_result,
            "is_good_answer": getattr(grade_result, 'binary_score', 'no') == 'yes',
            "reasoning": getattr(grade_result, 'reasoning', ''),
            "score": getattr(grade_result, 'score', 0)
        }
    except Exception as e:
        return {
            "grade": None,
            "is_good_answer": False,
            "reasoning": f"Grading error: {str(e)}",
            "score": 0,
            "error": str(e)
        }

def rag_node(state: MainWorkflowState):
    rag_response = run_adaptive_rag(state["query"])
    documents = []
    extractions = ""
    if isinstance(rag_response, dict):
        documents = rag_response.get("documents", [])
        extractions = rag_response.get("extractions", "")
        generation = rag_response.get("generation", "")
    else:
        generation = rag_response
    
    # Fact-check ONLY for factual accuracy
    fact_check_result = fact_check_response(str(rag_response), state["query"])
    
    return {
        "rag_response": rag_response,
        "synthesized_result": rag_response,
        "documents": documents,
        "extractions": extractions,
        "generation": generation,
        "current_mode": "rag",
        "fact_checked_responses": {"rag_response": fact_check_result}
    }

def router_node(state: MainWorkflowState):
    router_result = run_router_agent(state["query"])
    return {
        "router_result": router_result,
        "current_mode": "tooling"
    }

def agent_calls_node(state: MainWorkflowState):
    """Enhanced agent calls with parallel fact-checking (ONLY for facts)"""
    agent_responses = {}
    fact_checked_responses = {}
    agents = state["router_result"].get("agents", [])
    
    with ThreadPoolExecutor() as executor:
        # Submit all agent calls with fact-checking
        futures = {
            executor.submit(call_agent_with_fact_check, agent, state["query"]): agent 
            for agent in agents
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                agent_responses[agent_name] = result["response"]
                fact_checked_responses[agent_name] = result["fact_check"]
                
                # Log ONLY factual reliability information
                print(f"Agent {agent_name}: Factually Reliable = {result['is_factually_reliable']}, "
                      f"Factual Confidence = {result['factual_confidence']:.2f}")
                
            except Exception as e:
                agent_responses[agent_name] = f"Error: {str(e)}"
                fact_checked_responses[agent_name] = {
                    "is_factually_accurate": False,
                    "factual_confidence_score": 0,
                    "error": str(e)
                }
    
    return {
        "agent_responses": agent_responses,
        "fact_checked_responses": fact_checked_responses
    }

def synthesize_tooling_node(state: MainWorkflowState):
    """Enhanced synthesis considering ONLY factual accuracy"""
    # Filter responses based ONLY on factual accuracy
    factually_accurate_responses = []
    all_responses = []
    
    for agent_name, response in state["agent_responses"].items():
        fact_check = state["fact_checked_responses"].get(agent_name, {})
        all_responses.append(response)
        
        # Include response ONLY if it's factually accurate
        if fact_check.get("is_factually_accurate", False):
            factually_accurate_responses.append(response)
    
    # Use factually accurate responses if available, otherwise use all responses
    responses_to_synthesize = factually_accurate_responses if factually_accurate_responses else all_responses
    
    synthesized_result = synthesizer_agent.synthesize(responses_to_synthesize)
    
    return {
        "synthesized_result": synthesized_result,
        "factually_accurate_responses_count": len(factually_accurate_responses),
        "total_responses_count": len(all_responses)
    }

def grading_node(state: MainWorkflowState):
    """Separate fact checking (facts only) and answer grading (completeness/relevance)"""
    # Fact-check ONLY for factual accuracy
    fact_check_result = fact_check_response(str(state["synthesized_result"]), state["query"])
    
    # Grade for answer quality (completeness, relevance, etc.) - separate from fact checking
    answer_grade_result = grade_answer(state["query"], str(state["synthesized_result"]))
    
    return {
        "grading": fact_check_result,
        "answer_grade": answer_grade_result
    }

def mode_decision_edge(state: MainWorkflowState):
    """Decision logic based on factual accuracy and answer quality separately"""
    grading = state["grading"]
    answer_grade = state["answer_grade"]
    current_mode = state["current_mode"]
    has_switched_mode = state.get("has_switched_mode", False)
    
    # Check factual accuracy separately from answer quality
    is_factually_accurate = grading.get("is_factually_accurate", False)
    is_answer_complete = answer_grade.get("is_good_answer", False)
    
    print(f"Quality Assessment - Factually Accurate: {is_factually_accurate}, "
          f"Answer Complete: {is_answer_complete}")
    
    if is_factually_accurate or is_answer_complete:
        return "end"
    
    if has_switched_mode:
        return "query_rewrite"
    
    if current_mode == "tooling" and not is_factually_accurate and not is_answer_complete:
        return "switch_to_rag"
    elif current_mode == "rag" and not is_factually_accurate and not is_answer_complete:
        return "switch_to_tooling"
    
    return "query_rewrite"

def switch_to_rag_node(state: MainWorkflowState):
    rag_response = run_adaptive_rag(state["query"])
    documents = []
    extractions = ""
    if isinstance(rag_response, dict):
        documents = rag_response.get("documents", [])
        extractions = rag_response.get("extractions", "")
        generation = rag_response.get("generation", "")
    else:
        generation = rag_response
    
    # Fact-check ONLY for factual accuracy
    fact_check_result = fact_check_response(str(rag_response), state["query"])
    
    return {
        "rag_response": rag_response,
        "synthesized_result": rag_response,
        "documents": documents,
        "extractions": extractions,
        "generation": generation,
        "current_mode": "rag",
        "has_switched_mode": True,
        "fact_checked_responses": {"rag_response": fact_check_result}
    }

def switch_to_tooling_node(state: MainWorkflowState):
    router_result = run_router_agent(state["query"])
    return {
        "router_result": router_result,
        "current_mode": "tooling",
        "has_switched_mode": True
    }

def query_rewrite_node(state: MainWorkflowState):
    rewritten_query = query_rewriter_agent.rewrite(state["query"], state["synthesized_result"])
    return {
        "rewritten_query": rewritten_query,
        "query": rewritten_query,
        "has_switched_mode": False
    }

def start_routing_edge(state: MainWorkflowState):
    initial_mode = state["initial_mode"]
    if initial_mode == "rag":
        return "rag"
    else:
        return "router"

def build_hybrid_workflow_graph():
    graph = StateGraph(MainWorkflowState)
    
    graph.add_node("rag", rag_node)
    graph.add_node("router", router_node)
    graph.add_node("agent_calls", agent_calls_node)
    graph.add_node("synthesize_tooling", synthesize_tooling_node)
    graph.add_node("grading", grading_node)
    graph.add_node("switch_to_rag", switch_to_rag_node)
    graph.add_node("switch_to_tooling", switch_to_tooling_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    
    graph.add_conditional_edges(START, start_routing_edge, {"rag": "rag", "router": "router"})
    
    graph.add_edge("rag", "grading")
    graph.add_edge("router", "agent_calls")
    graph.add_edge("agent_calls", "synthesize_tooling")
    graph.add_edge("synthesize_tooling", "grading")
    
    graph.add_conditional_edges(
        "grading", 
        mode_decision_edge, 
        {
            "end": END, 
            "switch_to_rag": "switch_to_rag",
            "switch_to_tooling": "switch_to_tooling",
            "query_rewrite": "query_rewrite"
        }
    )
    
    graph.add_edge("switch_to_rag", "grading")
    graph.add_edge("switch_to_tooling", "agent_calls")
    graph.add_edge("query_rewrite", "router")
    
    return graph

hybrid_workflow_graph = build_hybrid_workflow_graph()
compiled_hybrid_graph = hybrid_workflow_graph.compile()

def run_workflow(query: str, mode: str = "rag") -> Dict[str, Any]:
    """Enhanced workflow execution with separated fact checking and answer grading"""
    state = MainWorkflowState(
        query=query,
        initial_mode=mode,
        current_mode=mode,
        rag_response="",
        router_result={},
        agent_responses={},
        fact_checked_responses={},
        synthesized_result="",
        grading=None,
        answer_grade=None,
        rewritten_query="",
        generation="",
        extractions="",
        documents=[],
        has_switched_mode=False
    )
    
    if mode.lower() not in ["rag", "tooling"]:
        raise ValueError("Mode must be either 'rag' or 'tooling'")
    
    final_state = compiled_hybrid_graph.invoke(state)
    
    # Return comprehensive results with clear separation
    return {
        "answer": final_state["synthesized_result"],
        "factual_accuracy_score": final_state.get("grading", {}).get("factual_confidence_score", 0),
        "answer_quality_grade": final_state.get("answer_grade", {}),
        "is_factually_accurate": final_state.get("grading", {}).get("is_factually_accurate", False),
        "is_answer_complete": final_state.get("answer_grade", {}).get("is_good_answer", False),
        "factually_accurate_responses_count": final_state.get("factually_accurate_responses_count", 0),
        "total_responses_count": final_state.get("total_responses_count", 0),
        "final_mode": final_state.get("current_mode", mode),
        "switched_modes": final_state.get("has_switched_mode", False)
    }

if __name__ == "__main__":
    import time
    questions = [
        "What are the best crops for Kharif season in Nashik and what is the weather forecast?",
        "Which fertilizer is recommended for rice in high rainfall regions?",
        "How can farmers manage pest outbreaks in cotton fields?",
        "What are the latest government policies for agricultural credit?",
        "Suggest sustainable practices for improving soil health in Maharashtra.",
        "What is the market price trend for wheat in India?",
        "How to prevent fungal diseases in tomato crops?",
        "What irrigation methods are best for sugarcane?",
        "Which crops are suitable for drought-prone areas?",
        "How to increase yield for maize in semi-arid regions?"
    ]
    
    mode = input("Select initial mode (rag/tooling): ").strip().lower()
    if mode not in ["rag", "tooling"]:
        print("Invalid mode. Using 'rag' as default.")
        mode = "rag"
    
    for idx, user_query in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}: {user_query}")
        print(f"Initial Mode: {mode.upper()}")
        print('='*80)
        
        start_time = time.time()
        result = run_workflow(user_query, mode)
        end_time = time.time()
        
        print(f"Answer: {result['answer']}")
        print(f"\nSeparated Quality Metrics:")
        print(f"  - Factual Accuracy Score: {result['factual_accuracy_score']:.2f}/5")
        print(f"  - Is Factually Accurate: {result['is_factually_accurate']}")
        print(f"  - Is Answer Complete: {result['is_answer_complete']}")
        print(f"  - Factually Accurate Responses: {result['factually_accurate_responses_count']}/{result['total_responses_count']}")
        print(f"  - Final Mode: {result['final_mode']}")
        print(f"  - Switched Modes: {result['switched_modes']}")
        print(f"  - Processing Time: {end_time - start_time:.2f}s")
        
        if result['answer_quality_grade'].get('reasoning'):
            print(f"  - Quality Grade Reasoning: {result['answer_quality_grade']['reasoning']}")