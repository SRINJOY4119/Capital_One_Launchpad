from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.arxiv import ArxivTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv
import json
from typing import Dict, List, Any

load_dotenv()

class SubsearchAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[ArxivTools(), WikipediaTools(), TavilyTools()],
            instructions=self._get_instructions()
        )
    
    def _get_instructions(self):
        return """You are a specialized subsearch agent for deep research pipelines. Your role is to perform comprehensive, multi-source information gathering and analysis.

TOOL USAGE STRATEGY:
- ArxivTools: Use for academic papers, research studies, scientific publications, technical documentation
- WikipediaTools: Use for general knowledge, definitions, historical context, basic factual information
- TavilyTools: Use for comprehensive web search, business information, latest developments, specialized queries, current events, and diverse web sources

RESEARCH METHODOLOGY:
1. Start with Wikipedia for foundational knowledge and context
2. Use Arxiv for peer-reviewed academic sources and technical depth
3. Leverage Tavily for comprehensive web analysis, current news, and business intelligence
4. Cross-reference findings across all sources for accuracy validation

OUTPUT REQUIREMENTS:
- Provide detailed, well-structured responses with source citations
- Include confidence levels for each claim
- Highlight conflicting information from different sources
- Synthesize information from multiple tools into coherent insights
- Always verify facts across at least 2 different source types

SEARCH PATTERNS:
- For technical topics: Wikipedia → Arxiv → Tavily
- For current events: Tavily → Wikipedia → Arxiv
- For business/market research: Tavily → Wikipedia → Arxiv
- For academic research: Arxiv → Wikipedia → Tavily"""

    def search_academic(self, query: str) -> Dict[str, Any]:
        prompt = f"""Conduct an academic research search on: {query}

SEARCH SEQUENCE:
1. Search Wikipedia for background context and definitions
2. Search Arxiv for peer-reviewed papers and technical studies
3. Search Tavily for recent academic developments and citations

Provide a comprehensive academic analysis with proper citations."""
        
        return self._execute_search(prompt)

    def search_current_events(self, query: str) -> Dict[str, Any]:
        prompt = f"""Research current events and recent developments about: {query}

SEARCH SEQUENCE:
1. Search Tavily for comprehensive recent coverage and current news
2. Search Wikipedia for background context
3. Search Arxiv for any related research studies

Focus on timeline, recent developments, and current status."""
        
        return self._execute_search(prompt)

    def search_business_intelligence(self, query: str) -> Dict[str, Any]:
        prompt = f"""Conduct business intelligence research on: {query}

SEARCH SEQUENCE:
1. Search Tavily for business news, market analysis, company information, and diverse business sources
2. Search Wikipedia for company history and industry background
3. Search Arxiv for relevant business research and case studies

Provide market insights, competitive analysis, and business trends."""
        
        return self._execute_search(prompt)

    def search_comprehensive(self, query: str) -> Dict[str, Any]:
        prompt = f"""Perform comprehensive multi-source research on: {query}

SEARCH SEQUENCE:
1. Search Wikipedia for foundational knowledge and context
2. Search Arxiv for academic and technical sources
3. Search Tavily for comprehensive web analysis, current information, and verification

Synthesize all sources into a complete research report with cross-validation."""
        
        return self._execute_search(prompt)

    def search_technical_deep_dive(self, query: str) -> Dict[str, Any]:
        prompt = f"""Conduct technical deep-dive research on: {query}

SEARCH SEQUENCE:
1. Search Arxiv for latest research papers and technical documentation
2. Search Wikipedia for technical definitions and foundational concepts
3. Search Tavily for industry applications, technical implementations, and technical forums

Focus on technical accuracy, implementation details, and expert opinions."""
        
        return self._execute_search(prompt)

    def _execute_search(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.agent.run(prompt)
            return {
                "success": True,
                "content": response.content,
                "metadata": {
                    "sources_used": ["arxiv", "wikipedia", "tavily"],
                    "search_type": "multi_source",
                    "confidence": "cross_validated"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_content": "Search execution failed"
            }

    def custom_search(self, query: str, tool_sequence: List[str]) -> Dict[str, Any]:
        tool_map = {
            "arxiv": "ArxivTools",
            "wikipedia": "WikipediaTools", 
            "tavily": "TavilyTools"
        }
        
        sequence_instruction = " → ".join([tool_map[tool] for tool in tool_sequence])
        
        prompt = f"""Execute custom search sequence for: {query}

CUSTOM SEARCH SEQUENCE: {sequence_instruction}

Follow the exact tool sequence provided and synthesize results comprehensively."""
        
        return self._execute_search(prompt)




if __name__ == "__main__":
    agent = SubsearchAgent()
    research_query = "sustainable agriculture AI technologies 2024"
    result = agent.search_comprehensive(research_query)
    print(result["content"])
    