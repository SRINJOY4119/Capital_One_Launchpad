from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team
from agno.tools.arxiv import ArxivTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv
import json
from typing import Dict, List, Any, Optional
from textwrap import dedent
import asyncio

load_dotenv()

# Main 3 Agents
arxiv_agent = Agent(
    name="ArXiv Research Agent",
    role="Academic papers and scholarly research specialist",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ArxivTools()],
    instructions=dedent("""
    You are the ArXiv Research Agent specializing in academic papers and scholarly publications.
    
    YOUR ROLE:
    - Search and analyze ArXiv papers
    - Focus on recent research and developments
    - Provide detailed paper summaries
    - Identify key research trends
    
    IMPORTANT:
    - Remove all emojis from responses
    - Use professional academic tone
    - Focus on your ArXiv research expertise
    - Complete your research independently
    - Do not try to coordinate other agents
    """),
)

wikipedia_agent = Agent(
    name="Wikipedia Knowledge Agent", 
    role="Foundational knowledge and background information specialist",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[WikipediaTools()],
    instructions=dedent("""
    You are the Wikipedia Knowledge Agent specializing in foundational knowledge and context.
    
    YOUR ROLE:
    - Provide encyclopedic knowledge and definitions
    - Give historical context and background
    - Explain fundamental concepts
    - Verify factual information
    
    IMPORTANT:
    - Remove all emojis from responses
    - Use clear, informative tone
    - Focus on your Wikipedia knowledge expertise
    - Complete your research independently
    - Do not try to coordinate other agents
    """),
)

tavily_agent = Agent(
    name="Web Intelligence Agent",
    role="Current web information and market intelligence specialist", 
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[TavilyTools()],
    instructions=dedent("""
    You are the Web Intelligence Agent specializing in current web information and developments.
    
    YOUR ROLE:
    - Search current web information via Tavily
    - Focus on recent developments and trends
    - Gather market intelligence and news
    - Provide real-time insights
    
    IMPORTANT:
    - Remove all emojis from responses
    - Use professional analytical tone
    - Focus on your web research expertise
    - Complete your research independently
    - Do not try to coordinate other agents
    """),
)

# One Main Research Team
research_team = Team(
    name="Multi-Agent Research Team",
    mode="sequential",
    model=Groq(id="llama-3.3-70b-versatile"),
    members=[
        wikipedia_agent,      # First: Get foundational knowledge
        arxiv_agent,         # Second: Get academic research
        tavily_agent,        # Third: Get current developments
    ],
    instructions=[
        "You are a comprehensive research team with 3 specialized agents.",
        "Each agent focuses on their specialty area and provides complete analysis.",
        "Wikipedia Agent: Provide foundational knowledge and context first.",
        "ArXiv Agent: Then provide academic research and papers.",
        "Tavily Agent: Finally provide current web developments and trends.",
        "Remove all emojis and maintain professional tone.",
        "Each agent works independently - no coordination between agents.",
    ],
    show_tool_calls=False,
    markdown=True,
)

class ResearchCoordinator:
    """Simple coordinator with one team and three agents"""
    
    def __init__(self):
        self.team = research_team
        self.arxiv_agent = arxiv_agent
        self.wikipedia_agent = wikipedia_agent 
        self.tavily_agent = tavily_agent
        
    def search(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Main research method using the team"""
        print("🔍 Starting research with Multi-Agent Research Team...")
        
        try:
            # Prepare the research query
            research_query = self._prepare_query(query, context)
            
            # Run the team research
            response = self.team.run(research_query)
            
            return {
                "success": True,
                "content": response.content if hasattr(response, 'content') else str(response),
                "query": query,
                "team_name": "Multi-Agent Research Team",
                "agents_used": ["Wikipedia Agent", "ArXiv Agent", "Tavily Agent"],
                "approach": "sequential_team_research"
            }
            
        except Exception as e:
            print(f"❌ Team research failed: {str(e)}")
            return self._fallback_search(query, context, str(e))
    
    def search_individual(self, query: str, agent_type: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Search using individual agent (fallback or specific needs)"""
        agents = {
            "arxiv": self.arxiv_agent,
            "wikipedia": self.wikipedia_agent,
            "tavily": self.tavily_agent
        }
        
        if agent_type not in agents:
            return {
                "success": False,
                "error": f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}",
                "query": query
            }
        
        print(f"🔍 Starting research with {agent_type.title()} Agent...")
        
        try:
            agent = agents[agent_type]
            research_query = self._prepare_query(query, context)
            response = agent.run(research_query)
            
            return {
                "success": True,
                "content": response.content if hasattr(response, 'content') else str(response),
                "query": query,
                "agent_name": agent.name,
                "agent_type": agent_type,
                "approach": "individual_agent_research"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "agent_type": agent_type
            }
    
    def _prepare_query(self, query: str, context: Optional[str] = None) -> str:
        """Prepare the research query"""
        research_query = f"Research Topic: {query}\n\n"
        
        if context:
            research_query += f"Additional Context: {context}\n\n"
            
        research_query += """Instructions:
- Conduct thorough research on this topic using your specialized tools
- Provide comprehensive analysis and insights
- Include relevant sources and evidence
- Focus on accuracy and depth
- Use professional tone without emojis
- Work independently and complete your analysis"""
        
        return research_query
    
    def _fallback_search(self, query: str, context: Optional[str], error: str) -> Dict[str, Any]:
        """Fallback to individual agents if team fails"""
        print("🔄 Team failed, trying fallback with individual agents...")
        
        results = []
        
        # Try each agent individually
        for agent_type in ["wikipedia", "arxiv", "tavily"]:
            result = self.search_individual(query, agent_type, context)
            if result["success"]:
                results.append(f"**{agent_type.title()} Agent Results:**\n{result['content']}\n")
        
        if results:
            combined_content = "\n".join(results)
            return {
                "success": True,
                "content": combined_content,
                "query": query,
                "team_name": "Fallback Individual Agents",
                "agents_used": ["Wikipedia Agent", "ArXiv Agent", "Tavily Agent"],
                "approach": "fallback_individual_research",
                "original_error": error
            }
        else:
            return {
                "success": False,
                "error": f"Both team and individual agents failed. Original error: {error}",
                "query": query
            }
    


# Example usage
if __name__ == "__main__":
    # Initialize coordinator
    coordinator = ResearchCoordinator()
    
    # Research query
    query = "sustainable agriculture AI technologies 2024"
    
    print("Starting research...")
    
    # Perform research
    result = coordinator.search(query)
    
    if result["success"]:
        print("Research completed successfully!")
        print("\nResults:")
        print("-" * 50)
        print(result['content'])
        
    else:
        print("Research failed:")
        print(f"Error: {result['error']}")
    
    print("\nDone!")