import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Task(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    agricultural_domain: str = Field(..., description="Agricultural domain (soil, crop, climate, etc.)")
    assigned_agent: str = Field(None, description="Assigned agricultural specialist")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    subsearch_agents: int = Field(default=1, description="Number of subsearch agents required")
    thought_process: List[str] = Field(default_factory=list, description="Chain of thought reasoning")

class ResearchPlan(BaseModel):
    plan_id: str = Field(..., description="Plan identifier")
    title: str = Field(..., description="Research title")
    objective: str = Field(..., description="Research objective")
    tasks: List[Task] = Field(..., description="List of tasks")
    agent_assignments: Dict[str, List[str]] = Field(..., description="Agent to tasks mapping")
    total_subsearch_agents: int = Field(default=0, description="Total number of subsearch agents")
    reasoning_chain: List[str] = Field(default_factory=list, description="Plan creation thought process")

class AgriculturalPlanningAgent:
    def __init__(self):
        load_dotenv()
        
        # Define available agricultural specialists
        self.available_agents = {
            "soil_scientist": "Soil health, nutrients, pH, fertilizers, soil testing",
            "crop_agronomist": "Crop selection, planting, growth monitoring, pest control",
            "field_researcher": "Field trials, experimental design, data collection",
            "data_analyst": "Agricultural data analysis, statistics, yield modeling",
            "climate_specialist": "Weather patterns, climate adaptation, seasonal planning",
            "sustainability_expert": "Organic farming, sustainable practices, certification"
        }
        
        # Initialize the planning agent
        self.planner = Agent(
            name="Agricultural Planning Agent",
            model=Gemini(id="gemini-2.0-flash"),
            response_model=ResearchPlan,
            use_json_mode=True,
            instructions=self._get_planning_instructions(),
        )
        
        self.total_subsearch_agents = 0

    def _get_planning_instructions(self) -> str:
        return """
        You are an Agricultural Research Planning Agent using chain-of-thought reasoning. Your job is to:
        
        1. Analyze the research objective
        2. Break down into logical task components
        3. Consider domain relationships and dependencies
        4. Assign specialists based on expertise matching
        5. Determine required number of subsearch agents per task
        
        Available specialists:
        - soil_scientist: Soil health, nutrients, pH, fertilizers, soil testing
        - crop_agronomist: Crop selection, planting, growth monitoring, pest control  
        - field_researcher: Field trials, experimental design, data collection
        - data_analyst: Agricultural data analysis, statistics, yield modeling
        - climate_specialist: Weather patterns, climate adaptation, seasonal planning
        - sustainability_expert: Organic farming, sustainable practices, certification
        
        For each task:
        1. Document thought process
        2. Determine agricultural domain
        3. Match with best specialist
        4. Identify task dependencies
        5. Calculate required subsearch agents
        """

    def _determine_domain(self, task_description: str) -> str:
        """Determine agricultural domain from task description"""
        desc = task_description.lower()
        
        if any(word in desc for word in ['soil', 'nutrient', 'fertilizer', 'ph']):
            return "soil"
        elif any(word in desc for word in ['crop', 'plant', 'seed', 'harvest', 'yield']):
            return "crop"
        elif any(word in desc for word in ['field', 'trial', 'experiment', 'test']):
            return "field"
        elif any(word in desc for word in ['data', 'analysis', 'model', 'statistics']):
            return "data"
        elif any(word in desc for word in ['climate', 'weather', 'season', 'temperature']):
            return "climate"
        elif any(word in desc for word in ['organic', 'sustainable', 'environment']):
            return "sustainability"
        else:
            return "general"

    def _assign_best_agent(self, domain: str, task_name: str) -> str:
        """Assign the best agent based on domain and task type"""
        # Primary assignment based on domain
        domain_mapping = {
            "soil": "soil_scientist",
            "crop": "crop_agronomist", 
            "field": "field_researcher",
            "data": "data_analyst",
            "climate": "climate_specialist",
            "sustainability": "sustainability_expert"
        }
        
        # Secondary assignment based on task type
        task_lower = task_name.lower()
        if "analysis" in task_lower or "data" in task_lower:
            return "data_analyst"
        elif "field" in task_lower or "trial" in task_lower:
            return "field_researcher"
        
        return domain_mapping.get(domain, "crop_agronomist")

    def _analyze_task_complexity(self, task_description: str) -> int:
        """Determine number of subsearch agents needed based on task complexity"""
        complexity_indicators = {
            'analyze': 2,
            'research': 2,
            'investigate': 2,
            'compare': 2,
            'evaluate': 3,
            'synthesize': 3,
            'integrate': 3,
            'optimize': 3
        }
        
        base_agents = 1
        for indicator, value in complexity_indicators.items():
            if indicator in task_description.lower():
                base_agents = max(base_agents, value)
        return base_agents

    def create_tasks(self, objective: str) -> List[Task]:
        """Break down objective into tasks using chain of thought"""
        thought_process = [
            f"Analyzing objective: {objective}",
            "Identifying key research components",
            "Determining task breakdown structure",
            "Evaluating domain relationships"
        ]
        
        base_tasks = [
            {
                "name": "Initial Analysis",
                "description": f"Analyze requirements and context for: {objective}",
                "thought": ["Consider scope", "Identify stakeholders", "Define boundaries"]
            },
            {
                "name": "Research Design",
                "description": f"Design research methodology for: {objective}",
                "thought": ["Review methods", "Select approaches", "Plan implementation"]
            },
            {
                "name": "Data Collection",
                "description": f"Gather and organize data for: {objective}",
                "thought": ["Define data needs", "Plan collection", "Ensure quality"]
            },
            {
                "name": "Analysis & Synthesis",
                "description": f"Analyze findings and synthesize results for: {objective}",
                "thought": ["Process data", "Identify patterns", "Draw conclusions"]
            }
        ]
        
        tasks = []
        for i, task_info in enumerate(base_tasks):
            domain = self._determine_domain(task_info["description"])
            agent = self._assign_best_agent(domain, task_info["name"])
            subsearch_count = self._analyze_task_complexity(task_info["description"])
            
            self.total_subsearch_agents += subsearch_count
            
            task = Task(
                task_id=f"T{i+1}",
                name=task_info["name"],
                description=task_info["description"],
                agricultural_domain=domain,
                assigned_agent=agent,
                dependencies=[f"T{i}"] if i > 0 else [],
                subsearch_agents=subsearch_count,
                thought_process=task_info["thought"]
            )
            tasks.append(task)
            
        return tasks

    def create_agent_assignments(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Create agent to tasks mapping"""
        assignments = {agent: [] for agent in self.available_agents.keys()}
        
        for task in tasks:
            if task.assigned_agent in assignments:
                assignments[task.assigned_agent].append(task.task_id)
        
        # Remove agents with no tasks
        return {agent: task_list for agent, task_list in assignments.items() if task_list}

    def create_plan(self, title: str, objective: str) -> ResearchPlan:
        """Create research plan with chain of thought reasoning"""
        reasoning_chain = [
            f"Analyzing research objective: {objective}",
            "Breaking down into component tasks",
            "Evaluating specialist requirements",
            "Determining subsearch agent needs"
        ]
        
        try:
            tasks = self.create_tasks(objective)
            agent_assignments = self.create_agent_assignments(tasks)
            
            plan = ResearchPlan(
                plan_id=f"AP{datetime.now().strftime('%Y%m%d%H%M')}",
                title=title,
                objective=objective,
                tasks=tasks,
                agent_assignments=agent_assignments,
                total_subsearch_agents=self.total_subsearch_agents,
                reasoning_chain=reasoning_chain
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            raise

    def display_plan(self, plan: ResearchPlan):
        """Display the research plan with thought process"""
        print(f"\n{'='*50}")
        print(f"AGRICULTURAL RESEARCH PLAN")
        print(f"{'='*50}")
        print(f"Title: {plan.title}")
        print(f"Objective: {plan.objective}")
        
        print("\nReasoning Chain:")
        for step in plan.reasoning_chain:
            print(f"- {step}")
        
        print("\nTasks:")
        for task in plan.tasks:
            print(f"\n{task.task_id}: {task.name}")
            print(f"Domain: {task.agricultural_domain}")
            print(f"Assigned: {task.assigned_agent}")
            print(f"Subsearch Agents: {task.subsearch_agents}")
            print("Thought Process:")
            for thought in task.thought_process:
                print(f"- {thought}")
        
        print(f"\nTotal Subsearch Agents Required: {plan.total_subsearch_agents}")

def main():
    planner = AgriculturalPlanningAgent()
    
    print("Agricultural Research Planning Assistant")
    print("=" * 40)
    
    # Get input
    title = input("Research Title: ").strip() or "Agricultural Research Study"
    objective = input("Research Objective: ").strip() or "Improve crop yield through better soil management"
    
    try:
        # Create and display plan
        plan = planner.create_plan(title, objective)
        planner.display_plan(plan)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()