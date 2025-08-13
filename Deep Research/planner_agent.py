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
    estimated_hours: int = Field(..., description="Estimated hours to complete")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")

class ResearchPlan(BaseModel):
    plan_id: str = Field(..., description="Plan identifier")
    title: str = Field(..., description="Research title")
    objective: str = Field(..., description="Research objective")
    tasks: List[Task] = Field(..., description="List of tasks")
    agent_assignments: Dict[str, List[str]] = Field(..., description="Agent to tasks mapping")

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
        
        self.task_counter = 1

    def _get_planning_instructions(self) -> str:
        return """
        You are an Agricultural Research Planning Agent. Your job is to:
        
        1. Break down research objectives into 4-6 manageable tasks
        2. Assign each task to the most suitable agricultural specialist
        3. Estimate realistic time requirements for each task
        4. Identify task dependencies
        
        Available specialists:
        - soil_scientist: Soil health, nutrients, pH, fertilizers, soil testing
        - crop_agronomist: Crop selection, planting, growth monitoring, pest control  
        - field_researcher: Field trials, experimental design, data collection
        - data_analyst: Agricultural data analysis, statistics, yield modeling
        - climate_specialist: Weather patterns, climate adaptation, seasonal planning
        - sustainability_expert: Organic farming, sustainable practices, certification
        
        For each task, determine:
        - Which agricultural domain it belongs to (soil, crop, field, data, climate, sustainability)
        - Which specialist is best suited for the work
        - Realistic time estimate in hours
        - Any dependencies on other tasks
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

    def create_tasks(self, objective: str) -> List[Task]:
        """Break down objective into agricultural tasks"""
        
        # Standard agricultural research workflow
        base_tasks = [
            {
                "name": "Literature Review",
                "description": f"Research existing studies and best practices related to: {objective}",
                "hours": 8
            },
            {
                "name": "Site Assessment", 
                "description": f"Evaluate field conditions, soil, climate for: {objective}",
                "hours": 12
            },
            {
                "name": "Methodology Design",
                "description": f"Design research approach and protocols for: {objective}",
                "hours": 16
            },
            {
                "name": "Implementation",
                "description": f"Execute the research plan and collect data for: {objective}",
                "hours": 24
            },
            {
                "name": "Data Analysis",
                "description": f"Analyze results and interpret findings for: {objective}",
                "hours": 12
            },
            {
                "name": "Recommendations",
                "description": f"Develop practical recommendations based on: {objective}",
                "hours": 8
            }
        ]
        
        tasks = []
        dependencies = []
        
        for i, task_info in enumerate(base_tasks):
            # Determine domain and assign agent
            domain = self._determine_domain(task_info["description"])
            agent = self._assign_best_agent(domain, task_info["name"])
            
            # Set dependencies (each task depends on previous)
            deps = [f"T{i}"] if i > 1 else []
            
            task = Task(
                task_id=f"T{i+1}",
                name=task_info["name"],
                description=task_info["description"],
                agricultural_domain=domain,
                assigned_agent=agent,
                estimated_hours=task_info["hours"],
                dependencies=deps
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
        """Create a simple research plan with task assignments"""
        try:
            # Break down into tasks
            tasks = self.create_tasks(objective)
            
            # Create agent assignments
            agent_assignments = self.create_agent_assignments(tasks)
            
            plan = ResearchPlan(
                plan_id=f"AP{datetime.now().strftime('%Y%m%d%H%M')}",
                title=title,
                objective=objective,
                tasks=tasks,
                agent_assignments=agent_assignments
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            raise

    def display_plan(self, plan: ResearchPlan):
        """Display the research plan in a clear format"""
        print(f"\n{'='*50}")
        print(f"AGRICULTURAL RESEARCH PLAN")
        print(f"{'='*50}")
        print(f"Title: {plan.title}")
        print(f"Objective: {plan.objective}")
        print(f"Plan ID: {plan.plan_id}")
        
        print(f"\n{'Tasks:':<20}")
        print("-" * 50)
        total_hours = 0
        for task in plan.tasks:
            print(f"{task.task_id}: {task.name}")
            print(f"   Domain: {task.agricultural_domain}")
            print(f"   Assigned: {task.assigned_agent}")
            print(f"   Hours: {task.estimated_hours}")
            print(f"   Dependencies: {', '.join(task.dependencies) if task.dependencies else 'None'}")
            print(f"   Description: {task.description}")
            print()
            total_hours += task.estimated_hours
        
        print(f"{'Agent Assignments:':<20}")
        print("-" * 50)
        for agent, task_ids in plan.agent_assignments.items():
            agent_name = agent.replace('_', ' ').title()
            specialization = self.available_agents[agent]
            task_hours = sum(task.estimated_hours for task in plan.tasks if task.task_id in task_ids)
            print(f"{agent_name}:")
            print(f"   Specialization: {specialization}")
            print(f"   Tasks: {', '.join(task_ids)} ({task_hours} hours)")
            print()
        
        print(f"Total Estimated Hours: {total_hours}")
        print(f"Estimated Duration: {total_hours // 8} working days")

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