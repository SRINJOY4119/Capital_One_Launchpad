from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from Deep_Research.workflow import DeepResearchWorkflow

router = APIRouter(prefix="/api/v1", tags=["Deep Research"])
workflow = DeepResearchWorkflow()

@router.post("/deep-research/")
def run_deep_research(
    objective: str,
    location: Optional[str] = "Global",
    focus_areas: Optional[List[str]] = Query(default=None)
):

    try:
        result = workflow.execute_research(
            objective=objective,
            location=location,
            focus_areas=focus_areas or []
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
