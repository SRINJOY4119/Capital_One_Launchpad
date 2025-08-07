from pprint import pprint
from adaptive_rag_class import ADAPTIVE_RAG
from stategraph import GraphState
from langgraph.graph import END, StateGraph, START
    

class Workflow:
    def __init__(self, model, api_key, k, csv_path):
        self.adaptive_rag = ADAPTIVE_RAG(model, api_key, k, csv_path)
        self.workflow = StateGraph(GraphState)
        
        # Add nodes without router
        self.workflow.add_node("web_search", self.adaptive_rag.web_search)
        self.workflow.add_node("retrieve", self.adaptive_rag.retrieve)
        self.workflow.add_node("grade_documents", self.adaptive_rag.grade_documents)
        self.workflow.add_node("generate", self.adaptive_rag.generate)
        self.workflow.add_node("transform_query", self.adaptive_rag.transform_query)
        self.workflow.add_node("fast_generate", self.adaptive_rag.fast_generate)
        self.workflow.add_node("introspective_agent_response", self.adaptive_rag.introspective_agent_response)
        
        # Start directly with retrieve (no router)
        self.workflow.add_edge(START, "retrieve")
        
        # Simplified workflow edges
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.adaptive_rag.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")

        self.workflow.add_conditional_edges(
            "generate",
            self.adaptive_rag.grade_generation_v_documents_and_question,
            {
                "not supported": "introspective_agent_response",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        self.workflow.add_edge("introspective_agent_response", END)
        self.app = self.workflow.compile()

    def build_workflow(self):
        pass

    def run_workflow(self, inputs):
        for output in self.app.stream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")
            pprint("\n---\n")

        # Return final results
        return {"generation": value["generation"], "extracted_info": value.get("extractions"), "documents": value["documents"]}
        