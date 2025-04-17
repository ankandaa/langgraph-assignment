from typing import Dict, Any
from langgraph.graph import StateGraph, Graph
from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()

# Safely get environment variables with default values
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "")

# Only set non-empty environment variables
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
if LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_ENDPOINT:
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT

langsmith_client = Client()

class GraphState:
    """State object for the LangGraph workflow."""
    def __init__(self):
        self.srs_content: str = ""
        self.requirements: Dict[str, Any] = {}
        self.api_endpoints: list = []
        self.db_schema: Dict[str, Any] = {}
        self.generated_code: Dict[str, str] = {}
        self.test_cases: Dict[str, list] = {}
        self.errors: list = []
        self.logs: list = []

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow for FastAPI project generation."""
    
    workflow = StateGraph(GraphState)

    # Define nodes
    srs_parser_node = srs_parser
    project_initializer_node = ProjectInitializerNode().run

    # Add nodes to the graph
    workflow.add_node("srs_parser", srs_parser_node)
    workflow.add_node("project_initializer", project_initializer_node)
    workflow.add_node("error_handler", lambda state: state)  # Placeholder error handler

    # Configure edges
    workflow.set_entry_point("srs_parser")

    workflow.add_edge("srs_parser", "project_initializer")
    workflow.add_edge("project_initializer", "error_handler")
    workflow.add_edge("srs_parser", "error_handler", condition=lambda state: len(state["errors"]) > 0)
    workflow.add_edge("project_initializer", "test_generator")

    # Compile
    workflow.compile()
    
    return workflow

if __name__ == "__main__":
    workflow = create_workflow()
    # TODO: Add workflow execution logic
