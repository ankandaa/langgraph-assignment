import asyncio
from typing import Dict, Any, Tuple, List
from langgraph.graph import Graph
from src.nodes.project_initializer import ProjectInitializerNode
from src.nodes.srs_parser import SRSParserNode
from src.nodes.test_generator import TestGeneratorNode
from src.nodes.code_generator import CodeGeneratorNode
from src.nodes.debugger import DebuggerNode

async def create_workflow() -> Graph:
    """Creates the workflow graph for API generation."""

    # Initialize nodes
    project_initializer = ProjectInitializerNode()
    srs_parser = SRSParserNode()
    test_generator = TestGeneratorNode()
    code_generator = CodeGeneratorNode()
    debugger = DebuggerNode()

    # Create workflow graph
    workflow = Graph()

    # Add nodes to graph with async functions - removed config parameter
    workflow.add_node("project_initializer", lambda x: project_initializer.run(x))
    workflow.add_node("srs_parser", lambda x: srs_parser.run(x))
    workflow.add_node("test_generator", lambda x: test_generator.run(x))
    workflow.add_node("code_generator", lambda x: code_generator.run(x))
    workflow.add_node("debugger", lambda x: debugger.run(x))

    # Define error handler
    async def error_handler(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Handles errors in the workflow."""
        print(f"Error occurred: {state.get('errors', [])}")
        return state, "error"

    # Add error handler node - removed config parameter
    workflow.add_node("error_handler", lambda x: error_handler(x))
    
    # Add completion handler
    async def complete_handler(state: Dict[str, Any]) -> Dict[str, Any]:
        """Completion handler."""
        state["logs"].append("Workflow completed successfully")
        return state
        
    workflow.add_node("complete", lambda x: complete_handler(x))

    # Define edges with conditions
    def is_success(x: Tuple[Dict[str, Any], str]) -> bool:
        return x[1] != "error_handler"

    def is_error(x: Tuple[Dict[str, Any], str]) -> bool:
        return x[1] == "error_handler"

    # Add edges with conditions
    workflow.add_edge("project_initializer", "srs_parser", condition=is_success)
    workflow.add_edge("project_initializer", "error_handler", condition=is_error)

    workflow.add_edge("srs_parser", "test_generator", condition=is_success)
    workflow.add_edge("srs_parser", "error_handler", condition=is_error)

    workflow.add_edge("test_generator", "code_generator", condition=is_success)
    workflow.add_edge("test_generator", "error_handler", condition=is_error)

    workflow.add_edge("code_generator", "debugger", condition=is_success)
    workflow.add_edge("code_generator", "error_handler", condition=is_error)

    workflow.add_edge("debugger", "complete", condition=is_success)
    workflow.add_edge("debugger", "error_handler", condition=is_error)

    # Set entry point
    workflow.set_entry_point("project_initializer")

    # Compile graph
    workflow.compile()

    return workflow

async def run_workflow(srs_doc_path: str) -> Dict[str, Any]:
    """Runs the API generation workflow."""
    try:
        # Create workflow
        workflow = await create_workflow()

        # Initialize state
        initial_state = {
            "srs_path": srs_doc_path,
            "requirements": {},
            "logs": [],
            "errors": []
        }

        # Run workflow - using invoke instead of arun for newer versions
        try:
            # First try with invoke (newer API)
            final_state = await workflow.invoke(initial_state)
        except AttributeError:
            # Fall back to arun (older API)
            final_state = await workflow.arun(initial_state)

        return final_state

    except Exception as e:
        print(f"Workflow error: {e}")
        return {
            "srs_path": srs_doc_path,
            "errors": [str(e)],
            "logs": [f"Workflow failed: {e}"]
        }

if __name__ == "__main__":
    # Example usage
    srs_doc = "test.docx"
    asyncio.run(run_workflow(srs_doc))
