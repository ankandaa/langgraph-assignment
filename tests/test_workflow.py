import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from src.workflow import create_workflow, run_workflow

@pytest.fixture
def mock_state():
    return {
        "srs_path": "test_requirements.docx",
        "requirements": {
            "endpoints": ["/api/users", "/api/items"],
            "models": ["User", "Item"],
            "auth": {"type": "JWT"}
        },
        "logs": [],
        "errors": []
    }

@pytest.fixture
def mock_nodes():
    # Project Initializer Mock
    project_initializer = MagicMock()
    project_initializer.run = AsyncMock(return_value=({
        "srs_path": "test_requirements.docx",
        "requirements": {},
        "logs": ["Project initialized"],
        "errors": []
    }, "srs_parser"))

    # SRS Parser Mock
    srs_parser = MagicMock()
    srs_parser.run = AsyncMock(return_value=({
        "srs_path": "test_requirements.docx",
        "requirements": {
            "endpoints": ["/api/users", "/api/items"],
            "models": ["User", "Item"]
        },
        "logs": ["Requirements parsed"],
        "errors": []
    }, "test_generator"))

    # Test Generator Mock
    test_generator = MagicMock()
    test_generator.run = AsyncMock(return_value=({
        "srs_path": "test_requirements.docx",
        "requirements": {
            "endpoints": ["/api/users", "/api/items"],
            "models": ["User", "Item"]
        },
        "logs": ["Tests generated"],
        "errors": []
    }, "code_generator"))

    # Code Generator Mock
    code_generator = MagicMock()
    code_generator.run = AsyncMock(return_value=({
        "srs_path": "test_requirements.docx",
        "requirements": {
            "endpoints": ["/api/users", "/api/items"],
            "models": ["User", "Item"]
        },
        "logs": ["Code generated"],
        "errors": []
    }, "debugger"))

    # Debugger Mock
    debugger = MagicMock()
    debugger.run = AsyncMock(return_value=({
        "srs_path": "test_requirements.docx",
        "requirements": {
            "endpoints": ["/api/users", "/api/items"],
            "models": ["User", "Item"]
        },
        "logs": ["Tests passed"],
        "errors": []
    }, "complete"))

    return {
        "ProjectInitializerNode": project_initializer,
        "SRSParserNode": srs_parser,
        "TestGeneratorNode": test_generator,
        "CodeGeneratorNode": code_generator,
        "DebuggerNode": debugger
    }

@pytest.mark.asyncio
async def test_create_workflow(mock_nodes):
    """Test workflow creation with all nodes."""
    with patch("src.workflow.ProjectInitializerNode", return_value=mock_nodes["ProjectInitializerNode"]), \
         patch("src.workflow.SRSParserNode", return_value=mock_nodes["SRSParserNode"]), \
         patch("src.workflow.TestGeneratorNode", return_value=mock_nodes["TestGeneratorNode"]), \
         patch("src.workflow.CodeGeneratorNode", return_value=mock_nodes["CodeGeneratorNode"]), \
         patch("src.workflow.DebuggerNode", return_value=mock_nodes["DebuggerNode"]):
        
        workflow = await create_workflow()
        assert workflow is not None
        assert hasattr(workflow, 'compile')
        assert hasattr(workflow, 'arun')

@pytest.mark.asyncio
async def test_workflow_successful_execution(mock_nodes, mock_state):
    """Test successful execution of the entire workflow."""
    with patch("src.workflow.ProjectInitializerNode", return_value=mock_nodes["ProjectInitializerNode"]), \
         patch("src.workflow.SRSParserNode", return_value=mock_nodes["SRSParserNode"]), \
         patch("src.workflow.TestGeneratorNode", return_value=mock_nodes["TestGeneratorNode"]), \
         patch("src.workflow.CodeGeneratorNode", return_value=mock_nodes["CodeGeneratorNode"]), \
         patch("src.workflow.DebuggerNode", return_value=mock_nodes["DebuggerNode"]):
        
        final_state = await run_workflow("test_requirements.docx")
        
        # Check state
        assert "errors" in final_state
        assert len(final_state["errors"]) == 0
        assert "logs" in final_state
        assert len(final_state["logs"]) == 5  # One log from each node
        
        # Check execution sequence through logs
        logs = final_state["logs"]
        assert "Project initialized" in logs[0]
        assert "Requirements parsed" in logs[1]
        assert "Tests generated" in logs[2]
        assert "Code generated" in logs[3]
        assert "Tests passed" in logs[4]

@pytest.mark.asyncio
async def test_workflow_error_handling(mock_nodes):
    """Test workflow error handling."""
    # Mock error in Project Initializer
    error_node = MagicMock()
    error_node.run = AsyncMock(side_effect=Exception("Project initialization failed"))
    mock_nodes["ProjectInitializerNode"] = error_node

    with patch("src.workflow.ProjectInitializerNode", return_value=mock_nodes["ProjectInitializerNode"]), \
         patch("src.workflow.SRSParserNode", return_value=mock_nodes["SRSParserNode"]), \
         patch("src.workflow.TestGeneratorNode", return_value=mock_nodes["TestGeneratorNode"]), \
         patch("src.workflow.CodeGeneratorNode", return_value=mock_nodes["CodeGeneratorNode"]), \
         patch("src.workflow.DebuggerNode", return_value=mock_nodes["DebuggerNode"]):
        
        final_state = await run_workflow("test_requirements.docx")
        
        assert "errors" in final_state
        assert len(final_state["errors"]) > 0
        assert "Project initialization failed" in str(final_state["errors"][0])
        assert "Workflow failed" in final_state["logs"][0]

@pytest.mark.asyncio
async def test_workflow_state_transitions(mock_nodes, mock_state):
    """Test state transitions between nodes."""
    states = []

    def create_state_tracking_mock(name, next_node):
        async def run_with_state_tracking(state):
            state["logs"].append(f"{name} executed")
            states.append(state.copy())
            return state, next_node
        
        node = MagicMock()
        node.run = run_with_state_tracking
        return node

    # Create nodes that track state
    mock_nodes["ProjectInitializerNode"] = create_state_tracking_mock("project_initializer", "srs_parser")
    mock_nodes["SRSParserNode"] = create_state_tracking_mock("srs_parser", "test_generator")
    mock_nodes["TestGeneratorNode"] = create_state_tracking_mock("test_generator", "code_generator")
    mock_nodes["CodeGeneratorNode"] = create_state_tracking_mock("code_generator", "debugger")
    mock_nodes["DebuggerNode"] = create_state_tracking_mock("debugger", "complete")

    with patch("src.workflow.ProjectInitializerNode", return_value=mock_nodes["ProjectInitializerNode"]), \
         patch("src.workflow.SRSParserNode", return_value=mock_nodes["SRSParserNode"]), \
         patch("src.workflow.TestGeneratorNode", return_value=mock_nodes["TestGeneratorNode"]), \
         patch("src.workflow.CodeGeneratorNode", return_value=mock_nodes["CodeGeneratorNode"]), \
         patch("src.workflow.DebuggerNode", return_value=mock_nodes["DebuggerNode"]):
        
        final_state = await run_workflow("test_requirements.docx")

        # Check state progression
        assert len(states) == 5  # One state per node
        for i, state in enumerate(states):
            assert "logs" in state
            assert f"{['project_initializer', 'srs_parser', 'test_generator', 'code_generator', 'debugger'][i]} executed" in state["logs"][-1]

@pytest.mark.asyncio
async def test_workflow_error_propagation(mock_nodes):
    """Test error propagation through the workflow."""
    error_states = []

    def create_error_propagation_mock(name, should_error=False):
        async def run_with_error(state):
            if should_error:
                error_msg = f"{name} failed"
                state["errors"].append(error_msg)
                error_states.append(state.copy())
                return state, "error_handler"
            state["logs"].append(f"{name} succeeded")
            return state, "complete"
        
        node = MagicMock()
        node.run = run_with_error
        return node

    # Make SRS Parser fail
    mock_nodes["ProjectInitializerNode"] = create_error_propagation_mock("project_initializer")
    mock_nodes["SRSParserNode"] = create_error_propagation_mock("srs_parser", should_error=True)

    with patch("src.workflow.ProjectInitializerNode", return_value=mock_nodes["ProjectInitializerNode"]), \
         patch("src.workflow.SRSParserNode", return_value=mock_nodes["SRSParserNode"]), \
         patch("src.workflow.TestGeneratorNode", return_value=mock_nodes["TestGeneratorNode"]), \
         patch("src.workflow.CodeGeneratorNode", return_value=mock_nodes["CodeGeneratorNode"]), \
         patch("src.workflow.DebuggerNode", return_value=mock_nodes["DebuggerNode"]):
        
        final_state = await run_workflow("test_requirements.docx")

        # Check error handling
        assert len(error_states) > 0
        assert "srs_parser failed" in error_states[0]["errors"]
        assert "error_handler" in str(final_state)
