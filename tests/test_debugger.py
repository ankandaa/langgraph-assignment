import pytest
import os
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

from src.nodes.debugger import DebuggerNode

@pytest.fixture
def mock_groq():
    mock = MagicMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="# Fixed test code"))]
    ))
    return mock

@pytest.fixture
def mock_run():
    return MagicMock(id="test_run_id")

@pytest.fixture
def mock_langsmith(mock_run):  # Properly inject the mock_run fixture
    mock = MagicMock()
    mock.create_run = AsyncMock(return_value=mock_run)  # Use the fixture value directly
    mock.update_run = AsyncMock()
    return mock

@pytest.fixture
def debugger(mock_groq, mock_langsmith):
    debugger = DebuggerNode()
    debugger.groq_client = mock_groq
    debugger.langsmith_client = mock_langsmith
    return debugger

@pytest.fixture
def sample_test_output():
    return """
============================= test session starts ==============================
platform win32 -- Python 3.9.0, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
collected 6 items

tests/test_api.py::test_create_user FAILED
tests/test_models.py::test_user_model FAILED

=================================== FAILURES ===================================
_____________________________ test_create_user ______________________________
def test_create_user():
>       response = client.post("/api/users", json=test_data)
E       AssertionError: No validation for required fields
    """

@pytest.fixture
def sample_failing_code():
    return """
    @app.post("/api/users")
    async def create_user(user: dict):
        return {"id": 1, **user}
    """

@pytest.fixture
def cleanup_tests():
    """Fixture to clean up the test environment."""
    test_dir = "generated_api"
    yield
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.mark.asyncio
async def test_create_debug_prompt(debugger, sample_test_output, sample_failing_code):
    """Test debug prompt creation."""
    prompt = await debugger._create_debug_prompt(sample_test_output, sample_failing_code)
    
    assert isinstance(prompt, str)
    assert "Analyze the following test failure" in prompt
    assert sample_test_output in prompt
    assert sample_failing_code in prompt

@pytest.mark.asyncio
async def test_run_tests_success(debugger):
    """Test successful test run."""
    with patch('pytest.main', return_value=0):
        success, output = await debugger.run_tests()
        assert success
        assert isinstance(output, str)

@pytest.mark.asyncio
async def test_run_tests_failure(debugger):
    """Test failed test run."""
    with patch('pytest.main', return_value=1):
        success, output = await debugger.run_tests()
        assert not success
        assert isinstance(output, str)

@pytest.mark.asyncio
async def test_fix_test_failures(debugger, sample_test_output, mock_groq, mock_langsmith, cleanup_tests):
    """Test fixing test failures."""
    # Create a test file to fix
    os.makedirs(os.path.join(debugger.test_dir), exist_ok=True)  # Create parent directory first
    test_file = os.path.join(debugger.test_dir, "test_api.py")
    with open(test_file, "w") as f:
        f.write("# Test code with failure")

    await debugger.fix_test_failures(sample_test_output)

    # Check if Groq was called with correct prompt
    assert mock_groq.chat.completions.create.called

    # Check if LangSmith logs were created
    assert mock_langsmith.create_run.called
    create_run_args = mock_langsmith.create_run.call_args_list[0][1]
    assert "debug_test_failures" in create_run_args["name"]

@pytest.mark.asyncio
async def test_extract_failing_files(debugger, sample_test_output, cleanup_tests):
    """Test extraction of failing test files."""
    # Create test files
    os.makedirs(debugger.test_dir, exist_ok=True)
    test_files = ["test_api.py", "test_models.py"]
    for file in test_files:
        file_path = os.path.join(debugger.test_dir, file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("# Test code")

    failing_files = debugger._extract_failing_files(sample_test_output)
    assert len(failing_files) > 0
    for file in failing_files:
        assert os.path.basename(file) in test_files

@pytest.mark.asyncio
async def test_run_success(debugger, mock_groq, mock_langsmith, cleanup_tests, mock_run):
    """Test successful execution of the debugger node."""
    # Mock tests passing on first try
    run_tests_mock = AsyncMock(return_value=(True, "All tests passed"))
    
    with patch.object(DebuggerNode, 'run_tests', run_tests_mock):
        state = {
            "requirements": {},
            "logs": [],
            "errors": []
        }

        new_state, next_node = await debugger.run(state)

        assert next_node == "documentation_generator"
        assert "All tests passed successfully" in new_state["logs"]
        assert len(new_state["errors"]) == 0

        # Verify LangSmith interactions
        assert mock_langsmith.create_run.called
        assert mock_langsmith.update_run.called
        assert mock_langsmith.update_run.call_args[1]["outputs"]["status"] == "success"

@pytest.mark.asyncio
async def test_run_with_fixes(debugger, mock_groq, mock_langsmith, cleanup_tests, mock_run):
    """Test debugger node with test failures that get fixed."""
    test_results = [(False, "Tests failed"), (True, "Tests passed")]
    test_iter = iter(test_results)
    
    async def mock_run_tests():
        return next(test_iter)

    with patch.object(DebuggerNode, 'run_tests', AsyncMock(side_effect=mock_run_tests)):
        state = {
            "requirements": {},
            "logs": [],
            "errors": []
        }

        new_state, next_node = await debugger.run(state)

        assert next_node == "documentation_generator"
        assert "Successfully fixed all test failures" in new_state["logs"]
        assert len(new_state["errors"]) == 0

        # Verify LangSmith interactions
        assert mock_langsmith.create_run.called
        assert mock_langsmith.update_run.called
        assert mock_langsmith.update_run.call_args[1]["outputs"]["status"] == "fixed"

@pytest.mark.asyncio
async def test_run_error(debugger, mock_groq, mock_langsmith, cleanup_tests, mock_run):
    """Test error handling in the debugger node."""
    with patch.object(DebuggerNode, 'run_tests', AsyncMock(side_effect=Exception("Test error"))):
        state = {
            "requirements": {},
            "logs": [],
            "errors": []
        }

        new_state, next_node = await debugger.run(state)

        assert next_node == "error_handler"
        assert "Debugging error" in new_state["errors"][0]
        assert "Error during debugging" in new_state["logs"][0]

        # Verify LangSmith error logging
        assert mock_langsmith.create_run.called
        assert mock_langsmith.update_run.called
        assert mock_langsmith.update_run.call_args[1]["error"] is not None
