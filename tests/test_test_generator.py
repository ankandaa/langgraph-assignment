import pytest
import os
import shutil
from unittest.mock import patch, Mock, MagicMock, AsyncMock  # Add AsyncMock import
from src.nodes.test_generator import TestGeneratorNode

@pytest.fixture
def mock_langsmith_run():
    mock_run = MagicMock()
    mock_run.id = "test_run_id"
    return mock_run

@pytest.fixture
def mock_llm():
    mock = Mock()
    # Use AsyncMock for the async method
    mock.ainvoke = AsyncMock(return_value=Mock(content="# Generated test content"))
    return mock

@pytest.fixture
def mock_langsmith():
    mock = Mock()
    mock.create_run = Mock(return_value=MagicMock(id="test_run_id"))
    mock.update_run = Mock()
    return mock

@pytest.fixture
def test_generator(mock_llm, mock_langsmith):
    generator = TestGeneratorNode()
    generator.llm = mock_llm
    generator.langsmith_client = mock_langsmith
    return generator

@pytest.fixture
def sample_requirements():
    return {
        "endpoints": ["/api/users", "/api/items"],
        "models": ["User", "Item"],
        "features": ["Authentication", "Authorization"],
        "auth": {
            "type": "JWT",
            "expiry": 3600
        }
    }

@pytest.fixture
def cleanup_tests():
    """Fixture to clean up the generated tests after each test."""
    test_dir = "generated_api"
    yield
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.mark.asyncio
async def test_generate_api_tests(test_generator, sample_requirements, cleanup_tests, mock_llm, mock_langsmith):
    """Test API test generation with LLM and LangSmith."""
    await test_generator.generate_api_tests(
        sample_requirements["endpoints"],
        sample_requirements
    )

    # Check if LLM was called with correct prompts
    assert mock_llm.ainvoke.called
    prompt_call = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(prompt_call, str)
    assert "Generate pytest test cases" in prompt_call

    # Check if LangSmith created runs
    assert mock_langsmith.create_run.called
    assert "generate_api_tests" in mock_langsmith.create_run.call_args_list[0][1]["name"]

    # Check if test files were created
    for endpoint in sample_requirements["endpoints"]:
        resource = endpoint.split('/')[-1]
        test_file = os.path.join(test_generator.test_dirs["routes"], f"test_{resource}.py")
        assert os.path.exists(test_file)

@pytest.mark.asyncio
async def test_generate_model_tests(test_generator, sample_requirements, cleanup_tests, mock_llm, mock_langsmith):
    """Test model test generation with LLM and LangSmith."""
    await test_generator.generate_model_tests(
        sample_requirements["models"],
        sample_requirements
    )

    # Check if LLM was called with correct prompts
    assert mock_llm.ainvoke.called
    prompt_call = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(prompt_call, str)
    assert "SQLAlchemy model" in prompt_call

    # Check if LangSmith created runs
    assert mock_langsmith.create_run.called
    assert "generate_model_tests" in mock_langsmith.create_run.call_args_list[0][1]["name"]

    # Check if test files were created
    for model in sample_requirements["models"]:
        test_file = os.path.join(test_generator.test_dirs["models"], f"test_{model.lower()}.py")
        assert os.path.exists(test_file)

@pytest.mark.asyncio
async def test_generate_auth_tests(test_generator, sample_requirements, cleanup_tests, mock_llm, mock_langsmith):
    """Test auth test generation with LLM and LangSmith."""
    await test_generator.generate_auth_tests(sample_requirements["auth"])

    # Check if LLM was called with correct prompts
    assert mock_llm.ainvoke.called
    prompt_call = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(prompt_call, str)
    assert "authentication and authorization" in prompt_call

    # Check if LangSmith created runs
    assert mock_langsmith.create_run.called
    assert "generate_auth_tests" in mock_langsmith.create_run.call_args_list[0][1]["name"]

    # Check if auth test file was created
    test_file = os.path.join(test_generator.test_dirs["auth"], "test_auth.py")
    assert os.path.exists(test_file)

@pytest.mark.asyncio
async def test_run_success(test_generator, sample_requirements, cleanup_tests, mock_llm, mock_langsmith):
    """Test successful execution of the test generator node."""
    state = {
        "requirements": sample_requirements,
        "logs": [],
        "errors": []
    }

    new_state, next_node = await test_generator.run(state)

    # Check node execution
    assert next_node == "code_generator"
    assert "Successfully generated test cases" in new_state["logs"]
    assert len(new_state["errors"]) == 0

    # Check LangSmith run creation and update
    assert mock_langsmith.create_run.called
    assert mock_langsmith.update_run.called
    assert mock_langsmith.create_run.call_args[1].get("name") == "test_generator_run"
    assert "success" in mock_langsmith.update_run.call_args[1]["outputs"]["status"]

@pytest.mark.asyncio
async def test_run_error(test_generator, sample_requirements, cleanup_tests, mock_llm, mock_langsmith):
    """Test error handling in the test generator node."""
    # Set up LLM to raise an exception
    mock_llm.ainvoke.side_effect = Exception("LLM error")

    state = {
        "requirements": sample_requirements,
        "logs": [],
        "errors": []
    }

    new_state, next_node = await test_generator.run(state)

    # Check error handling
    assert next_node == "error_handler"
    assert "Test generation error" in new_state["errors"][0]
    assert "Error generating tests" in new_state["logs"][0]

    # Check LangSmith error logging
    assert mock_langsmith.create_run.called
    assert mock_langsmith.update_run.called
    assert "error" in mock_langsmith.update_run.call_args[1]

@pytest.mark.asyncio
async def test_file_write_error(test_generator, sample_requirements, cleanup_tests, mock_langsmith):
    """Test handling of file write errors."""
    with patch("builtins.open", side_effect=IOError("Test error")):
        with pytest.raises(Exception) as exc_info:
            await test_generator.generate_api_tests(
                sample_requirements["endpoints"],
                sample_requirements
            )
        assert "Error generating API tests" in str(exc_info.value)
        
        # Check LangSmith error run creation
        assert mock_langsmith.create_run.called
        assert "api_test_error" in mock_langsmith.create_run.call_args_list[-1][1]["name"]
