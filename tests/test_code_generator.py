import pytest
import os
import shutil
from unittest.mock import patch, Mock, MagicMock, AsyncMock  # Import AsyncMock
from src.nodes.code_generator import CodeGeneratorNode

@pytest.fixture
def mock_langsmith_run():
    mock_run = MagicMock()
    mock_run.id = "test_run_id"
    return mock_run

@pytest.fixture
def mock_llm():
    mock = Mock()
    # Replace Mock with AsyncMock for the async method
    mock.ainvoke = AsyncMock(return_value=Mock(content="# Generated code"))
    return mock

@pytest.fixture
def mock_langsmith():
    mock = Mock()
    mock.create_run = Mock(return_value=MagicMock(id="test_run_id"))
    mock.update_run = Mock()
    return mock

@pytest.fixture
def code_generator(mock_llm, mock_langsmith):
    generator = CodeGeneratorNode()
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
def cleanup_code():
    """Fixture to clean up the generated code after each test."""
    app_dir = "generated_api/app"
    yield
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)

@pytest.mark.asyncio
async def test_generate_models(code_generator, sample_requirements, cleanup_code, mock_llm, mock_langsmith):
    """Test model code generation."""
    await code_generator.generate_models(
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
    assert "generate_model_code" in mock_langsmith.create_run.call_args_list[0][1]["name"]

    # Check if model files were created
    for model in sample_requirements["models"]:
        model_file = os.path.join(code_generator.app_dirs["models"], f"{model.lower()}.py")
        assert os.path.exists(model_file)

@pytest.mark.asyncio
async def test_generate_routes(code_generator, sample_requirements, cleanup_code, mock_llm, mock_langsmith):
    """Test route code generation."""
    await code_generator.generate_routes(
        sample_requirements["endpoints"],
        sample_requirements
    )

    # Check if LLM was called with correct prompts
    assert mock_llm.ainvoke.called
    prompt_call = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(prompt_call, str)
    assert "FastAPI route handler" in prompt_call

    # Check if LangSmith created runs
    assert mock_langsmith.create_run.called
    assert "generate_route_code" in mock_langsmith.create_run.call_args_list[0][1]["name"]

    # Check if route files were created
    for endpoint in sample_requirements["endpoints"]:
        resource = endpoint.split('/')[-1]
        route_file = os.path.join(code_generator.app_dirs["routes"], f"{resource}.py")
        assert os.path.exists(route_file)

@pytest.mark.asyncio
async def test_generate_services(code_generator, sample_requirements, cleanup_code, mock_llm, mock_langsmith):
    """Test service code generation."""
    await code_generator.generate_services(
        sample_requirements["models"],
        sample_requirements
    )

    # Check if LLM was called with correct prompts
    assert mock_llm.ainvoke.called
    prompt_call = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(prompt_call, str)
    assert "service class" in prompt_call

    # Check if LangSmith created runs
    assert mock_langsmith.create_run.called
    assert "generate_service_code" in mock_langsmith.create_run.call_args_list[0][1]["name"]

    # Check if service files were created
    for model in sample_requirements["models"]:
        service_file = os.path.join(code_generator.app_dirs["services"], f"{model.lower()}_service.py")
        assert os.path.exists(service_file)

@pytest.mark.asyncio
async def test_run_success(code_generator, sample_requirements, cleanup_code, mock_llm, mock_langsmith):
    """Test successful execution of the code generator node."""
    state = {
        "requirements": sample_requirements,
        "logs": [],
        "errors": []
    }

    new_state, next_node = await code_generator.run(state)

    # Check node execution
    assert next_node == "debugger"
    assert "Successfully generated application code" in new_state["logs"]
    assert len(new_state["errors"]) == 0

    # Check LangSmith run creation and update
    assert mock_langsmith.create_run.called
    assert mock_langsmith.update_run.called
    assert "code_generator_run" in mock_langsmith.create_run.call_args[0]
    assert "success" in mock_langsmith.update_run.call_args[1]["outputs"]["status"]

@pytest.mark.asyncio
async def test_run_error(code_generator, sample_requirements, cleanup_code, mock_llm, mock_langsmith):
    """Test error handling in the code generator node."""
    # Set up LLM to raise an exception
    mock_llm.ainvoke.side_effect = Exception("LLM error")

    state = {
        "requirements": sample_requirements,
        "logs": [],
        "errors": []
    }

    new_state, next_node = await code_generator.run(state)

    # Check error handling
    assert next_node == "error_handler"
    assert "Code generation error" in new_state["errors"][0]
    assert "Error generating code" in new_state["logs"][0]

    # Check LangSmith error logging
    assert mock_langsmith.create_run.called
    assert mock_langsmith.update_run.called
    assert "error" in mock_langsmith.update_run.call_args[1]

@pytest.mark.asyncio
async def test_file_write_error(code_generator, sample_requirements, cleanup_code, mock_langsmith):
    """Test handling of file write errors."""
    with patch("builtins.open", side_effect=IOError("Test error")):
        with pytest.raises(Exception) as exc_info:
            await code_generator.generate_models(
                sample_requirements["models"],
                sample_requirements
            )
        assert "Error generating model code" in str(exc_info.value)
        
        # Check LangSmith error run creation
        assert mock_langsmith.create_run.called
        assert "model_generation_error" in mock_langsmith.create_run.call_args_list[-1][1]["name"]
