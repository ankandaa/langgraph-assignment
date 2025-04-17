import pytest
import os
import shutil
from unittest.mock import Mock, patch
from dotenv import load_dotenv
from src.nodes.project_initializer import ProjectInitializerNode

# Load environment variables from .env file if it exists
load_dotenv()

# Define test environment variables
TEST_ENV_VARS = {
    "LANGCHAIN_API_KEY": "test_key",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com"
}

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Automatically set up test environment variables for all tests"""
    for key, value in TEST_ENV_VARS.items():
        monkeypatch.setenv(key, value)

@pytest.fixture()
def project_initializer():
    return ProjectInitializerNode()

@pytest.fixture()
def sample_requirements():
    return {
        "endpoints": ["/api/users", "/api/items"],
        "models": ["User", "Item"],
        "features": ["Authentication", "Authorization"],
        "auth": "JWT"
    }

@pytest.fixture()
def cleanup_project():
    """Fixture to clean up the generated project after the test."""
    project_name = "generated_api"
    yield
    if os.path.exists(project_name):
        shutil.rmtree(project_name)

@pytest.mark.asyncio
async def test_create_project_structure(project_initializer, sample_requirements, cleanup_project):
    """Test that the project structure is created correctly."""
    await project_initializer.create_project_structure(sample_requirements)

    project_name = "generated_api"
    assert os.path.exists(project_name)
    assert os.path.exists(os.path.join(project_name, "app"))
    assert os.path.exists(os.path.join(project_name, "app", "api"))
    assert os.path.exists(os.path.join(project_name, "app", "models"))
    assert os.path.exists(os.path.join(project_name, "app", "services"))
    assert os.path.exists(os.path.join(project_name, "app", "api", "routes"))
    assert os.path.exists(os.path.join(project_name, "app", "main.py"))
    assert os.path.exists(os.path.join(project_name, "requirements.txt"))

@pytest.mark.asyncio
async def test_setup_podman_postgres(project_initializer, sample_requirements, cleanup_project):
    """Test that the Podman PostgreSQL setup is called correctly."""
    with patch("subprocess.run") as mock_run:
        await project_initializer.setup_podman_postgres(sample_requirements)
        mock_run.assert_called()

@pytest.mark.asyncio
async def test_create_virtual_environment(project_initializer, sample_requirements, cleanup_project):
    """Test that the virtual environment is created and dependencies are installed."""
    with patch("subprocess.run") as mock_run:
        await project_initializer.create_virtual_environment(sample_requirements)
        mock_run.assert_called()

@pytest.mark.asyncio
async def test_run_success(project_initializer, sample_requirements, cleanup_project):
    """Test successful execution of the project initializer node."""
    state = {"requirements": sample_requirements, "logs": [], "errors": []}
    new_state, next_node = await project_initializer.run(state)

    assert next_node == "test_generator"
    assert "Successfully initialized project structure" in new_state["logs"]
    assert len(new_state["errors"]) == 0

@pytest.mark.asyncio
async def test_run_error(project_initializer, sample_requirements, cleanup_project):
    """Test error handling in the project initializer node."""
    with patch.object(ProjectInitializerNode, "create_project_structure", side_effect=Exception("Test Error")):
        state = {"requirements": sample_requirements, "logs": [], "errors": []}
        new_state, next_node = await project_initializer.run(state)

        assert next_node == "error_handler"
        assert "Project initialization error" in new_state["errors"][0]
