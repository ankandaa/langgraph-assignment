import pytest
import os
from src.workflow import create_workflow, run_workflow

@pytest.mark.asyncio
async def test_workflow_with_srs_document():
    """Test workflow with actual SRS document."""
    # Path to the test SRS document
    srs_path = os.path.join("tests", "test.docx")
    
    # Ensure test document exists
    assert os.path.exists(srs_path), "test.docx not found in tests directory"
    
    # Run workflow
    final_state = await run_workflow(srs_path)
    
    # Verify state
    assert "errors" in final_state
    assert len(final_state["errors"]) == 0, f"Workflow errors: {final_state['errors']}"
    
    # Check if requirements were parsed
    assert "requirements" in final_state
    requirements = final_state["requirements"]
    
    # Verify parsed requirements
    assert "endpoints" in requirements
    assert "models" in requirements
    
    # Check generated files
    generated_api_dir = "generated_api"
    assert os.path.exists(generated_api_dir)
    
    # Check FastAPI app structure
    app_dir = os.path.join(generated_api_dir, "app")
    assert os.path.exists(app_dir)
    
    # Verify key files
    assert os.path.exists(os.path.join(app_dir, "main.py"))
    assert os.path.exists(os.path.join(app_dir, "models"))
    assert os.path.exists(os.path.join(app_dir, "api"))
    assert os.path.exists(os.path.join(app_dir, "tests"))

@pytest.mark.asyncio
async def test_workflow_state_progression():
    """Test workflow state changes through each node."""
    srs_path = os.path.join("tests", "test.docx")
    final_state = await run_workflow(srs_path)
    
    # Check logs for node execution sequence
    logs = final_state["logs"]
    expected_steps = [
        "Project initialized",
        "Requirements parsed",
        "Tests generated",
        "Code generated",
        "Tests passed"
    ]
    
    for step in expected_steps:
        assert any(step in log for log in logs), f"Missing step: {step}"
    
    # Verify final requirements
    requirements = final_state["requirements"]
    assert "endpoints" in requirements
    assert "models" in requirements
    assert len(requirements["endpoints"]) > 0
    assert len(requirements["models"]) > 0

@pytest.mark.asyncio
async def test_generated_code_structure():
    """Test the structure and content of generated code."""
    srs_path = os.path.join("tests", "test.docx")
    await run_workflow(srs_path)
    
    app_dir = os.path.join("generated_api", "app")
    
    # Check main FastAPI application
    main_py = os.path.join(app_dir, "main.py")
    assert os.path.exists(main_py)
    with open(main_py, 'r') as f:
        content = f.read()
        assert "FastAPI" in content
        assert "app = FastAPI()" in content
    
    # Check models
    models_dir = os.path.join(app_dir, "models")
    assert os.path.exists(models_dir)
    assert len(os.listdir(models_dir)) > 0
    
    # Check API routes
    api_dir = os.path.join(app_dir, "api")
    assert os.path.exists(api_dir)
    assert len(os.listdir(api_dir)) > 0
    
    # Check tests
    tests_dir = os.path.join(app_dir, "tests")
    assert os.path.exists(tests_dir)
    assert len(os.listdir(tests_dir)) > 0

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up generated files after tests."""
    yield
    generated_dir = "generated_api"
    if os.path.exists(generated_dir):
        import shutil
        shutil.rmtree(generated_dir)
