import pytest
from unittest.mock import Mock, patch
import json
from src.nodes.srs_parser import analyze_requirements, process_docx, srs_parser

@pytest.fixture
def sample_srs_file():
    return "tests/test.docx"

@pytest.fixture
def sample_srs_content(sample_srs_file):
    from src.nodes.srs_parser import process_docx
    return process_docx(sample_srs_file)

@pytest.fixture
def mock_groq_response():
    return Mock(choices=[
        Mock(message=Mock(content=json.dumps({
            "functional_requirements": [
                "User management system",
                "Post creation and management"
            ],
            "api_endpoints": [
                {
                    "path": "/users",
                    "method": "POST",
                    "description": "Create new user"
                }
            ],
            "db_schema": {
                "users": {
                    "fields": ["id", "username", "email"]
                }
            },
            "auth_requirements": {
                "type": "JWT",
                "features": ["RBAC"]
            }
        })))
    ])

@pytest.mark.asyncio
async def test_analyze_requirements(sample_srs_content, mock_groq_response):
    """Test that SRS content is properly analyzed."""
    with patch('src.nodes.srs_parser.get_groq_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_groq_response
        mock_get_client.return_value = mock_client
        
        result = await analyze_requirements(sample_srs_content)
        
        # Verify structure of parsed requirements
        parsed = json.loads(result)
        assert "functional_requirements" in parsed
        assert "api_endpoints" in parsed
        assert "db_schema" in parsed
        assert "auth_requirements" in parsed
        
        # Verify content
        assert len(parsed["functional_requirements"]) > 0
        assert len(parsed["api_endpoints"]) > 0
        assert "users" in parsed["db_schema"]

@pytest.mark.asyncio
async def test_process_docx():
    """Test processing of .docx files."""
    content = process_docx("tests/test.docx")
    
    # Verify the content is properly extracted
    assert content.strip() != ""
    assert len(content) > 0

@pytest.mark.asyncio
async def test_srs_parser_success(sample_srs_content, mock_groq_response):
    """Test successful execution of the parser."""
    with patch('src.nodes.srs_parser.get_groq_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_groq_response
        mock_get_client.return_value = mock_client
        
        state = {"srs_content": "tests/test.docx", "logs": [], "errors": []}
        new_state = await srs_parser(state)
        
        assert "requirements" in new_state
        assert len(new_state["logs"]) > 0
        assert len(new_state["errors"]) == 0

@pytest.mark.asyncio
async def test_srs_parser_error():
    """Test error handling in the parser."""
    state = {"srs_content": "", "logs": [], "errors": []}
    new_state = await srs_parser(state)
    
    assert len(new_state["errors"]) > 0
    assert "SRS parsing error" in new_state["errors"][0]
