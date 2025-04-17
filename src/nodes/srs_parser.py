from typing import Dict, Any, Annotated, Callable
from langgraph.graph import MessageGraph, StateGraph
from groq import Groq
import os
from docx import Document
import json

# Constants
MODEL_NAME = "mixtral-8x7b"  # Using latest Mixtral model on Groq

def get_groq_client():
    """Get or initialize Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable must be set")
    return Groq(api_key=api_key)

def process_docx(file_path: str) -> str:
    """Extract text content from .docx file."""
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

async def analyze_requirements(content: str) -> str:
    """Extract functional requirements from SRS content."""
    prompt = f"""You are a software engineer analyzing a Software Requirements Specification (SRS) document.
    Extract the following information and return it as a valid JSON:
    1. Core functional requirements
    2. Required API endpoints with their parameters
    3. Database schema requirements
    4. Authentication and authorization requirements

    SRS Content:
    {content}

    Return the analysis as a structured JSON with these exact keys:
    - functional_requirements: list of requirements
    - api_endpoints: list of endpoint specs with path, method, description
    - db_schema: database table specifications with fields
    - auth_requirements: auth system specifications
    
    Ensure the response is valid JSON that can be parsed."""
    
    response = get_groq_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,
        max_tokens=4000,
    )
    
    return response.choices[0].message.content

async def srs_parser(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse SRS document and extract requirements."""
    try:
        if not state["srs_content"]:
            raise ValueError("Empty SRS content")

        if state["srs_content"].endswith(".docx"):
            content = process_docx(state["srs_content"])
        else:
            content = state["srs_content"]

        requirements = await analyze_requirements(content)
        
        # Validate JSON response
        try:
            parsed_requirements = json.loads(requirements)
            required_keys = ["functional_requirements", "api_endpoints", "db_schema", "auth_requirements"]
            if not all(key in parsed_requirements for key in required_keys):
                raise ValueError("Missing required keys in parsed requirements")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from requirements analysis")
        
        # Update state with extracted requirements
        state["requirements"] = parsed_requirements
        state["logs"].append("Successfully parsed SRS document")
        
        return state
        
    except Exception as e:
        state["errors"].append(f"SRS parsing error: {str(e)}")
        state["logs"].append(f"Error in SRS parsing: {str(e)}")
        return state
