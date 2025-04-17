from typing import Dict, Any, Tuple
from langgraph.graph import Node
from groq import Groq
import os
from docx import Document
import json

# Initialize Groq client for LLaMA 3 70B
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLAMA_3_70B = "llama2-70b-4096"

class SRSParserNode(Node):
    """Node for parsing SRS documents and extracting requirements using LLaMA 3 70B."""
    
    async def analyze_requirements(self, content: str) -> str:
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
        
        response = groq_client.chat.completions.create(
            model=LLAMA_3_70B,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2,
            max_tokens=4000,
        )
        
        return response.choices[0].message.content

    def process_docx(self, file_path: str) -> str:
        """Extract text content from .docx file."""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    async def run(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Run the SRS parsing node."""
        try:
            if not state["srs_content"]:
                raise ValueError("Empty SRS content")

            if state["srs_content"].endswith(".docx"):
                content = self.process_docx(state["srs_content"])
            else:
                content = state["srs_content"]

            requirements = await self.analyze_requirements(content)
            
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
            
            # Return updated state and next node
            return state, "project_initializer"
            
        except Exception as e:
            state["errors"].append(f"SRS parsing error: {str(e)}")
            state["logs"].append(f"Error in SRS parsing: {str(e)}")
            return state, "error_handler"
