import os
import groq
from typing import Dict, Any, Tuple, List
from langchain.prompts import PromptTemplate
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

class TestGeneratorNode:
    """Node for generating test cases based on the extracted requirements using LLM."""

    def __init__(self):
        self.test_dir = "generated_api/tests"
        self.test_dirs = {
            "routes": os.path.join(self.test_dir, "test_routes"),
            "models": os.path.join(self.test_dir, "test_models"),
            "auth": self.test_dir
        }
        
        # Initialize both direct client and LangChain-compatible interface
        self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mistral-saba-24b"  # Updated model name
        
        # This will be replaced by a mock in tests
        self.llm = self
        self.langsmith_client = Client()
    
    async def ainvoke(self, prompt: str) -> Any:
        """Method to match LangChain's interface, calls Groq directly"""
        return await self._invoke_groq(prompt)
    
    async def _invoke_groq(self, prompt: str) -> Any:
        """Helper method to call Groq API directly"""
        completion = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        # Create a response object with a .content attribute
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(completion.choices[0].message.content)

    async def _create_api_test_prompt(self, endpoint: str, requirements: Dict[str, Any]) -> str:
        """Creates a prompt for API test generation."""
        return f"""
        Generate pytest test cases for the FastAPI endpoint: {endpoint}
        
        Requirements: {requirements}
        
        Include tests for:
        - Valid requests
        - Invalid requests
        - Authentication/authorization
        - Edge cases
        
        Use FastAPI TestClient for all tests.
        """

    async def _create_model_test_prompt(self, model: str, requirements: Dict[str, Any]) -> str:
        """Creates a prompt for model test generation."""
        return f"""
        Generate pytest test cases for the SQLAlchemy model: {model}
        
        Requirements: {requirements}
        
        Include tests for:
        - Model instantiation
        - Field validation
        - Relationships
        - CRUD operations
        
        Use pytest fixtures and a test database.
        """

    async def _ensure_directories_exist(self):
        """Ensure all required directories exist."""
        for dir_path in self.test_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    async def generate_api_tests(self, endpoints: List[str], requirements: Dict[str, Any]) -> None:
        """Generates test cases for API endpoints using LLM."""
        try:
            self.langsmith_client.create_run(
                name="generate_api_tests",
                inputs={"endpoints": endpoints, "requirements": requirements}
            )
            
            await self._ensure_directories_exist()
            
            for endpoint in endpoints:
                prompt = await self._create_api_test_prompt(endpoint, requirements)
                response = await self.llm.ainvoke(prompt)  # Use llm interface for mockability
                test_content = response.content
                
                resource = endpoint.split('/')[-1]
                test_file = os.path.join(self.test_dirs["routes"], f"test_{resource}.py")
                
                with open(test_file, "w") as f:
                    f.write(test_content)
                
                self.langsmith_client.create_run(
                    name="api_test_generated", 
                    inputs={"file": test_file}
                )
        
        except Exception as e:
            self.langsmith_client.create_run(
                name="api_test_error",
                error=str(e)
            )
            raise Exception(f"Error generating API tests: {e}")

    async def generate_model_tests(self, models: List[str], requirements: Dict[str, Any]) -> None:
        """Generates test cases for database models using LLM."""
        try:
            self.langsmith_client.create_run(
                name="generate_model_tests",
                inputs={"models": models, "requirements": requirements}
            )
            
            await self._ensure_directories_exist()
            
            for model in models:
                prompt = await self._create_model_test_prompt(model, requirements)
                response = await self.llm.ainvoke(prompt)  # Use llm interface for mockability
                test_content = response.content
                
                test_file = os.path.join(self.test_dirs["models"], f"test_{model.lower()}.py")
                
                with open(test_file, "w") as f:
                    f.write(test_content)
                
                self.langsmith_client.create_run(
                    name="model_test_generated",
                    inputs={"file": test_file}
                )
        
        except Exception as e:
            self.langsmith_client.create_run(
                name="model_test_error",
                error=str(e)
            )
            raise Exception(f"Error generating model tests: {e}")

    async def generate_auth_tests(self, auth_config: Dict[str, Any]) -> None:
        """Generates test cases for authentication and authorization using LLM."""
        try:
            self.langsmith_client.create_run(
                name="generate_auth_tests",
                inputs={"auth_config": auth_config}
            )

            await self._ensure_directories_exist()
            
            prompt_template = PromptTemplate(
                input_variables=["auth_config"],
                template="""
                Generate pytest test cases for FastAPI authentication and authorization.

                Authentication Config: {auth_config}

                Include tests for:
                1. User registration
                2. Login/logout
                3. Token handling
                4. Protected routes
                5. Permission checks

                Follow these rules:
                - Use FastAPI TestClient
                - Test both success and failure cases
                - Include token validation
                - Test expiry and refresh
                - Use secure test credentials

                The test code should be complete and ready to use.
                """
            )

            # Generate test code using LLM
            prompt = prompt_template.format(auth_config=auth_config)
            response = await self.llm.ainvoke(prompt)  # Use llm interface for mockability
            test_content = response.content

            # Write test file
            test_file = os.path.join(self.test_dirs["auth"], "test_auth.py")
            
            with open(test_file, "w") as f:
                f.write(test_content)

            self.langsmith_client.create_run(
                name="auth_test_generated",
                inputs={"file": test_file}
            )

        except Exception as e:
            self.langsmith_client.create_run(
                name="auth_test_error",
                error=str(e)
            )
            raise Exception(f"Error generating auth tests: {e}")

    async def run(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Runs the test generation node."""
        try:
            # Change from keyword args to positional args (name as first arg)
            self.langsmith_client.create_run(
                "test_generator_run",  # Changed from name="test_generator_run"
                inputs={"state": state}
            )
            
            requirements = state.get("requirements", {})
            
            # Create test directories
            await self._ensure_directories_exist()
            
            # Generate tests for endpoints
            if "endpoints" in requirements:
                await self.generate_api_tests(requirements["endpoints"], requirements)
            
            # Generate tests for models
            if "models" in requirements:
                await self.generate_model_tests(requirements["models"], requirements)
            
            # Generate authentication tests
            if "auth" in requirements:
                await self.generate_auth_tests(requirements["auth"])
            
            # Update state
            state["logs"].append("Successfully generated test cases")
            
            self.langsmith_client.update_run(
                "test_generator_run",
                outputs={"status": "success", "state": state}
            )
            
            return state, "code_generator"
        
        except Exception as e:
            error_msg = f"Test generation error: {e}"
            state["errors"].append(error_msg)
            state["logs"].append(f"Error generating tests: {e}")
            
            self.langsmith_client.update_run(
                run_id="test_run_id",
                error=str(e)
            )
            
            return state, "error_handler"
