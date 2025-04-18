import os
import groq
from typing import Dict, Any, Tuple, List
from langchain.prompts import PromptTemplate
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

class CodeGeneratorNode:
    """Node for generating application code based on the extracted requirements using LLM."""

    def __init__(self):
        self.app_dir = "generated_api/app"
        self.app_dirs = {
            "routes": os.path.join(self.app_dir, "api", "routes"),
            "models": os.path.join(self.app_dir, "models"),
            "services": os.path.join(self.app_dir, "services"),
            "root": self.app_dir,
            "api": os.path.join(self.app_dir, "api")
        }
        
        # Initialize Groq client directly instead of using GroqWrapper
        self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"  # Update to your preferred model
        
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
    
    async def _ensure_directories_exist(self):
        """Ensure all required directories exist."""
        for dir_path in self.app_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    async def _create_model_prompt(self, model: str, requirements: Dict[str, Any]) -> str:
        """Creates a prompt for SQLAlchemy model generation."""
        prompt_template = PromptTemplate(
            input_variables=["model", "requirements"],
            template="""
            Generate a SQLAlchemy model class for FastAPI.
            
            Model: {model}
            Requirements: {requirements}
            
            Include:
            1. Model class with proper inheritance
            2. All required fields with proper types
            3. Relationships to other models
            4. Database constraints
            5. Field validation
            6. Pydantic model for API
            
            Follow these rules:
            - Use SQLAlchemy declarative base
            - Add proper indices
            - Include proper foreign keys
            - Handle cascading deletes
            - Add __repr__ method
            - Add any required methods
            
            The code should be complete and ready to use.
            """
        )
        return prompt_template.format(
            model=model,
            requirements=requirements
        )

    async def _create_route_prompt(self, endpoint: str, requirements: Dict[str, Any]) -> str:
        """Creates a prompt for FastAPI route generation."""
        prompt_template = PromptTemplate(
            input_variables=["endpoint", "requirements"],
            template="""
            Generate a FastAPI route handler for the endpoint.
            
            Endpoint: {endpoint}
            Requirements: {requirements}
            
            Include:
            1. All CRUD operations
            2. Input validation
            3. Error handling
            4. Authentication checks
            5. Database operations
            6. Response models
            
            Follow these rules:
            - Use FastAPI dependency injection
            - Include proper status codes
            - Add OpenAPI documentation
            - Handle database sessions
            - Include error responses
            - Add proper logging
            
            The code should be complete and ready to use.
            """
        )
        return prompt_template.format(
            endpoint=endpoint,
            requirements=requirements
        )

    async def _create_service_prompt(self, model: str, requirements: Dict[str, Any]) -> str:
        """Creates a prompt for service layer generation."""
        prompt_template = PromptTemplate(
            input_variables=["model", "requirements"],
            template="""
            Generate a service class for business logic.
            
            Model: {model}
            Requirements: {requirements}
            
            Include:
            1. Business logic methods
            2. Database operations
            3. Data validation
            4. Error handling
            5. Integration points
            
            Follow these rules:
            - Use dependency injection
            - Handle transactions
            - Add proper error types
            - Include logging
            - Add docstrings
            - Handle edge cases
            
            The code should be complete and ready to use.
            """
        )
        return prompt_template.format(
            model=model,
            requirements=requirements
        )

    async def generate_models(self, models: List[str], requirements: Dict[str, Any]) -> None:
        """Generates SQLAlchemy models."""
        try:
            os.makedirs(self.app_dirs["models"], exist_ok=True)

            for model in models:
                run = self.langsmith_client.create_run(
                    name="generate_model_code",
                    inputs={"model": model}
                )

                prompt = await self._create_model_prompt(model, requirements)
                response = await self.llm.ainvoke(prompt)
                model_code = response.content

                model_file = os.path.join(self.app_dirs["models"], f"{model.lower()}.py")
                with open(model_file, "w") as f:
                    f.write(model_code)

                self.langsmith_client.update_run(
                    run.id,
                    outputs={"file": model_file}
                )

        except Exception as e:
            self.langsmith_client.create_run(
                name="model_generation_error",
                error=str(e)
            )
            raise Exception(f"Error generating model code: {e}")

    async def generate_routes(self, endpoints: List[str], requirements: Dict[str, Any]) -> None:
        """Generates FastAPI route handlers."""
        try:
            os.makedirs(self.app_dirs["routes"], exist_ok=True)

            for endpoint in endpoints:
                run = self.langsmith_client.create_run(
                    name="generate_route_code",
                    inputs={"endpoint": endpoint}
                )

                prompt = await self._create_route_prompt(endpoint, requirements)
                response = await self.llm.ainvoke(prompt)
                route_code = response.content

                resource = endpoint.split('/')[-1]
                route_file = os.path.join(self.app_dirs["routes"], f"{resource}.py")
                with open(route_file, "w") as f:
                    f.write(route_code)

                self.langsmith_client.update_run(
                    run.id,
                    outputs={"file": route_file}
                )

        except Exception as e:
            self.langsmith_client.create_run(
                name="route_generation_error",
                error=str(e)
            )
            raise Exception(f"Error generating route code: {e}")

    async def generate_services(self, models: List[str], requirements: Dict[str, Any]) -> None:
        """Generates service layer classes."""
        try:
            os.makedirs(self.app_dirs["services"], exist_ok=True)

            for model in models:
                run = self.langsmith_client.create_run(
                    name="generate_service_code",
                    inputs={"model": model}
                )

                prompt = await self._create_service_prompt(model, requirements)
                response = await self.llm.ainvoke(prompt)
                service_code = response.content

                service_file = os.path.join(self.app_dirs["services"], f"{model.lower()}_service.py")
                with open(service_file, "w") as f:
                    f.write(service_code)

                self.langsmith_client.update_run(
                    run.id,
                    outputs={"file": service_file}
                )

        except Exception as e:
            self.langsmith_client.create_run(
                name="service_generation_error",
                error=str(e)
            )
            raise Exception(f"Error generating service code: {e}")

    async def run(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Runs the code generation node."""
        # Change from keyword args to positional args (name as first arg)
        run = self.langsmith_client.create_run(
            "code_generator_run",  # Changed from name="code_generator_run"
            inputs=state.get("requirements", {})
        )
        try:
            requirements = state.get("requirements", {})
            endpoints = requirements.get("endpoints", [])
            models = requirements.get("models", [])

            # Create directories first
            await self._ensure_directories_exist()
            
            await self.generate_models(models, requirements)
            await self.generate_routes(endpoints, requirements)
            await self.generate_services(models, requirements)

            state["logs"].append("Successfully generated application code")
            self.langsmith_client.update_run(
                run.id,
                outputs={"status": "success"}
            )
            return state, "debugger"  # Next node in the graph

        except Exception as e:
            error_msg = f"Code generation error: {e}"
            state["errors"].append(error_msg)
            state["logs"].append(f"Error generating code: {e}")
            self.langsmith_client.update_run(
                run.id,
                error=error_msg
            )
            return state, "error_handler"
