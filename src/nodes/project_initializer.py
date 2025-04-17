import os
import subprocess
from typing import Dict, Any, Tuple
from langchain_core.runnables import Runnable

class ProjectInitializerNode:
    """Node for initializing the FastAPI project structure."""

    async def create_project_structure(self, requirements: Dict[str, Any]) -> None:
        """Creates the basic folder structure for the FastAPI project."""
        try:
            # Define the project structure
            project_name = "generated_api"  # You might want to extract this from the SRS
            base_dir = project_name
            dirs = [
                base_dir,
                os.path.join(base_dir, "app"),
                os.path.join(base_dir, "app", "api"),
                os.path.join(base_dir, "app", "models"),
                os.path.join(base_dir, "app", "services"),
                os.path.join(base_dir, "app", "api", "routes"),
            ]

            # Create the directories
            for dir_path in dirs:
                os.makedirs(dir_path, exist_ok=True)

            # Create a basic main.py file
            main_py_content = """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
"""
            with open(os.path.join(base_dir, "app", "main.py"), "w") as f:
                f.write(main_py_content)

            # Create a basic requirements.txt file
            requirements_content = """
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
"""
            with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
                f.write(requirements_content)

        except Exception as e:
            raise Exception(f"Error creating project structure: {e}")

    async def setup_podman_postgres(self, requirements: Dict[str, Any]) -> None:
        """Sets up a PostgreSQL database using Podman."""
        try:
            # Define the podman command
            podman_command = [
                "podman",
                "run",
                "-d",
                "--name",
                "postgres_db",
                "-e",
                "POSTGRES_USER=postgres",
                "-e",
                "POSTGRES_PASSWORD=postgres",
                "-p",
                "5432:5432",
                "postgres:latest",
            ]

            # Execute the podman command
            subprocess.run(podman_command, check=True)

        except subprocess.CalledProcessError as e:
            raise Exception(f"Error setting up Podman PostgreSQL: {e}")

    async def create_virtual_environment(self, requirements: Dict[str, Any]) -> None:
        """Creates a virtual environment and installs dependencies."""
        try:
            project_name = "generated_api"
            venv_path = os.path.join(project_name, "venv")

            # Create the virtual environment
            subprocess.run(["python", "-m", "venv", venv_path], check=True)

            # Activate the virtual environment (platform-specific)
            activate_script = os.path.join(venv_path, "bin", "activate")  # Linux/macOS
            if os.name == "nt":  # Windows
                activate_script = os.path.join(venv_path, "Scripts", "activate")

            # Install dependencies
            pip_install_command = [activate_script, "&&", "pip", "install", "-r", os.path.join(project_name, "requirements.txt")]
            subprocess.run(pip_install_command, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            raise Exception(f"Error creating virtual environment: {e}")

    async def run(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Runs the project initialization node."""
        try:
            requirements = state.get("requirements", {})

            await self.create_project_structure(requirements)
            await self.setup_podman_postgres(requirements)
            await self.create_virtual_environment(requirements)

            state["logs"].append("Successfully initialized project structure")
            return state, "test_generator"  # Next node in the graph

        except Exception as e:
            state["errors"].append(f"Project initialization error: {e}")
            state["logs"].append(f"Error initializing project: {e}")
            return state, "error_handler"
