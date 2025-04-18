import os
import pytest
import groq
from typing import Dict, Any, Tuple, List
from langchain_core.runnables import Runnable
from langsmith import Client
from langchain.prompts import PromptTemplate

class DebuggerNode:
    """Node for analyzing test failures and suggesting/implementing fixes."""

    def __init__(self):
        self.app_dir = "generated_api/app"
        self.test_dir = "generated_api/tests"
        # Initialize Groq and LangSmith clients
        self.groq_client = groq.Client()
        self.model = "mistral-saba-24b"  # Updated model name
        self.langsmith_client = Client()

    async def _create_debug_prompt(self, test_output: str, file_content: str) -> str:
        """Creates a prompt for analyzing test failures."""
        prompt_template = PromptTemplate(
            input_variables=["test_output", "file_content"],
            template="""
            Analyze the following test failure and suggest fixes.
            
            Test Output:
            {test_output}
            
            Current Code:
            {file_content}
            
            Please:
            1. Identify the root cause of the failure
            2. Suggest specific code changes
            3. Consider edge cases and error handling
            4. Ensure compliance with FastAPI best practices
            5. Maintain existing functionality
            
            Provide the corrected code that should fix the test failure.
            """
        )
        return prompt_template.format(
            test_output=test_output,
            file_content=file_content
        )

    async def run_tests(self) -> Tuple[bool, str]:
        """Runs the test suite and returns results."""
        try:
            pytest_output = pytest.main([self.test_dir, "-v"])
            return pytest_output == 0, str(pytest_output)
        except Exception as e:
            return False, str(e)

    def _extract_failing_files(self, test_output: str) -> List[str]:
        """Extract failing test files from test output."""
        failing_files = []
        
        # Look for lines with FAILED
        for line in test_output.splitlines():
            line = line.strip()
            if "::" in line and "FAILED" in line:
                parts = line.split("::")
                file_path = parts[0]
                
                # Handle the path correctly - if it starts with "tests/"
                if file_path.startswith("tests/") or file_path.startswith("tests\\"):
                    # Get just the file name - remove the "tests/" part
                    file_name = os.path.basename(file_path)
                    # Create the full path properly
                    full_path = os.path.join(self.test_dir, file_name)
                else:
                    # If it doesn't already have "tests/" prefix
                    full_path = os.path.join(self.test_dir, file_path)
                
                failing_files.append(full_path)
        
        return failing_files

    async def fix_test_failures(self, test_output: str) -> None:
        """Analyzes test failures and implements fixes."""
        try:
            run = await self.langsmith_client.create_run(
                name="debug_test_failures",
                inputs={"test_output": test_output}
            )

            # Analyze each failing test
            failing_files = self._extract_failing_files(test_output)
            for file_path in failing_files:
                with open(file_path, 'r') as f:
                    current_code = f.read()

                # Generate fix using LLM
                prompt = await self._create_debug_prompt(test_output, current_code)
                chat_completion = await self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                fixed_code = chat_completion.choices[0].message.content

                # Apply fix
                with open(file_path, 'w') as f:
                    f.write(fixed_code)

                await self.langsmith_client.create_run(
                    name="fix_applied",
                    inputs={"file": file_path}
                )

            await self.langsmith_client.update_run(
                run.id,
                outputs={"status": "fixes_applied"}
            )

        except Exception as e:
            await self.langsmith_client.create_run(
                name="debug_error",
                error=str(e)
            )
            raise Exception(f"Error during debugging: {e}")

    async def run(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Runs the debugger node."""
        run = await self.langsmith_client.create_run(
            name="debugger_run",
            inputs=state.get("requirements", {})
        )
        try:
            # Run tests
            tests_passed, test_output = await self.run_tests()

            if tests_passed:
                state["logs"].append("All tests passed successfully")
                await self.langsmith_client.update_run(
                    run.id,
                    outputs={"status": "success"}
                )
                return state, "documentation_generator"  # Next node in the graph

            # Fix failing tests
            state["logs"].append("Found test failures, attempting fixes")
            await self.fix_test_failures(test_output)

            # Re-run tests after fixes
            tests_passed, test_output = await self.run_tests()
            if tests_passed:
                state["logs"].append("Successfully fixed all test failures")
                await self.langsmith_client.update_run(
                    run.id,
                    outputs={"status": "fixed"}
                )
                return state, "documentation_generator"
            else:
                raise Exception("Unable to fix all test failures")

        except Exception as e:
            error_msg = f"Debugging error: {e}"
            state["errors"].append(error_msg)
            state["logs"].append(f"Error during debugging: {e}")
            await self.langsmith_client.update_run(
                run.id,
                error=error_msg
            )
            return state, "error_handler"
