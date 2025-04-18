import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

print(os.getenv("LANGCHAIN_API_KEY"))