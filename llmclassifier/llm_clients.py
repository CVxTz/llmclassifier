import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

if (Path(__file__).parents[1] / ".env").is_file():
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct")

llm_openai_client = ChatOpenAI(
    openai_api_base=os.environ.get("LLM_BASE_URL"),
    model=MODEL_NAME,
    openai_api_key=os.environ.get("LLM_API_KEY"),
    temperature=0,
    max_retries=10,
)
