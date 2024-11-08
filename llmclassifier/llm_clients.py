import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

if (Path(__file__).parents[1] / ".env").is_file():
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


MEDIUM_MODEL = os.environ.get("MEDIUM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LARGE_MODEL = os.environ.get("LARGE_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")

llm_medium = ChatOpenAI(
    openai_api_base=os.environ.get("LLM_BASE_URL"),
    model=MEDIUM_MODEL,
    openai_api_key=os.environ.get("LLM_API_KEY"),
    temperature=0,
    max_retries=10,
)

lmm_large = ChatOpenAI(
    openai_api_base=os.environ.get("LLM_BASE_URL"),
    model=LARGE_MODEL,
    openai_api_key=os.environ.get("LLM_API_KEY"),
    temperature=0,
    max_retries=10,
)
