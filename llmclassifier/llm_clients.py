import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from llmclassifier.logger import logger

if (Path(__file__).parents[1] / ".env").is_file():
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


OPENAI_MODEL_NAME = os.environ.get(
    "OPENAI_MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct"
)
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "llama-3.2-90b-vision-preview")
FIREWORKS_MODEL_NAME = os.environ.get(
    "FIREWORKS_MODEL_NAME", "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
)
GOOGLE_MODEL_NAME = os.environ.get("GOOGLE_MODEL_NAME", "gemini-1.5-flash-002")

if os.environ.get("OPENAI_API_KEY"):
    llm_openai_client = ChatOpenAI(
        openai_api_base=os.environ.get("LLM_BASE_URL"),
        model=OPENAI_MODEL_NAME,
        temperature=0,
        max_retries=10,
    )
else:
    logger.warning("OpenAI Client not set")
    llm_openai_client = None


if os.environ.get("GROQ_API_KEY"):
    llm_groq_client = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0,
        max_retries=10,
    )
else:
    logger.warning("Groq Client not set")
    llm_groq_client = None

if os.environ.get("GROQ_API_KEY"):
    llm_groq_client = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0,
        max_retries=10,
    )
else:
    logger.warning("Groq Client not set")
    llm_groq_client = None

if os.environ.get("FIREWORKS_API_KEY"):
    llm_fireworks_client = ChatFireworks(
        model=FIREWORKS_MODEL_NAME,
        temperature=0,
        max_retries=10,
    )
else:
    logger.warning("Fireworks Client not set")
    llm_fireworks_client = None


if os.environ.get("GOOGLE_API_KEY"):
    llm_google_client = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL_NAME,
        temperature=0,
        max_retries=10,
    )
else:
    logger.warning("Google Client not set")
    llm_google_client = None
