import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Optional: used for RAG embeddings only (DeepSeek has no embeddings API).
# Falls back to a local HuggingFace model when unset -- see get_embeddings().
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Optional: LangSmith tracing. Wiring the env vars here (once, at import time)
# is enough for every LangChain/LangGraph call in the process to get traced.
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "deep-research-system")
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY


def get_llm(temperature: float = 0.5) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=temperature,
    )


def get_search_client(max_results: int = 5) -> TavilySearch:
    return TavilySearch(max_results=max_results)


_embeddings = None


def get_embeddings():
    """
    Embeddings for the RAG vector store. OpenAI's if OPENAI_API_KEY is set
    (best quality, needs a key); otherwise a local HuggingFace model, so RAG
    still works with zero extra cost or keys.

    Cached at module level: loading the local model is expensive and every
    research() call should reuse the same instance.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings

        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    return _embeddings
