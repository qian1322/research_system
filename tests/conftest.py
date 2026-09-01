import pytest

from research_system import config
from research_system.nodes.critic import CriticOutput
from research_system.nodes.planner import ResearchPlan


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeStructuredLLM:
    """Stands in for `ChatModel.with_structured_output(schema)`."""

    def __init__(self, schema):
        self._schema = schema

    def invoke(self, messages):
        if self._schema is ResearchPlan:
            return ResearchPlan(
                thinking="fake plan",
                sub_questions=["子问题一", "子问题二"],
            )
        if self._schema is CriticOutput:
            return CriticOutput(
                score=9,
                approved=True,
                improvements=[],
                critique_summary="looks good",
            )
        raise ValueError(f"No fake response configured for schema {self._schema!r}")


class FakeLLM:
    """Stands in for `ChatOpenAI` so tests never hit the network."""

    def invoke(self, messages):
        return FakeMessage("fake llm response")

    def with_structured_output(self, schema, method=None):
        return FakeStructuredLLM(schema)


class FakeTavilySearch:
    def __init__(self, max_results=None, **kwargs):
        pass

    def invoke(self, input):
        return {"results": [{"title": "fake title", "url": "https://example.com", "content": "fake snippet"}]}


class FakeEmbeddings:
    """Deterministic stand-in for a real embedding model, so RAG's Chroma
    step never downloads a model or calls a network API in tests."""

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        return [float(len(text) % 7), float(hash(text) % 11)]


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(config, "get_llm", lambda temperature=0.5: FakeLLM())
    monkeypatch.setattr(config, "get_search_client", lambda max_results=5: FakeTavilySearch())
    monkeypatch.setattr(config, "get_embeddings", lambda: FakeEmbeddings())
    return FakeLLM()
