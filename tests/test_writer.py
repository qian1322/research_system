from research_system import config
from research_system.nodes.writer import writer_node

# Deliberately a single section: with only one candidate, retrieval can't
# pick "the wrong one" -- makes the section-edit path's outcome deterministic
# without depending on FakeEmbeddings' non-semantic vectors.
REPORT_ONE_SECTION = (
    "# Topic Research Report\n"
    "Some preamble.\n\n"
    "## Key Findings\n"
    "Findings body [1].\n\n"
    "## References\n"
    "[1] https://example.com/a\n"
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    """Returns responses[i] on the i-th call (last one repeats if exhausted)."""

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def invoke(self, messages):
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return FakeMessage(content)


def _edit_state(**overrides):
    state = {
        "topic": "test topic",
        "rag_context": "context",
        "final_report": REPORT_ONE_SECTION,
        "search_results": [],
        "edit_instructions": "add a case to key findings",
        "edit_history": ["add a case to key findings"],
        "edit_section_k": 2,
        "revision_count": 0,
    }
    state.update(overrides)
    return state


def test_writer_edits_only_matched_section(fake_llm, monkeypatch):
    scripted = ScriptedLLM(["## Key Findings\nFindings body [1], now with a case study."])
    monkeypatch.setattr(config, "get_llm", lambda temperature=0.5: scripted)

    result = writer_node(_edit_state())
    draft = result["draft_report"]

    assert "Findings body [1], now with a case study." in draft
    assert draft.startswith("# Topic Research Report\nSome preamble.\n\n")
    assert draft.count("## References") == 1
    assert "[1] https://example.com/a" in draft
    assert scripted.calls == 1  # section edit succeeded, no fallback needed


REPORT_TWO_SECTIONS = (
    "# Topic Research Report\n"
    "Some preamble.\n\n"
    "## Executive Summary\n"
    "Summary body.\n\n"
    "## Conclusions\n"
    "Conclusions body [1].\n\n"
    "## References\n"
    "[1] https://example.com/a\n"
)


def test_writer_prefers_keyword_match_over_embedding_retrieval(fake_llm, monkeypatch):
    """Regression test for a real manual run: an instruction naming a
    section in Chinese ("结论部分") got matched to the wrong section by
    embedding retrieval alone. Keyword matching should pick "## Conclusions"
    deterministically regardless of what the (fake, non-semantic) embeddings
    would have picked."""
    scripted = ScriptedLLM(["## Conclusions\nConclusions body [1], now with a limitations note."])
    monkeypatch.setattr(config, "get_llm", lambda temperature=0.5: scripted)

    state = _edit_state(
        final_report=REPORT_TWO_SECTIONS,
        edit_instructions="在结论部分补充一句关于局限性的说明",
    )
    draft = writer_node(state)["draft_report"]

    assert "Conclusions body [1], now with a limitations note." in draft
    assert "## Executive Summary\nSummary body." in draft  # untouched, byte-identical
    assert scripted.calls == 1


def test_writer_falls_back_to_whole_report_when_llm_ignores_format(fake_llm, monkeypatch):
    scripted = ScriptedLLM(["Sorry, I can't help with that."])  # no "## " heading at all
    monkeypatch.setattr(config, "get_llm", lambda temperature=0.5: scripted)

    result = writer_node(_edit_state())

    assert result["draft_report"] == "Sorry, I can't help with that."
    assert scripted.calls == 2  # one failed section-edit attempt + one whole-report fallback
