from langgraph.types import Command

from research_system.graph import build_graph


def _initial_state(topic="测试主题"):
    return {
        "topic": topic,
        "research_plan": [],
        "citation_style": "",
        "search_results": [],
        "numbered_sources": [],
        "source_titles": [],
        "rag_context": "",
        "draft_report": "",
        "critique": "",
        "revision_count": 0,
        "quality_approved": False,
        "final_report": "",
        "edit_instructions": "",
        "edit_history": [],
        "edit_section_k": 2,
    }


def test_full_pipeline_runs_end_to_end(fake_llm):
    app = build_graph()
    config = {"configurable": {"thread_id": "test-run"}}

    # Planner runs, then the graph pauses for human review of the plan.
    result = app.invoke(_initial_state(), config=config)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["research_plan"] == ["子问题一", "子问题二"]

    # Approve the plan as-is to resume.
    result = app.invoke(
        Command(resume={"research_plan": payload["research_plan"]}), config=config
    )

    # Planner -> 2 fake sub-questions -> 2 parallel search_agent results merged via operator.add
    assert len(result["search_results"]) == 2
    assert result["rag_context"] == "fake llm response"
    assert result["draft_report"] == "fake llm response"
    # Fake critic always approves on the first pass
    assert result["quality_approved"] is True
    assert result["final_report"] == "fake llm response"
    assert result["revision_count"] == 0

    # Approval no longer goes straight to END -- it pauses at edit_review
    # for an optional follow-up instruction.
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "edit_review"
    assert payload["final_report"] == "fake llm response"

    # Decline to edit further -> graph proceeds to END.
    result = app.invoke(Command(resume={"edit_instructions": None}), config=config)
    assert "__interrupt__" not in result
    assert result["final_report"] == "fake llm response"


def test_edit_review_loops_back_through_writer_and_critic(fake_llm):
    """A follow-up edit instruction should route writer -> critic -> edit_review
    again (not straight to END), with a fresh revision_count for the new round.

    Note: fake_llm's fixed "fake llm response" text has no "## " headings, so
    this only exercises the whole-report edit fallback in writer.py's
    _apply_section_edit, not the section-targeted retrieval path -- see
    tests/test_writer.py for that."""
    app = build_graph()
    config = {"configurable": {"thread_id": "test-run-edit-loop"}}

    result = app.invoke(_initial_state(), config=config)
    payload = result["__interrupt__"][0].value
    result = app.invoke(
        Command(resume={"research_plan": payload["research_plan"]}), config=config
    )
    assert result["__interrupt__"][0].value["type"] == "edit_review"

    # Submit a follow-up edit instruction.
    result = app.invoke(
        Command(resume={"edit_instructions": "add a limitations note"}), config=config
    )

    # Fake critic always approves on the first pass, so this edit round's
    # writer/critic loop runs exactly once before pausing at edit_review again.
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "edit_review"
    assert result["revision_count"] == 0
    assert result["edit_history"] == ["add a limitations note"]

    # Decline further edits -> graph proceeds to END.
    result = app.invoke(Command(resume={"edit_instructions": None}), config=config)
    assert "__interrupt__" not in result


def test_human_review_can_edit_plan(fake_llm):
    """Trimming a sub-question at the human-review step should fan out fewer search agents."""
    app = build_graph()
    config = {"configurable": {"thread_id": "test-run-edit"}}

    result = app.invoke(_initial_state(), config=config)
    payload = result["__interrupt__"][0].value

    edited_plan = payload["research_plan"][:1]
    result = app.invoke(Command(resume={"research_plan": edited_plan}), config=config)

    assert result["research_plan"] == edited_plan
    assert len(result["search_results"]) == 1
