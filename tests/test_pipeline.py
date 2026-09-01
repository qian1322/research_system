from langgraph.types import Command

from research_system.graph import build_graph


def _initial_state(topic="测试主题"):
    return {
        "topic": topic,
        "research_plan": [],
        "search_results": [],
        "numbered_sources": [],
        "rag_context": "",
        "draft_report": "",
        "critique": "",
        "revision_count": 0,
        "quality_approved": False,
        "final_report": "",
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
