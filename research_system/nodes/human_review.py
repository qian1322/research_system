from langgraph.types import interrupt

from research_system.state import ResearchState


def human_review_node(state: ResearchState) -> dict:
    """
    Pause the graph and hand the Planner's sub-questions to a human.
    Resumed via Command(resume={"research_plan": [...]}) with the
    approved (possibly edited) list of sub-questions.
    """
    plan = state["research_plan"]
    print(f"[Human Review] Waiting for approval of {len(plan)} sub-questions")
    decision = interrupt(
        {
            "type": "plan_review",
            "topic": state["topic"],
            "research_plan": plan,
        }
    )
    approved_plan = decision.get("research_plan", plan) if isinstance(decision, dict) else plan
    print(f"  -> Approved {len(approved_plan)} sub-questions")
    return {"research_plan": approved_plan}
