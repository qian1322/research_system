from typing import List

from langgraph.types import Send
from pydantic import BaseModel

from research_system import config
from research_system.prompts import planner_prompt
from research_system.state import ResearchState


class ResearchPlan(BaseModel):
    thinking: str
    sub_questions: List[str]  # 3-5 focused sub-questions


def planner_node(state: ResearchState) -> dict:
    print(f'[Planner] {state["topic"][:50]}')
    prompt = planner_prompt(state["topic"])
    planner_llm = config.get_llm(temperature=0.3).with_structured_output(
        ResearchPlan, method="function_calling"
    )
    plan = planner_llm.invoke([("user", prompt)])
    print(f"  -> {len(plan.sub_questions)} sub-questions")
    for i, q in enumerate(plan.sub_questions, 1):
        print(f"     {i}. {q[:60]}")
    return {"research_plan": plan.sub_questions, "search_results": []}


def dispatch_search(state: ResearchState):
    """
    Returns List[Send] -> LangGraph runs ALL in parallel.

    Each Send(node_name, sub_state) is an independent invocation.
    Results from all agents merge via the operator.add reducer on search_results.
    """
    print(f'[Send API] Dispatching {len(state["research_plan"])} parallel agents')
    return [
        Send(
            "search_agent",
            {
                "topic": state["topic"],
                "sub_question": q,
                # Must initialize all fields for the sub-state
                "research_plan": [],
                "search_results": [],
                "numbered_sources": [],
                "source_titles": [],
                "rag_context": "",
                "draft_report": "",
                "critique": "",
                "revision_count": 0,
                "quality_approved": False,
                "final_report": "",
            },
        )
        for q in state["research_plan"]
    ]
