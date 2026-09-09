from langgraph.types import interrupt

from research_system.state import ResearchState

# Keep only this many most-recent edit instructions in state.edit_history --
# see state.py's comment on why this field is a plain list, not operator.add.
MAX_EDIT_HISTORY = 5


def edit_review_node(state: ResearchState) -> dict:
    """
    Pause the graph after a report is approved and hand it to a human for
    optional follow-up edits. Resumed via
    Command(resume={"edit_instructions": "..."}) with the next instruction,
    or Command(resume={"edit_instructions": None}) (or no instruction at
    all) to stop editing and proceed to END.
    """
    print("[Edit Review] Waiting for a follow-up edit instruction (or none, to finish)")
    decision = interrupt({"type": "edit_review", "final_report": state["final_report"]})
    instruction = (
        (decision.get("edit_instructions") or "").strip() if isinstance(decision, dict) else ""
    )

    if not instruction:
        print("  -> No further edits. Finishing.")
        return {"edit_instructions": ""}

    k = decision.get("edit_k", 2) if isinstance(decision, dict) else 2
    print(f"  -> Edit requested (k={k}): {instruction}")
    history = (state.get("edit_history", []) + [instruction])[-MAX_EDIT_HISTORY:]
    return {
        "edit_instructions": instruction,
        "edit_history": history,
        "edit_section_k": k,
        "revision_count": 0,
        "quality_approved": False,
    }


def should_continue_editing(state: ResearchState) -> str:
    return "edit" if state.get("edit_instructions") else "done"
