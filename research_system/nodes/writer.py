from research_system import config
from research_system.state import ResearchState


def writer_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    if revision == 0:
        print("[Writer] Writing initial draft...")
        task = "Write a complete, professional research report."
    else:
        print(f"[Writer] Revising (attempt {revision})...")
        task = f'Revise based on:\n{state.get("critique", "")}'

    numbered_sources = state.get("numbered_sources", [])
    sources_text = "\n".join(
        f"[{i}] {s}" for i, s in enumerate(numbered_sources, 1) if s
    )

    prompt = (
        f'Topic: {state["topic"]}\n'
        f'Context (from RAG retrieval; [n] markers are citation numbers):\n{state["rag_context"]}\n\n'
        f"Sources for each citation number:\n{sources_text}\n\n"
        f"Task: {task}\n\n"
        "Format (in Chinese):\n"
        "# [Topic] Research Report\n"
        "## Executive Summary\n"
        "## Background\n"
        "## Key Findings\n"
        "## Case Analysis\n"
        "## Conclusions\n"
        "## References\n\n"
        "Citation rule: keep every [n] marker from the context exactly as written "
        "in the body -- don't remove, renumber, or invent one. In '## References', "
        "list only the numbers actually cited in the body, formatted as '[n] url'."
    )
    writer_llm = config.get_llm(temperature=0.7)
    response = writer_llm.invoke([("user", prompt)])
    print(f"  -> Draft: {len(response.content)} chars")
    return {"draft_report": response.content}
