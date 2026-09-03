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
        "list only the numbers actually cited in the body, formatted as '[n] url'.\n\n"
        "Fact rule: do not invent specific numbers, percentages, statistics, company "
        "names, or case studies that do not appear in the context above -- and do not "
        "round or blend real numbers from the context into a new one that doesn't "
        "actually appear there either (e.g. don't turn a source's 79% into 85%). If "
        "the context doesn't contain a concrete figure or example for a point you "
        "want to make, use honest hedged phrasing instead, such as '数据显示...呈上升"
        "趋势' or '部分来源提及...' -- an accurate hedge beats a precise fabrication. "
        "This applies to '## Case Analysis' too: only analyze cases that are actually "
        "present in the context; if none are present, discuss patterns or trends from "
        "the context instead of inventing a fictional case."
    )
    writer_llm = config.get_llm(temperature=0.7)
    response = writer_llm.invoke([("user", prompt)])
    print(f"  -> Draft: {len(response.content)} chars")
    return {"draft_report": response.content}
