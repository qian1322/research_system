from research_system import config
from research_system.nodes.critic import build_source_excerpt_map_from_numbered_sources
from research_system.state import ResearchState


def writer_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    if revision == 0:
        print("[Writer] Writing initial draft...")
        task = "Write a complete, professional research report."
    else:
        print(f"[Writer] Revising (attempt {revision})...")
        task = f'Revise based on:\n{state.get("critique", "")}'

    # Previously this was just "[n] url" -- the Writer could see a citation
    # number existed but never the actual source text behind it, so it wrote
    # specific claims from a blurry, multiply-summarized rag_context and
    # attached whichever [n] felt contextually close. The eval harness's
    # judge caught the result: faithfulness/citation_appropriateness scores
    # never once reached 4/5 across 24 real eval runs (see README's
    # Evaluation section) -- every run had claims invented or misattributed
    # to a source that didn't say them. Giving the Writer the same per-
    # citation excerpts the Critic already fact-checks against lets it
    # ground claims while drafting instead of only being told after the fact.
    numbered_sources = state.get("numbered_sources", [])
    excerpts = build_source_excerpt_map_from_numbered_sources(
        numbered_sources, state.get("search_results", [])
    )
    sources_text = "\n\n".join(
        f"[{i}] {url}\n{excerpts.get(i, '')}"
        for i, url in enumerate(numbered_sources, 1) if url
    )

    prompt = (
        f'Topic: {state["topic"]}\n'
        f'Context (from RAG retrieval; [n] markers are citation numbers):\n{state["rag_context"]}\n\n'
        f"Sources for each citation number:\n{sources_text}\n\n"
        f"Task: {task}\n\n"
        "Format (respond in the same language as the topic above):\n"
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
        "Fact rule: every specific number, percentage, statistic, company name, or "
        "case study you attach to a [n] must actually appear in THAT number's excerpt "
        "above, not just somewhere in the general rag_context impression -- check the "
        "excerpt for the exact [n] you're about to cite before writing the claim. Do "
        "not invent facts, and do not round or blend a real number from an excerpt "
        "into a new one that doesn't actually appear there either (e.g. don't turn a "
        "source's 79% into 85%). If no excerpt supports a concrete figure or example "
        "for a point you want to make, use honest hedged phrasing instead, such as "
        "'数据显示...呈上升趋势' or '部分来源提及...' -- an accurate hedge beats a "
        "precise fabrication. This applies to '## Case Analysis' too: only analyze "
        "cases that actually appear in an excerpt; if none are present, discuss "
        "patterns or trends from the context instead of inventing a fictional case."
    )
    writer_llm = config.get_llm(temperature=0.7)
    response = writer_llm.invoke([("user", prompt)])
    print(f"  -> Draft: {len(response.content)} chars")
    return {"draft_report": response.content}
