from research_system import config
from research_system.prompts import build_source_excerpt_map_from_numbered_sources, writer_prompt
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

    prompt = writer_prompt(
        topic=state["topic"],
        rag_context=state["rag_context"],
        sources_text=sources_text,
        task=task,
    )
    writer_llm = config.get_llm(temperature=0.7)
    response = writer_llm.invoke([("user", prompt)])
    print(f"  -> Draft: {len(response.content)} chars")
    return {"draft_report": response.content}
