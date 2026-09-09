from research_system import config
from research_system.prompts import (
    build_edit_task,
    build_section_edit_task,
    build_sources_text_from_numbered_sources,
    build_sources_text_from_report,
    writer_prompt,
)
from research_system.report_sections import (
    build_section_index,
    match_headings_by_keyword,
    parse_section_edits,
    reassemble_report,
    retrieve_target_sections,
    split_report_into_sections,
)
from research_system.state import ResearchState


def _apply_section_edit(state: ResearchState, search_results: list) -> str:
    """
    New edit round's first pass: retrieve and revise only the section(s)
    relevant to edit_instructions, splicing the LLM's output into the
    untouched rest of the report so anything not targeted is provably
    byte-identical rather than just "hopefully left alone" by the LLM.
    Falls back to a whole-report edit (build_edit_task, this branch's
    pre-section-retrieval behavior) if the report can't be split into
    sections, or the LLM doesn't follow the "output only these headings"
    format closely enough for parse_section_edits to find anything.
    """
    final_report = state["final_report"]
    edit_instructions = state["edit_instructions"]
    edit_history = state.get("edit_history", [])
    # The report's citations may already be compacted/reordered by
    # renumber_citations(), so the current [n] <-> source mapping must come
    # from the report's own References section, not the original
    # numbered_sources list (see prompts.py's build_sources_text_from_report).
    sources_text = build_sources_text_from_report(final_report, search_results)
    writer_llm = config.get_llm(temperature=0.7)

    sections = split_report_into_sections(final_report)
    if sections:
        target_headings = match_headings_by_keyword(edit_instructions, sections)
        if target_headings:
            print(f"  -> Matched section(s) by keyword: {', '.join(target_headings)}")
        else:
            k = min(state.get("edit_section_k", 2), len(sections))
            index = build_section_index(sections)
            target_headings = retrieve_target_sections(index, edit_instructions, k=k)
        target_sections = [s for s in sections if s.heading in target_headings]

        task = build_section_edit_task(target_sections, edit_instructions, edit_history)
        prompt = writer_prompt(
            topic=state["topic"], rag_context=state["rag_context"],
            sources_text=sources_text, task=task,
        )
        response = writer_llm.invoke([("user", prompt)])
        edited = parse_section_edits(response.content, target_headings)
        if edited:
            print(f"  -> Edited section(s): {', '.join(edited)}")
            return reassemble_report(final_report, sections, edited)
        print("  -> LLM didn't follow section-edit format; falling back to whole-report edit")

    task = build_edit_task(final_report, edit_instructions, edit_history)
    prompt = writer_prompt(
        topic=state["topic"], rag_context=state["rag_context"],
        sources_text=sources_text, task=task,
    )
    response = writer_llm.invoke([("user", prompt)])
    return response.content


def writer_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    edit_instructions = state.get("edit_instructions", "")
    search_results = state.get("search_results", [])

    if edit_instructions and revision == 0:
        print("[Writer] Applying edit instruction...")
        draft_report = _apply_section_edit(state, search_results)
        print(f"  -> Draft: {len(draft_report)} chars")
        return {"draft_report": draft_report}

    if edit_instructions:
        print(f"[Writer] Revising edit (attempt {revision})...")
        task = f'Revise based on:\n{state.get("critique", "")}'
        sources_text = build_sources_text_from_report(state["final_report"], search_results)
    elif revision == 0:
        print("[Writer] Writing initial draft...")
        task = "Write a complete, professional research report."
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
        sources_text = build_sources_text_from_numbered_sources(
            state.get("numbered_sources", []), search_results
        )
    else:
        print(f"[Writer] Revising (attempt {revision})...")
        task = f'Revise based on:\n{state.get("critique", "")}'
        sources_text = build_sources_text_from_numbered_sources(
            state.get("numbered_sources", []), search_results
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
