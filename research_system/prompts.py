"""
Every prompt sent to an LLM in this pipeline, in one place.

Each node file (planner.py, search.py, rag.py, writer.py, critic.py) still
owns its own business logic -- state handling, LLM invocation, retries -- it
just imports the prompt-building function it needs from here instead of
inlining an f-string. Kept out of this file on purpose: the Pydantic
structured-output schemas (ResearchPlan in planner.py, QualityVerdict in
critic.py). Their Field(...) descriptions do end up as text the model sees
(compiled into the tool/function schema for with_structured_output), but
they're a data CONTRACT, not an instruction prompt -- they stay colocated
with the node that defines and consumes that contract.

build_quality_prompt (used by both critic_node and eval/judge.py's
judge_report -- the two are compared precisely because they share this exact
prompt, see critic.py) needs a "citation number -> source excerpt" lookup.
Those helpers (_extract_reference_urls, build_source_excerpt_map,
build_source_excerpt_map_from_numbered_sources) live here too rather than in
critic.py: they have no purpose except feeding a prompt string, so moving
them alongside avoids a critic.py <-> prompts.py import cycle (critic.py
needs build_quality_prompt; if these helpers stayed in critic.py, this
module would need to import back from critic.py to build it).
"""
import re
from typing import List

REFERENCES_MARKER = "## References"


# ---------------------------------------------------------------------------
# planner.py
# ---------------------------------------------------------------------------
def planner_prompt(topic: str) -> str:
    return (
        "You are a research planning expert.\n"
        f"Topic: {topic}\n"
        "Break this into 3-5 specific, independently-answerable sub-questions.\n"
        "Output thinking + sub_questions, written in the same language as the topic above."
    )


# ---------------------------------------------------------------------------
# search.py
# ---------------------------------------------------------------------------
def search_agent_prompt(topic: str, sub_question: str, sources_block: str, n_sources: int) -> str:
    return (
        f"Research topic: {topic}\n"
        f"Sub-question: {sub_question}\n\n"
        f"Web search results (numbered):\n{sources_block}\n\n"
        "Using ONLY the search results above, extract the key findings, data, "
        "and examples that answer the sub-question.\n"
        "Citation rule: after each fact, cite the result it came from using its "
        f"bracketed number, e.g. '...market size reached $X [2]'. Only use numbers "
        f"1-{n_sources} that appear above -- never invent one, and never write "
        "out the URL itself. A sentence may cite multiple sources, e.g. [1][3].\n"
        "Respond in the same language as the topic above, under 300 characters."
    )


# ---------------------------------------------------------------------------
# rag.py
# ---------------------------------------------------------------------------
def rag_synthesis_prompt(topic: str, retrieved_text: str) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Most relevant retrieved chunks below (the [n] markers inside them are "
        f"citation numbers pointing to source URLs):\n{retrieved_text}\n\n"
        "Synthesize the key findings, data, and examples most useful for writing "
        "the report.\n"
        "Citation rule: keep every [n] marker exactly as written -- don't remove, "
        "renumber, or invent one. If a piece of information has no marker, don't add one.\n"
        # Previously capped at 500 chars for a report covering 3-5 sub-questions --
        # too little real material for the Writer to draw on, which pushed it
        # toward fabricating specifics to fill the gaps (see README's Evaluation
        # section for a real example this surfaced).
        "Respond in the same language as the topic above, under 2000 characters."
    )


# ---------------------------------------------------------------------------
# writer.py
# ---------------------------------------------------------------------------
def writer_prompt(topic: str, rag_context: str, sources_text: str, task: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Context (from RAG retrieval; [n] markers are citation numbers):\n{rag_context}\n\n"
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


# ---------------------------------------------------------------------------
# critic.py / eval/judge.py -- shared quality-rubric prompt + its excerpt helpers
# ---------------------------------------------------------------------------

# report's own "## References" section is the ground truth for what [n]
# currently means IN THIS REPORT -- assuming report [n] == numbered_sources[n-1]
# (indexing into the pipeline's original, uncompressed source list) breaks the
# moment renumber_citations() compresses/reorders citation numbers, which
# happens whenever the report doesn't cite every retrieved source (i.e. almost
# always -- citation_coverage has never hit 100% in eval/results/). Parsing the
# report's own References section instead is correct whether or not
# renumbering has happened yet.
def _extract_reference_urls(report: str) -> dict:
    idx = report.find(REFERENCES_MARKER)
    if idx == -1:
        return {}
    refs = {}
    for line in report[idx + len(REFERENCES_MARKER):].splitlines():
        m = re.match(r"^\[(\d+)\]\s*(\S+)", line.strip())
        if m:
            refs[int(m.group(1))] = m.group(2)
    return refs


# The raw source text lives in search_results[*]["sources"][*]["content"];
# numbered_sources only ever had bare URLs. Build url -> content once, shared
# by both excerpt-map builders below.
def _url_to_content(search_results: List[dict]) -> dict:
    url_to_content: dict = {}
    for r in search_results:
        for s in r.get("sources", []):
            url = s.get("url")
            if url and url not in url_to_content:
                url_to_content[url] = s.get("content", "")
    return url_to_content


# Rebuild citation number -> source excerpt so a QualityVerdict caller can
# check a specific claim against the actual text a citation points to,
# instead of a single blended summary blob. Ground truth for what [n] means
# is the report's own References section (see _extract_reference_urls) --
# correct whether or not renumber_citations() has run yet.
def build_source_excerpt_map(
    report: str, search_results: List[dict], max_chars_per_source: int = 400
) -> dict:
    url_to_content = _url_to_content(search_results)
    return {
        n: url_to_content.get(url, "")[:max_chars_per_source]
        for n, url in _extract_reference_urls(report).items()
    }


# Same idea, but for the Writer -- there's no report yet to parse a
# References section out of, only the flattened numbered_sources list
# (index i, 0-based) <-> citation [i+1], which is the correct correspondence
# up until renumber_citations() compresses it later. Lets the Writer ground
# claims in real source text while drafting, instead of only being told
# after the fact (by the Critic) which claims didn't check out.
def build_source_excerpt_map_from_numbered_sources(
    numbered_sources: List[str], search_results: List[dict], max_chars_per_source: int = 400
) -> dict:
    url_to_content = _url_to_content(search_results)
    return {
        i: url_to_content.get(url, "")[:max_chars_per_source]
        for i, url in enumerate(numbered_sources, start=1)
        if url
    }


# Shared prompt builder -- the actual mechanism that keeps critic_node and
# eval/judge.py's judge_report evaluating the same report the same way.
# Faithfulness explicitly calls out fabricated vs. honestly-hedged numbers so
# a report isn't punished for saying "约" / "数据显示..." instead of quoting a
# figure the source doesn't have (see the writer-factrule / graded-severity
# history in README.md's Evaluation section for why that distinction matters).
def build_quality_prompt(
    topic: str,
    research_plan: List[str],
    report: str,
    search_results: List[dict],
) -> str:
    ref_urls = _extract_reference_urls(report)
    excerpts = build_source_excerpt_map(report, search_results)
    sources_block = "\n".join(f"[{n}] {ref_urls[n]}\n{text}" for n, text in excerpts.items())
    plan_block = "\n".join(f"- {q}" for q in research_plan) if research_plan else "(not available)"

    return (
        f"Topic: {topic}\n\n"
        f"Planned sub-questions:\n{plan_block}\n\n"
        f"Source excerpts (numbered, matching [n] citations in the report below; the "
        f"ONLY material this report is allowed to be grounded in):\n{sources_block}\n\n"
        f"Report:\n{report}\n\n"
        "Score the report on 4 dimensions, 1 (poor) to 5 (excellent):\n"
        "- coverage_score: does it address every planned sub-question above?\n"
        "- faithfulness_score: is every specific number, percentage, statistic, "
        "company name, or case study in the report traceable to its cited excerpt? "
        "Score low (1-2) for anything invented, contradicted, or a rounded/blended "
        "version of a real number that doesn't actually appear in the excerpt (e.g. "
        "citing 85% when the excerpt says 79%). Honest hedged language ('约', '数据"
        "显示...呈上升趋势', '部分来源提及...') that doesn't claim false precision "
        "should NOT be penalized.\n"
        "- coherence_score: is it well-organized, readable, non-repetitive?\n"
        "- citation_appropriateness_score: are [n] citations placed on the claims "
        "they actually support, rather than missing or misattributed?\n"
        "Give a short reasoning explaining the scores, itemized by dimension."
    )
