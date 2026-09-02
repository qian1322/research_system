import re
from typing import List

from pydantic import BaseModel

from research_system import config
from research_system.state import ResearchState

REFERENCES_MARKER = "## References"


class CriticOutput(BaseModel):
    score: int  # 1-10
    approved: bool
    improvements: List[str]
    critique_summary: str


def renumber_citations(report: str, numbered_sources: List[str]) -> str:
    """
    Writer-Critic revisions can drop cited facts, leaving gaps in the [n]
    sequence (e.g. [1][2][5][6]). Compress the surviving numbers to a
    contiguous 1..k run, in order of first appearance, and rebuild the
    References section to match -- listing only sources actually cited.
    """
    idx = report.find(REFERENCES_MARKER)
    body = report[:idx] if idx != -1 else report

    used_old_numbers = []
    seen = set()
    for m in re.finditer(r"\[(\d+)\]", body):
        n = int(m.group(1))
        if n not in seen and 1 <= n <= len(numbered_sources):
            seen.add(n)
            used_old_numbers.append(n)

    old_to_new = {old: new for new, old in enumerate(used_old_numbers, start=1)}

    def _replace(m):
        n = int(m.group(1))
        return f"[{old_to_new[n]}]" if n in old_to_new else m.group(0)

    new_body = re.sub(r"\[(\d+)\]", _replace, body).rstrip()

    if not used_old_numbers:
        return new_body

    refs = "\n".join(
        f"[{new}] {numbered_sources[old - 1]}"
        for old, new in sorted(old_to_new.items(), key=lambda kv: kv[1])
    )
    return f"{new_body}\n\n{REFERENCES_MARKER}\n{refs}\n"


# numbered_sources only retains bare URLs (see rag._renumber_to_global); the raw
# source text lives in state["search_results"][*]["sources"][*]["content"]. Rebuild
# citation number -> source excerpt so the Critic (and eval/judge.py, which imports
# this) can check a specific claim against the actual text a citation points to,
# instead of a single blended summary blob.
def build_source_excerpt_map(
    search_results: List[dict], numbered_sources: List[str], max_chars_per_source: int = 400
) -> dict:
    url_to_content: dict = {}
    for r in search_results:
        for s in r.get("sources", []):
            url = s.get("url")
            if url and url not in url_to_content:
                url_to_content[url] = s.get("content", "")

    return {
        i: url_to_content.get(url, "")[:max_chars_per_source]
        for i, url in enumerate(numbered_sources, start=1)
    }


# DeepSeek's tool_choice="any" isn't always honored -- the model sometimes
# replies without calling the tool at all, in which case with_structured_output
# returns None (documented langchain_core behavior, not an error) instead of a
# CriticOutput. Retry a couple times before giving up, since this is usually
# transient.
def _invoke_critic_with_retry(critic_llm, prompt: str, max_retries: int = 2) -> CriticOutput | None:
    for attempt in range(max_retries + 1):
        result = critic_llm.invoke([("user", prompt)])
        if result is not None:
            return result
        print(f"  -> Critic returned no structured output (attempt {attempt + 1}/{max_retries + 1}), retrying...")
    return None


def critic_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    print(f"[Critic] Reviewing (revision_count={revision})")
    numbered_sources = state.get("numbered_sources", [])

    # Hard stop: prevent infinite loop
    if revision >= 3:
        print("  -> Max revisions. Force approving.")
        return {
            "quality_approved": True,
            "final_report": renumber_citations(state["draft_report"], numbered_sources),
            "critique": "Max revisions reached.",
        }

    excerpts = build_source_excerpt_map(state.get("search_results", []), numbered_sources)
    sources_block = "\n".join(f"[{n}] {numbered_sources[n - 1]}\n{text}" for n, text in excerpts.items())

    prompt = (
        f'Topic: {state["topic"]}\n\n'
        f'Source excerpts (numbered, matching [n] citations in the report below; the '
        f'ONLY material this report is allowed to be grounded in):\n{sources_block}\n\n'
        f'Report:\n{state["draft_report"]}\n\n'
        "Rate the report: score(1-10), approved(>=7), improvements, critique_summary.\n"
        "Fact-check requirement: for EACH [n] citation used in the report, check it "
        "against source excerpt [n] above -- the claim is only valid if that excerpt "
        "actually supports it. Flag any specific number, percentage, statistic, "
        "company name, or case study that is not an exact or clearly reasonable match "
        "to its cited excerpt -- including numbers that look like a rounded or blended "
        "version of a real number in the source (e.g. citing 85% when the excerpt says "
        "79%). If the report contains such unsupported specifics, do NOT approve it "
        "(approved=false) no matter how well-organized it reads, and list each one in "
        "improvements.\n"
        "Respond in Chinese."
    )
    # method="function_calling": ChatOpenAI defaults to method="json_schema", whose
    # response_format DeepSeek's API rejects with a 400 (see planner_node for the
    # same workaround).
    critic_llm = config.get_llm(temperature=0.2).with_structured_output(
        CriticOutput, method="function_calling"
    )
    result = _invoke_critic_with_retry(critic_llm, prompt)

    if result is None:
        print("  -> Critic produced no verdict after retries. Treating as not approved.")
        return {
            "quality_approved": False,
            "critique": "Critic did not return a valid verdict after retries; please revise for clarity and structure.",
            "revision_count": revision + 1,
        }

    print(f"  -> Score: {result.score}/10 | Approved: {result.approved}")

    if result.approved:
        return {
            "quality_approved": True,
            "final_report": renumber_citations(state["draft_report"], numbered_sources),
            "critique": result.critique_summary,
            "revision_count": revision,
        }
    else:
        fixes = "\n".join(f"- {i}" for i in result.improvements)
        return {
            "quality_approved": False,
            "critique": f"Please improve:\n{fixes}",
            "revision_count": revision + 1,
        }


def should_revise(state: ResearchState) -> str:
    if state.get("quality_approved", False):
        return "approved"
    return "revise"
