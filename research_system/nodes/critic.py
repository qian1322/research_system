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

    prompt = (
        f'Topic: {state["topic"]}\n\n'
        f'Retrieved source material (the ONLY material this report is allowed to be '
        f'grounded in; [n] markers are citation numbers):\n{state.get("rag_context", "")}\n\n'
        f'Report:\n{state["draft_report"]}\n\n'
        "Rate the report: score(1-10), approved(>=7), improvements, critique_summary.\n"
        "Fact-check requirement: check every specific number, statistic, named case "
        "study, company, or example in the report against the retrieved source material "
        "above. Anything not traceable to that material is fabricated. If the report "
        "contains fabricated specifics, do NOT approve it (approved=false) no matter how "
        "well-organized it reads, and list each fabricated claim in improvements.\n"
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
