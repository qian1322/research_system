import re
from typing import List

from pydantic import BaseModel, Field

from research_system import config
from research_system.prompts import REFERENCES_MARKER, build_quality_prompt
from research_system.state import ResearchState

MAX_REVISIONS = 3  # hard stop for the Writer-Critic loop
PASS_THRESHOLD = 4  # out of 5, on the averaged 4-dimension score


# Same schema critic_node (real-time, DeepSeek) and eval/judge.py's judge_report
# (offline, optionally a different provider) both score against -- previously
# the pipeline's own Critic used a vague holistic 1-10 "approved(>=7)" score
# while eval/judge.py used this explicit 4-dimension rubric, so a report the
# Critic approved could still fail the independent judge on the very same
# report (observed: Critic 7/10 approved, judge averaged 3.5/5, still below
# PASS_THRESHOLD). Sharing the schema AND the prompt (build_quality_prompt
# below) doesn't guarantee the two agree -- they can still be different model
# calls with different temperature/model -- but it removes "they were
# scoring different things, worded differently" as a source of the gap.
class QualityVerdict(BaseModel):
    coverage_score: int = Field(..., ge=1, le=5, description="Does the report address every planned sub-question?")
    faithfulness_score: int = Field(..., ge=1, le=5, description="Is every specific claim traceable to its cited source excerpt?")
    coherence_score: int = Field(..., ge=1, le=5, description="Is the report well-organized, readable, non-repetitive?")
    citation_appropriateness_score: int = Field(..., ge=1, le=5, description="Are [n] citations placed on the claims they actually support?")
    reasoning: str = Field(..., description="Short justification for the scores, itemized by dimension.")


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
# QualityVerdict. Retry a couple times before giving up, since this is usually
# transient.
def _invoke_critic_with_retry(critic_llm, prompt: str, max_retries: int = 2) -> QualityVerdict | None:
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
    if revision >= MAX_REVISIONS:
        print("  -> Max revisions. Force approving.")
        return {
            "quality_approved": True,
            "final_report": renumber_citations(state["draft_report"], numbered_sources),
            "critique": "Max revisions reached.",
        }

    prompt = build_quality_prompt(
        topic=state["topic"],
        research_plan=state.get("research_plan", []),
        report=state["draft_report"],
        search_results=state.get("search_results", []),
    )
    # method="function_calling": ChatOpenAI defaults to method="json_schema", whose
    # response_format DeepSeek's API rejects with a 400 (see planner_node for the
    # same workaround).
    critic_llm = config.get_llm(temperature=0.2).with_structured_output(
        QualityVerdict, method="function_calling"
    )
    result = _invoke_critic_with_retry(critic_llm, prompt)

    if result is None:
        print("  -> Critic produced no verdict after retries. Treating as not approved.")
        return {
            "quality_approved": False,
            "critique": "Critic did not return a valid verdict after retries; please revise for clarity and structure.",
            "revision_count": revision + 1,
        }

    scores = [
        result.coverage_score,
        result.faithfulness_score,
        result.coherence_score,
        result.citation_appropriateness_score,
    ]
    average_score = sum(scores) / len(scores)
    approved = average_score >= PASS_THRESHOLD
    print(
        f"  -> coverage={result.coverage_score} faithfulness={result.faithfulness_score} "
        f"coherence={result.coherence_score} citations={result.citation_appropriateness_score} "
        f"avg={average_score:.2f} | Approved: {approved}"
    )

    if approved:
        return {
            "quality_approved": True,
            "final_report": renumber_citations(state["draft_report"], numbered_sources),
            "critique": result.reasoning,
            "revision_count": revision,
        }
    else:
        return {
            "quality_approved": False,
            "critique": f"Please improve:\n{result.reasoning}",
            "revision_count": revision + 1,
        }


def should_revise(state: ResearchState) -> str:
    if state.get("quality_approved", False):
        return "approved"
    return "revise"
