"""
Classifies a research topic into one of five academic domains, then maps
that onto one of the citation styles citation_styles.py knows how to
render. A bounded 5-way classification, not open generation -- worst case
is picking a slightly-off style, not fabricating content, so this is a
safe use of an LLM call, unlike the report-body fabrication risk this
project spent a long debugging trail eliminating (see README's Evaluation
section).

Two of the five domains (social_business_english -> APA7, literature_language
-> MLA9) don't have a real renderer yet -- both use author-date in-text
citations, not the numbered [n] markers this pipeline uses everywhere else,
which is a much bigger change (see citation_styles.py's module docstring).
They fall back to DEFAULT_CITATION_STYLE for now rather than mislabeling
output as a style that isn't actually implemented.
"""
from typing import Literal

from pydantic import BaseModel

from research_system import config
from research_system.citation_styles import DEFAULT_CITATION_STYLE
from research_system.prompts import classify_citation_style_prompt
from research_system.state import ResearchState

Domain = Literal[
    "cn_thesis", "cs_ee_english", "social_business_english", "literature_language", "history"
]


class CitationStyleChoice(BaseModel):
    domain: Domain


DOMAIN_TO_STYLE = {
    "cn_thesis": "gbt7714",
    "cs_ee_english": "ieee",
    "history": "chicago_notes",
    "social_business_english": DEFAULT_CITATION_STYLE,  # APA7 not implemented yet
    "literature_language": DEFAULT_CITATION_STYLE,  # MLA9 not implemented yet
}


def citation_style_node(state: ResearchState) -> dict:
    prompt = classify_citation_style_prompt(state["topic"])
    # method="function_calling": same DeepSeek 400-on-json_schema workaround
    # used everywhere else in this codebase (see planner_node).
    llm = config.get_llm(temperature=0).with_structured_output(
        CitationStyleChoice, method="function_calling"
    )
    result = llm.invoke([("user", prompt)])
    # DeepSeek's tool_choice isn't always honored (see critic.py's
    # _invoke_critic_with_retry for the same behavior) -- this decision is
    # low-stakes enough that a retry loop isn't worth it, just default.
    if result is None:
        print(f"  -> Citation-style classifier returned no result, defaulting to {DEFAULT_CITATION_STYLE!r}")
        return {"citation_style": DEFAULT_CITATION_STYLE}
    style = DOMAIN_TO_STYLE[result.domain]
    print(f"[CitationStyle] domain={result.domain} -> style={style}")
    return {"citation_style": style}
