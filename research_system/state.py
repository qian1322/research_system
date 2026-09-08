import operator
from typing import Annotated, List, TypedDict


class ResearchState(TypedDict):
    topic: str
    research_plan: List[str]
    # one of citation_styles.CITATION_STYLES, set by nodes/citation_style.py
    # right after planning; consumed by critic.py's renumber_citations when
    # it builds the final References section.
    citation_style: str
    # operator.add: parallel search agents append their result, never overwrite the list
    search_results: Annotated[List[dict], operator.add]
    # flattened, pipeline-wide source list; index i (0-based) <-> citation [i+1] in the text
    numbered_sources: List[str]
    # parallel to numbered_sources (same length, same order, same index <-> citation
    # mapping) -- Tavily's title for each source, kept separate rather than folded into
    # numbered_sources so every existing consumer of that plain URL list is untouched.
    # Used only for citation-style formatting (see citation_styles.py).
    source_titles: List[str]
    rag_context: str
    draft_report: str
    critique: str
    revision_count: int  # hard stop for the Writer-Critic loop
    quality_approved: bool
    final_report: str
