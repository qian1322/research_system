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
    # current post-approval edit round's raw instruction; "" means no edit in
    # progress. Set by nodes/edit_review.py, consumed by writer.py.
    edit_instructions: str
    # bounded (kept to the last few entries by edit_review_node itself) --
    # deliberately NOT Annotated[..., operator.add]: a reducer always
    # concatenates whatever a node returns, so capping length only works if
    # the node's return value fully replaces this field instead.
    edit_history: List[str]
    # how many report sections a section-targeted edit round should retrieve
    # and revise (see report_sections.py); user-configurable per round via
    # edit_review_node, defaults to 2 if not provided.
    edit_section_k: int
