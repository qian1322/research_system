"""
Splits a report into '## '-headed sections so a post-approval edit round can
retrieve and revise only the section(s) relevant to the user's instruction,
instead of resending the whole report to the LLM every round. See
nodes/writer.py's edit-first-pass branch for the caller.

Front matter (the '# Title' line and anything before the first '## ') and
the '## References' block are never returned as sections -- neither is a
retrieval/edit target: front matter has nothing to edit, and References is
regenerated from scratch by critic.py's renumber_citations() regardless of
what's here.
"""
import re
import uuid
from typing import Dict, List, NamedTuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from research_system import config
from research_system.prompts import REFERENCES_MARKER

_HEADING_RE = re.compile(r"^## .*$", re.MULTILINE)

# Chinese/English aliases for the fixed section headings writer_prompt's
# format spec always produces (see prompts.py's writer_prompt). Checked
# before falling back to embedding retrieval: a small Chinese embedding
# model (config.get_embeddings()'s HuggingFace fallback) doesn't reliably
# match a Chinese instruction naming a section (e.g. "结论部分") to its
# English heading ("## Conclusions") -- confirmed by a real manual run where
# "在结论部分补充..." with k=1 matched "## Executive Summary" instead. An
# exact/substring alias check is cheap, deterministic, and sidesteps that
# failure mode whenever the instruction names a section directly; only
# instructions that don't name a section at all still need embedding search.
_HEADING_ALIASES: Dict[str, List[str]] = {
    "## Executive Summary": ["摘要", "概述", "执行摘要", "executive summary"],
    "## Background": ["背景", "背景介绍", "background"],
    "## Key Findings": ["关键发现", "主要发现", "发现", "key findings"],
    "## Case Analysis": ["案例分析", "案例", "case analysis"],
    "## Conclusions": ["结论", "总结", "conclusion"],
}


def match_headings_by_keyword(instruction: str, sections: List[ReportSection]) -> List[str]:
    lowered = instruction.lower()
    matched = []
    for s in sections:
        aliases = _HEADING_ALIASES.get(s.heading, [])
        if any(alias.lower() in lowered for alias in aliases):
            matched.append(s.heading)
    return matched


class ReportSection(NamedTuple):
    heading: str  # exact "## ..." line, e.g. "## Key Findings"
    body: str


def split_report_into_sections(report: str) -> List[ReportSection]:
    matches = list(_HEADING_RE.finditer(report))
    sections = []
    for i, m in enumerate(matches):
        heading = m.group().strip()
        if heading == REFERENCES_MARKER:
            break
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        body = report[m.end():end].strip()
        sections.append(ReportSection(heading=heading, body=body))
    return sections


def _front_matter(report: str) -> str:
    m = _HEADING_RE.search(report)
    return report[: m.start()] if m else report


def _references_block(report: str) -> str:
    idx = report.find(REFERENCES_MARKER)
    return report[idx:] if idx != -1 else ""


def build_section_index(sections: List[ReportSection]) -> Chroma:
    docs = [
        Document(page_content=f"{s.heading}\n{s.body}", metadata={"heading": s.heading})
        for s in sections
    ]
    # Chroma.from_documents() defaults collection_name to the fixed string
    # "langchain" -- without a unique name, successive calls in the same
    # process share (and accumulate into) one collection instead of each
    # getting an isolated index, so an edit round could retrieve leftover
    # sections from an earlier report. A fresh uuid per call guarantees
    # isolation.
    return Chroma.from_documents(
        documents=docs,
        embedding=config.get_embeddings(),
        collection_name=f"report-sections-{uuid.uuid4().hex}",
    )


def retrieve_target_sections(index: Chroma, instruction: str, k: int = 2) -> List[str]:
    retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": k})
    headings = []
    for doc in retriever.invoke(instruction):
        heading = doc.metadata["heading"]
        if heading not in headings:
            headings.append(heading)
    return headings


def reassemble_report(
    original_report: str, sections: List[ReportSection], edited: Dict[str, str]
) -> str:
    body = "\n\n".join(
        f"{s.heading}\n{edited.get(s.heading, s.body)}" for s in sections
    )
    # References block is reproduced verbatim, pre-renumbering -- fine
    # because renumber_citations() rebuilds it from scratch from whichever
    # [n] markers survive in the body above, discarding whatever is here.
    return f"{_front_matter(original_report)}{body}\n\n{_references_block(original_report)}"


def parse_section_edits(raw_output: str, target_headings: List[str]) -> Dict[str, str]:
    matches = list(_HEADING_RE.finditer(raw_output))
    edited = {}
    for i, m in enumerate(matches):
        heading = m.group().strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_output)
        body = raw_output[m.end():end].strip()
        if heading in target_headings:
            edited[heading] = body
        else:
            print(f"  -> [Writer] Ignoring unexpected section heading in edit output: {heading!r}")
    return edited
