"""
Regression tests for eval/metrics.py against REAL LLM-generated reports,
not hand-written fake strings.

Why this file exists separately from test_eval_metrics.py: hand-written test
strings tend to accidentally avoid edge cases real LLM output actually
contains (e.g. two-digit citation numbers like [12], since it's natural to
write [1][2][3] by hand). A regex change that silently breaks on [12] would
pass every fake-string test here and still be wrong. The fixtures below are
real final_report text pulled from real eval/results/*.json runs (cost real
money to generate once; frozen here so re-checking them costs nothing).

These fixtures are picked for STRUCTURAL diversity, not quality -- one is
the lowest-faithfulness report ever recorded in this project's eval history
(faithfulness=1/5, largely fabricated content) and one is a passing report
(faithfulness=4/5). That's deliberate: citation_validity and
reference_list_consistency only check the report's own citation bookkeeping
(does every [n] resolve, does the body agree with References) -- they say
nothing about whether a claim is factually true, so a low-quality report is
just as valid a fixture for these checks as a high-quality one. (Faithfulness
itself can only be judged by a paid LLM call -- see eval/judge.py -- and
deliberately isn't part of this free regression gate.)

numbered_sources here is reconstructed from each fixture's OWN "## References"
section length, not the original pipeline's pre-renumbering source list --
these fixtures are already-renumbered final reports, so that's the correct
correspondence, but it does make citation_coverage trivially 1.0 by
construction. Only citation_validity/reference_list_consistency (and the
raw citation counts) are meaningful regression targets here.
"""
from pathlib import Path

from eval.metrics import (
    citation_validity,
    extract_reference_entries,
    reference_list_consistency,
    report_length,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(filename: str) -> str:
    return (FIXTURES_DIR / filename).read_text(encoding="utf-8")


def _numbered_sources_for(report: str) -> list[str]:
    """Placeholder source list sized to match the report's own References
    section -- these functions only ever check the COUNT (range 1..N), never
    the content, so a real URL isn't needed here."""
    entries = extract_reference_entries(report)
    return [""] * (max(entries) if entries else 0)


def test_low_faithfulness_report_is_still_structurally_valid():
    """faithfulness=1/5 in eval/results/ -- largely fabricated content -- but
    the citation bookkeeping itself must still be internally consistent."""
    report = _load("real_report_low_faithfulness_en.md")
    numbered_sources = _numbered_sources_for(report)

    validity = citation_validity(report, numbered_sources)
    assert validity == {
        "total_citation_marks": 25,
        "unique_cited_numbers": 3,
        "invalid_citation_numbers": [],
        "all_valid": True,
    }

    consistency = reference_list_consistency(report, numbered_sources)
    assert consistency == {
        "missing_from_references": [],
        "orphan_references": [],
        "is_consistent": True,
    }


def test_passed_report_is_structurally_valid():
    """faithfulness=4/5, judge-approved report -- sanity check the same
    machinery on a real, larger (22-source) passing example."""
    report = _load("real_report_passed_zh.md")
    numbered_sources = _numbered_sources_for(report)

    validity = citation_validity(report, numbered_sources)
    assert validity == {
        "total_citation_marks": 66,
        "unique_cited_numbers": 22,
        "invalid_citation_numbers": [],
        "all_valid": True,
    }

    consistency = reference_list_consistency(report, numbered_sources)
    assert consistency["is_consistent"] is True


def test_report_length_is_nonzero_on_real_reports():
    for filename in ("real_report_low_faithfulness_en.md", "real_report_passed_zh.md"):
        length = report_length(_load(filename))
        assert length["char_count"] > 0
        assert length["token_count"] > 0
