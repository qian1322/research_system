from research_system.report_sections import (
    ReportSection,
    build_section_index,
    match_headings_by_keyword,
    parse_section_edits,
    reassemble_report,
    retrieve_target_sections,
    split_report_into_sections,
)

REPORT = (
    "# Topic Research Report\n"
    "Some preamble.\n\n"
    "## Executive Summary\n"
    "Summary body.\n\n"
    "## Key Findings\n"
    "Findings body [1].\n\n"
    "## References\n"
    "[1] https://example.com/a\n"
)


def test_split_report_into_sections():
    sections = split_report_into_sections(REPORT)
    assert sections == [
        ReportSection("## Executive Summary", "Summary body."),
        ReportSection("## Key Findings", "Findings body [1]."),
    ]


def test_split_report_with_no_headings_returns_empty():
    assert split_report_into_sections("just plain text, no headings") == []


def test_reassemble_report_replaces_only_edited_sections():
    sections = split_report_into_sections(REPORT)
    result = reassemble_report(REPORT, sections, {"## Key Findings": "New findings body [1]."})

    assert "## Executive Summary\nSummary body." in result
    assert "## Key Findings\nNew findings body [1]." in result
    assert "Findings body [1]." not in result  # old body is gone, not duplicated
    assert result.startswith("# Topic Research Report\nSome preamble.\n\n")
    assert result.count("## References") == 1
    assert "[1] https://example.com/a" in result


def test_reassemble_report_with_no_edits_is_unchanged_content():
    sections = split_report_into_sections(REPORT)
    result = reassemble_report(REPORT, sections, {})
    assert "Summary body." in result
    assert "Findings body [1]." in result


def test_parse_section_edits_keeps_only_target_headings():
    raw = "## Key Findings\nRevised findings.\n\n## Unexpected Heading\nShould be dropped.\n"
    edited = parse_section_edits(raw, target_headings=["## Key Findings"])
    assert edited == {"## Key Findings": "Revised findings."}


def test_parse_section_edits_with_no_headings_returns_empty():
    assert parse_section_edits("just prose, no headings", target_headings=["## Key Findings"]) == {}


def test_match_headings_by_keyword_finds_named_section():
    sections = [
        ReportSection("## Executive Summary", "Summary body."),
        ReportSection("## Conclusions", "Conclusions body."),
    ]
    assert match_headings_by_keyword(
        "在结论部分补充一句关于局限性的说明", sections
    ) == ["## Conclusions"]


def test_match_headings_by_keyword_no_match_returns_empty():
    sections = [ReportSection("## Executive Summary", "Summary body.")]
    assert match_headings_by_keyword("make this punchier", sections) == []


def test_match_headings_by_keyword_only_matches_present_sections():
    # "## Conclusions" isn't in `sections`, so its alias shouldn't match.
    sections = [ReportSection("## Executive Summary", "Summary body.")]
    assert match_headings_by_keyword("结论部分", sections) == []


def test_retrieve_target_sections_returns_k_known_headings(fake_llm):
    sections = [
        ReportSection("## Executive Summary", "Summary body."),
        ReportSection("## Background", "Background body."),
        ReportSection("## Key Findings", "Findings body."),
        ReportSection("## Case Analysis", "Case body."),
        ReportSection("## Conclusions", "Conclusions body."),
    ]
    index = build_section_index(sections)
    headings = retrieve_target_sections(index, "add a case study", k=2)

    assert len(headings) == 2
    assert all(h in {s.heading for s in sections} for h in headings)


def test_retrieve_target_sections_clamped_to_available_sections(fake_llm):
    sections = [ReportSection("## Executive Summary", "Summary body.")]
    index = build_section_index(sections)
    headings = retrieve_target_sections(index, "shorten this", k=1)
    assert headings == ["## Executive Summary"]
