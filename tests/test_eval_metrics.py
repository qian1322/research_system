from eval.metrics import (
    MAX_REVISIONS,
    citation_coverage,
    citation_validity,
    compute_rule_based_metrics,
    reference_list_consistency,
    report_length,
    revision_stats,
)

SOURCES = ["https://a.example.com", "https://b.example.com", "https://c.example.com"]

VALID_REPORT = (
    "# Report\n\n研究发现 A [1],同时 B 也证实了这一点 [2]。\n\n"
    "## References\n[1] https://a.example.com\n[2] https://b.example.com\n"
)


def test_citation_validity_all_valid():
    result = citation_validity(VALID_REPORT, SOURCES)
    assert result["all_valid"] is True
    assert result["invalid_citation_numbers"] == []
    assert result["total_citation_marks"] == 2


def test_citation_validity_detects_out_of_range():
    report = "Finding X [1] and a fabricated one [9]。\n\n## References\n[1] https://a.example.com\n"
    result = citation_validity(report, SOURCES)
    assert result["all_valid"] is False
    assert result["invalid_citation_numbers"] == [9]


def test_reference_list_consistency_detects_missing_and_orphan():
    report = "Body cites [1] and [2].\n\n## References\n[1] https://a.example.com\n[3] https://c.example.com\n"
    result = reference_list_consistency(report, SOURCES)
    assert result["missing_from_references"] == [2]
    assert result["orphan_references"] == [3]
    assert result["is_consistent"] is False


def test_reference_list_consistency_matches_when_aligned():
    assert reference_list_consistency(VALID_REPORT, SOURCES)["is_consistent"] is True


def test_reference_list_consistency_handles_missing_references_section():
    report = "Just a body with a claim [1], no references section at all."
    result = reference_list_consistency(report, SOURCES)
    assert result["missing_from_references"] == [1]
    assert result["orphan_references"] == []


def test_reference_list_consistency_ignores_out_of_range_citation():
    # [9] is out of range (only 3 sources) -- citation_validity's job to flag,
    # not this function's, so it should be excluded from both sides here.
    report = "Body cites [1] and a fabricated [9].\n\n## References\n[1] https://a.example.com\n"
    result = reference_list_consistency(report, SOURCES)
    assert result["missing_from_references"] == []
    assert result["orphan_references"] == []
    assert result["is_consistent"] is True


def test_citation_coverage_ratio():
    result = citation_coverage(VALID_REPORT, SOURCES)
    assert result["cited_source_count"] == 2
    assert result["total_source_count"] == 3
    assert result["coverage_ratio"] == 2 / 3


def test_citation_coverage_handles_zero_sources():
    result = citation_coverage("no sources here", [])
    assert result["coverage_ratio"] == 0.0
    assert result["total_source_count"] == 0


def test_report_length_counts_chars_and_tokens():
    result = report_length("hello world")
    assert result["char_count"] == 11
    assert result["token_count"] > 0


def test_revision_stats_hit_max():
    assert revision_stats({"revision_count": MAX_REVISIONS})["hit_max_revisions"] is True
    assert revision_stats({"revision_count": 1})["hit_max_revisions"] is False


def test_revision_stats_default_missing_key():
    assert revision_stats({}) == {"revision_count": 0, "hit_max_revisions": False}


def test_compute_rule_based_metrics_aggregates_all_keys():
    result = {"final_report": VALID_REPORT, "numbered_sources": SOURCES, "revision_count": 0}
    metrics = compute_rule_based_metrics(result)
    assert set(metrics.keys()) == {
        "citation_validity",
        "reference_list_consistency",
        "citation_coverage",
        "length",
        "revision",
    }
