import re
from datetime import date

from research_system.citation_styles import (
    CitedSource,
    format_reference,
    format_reference_chicago_notes,
    format_reference_gbt7714,
    format_reference_ieee,
)

SOURCE = CitedSource(
    title="2025年全球AI Agent行业洞察报告",
    url="https://www.moonfox.cn/insight/report/1766",
)
FIXED_DATE = date(2026, 9, 7)


def test_site_name_strips_www_and_keeps_domain():
    # _site_name (author position) should strip "www." -- the URL field at
    # the end of the citation is untouched and keeps the real "www." host.
    result = format_reference_gbt7714(1, SOURCE, FIXED_DATE)
    # "[1] <author_position>. <rest>" -- match up to the field-separating
    # ". " (period+space), not "moonfox.cn"'s own internal period.
    author_position = re.match(r"^\[\d+\]\s+(.*?)\.\s", result).group(1)
    assert author_position == "moonfox.cn"


def test_gbt7714_format():
    result = format_reference_gbt7714(1, SOURCE, FIXED_DATE)
    assert result == (
        "[1] moonfox.cn. 2025年全球AI Agent行业洞察报告[EB/OL]. "
        "[2026-09-07]. https://www.moonfox.cn/insight/report/1766."
    )


def test_ieee_format():
    result = format_reference_ieee(1, SOURCE, FIXED_DATE)
    assert result == (
        '[1] moonfox.cn, "2025年全球AI Agent行业洞察报告," [Online]. '
        "Available: https://www.moonfox.cn/insight/report/1766. [Accessed: Sep. 7, 2026]."
    )


def test_chicago_notes_format():
    result = format_reference_chicago_notes(1, SOURCE, FIXED_DATE)
    assert result == (
        '1. moonfox.cn, "2025年全球AI Agent行业洞察报告," '
        "访问于2026年9月7日, https://www.moonfox.cn/insight/report/1766."
    )


def test_index_is_used_verbatim_not_recomputed():
    # format_reference must not silently renumber -- the caller (renumber_citations)
    # is the single source of truth for numbering.
    result = format_reference_gbt7714(7, SOURCE, FIXED_DATE)
    assert result.startswith("[7]")


def test_format_reference_dispatches_by_style_name():
    for style in ("gbt7714", "ieee", "chicago_notes"):
        assert format_reference(style, 1, SOURCE, FIXED_DATE) == FORMATTERS_FOR_TEST[style](1, SOURCE, FIXED_DATE)


FORMATTERS_FOR_TEST = {
    "gbt7714": format_reference_gbt7714,
    "ieee": format_reference_ieee,
    "chicago_notes": format_reference_chicago_notes,
}


def test_format_reference_falls_back_to_default_on_unknown_style():
    # Shouldn't crash a run that already paid for search + writing --
    # degrade to the default style instead.
    result = format_reference("apa7", 1, SOURCE, FIXED_DATE)
    assert result == format_reference_gbt7714(1, SOURCE, FIXED_DATE)


def test_defaults_to_todays_date_when_access_date_omitted():
    # Don't assert an exact string (today's date isn't fixed) -- just confirm
    # it doesn't crash and produces *some* ISO-looking date.
    result = format_reference_gbt7714(1, SOURCE)
    assert date.today().isoformat() in result
