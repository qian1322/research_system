"""
Deterministic (no-LLM) formatters that render one cited web source into a
References-list entry, in one of three citation styles. Kept separate from
prompts.py: this is post-processing string formatting applied to the final
report, not text sent to an LLM.

Why author/date are always treated as missing: Tavily's search results
never include a byline or publish date -- confirmed against a real API
response (fields are url/title/content/score/raw_content, nothing else) --
so every style below follows its own convention for a source with no
individual author and no known publish date: fall back to the site's
domain name in author position, and cite the day the pipeline actually
retrieved the source (which we know for certain) instead of guessing a
publish date it doesn't have.

These are deliberately simplified renderings -- close to GB/T 7714-2015 /
IEEE / Chicago notes-bibliography style for a web source, not full
implementations of every edge case those manuals define for other source
types (journal articles, books, etc.). This pipeline only ever cites web
pages, so only that one case needs to be right.
"""
import calendar
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

CITATION_STYLES = ("gbt7714", "ieee", "chicago_notes")


@dataclass(frozen=True)
class CitedSource:
    title: str
    url: str


# 没有作者字段时，三种格式都用"网站域名"顶替作者位置——不是瞎编一个机构名，
# 是从 URL 本身解析出来的，可验证、零幻觉风险。
def _site_name(url: str) -> str:
    netloc = urlparse(url).netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or url


def format_reference_gbt7714(index: int, source: CitedSource, access_date: date | None = None) -> str:
    access_date = access_date or date.today()
    site = _site_name(source.url)
    return f"[{index}] {site}. {source.title}[EB/OL]. [{access_date.isoformat()}]. {source.url}."


def format_reference_ieee(index: int, source: CitedSource, access_date: date | None = None) -> str:
    access_date = access_date or date.today()
    site = _site_name(source.url)
    accessed = f"{calendar.month_abbr[access_date.month]}. {access_date.day}, {access_date.year}"
    return f'[{index}] {site}, "{source.title}," [Online]. Available: {source.url}. [Accessed: {accessed}].'


def format_reference_chicago_notes(index: int, source: CitedSource, access_date: date | None = None) -> str:
    access_date = access_date or date.today()
    site = _site_name(source.url)
    accessed = f"{access_date.year}年{access_date.month}月{access_date.day}日"
    return f'{index}. {site}, "{source.title}," 访问于{accessed}, {source.url}.'


FORMATTERS = {
    "gbt7714": format_reference_gbt7714,
    "ieee": format_reference_ieee,
    "chicago_notes": format_reference_chicago_notes,
}


def format_reference(style: str, index: int, source: CitedSource, access_date: date | None = None) -> str:
    formatter = FORMATTERS.get(style)
    if formatter is None:
        raise ValueError(f"Unknown citation style: {style!r}. Expected one of {CITATION_STYLES}.")
    return formatter(index, source, access_date)
