r"""
Deterministic, API-free metrics over a finished ResearchState result.

Every function here is a pure function of a report string / numbered_sources
list / plain result dict -- no LLM calls, no network I/O, safe to unit test
directly. Citation parsing mirrors research_system.nodes.critic's
`\[(\d+)\]` / "## References" contract on purpose: these metrics double as a
regression check on renumber_citations' output against REAL LLM citation
behavior (the existing tests only exercise it against FakeLLM strings).
"""
import re
from functools import lru_cache

import tiktoken

from research_system.nodes.critic import MAX_REVISIONS

REFERENCES_MARKER = "## References"
CITATION_RE = re.compile(r"\[(\d+)\]")
REFERENCE_LINE_RE = re.compile(r"^\[(\d+)\]")

# deepseek-chat has no tiktoken encoding registered (tiktoken.encoding_for_model
# would raise KeyError), so an explicit general-purpose encoding is used as a
# consistent length proxy -- NOT an exact token-billing count for DeepSeek.
TOKEN_ENCODING = "cl100k_base"


# 加载 tiktoken 分词器(有一定开销),用 lru_cache 缓存住,
# 同一个 encoding_name 只会真正加载一次,后面重复调用直接返回缓存结果。
@lru_cache(maxsize=None)
def _get_encoding(name: str):
    return tiktoken.get_encoding(name)


# 把报告从 "## References" 这个标题处切成 (正文, 参考文献) 两段。
# 后面所有函数都是在这两段各自独立的文本上做提取/判断，
# 不会跨段落误判 —— 比如正文里提到的 [n] 不会被当成参考文献条目。
def split_body_and_references(report: str) -> tuple[str, str]:
    """Everything before '## References' is body; everything after is refs.
    No marker found -> whole report is treated as body."""
    idx = report.find(REFERENCES_MARKER)
    if idx == -1:
        return report, ""
    return report[:idx], report[idx + len(REFERENCES_MARKER):]


# “读”正文：把正文里所有 [n] 引用标记的编号按出现顺序提取出来，
# 保留重复（同一个来源引用两次就出现两次），不做去重、不做合法性判断，
# 这些留给下游的 citation_validity / reference_list_consistency 等函数处理。
def extract_body_citations(report: str) -> list[int]:
    """All [n] numbers cited in the body, in order of appearance, WITH duplicates."""
    body, _ = split_body_and_references(report)
    return [int(m.group(1)) for m in CITATION_RE.finditer(body)]


# “读”参考文献列表：逐行检查每一行是不是以 [n] 开头（用 ^ 锚定行首），
# 是的话把编号记下来。用 .match() 而不是 .finditer()，
# 是为了避免误把正文里出现的 [n] 当成参考文献条目。
def extract_reference_entries(report: str) -> list[int]:
    """[n] numbers that head a line in the References section."""
    _, refs = split_body_and_references(report)
    entries = []
    for line in refs.splitlines():
        m = REFERENCE_LINE_RE.match(line.strip())
        if m:
            entries.append(int(m.group(1)))
    return entries


# 检查“有没有编造引用”：正文里每个 [n] 的 n 必须落在
# 1..来源总数 的合法范围内，超出范围就说明这个引用是 LLM 凭空写出来的、
# 根本没对应的真实来源（比如只检索到 3 篇来源，正文却写了 [9]）。
def citation_validity(report: str, numbered_sources: list[str]) -> dict:
    """Every [n] cited in-body must resolve within 1..len(numbered_sources)."""
    cited = extract_body_citations(report)
    n_sources = len(numbered_sources)
    invalid = sorted({n for n in cited if not (1 <= n <= n_sources)})
    return {
        "total_citation_marks": len(cited),
        "unique_cited_numbers": len(set(cited)),
        "invalid_citation_numbers": invalid,
        "all_valid": len(invalid) == 0,
    }


# 检查“正文引用”和“参考文献列表”这两份清单对不对得上：
# 只看落在合法范围内的引用（越界的交给 citation_validity 管），
# 求差集找出“正文引用了但列表漏列”和“列表列了但正文没引用过”两种不一致。
# 这本质上是在给流水线自己 renumber_citations() 的输出做回归检查。
def reference_list_consistency(report: str, numbered_sources: list[str]) -> dict:
    """In-body cites <-> '## References' entries must match exactly:
    no cited number missing a References entry, no orphan References entry."""
    n_sources = len(numbered_sources)
    body_cited = {n for n in extract_body_citations(report) if 1 <= n <= n_sources}
    ref_entries = set(extract_reference_entries(report))
    missing_from_references = sorted(body_cited - ref_entries)
    orphan_references = sorted(ref_entries - body_cited)
    return {
        "missing_from_references": missing_from_references,
        "orphan_references": orphan_references,
        "is_consistent": not missing_from_references and not orphan_references,
    }


# 算“来源利用率”：RAG 检索到了一堆来源（numbered_sources），
# 但报告不一定每个都用上了。这个比例越低，说明检索到的内容
# 和最终报告的相关性可能不够高，是衡量 RAG 效果的一个侧面指标。
#
# 警示：这个比例有一个结构性天花板，跟流水线质量无关 —— rag.py 里
# top-k=3 只挑 3 个"子问题级别"的 chunk 喂给 Writer，没被选中的 chunk
# 对应的来源永远不可能被引用到。所以单次运行的绝对值不适合直接当
# "好/坏"判断标准，更适合同一套流水线不同版本之间做横向对比
# （比如改了 k 或者改了 prompt 前后，看这个比例涨了还是跌了）。
def citation_coverage(report: str, numbered_sources: list[str]) -> dict:
    """Fraction of retrieved numbered_sources actually cited in the final report."""
    n_sources = len(numbered_sources)
    if n_sources == 0:
        return {"cited_source_count": 0, "total_source_count": 0, "coverage_ratio": 0.0}
    body_cited = {n for n in extract_body_citations(report) if 1 <= n <= n_sources}
    return {
        "cited_source_count": len(body_cited),
        "total_source_count": n_sources,
        "coverage_ratio": len(body_cited) / n_sources,
    }


# 算报告长度：字符数直接用 len()，token 数则借助 tiktoken 分词器。
# 注意 cl100k_base 不是 DeepSeek 真实计费用的编码方式，
# 只是一把统一的“尺子”，方便不同版本的报告长度互相比较。
def report_length(report: str, encoding_name: str = TOKEN_ENCODING) -> dict:
    encoding = _get_encoding(encoding_name)
    return {"char_count": len(report), "token_count": len(encoding.encode(report))}


# 读取 Writer-Critic 循环跑了几轮修订、有没有撞到硬上限（MAX_REVISIONS，
# 从 critic.py 导入，跟流水线实际用的值保持同步，不在这里另外写死一份）。
# 撞到上限意味着 critic 从未真正“满意”过，是被强制通过的，
# 值得在评测报告里单独标出来。
def revision_stats(result: dict, max_revisions: int = MAX_REVISIONS) -> dict:
    revision_count = result.get("revision_count", 0)
    return {"revision_count": revision_count, "hit_max_revisions": revision_count >= max_revisions}


# 总入口：接收 DeepResearchSystem.research() 返回的完整结果字典，
# 依次调用上面所有的小函数，把所有确定性指标打包成一个嵌套字典返回。
# eval/run_eval.py 只需要调用这一个函数，不用逐个手动拼装。
def compute_rule_based_metrics(result: dict) -> dict:
    """Aggregate all deterministic metrics for one finished pipeline result
    (the dict returned by DeepResearchSystem.research()). Does NOT include
    latency -- that's wall-clock, measured by the caller around the
    .research() call, not derivable from the result dict."""
    report = result.get("final_report", "")
    numbered_sources = result.get("numbered_sources", [])
    return {
        "citation_validity": citation_validity(report, numbered_sources),
        "reference_list_consistency": reference_list_consistency(report, numbered_sources),
        "citation_coverage": citation_coverage(report, numbered_sources),
        "length": report_length(report),
        "revision": revision_stats(result),
    }
