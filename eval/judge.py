"""
LLM-as-judge scoring for eval reports.

Uses its own model-selection logic, separate from research_system.config,
on purpose: when OPENAI_API_KEY is set, judging with a DIFFERENT provider
than the one that generated the report (DeepSeek) reduces same-model
self-evaluation bias. Falls back to the same DeepSeek model the pipeline
itself uses when no OpenAI key is present, so the harness still works with
zero extra keys -- just with a weaker bias guarantee (noted in the README).
"""
from pydantic import BaseModel, Field

from research_system import config as pipeline_config

PASS_THRESHOLD = 4  # out of 5, on the averaged 4-dimension score


class JudgeVerdict(BaseModel):
    coverage_score: int = Field(..., ge=1, le=5, description="Does the report address every planned sub-question?")
    faithfulness_score: int = Field(..., ge=1, le=5, description="Are claims supported by the given source excerpts (not invented/contradicted)?")
    coherence_score: int = Field(..., ge=1, le=5, description="Is the report well-organized, readable, non-repetitive?")
    citation_appropriateness_score: int = Field(..., ge=1, le=5, description="Are [n] citations placed on the claims they actually support?")
    reasoning: str = Field(..., description="Short free-text justification for the scores.")


# 选裁判模型：有 OPENAI_API_KEY 就用 gpt-4o-mini（跨供应商打分，
# 避免 DeepSeek 自己给自己批卷子）；没有的话退回流水线自己用的
# DeepSeek 模型，保证零额外 key 也能跑，只是偏差保证弱一些。
def get_judge_llm(temperature: float = 0.0):
    if pipeline_config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", api_key=pipeline_config.OPENAI_API_KEY, temperature=temperature)
    return pipeline_config.get_llm(temperature=temperature)


# numbered_sources 只保留了裸 URL（见 rag._renumber_to_global），真实的
# 来源文本还留在 search_results[*]["sources"][*]["content"] 里 —— 这里把
# 两者重新对上号，重建"引用编号 -> 来源摘录"，好让裁判模型看到真实内容，
# 而不是只凭一个 URL 字符串去判断忠实度。同一个 URL 在多个子问题里重复
# 出现时，保留第一次遇到的内容。
def build_source_excerpt_map(
    search_results: list[dict], numbered_sources: list[str], max_chars_per_source: int = 400
) -> dict[int, str]:
    url_to_content: dict[str, str] = {}
    for r in search_results:
        for s in r.get("sources", []):
            url = s.get("url")
            if url and url not in url_to_content:
                url_to_content[url] = s.get("content", "")

    return {
        i: url_to_content.get(url, "")[:max_chars_per_source]
        for i, url in enumerate(numbered_sources, start=1)
    }


# 总入口：拼一个包含 topic、计划的子问题、编号来源摘录、报告正文的 prompt，
# 让裁判模型按 4 个维度打分（1-5），method="function_calling" 是硬性要求
# （DeepSeek 的 API 对默认的 json_schema 方式会 400，跟 critic.py/planner.py
# 里的规则一致）。返回值在 4 个打分字段之外，额外算出均分和是否达标。
def judge_report(
    topic: str,
    research_plan: list[str],
    report: str,
    numbered_sources: list[str],
    search_results: list[dict],
    pass_threshold: int = PASS_THRESHOLD,
) -> dict:
    excerpts = build_source_excerpt_map(search_results, numbered_sources)
    sources_block = "\n".join(f"[{n}] {numbered_sources[n - 1]}\n{text}" for n, text in excerpts.items())
    plan_block = "\n".join(f"- {q}" for q in research_plan)

    prompt = (
        f"Topic: {topic}\n\n"
        f"Planned sub-questions:\n{plan_block}\n\n"
        f"Source excerpts (numbered, matching [n] citations in the report below):\n{sources_block}\n\n"
        f"Report:\n{report}\n\n"
        "Score the report on 4 dimensions, 1 (poor) to 5 (excellent):\n"
        "- coverage_score: does it address every planned sub-question above?\n"
        "- faithfulness_score: is every factual claim actually supported by the source "
        "excerpts above (not invented, not contradicted)?\n"
        "- coherence_score: is it well-organized, readable, non-repetitive?\n"
        "- citation_appropriateness_score: are [n] citations placed on the claims they "
        "actually support, rather than missing or misattributed?\n"
        "Give a short reasoning explaining the scores."
    )
    judge_llm = get_judge_llm().with_structured_output(JudgeVerdict, method="function_calling")
    verdict: JudgeVerdict = judge_llm.invoke([("user", prompt)])

    scores = [
        verdict.coverage_score,
        verdict.faithfulness_score,
        verdict.coherence_score,
        verdict.citation_appropriateness_score,
    ]
    average_score = sum(scores) / len(scores)
    return {
        **verdict.model_dump(),
        "average_score": average_score,
        "passed": average_score >= pass_threshold,
    }
