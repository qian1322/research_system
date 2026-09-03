"""
LLM-as-judge scoring for eval reports.

Reuses research_system.nodes.critic's QualityVerdict schema and
build_quality_prompt -- the same rubric the pipeline's own Critic scores
against -- so a gap between "the pipeline approved it" and "the judge liked
it" reflects an actual disagreement, not two different rubrics worded
differently. What stays independent here is the model: when OPENAI_API_KEY
is set, judging uses a DIFFERENT provider (gpt-4o-mini) than the one that
generated and critiqued the report (DeepSeek), reducing same-model
self-evaluation bias. Falls back to the same DeepSeek model the pipeline
itself uses when no OpenAI key is present, so the harness still works with
zero extra keys -- just with a weaker bias guarantee (noted in the README).
"""
from research_system import config as pipeline_config
from research_system.nodes.critic import PASS_THRESHOLD, QualityVerdict, build_quality_prompt


# 选裁判模型：有 OPENAI_API_KEY 就用 gpt-4o-mini（跨供应商打分，
# 避免 DeepSeek 自己给自己批卷子）；没有的话退回流水线自己用的
# DeepSeek 模型，保证零额外 key 也能跑，只是偏差保证弱一些。
def get_judge_llm(temperature: float = 0.0):
    if pipeline_config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", api_key=pipeline_config.OPENAI_API_KEY, temperature=temperature)
    return pipeline_config.get_llm(temperature=temperature)


# 总入口：跟 critic_node 共用同一份 build_quality_prompt + QualityVerdict，
# 唯一的区别是裁判模型可能换了供应商。method="function_calling" 是硬性要求
# （DeepSeek 的 API 对默认的 json_schema 方式会 400，跟 critic.py/planner.py
# 里的规则一致）。返回值在 4 个打分字段之外，额外算出均分和是否达标。
def judge_report(
    topic: str,
    research_plan: list[str],
    report: str,
    search_results: list[dict],
    pass_threshold: int = PASS_THRESHOLD,
) -> dict:
    prompt = build_quality_prompt(topic, research_plan, report, search_results)
    judge_llm = get_judge_llm().with_structured_output(QualityVerdict, method="function_calling")
    verdict: QualityVerdict = judge_llm.invoke([("user", prompt)])

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
