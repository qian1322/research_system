"""
CLI: run the REAL DeepResearchSystem pipeline (real DeepSeek + Tavily calls,
optionally a real OpenAI judge call) over eval/cases.py, score each run with
eval/metrics.py (deterministic) and eval/judge.py (LLM-as-judge), and write a
timestamped JSON + markdown summary to eval/results/.

This costs real money and takes real minutes -- it is a manual, opt-in
workflow, NOT part of `pytest` or CI. See README.md's Evaluation section.

Usage:
    python -m eval.run_eval --label baseline
    python -m eval.run_eval --label baseline --cases tech-agentic-ai-zh,history-silk-road-zh
    python -m eval.run_eval --label baseline --no-judge   # skip the judge LLM call
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.cases import EVAL_CASES, EvalCase
from eval.judge import judge_report
from eval.metrics import compute_rule_based_metrics
from research_system.system import DeepResearchSystem

RESULTS_DIR = Path(__file__).parent / "results"


# 跑单个 case：走 DeepResearchSystem.research() 这个公开入口（不能直接怼
# build_graph()，dispatch_search 用 Send API 手工构造子 state，绕过 research()
# 会漏掉正确的 interrupt/resume 处理）。review_plan=lambda plan: plan 让 Planner
# 提的子问题原样自动通过，全程无人值守。计时只包在 .research() 调用外面，
# 不含打分本身的耗时。
def run_case(case: EvalCase, run_judge: bool = True) -> dict:
    system = DeepResearchSystem()
    start = time.perf_counter()
    result = system.research(case.topic, review_plan=lambda plan: plan)
    latency_seconds = time.perf_counter() - start

    rule_metrics = compute_rule_based_metrics(result)
    rule_metrics["latency_seconds"] = latency_seconds

    judge_metrics = None
    if run_judge:
        judge_metrics = judge_report(
            topic=case.topic,
            research_plan=result.get("research_plan", []),
            report=result.get("final_report", ""),
            search_results=result.get("search_results", []),
        )

    return {
        "case_id": case.id,
        "topic": case.topic,
        "domain": case.domain,
        "language": case.language,
        "error": None,
        "research_plan": result.get("research_plan", []),
        "final_report": result.get("final_report", ""),
        "critique": result.get("critique", ""),
        "rule_based_metrics": rule_metrics,
        "judge_metrics": judge_metrics,
    }


# 跑整个评测集（或者按 --cases 过滤出的一部分），案例之间串行执行 ——
# 每个 case 内部已经在并行 fan-out 好几个 search agent 了，案例间再并行
# 会在 DeepSeek + Tavily + 可选 OpenAI 三个供应商上叠加限流风险，不值得。
# 单个 case 的 API 抖动包在 try/except 里，不会打断整次评测，失败的 case
# 记录 error 并从 aggregate 里排除（但计入 n_errors）。
def run_eval(label: str = "run", case_ids: list[str] | None = None, run_judge: bool = True) -> dict:
    cases = [c for c in EVAL_CASES if case_ids is None or c.id in case_ids]
    case_results = []
    for case in cases:
        print(f"\n=== [{case.id}] {case.topic} ===")
        try:
            case_results.append(run_case(case, run_judge=run_judge))
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            case_results.append({
                "case_id": case.id, "topic": case.topic, "domain": case.domain,
                "language": case.language, "error": str(exc),
                "research_plan": [], "final_report": "", "critique": "",
                "rule_based_metrics": None, "judge_metrics": None,
            })
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "judge_enabled": run_judge,
        "n_cases": len(case_results),
        "cases": case_results,
        "aggregate": aggregate_metrics(case_results),
    }


# 把逐 case 结果汇总成整体统计。只在 rule_based_metrics 不是 None 的 case
# （也就是没报错的）里取平均，judge 相关字段在 --no-judge 或全部 judge
# 调用都失败时为 None，避免误导性的假数据。
def aggregate_metrics(case_results: list[dict]) -> dict:
    valid = [r for r in case_results if r.get("rule_based_metrics")]
    n = len(valid)
    if n == 0:
        return {"n_errors": len(case_results)}

    judged = [r for r in valid if r.get("judge_metrics")]

    return {
        "n_errors": len(case_results) - n,
        "citation_valid_rate": sum(r["rule_based_metrics"]["citation_validity"]["all_valid"] for r in valid) / n,
        "reference_consistent_rate": sum(r["rule_based_metrics"]["reference_list_consistency"]["is_consistent"] for r in valid) / n,
        "avg_citation_coverage_ratio": sum(r["rule_based_metrics"]["citation_coverage"]["coverage_ratio"] for r in valid) / n,
        "avg_token_count": sum(r["rule_based_metrics"]["length"]["token_count"] for r in valid) / n,
        "avg_latency_seconds": sum(r["rule_based_metrics"]["latency_seconds"] for r in valid) / n,
        "max_revision_hit_rate": sum(r["rule_based_metrics"]["revision"]["hit_max_revisions"] for r in valid) / n,
        "avg_judge_score": (sum(r["judge_metrics"]["average_score"] for r in judged) / len(judged)) if judged else None,
        "judge_pass_rate": (sum(r["judge_metrics"]["passed"] for r in judged) / len(judged)) if judged else None,
    }


def _bool_mark(b: bool) -> str:
    return "✓" if b else "✗"


# 把整份评测结果渲染成人类可读的 Markdown 表格 —— 汇总表 + 逐 case 明细表。
def render_markdown(report: dict) -> str:
    agg = report["aggregate"]
    judge_avg_line = (
        f"| Avg judge score (1-5) | {agg['avg_judge_score']:.2f} |"
        if agg.get("avg_judge_score") is not None
        else "| Avg judge score (1-5) | n/a |"
    )
    judge_pass_line = (
        f"| Judge pass rate | {agg['judge_pass_rate']:.1%} |"
        if agg.get("judge_pass_rate") is not None
        else "| Judge pass rate | n/a |"
    )
    lines = [
        f"# Eval Run: {report['label']} ({report['timestamp']})",
        "",
        f"Cases: {report['n_cases']}  |  Judge enabled: {report['judge_enabled']}  |  Errors: {agg.get('n_errors', 0)}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Citation valid rate | {agg.get('citation_valid_rate', 0):.1%} |",
        f"| Reference-list consistent rate | {agg.get('reference_consistent_rate', 0):.1%} |",
        f"| Avg citation coverage | {agg.get('avg_citation_coverage_ratio', 0):.1%} |",
        f"| Avg report length (tokens) | {agg.get('avg_token_count', 0):.0f} |",
        f"| Avg latency (s) | {agg.get('avg_latency_seconds', 0):.1f} |",
        f"| Max-revision hit rate | {agg.get('max_revision_hit_rate', 0):.1%} |",
        judge_avg_line,
        judge_pass_line,
        "",
        "## Per-Case Results",
        "",
        "| ID | Domain | Tokens | Revisions | Latency(s) | Cit Valid | Ref Consistent | Coverage | Judge Avg | Judge Pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        if c.get("error"):
            lines.append(f"| {c['case_id']} | {c['domain']} | ERROR: {c['error']} | | | | | | | |")
            continue
        rm = c["rule_based_metrics"]
        jm = c.get("judge_metrics")
        judge_avg_cell = f"{jm['average_score']:.2f}" if jm else "n/a"
        judge_pass_cell = _bool_mark(jm["passed"]) if jm else "n/a"
        lines.append(
            f"| {c['case_id']} | {c['domain']} | {rm['length']['token_count']} | "
            f"{rm['revision']['revision_count']} | {rm['latency_seconds']:.1f} | "
            f"{_bool_mark(rm['citation_validity']['all_valid'])} | "
            f"{_bool_mark(rm['reference_list_consistency']['is_consistent'])} | "
            f"{rm['citation_coverage']['coverage_ratio']:.0%} | "
            f"{judge_avg_cell} | {judge_pass_cell} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval harness against the real pipeline.")
    parser.add_argument("--label", default="run", help="Short label embedded in the output filenames/report.")
    parser.add_argument("--cases", default=None, help="Comma-separated case ids to run (default: all).")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge scoring step.")
    args = parser.parse_args()

    case_ids = args.cases.split(",") if args.cases else None
    report = run_eval(label=args.label, case_ids=case_ids, run_judge=not args.no_judge)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"{ts}-{args.label}.json"
    md_path = RESULTS_DIR / f"{ts}-{args.label}.md"
    # ensure_ascii=False：不然中文 topic/报告会被转义成 \uXXXX，JSON 文件人眼没法直接读。
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
