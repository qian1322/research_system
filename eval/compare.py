"""
CLI: diff two eval/run_eval.py JSON result files and print a per-metric delta
table -- the regression-testing workflow (e.g. before/after a prompt change,
model swap, or RAG top-k tweak).

Usage:
    python -m eval.compare eval/results/20260830-120000-baseline.json eval/results/20260831-093000-after-prompt-fix.json

Caveat: pipeline temperature is > 0 in several nodes (writer=0.7, search=0.3,
rag=0.3), so LLM output is non-deterministic across runs -- a delta here
reflects code/prompt/model changes PLUS run-to-run noise, not code changes
alone. For a real before/after claim, run the harness 2x on each side of the
change and eyeball whether the delta exceeds the noise floor.
"""
import argparse
import json
from pathlib import Path

AGGREGATE_METRICS = [
    "citation_valid_rate",
    "reference_consistent_rate",
    "avg_citation_coverage_ratio",
    "avg_token_count",
    "avg_latency_seconds",
    "max_revision_hit_rate",
    "avg_judge_score",
    "judge_pass_rate",
]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 只对比两次运行都存在的聚合指标；某个字段在某一边是 None（比如那次
# 跑了 --no-judge）就把 delta 记成 None，不硬算一个没有意义的差值。
def diff_aggregate(before: dict, after: dict) -> list[dict]:
    rows = []
    for key in AGGREGATE_METRICS:
        b = before["aggregate"].get(key)
        a = after["aggregate"].get(key)
        delta = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        rows.append({"metric": key, "before": b, "after": a, "delta": delta})
    return rows


# 逐 case 对比 judge 均分：只看两次运行里都出现过的 case_id（用集合交集），
# 新增/删掉的 case 不参与这张表，避免"对比了根本没跑过的东西"。
def diff_per_case(before: dict, after: dict) -> list[dict]:
    before_by_id = {c["case_id"]: c for c in before["cases"]}
    after_by_id = {c["case_id"]: c for c in after["cases"]}
    rows = []
    for case_id in sorted(set(before_by_id) & set(after_by_id)):
        b, a = before_by_id[case_id], after_by_id[case_id]
        b_score = b.get("judge_metrics", {}).get("average_score") if b.get("judge_metrics") else None
        a_score = a.get("judge_metrics", {}).get("average_score") if a.get("judge_metrics") else None
        delta = (a_score - b_score) if b_score is not None and a_score is not None else None
        rows.append({
            "case_id": case_id,
            "before_judge_avg": b_score,
            "after_judge_avg": a_score,
            "delta_judge_avg": delta,
        })
    return rows


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def render_delta_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = "\n".join("| " + " | ".join(_fmt(row[c]) for c in columns) + " |" for row in rows)
    return "\n".join([header, sep, body]) if rows else header + "\n" + sep


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two eval result JSON files.")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)

    print(f"Before: {before.get('label')} ({before.get('timestamp')})")
    print(f"After:  {after.get('label')} ({after.get('timestamp')})\n")

    print("## Aggregate Delta\n")
    print(render_delta_table(diff_aggregate(before, after), ["metric", "before", "after", "delta"]))

    print("\n## Per-Case Judge Score Delta (cases present in both runs)\n")
    print(render_delta_table(
        diff_per_case(before, after),
        ["case_id", "before_judge_avg", "after_judge_avg", "delta_judge_avg"],
    ))


if __name__ == "__main__":
    main()
