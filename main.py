import argparse
import sys

from research_system import config
from research_system.system import DeepResearchSystem


def review_plan_cli(sub_questions: list) -> list:
    print("\n[Human Review] Planner 生成了以下子问题:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")
    choice = input(
        "\n直接回车批准原计划;输入 'e' 逐条编辑;或输入序号(如 '1,3')只保留这些子问题: "
    ).strip()

    if not choice:
        return sub_questions

    if choice.lower() == "e":
        edited = []
        for i, q in enumerate(sub_questions, 1):
            new_q = input(f"  {i}. [{q}]\n     改为(回车保留): ").strip()
            edited.append(new_q or q)
        return edited

    try:
        keep = [int(x) for x in choice.split(",")]
        return [sub_questions[i - 1] for i in keep if 1 <= i <= len(sub_questions)]
    except ValueError:
        print("  输入无法识别,按原计划继续。")
        return sub_questions


def review_edit_cli(final_report: str) -> tuple[str, int] | None:
    print("\n[Human Review] 报告已生成/已更新。")
    instruction = input("输入后续修改指令(直接回车结束编辑): ").strip()
    if not instruction:
        return None
    k_input = input("涉及几段?(直接回车默认 2): ").strip()
    k = int(k_input) if k_input.isdigit() else 2
    return instruction, k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent deep research system (LangGraph + DeepSeek)"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="AI Agent技术发展趋势",
        help="Research topic (default: %(default)s)",
    )
    args = parser.parse_args()

    if not config.DEEPSEEK_API_KEY:
        sys.exit("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in.")

    system = DeepResearchSystem()
    result = system.research(args.topic, review_plan=review_plan_cli, review_edit=review_edit_cli)
    system.print_report(result)


if __name__ == "__main__":
    main()
