from langgraph.types import Command

from research_system.graph import build_graph


class DeepResearchSystem:
    def __init__(self):
        self.app = build_graph()
        self._id = 0

    def start_research(self, topic: str, callbacks: list | None = None) -> dict:
        """Run until the graph interrupts for human review, or finishes.

        callbacks: optional LangChain callback handlers (e.g.
        UsageMetadataCallbackHandler) -- passed through LangGraph's config so
        they see every node's LLM call, not just a top-level one. Used by
        eval/run_eval.py to measure real token usage per run; None in normal
        use, so this doesn't change default behavior.
        """
        self._id += 1
        config = {"configurable": {"thread_id": f"run-{self._id}"}}
        if callbacks:
            config["callbacks"] = callbacks
        initial = {
            "topic": topic,
            "research_plan": [],
            "search_results": [],
            "numbered_sources": [],
            "source_titles": [],
            "rag_context": "",
            "draft_report": "",
            "critique": "",
            "revision_count": 0,
            "quality_approved": False,
            "final_report": "",
        }
        print(f"\nSTART: {topic}\n" + "=" * 60)
        result = self.app.invoke(initial, config=config)
        return {"config": config, "result": result}

    def resume_research(self, config: dict, resume_value) -> dict:
        """Continue a run paused by start_research/resume_research with the human's decision."""
        result = self.app.invoke(Command(resume=resume_value), config=config)
        return {"config": config, "result": result}

    @staticmethod
    def is_interrupted(result: dict) -> bool:
        return bool(result.get("__interrupt__"))

    @staticmethod
    def get_interrupt_payload(result: dict) -> dict:
        return result["__interrupt__"][0].value

    def research(self, topic: str, review_plan=None, callbacks: list | None = None) -> dict:
        """
        End-to-end run with an in-process human review loop.

        review_plan: optional callable(sub_questions: list[str]) -> list[str],
            invoked with the Planner's proposed sub-questions; return the
            (possibly edited) list to approve. If omitted, the plan is
            approved as-is (no pause).
        callbacks: see start_research.
        """
        state = self.start_research(topic, callbacks=callbacks)
        result = state["result"]

        while self.is_interrupted(result):
            payload = self.get_interrupt_payload(result)
            plan = payload["research_plan"]
            approved_plan = review_plan(plan) if review_plan else plan
            state = self.resume_research(state["config"], {"research_plan": approved_plan})
            result = state["result"]

        print(
            f'DONE. {len(result.get("final_report", ""))} chars, '
            f'{result.get("revision_count", 0)} revisions'
        )
        return result

    def print_report(self, r: dict):
        print("\n" + "=" * 70)
        print(r.get("final_report", "No report generated."))
        print("=" * 70)

    def save_report(self, result: dict, filename: str = "report.md") -> str:
        report = result.get("final_report", "")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved: {filename}")
        return filename
