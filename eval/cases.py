"""
Fixed, versioned set of eval topics. Deliberately small and spans multiple
domains and both languages the pipeline is used with, so the harness
exercises different vocabularies, source availability, and citation density
without becoming an expensive, slow-to-run benchmark.

Adding/removing a case changes what "the eval set" means for compare.py's
regression story -- treat edits to this file like a schema change.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str        # stable slug; used as filename-safe key and in compare.py matching
    topic: str      # passed verbatim to DeepResearchSystem.research()
    domain: str      # technology | finance | health | climate | history | science | business | education
    language: str     # "zh" | "en" -- informational only, not enforced


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="tech-agentic-ai-zh",
        topic="AI Agent技术在2025年的发展趋势",
        domain="technology",
        language="zh",
    ),
    EvalCase(
        id="finance-llm-investing-en",
        topic="How are hedge funds using large language models for equity research in 2025",
        domain="finance",
        language="en",
    ),
    EvalCase(
        id="health-ai-diagnostics-zh",
        topic="人工智能在医学影像诊断中的最新进展",
        domain="health",
        language="zh",
    ),
    EvalCase(
        id="climate-carbon-capture-en",
        topic="Recent advances in direct air capture technology for carbon removal",
        domain="climate",
        language="en",
    ),
    EvalCase(
        id="history-silk-road-zh",
        topic="丝绸之路对中西方文化交流的历史影响",
        domain="history",
        language="zh",
    ),
    EvalCase(
        id="science-quantum-computing-en",
        topic="Progress toward fault-tolerant quantum computing in 2025",
        domain="science",
        language="en",
    ),
]
