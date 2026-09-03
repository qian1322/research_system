# Multi-Agent Deep Research System

A LangGraph-based multi-agent pipeline that takes a research topic, plans it into
sub-questions, researches them in parallel, synthesizes the findings, and writes
a report through an automated Writer-Critic revision loop.

```
User Input
  -> Planner (splits topic into 3-5 sub-questions)
  -> Send API (parallel Search Agents, one per sub-question)
       each agent: Tavily web search -> LLM extracts findings, citing sources as [n]
  -> RAG Retriever (remaps citations to a global numbering, embeds results into
       Chroma, retrieves the top-k chunks most relevant to the topic)
  -> Writer (drafts the report, keeping [n] citation markers + a References section)
  -> Critic (scores the draft, loops back to Writer up to 3x if not approved;
       on approval, compresses citation numbers and finalizes the report)
  -> Final Report
```

LLM backend: [DeepSeek](https://api-docs.deepseek.com/) (`deepseek-chat`), via an
OpenAI-compatible `langchain-openai` client. Web search: [Tavily](https://tavily.com/).
Vector store: [Chroma](https://www.trychroma.com/) (in-memory, per research run).
Optional tracing: [LangSmith](https://smith.langchain.com/).

## Project layout

```
research_system/
├── main.py                     # CLI entry point
├── app.py                      # Streamlit UI
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── research_system/
│   ├── config.py                # env loading + get_llm() / get_search_client() / get_embeddings()
│   ├── state.py                 # ResearchState (LangGraph state schema)
│   ├── graph.py                 # builds & compiles the StateGraph
│   ├── system.py                # DeepResearchSystem: run + print/save convenience wrapper
│   └── nodes/
│       ├── planner.py           # planner_node, dispatch_search (Send API fan-out)
│       ├── search.py            # search_agent: Tavily web search + LLM extraction, cites [n]
│       ├── rag.py                # rag_retriever_node: citation remap + Chroma vector retrieval
│       ├── writer.py             # writer_node: drafts report, preserves [n] citations
│       └── critic.py             # critic_node, should_revise, renumber_citations
├── eval/                        # opt-in evaluation harness, runs the REAL pipeline (see ## Evaluation)
│   ├── cases.py                  # fixed set of eval topics
│   ├── metrics.py                # deterministic, API-free citation/length metrics
│   ├── judge.py                  # LLM-as-judge scoring (coverage/faithfulness/coherence/citations)
│   ├── run_eval.py                # CLI: run real pipeline + score + persist to eval/results/
│   └── compare.py                 # CLI: regression diff between two saved eval runs
└── tests/
    ├── conftest.py               # FakeLLM/FakeTavilySearch/FakeEmbeddings so tests never hit the network
    ├── test_critic.py            # unit test for the revision-loop routing logic
    ├── test_pipeline.py          # end-to-end graph run with the fakes
    └── test_eval_metrics.py      # unit tests for eval/metrics.py's pure functions
```

## Key design points

- **Parallel fan-out with the Send API.** `dispatch_search` returns a `list[Send]`,
  one per sub-question; LangGraph executes all of them concurrently.
  `search_results` uses an `operator.add` reducer so every parallel branch appends
  to the same list instead of overwriting it.
- **Writer-Critic revision loop with a hard stop.** The critic returns a
  structured (Pydantic) verdict; if not approved, `revision_count` increments and
  control routes back to the writer. `revision_count >= MAX_REVISIONS` (3)
  force-approves to guarantee termination.
- **Checkpointed runs.** The graph is compiled with `MemorySaver`, so each call to
  `DeepResearchSystem.research()` gets its own `thread_id` and is independently
  resumable/inspectable.
- **Grounded search.** Each parallel `search_agent` call runs a real
  [Tavily](https://tavily.com/) web search for its sub-question first, then asks
  the LLM to extract findings using only those results — the LLM isn't answering
  from parametric memory. Each fact is cited with a bracketed number (`[n]`)
  pointing back to the source it came from.
- **Real vector retrieval.** `rag_retriever_node` embeds every search result
  (via OpenAI embeddings if `OPENAI_API_KEY` is set, otherwise a local
  HuggingFace model) into an in-memory Chroma index scoped to that research
  run, then retrieves only the top-k chunks most relevant to the topic —
  the writer never sees every raw result, just the retrieved subset.
- **Citations survive the whole pipeline.** Search agents run in parallel and
  each cites sources with its own locally-scoped `[1][2][3]`; the RAG step
  remaps every result onto one pipeline-wide numbering (`numbered_sources`)
  before embedding, so `[n]` means the same source everywhere downstream. The
  writer keeps those markers and adds a `## References` section; once the
  critic approves (or the revision cap forces approval), `renumber_citations`
  compresses any gaps in the numbering and rebuilds `## References` to list
  only the sources actually cited in the final text.
- **Optional LangSmith tracing.** Setting `LANGCHAIN_API_KEY` turns on full
  pipeline tracing with zero code changes elsewhere — `config.py` wires the
  env vars once at import time.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt + pytest
cp .env.example .env                  # then fill in DEEPSEEK_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
python main.py "AI Agent技术发展趋势"
# or, with the default topic:
python main.py
```

```python
from research_system.system import DeepResearchSystem

system = DeepResearchSystem()
result = system.research("AI Agent技术发展趋势")
system.print_report(result)
system.save_report(result, "report.md")
```

Or launch the Streamlit UI:

```bash
streamlit run app.py
```

## Testing

Tests mock the LLM layer (`research_system.config.get_llm`), the search layer
(`research_system.config.get_search_client`), and the embeddings layer
(`research_system.config.get_embeddings`) so they run offline, deterministically,
and without any API keys:

```bash
pytest
```

## Evaluation

`pytest` (above) checks pipeline *logic* against fakes — routing, state shapes.
It can't tell you whether the pipeline produces a good, faithfully-cited
report from a real model and real search results, since `FakeLLM` always
returns the same static string regardless of the prompt. `eval/` is a
separate, opt-in harness that runs the REAL pipeline (real DeepSeek + Tavily
calls) against a fixed set of topics (`eval/cases.py`) and scores each run
two ways:

**Rule-based (deterministic, no API calls — `eval/metrics.py`, unit-tested
offline in `tests/test_eval_metrics.py`):**
- citation validity — every `[n]` in the report resolves to a real, in-range source
- reference-list consistency — body citations and `## References` entries
  match exactly (this doubles as a regression check on `renumber_citations()`
  against real LLM output, which the fake-based tests can't reach)
- citation coverage — fraction of retrieved sources actually cited (has a
  structural ceiling tied to RAG's top-k chunk retrieval — see the caveat in
  `eval/metrics.py`; more useful for before/after comparison than as an
  absolute score)
- report length (chars + tokens, via `tiktoken`'s `cl100k_base` as a
  consistent length proxy, not an exact DeepSeek token count)
- revision count — did the Writer-Critic loop hit its `MAX_REVISIONS` hard stop
- wall-clock latency

**LLM-as-judge (`eval/judge.py`, structured output, 1-5 per dimension):**
coverage (does the report address every planned sub-question), faithfulness
(checked against real retrieved source excerpts, not just the report text),
coherence, and citation appropriateness. When `OPENAI_API_KEY` is set,
judging uses `gpt-4o-mini` instead of DeepSeek, to avoid the pipeline's own
model grading its own output.

### Running it

```bash
python -m eval.run_eval --label baseline
python -m eval.run_eval --label baseline --cases tech-agentic-ai-zh,history-silk-road-zh
python -m eval.run_eval --label baseline --no-judge   # skip the judge LLM call
```

Writes `eval/results/<timestamp>-<label>.json` (full data, human-readable
Chinese — not escaped) and a matching `.md` summary table. `eval/results/`
is gitignored.

### Regression comparison

```bash
python -m eval.compare eval/results/<before>.json eval/results/<after>.json
```

Prints an aggregate delta table plus a per-case judge-score delta — the
intended workflow for checking whether a prompt/model/RAG-`k` change
actually helped, not just "feels different." Caveat: several nodes run
above temperature 0 (writer=0.7, search=0.3, rag=0.3), so a single
before/after pair reflects the change *plus* run-to-run noise — run each
side twice if you need a confident before/after claim.

### Sample output (real 2-case run, no code changes yet)

| Metric | Value |
|---|---|
| Citation valid rate | 100% |
| Reference-list consistent rate | 100% |
| Avg citation coverage | 46% |
| Avg report length (tokens) | 4816 |
| Avg latency (s) | 61.9 |
| Max-revision hit rate | 0% |
| Avg judge score (1-5) | 2.75 |
| Judge pass rate | 0% |

This run's `judge_metrics.reasoning` fields caught a real failure mode
`pytest`'s fakes structurally can't reach: the Writer fabricating specific
case studies and statistics not present in any retrieved source, which the
pipeline's own Critic (checking structure/coherence, not fact-grounding)
approved on the first draft both times (`revision_count: 0`). See a saved
run's JSON for the itemized per-claim findings.

### Cost caveat

`eval/run_eval.py` makes real DeepSeek + Tavily API calls (and OpenAI calls
if `OPENAI_API_KEY` is set) for every case — this costs money and takes
minutes, so it is intentionally **not** part of `pytest`/CI. The rule-based
metric functions themselves are pure and fully unit-tested offline.

## Resume bullet

```
Multi-Agent Deep Research System — LangGraph, DeepSeek

- Architected a multi-agent research pipeline: Planner -> parallel Search Agents
  (LangGraph Send API fan-out, grounded in live Tavily web search) -> Chroma
  vector retrieval -> Writer -> Critic revision loop.
- Implemented Send API fan-out with an operator.add reducer, enabling N parallel
  search agents with automatic result merging.
- Built a citation pipeline that survives parallel fan-out: per-agent [n]
  markers are remapped to a global numbering, embedded into Chroma for top-k
  retrieval, and compressed/deduplicated into a References section once the
  report is finalized.
- Built a Writer-Critic revision loop with Pydantic structured output and a hard
  revision-count limit to guarantee termination.
- Backed all agents with DeepSeek's OpenAI-compatible API for ~95% lower LLM cost
  vs. GPT-4-class models; added optional LangSmith tracing and a Streamlit UI.
- Built an offline evaluation harness (deterministic citation/reference-integrity
  checks + a cross-provider LLM-as-judge on coverage/faithfulness/coherence) that
  runs the real pipeline end-to-end and supports before/after regression
  comparison; it surfaced a real hallucination failure mode (fabricated case
  studies/statistics) that the pipeline's own Critic and the fake-based unit
  tests both missed.
```
