# Multi-Agent Deep Research System

**English** | [中文](README.zh-CN.md)

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

### Iteration history: a real debugging trail, not a single number

A single eval run's absolute score isn't that meaningful on its own (see the
citation-coverage caveat above) — what matters is whether a specific,
isolated change, validated against the same 2 cases, actually moved the
needle. Here's the full trail, including a change that turned out to be a
mistake:

| Run | Change | Avg judge score (1-5) | Judge pass rate | Max-revision hit rate | Avg latency (s) |
|---|---|---|---|---|---|
| `baseline` | — | 2.75 | 0% | 0% | 61.9 |
| `critic-rag-fix` | Critic fact-checks against `rag_context` (previously had no retrieved material at all); RAG `k` 3→6, context cap 500→2000 chars | 3.13 | 0% | 0% | 79.5 |
| `writer-factrule` | Writer told not to invent numbers/case studies, hedge instead | 3.50 | 50% | 0% | 52.7 |
| `critic-per-citation` | Critic checks each `[n]` against its own source excerpt, not one blended summary blob | 3.63 | 0% | 50% | 197.8 |
| `max-revisions-5` *(reverted)* | Revision cap 3→5, hoping more attempts would resolve disagreements | 3.50 | 0% | **100%** | **386.7** |
| `graded-severity` | Critic separates hard fabrication from honest hedged language; cap back to 3 | 3.50 | 0% | 50% | 129.1 |

What each row actually found:
- **`critic-rag-fix`**: the Critic originally had *no* retrieved material to
  check the draft against — it was rating structure and tone, not truth.
  Giving it `rag_context`, plus widening the RAG budget so the Writer had
  enough real material to draw from, nearly doubled citation coverage
  (46%→94%) and lifted the judge score — but didn't fix fabrication on its own.
- **`writer-factrule`**: telling the Writer directly not to invent specifics
  — and to prefer honest, vague language over a fabricated precise one — was
  the single highest-leverage change in the series (first case ever to pass;
  biggest single-step jump in judge score).
- **`critic-per-citation`**: the Writer's fabrication had shifted from
  *inventing* numbers to *blending* real ones (citing 85% when a source said
  79%) — too subtle for one compressed context blob to catch. Reconstructing
  per-citation source excerpts (the shared implementation now lives in
  `critic.py`; `eval/judge.py` imports it instead of duplicating it) caught
  this, but its zero-tolerance rejection rule created a **deadlock**: Critic
  scores oscillated 4-6/10 across every revision, never reaching the approval
  bar, because report-writing style keeps pulling the Writer toward specifics
  the fact-check kept rejecting.
- **`max-revisions-5` — the failed attempt**: raising the revision cap to
  give that deadlock more room to resolve seemed reasonable. It wasn't:
  **both** cases hit the new cap and got force-approved anyway (which skips
  fact-checking entirely), latency nearly doubled, and the judge score didn't
  move. More attempts don't help when the approval bar itself is structurally
  unsatisfiable — that was the actual bug. Reverted.
- **`graded-severity`**: splitting fact-check violations into HARD (invented
  facts, unsupported/blended numbers — still blocks approval) vs. SOFT
  (honest hedged phrasing — noted, doesn't block) broke the deadlock: one
  case was genuinely approved by the Critic for the first time, not
  force-approved, and latency dropped 3x from the previous run. Notably, the
  *independent* judge score for that same case didn't move at all (3.5 before
  and after) — this fixed the Critic's internal consistency, not the report's
  objective quality. The gap between "the pipeline's own Critic approves it"
  and "an independent judge thinks it's good" is still open — which is
  exactly why this harness exists as a check independent of the pipeline's
  own judgment, rather than trusting the Critic's self-report.

Full per-run JSON/MD (topics, drafts, and each judge's itemized `reasoning`)
lives in `eval/results/`, which is gitignored — commit history and the
commit message on each change above are the durable record of what was
tried, in what order, and why (including the revert).

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
  comparison; drove 6 validated iterations with it (one deliberately reverted
  after data showed it made things worse), raising the average judge score
  from 2.75 to 3.50 (out of 5) and getting the first case to pass, while
  surfacing failure modes (fabricated case studies, blended statistics, a
  Critic approval deadlock) the pipeline's own Critic and the fake-based unit
  tests both missed.
```
