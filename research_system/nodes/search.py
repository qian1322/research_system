from research_system import config
from research_system.prompts import search_agent_prompt
from research_system.state import ResearchState


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Real web search via Tavily. Returns [{title, url, content}, ...]."""
    tool = config.get_search_client(max_results=max_results)
    response = tool.invoke({"query": query})
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response.get("results", [])
    ]


def search_agent(state: ResearchState) -> dict:
    """
    Handles ONE sub-question per call.
    Returns {"search_results": [one_result]}.
    operator.add merges all parallel results into the main state.
    """
    sub_q = state.get("sub_question", "")
    print(f"  [Search] {sub_q[:50]}")

    sources = web_search(sub_q)
    print(f"    -> {len(sources)} web results")

    sources_block = "\n\n".join(
        f'[{i}] {s["title"]} ({s["url"]})\n{s["content"]}'
        for i, s in enumerate(sources, 1)
    )

    prompt = search_agent_prompt(
        topic=state["topic"], sub_question=sub_q, sources_block=sources_block, n_sources=len(sources)
    )
    search_llm = config.get_llm(temperature=0.3)
    response = search_llm.invoke([("user", prompt)])
    return {
        "search_results": [
            {"question": sub_q, "content": response.content, "sources": sources}
        ]
    }
