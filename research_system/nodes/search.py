from research_system import config
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

    prompt = (
        f'Research topic: {state["topic"]}\n'
        f"Sub-question: {sub_q}\n\n"
        f"Web search results (numbered):\n{sources_block}\n\n"
        "Using ONLY the search results above, extract the key findings, data, "
        "and examples that answer the sub-question.\n"
        "Citation rule: after each fact, cite the result it came from using its "
        f"bracketed number, e.g. '...market size reached $X [2]'. Only use numbers "
        f"1-{len(sources)} that appear above -- never invent one, and never write "
        "out the URL itself. A sentence may cite multiple sources, e.g. [1][3].\n"
        "Respond in the same language as the topic above, under 300 characters."
    )
    search_llm = config.get_llm(temperature=0.3)
    response = search_llm.invoke([("user", prompt)])
    return {
        "search_results": [
            {"question": sub_q, "content": response.content, "sources": sources}
        ]
    }
