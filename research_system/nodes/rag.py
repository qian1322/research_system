import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from research_system import config
from research_system.state import ResearchState


def _renumber_to_global(search_results: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Each search_agent runs in parallel and cites sources with its own locally
    scoped [1][2][3]. Before merging, remap every result's citations onto one
    pipeline-wide numbering so [n] means the same source everywhere and lines
    up 1:1 with the flattened source list.
    """
    global_sources: list[str] = []
    renumbered = []
    for r in search_results:
        local_sources = r.get("sources", [])
        offset = len(global_sources)

        def _remap(match, offset=offset, n_local=len(local_sources)):
            n = int(match.group(1))
            if 1 <= n <= n_local:
                return f"[{offset + n}]"
            return match.group(0)  # out-of-range number: leave it untouched

        new_content = re.sub(r"\[(\d+)\]", _remap, r["content"])
        renumbered.append({**r, "content": new_content})
        global_sources.extend(s["url"] for s in local_sources)

    return renumbered, global_sources


def rag_retriever_node(state: ResearchState) -> dict:
    """
    Real vector-based retrieval (Chroma), not just LLM summarization:
    1. Remap each parallel search agent's local [n] citations onto one global numbering.
    2. Embed every (renumbered) search result into an in-memory Chroma index.
    3. Retrieve the top-k chunks most relevant to the overall research topic.
    4. Ask the LLM to synthesize those chunks into a compact context, preserving citations.
    """
    search_results = state["search_results"]
    print(f"[RAG] Embedding {len(search_results)} results into Chroma")

    renumbered_results, global_sources = _renumber_to_global(search_results)

    docs = [
        Document(page_content=r["content"], metadata={"question": r["question"]})
        for r in renumbered_results
    ]
    # Fresh in-memory collection per research run -- no cross-run leakage.
    vectorstore = Chroma.from_documents(documents=docs, embedding=config.get_embeddings())
    # k=6 comfortably covers the planner's max of 5 sub-questions (one chunk per
    # sub-question, see docs above) -- previously k=3 silently dropped whichever
    # chunks scored lowest on topic similarity, starving the Writer of real
    # material for those sub-questions and pushing it toward inventing content.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke(state["topic"])
    print(f"  -> retrieved {len(retrieved_docs)} relevant chunks")

    retrieved_text = "\n\n".join(
        f"[Chunk {i}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs, 1)
    )

    prompt = (
        f'Topic: {state["topic"]}\n\n'
        f"Most relevant retrieved chunks below (the [n] markers inside them are "
        f"citation numbers pointing to source URLs):\n{retrieved_text}\n\n"
        "Synthesize the key findings, data, and examples most useful for writing "
        "the report.\n"
        "Citation rule: keep every [n] marker exactly as written -- don't remove, "
        "renumber, or invent one. If a piece of information has no marker, don't add one.\n"
        # Previously capped at 500 chars for a report covering 3-5 sub-questions --
        # too little real material for the Writer to draw on, which pushed it
        # toward fabricating specifics to fill the gaps (see README's Evaluation
        # section for a real example this surfaced).
        "Respond in Chinese, under 2000 characters."
    )
    rag_llm = config.get_llm(temperature=0.3)
    response = rag_llm.invoke([("user", prompt)])
    print(f"  -> Context: {len(response.content)} chars")
    return {"rag_context": response.content, "numbered_sources": global_sources}
