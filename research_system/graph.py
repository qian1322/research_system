from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_system.nodes.critic import critic_node, should_revise
from research_system.nodes.human_review import human_review_node
from research_system.nodes.planner import dispatch_search, planner_node
from research_system.nodes.rag import rag_retriever_node
from research_system.nodes.search import search_agent
from research_system.nodes.writer import writer_node
from research_system.state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("search_agent", search_agent)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "human_review")

    # Fan-out: human-approved plan -> N parallel Search Agents
    graph.add_conditional_edges("human_review", dispatch_search)

    # Fan-in: all Search Agents done -> RAG
    graph.add_edge("search_agent", "rag_retriever")

    graph.add_edge("rag_retriever", "writer")
    graph.add_edge("writer", "critic")

    graph.add_conditional_edges(
        "critic",
        should_revise,
        {"approved": END, "revise": "writer"},
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
