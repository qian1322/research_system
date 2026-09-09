from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_system.nodes.citation_style import citation_style_node
from research_system.nodes.critic import critic_node, should_revise
from research_system.nodes.edit_review import edit_review_node, should_continue_editing
from research_system.nodes.human_review import human_review_node
from research_system.nodes.planner import dispatch_search, planner_node
from research_system.nodes.rag import rag_retriever_node
from research_system.nodes.search import search_agent
from research_system.nodes.writer import writer_node
from research_system.state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("citation_style", citation_style_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("search_agent", search_agent)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("edit_review", edit_review_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "citation_style")
    graph.add_edge("citation_style", "human_review")

    # Fan-out: human-approved plan -> N parallel Search Agents
    graph.add_conditional_edges("human_review", dispatch_search)

    # Fan-in: all Search Agents done -> RAG
    graph.add_edge("search_agent", "rag_retriever")

    graph.add_edge("rag_retriever", "writer")
    graph.add_edge("writer", "critic")

    graph.add_conditional_edges(
        "critic",
        should_revise,
        {"approved": "edit_review", "revise": "writer"},
    )

    # Post-approval edit loop: pause for an optional follow-up instruction,
    # loop back through writer/critic if one is given, or finish.
    graph.add_conditional_edges(
        "edit_review",
        should_continue_editing,
        {"edit": "writer", "done": END},
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
