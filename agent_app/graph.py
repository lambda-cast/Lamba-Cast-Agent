from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from .nodes.agent import node_agent
from .registry import TOOL_REGISTRY
from .state import GraphState
from agent_app.nodes.installations import node_fetch_installations, route_installations


def auth_guard(state: GraphState):
    """
    Check if user is authenticated before allowing access to the agent.

    No role check here: this graph is only ever served from the admin UI,
    so reaching it at all implies admin access. If that ever stops being
    true (e.g. this graph gets exposed anywhere else), reinstate a
    session.get("role") != "admin" check here.
    """
    session = state.get("session")

    if not session or not session.get("user_id"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "🔒 Authentication Required\n\n"
                        "You need to be authenticated to use this service. "
                        "Please log in to your account to access your solar installations and data."
                    )
                )
            ],
            "authenticated": False,
        }

    return {"authenticated": True}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_auth(state: GraphState):
    """Route based on authentication status."""
    if state.get("authenticated", False):
        return "installations"
    return END


def route_agent(state: GraphState):
    """Route based on agent's response (tool calls or end)."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build and compile the graph.

    Pass a checkpointer (e.g. langgraph.checkpoint.memory.MemorySaver, or a
    persistent one in production) to keep state across separate
    graph.invoke() calls for the same thread_id -- required for the
    installations node's "which one?" -> admin reply round trip to work
    over separate HTTP requests. Without one, each invoke() starts fresh
    every call.
    """
    graph_builder = StateGraph(GraphState)
    tool_node = ToolNode(TOOL_REGISTRY)

    graph_builder.add_node("auth_guard", auth_guard)
    graph_builder.add_node("installations", node_fetch_installations)
    graph_builder.add_node("agent", node_agent)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "auth_guard")
    graph_builder.add_conditional_edges(
        "auth_guard", route_auth, {"installations": "installations", END: END}
    )
    graph_builder.add_conditional_edges(
        "installations", route_installations, {"agent": "agent", END: END}
    )
    graph_builder.add_conditional_edges("agent", route_agent, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile(checkpointer=checkpointer)


# Module-level default: uncheckpointed, single-shot use (e.g. tests,
# one-off scripts). The API server builds its own checkpointed instance.
graph = build_graph()
graph.get_graph().draw_mermaid()