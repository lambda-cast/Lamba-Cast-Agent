import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agent_app.state import GraphState, Installation
from agent_app.database.platform_db import PlatformDatabase


def _fetch_installations_for_owner(owner_user_id: int) -> list[Installation]:
    """Fetch installations for a user using PlatformDatabase."""
    db = PlatformDatabase()
    installations = db.list_installations_for_user(owner_user_id)
    return [{"id": inst["id"], "name": inst["name"]} for inst in installations]


def _format_options(installations: list[Installation]) -> str:
    return "\n".join(
        f"{i + 1}. {inst['name']} (id={inst['id']})" for i, inst in enumerate(installations)
    )


def _last_human_text(state: GraphState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return ""


def _match_installation(text: str, installations: list[Installation]):
    if not text:
        return None
    text_lower = text.lower()

    # 1) bare number — position in the list OR literal installation id
    m = re.search(r"\b(\d+)\b", text_lower)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(installations):
            return installations[idx - 1]
        for inst in installations:
            if inst["id"] == idx:
                return inst

    # 2) unambiguous name substring match
    name_matches = [inst for inst in installations if inst["name"].lower() in text_lower]
    if len(name_matches) == 1:
        return name_matches[0]

    return None


def node_fetch_installations(state: GraphState) -> dict:
    """Resolve state['installation_id'].

    - No installations → inform the user and end.
    - One installation  → auto-select.
    - Multiple          → try to resolve from what the user just said;
                          if still ambiguous, ask them to pick (no interrupt,
                          just an AIMessage) and return without an
                          installation_id so the graph routes to END and
                          waits for the next user message.
    """
    session = state.get("session") or {}
    owner_user_id = session.get("user_id")

    installations = _fetch_installations_for_owner(owner_user_id)

    if not installations:
        return {
            "installations": [],
            "installation_id": None,
            "messages": [
                AIMessage(content="No installations are registered to your account yet.")
            ],
        }

    # Already resolved in a previous turn — keep it.
    current_id = state.get("installation_id")
    if current_id is not None and any(i["id"] == current_id for i in installations):
        return {"installations": installations}

    # Exactly one → auto-select silently.
    if len(installations) == 1:
        return {"installations": installations, "installation_id": installations[0]["id"]}

    # Multiple: try to resolve from the user's latest message.
    match = _match_installation(_last_human_text(state), installations)
    if match:
        return {"installations": installations, "installation_id": match["id"]}

    # Still ambiguous → ask the user to pick; do NOT use interrupt().
    # The graph will route to END; the next user message will come back
    # here and the _match_installation above will pick it up.
    options_text = _format_options(installations)
    return {
        "installations": installations,
        "installation_id": None,
        "messages": [
            AIMessage(
                content=(
                    "You have multiple installations. Which one would you like to work with?\n\n"
                    f"{options_text}\n\n"
                    "Reply with the number, name, or installation ID."
                )
            )
        ],
    }


def route_installations(state: GraphState):
    """Let the agent run once an installation has been resolved."""
    if state.get("installation_id") is not None:
        return "agent"
    return END
