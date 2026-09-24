import logging
import os
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from agent_app.prompts import SYSTEM_PROMPT
from agent_app.registry import TOOL_REGISTRY
from agent_app.state import GraphState

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

llm = ChatLiteLLM(
    api_base=os.getenv("LITELLM_HOST"),
    model=os.getenv("LITELLM_MODEL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    },
    temperature=0.2,
)

def node_agent(state: GraphState):
    # Extract user context from session
    session = state.get("session") or {}
    user_role = session.get("role", "user")
    username = session.get("username", "User")
    user_id = session.get("user_id")
    
    # Optionally customize system prompt based on role
    role_context = ""
    if user_role == "admin":
        role_context = f"\n\nCurrent user '{username}' (ID: {user_id}) is an ADMIN with access to all installations."
    else:
        role_context = f"\n\nCurrent user '{username}' (ID: {user_id}) is a regular USER with access to only their own installations."
    
    system_message = SYSTEM_PROMPT + role_context
    
    llm_with_tools = llm.bind_tools(TOOL_REGISTRY) if TOOL_REGISTRY else llm
    response = llm_with_tools.invoke(
        [{"role": "system", "content": system_message}] + state["messages"]
    )
    return {"messages": [response]}
