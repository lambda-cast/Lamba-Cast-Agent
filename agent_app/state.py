from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph.message import AnyMessage, add_messages


class Session(TypedDict):
    user_id: int
    

class Installation(TypedDict):
    id: int
    name: str


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: bool
    session: Optional[Session]
    authenticated: Optional[bool]

    installations: Optional[list[Installation]]   # every installation owned by this admin
    installation_id: Optional[int]                 # the one picked; every DB/forecast tool is scoped to it
    user_choice: Optional[dict]                    # stores user's choice from human-in-the-loop interrupts