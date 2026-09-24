"""    uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import logging
import uuid
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from agent_app.graph import build_graph
from agent_app.state import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

checkpointer = MemorySaver()
graph = build_graph(checkpointer=checkpointer)


def get_session(
    x_user_id: Optional[str] = Header(default=None)
) -> Session:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id header.")
    try:
        user_id = int(x_user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="X-User-Id must be an integer.")
    return {"user_id": user_id}


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(title="Solar Admin Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="The admin's message.")
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Conversation id. Omit to start a new conversation -- the "
            "response returns an id to reuse for every follow-up turn "
            "(e.g. when replying to the 'which installation?' question)."
        ),
    )


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    installations: Optional[list] = None
    installation_id: Optional[int] = None
    authenticated: Optional[bool] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session: Session = Depends(get_session)):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "session": session,
            },
            config=config,
        )
    except Exception:
        logger.exception("graph.invoke failed for thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail="Agent failed to process the message.")

    reply = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break

    return ChatResponse(
        thread_id=thread_id,
        reply=reply,
        installations=result.get("installations"),
        installation_id=result.get("installation_id"),
        authenticated=result.get("authenticated"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)