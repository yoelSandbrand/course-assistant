"""FastAPI web server for the course assistant (stage 6).

Wraps the agent in a web interface. Start the MCP server first, then:

    .venv/Scripts/python.exe main.py

and open http://localhost:8000

Ports:  8001 MCP server (the tools)   8000 this web server

NOTE ON SESSIONS: there is deliberately no sessions dictionary here. The
conversation id goes straight through to the agent, whose SQLite database has
owned the history since stage 5. Keeping a second copy in this file would
create two sources of truth that can drift apart. The conversation id separates
CONVERSATIONS, not users - everything in data/ is shared by all of them.
"""

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# The agent module knows nothing about HTTP. We import two things from it: a
# way to build the knowledge base once, and a function that answers a question.
from agent import answer, build_knowledge

ROOT = Path(__file__).parent
DATA = ROOT / "data"
STATIC = ROOT / "static"


class Question(BaseModel):
    question: str
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the knowledge base once, at startup - never per request."""
    print("Building the syllabus knowledge base...")
    app.state.knowledge = build_knowledge()
    print("Ready on http://localhost:8000")
    yield


app = FastAPI(title="Course Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def home() -> FileResponse:
    """The chat page."""
    return FileResponse(STATIC / "index.html")


@app.post("/api/session")
async def new_session() -> dict:
    """Open a new conversation and hand back its id.

    Nothing is stored here. The id is just a label; the first time the agent is
    asked something under it, its SQLite database starts a history for it.
    """
    return {"session_id": f"web_{uuid.uuid4().hex[:12]}"}


@app.post("/api/ask")
async def ask(payload: Question) -> dict:
    """Put one question to the agent and return its answer."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="The question is empty.")
    try:
        text = await answer(
            question=payload.question,
            session_id=payload.session_id,
            knowledge=app.state.knowledge,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The agent could not answer ({type(exc).__name__}). "
                   f"Is the MCP server running on port 8001?",
        ) from exc
    return {"answer": text}


@app.get("/api/data")
async def data() -> dict:
    """Return the data files exactly as they are on disk.

    This is what the state panel reads. It deliberately does NOT go through the
    agent: reading the files directly is the only way to catch the agent saying
    "added!" when nothing was actually written.
    """
    def read(name: str) -> list:
        path = DATA / name
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    return {
        "courses": read("courses.json"),
        "assignments": read("assignments.json"),
        "notes": read("notes.json"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
