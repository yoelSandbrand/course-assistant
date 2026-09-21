"""The course assistant agent - "the brain" that drives the MCP tools.

Connects to an MCP server that is ALREADY RUNNING. It does not launch one.
Start the server first, in its own window:

    .venv/Scripts/python.exe mcp_server.py

Then:

    .venv/Scripts/python.exe agent.py            stage 3/4  five test requests
    .venv/Scripts/python.exe agent.py rag        stage 5a   three document questions
    .venv/Scripts/python.exe agent.py memory     stage 5b   three session tests

WHAT COMES FROM WHERE
    Tools     dynamic data that changes: the course list, assignments, deadlines,
              today's date. Anything the student adds or edits.
    Documents static data that does not change: grading weights, late policy,
              prerequisites, required reading, attendance rules. The syllabus.

Mixing the two is the mistake. A due date in a syllabus goes stale the moment
the lecturer moves it; a grading weight in a JSON file is data nobody maintains.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools
from agno.vectordb.lancedb import LanceDb

# Absolute paths throughout, same reasoning as the MCP server: this has to work
# regardless of which directory the script was launched from.
ROOT = Path(__file__).parent
DATA = ROOT / "data"
LANCE_URI = DATA / "lancedb"
MEMORY_DB = DATA / "agent_memory.db"

load_dotenv(ROOT / ".env")

MCP_URL = "http://127.0.0.1:8001/mcp"
MODEL_ID = os.getenv("COURSE_AGENT_MODEL", "claude-sonnet-5")

# A multilingual embedder, because the student writes in Hebrew while the
# syllabus documents are in English. It runs locally on the CPU and costs
# nothing - no embedding API key is required.
EMBEDDER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Named for what it holds. Not "documents", not the tutorial's table name.
TABLE_NAME = "syllabi"

DESCRIPTION = """\
You are a course assistant for a university student. You manage their courses,
assignments and deadlines entirely through conversation, and you can answer
questions about course syllabi. The student never edits a file by hand."""

INSTRUCTIONS = [
    "Always use the tools to read or change course data. Never answer about "
    "courses, assignments or deadlines from memory, and never invent a course, "
    "a lecturer, a due date or a weight.",
    "You do not know today's date. Whenever a request involves a date, a deadline "
    "or a relative expression, call get_today first and work the real date out from it.",
    "Course names must be passed to tools exactly as stored. When the student names "
    "a course, call get_courses to find its exact spelling before using it as an argument.",
    # The restriction the brief asks for. Without this the model happily invents a
    # plausible late policy for a course it has never seen a syllabus for.
    "Questions about grading weights, late submission policy, prerequisites, required "
    "reading, attendance rules or academic integrity must be answered ONLY from the "
    "syllabus documents in your knowledge base. Search the knowledge base for them.",
    "If the syllabus documents do not contain the answer, say plainly that you do not "
    "have a syllabus for that course and therefore do not know. Do NOT guess, do NOT "
    "reason by analogy from another course, and do NOT state a general university policy "
    "as if it were that course's policy.",
    "When a tool returns a message saying it could not do something, explain that to the "
    "student in plain language. Do not retry the same call.",
    "Answer in the same language the student wrote in.",
]

# Stage 3/4 - the five test requests, exactly as the brief specifies them.
TEST_REQUESTS = [
    "אילו קורסים אני לומד?",
    'הוסף את הקורס אתיקה, מרצה ד"ר דוד שטטר, 2 נק"ז',
    "לקורס אתיקה הוסף מטלה 'תרגיל בית 1' לתאריך 28.8, משקל 10",
    "הוסף את הקורס אתיקה שוב",
    "לקורס מערכות הפעלה הוסף מטלה 'תרגיל 1' לתאריך 1.9, משקל 20",
]

# Stage 5a - three questions with three different correct behaviours.
RAG_QUESTIONS = [
    ("מה משקל הפרוייקט בציון בקורס אתיקה?", "SYLLABUS", "should answer 50%"),
    ("מה עלי להגיש בקורס אתיקה?", "TOOLS", "should list assignments from the tools"),
    ("מה מדיניות האיחורים בקורס בינה מלאכותית מתקדמת?", "NOWHERE",
     "no syllabus exists - it MUST admit it does not know"),
]

# Stage 5b - same question twice in one session, once in another.
MEMORY_TESTS = [
    ("מה המטלות בקורס אתיקה?", "chat_1", "opens the conversation"),
    ("ומתי צריך להגיש את הראשונה?", "chat_1", "must understand 'the first' from context"),
    ("ומתי צריך להגיש את הראשונה?", "chat_2", "new session - must NOT understand it"),
]


def _utf8() -> None:
    """Windows consoles default to a codepage that cannot print Hebrew."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def build_knowledge() -> Knowledge:
    """The syllabus knowledge base, loaded from disk exactly once.

    The PDFs are embedded into LanceDB on first run only. On every later run the
    table already exists and we skip straight past it - re-embedding on each
    launch would be slow and, with a paid embedder, expensive.
    """
    vector_db = LanceDb(
        uri=str(LANCE_URI),
        table_name=TABLE_NAME,
        embedder=FastEmbedEmbedder(id=EMBEDDER_ID, dimensions=384),
    )
    knowledge = Knowledge(name="Course syllabi", vector_db=vector_db, max_results=5)

    # exists() is NOT enough: constructing the LanceDb creates the table, so it
    # reports True even when the table is empty and nothing has been embedded.
    # Count the rows instead - that is the question actually being asked.
    already = vector_db.get_count() if vector_db.exists() else 0
    if already:
        print(f"Knowledge: table '{TABLE_NAME}' already holds {already} chunks, "
              f"skipping load.")
        return knowledge

    syllabi = sorted(DATA.glob("syllabus_*.pdf"))
    print(f"Knowledge: first run, embedding {len(syllabi)} syllabus PDFs "
          f"(downloads the model once, please wait)...")
    # A whole syllabus in one chunk is useless: every query then retrieves both
    # documents in full, and nothing is actually being *retrieved*. The default
    # chunk size is 5000 characters and these syllabi are ~1500, so they would
    # never split. 600 with overlap splits them roughly per section, which is
    # the granularity the questions are asked at.
    reader = PDFReader(chunking_strategy=RecursiveChunking(chunk_size=600, overlap=80))
    for pdf in syllabi:
        knowledge.insert(name=pdf.stem, path=str(pdf), reader=reader, skip_if_exists=True)
        print(f"   loaded {pdf.name}")
    print(f"Knowledge: {vector_db.get_count()} chunks embedded.")
    return knowledge


def build_agent(mcp_tools: MCPTools, knowledge: Knowledge | None = None) -> Agent:
    """Create the course assistant. Everything it needs is passed in."""
    return Agent(
        model=Claude(id=MODEL_ID),
        tools=[mcp_tools],
        knowledge=knowledge,
        search_knowledge=knowledge is not None,
        db=SqliteDb(db_file=str(MEMORY_DB)),
        add_history_to_context=True,
        num_history_runs=5,
        description=DESCRIPTION,
        instructions=INSTRUCTIONS,
        markdown=True,
    )


async def answer(question: str, session_id: str,
                 knowledge: Knowledge | None = None) -> str:
    """Answer one question. This is the entry point for callers outside this file.

    STAGE 6a. The signature is only these three things because Agno has owned
    the conversation history since stage 5: the caller passes a session id and
    the agent's SQLite database does the remembering. A caller that also kept
    its own history would create a second source of truth, and the two would
    drift apart.

    This function knows nothing about HTTP, FastAPI or browsers, and it must
    not learn. It is called from main.py, and it would work identically from a
    command line, a test, or a cron job.

    The MCP connection is opened per call rather than held open for the life of
    the process, so that restarting the MCP server does not leave this side
    holding a dead socket.
    """
    async with MCPTools(url=MCP_URL, transport="streamable-http") as tools:
        agent = build_agent(tools, knowledge)
        response = await agent.arun(question, session_id=session_id)
        return response.content


def _tokens(response) -> str:
    m = getattr(response, "metrics", None)
    if m is None:
        return "tokens: n/a"
    used = {}
    for attr in ("input_tokens", "output_tokens", "total_tokens"):
        v = getattr(m, attr, None)
        if isinstance(v, (int, float)) and v:
            used[attr] = int(v)
    return ", ".join(f"{k}={v}" for k, v in used.items()) if used else "tokens: n/a"


async def ask(agent: Agent, label: str, question: str, note: str = "",
              session_id: str | None = None) -> None:
    """Run one question and report tools, retrieved documents, and the answer."""
    print("\n" + "=" * 72)
    print(f"{label}: {question}")
    if note:
        print(f"expected: {note}")
    if session_id:
        print(f"session_id: {session_id}")
    print("=" * 72)

    response = (await agent.arun(question, session_id=session_id)
                if session_id else await agent.arun(question))

    calls = getattr(response, "tools", None) or []
    print(f"\nTOOLS CALLED ({len(calls)}), in order:")
    if not calls:
        print("  (none)")
    for i, c in enumerate(calls, 1):
        name = getattr(c, "tool_name", "?")
        args = getattr(c, "tool_args", None) or {}
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  {i}. {name}({shown})")
        # The brief asks you to find the number of documents the search returned.
        if "knowledge" in name.lower() or "search" in name.lower():
            result = getattr(c, "result", None)
            if isinstance(result, str):
                print(f"     -> retrieved {len(result)} chars of syllabus text "
                      f"(agno logs the chunk count above as 'Found N documents')")

    print(f"\nANSWER:\n{response.content}")
    print(f"\n[{_tokens(response)}]")


async def run_requests() -> None:
    """Stage 3 and 4: the five test requests."""
    async with MCPTools(url=MCP_URL, transport="streamable-http") as tools:
        agent = build_agent(tools)
        for i, request in enumerate(TEST_REQUESTS, 1):
            await ask(agent, f"REQUEST {i}", request)
    print("\n" + "=" * 72)
    print("Five requests finished. Now OPEN data/courses.json and")
    print("data/assignments.json. Do not trust what the agent said it did.")
    print("=" * 72)


async def run_rag() -> None:
    """Stage 5a: three questions, three different correct behaviours."""
    knowledge = build_knowledge()
    async with MCPTools(url=MCP_URL, transport="streamable-http") as tools:
        agent = build_agent(tools, knowledge)
        for i, (q, source, note) in enumerate(RAG_QUESTIONS, 1):
            await ask(agent, f"QUESTION {i}  [answer source: {source}]", q, note)
    print("\n" + "=" * 72)
    print("Question 3 is the real test. If the agent invented a late policy")
    print("for a course with no syllabus, the instruction restriction failed.")
    print("=" * 72)


async def run_memory() -> None:
    """Stage 5b: two questions in one session, the same question in another."""
    knowledge = build_knowledge()
    async with MCPTools(url=MCP_URL, transport="streamable-http") as tools:
        agent = build_agent(tools, knowledge)
        for i, (q, session, note) in enumerate(MEMORY_TESTS, 1):
            await ask(agent, f"TEST {i}", q, note, session_id=session)
    print("\n" + "=" * 72)
    print("Test 2 should understand 'the first' from test 1.")
    print("Test 3 asks the identical question in a different session and")
    print("should NOT understand it - it has no history to lean on.")
    print("=" * 72)


MODES = {"requests": run_requests, "rag": run_rag, "memory": run_memory}


async def main() -> None:
    _utf8()
    mode = sys.argv[1] if len(sys.argv) > 1 else "requests"
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; use one of: {', '.join(MODES)}")

    print(f"Connecting to MCP server at {MCP_URL}")
    print(f"Model: {MODEL_ID}   |   mode: {mode}")

    try:
        await MODES[mode]()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print("\nIs the MCP server running? In its own window:\n"
              "    .venv/Scripts/python.exe mcp_server.py\n"
              "and wait for 'Uvicorn running on http://127.0.0.1:8001'.")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
