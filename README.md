# Course Assistant

A course and assignment manager driven entirely by natural language. There are
no forms and no editing files by hand: you write *"add the course Secure
Development"* or *"add an assignment to Secure Development due 28.8"* and the
system does it.

Two halves, the same code underneath:

- **Part A, interactive.** A chat page in the browser. You ask, the agent acts.
- **Part B, autonomous.** The same agent running on a schedule, checking which
  deadlines are approaching and emailing an alert. Nobody asks it anything.

---

## Architecture

```
Browser  ──HTTP──▶  main.py (FastAPI :8000)
                        │
                        │ calls answer(question, session_id)
                        ▼
                    agent.py  ────────────▶  LanceDB          (syllabus documents)
                    (Claude)  ────────────▶  SQLite           (conversation history)
                        │
                        │ MCP over streamable-http
                        ▼
                 mcp_server.py (:8001)
                        │
                        ▼
                    data/*.json            (courses, assignments, notes)
```

The state panel in the browser reads `data/*.json` through `main.py` **directly**,
never through the agent. That is deliberate: it is the only way to notice the
agent claiming it saved something when it did not.

---

## Requirements

- **Python 3.12.** Not 3.13 or 3.14 — `lancedb` and `pyarrow` have no wheels for
  those, and the failure only appears at the document-search stage.
- An **Anthropic API key** with credit on the account.

---

## Setup

```bash
git clone <this-repo>
cd final_pro

py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=not-used
```

`OPENAI_API_KEY` is listed for completeness but is **not used**. Document
embedding runs locally through FastEmbed, so no embedding API key is needed.

Check the install:

```bash
python -c "import mcp, agno, fastapi; print('OK')"
```

---

## Running it

Three servers on three ports. **Start them in this order**, each in its own
terminal window, and leave each one running — a server that has started prints
a line and then sits silently, which is correct.

### 1. The MCP server — port 8001, always required

```bash
.venv\Scripts\python.exe mcp_server.py
```

Wait for `Uvicorn running on http://127.0.0.1:8001`. Everything else talks to
this. Nothing works without it.

### 2. The web interface — port 8000

```bash
.venv\Scripts\python.exe main.py
```

Then open **http://localhost:8000**. First start takes about 20 seconds because
it downloads the embedding model once.

### 3. The autonomous agent — port 8002

```bash
.venv\Scripts\python.exe scheduled_agent.py
```

Swagger UI at **http://localhost:8002/docs** to trigger it by hand.

| Port | Process | Needed for |
|---|---|---|
| 8001 | `mcp_server.py` | everything |
| 8000 | `main.py` | the browser interface |
| 8002 | `scheduled_agent.py` | the scheduled email alerts |

---

## The command line agent

`agent.py` runs against the MCP server without the browser. Three modes:

```bash
.venv\Scripts\python.exe agent.py            # five test requests
.venv\Scripts\python.exe agent.py rag        # three syllabus questions
.venv\Scripts\python.exe agent.py memory     # three conversation-memory tests
```

Each prints which tools the agent decided to call, in order, plus token counts.

Set `COURSE_AGENT_MODEL` to run against a cheaper model while developing:

```bash
set COURSE_AGENT_MODEL=claude-haiku-4-5-20251001
```

---

## The tools

Seven tools, exposed over MCP. Four read, three write.

| Tool | Type | Validation |
|---|---|---|
| `get_today()` | read | — |
| `get_courses()` | read | — |
| `get_assignments(course_name)` | read | — |
| `get_upcoming_deadlines(days)` | read | skips past and malformed dates |
| `add_course(name, lecturer, credits)` | write | rejects a duplicate course name |
| `add_assignment(course_name, title, due_date, weight)` | write | course must exist, matched exactly; rejects a repeated course + title |
| `add_note(course_name, note)` | write | — |

Every rejection returns a **message**, never an exception, so the agent can
explain what happened instead of crashing.

To inspect what the server advertises, start it and run in a second terminal:

```bash
.venv\Scripts\python.exe discover_tools.py
```

It prints all seven with descriptions and JSON schemas.

---

## Using it from Claude Desktop

The MCP server is a standard server, so any MCP client can drive it. For Claude
Desktop, add this to `claude_desktop_config.json` and restart the app completely:

```json
{
  "mcpServers": {
    "courses": {
      "command": "C:\\path\\to\\final_pro\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\final_pro\\mcp_server.py", "stdio"]
    }
  }
}
```

The `stdio` argument is the only difference. Claude Desktop launches the server
itself and speaks over stdin/stdout rather than connecting to a port, so it
cannot use the HTTP transport. With no argument the server defaults to
streamable-http on 8001.

---

## Where information comes from

| Kind | Source | Examples |
|---|---|---|
| Changes over time | **tools** | course list, assignments, deadlines, today's date |
| Fixed for the semester | **documents** | grading weights, late policy, prerequisites, reading |

Syllabus PDFs live in `data/` as `syllabus_*.pdf` and are embedded into LanceDB
on first run only. Delete `data/lancedb/` to force a rebuild.

---

## Project layout

```
final_pro/
├── data/
│   ├── courses.json          course list
│   ├── assignments.json      assignments and due dates
│   ├── notes.json            created by add_note on first use
│   ├── syllabus_*.pdf        syllabus documents
│   ├── lancedb/              vector index (gitignored, rebuilt on demand)
│   └── agent_memory.db       conversation history (gitignored)
├── mcp_server.py             the seven tools
├── discover_tools.py         prints what the server advertises
├── agent.py                  the agent; knows nothing about HTTP
├── main.py                   FastAPI web server
├── static/index.html         chat + read-only state panel
├── scheduled_agent.py        autonomous agent, runs on a schedule
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Troubleshooting

**`All connection attempts failed`, or a long traceback ending in
`CancelledError`** — the MCP server is not running. Start it and wait for the
Uvicorn line. This is by far the most common failure.

**`address already in use`** — that port already has a server on it. Either use
the one that is running, or stop it first.

**`No module named mcp`** — you ran `python` instead of the virtual
environment's interpreter. Use `.venv\Scripts\python.exe`.

**`credit balance is too low`** — the Anthropic account has no credit. The key
is fine; the balance is not.

**The agent answers but nothing changes in the files** — look at the state
panel, not the chat. If the panel does not change, the write did not happen.
