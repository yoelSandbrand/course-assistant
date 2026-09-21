"""The autonomous agent (Part B) - the same system, a different trigger.

Everything so far has been reactive: the student asks, the agent acts. This is
proactive. At 08:00 every morning, with nobody asking, the agent checks which
deadlines are approaching and emails an alert if anything is urgent.

Run it with the MCP server already up:

    .venv/Scripts/python.exe scheduled_agent.py

Then open http://localhost:8002/docs and use "Try it out" to trigger the agent
by hand, BEFORE trusting any schedule.

Ports:  8001 MCP server   8000 web interface   8002 this service

NOTE: get_today matters more here than in Part A. An agent checking deadlines
without knowing today's date is meaningless, and unlike Part A there is no
human reading the answer who would notice the mistake.
"""

import json
import os
import smtplib
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.scheduler.manager import ScheduleManager
from agno.tools.mcp import MCPTools
from agno.tools.reasoning import ReasoningTools

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

MCP_URL = "http://127.0.0.1:8001/mcp"
PORT = 8002                      # clashes with neither 8001 (MCP) nor 8000 (web)
MODEL_ID = os.getenv("COURSE_AGENT_MODEL", "claude-sonnet-5")
SCHEDULE_DB = ROOT / "data" / "scheduler.db"
SHARED_DB = SqliteDb(db_file=str(SCHEDULE_DB))

AGENT_ID = "deadline-watcher"
SCHEDULE_NAME = "daily-deadline-alert"
ALERT_SESSION_ID = "deadline-watch"

# Daily at 08:00. Cron is five fields: minute hour day-of-month month day-of-week.
# Stage 9 has you temporarily change this to every minute ("* * * * *") so the
# repeat-alert bug is visible today instead of tomorrow. PUT IT BACK before
# submitting.
CRON_DAILY_0800 = "0 8 * * *"

# Stage 9a needs the agent running every minute so the repeat-alert bug shows up
# today rather than tomorrow morning. Rather than editing the line above and
# risking the brief's listed pitfall ("you forgot to put it back to daily before
# submitting"), the every-minute schedule is opt-in at launch:
#
#     set ALERT_CRON=* * * * *        only for the stage 9 experiment
#
# With the variable unset - which is how it ships - the schedule is daily 08:00.
CRON = os.getenv("ALERT_CRON", CRON_DAILY_0800)

# Without this the schedule fires at 08:00 UTC, which is not 08:00 here.
TIMEZONE = "Asia/Jerusalem"

ALERT_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# the email tool
# ---------------------------------------------------------------------------
ALERT_LOG = ROOT / "data" / "alert_log.json"
COOLDOWN_HOURS = 24


def _assignment_key(course: str, title: str) -> str:
    """The stable identifier for an assignment.

    STAGE 9c. A stock has a ticker; an assignment here does not. The due date
    is no good - it is exactly what stays the same between runs, and it can be
    edited. The title alone collides across courses. Course + title is the
    identifier, and it is only usable as one because add_assignment refuses a
    repeated course+title pair, which is the check added back in stage 4.
    """
    return f"{course.strip()}||{title.strip()}"


def _load_alert_log() -> dict:
    if not ALERT_LOG.exists():
        return {}
    try:
        return json.loads(ALERT_LOG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_alert_log(log: dict) -> None:
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    ALERT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def _send(subject: str, body: str) -> str:
    """Actually put the mail on the wire. No policy here, just SMTP."""
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to = os.getenv("ALERT_TO", user)

    if not user or not password:
        return ("Email is not configured: SMTP_USER and SMTP_PASSWORD are missing "
                "from .env, so no email was sent.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = user
    message["To"] = to
    message.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
    except smtplib.SMTPAuthenticationError:
        return ("Email rejected the login. For Gmail you need an app password, "
                "not your normal account password.")
    except Exception as exc:
        return f"Email failed ({type(exc).__name__}): {exc}"
    return f"sent to {to}"


def send_deadline_alert(assignments: list[dict]) -> str:
    """Email the student about urgent assignments.

    Pass EVERY assignment you judge to be urgent. Each one must be a dictionary
    with the keys: course, title, due, days_left, weight. Take these values
    straight from get_upcoming_deadlines - do not retype or reword them.

    You do not need to remember what was sent before, and you must not try to.
    This tool keeps its own record and will silently skip any assignment the
    student was already warned about recently. Simply report everything urgent
    every time; the tool decides what actually goes out.

    Returns a summary of what was sent and what was suppressed.
    """
    # STAGE 9c - THE REAL FIX.
    #
    # Deciding WHAT is urgent is the model's job and it does it well. Deciding
    # WHETHER an alert may be sent is not a judgement at all, so it does not
    # belong to the model. It is a lookup against a record on disk.
    #
    # Per assignment, not global: a new urgent assignment must still get through
    # while an old one stays silenced. A single global "already alerted" flag
    # would suppress the new one too.
    #
    # On disk, not in the conversation: the stage 9b version behaved correctly
    # for 39 runs and then re-sent everything the moment the history was
    # cleared, because its guarantee lived in the chat. This one survives a
    # restart, a redeploy and a new session, because the record is a file.
    if not assignments:
        return "No assignments were passed, so nothing was sent."

    log = _load_alert_log()
    now = datetime.now()
    cutoff = timedelta(hours=COOLDOWN_HOURS)

    due_to_send, suppressed = [], []
    for a in assignments:
        course = str(a.get("course", "")).strip()
        title = str(a.get("title", "")).strip()
        if not course or not title:
            continue
        key = _assignment_key(course, title)
        last = log.get(key)
        if last:
            try:
                if now - datetime.fromisoformat(last) < cutoff:
                    suppressed.append(f"{course} / {title}")
                    continue
            except ValueError:
                pass
        due_to_send.append((key, course, title, a))

    if not due_to_send:
        return (f"Nothing sent. All {len(suppressed)} urgent assignment(s) were "
                f"already alerted within the last {COOLDOWN_HOURS} hours: "
                + "; ".join(suppressed))

    lines = ["You have urgent assignment deadlines coming up:", ""]
    for _, course, title, a in due_to_send:
        lines.append(f"- {course}: {title}")
        lines.append(f"    due {a.get('due')}  ({a.get('days_left')} days left, "
                     f"weight {a.get('weight')}%)")
    body = chr(10).join(lines)
    subject = (f"Urgent: {len(due_to_send)} assignment"
               f"{'s' if len(due_to_send) > 1 else ''} due soon")

    result = _send(subject, body)
    if not result.startswith("sent to"):
        return f"Nothing was recorded because the email failed: {result}"

    # Only record AFTER the send succeeded - otherwise a failed send would
    # silence the assignment for a day and the student would never hear about it.
    for key, _, _, _ in due_to_send:
        log[key] = now.isoformat(timespec="seconds")
    _save_alert_log(log)

    report = f"Emailed {len(due_to_send)} assignment(s): " + "; ".join(
        f"{c} / {t}" for _, c, t, _ in due_to_send)
    if suppressed:
        report += (f". Suppressed {len(suppressed)} already alerted within "
                   f"{COOLDOWN_HOURS}h: " + "; ".join(suppressed))
    return report


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------
DESCRIPTION = """\
You watch a university student's assignment deadlines and warn them by email
when something is about to be due. You run on a schedule, unprompted. Nobody is
reading your output, so the email is the only thing that matters."""

INSTRUCTIONS = [
    "You do not know today's date. ALWAYS call get_today first. Every judgement "
    "you make about urgency depends on it, and you cannot work it out any other way.",
    f"Call get_upcoming_deadlines with days={ALERT_WINDOW_DAYS} to find assignments "
    f"due soon.",
    "Decide what is urgent: an assignment due within the next "
    f"{ALERT_WINDOW_DAYS} days is urgent. Anything further out is not.",
    "If anything is urgent, call send_deadline_alert ONCE, passing every urgent "
    "assignment in the list, with the values exactly as get_upcoming_deadlines "
    "returned them.",
    # STAGE 9c. The 9b instruction that used to live here - "check history, do not
    # send about the same assignment twice" - has been deleted on purpose. That job
    # now belongs to the tool, which checks a record on disk. Leaving the
    # instruction in would imply the model is still responsible for a guarantee it
    # cannot actually make.
    "Do not try to remember or work out what was already sent. Report everything "
    "urgent every time. The tool decides what actually goes out.",
    "If NOTHING is urgent, call nothing and simply report that there was nothing to send.",
    "Never invent an assignment or a date. Everything you report must come from the tools.",
    # STAGE 9d. Having ReasoningTools available is not enough - the agent has to
    # be told to use them, or it goes straight to the answer and there is no
    # chain of thought to look at.
    "Use the think tool before deciding. Work through it one assignment at a "
    "time: how many days remain, whether that clears the urgency threshold, and "
    "therefore whether it belongs in the alert. Then use analyze to check your "
    "conclusion before you act on it.",
]


def build_scheduled_agent(mcp_tools: MCPTools) -> Agent:
    """The deadline watcher. Tools come from the MCP server, not a local file."""
    return Agent(
        id=AGENT_ID,
        name="Deadline Watcher",
        model=Claude(id=MODEL_ID),
        tools=[mcp_tools, send_deadline_alert, ReasoningTools(add_instructions=True)],
        db=SHARED_DB,
        description=DESCRIPTION,
        instructions=INSTRUCTIONS,
        # STAGE 9b needed 30 runs of history for its prompt-based guard, and
        # that history was resent on every single call - input tokens grew from
        # 7,409 to 24,214 in half an hour. With the cooldown living on disk the
        # agent no longer needs to remember anything between runs, so the
        # history goes back to a small window and the cost stops growing.
        add_history_to_context=True,
        num_history_runs=3,
        # STAGE 9d. The brief writes this as reasoning=True, which agno 3.0.5
        # does not accept. Two further attempts also failed: reasoning_model
        # alone is rejected ("Claude is not a native reasoning model"), and
        # enabling extended thinking gets as far as the API but agno then
        # reports "No reasoning content" because it does not read adaptive
        # thinking blocks. ReasoningTools is agno's own suggested route and is
        # model-agnostic: the agent thinks in explicit, visible steps.
        # Displaying them is a separate flag on print_response, below.
        markdown=False,
    )


def register_schedule(db: SqliteDb, enabled: bool = True) -> None:
    """Register the 08:00 run, updating it if one with this name already exists.

    THE TRAP the brief warns about: this runs on every launch of the script.
    Creating the schedule unconditionally would add ANOTHER identical job each
    time, and you would get one email per duplicate for the same assignment.

    `if_exists="update"` is exactly the parameter that prevents it. Going
    through ScheduleManager rather than writing to the database by hand also
    means `next_run_at` gets computed from the cron expression - written
    directly, that column stays empty and the poller never considers the job
    due, so it silently never fires.
    """
    ScheduleManager(db).create(
        name=SCHEDULE_NAME,
        cron=CRON,
        endpoint=f"/agents/{AGENT_ID}/runs",
        method="POST",
        description="Checks for approaching assignment deadlines and alerts by email.",
        payload={
            "message": "Check which assignments are due soon and alert me if any are urgent.",
            # Without this every scheduled run starts a fresh session and the
            # agent's "history" is always empty.
            "session_id": ALERT_SESSION_ID,
        },
        timezone=TIMEZONE,
        if_exists="update",
    )
    print(f"Schedule '{SCHEDULE_NAME}' registered (cron: {CRON}, tz: {TIMEZONE}).")


async def run_once_with_reasoning() -> None:
    """STAGE 9d: one run, with the full chain of thought printed.

    Run it with three urgent assignments outstanding:

        .venv/Scripts/python.exe scheduled_agent.py reasoning
    """
    import mcp_server  # only to show what the agent is about to reason over

    print("Urgent assignments the agent will reason about:")
    for a in mcp_server.get_upcoming_deadlines(ALERT_WINDOW_DAYS):
        print(f"   {a['course']} / {a['title']}  due {a['due']}  ({a['days_left']}d)")
    print()

    async with MCPTools(url=MCP_URL, transport="streamable-http") as tools:
        agent = build_scheduled_agent(tools)
        await agent.aprint_response(
            "Check which assignments are due soon and alert me if any are urgent.",
            # A fresh session: with the usual history the agent replies "same
            # as before" and never reasons at all.
            session_id=f"reasoning-demo-{int(datetime.now().timestamp())}",
            show_full_reasoning=True,
            stream=True,
        )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    db = SHARED_DB

    # MCPTools must be opened in the SAME event loop that later serves requests.
    # Opening it with asyncio.run() here would connect inside a throwaway loop
    # that closes immediately, leaving the agent holding a session bound to a
    # dead loop - every tool call then fails at runtime while the agent itself
    # looks perfectly healthy. So the object is built now and connected inside
    # the server's lifespan, which runs on the serving loop.
    mcp_tools = MCPTools(url=MCP_URL, transport="streamable-http")
    agent = build_scheduled_agent(mcp_tools)

    @asynccontextmanager
    async def lifespan(app):
        await mcp_tools.__aenter__()
        print(f"Connected to MCP server at {MCP_URL}")
        try:
            yield
        finally:
            await mcp_tools.__aexit__(None, None, None)

    register_schedule(db)

    agent_os = AgentOS(
        name="Course Deadline Service",
        description="Watches assignment deadlines and sends email alerts.",
        agents=[agent],
        db=db,
        lifespan=lifespan,
        # Registering a schedule row is NOT enough: without scheduler=True
        # nothing ever polls the table, and the job silently never runs.
        scheduler=True,
        scheduler_poll_interval=10,
        # Defaults to port 7777. It has to point at THIS server, or the
        # scheduler fires into nothing.
        scheduler_base_url=f"http://127.0.0.1:{PORT}",
    )

    print("")
    print(f"AgentOS starting on http://localhost:{PORT}")
    print(f"Swagger UI:  http://localhost:{PORT}/docs")
    print(f"Trigger by hand at:  POST /agents/{AGENT_ID}/runs")
    print("")
    agent_os.serve(app=agent_os.get_app(), host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    if "reasoning" in sys.argv[1:]:
        import asyncio
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass
        asyncio.run(run_once_with_reasoning())
    else:
        main()
