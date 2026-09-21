"""MCP server for the course assistant - "the hands" of the system.

Exposes seven tools over the Model Context Protocol so that ANY MCP client
(our own agent, Claude Desktop, or anything else) can read and write the
course data in data/*.json.

Run with:  .venv/Scripts/python.exe mcp_server.py
Listens on http://127.0.0.1:8001/mcp (streamable-http).
"""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Absolute paths - the server must work no matter which directory it is
# launched from. Claude Desktop (stage 2) starts it from somewhere else
# entirely, and a relative "data/courses.json" would fail there.
DATA_DIR = Path(__file__).parent / "data"
COURSES_FILE = DATA_DIR / "courses.json"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"
NOTES_FILE = DATA_DIR / "notes.json"

mcp = FastMCP("courses", host="127.0.0.1", port=8001)


# --------------------------------------------------------------------------
# helpers (not exposed as tools)
# --------------------------------------------------------------------------
def _read_json(path: Path) -> list[dict]:
    """Read a JSON list from disk, returning [] if the file does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: list[dict]) -> None:
    """Write a JSON list to disk, pretty-printed and UTF-8 safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# READ tools
# --------------------------------------------------------------------------
@mcp.tool()
def get_today() -> str:
    """Return today's date in YYYY-MM-DD format.

    USE THIS FIRST whenever a request involves a date, a deadline, or any
    relative time expression. You do NOT know today's date on your own and
    you must never guess it.

    Call this tool before you save any assignment whose due date was given
    without a year (for example "28.8", "next Tuesday", "in two weeks").
    Combine today's date with the user's text to work out the correct year:
    a date that has already passed this year almost always means next year.
    Guessing the year silently corrupts the data.
    """
    return date.today().isoformat()


@mcp.tool()
def get_courses() -> list[dict]:
    """Return every course the student is enrolled in.

    Each course has: name, lecturer, credits.

    Use this to answer "which courses am I taking?", to look up a lecturer or
    a credit count, and - importantly - to discover the EXACT spelling of a
    course name before calling any tool that takes a course name. The other
    tools match course names exactly, so always confirm the real name here
    rather than passing the user's paraphrase straight through.
    """
    return _read_json(COURSES_FILE)


@mcp.tool()
def get_assignments(course_name: str) -> list[dict]:
    """Return all assignments for one specific course, with their due dates.

    Each assignment has: course, title, due (YYYY-MM-DD), weight (percent of
    the final grade).

    Use this when the student asks what they have to submit in a particular
    course, or when they ask about the deadline or weight of a specific task.
    The course name must match an existing course exactly - call get_courses
    first if you are not certain of the spelling.
    """
    assignments = _read_json(ASSIGNMENTS_FILE)
    return [a for a in assignments if a.get("course") == course_name]


@mcp.tool()
def get_upcoming_deadlines(days: int) -> list[dict]:
    """Return assignments due within the next `days` days, across ALL courses.

    Results are sorted nearest-deadline-first. Assignments whose due date has
    already passed are excluded.

    Use this for "what is due soon?", "what do I have this week?" (days=7), or
    any question about workload in the near future. This is also the tool to
    use when deciding whether something is urgent enough to warrant alerting
    the student. Prefer this over reading every course one by one.
    """
    today = date.today()
    cutoff = today + timedelta(days=days)

    upcoming = []
    for a in _read_json(ASSIGNMENTS_FILE):
        try:
            due = datetime.strptime(a["due"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue  # skip malformed rows rather than crashing the tool
        if today <= due <= cutoff:
            upcoming.append({**a, "days_left": (due - today).days})

    return sorted(upcoming, key=lambda a: a["due"])


# --------------------------------------------------------------------------
# WRITE tools
# --------------------------------------------------------------------------
@mcp.tool()
def add_course(name: str, lecturer: str, credits: int) -> str:
    """Add a new course to the student's course list.

    Use this when the student says they are taking / registered for / want to
    add a course. Requires all three details: the course name, the lecturer's
    name, and the number of credits. If the student did not supply all three,
    ask them before calling this tool.

    A course name must be unique. If one with this name already exists the
    tool changes nothing and says so.

    Returns a human-readable confirmation or refusal message.
    """
    courses = _read_json(COURSES_FILE)

    # STAGE 4a. The validation lives HERE, inside the tool - not in the
    # agent's instructions. An instruction is a request the model may or may
    # not honour; this is a rule it cannot route around. Note that we return
    # a message instead of raising: that lets the agent explain the situation
    # to the student in plain language rather than crashing the run.
    for existing in courses:
        if existing.get("name") == name:
            return (
                f"Course '{name}' already exists (lecturer: "
                f"{existing.get('lecturer')}, {existing.get('credits')} credits). "
                f"Nothing was added."
            )

    courses.append({"name": name, "lecturer": lecturer, "credits": credits})
    _write_json(COURSES_FILE, courses)
    return f"Added course '{name}' (lecturer: {lecturer}, {credits} credits)."


@mcp.tool()
def add_assignment(course_name: str, title: str, due_date: str, weight: int) -> str:
    """Add an assignment to an existing course.

    Arguments:
        course_name: must match an existing course EXACTLY
        title:       what the assignment is called
        due_date:    the deadline in YYYY-MM-DD format
        weight:      percent of the final grade

    Before calling this tool: call get_today if the student gave a date
    without a year, and call get_courses to confirm the exact course name.

    The course must already exist, matched exactly. If it does not, the tool
    changes nothing and returns a message listing the courses that do exist.

    Returns a human-readable confirmation or refusal message.
    """
    # STAGE 4b. The course must already exist, matched EXACTLY. No partial
    # and no fuzzy matching: quietly guessing which course the student meant
    # is how an assignment ends up filed against the wrong one.
    known = [c.get("name", "") for c in _read_json(COURSES_FILE)]
    if course_name not in known:
        return (
            f"No course named '{course_name}' exists, so no assignment was added. "
            f"Existing courses: {', '.join(known)}. "
            f"Add the course first, or use one of those names exactly."
        )

    assignments = _read_json(ASSIGNMENTS_FILE)

    # Same class of bug as 4a, found in a third place. The brief does not ask
    # for this one, but stage 9 does: silencing a repeated alert PER ASSIGNMENT
    # needs a stable identifier for an assignment, and (course, title) can only
    # serve as one if the pair is unique.
    for existing in assignments:
        if existing.get("course") == course_name and existing.get("title") == title:
            return (
                f"Assignment '{title}' already exists for {course_name} "
                f"(due {existing.get('due')}, weight {existing.get('weight')}%). "
                f"Nothing was added."
            )

    assignments.append(
        {"course": course_name, "title": title, "due": due_date, "weight": weight}
    )
    _write_json(ASSIGNMENTS_FILE, assignments)
    return f"Added assignment '{title}' to {course_name}, due {due_date} (weight {weight}%)."


@mcp.tool()
def add_note(course_name: str, note: str) -> str:
    """Save a free-text note attached to a course.

    Use this for anything the student wants to remember that is not a formal
    assignment: "the lecturer said the exam is open book", "office hours moved
    to Tuesday", "focus on chapters 3-5". Notes are timestamped automatically.

    Returns a human-readable confirmation message.
    """
    notes = _read_json(NOTES_FILE)
    notes.append(
        {
            "course": course_name,
            "note": note,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_json(NOTES_FILE, notes)
    return f"Saved note for {course_name}."


if __name__ == "__main__":
    # The transport is chosen at launch instead of by editing this file.
    #
    #   (no argument)     streamable-http on port 8001. Our own agent connects
    #                     to this already-running server (stages 1, 3-9).
    #   stdio             Claude Desktop launches this script itself and speaks
    #                     over stdin/stdout (stage 2).
    #
    # Identical tools and identical logic either way - only the pipe changes.
    # That is the whole point of stage 2, and keeping both available means the
    # file never has to be edited back and forth between stages.
    transport = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"
    if transport not in ("stdio", "streamable-http"):
        raise SystemExit(
            f"unknown transport {transport!r} - use 'stdio' or 'streamable-http'"
        )
    mcp.run(transport=transport)
