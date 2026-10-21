"""Shared real run-provenance helpers - added to close 3 of the gaps
schema_report.md flagged as MISSING/PARTIAL (model version, run id, and a
consistent UTC-aware timestamp), so any generator can stamp its output
with "what code produced this, and when" without re-deriving it.

get_model_version() shells out to the real git short hash of HEAD - the
same real commit that's actually checked out when a generator runs, not
an asserted version string. new_run_id() is a short, real random id, not
a counter (no shared state file needed, safe for concurrent runs).
"""

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_cached_model_version = None


def get_model_version():
    """Real git short hash of the currently checked-out commit. Falls back
    to 'unknown' (never raises) if git isn't available or this isn't a
    real git checkout - a provenance tag that degrades gracefully is more
    useful than a crashing pipeline."""
    global _cached_model_version
    if _cached_model_version is not None:
        return _cached_model_version
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        _cached_model_version = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        _cached_model_version = "unknown"
    return _cached_model_version


def new_run_id():
    """Real, unique per-run id - 12 hex chars of a real uuid4, short enough
    to read in a log line, random enough that concurrent runs never collide."""
    return uuid.uuid4().hex[:12]


def utc_now_iso():
    """Real UTC-aware timestamp, matching generation_timestamps.py's
    existing convention (NOT refresh_weekly.py's naive datetime.now() -
    see schema_report.md's timezone-awareness finding)."""
    return datetime.now(timezone.utc).isoformat()


def parse_any_iso_to_utc(ts):
    """Real, defensive ISO-timestamp parser for this project's own mixed
    history: some real timestamps are UTC-aware (e.g. ESPN's own event
    date, 'Z'-suffixed), others are naive local wall-clock time (e.g.
    espn_odds_client's old fetched_at, refresh_weekly's step log). A naive
    string is assumed to be local system time (what it actually was when
    written) and converted to real UTC; an aware string is converted
    directly. Returns None on anything unparseable rather than raising -
    a single bad historical row shouldn't crash a real comparison across
    hundreds of rows."""
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone()  # real local system tz, then normalized below
    return dt.astimezone(timezone.utc)
