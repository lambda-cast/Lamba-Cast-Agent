"""Tools for querying the solar panel / weather PostgreSQL table.
...
"""

import json
import logging
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Optional
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from psycopg import sql
from pydantic import Field

from agent_app.db import get_cursor
from agent_app.state import GraphState

logger = logging.getLogger(__name__)

TABLE_NAME = os.getenv("SOLAR_TABLE_NAME", "solar_weather_data")

TUNIS_TZ = ZoneInfo("Africa/Tunis")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _convert_row_tz(row: dict) -> dict:
    row = dict(row)
    for key, value in row.items():
        if isinstance(value, datetime) and value.tzinfo is not None:
            row[key] = value.astimezone(TUNIS_TZ)
    return row


def _convert_rows_tz(rows: list[dict]) -> list[dict]:
    return [_convert_row_tz(row) for row in rows]


def _rows_to_json(rows: list[dict]) -> str:
    return json.dumps(_convert_rows_tz(rows), default=_json_default, indent=2)


def _require_installation_id(state: GraphState) -> Optional[int]:
    return state.get("installation_id")


# --------------------------------------------------------------------------
# tool: latest reading
# --------------------------------------------------------------------------

@tool("get_latest_reading")
def get_latest_reading(runtime: ToolRuntime) -> str:
    """Get the single most recent solar/weather reading row for the
    admin's currently selected installation.

    Use this for "what's happening right now" questions: current power
    output, current weather, current daily yield, etc.
    """
    state: GraphState = runtime.state
    installation_id = _require_installation_id(state)
    if installation_id is None:
        return "No installation selected yet."

    query = sql.SQL(
        "SELECT * FROM {table} WHERE installation_id = %(installation_id)s "
        "ORDER BY ts DESC LIMIT 1"
    ).format(table=sql.Identifier(TABLE_NAME))
    with get_cursor() as cur:
        cur.execute(query, {"installation_id": installation_id})
        row = cur.fetchone()
    if not row:
        return "No readings found for this installation."
    return _rows_to_json([row])


# --------------------------------------------------------------------------
# tool: time-range fetch (also useful as forecast tool input)
# --------------------------------------------------------------------------

@tool("get_readings_in_range")
def get_readings_in_range(
    start_ts: Annotated[str, Field(description="Start of range, ISO 8601, e.g. 2026-09-20T00:00:00Z")],
    end_ts: Annotated[str, Field(description="End of range, ISO 8601, e.g. 2026-09-21T00:00:00Z")],
    runtime: ToolRuntime,
    limit: Annotated[int, Field(description="Max rows to return, capped at 5000.")] = 500,
) -> str:
    """Fetch readings between two timestamps (inclusive) for the admin's
    currently selected installation, ordered oldest to newest.

    Use this to build a time series for analysis, charting, or as the
    feature/history window fed into the ARX / XGBoost forecast tool.
    """
    state: GraphState = runtime.state
    installation_id = _require_installation_id(state)
    if installation_id is None:
        return "No installation selected yet."

    limit = min(max(limit, 1), 5000)
    query = sql.SQL(
        "SELECT * FROM {table} "
        "WHERE installation_id = %(installation_id)s AND ts BETWEEN %(start_ts)s AND %(end_ts)s "
        "ORDER BY ts ASC LIMIT %(limit)s"
    ).format(table=sql.Identifier(TABLE_NAME))

    with get_cursor() as cur:
        cur.execute(
            query,
            {
                "installation_id": installation_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "limit": limit,
            },
        )
        rows = cur.fetchall()
    return _rows_to_json(rows)


# --------------------------------------------------------------------------
# tool: guarded read-only SQL (for aggregations, grouping, etc.)
# --------------------------------------------------------------------------

_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "truncate", "grant",
    "revoke", "create", "call", "copy", "vacuum", ";", "--",
)

_TABLE_NAME_RE = re.compile(r"\b" + re.escape(TABLE_NAME) + r"\b", re.IGNORECASE)


@tool("run_readonly_sql")
def run_readonly_sql(
    query: Annotated[
        str,
        Field(
            description=(
                "A single read-only SQL SELECT statement against the "
                f"'{TABLE_NAME}' table. Must start with SELECT. "
                "No INSERT/UPDATE/DELETE/DDL, no multiple statements. "
                "Do not add your own installation_id filter -- it is applied "
                "automatically for the currently selected installation."
            )
        ),
    ],
    runtime: ToolRuntime,
) -> str:
    """Run a read-only SQL SELECT against the solar/weather table, scoped
    to the admin's currently selected installation.

    Use this for aggregations the other tools don't cover, e.g. average
    power by hour, daily totals, correlation between solar_radiation_wm2
    and power_w. Only SELECT is allowed; the query is rejected if it
    contains write/DDL keywords, a statement separator, or doesn't
    reference the readings table. A LIMIT is appended automatically if
    missing.

    Note: this is basic guardrailing (keyword + prefix checks, plus a
    forced installation_id filter on every reference to the table), not a
    substitute for a read-only, row-level-secured DB role. Point
    POSTGRES_DSN at a read-only user in production for real defense in
    depth.
    """
    state: GraphState = runtime.state
    installation_id = _require_installation_id(state)
    if installation_id is None:
        return "No installation selected yet."

    normalized = query.strip().rstrip(";")
    lowered = normalized.lower()

    if not lowered.startswith("select"):
        return "Rejected: only SELECT statements are allowed."
    if any(kw in lowered for kw in _FORBIDDEN_KEYWORDS):
        return "Rejected: query contains a disallowed keyword or statement separator."
    if not _TABLE_NAME_RE.search(normalized):
        return f"Rejected: query must reference the '{TABLE_NAME}' table."
    if "limit" not in lowered:
        normalized += " LIMIT 1000"

    scoped_subquery = (
        f"(SELECT * FROM {TABLE_NAME} WHERE installation_id = {int(installation_id)}) "
        f"AS {TABLE_NAME}"
    )
    scoped_query = _TABLE_NAME_RE.sub(scoped_subquery, normalized)

    try:
        with get_cursor() as cur:
            cur.execute(scoped_query)
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_readonly_sql failed")
        return f"Query failed: {exc}"

    return _rows_to_json(rows)