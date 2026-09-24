import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal, Optional
import numpy as np
import pandas as pd

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from psycopg import sql
from pydantic import BaseModel, Field

from agent_app.db import get_cursor
from agent_app.state import GraphState
from models import (
    EXOG_COLUMNS_DB,
    MODEL_KEYS,
    TARGET_COLUMN,
    get_model,
    model_family,
    model_label,
    model_needs_exog,
)
from agent_app.tools.postgres_tools import TABLE_NAME

logger = logging.getLogger(__name__)

ModelKey = Literal[
    "ar_filtered",
    "ar_unfiltered",
    "arx_filtered",
    "arx_unfiltered",
    "armax_filtered",
    "armax_unfiltered",
    "sarimax_unfiltered",
]


def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _fetch_recent_history(installation_id: int, n_rows: int, need_exog: bool):
    cols = ["ts", TARGET_COLUMN] + (EXOG_COLUMNS_DB if need_exog else [])
    query = sql.SQL(
        "SELECT {cols} FROM {table} WHERE installation_id = %(installation_id)s "
        "ORDER BY ts DESC LIMIT %(n)s"
    ).format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols),
        table=sql.Identifier(TABLE_NAME),
    )
    with get_cursor() as cur:
        cur.execute(query, {"n": n_rows, "installation_id": installation_id})
        rows = cur.fetchall()
    rows.reverse()  # oldest -> newest, as the models expect
    return rows


def _lookback_and_min_required(fitted, family: str) -> tuple[int, int]:
    """How many rows to pull from the DB, and the minimum needed to seed
    the model, based on its statsmodels family.

    - 'autoreg' (AR/ARX, statsmodels AutoReg): lag info is on
      `.model.ar_lags`.
    - 'sarimax' (ARMAX/SARIMAX, statsmodels statespace SARIMAX): lag info
      is on `.model.order` (p, d, q) and `.model.seasonal_order`
      (P, D, Q, s).
    """
    if family == "autoreg":
        ar_lags = fitted.model.ar_lags
        max_lag = max(ar_lags) if ar_lags else 1
        lookback = max(200, 3 * max_lag)
        min_required = max_lag + 5
    else:  # "sarimax" -- covers both ARMAX and SARIMAX
        p, d, q = fitted.model.order
        P, D, Q, s = getattr(fitted.model, "seasonal_order", (0, 0, 0, 0))
        # rough worst-case span the model needs to see before it can
        # produce a differenced/seasonal state: non-seasonal order + d,
        # plus whatever the seasonal terms reach back across.
        max_lag = max(p + d, 1) + (P + D) * s
        lookback = max(200, 5 * max_lag)
        min_required = max_lag + 10
    return lookback, min_required


class ForecastArgs(BaseModel):
    model: ModelKey = Field(
        description=(
            "Which fitted model to forecast with:\n"
            "- 'ar_filtered' / 'ar_unfiltered': history-only (AutoReg), fastest, least accurate.\n"
            "- 'arx_filtered' / 'arx_unfiltered': history + weather (AutoReg w/ exog), good accuracy.\n"
            "- 'armax_filtered' / 'armax_unfiltered': history + weather (SARIMAX, no differencing), "
            "generally more accurate than ARX.\n"
            "- 'sarimax_unfiltered': history + weather + daily seasonality (SARIMAX with seasonal "
            "order), most expressive model; no filtered variant exists yet.\n"
            "'filtered' variants were fit on cleaned data and are recommended by default; "
            "'unfiltered' variants were fit on raw data. All exog-based models assume weather "
            "stays constant at last observed values for the forecast horizon."
        )
    )
    horizon_steps: int = Field(
        default=6,
        description=(
            "How many future steps to forecast, capped at 48. A 'step' is "
            "one future reading at the data's natural cadence, not "
            "necessarily one hour — see the caveat in the result."
        ),
    )


@tool("forecast_power", args_schema=ForecastArgs)
def forecast_power(
    model: str,
    state: Annotated[GraphState, InjectedState],
    horizon_steps: int = 6,
) -> str:
    """Forecast future solar power output using a pretrained model for the
    admin's currently selected installation.

    Use this when asked to predict/forecast future solar generation.
    See the 'model' field description for the available models and when
    to use each. Read the 'caveat' field in the result before presenting
    numbers as precise hourly predictions.
    """
    installation_id = state.get("installation_id")
    if installation_id is None:
        return "No installation selected yet."

    if model not in MODEL_KEYS:
        return f"Unknown model '{model}'. Valid options: {', '.join(MODEL_KEYS)}"

    horizon_steps = min(max(horizon_steps, 1), 48)
    need_exog = model_needs_exog(model)
    family = model_family(model)

    try:
        fitted = get_model(model)
    except FileNotFoundError as exc:
        return str(exc)

    lookback, min_required = _lookback_and_min_required(fitted, family)
    history = _fetch_recent_history(installation_id, lookback, need_exog)

    if len(history) < min_required:
        return (
            f"Not enough historical data to seed the {model_label(model)} forecast "
            f"(need at least {min_required} rows, got {len(history)}, "
            f"installation_id={installation_id})."
        )

    endog = [row[TARGET_COLUMN] for row in history]
    exog_note = ""

    try:
        if need_exog:
            exog = [[row[c] for c in EXOG_COLUMNS_DB] for row in history]
            applied = fitted.apply(endog, exog=exog, refit=False)
            future_exog = [exog[-1] for _ in range(horizon_steps)]
            forecast_vals = applied.forecast(steps=horizon_steps, exog=future_exog)
            exog_note = (
                " Weather inputs for the forecast horizon are held constant "
                "at the most recently observed values (no live weather "
                "forecast is wired in yet)."
            )
        else:
            applied = fitted.apply(endog, refit=False)
            forecast_vals = applied.forecast(steps=horizon_steps)

    except Exception as exc:
        logger.exception(f"forecast_power failed with model={model}")
        return f"Forecast failed: {exc}"

    last_row = history[-1]
    result = {
        "model": model,
        "model_label": model_label(model),
        "installation_id": installation_id,
        "target_column": TARGET_COLUMN,
        "last_observed_ts": _json_default(last_row["ts"])
        if isinstance(last_row["ts"], (datetime, date))
        else str(last_row["ts"]),
        "last_observed_value": float(last_row[TARGET_COLUMN]),
        "horizon_steps": horizon_steps,
        "forecast": [round(float(v), 2) for v in forecast_vals],
        "caveat": (
            "Values are indexed by step, not a fixed time interval: the "
            "models were trained on daytime-only readings with irregular "
            "gaps between them, so 'step N' means the Nth future reading at "
            "whatever cadence your live data arrives in, not necessarily N "
            "hours ahead." + exog_note
        ),
    }
    return json.dumps(result, default=_json_default, indent=2)


@tool("forecast_power_with_exog", args_schema=ForecastArgs)
def forecast_power_with_exog(
    model: str,
    state: Annotated[GraphState, InjectedState],
    horizon_steps: int = 6,
) -> str:
    """Backward-compatible wrapper for the legacy exogenous forecast name."""
    return forecast_power.func(model=model, horizon_steps=horizon_steps, state=state)
