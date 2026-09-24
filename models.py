"""Loading the pretrained forecasting models: AR, ARX, ARMAX, SARIMAX,
each in a 'filtered' and (where available) 'unfiltered' variant.

AR / ARX were fit with statsmodels' AutoReg and saved with
`results.save(path)`. ARMAX / SARIMAX were fit with statsmodels'
SARIMAX (statespace) and saved the same way. Only an unfiltered
SARIMAX exists -- there's no "sarimax_filtered".
"""

import os
from functools import lru_cache

import statsmodels.api as sm

# The column all models were trained to predict ("DC" in the notebooks).
TARGET_COLUMN = os.getenv("FORECAST_TARGET_COLUMN", "power_w")

# Exog columns used by every exogenous model (ARX, ARMAX, SARIMAX), as
# stored in the DB (snake_case), mirroring the notebooks' `exog_cols`:
#   ['Solar_Radiation_Wm2', 'Outdoor_Temp_C', 'Wind_Speed_ms',
#    'Humidity_pct', 'AQI_US']
EXOG_COLUMNS_DB = [
    "solar_radiation_wm2",
    "outdoor_temp_c",
    "wind_speed_ms",
    "humidity_pct",
    "aqi_us",
]

# "family" tells forecast_tools.py how to read the fitted lag/order info
# and how to call it:
#   "autoreg" -> statsmodels AutoReg results (AR, ARX): lag info lives on
#                `.model.ar_lags`.
#   "sarimax" -> statsmodels SARIMAX/statespace results (ARMAX, SARIMAX):
#                lag info lives on `.model.order` / `.model.seasonal_order`.
_MODEL_SPECS = {
    "ar_filtered": {
        "path_env": "AR_FILTERED_MODEL_PATH",
        "default_path": "models/ar_model.pkl",
        "family": "autoreg",
        "needs_exog": False,
        "label": "AR (filtered)",
    },
    "ar_unfiltered": {
        "path_env": "AR_UNFILTERED_MODEL_PATH",
        "default_path": "models/ar_model_unfiltered.pkl",
        "family": "autoreg",
        "needs_exog": False,
        "label": "AR (unfiltered)",
    },
    "arx_filtered": {
        "path_env": "ARX_FILTERED_MODEL_PATH",
        "default_path": "models/arx_model.pkl",
        "family": "autoreg",
        "needs_exog": True,
        "label": "ARX (filtered)",
    },
    "arx_unfiltered": {
        "path_env": "ARX_UNFILTERED_MODEL_PATH",
        "default_path": "models/arx_model_unfiltered.pkl",
        "family": "autoreg",
        "needs_exog": True,
        "label": "ARX (unfiltered)",
    },
    "armax_filtered": {
        "path_env": "ARMAX_FILTERED_MODEL_PATH",
        "default_path": "models/armax_model_filtered.pkl",
        "family": "sarimax",
        "needs_exog": True,
        "label": "ARMAX (filtered)",
    },
    "armax_unfiltered": {
        "path_env": "ARMAX_UNFILTERED_MODEL_PATH",
        "default_path": "models/armax_model_unfiltered.pkl",
        "family": "sarimax",
        "needs_exog": True,
        "label": "ARMAX (unfiltered)",
    },
    "sarimax_unfiltered": {
        "path_env": "SARIMAX_UNFILTERED_MODEL_PATH",
        "default_path": "models/sarimax_model_unfiltered.pkl",
        "family": "sarimax",
        "needs_exog": True,
        "label": "SARIMAX (unfiltered)",
    },
}

MODEL_KEYS = list(_MODEL_SPECS.keys())


def model_family(model_key: str) -> str:
    return _MODEL_SPECS[model_key]["family"]


def model_needs_exog(model_key: str) -> bool:
    return _MODEL_SPECS[model_key]["needs_exog"]


def model_label(model_key: str) -> str:
    return _MODEL_SPECS[model_key]["label"]


@lru_cache(maxsize=None)
def get_model(model_key: str):
    """Load (and cache) a fitted model by its registry key, e.g. 'arx_filtered'."""
    if model_key not in _MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_key}'. Valid: {MODEL_KEYS}")
    spec = _MODEL_SPECS[model_key]
    path = os.getenv(spec["path_env"], spec["default_path"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{model_key} model not found at '{path}'. Set {spec['path_env']} in .env."
        )
    return sm.load(path)
