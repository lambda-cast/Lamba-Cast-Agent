from agent_app.tools.postgres_tools import get_latest_reading, get_readings_in_range, run_readonly_sql
from agent_app.tools.forecast_tools import forecast_power, forecast_power_with_exog

TOOL_REGISTRY = [
    get_latest_reading,
    get_readings_in_range,
    run_readonly_sql,
    forecast_power,
    forecast_power_with_exog,
]