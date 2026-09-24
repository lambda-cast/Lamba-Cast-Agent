# Energy Data Agent

This project contains a LangGraph agent for energy analysis. The planned agent will:

1. Fetch weather and solar-panel data from PostgreSQL.
2. Use a forecasting tool to predict energy production or consumption.
3. Explain results and ask for clarification when a request is incomplete.

## Graph

![LangGraph workflow](graph.png)

## Project layout

```text
agent/
├── agent_app/
│   ├── agent.py       # LLM node and tool binding
│   ├── graph.py       # LangGraph workflow entrypoint
│   ├── prompts.py     # System prompt
│   ├── registry.py    # Tools exposed to the LLM
│   ├── state.py       # Shared graph state
│   └── tools/
│       ├── database.py    # Planned PostgreSQL access tools
│       └── forecasting.py # Planned energy forecasting tool
├── tests/             # Focused graph and tool tests
├── .env               # Local secrets and service configuration
├── langgraph.json     # LangGraph CLI configuration
└── pyproject.toml
```

## Planned tools

### PostgreSQL data tool

The database tool will expose narrow, read-only operations for querying weather and panel measurements. Database credentials will come from environment variables; SQL will use parameterized queries and return validated, serializable records rather than raw database objects.

Expected configuration:

```text
POSTGRES_DSN=postgresql://user:password@host:5432/database
```

### Forecasting tool

The forecasting tool will accept a validated time series, forecast horizon, and model options, then return predictions with timestamps and useful metadata such as the model name and forecast interval. It should remain independent from the database tool so forecasts can also be tested with local or synthetic data.

## Development

Install dependencies and run the LangGraph development server from this directory:

```bash
uv sync
uv run langgraph dev
```

The graph configured for the CLI is `agent_app.graph:graph`.

Before connecting production data, add the required LiteLLM settings to `.env` and configure PostgreSQL access through `POSTGRES_DSN`.