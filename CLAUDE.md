# Project Overview
An AI agent app for AI data center buildout analysis. An MCP server exposes tools over Epoch AI's data center dataset, and a user prompts an LLM to retrieve data and run analysis (IT power, capex, GPUs/compute, timelines, etc.).

## How the user interacts
Both paths share one MCP server:
- **Path A:** Claude Code → MCP server
- **Path B:** CLI → custom AI agent (Qwen on Ollama) → MCP server

The custom agent takes the place of Claude Code in Path B: it is the MCP client, runs the agent loop, and handles human-in-the-loop approval. It talks to models through a provider abstraction so the model can be swapped by config.

## Data source
[Epoch AI — AI Data Centers](https://epoch.ai/data/ai-data-centers), licensed CC BY 4.0 (attribution to Epoch AI required in any output that reproduces the data).

CSVs at `https://epoch.ai/data/data_centers/` (or the bundle `data_centers.zip`):
- `data_centers.csv` — one row per data center: location, owner, users, IT power, compute (H100e), capital cost
- `data_center_timelines.csv` — construction/operation dates and yearly IT power, compute, cost
- `data_centers_chip_quantities.csv` — chip counts per data center
- `data_center_chillers.csv`, `data_center_cooling_towers.csv` — cooling equipment

Chip types: `https://epoch.ai/data/ai_chip_sales_chip_types.csv`

Figures are estimates (Epoch's stated 80% intervals: ~1.4x for IT power, ~1.5x for compute, ~1.6x for cost, ~6 months for dates). Analysis should treat them as estimates, not exact values. Some user affiliations are marked "Likely" or "Speculative".

## Conventions
- Tracing and metrics use OpenTelemetry.

## Observability

Requires `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`.

Start the local Aspire dashboard:

```powershell
docker run --rm -d --name aspire-dashboard -p 127.0.0.1:18888:18888 -p 127.0.0.1:4318:18890 -e ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS=true mcr.microsoft.com/dotnet/aspire-dashboard:latest
```

Open `http://localhost:18888`.

## Evals
- Golden dataset: `evals/golden_dataset.jsonl`, answers computed from `data/data_centers.zip`.
- The server never downloads data on its own, so the zip only changes when you update it. To update: fetch the newest zip with `uv run python -m mcp_server.data.fetch`, regenerate the answers with `uv run python -m evals.build_golden_dataset`, and review what changed. The eval refuses to run if the golden dataset doesn't match the zip.
- The golden answers are computed for one day (`AS_OF` in `evals/build_golden_dataset.py`, stored as `as_of` on each case). During an eval run, the tools and the agent's prompt use that day as "today" (the eval sets `EVAL_TODAY`), so results don't change with the day the eval runs.
