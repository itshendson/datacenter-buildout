# Project Overview
An AI agent app for AI data center buildout analysis. An MCP server exposes tools over Epoch AI's data center dataset, and a user prompts an LLM to retrieve data and run analysis (IT power, capex, GPUs/compute, timelines, etc.).

## How the user interacts
Both paths share one MCP server:
- **Path A:** Codex → MCP server
- **Path B:** CLI → custom AI agent (Qwen on Ollama) → MCP server

The custom agent takes the place of Codex in Path B: it is the MCP client, runs the agent loop, and handles human-in-the-loop approval. It talks to models through a provider abstraction so the model can be swapped by config.

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
