# AI Data Center Buildout Agent

Ask questions about the global AI data center buildout in plain English, such as *"Which 5 data centers have the most IT power right now?"* or *"How much compute will be online by the end of 2027?"*, and get answers from a real dataset instead of a model's memory.

An **MCP server** exposes 7 tools over [Epoch AI's AI Data Centers dataset](https://epoch.ai/data/ai-data-centers). Any MCP client can use it: **Claude Code**, or a **custom agent** I built that runs through Claude API or a local model (Qwen3 14B on Ollama) with workloads running on my RTX 4070 TI through the same code.

## Highlights

- **One MCP server, two clients.** Claude Code and my own agent share the same tools, so models can be compared fairly.
- **Custom agent loop** with a provider abstraction. Switching between a local model and Claude is a command-line flag.
- **CI evaluation pipeline**: a 40-question golden dataset generated from the data, graded by code checks plus an LLM judge (Claude Opus 5.5), and run in **GitHub Actions** with a 90% pass gate.
- **Evals that drove the fixes.** Every improvement below came from reading failed eval results.
- **OpenTelemetry tracing** of every model call and tool call, with input/output token counts, viewable in the Aspire dashboard.
- **Honest answers about uncertain data.** Tools flag estimates, projections and unconfirmed ownership so the model can't present them as fact.

## Eval results

Same 40 questions throughout graded by Claude Opus 5.5.

| Step | qwen3:14b (local) | What changed |
|---|---|---|
| Baseline | **24/40** (60%) | |
| Fix tools and context | **35/40** (88%) | See below |
| Final | **37/40** (93%) | Rounding tolerance, refreshed data |

| Model | Final score | Time |
|---|---|---|
| Claude Sonnet 5.5 (CI) | **38/40** (95%) | 2 min |
| qwen3:14b (local GPU) | **37/40** (93%) | 22 min |
| Claude Opus 5.5 | **37/40** (93%)* | 4 min |

\*Run before the rounding tolerance was raised from 1% to 2.5%. Two of its three misses were correctly rounded answers ("about 16 million" for 15.77M), so it would likely score 39/40 today.

What the failed questions revealed:

- **Ollama silently cut off prompts.** Its default context window is 4,096 tokens, and the tool definitions alone used ~3,500. Large tool results, and sometimes the question itself, were dropped without any error. Spotted because results 3x larger cost the same number of tokens. Fixed by setting the context to 16,384.
- **Some questions were unanswerable.** The data had water use and total facility power, but no tool returned them.
- **The model didn't know today's date**, so it guessed 2023 for "right now" questions.
- **The answer key drifted daily.** The data includes future projections, so a "today" answer changes as days pass. Evals now pin "today" to the date the answer key was built.
- **Totals hid unconfirmed data.** A sum like "Meta's IT power" now reports how much comes from sites with *likely* or *speculative* ownership.

## Tech stack

Python 3.14 · uv · MCP Python SDK · pandas · Pydantic · Anthropic SDK · Ollama (Qwen3 14B) · OpenTelemetry · .NET Aspire dashboard · GitHub Actions

## Project layout

| Folder | What it is |
|---|---|
| `mcp_server/` | MCP server and tools: lookup, filter/aggregate, timelines, chips, cooling |
| `agent/` | Custom agent: CLI, agent loop, Anthropic and Ollama providers |
| `evals/` | Golden dataset builder, eval runner, LLM judge |
| `observability/` | OpenTelemetry setup |
| `data/` | Epoch AI dataset (zip) and its download date |

## Try it

Needs [uv](https://docs.astral.sh/uv/). For a local model, install [Ollama](https://ollama.com) and run `ollama pull qwen3:14b`. For Claude, set `ANTHROPIC_API_KEY`.

```bash
git clone https://github.com/itshendson/datacenter-buildout.git
cd datacenter-buildout
uv sync
```

**Chat with the agent:**

```bash
uv run python -m agent --provider ollama      # local Qwen3 14B
uv run python -m agent --provider anthropic   # Claude
```

**Use the tools from Claude Code:**

```bash
claude mcp add epoch-datacenters -- uv run python -m mcp_server
```

**Run the evals** (the judge uses Claude, so `ANTHROPIC_API_KEY` is required):

```bash
uv run python -m evals --provider ollama --model qwen3:14b
```

**See traces:** start the dashboard, set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`, then open http://localhost:18888.

```bash
docker run --rm -d --name aspire-dashboard -p 127.0.0.1:18888:18888 -p 127.0.0.1:4318:18890 -e ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS=true mcr.microsoft.com/dotnet/aspire-dashboard:latest
```

## Data

Data from [Epoch AI, AI Data Centers](https://epoch.ai/data/ai-data-centers), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All figures are Epoch's estimates.
