"""Eval script: run each task in golden_dataset.jsonl through the agent with
the MCP tools, record what happened, and grade the answers.

The judge always uses Claude, so ANTHROPIC_API_KEY is needed even when the
model under test is Qwen. Results are written to evals/results/<timestamp>.json,
one file per run.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import anthropic
from mcp import Client
from pydantic import BaseModel

from agent.agent import Agent, list_tools, system_prompt
from agent.cli import DEFAULT_MODELS, DEFAULT_PROVIDER
from agent.providers import make_provider
from agent.providers.base import (
    Message,
    ModelReply,
    ProviderError,
    ToolResult,
    UserMessage,
)
from mcp_server.data.fetch import ZIP_PATH
from mcp_server.server import TODAY_ENV, mcp

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "medium"

class Verdict(BaseModel):
    reason: str
    passed: bool

def check_expected(expected: dict, answer: str) -> bool | None:
    answer = answer.lower()

    kind = expected["type"]
    if kind == "text":
        return expected["value"].lower() in answer
    if kind == "set":
        return all(item.lower() in answer for item in expected["value"])
    if kind == "ranked":
        positions = [answer.find(item.lower()) for item in expected ["value"]]
        return -1 not in positions and positions == sorted(positions)

    return None

async def check_rubric(judge: anthropic.AsyncAnthropic, task: dict, answer: str) -> tuple[bool, str]:
    rubric = "\n".join(f"- {item}" for item in task["rubric"]) or "- (none)"
    must_not = "\n".join(f"- {item}" for item in task["must_not"]) or "- (none)"

    prompt = f"""Grade an AI agent's answer to a question about Epoch AI's data center dataset.

Question: {task["question"]}

Agent's answer: {answer}

Expected answer {json.dumps(task["expected"])}
- If its type is "not_in_dataset", the answer should say the dataset doesn't cover this.
- If its type is "judge", there is no fixed answer; grade on the rules below.
- Numbers within the given rel_tol count as matching.

The answer must:
{rubric}

The answer must not:
{must_not}

Pass only if the answer matches the expected answer, follows every "must" rule, and breaks no "must not" rule."""

    response = await judge.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        output_config={"effort": JUDGE_EFFORT},
        output_format=Verdict,
        messages=[{"role": "user", "content": prompt}]
    )

    verdict = response.parsed_output
    if verdict is None:
        return False, f"Judge gave no verdict (stop reason {response.stop_reason})"
    return verdict.passed, verdict.reason


async def run_agent(agent: Agent, question: str) -> dict:
    messages: list[Message] = [UserMessage(question)]

    tool_calls:list[dict] = [] # Gets filled by agent.run_turn(...). Represents list of tool name and argument inputs

    def record_tool_call(name: str, arguments: dict) -> None:
        tool_calls.append({"name": name, "arguments": arguments})

    # Measure and run agents
    started = time.perf_counter()
    reply = await agent.run_turn(messages, on_tool_call=record_tool_call)
    seconds = time.perf_counter() - started

    # Record tool results from agent
    tool_results = [m for m in messages if isinstance(m, ToolResult)] # Represents results of each tool call

    for call, result in zip(tool_calls, tool_results, strict=True):
        call["is_error"] = result.is_error
        call["result"] = result.content

    replies = [m for m in messages if isinstance(m, ModelReply)]
    input_tokens = sum(r.input_tokens or 0 for r in replies)
    output_tokens = sum(r.output_tokens or 0 for r in replies)

    return {
        "answer": reply.text,
        "stop_reason": reply.stop_reason,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": seconds,
    }

async def run_task(judge: anthropic.AsyncAnthropic, agent: Agent, task: dict) -> dict:
    print(f"  running {task['id']}...", flush=True)
    try:
        run = await run_agent(agent, task["question"])
    except ProviderError as e:
        print(f"  {task['id']}: API error: {e}", flush=True)
        return {"id": task["id"], "passed": False, "error": str(e)}
    
    expected_ok = check_expected(task["expected"], run["answer"])
    rubric_ok, judge_reason = await check_rubric(judge, task, run["answer"])
    return {
        "id": task["id"],
        "judge_reason": judge_reason,
        "passed": rubric_ok if expected_ok is None else expected_ok and rubric_ok,
        "expected_ok": expected_ok,
        "rubric_ok": rubric_ok,
        "error": None,
        "answer": run["answer"],
        "stop_reason": run["stop_reason"],
        "tool_call_count": len(run["tool_calls"]),
        "tool_errors": sum(call["is_error"] for call in run["tool_calls"]),
        "tool_calls": run["tool_calls"],
        "input_tokens": run["input_tokens"],
        "output_tokens": run["output_tokens"],
        "seconds": run["seconds"],
    }

async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="*", help="Run only these task ids.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["anthropic", "ollama"], help="Provider of the model being evaluated.")
    parser.add_argument("--model", help="Model being evaluated. Default depends on --provider.")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("No Claude API credentials found. The judge uses Claude, so set " \
        "ANTHROPIC_API_KEY (from console.anthropic.com) even when testing Qwen.")

    # Load golden dataset
    dataset_path = Path(__file__).parent / "golden_dataset.jsonl"
    tasks: list[dict] = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Hash check task against data_centers.zip to ensure answer freshness
    dataset_hash = zip_sha256()
    if args.only:
        tasks = [t for t in tasks if t["id"] in args.only]
    stale = stale_task_ids(tasks, dataset_hash)
    if stale:
        sys.exit(f"Golden dataset is out of date for: {', '.join(stale)}. "
                 "Regenerate it with: uv run python -m evals.build_golden_dataset")

    # The golden answers were computed for one day (as_of). Pin "today" to that day for both the
    # tools and the agent's prompt, so "right now" questions don't drift as real days pass.
    as_of_dates = {t["as_of"] for t in tasks}
    if len(as_of_dates) != 1:
        sys.exit(f"Expected one as_of date across the tasks, found: {sorted(as_of_dates) or 'no tasks'}.")
    as_of = as_of_dates.pop()
    os.environ[TODAY_ENV] = as_of

    # Output result location
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    # Run actual tasks and return results
    claude = anthropic.AsyncAnthropic()
    async with Client(mcp) as mcp_client:
        tools = await list_tools(mcp_client)
        agent = Agent(
            provider=make_provider(
                args.provider, args.model, args.effort
            ),
            client=mcp_client,
            tools=tools,
            system=system_prompt(mcp_client.instructions, today=date.fromisoformat(as_of)),
        )
        print(
            f"{len(tasks)} tasks · evaluating model {args.model} {args.effort} · judge {JUDGE_MODEL} {JUDGE_EFFORT} · today pinned to {as_of}"
        )
        rows = []
        for task in tasks:
            rows.append(await run_task(claude, agent, task))

        results = {
            "provider": args.provider,
            "model": args.model,
            "effort": args.effort,
            "judge_model": JUDGE_MODEL,
            "judge_effort": JUDGE_EFFORT,
            "data_sha256": dataset_hash,
            "as_of": as_of,
            "tasks": rows
        }
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        passed = sum(row["passed"]for row in rows)
        print(f"Passed {passed}/{len(rows)}")
        print(f"\nResults: {results_path}")

def stale_task_ids(tasks: list[dict], dataset_hash: str) -> list[str]:
    return [t["id"] for t in tasks if t["data_sha256"] != dataset_hash]

def zip_sha256() -> str:
    return hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()