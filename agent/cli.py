"""Chat with an AI agent about data center buildout, using Epoch AI's dataset.

Usage (from the project root):
    uv run python -m agent

The agent launches the MCP server as a subprocess and talks to it over
stdio, the same way Claude Code does. You don't need to start it separately.

Model: a local model running in Ollama, or Claude via the Anthropic API
(set the ANTHROPIC_API_KEY environment variable).
"""

import argparse
import asyncio
import contextlib
import json
import os
import shlex
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from agent.agent import Agent, list_tools, system_prompt
from agent.providers import make_provider
from agent.providers.base import Message, ProviderError, UserMessage

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "ollama": "qwen3"}

DEFAULT_SERVER_COMMAND = "uv run python -m mcp_server"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run(
    provider: str, model: str, effort: str, server_command: str, show_server_log: bool
) -> None:
    """Start the MCP server subprocess, build the agent, then initiate the agent loop"""
    command, *args = shlex.split(server_command)
    server = StdioServerParameters(command=command, args=args, cwd=PROJECT_ROOT)
    with (
        contextlib.nullcontext(sys.stderr) if show_server_log else open(os.devnull, "w")
    ) as errlog:
        async with Client(stdio_client(server, errlog=errlog)) as mcp_client:
            agent = Agent(
                provider=make_provider(provider, model, effort),
                client=mcp_client,
                tools=await list_tools(mcp_client),
                system=system_prompt(mcp_client.instructions),
            )

            print(f"MCP server with Epoch AI dataset - {provider} {model}. Effort {effort}.\n")
            await chat(agent)


def show_tool_call(name: str, arguments: dict) -> None:
    """Print each tool call the agent makes, so you can see what it's doing."""
    print(f"  [tool] {name}({json.dumps(arguments)})")


async def chat(agent: Agent) -> None:
    messages: list[Message] = []  # the whole conversation, kept across user messages
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"exit", "quit"}:
            return
        if not text:
            continue

        before = len(messages)
        messages.append(UserMessage(text))
        try:
            reply = await agent.run_turn(messages, on_tool_call=show_tool_call)
        except ProviderError as e:
            del messages[before:]
            print(f"  ({e})\n")
            continue
        if reply.stop_reason == "max_model_calls":
            print("  (stopped: too many steps for one message)")
        print(f"\nagent> {reply.text}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["anthropic", "ollama"],
        help=f"Provider (default {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        help=f"Model (default {DEFAULT_MODELS['anthropic']} for anthropic, {DEFAULT_MODELS['ollama']} for ollama).",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="How much the model thinks before answering (default medium). Claude only; Ollama ignores it.",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_COMMAND,
        help=f"Command that starts the MCP server over stdio (default: {DEFAULT_SERVER_COMMAND!r}, run from the project root).",
    )
    parser.add_argument(
        "--show-server-log",
        action="store_true",
        help="Show the server's log output (dataset update checks, loaded tables, warnings).",
    )

    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    if args.provider == "anthropic" and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        sys.exit(
            "No Claude API credentials found. Set ANTHROPIC_API_KEY (from console.anthropic.com) and try again."
        )

    try:
        asyncio.run(run(args.provider, args.model, args.effort, args.server, args.show_server_log))
    except* KeyboardInterrupt:
        print("Process was cancelled.")
    except* FileNotFoundError:
        sys.exit(
            f"Could not start the MCP server: {shlex.split(args.server)[0]!r} was not found. Check --server."
        )
    except* MCPError:
        sys.exit(
            "The MCP server stopped unexpectedly. Run with --show-server-log to see its error output."
        )
