"""The agent loop: the model calls the MCP server's tools until it can answer the user.

`Agent.run_turn` handles one user message. It asks the model for a reply, runs any tools the
model asks for, sends the results back, and repeats until the model answers without a tool call.

The conversation history belongs to the caller (the chat CLI, or an eval). `run_turn` appends
to it in place, so the next user message continues the same conversation.
"""

from collections.abc import Callable
from dataclasses import dataclass

from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from agent.providers.base import (
    Message,
    ModelReply,
    Provider,
    ToolCall,
    ToolResult,
    ToolSpec,
)

MAX_MODEL_CALLS = 20

# The agent's own role. The dataset rules (estimates, "Likely" labels, attribution) live in
# the MCP server's instructions and are appended by `system_prompt`, so there's one copy.
SYSTEM = """You answer questions about AI data center buildout: IT power, capital cost, chips
and compute, construction timelines, and cooling. Look up the data with the tools; make as
many tool calls as the question needs before answering. Keep answers concise."""

tracer = trace.get_tracer(__name__)


def system_prompt(server_instructions: str | None) -> str:
    """The agent's role plus the MCP server's instructions, if it sent any."""
    if not server_instructions:
        return SYSTEM
    return f"{SYSTEM}\n\n{server_instructions}"


async def list_tools(client: Client) -> list[ToolSpec]:
    """Every tool the MCP server offers, in the neutral format the providers take."""
    tools = []
    cursor = None
    while True:  # the server may split a long tool list into pages
        page = await client.list_tools(cursor=cursor)
        tools += [
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema,
            )
            for tool in page.tools
        ]
        cursor = page.next_cursor
        if cursor is None:
            return tools


@dataclass
class Agent:
    provider: Provider
    client: Client
    tools: list[ToolSpec]
    system: str

    async def run_turn(
        self, messages: list[Message], on_tool_call: Callable[[str, dict], None]
    ) -> ModelReply:
        """Run one user message to completion. Appends every reply and tool result to `messages`.

        Raises ProviderError if a model call fails.
        """
        with tracer.start_as_current_span(
            "invoke_agent",
            attributes={"gen_ai.operation.name": "invoke_agent"},
        ):
            for _ in range(MAX_MODEL_CALLS):
                with tracer.start_as_current_span("provider.reply") as span:
                    reply = await self.provider.reply(
                        self.system, messages, self.tools
                    )

                    if reply.input_tokens is not None:
                        span.set_attribute(
                            "gen_ai.usage.input_tokens", reply.input_tokens
                        )
                    if reply.output_tokens is not None:
                        span.set_attribute(
                            "gen_ai.usage.output_tokens", reply.output_tokens
                        )

                messages.append(reply)
                if not reply.tool_calls:
                    return reply

                for call in reply.tool_calls:
                    on_tool_call(call.name, call.arguments)
                    messages.append(await self.run_tool(call))

            return ModelReply(text="", stop_reason="max_model_calls")

    async def run_tool(self, call: ToolCall) -> ToolResult:
        """Run one tool on the MCP server. A failure becomes an error result the model can read
        and react to (e.g. fix its arguments), instead of ending the chat."""
        with tracer.start_as_current_span(
            f"execute_tool {call.name}",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": call.name,
            },
        ) as span:
            try:
                result = await self.client.call_tool(call.name, call.arguments)
            except MCPError as e:
                error_type = type(e).__name__
                span.record_exception(e)
                span.set_attribute("error.type", error_type)
                span.set_status(StatusCode.ERROR)

                return ToolResult(
                    call.id, call.name, f"Tool call failed: {e}", is_error=True
                )

            if result.is_error:
                span.set_attribute("error.type", "tool_error")
                span.set_status(StatusCode.ERROR)
            return ToolResult(
                call.id, call.name, result_text(result), is_error=result.is_error
            )


def result_text(result: CallToolResult) -> str:
    """The text the tool returned. MCP results are a list of blocks; tools here return text."""
    return "\n".join(block.text for block in result.content if block.type == "text")
