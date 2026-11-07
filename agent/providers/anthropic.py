"""Claude, through the Anthropic API."""

import anthropic

from agent.providers.base import (
    Message,
    ModelReply,
    Provider,
    ProviderError,
    ToolCall,
    ToolResult,
    ToolSpec,
    UserMessage,
)

MAX_TOKENS = 16000

# Claude's stop reasons -> the neutral ones in ModelReply.
STOP_REASONS = {
    "end_turn": "end",
    "tool_use": "tool_calls",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicProvider(Provider):
    def __init__(self, model: str, effort: str) -> None:
        super().__init__(model)
        self.effort = effort
        self.client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

    async def reply(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelReply:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=to_anthropic_messages(messages),
                tools=[to_anthropic_tool(tool) for tool in tools],
                output_config={"effort": self.effort},
                # The tools and system prompt are the same on every call, so cache them.
                cache_control={"type": "ephemeral"},
            )
        except anthropic.APIError as e:
            raise ProviderError(f"Claude API error: {e}") from e

        usage = response.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0

        return ModelReply(
            text="".join(block.text for block in response.content if block.type == "text"),
            tool_calls=[
                ToolCall(id=block.id, name=block.name, arguments=block.input)
                for block in response.content
                if block.type == "tool_use"
            ],
            stop_reason=STOP_REASONS.get(response.stop_reason, response.stop_reason),
            input_tokens=usage.input_tokens + cache_read + cache_write,
            output_tokens=usage.output_tokens,
            raw=response.content,
        )


def to_anthropic_tool(tool: ToolSpec) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def to_anthropic_messages(messages: list[Message]) -> list[dict]:
    result = []
    for message in messages:
        if isinstance(message, UserMessage):
            result.append({"role": "user", "content": message.text})

        elif isinstance(message, ModelReply):
            # Send Claude's own reply back unchanged (keeps thinking blocks).
            result.append({"role": "assistant", "content": message.raw})

        elif isinstance(message, ToolResult):
            block = {
                "type": "tool_result",
                "tool_use_id": message.call_id,
                "content": message.content,
                "is_error": message.is_error,
            }
            # Claude expects all results for one reply in a single user message.
            previous = result[-1] if result else None
            if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                previous["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})
    return result
