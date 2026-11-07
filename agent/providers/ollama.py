"""Local models (e.g. Qwen) through Ollama."""

import uuid

import ollama

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


class OllamaProvider(Provider):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.client = ollama.AsyncClient()  # reads OLLAMA_HOST, defaults to localhost:11434

    async def reply(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelReply:
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, *to_ollama_messages(messages)],
                tools=[to_ollama_tool(tool) for tool in tools],
            )
        except (ollama.ResponseError, ollama.RequestError, ConnectionError) as e:
            raise ProviderError(f"Ollama error: {e}") from e

        # Ollama doesn't give tool calls an id, so make one to pair each call with its result.
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=call.function.name,
                arguments=dict(call.function.arguments),
            )
            for call in response.message.tool_calls or []
        ]
        if tool_calls:
            stop_reason = "tool_calls"
        elif response.done_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end"

        return ModelReply(
            text=response.message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
        )


def to_ollama_tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def to_ollama_messages(messages: list[Message]) -> list[dict]:
    result = []
    for message in messages:
        if isinstance(message, UserMessage):
            result.append({"role": "user", "content": message.text})

        elif isinstance(message, ModelReply):
            entry = {"role": "assistant", "content": message.text}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in message.tool_calls
                ]
            result.append(entry)

        elif isinstance(message, ToolResult):
            # Ollama pairs a result with its call by tool name, not id.
            result.append({"role": "tool", "content": message.content, "tool_name": message.name})
    return result
