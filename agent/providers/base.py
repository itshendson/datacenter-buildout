"""Base interface for different providers including Ollama, Anthropic, and other future providers.

Also defines the neutral types the agent loop works with. The loop only ever sees these;
each provider converts them to and from its own API's format, so swapping models never
touches the loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool the model may call, as listed by the MCP server."""

    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    """The model asking to run one tool. `id` pairs it with its ToolResult."""

    id: str
    name: str
    arguments: dict


@dataclass
class UserMessage:
    """What the user typed."""

    text: str


@dataclass
class ModelReply:
    """One reply from the model. An empty `tool_calls` list means the model is done."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end"  # "end", "tool_calls", "max_tokens", or "refusal"
    input_tokens: int | None = None
    output_tokens: int | None = None
    # The provider's own copy of this reply, so it can be sent back unchanged on the
    # next call. Claude needs this: its replies carry thinking blocks that must be
    # returned exactly as they came. Providers that don't need it leave it as None.
    raw: Any = None


@dataclass
class ToolResult:
    """The output of one tool call, sent back to the model."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


# One entry in the conversation history.
Message = UserMessage | ModelReply | ToolResult


class ProviderError(Exception):
    """A call to the model's API failed (network, auth, rate limit, bad request, ...)."""


class Provider(ABC):
    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def reply(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelReply:
        """Send the conversation to the model and return its next reply.

        Raises ProviderError if the API call fails.
        """
