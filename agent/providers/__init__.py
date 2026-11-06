"""Model providers. `make_provider` is the one place that picks which one to use."""

from agent.providers.base import Provider


def make_provider(name: str, model: str, effort: str) -> Provider:
    # Each provider's SDK is imported only when that provider is picked, so running
    # with Claude never loads the Ollama SDK (and the other way round).
    if name == "anthropic":
        from agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model, effort)
    if name == "ollama":
        from agent.providers.ollama import OllamaProvider

        return OllamaProvider(model)
    raise ValueError(f"Unknown provider: {name!r}")
