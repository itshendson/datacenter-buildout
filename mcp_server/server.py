import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from mcp_server.data.fetch import ZIP_PATH, refresh
from mcp_server.data.store import DataStore

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: MCPServer) -> AsyncGenerator[DataStore]:
    logger.info(f"changed: {refresh()}")

    store = DataStore(ZIP_PATH)
    for name, df in vars(store).items():
        logger.info(f"{name}: {df.shape}")

    yield store


mcp = MCPServer(
    "epoch-ai-datacenters",
    instructions=(
        "Tools over Epoch AI's AI datacenters dataset: data center locations, owners, users, IT power, compute (in H100 equivalents), capital cost, construction timelines, chip counts, and cooling equipment.\n"
        "When answering:\n"
        "Treat every figure as an estimate. Present totals and comparisons as approximate, and avoid false precision.\n"
        'Some user affiliations are marked "Likely" or "Speculative". Include the label when you report them. Don\'t state them as confirmed.\n'
        "Answer only from what the tools return. If the dataset doesn't cover something, say so instead of filling the gap from general knowledge.\n"
        "Whenever you reproduce or summarize the data, attribute it to Epoch AI (CC BY 4.0).\n"
    ),
    lifespan=app_lifespan
)
