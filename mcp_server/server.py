from mcp.server import MCPServer

INSTRUCTIONS = """\
Tools over Epoch AI's AI datacenters dataset: data center locations, owners, users, IT power, compute (in H100 equivalents), capital cost, construction timelines, chip counts, and cooling equipment.

When answering:
- Treat every figure as an estimate. Present totals and comparisons as approximate, and avoid false precision.
- Some user affiliations are marked "Likely" or "Speculative". Include the label when you report them. Don't state them as confirmed.
- Answer only from what the tools return. If the dataset doesn't cover something, say so instead of filling the gap from general knowledge.
- Whenever you reproduce or summarize the data, attribute it to Epoch AI (CC BY 4.0).
"""

mcp = MCPServer(
    "epoch-ai-datacenters",
    instructions=INSTRUCTIONS,
)
