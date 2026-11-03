"""query_chips: find information about data center chips"""

from typing import Annotated, Literal

import pandas as pd
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, mcp

# The metric names the model can pass, and the chip_types column each one reads.
METRIC_COLUMNS = {
    "tdp_w": "TDP (W) (from ML Hardware (linked))",
    "flops_16bit": "16-bit FLOP/s",
    "ops_8bit": "8-bit OP/s",
    "memory_bytes": "Memory",
    "memory_bandwidth": "Memory bandwidth",
    "h100e": "H100e",
    "cost_usd": "Cost per chip (approx.)",
}

# The group_by names the model can pass, and the chip_types column each one reads.
GROUP_COLUMNS = {
    "designer": "Designer",
    "chinese_market": "Primarily for Chinese market",
}


class ChipResult(BaseModel):
    rows: list[dict] = Field(
        description=(
            "The answer. list and rank: one dict per chip type with all its columns. "
            "count: [{'count': n}]. "
            "With group_by: one dict per group, e.g. {'designer': 'Nvidia', 'count': 9}."
        )
    )
    total_rows: int = Field(
        description="Number of chip types that matched the filters, before limit"
    )
    message: str | None = Field(
        description="Set when results were cut off by limit, e.g. 'Showing 20 of 32.'"
    )
    notes: str = Field(
        default=(
            "Specs come from Epoch AI's ML Hardware database; H100e and cost per chip are "
            "Epoch AI estimates. Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Caveats and attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def query_chips(
    chip_type: Annotated[
        str | None,
        Field(description="Short name of the chip, e.g. 'TPU v3', 'B300'. Case-insensitive."),
    ] = None,
    full_name: Annotated[
        str | None,
        Field(
            description=(
                "Full product name, e.g. 'NVIDIA GB200', 'TPU v7 Ironwood'. Case-insensitive. "
                "Use when the question names the product rather than the short name."
            )
        ),
    ] = None,
    designer: Annotated[
        Literal["Nvidia", "Google", "AMD", "Amazon", "Huawei", "Cambricon"] | None,
        Field(description="Company that designed the chip."),
    ] = None,
    chinese_market: Annotated[
        bool | None,
        Field(
            description=(
                "True: only chips made mainly for the Chinese market, such as Huawei Ascend "
                "or the A800 (a China version of the A100). False: leave those out."
            )
        ),
    ] = None,
    min_tdp_w: Annotated[
        float | None,
        Field(description="Only chips with at least this thermal design power (max power draw), in watts, e.g. 700."),
    ] = None,
    max_tdp_w: Annotated[
        float | None,
        Field(description="Only chips with at most this thermal design power, in watts."),
    ] = None,
    min_flops_16bit: Annotated[
        float | None,
        Field(description="Only chips with at least this 16-bit performance, in FLOP/s, e.g. 1e15."),
    ] = None,
    max_flops_16bit: Annotated[
        float | None,
        Field(description="Only chips with at most this 16-bit performance, in FLOP/s."),
    ] = None,
    min_ops_8bit: Annotated[
        float | None,
        Field(description="Only chips with at least this 8-bit performance, in OP/s, e.g. 2e15."),
    ] = None,
    max_ops_8bit: Annotated[
        float | None,
        Field(description="Only chips with at most this 8-bit performance, in OP/s."),
    ] = None,
    min_memory_bytes: Annotated[
        float | None,
        Field(description="Only chips with at least this much memory, in bytes, e.g. 8e10 for 80 GB."),
    ] = None,
    max_memory_bytes: Annotated[
        float | None,
        Field(description="Only chips with at most this much memory, in bytes."),
    ] = None,
    min_memory_bandwidth: Annotated[
        float | None,
        Field(description="Only chips with at least this memory bandwidth, in bytes per second, e.g. 3.35e12 for 3.35 TB/s."),
    ] = None,
    max_memory_bandwidth: Annotated[
        float | None,
        Field(description="Only chips with at most this memory bandwidth, in bytes per second."),
    ] = None,
    min_h100e: Annotated[
        float | None,
        Field(description="Only chips with at least this much compute per chip, in H100 equivalents (H100 = 1)."),
    ] = None,
    max_h100e: Annotated[
        float | None,
        Field(description="Only chips with at most this much compute per chip, in H100 equivalents."),
    ] = None,
    min_cost_usd: Annotated[
        float | None,
        Field(description="Only chips with at least this approximate price per chip, in US dollars, e.g. 25000."),
    ] = None,
    max_cost_usd: Annotated[
        float | None,
        Field(description="Only chips with at most this approximate price per chip, in US dollars."),
    ] = None,
    operation: Annotated[
        Literal["list", "count", "rank"],
        Field(
            description=(
                "What to do with the matching chips. "
                "'list': return them. 'count': how many. "
                "'rank': sort by metric, highest first."
            )
        ),
    ] = "list",
    metric: Annotated[
        Literal["tdp_w", "flops_16bit", "ops_8bit", "memory_bytes", "memory_bandwidth", "h100e", "cost_usd"] | None,
        Field(
            description=(
                "Which number to rank by: 'tdp_w' (watts), 'flops_16bit' (FLOP/s), 'ops_8bit' (OP/s), "
                "'memory_bytes', 'memory_bandwidth' (bytes per second), 'h100e' (H100 equivalents), "
                "'cost_usd' (approximate price per chip). Required for rank."
            )
        ),
    ] = None,
    group_by: Annotated[
        Literal["designer", "chinese_market"] | None,
        Field(description="Split results into one row per designer, or Chinese-market vs not. Only for count."),
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=100, description="Maximum rows to return.")
    ] = 20,
    ctx: Context = None,
) -> ChipResult:
    """Find chip types by name, designer or specs, then list, count or rank them.

    Use for questions about chip specs, e.g. "what's the B300's TDP", "which chip has the most
    memory", "how many Nvidia chip types are there", "chips above 1 H100e that cost under $20,000".

    Not for how many chips are installed at data centers; use query_installed_chips.
    """
    store: DataStore = ctx.request_context.lifespan_context

    df = _filter_chips(
        store.chip_types,
        chip_type=chip_type,
        full_name=full_name,
        designer=designer,
        chinese_market=chinese_market,
        min_tdp_w=min_tdp_w,
        max_tdp_w=max_tdp_w,
        min_flops_16bit=min_flops_16bit,
        max_flops_16bit=max_flops_16bit,
        min_ops_8bit=min_ops_8bit,
        max_ops_8bit=max_ops_8bit,
        min_memory_bytes=min_memory_bytes,
        max_memory_bytes=max_memory_bytes,
        min_memory_bandwidth=min_memory_bandwidth,
        max_memory_bandwidth=max_memory_bandwidth,
        min_h100e=min_h100e,
        max_h100e=max_h100e,
        min_cost_usd=min_cost_usd,
        max_cost_usd=max_cost_usd,
    )

    rows = []
    message = None
    
    # Chinese market cells hold True or are empty, so empty means False. Every chip has a designer.
    groups = df[GROUP_COLUMNS[group_by]].fillna(False) if group_by else None

    match operation:
        case "list":
            if group_by:
                raise ToolError("group_by works with count only, not list.")
            rows = df.head(limit).to_dict("records")
            if len(df) > limit:
                message = f"Showing {limit} of {len(df)}. Add filters or raise limit."
        case "count":
            if group_by:
                counts = df.groupby(groups).size().sort_values(ascending=False)
                rows = [{group_by: key, "count": int(n)} for key, n in counts.items()]
            else:
                rows = [{"count": len(df)}]
        case "rank":
            # Rows are chip types, not installed chips, so a total per group means nothing.
            if group_by:
                raise ToolError("group_by works with count only, not rank.")
            if not metric:
                raise ToolError(
                    "rank requires metric: 'tdp_w', 'flops_16bit', 'ops_8bit', 'memory_bytes', "
                    "'memory_bandwidth', 'h100e' or 'cost_usd'."
                )
            rows = df.nlargest(limit, METRIC_COLUMNS[metric]).to_dict("records")

    return ChipResult(rows=rows, total_rows=len(df), message=message)


def _filter_chips(
    matches: pd.DataFrame,
    chip_type: str | None,
    full_name: str | None,
    designer: str | None,
    chinese_market: bool | None,
    min_tdp_w: float | None,
    max_tdp_w: float | None,
    min_flops_16bit: float | None,
    max_flops_16bit: float | None,
    min_ops_8bit: float | None,
    max_ops_8bit: float | None,
    min_memory_bytes: float | None,
    max_memory_bytes: float | None,
    min_memory_bandwidth: float | None,
    max_memory_bandwidth: float | None,
    min_h100e: float | None,
    max_h100e: float | None,
    min_cost_usd: float | None,
    max_cost_usd: float | None,
) -> pd.DataFrame:
    """The chips that pass every filter that was set."""
    text_filter = {
        "Name": chip_type,
        "ML Hardware (linked)": full_name,
        "Designer": designer,
    }

    for column, value in text_filter.items():
        if value:
            matches = matches[matches[column].str.contains(value, case=False, regex=False)]

    # The column holds True or is empty, never False, so empty cells count as False.
    if chinese_market is not None:
        matches = matches[matches["Primarily for Chinese market"].fillna(False).astype(bool) == chinese_market]

    number_filter = [
        (METRIC_COLUMNS["tdp_w"], min_tdp_w, max_tdp_w),
        (METRIC_COLUMNS["flops_16bit"], min_flops_16bit, max_flops_16bit),
        (METRIC_COLUMNS["ops_8bit"], min_ops_8bit, max_ops_8bit),
        (METRIC_COLUMNS["memory_bytes"], min_memory_bytes, max_memory_bytes),
        (METRIC_COLUMNS["memory_bandwidth"], min_memory_bandwidth, max_memory_bandwidth),
        (METRIC_COLUMNS["h100e"], min_h100e, max_h100e),
        (METRIC_COLUMNS["cost_usd"], min_cost_usd, max_cost_usd),
    ]

    for column, low, high in number_filter:
        if low is not None:
            matches = matches[matches[column] >= low]
        if high is not None:
            matches = matches[matches[column] <= high]

    return matches
