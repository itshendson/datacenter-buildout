"""query_cooling_towers: find information about data center cooling towers"""

from typing import Annotated, Literal

import pandas as pd
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, mcp

# "Cooling capacity (stated, kW)" is text and sometimes a range ("685.9 - 738.6"),
# so filters and rank use Epoch's single central estimate instead.
CAPACITY_COLUMN = "Cooling capacity (central estimate, kW)"


class CoolingTowerResult(BaseModel):
    rows: list[dict] = Field(
        description=(
            "The answer. list and rank: one dict per cooling tower model with all its columns. "
            "count: [{'count': n}]. "
            "With group_by: one dict per manufacturer, e.g. {'manufacturer': 'BAC', 'count': 362}."
        )
    )
    total_rows: int = Field(
        description="Number of cooling towers that matched the filters, before limit"
    )
    message: str | None = Field(
        description="Set when results were cut off by limit, e.g. 'Showing 20 of 527.'"
    )
    notes: str = Field(
        default=(
            "Capacities are Epoch AI's central estimates; some manufacturers state a range. "
            "Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Caveats and attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def query_cooling_towers(
    model: Annotated[
        str | None,
        Field(
            description="Cooling tower model name, e.g. 'AT 456-5O26', 'XES3E'. Case-insensitive."
        ),
    ] = None,
    manufacturer: Annotated[
        str | None,
        Field(
            description="Manufacturer of the cooling tower, e.g. 'BAC', 'Evapco', 'SPX'. Case-insensitive."
        ),
    ] = None,
    min_cooling_kw: Annotated[
        float | None,
        Field(
            description="Only cooling towers with at least this much cooling capacity, in kilowatts."
        ),
    ] = None,
    max_cooling_kw: Annotated[
        float | None,
        Field(
            description="Only cooling towers with at most this much cooling capacity, in kilowatts."
        ),
    ] = None,
    operation: Annotated[
        Literal["list", "count", "rank"],
        Field(
            description=(
                "What operation to perform on the data. "
                "'list': return them. 'count': how many. "
                "'rank': sort by cooling capacity, highest first."
            )
        ),
    ] = "list",
    group_by: Annotated[
        Literal["manufacturer"] | None,
        Field(description="Split results into one row per manufacturer. Only for count."),
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=100, description="Maximum rows to return.")
    ] = 20,
    ctx: Context = None,
) -> CoolingTowerResult:
    """Find data about cooling towers by model name, manufacturer, or cooling capacity.

    Use for questions across many manufacturers or cooling towers, e.g. "what's the
    highest-capacity cooling tower", "how many cooling tower models does Evapco make",
    "which cooling towers can handle at least 10,000 kW".

    The tool's dataset cannot answer questions about data centers, such as "what cooling towers
    does Colossus 2 use?"
    """
    store: DataStore = ctx.request_context.lifespan_context

    df = _filter_cooling_towers(
        store.cooling_towers,
        model=model,
        manufacturer=manufacturer,
        min_cooling_kw=min_cooling_kw,
        max_cooling_kw=max_cooling_kw,
    )

    rows = []
    message = None

    match operation:
        case "list":
            if group_by:
                raise ToolError("group_by works with count only, not list.")
            rows = df.head(limit).to_dict("records")
            if len(df) > limit:
                message = f"Showing {limit} of {len(df)}. Add filters or raise limit."
        case "count":
            if group_by:
                counts = df.groupby("Manufacturer").size().sort_values(ascending=False)
                rows = [{group_by: key, "count": int(n)} for key, n in counts.items()]
            else:
                rows = [{"count": len(df)}]
        case "rank":
            # Rows are catalog models, not installed units, so a total per group means nothing.
            if group_by:
                raise ToolError("group_by works with count only, not rank.")
            rows = df.nlargest(limit, CAPACITY_COLUMN).to_dict("records")

    return CoolingTowerResult(rows=rows, total_rows=len(df), message=message)


def _filter_cooling_towers(
    matches: pd.DataFrame,
    model: str | None,
    manufacturer: str | None,
    min_cooling_kw: float | None,
    max_cooling_kw: float | None,
) -> pd.DataFrame:
    """The cooling towers that pass every filter that was set."""
    text_filter = {
        "Model": model,
        "Manufacturer": manufacturer,
    }

    for column, value in text_filter.items():
        if value:
            matches = matches[
                matches[column].str.contains(value, case=False, regex=False)
            ]

    if min_cooling_kw is not None:
        matches = matches[matches[CAPACITY_COLUMN] >= min_cooling_kw]
    if max_cooling_kw is not None:
        matches = matches[matches[CAPACITY_COLUMN] <= max_cooling_kw]

    return matches
