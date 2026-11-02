"""query_chillers: find information about data center chillers"""

from typing import Annotated, Literal

import pandas as pd
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, mcp

# The chiller_type names the model can pass, and the Type value each one matches.
CHILLER_TYPES = {
    "air_cooled": "Air-cooled",
    "water_cooled": "Water-cooled",
}

# The group_by names the model can pass, and the chillers column each one reads.
GROUP_COLUMNS = {
    "manufacturer": "Manufacturer",
    "type": "Type",
}

# The metric names the model can pass, and the chillers column each one reads.
METRIC_COLUMNS = {
    "cooling_capacity_kw": "Cooling capacity (kW)",
}


class ChillerResult(BaseModel):
    rows: list[dict] = Field(
        description=(
            "The answer. list and rank: one dict per chiller model with all its columns. "
            "count: [{'count': n}]. "
            "With group_by: one dict per group, e.g. {'manufacturer': 'Daikin', 'count': 19}."
        )
    )
    total_rows: int = Field(
        description="Number of chillers that matched the filters, before limit"
    )
    message: str | None = Field(
        description="Set when results were cut off by limit, e.g. 'Showing 20 of 134.'"
    )
    notes: str = Field(
        default=(
            "All figures are Epoch AI estimates. "
            "Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def query_chillers(
    model: Annotated[
        str | None,
        Field(
            description="Chiller model name, e.g. 'ACCPS161', 'TurboChill Hydro'. Case-insensitive."
        ),
    ] = None,
    manufacturer: Annotated[
        str | None,
        Field(
            description="Manufacturer of the chillers, e.g. 'Mgreenbelt', 'Yizhong'. Case-insensitive."
        ),
    ] = None,
    chiller_type: Annotated[
        Literal["air_cooled", "water_cooled"] | None,
        Field(description="The chiller type: 'air_cooled', 'water_cooled'."),
    ] = None,
    min_cooling_kw: Annotated[
        float | None,
        Field(
            description="Only chillers with at least this much cooling capacity, in kilowatts."
        ),
    ] = None,
    max_cooling_kw: Annotated[
        float | None,
        Field(
            description="Only chillers with at most this much cooling capacity, in kilowatts."
        ),
    ] = None,
    operation: Annotated[
        Literal["list", "count", "rank"],
        Field(
            description=(
                "What operation to perform on the data. "
                "'list': return them. 'count': how many. "
                "'rank': sort by the metric, highest first."
            )
        ),
    ] = "list",
    metric: Annotated[
        Literal["cooling_capacity_kw"],
        Field(
            description="Which number to rank by: 'cooling_capacity_kw' (cooling capacity in kilowatts)."
        ),
    ] = "cooling_capacity_kw",
    group_by: Annotated[
        Literal["manufacturer", "type"] | None,
        Field(
            description=(
                "Split results into one row per manufacturer or type. Only for count."
            )
        ),
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=100, description="Maximum rows to return.")
    ] = 20,
    ctx: Context = None,
) -> ChillerResult:
    """Find data about chillers by model name, manufacturer, cooling type, or cooling capacity.

    Use for questions across many manufacturers or chillers, e.g. "what chillers are water-cooled",
    "what chillers are manufactured by Mitsubishi Heavy Industries", "which manufacturer has the
    most chiller models", "which chiller has the most cooling capacity".

    The tool's dataset cannot answer questions about data centers, such as "what chillers does Colossus 2 use?"
    """
    store: DataStore = ctx.request_context.lifespan_context
    chillers = store.chillers.drop_duplicates(subset=["Manufacturer", "Model"])

    df = _filter_chillers(
        chillers,
        model=model,
        manufacturer=manufacturer,
        chiller_type=_normalize_chiller_name(chiller_type),
        min_cooling_kw=min_cooling_kw,
        max_cooling_kw=max_cooling_kw,
    )

    rows = []
    message = None

    groups = df[GROUP_COLUMNS[group_by]].fillna("Unknown") if group_by else None

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
            # Rows are catalog models, not installed units, so a total per group means nothing.
            if group_by:
                raise ToolError("group_by works with count only, not rank.")
            rows = df.nlargest(limit, METRIC_COLUMNS[metric]).to_dict("records")

    return ChillerResult(rows=rows, total_rows=len(df), message=message)


def _normalize_chiller_name(chiller_type: str | None) -> str | None:
    return CHILLER_TYPES[chiller_type] if chiller_type else None


def _filter_chillers(
    matches: pd.DataFrame,
    model: str | None,
    manufacturer: str | None,
    chiller_type: str | None,
    min_cooling_kw: float | None,
    max_cooling_kw: float | None,
) -> pd.DataFrame:
    """The chillers that pass every filter that was set."""
    text_filter = {
        "Model": model,
        "Manufacturer": manufacturer,
        "Type": chiller_type,
    }

    for column, value in text_filter.items():
        if value:
            matches = matches[
                matches[column].str.contains(value, case=False, regex=False)
            ]

    number_filter = [
        ("Cooling capacity (kW)", min_cooling_kw, max_cooling_kw),
    ]

    for column, low, high in number_filter:
        if low is not None:
            matches = matches[matches[column] >= low]
        if high is not None:
            matches = matches[matches[column] <= high]

    return matches
