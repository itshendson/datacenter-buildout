"""query_data_centers: filter, list, count, sum and rank data centers by their attributes and metrics"""

from typing import Annotated, Literal

import pandas as pd
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, mcp

# The metric names the model can pass, and the data_centers column each one reads.
METRIC_COLUMNS = {
    "power_mw": "Current power (MW)",
    "h100e": "Current H100 equivalents",
    "capital_cost": "Current total capital cost (2025 USD billions)",
}

# The group_by names the model can pass, and the data_centers column each one reads.
GROUP_COLUMNS = {
    "owner": "Owner",
    "country": "Country",
}


class DataCenterResult(BaseModel):
    rows: list[dict] = Field(
        description=(
            "The answer. list and rank: one dict per data center with all its columns. "
            "count: [{'count': n}]. sum: [{metric: total}]. "
            "With group_by: one dict per group, e.g. {'owner': 'Google', 'power_mw': 2100}. "
            "Owner groups merge all confidence tags."
        )
    )
    total_rows: int = Field(
        description="Number of data centers that matched the filters, before limit"
    )
    message: str | None = Field(
        description="Set when results were cut off by limit, e.g. 'Showing 20 of 77.'"
    )
    notes: str = Field(
        default=(
            "All figures are Epoch AI estimates (80% intervals: ~1.4x for power, ~1.5x for compute, "
            "~1.6x for cost). Owner, Users and Project are tagged #confident, #likely or #speculative; "
            "report the tag, and don't state #likely or #speculative as confirmed. "
            "Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Caveats and attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def query_data_centers(
    owner: Annotated[
        str | None,
        Field(
            description=(
                "Company that owns the data center, e.g. 'Microsoft', 'Google', 'Oracle'. Case-insensitive."
            )
        ),
    ] = None,
    country: Annotated[
        str | None,
        Field(
            description=(
                "Country where the data center is located, e.g. 'United States', 'China'. Case-insensitive."
            )
        ),
    ] = None,
    users: Annotated[
        str | None,
        Field(
            description=(
                "Company using the compute, e.g. 'OpenAI', 'Anthropic'. Matches if it's any one of the data center's users."
            )
        ),
    ] = None,
    project: Annotated[
        str | None,
        Field(
            description=(
                "Larger project the data center belongs to, e.g. 'Stargate', 'Hyperion'."
            )
        ),
    ] = None,
    chip_type: Annotated[
        str | None,
        Field(
            description=(
                "Chip installed at the data center, e.g. 'B200', 'TPU v6e', 'H100'. Matches if it's any one of its chip types."
            )
        ),
    ] = None,
    min_power_mw: Annotated[
        float | None,
        Field(
            description=(
                "Only data centers with at least this much current IT power, in megawatts, e.g. 1000 for 1 GW."
            )
        ),
    ] = None,
    max_power_mw: Annotated[
        float | None,
        Field(
            description=(
                "Only data centers with at most this much current IT power, in megawatts."
            )
        ),
    ] = None,
    min_h100e: Annotated[
        float | None,
        Field(
            description=(
                "Only data centers with at least this much current compute, in H100 equivalents, e.g. 100000."
            )
        ),
    ] = None,
    max_h100e: Annotated[
        float | None,
        Field(
            description=(
                "Only data centers with at most this much current compute, in H100 equivalents."
            )
        ),
    ] = None,
    min_capital_cost: Annotated[
        float | None,
        Field(
            description=(
                "Only data centers with at least this total capital cost, in 2025 USD billions, e.g. 10 for $10B."
            )
        ),
    ] = None,
    max_capital_cost: Annotated[
        float | None,
        Field(
            description=(
                "Only data centers with at most this total capital cost, in 2025 USD billions."
            )
        ),
    ] = None,
    group_by: Annotated[
        Literal["owner", "country"] | None,
        Field(
            description=(
                "Split results into one row per owner or country. Only for count, sum and rank."
            )
        ),
    ] = None,
    operation: Annotated[
        Literal["list", "count", "sum", "rank"],
        Field(
            description=(
                "What to do with the matching data centers. "
                "'list': return them. 'count': how many. "
                "'sum': add up the metric. 'rank': sort by the metric, highest first. "
                "sum and rank require metric."
            )
        ),
    ] = "list",
    metric: Annotated[
        Literal["power_mw", "h100e", "capital_cost"] | None,
        Field(
            description=(
                "Which number to sum or rank by: 'power_mw' (current IT power), "
                "'h100e' (compute in H100 equivalents), 'capital_cost' (2025 USD billions). "
                "Required for sum and rank."
            )
        ),
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=100, description="Maximum rows to return.")
    ] = 20,
    ctx: Context = None,
) -> DataCenterResult:
    """Find data centers by their attributes, then list, count, sum or rank them.

    Use for questions across many data centers, e.g. "which data centers does Microsoft own",
    "how much IT power is in China", "top 5 by compute in the US", "total capital cost per owner".
    For the full profile of one named data center, use lookup instead.

    Filters are combined (all must match); leave out the ones you don't need. Then operation
    decides what to return: 'list' the data centers, 'count' them, 'sum' a metric, or 'rank'
    them by a metric. group_by splits count, sum and rank into one result per owner or country;
    data centers with no owner are grouped as 'Unknown'.
    """
    store: DataStore = ctx.request_context.lifespan_context
    df = _filter_data_centers(
        store.data_centers,
        country=country,
        owner=owner,
        users=users,
        project=project,
        chip_type=chip_type,
        min_power_mw=min_power_mw,
        max_power_mw=max_power_mw,
        min_h100e=min_h100e,
        max_h100e=max_h100e,
        min_capital_cost=min_capital_cost,
        max_capital_cost=max_capital_cost,
    )

    rows = []
    message = None

    # Rows with an empty group column (e.g. data centers with no owner) go in an "Unknown"
    # group, since pandas would otherwise drop them and grouped totals would come up short.
    # The confidence tag is stripped so "Google #confident" and "Google #speculative" land
    # in one "Google" group instead of splitting its total.
    groups = (
        df[GROUP_COLUMNS[group_by]]
        .fillna("Unknown")
        .str.replace(r"\s*#\w+$", "", regex=True)
        if group_by
        else None
    )

    # Operations on data
    if operation == "list":
        if group_by:
            raise ToolError("group_by works with count, sum and rank, not list.")
        rows = df.head(limit).to_dict("records")
        if len(df) > limit:
            message = f"Showing {limit} of {len(df)}. Add filters or raise limit."
    elif operation == "count":
        if group_by:
            counts = df.groupby(groups).size().sort_values(ascending=False)
            rows = [{group_by: key, "count": int(n)} for key, n in counts.items()]
        else:
            rows = [{"count": len(df)}]
    elif operation == "rank":
        if not metric:
            raise ToolError(
                f"{operation} requires metric: 'power_mw', 'h100e' or 'capital_cost'."
            )
        if group_by:
            totals = df.groupby(groups)[METRIC_COLUMNS[metric]].sum().nlargest(limit)
            rows = [{group_by: key, metric: value} for key, value in totals.items()]
        else:
            rows = df.nlargest(limit, METRIC_COLUMNS[metric]).to_dict("records")
    elif operation == "sum":
        if not metric:
            raise ToolError(
                f"{operation} requires metric: 'power_mw', 'h100e' or 'capital_cost'."
            )
        if group_by:
            totals = (
                df.groupby(groups)[METRIC_COLUMNS[metric]]
                .sum()
                .sort_values(ascending=False)
            )
            rows = [{group_by: key, metric: value} for key, value in totals.items()]
        else:
            rows = [{metric: df[METRIC_COLUMNS[metric]].sum()}]

    return DataCenterResult(
        rows=rows,
        total_rows=len(df),
        message=message,
    )


def _filter_data_centers(
    matches: pd.DataFrame,
    country: str | None,
    owner: str | None,
    users: str | None,
    project: str | None,
    chip_type: str | None,
    min_power_mw: float | None,
    max_power_mw: float | None,
    min_h100e: float | None,
    max_h100e: float | None,
    min_capital_cost: float | None,
    max_capital_cost: float | None,
) -> pd.DataFrame:
    """The data centers that pass every filter that was set."""
    if country:
        matches = matches[matches["Country"].str.lower() == country.lower()]

    text_filter = {
        "Owner": owner,
        "Users": users,
        "Project": project,
        "Current chip types": chip_type,
    }

    for column, value in text_filter.items():
        if value:
            matches = matches[
                matches[column].str.contains(value, case=False, regex=False)
            ]

    number_filter = [
        ("Current power (MW)", min_power_mw, max_power_mw),
        ("Current H100 equivalents", min_h100e, max_h100e),
        (
            "Current total capital cost (2025 USD billions)",
            min_capital_cost,
            max_capital_cost,
        ),
    ]

    for column, low, high in number_filter:
        if low is not None:
            matches = matches[matches[column] >= low]
        if high is not None:
            matches = matches[matches[column] <= high]

    return matches
