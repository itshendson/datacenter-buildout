"""query_timeline: IT power, compute and cost of data centers at a date, change between dates, and first online dates"""

from datetime import date
from typing import Annotated, Literal

import pandas as pd
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, mcp
from mcp_server.tools.lookup import _close_names

# The metric names the model can pass, and the timelines column each one reads.
METRIC_COLUMNS = {
    "power_mw": "IT power (MW)",
    "h100e": "H100 equivalents",
    "capital_cost": "Total capital cost (2025 USD billions)",
}


class TimelineResult(BaseModel):
    rows: list[dict] = Field(
        description=(
            "One dict per data center, highest first (first_online: earliest first). "
            "as_of: {'data_center', 'as_of_row', metric}. "
            "change: {'data_center', 'start', 'end', 'change'}. "
            "first_online: {'data_center', 'first_online', 'power_mw'}."
        )
    )
    total: float | None = Field(
        description="as_of: the metric summed over all matching data centers. change: the summed change. null for first_online."
    )
    total_rows: int = Field(description="Number of data centers in the answer, before limit")
    message: str | None = Field(
        description="Set when results were cut off by limit, e.g. 'Showing 20 of 96.'"
    )
    notes: str = Field(
        default=(
            "All figures are Epoch AI estimates (80% intervals: ~1.4x for power, ~1.5x for compute, "
            "~1.6x for cost, ~6 months for dates). Dates after today are Epoch's projections, not "
            "observed values. Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Caveats and attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def query_timeline(
    operation: Annotated[
        Literal["as_of", "change", "first_online"],
        Field(
            description=(
                "'as_of': the metric at a date. "
                "'change': how much the metric grew between start_date and end_date. "
                "'first_online': the first date each data center had IT power above 0."
            )
        ),
    ],
    dc_name: Annotated[
        str | None,
        Field(
            description=(
                "Full or partial name of one data center, e.g. 'Stargate Abilene', 'colossus 2'. "
                "Case-insensitive. Leave out to cover all data centers."
            )
        ),
    ] = None,
    date: Annotated[
        date | None,
        Field(description="For as_of: the date to read, e.g. '2027-12-31' for end of 2027."),
    ] = None,
    start_date: Annotated[
        date | None,
        Field(description="For change: the start, e.g. '2024-12-31' for power added during 2025."),
    ] = None,
    end_date: Annotated[
        date | None,
        Field(description="For change: the end, e.g. '2025-12-31'."),
    ] = None,
    year: Annotated[
        int | None,
        Field(description="For first_online: only data centers that first come online in this year, e.g. 2027."),
    ] = None,
    metric: Annotated[
        Literal["power_mw", "h100e", "capital_cost"],
        Field(
            description=(
                "For as_of and change: 'power_mw' (IT power), 'h100e' (compute in H100 equivalents), "
                "'capital_cost' (2025 USD billions)."
            )
        ),
    ] = "power_mw",
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum rows to return.")] = 20,
    ctx: Context = None,
) -> TimelineResult:
    """IT power, compute or capital cost of data centers over time, including Epoch's projections.

    Use for questions about dates, e.g. "Colossus 2's power by March 2027", "total power online at
    the end of 2027", "power added during 2025", "when did Abilene first come online",
    "which data centers come online in 2027". For one data center's current profile, use lookup.

    Each timeline row holds until the next one, so as_of reads the latest row on or before the date.
    For a whole year, pass the year's last day ('2027-12-31'); for change during a year, start at
    the previous year's last day.
    """
    store: DataStore = ctx.request_context.lifespan_context
    timeline = store.timelines.sort_values("Date")
    if dc_name:
        timeline = timeline[timeline["Data center"] == _match_name(dc_name, store)]

    total = None
    column = METRIC_COLUMNS[metric]

    if operation == "as_of":
        if date is None:
            raise ToolError("as_of requires date, e.g. '2027-12-31'.")
        values = _values_at(timeline, date, column)
        total = float(values[column].sum())
        rows = [
            {"data_center": name, "as_of_row": _day(row["Date"]), metric: row[column]}
            for name, row in values.sort_values(column, ascending=False).iterrows()
        ]
    elif operation == "change":
        if start_date is None or end_date is None:
            raise ToolError("change requires start_date and end_date, e.g. '2024-12-31' and '2025-12-31'.")
        if start_date >= end_date:
            raise ToolError("start_date must be before end_date.")
        # Net change, not the sum of increases: power can drop (Colossus 1 went 340 -> 268 -> 340 MW).
        start = _values_at(timeline, start_date, column)[column]
        end = _values_at(timeline, end_date, column)[column]
        both = pd.DataFrame({"start": start, "end": end}).fillna(0)
        both["change"] = both["end"] - both["start"]
        both = both[both["change"] != 0].sort_values("change", ascending=False)
        total = float(both["change"].sum())
        rows = [{"data_center": name, **values} for name, values in both.to_dict("index").items()]
    else:
        online = timeline[timeline["IT power (MW)"] > 0].groupby("Data center").first()
        if year is not None:
            online = online[online["Date"].dt.year == year]
        rows = [
            {"data_center": name, "first_online": _day(row["Date"]), "power_mw": row["IT power (MW)"]}
            for name, row in online.sort_values("Date").iterrows()
        ]
        if dc_name and not rows:
            raise ToolError(f"{dc_name} has no IT power above 0 anywhere in its timeline, including projections.")

    message = None
    if len(rows) > limit:
        message = f"Showing {limit} of {len(rows)}. total covers all of them."
    return TimelineResult(rows=rows[:limit], total=total, total_rows=len(rows), message=message)


def _match_name(dc_name: str, store: DataStore) -> str:
    """The one data center name that contains dc_name, or a ToolError saying how to fix the call."""
    names = store.data_centers["Name"]
    matches = names[names.str.contains(dc_name, case=False, regex=False)].tolist()
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ToolError(
            f"{len(matches)} data centers match '{dc_name}': {', '.join(matches)}. "
            "Call again with one of these names."
        )
    suggestions = _close_names(dc_name, names.tolist())
    if suggestions:
        raise ToolError(f"No data center matches '{dc_name}'. Did you mean: {', '.join(suggestions)}?")
    raise ToolError(f"No data center matches '{dc_name}'.")


def _values_at(timeline: pd.DataFrame, day: date, column: str) -> pd.DataFrame:
    """Each data center's latest row on or before day, indexed by name. Data centers whose timeline
    starts after day are left out (they had nothing yet). last() skips empty cells, so a blank
    value falls back to the row before it."""
    before = timeline[timeline["Date"] <= pd.Timestamp(day)]
    return before.groupby("Data center")[["Date", column]].last()


def _day(timestamp: pd.Timestamp) -> str:
    return timestamp.date().isoformat()
