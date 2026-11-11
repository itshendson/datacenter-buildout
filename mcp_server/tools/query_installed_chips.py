"""query_installed_chips: how many chips of each type are installed across data centers at a date"""

from datetime import date
from typing import Annotated, Literal

import pandas as pd
from mcp.server.mcpserver import Context
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, get_today, mcp
from mcp_server.tools.query_timeline import _match_name

# The group_by names the model can pass, and the chip_quantities column each one reads.
GROUP_COLUMNS = {
    "chip_type": "Chip type",
    "data_center": "Data center",
}


class InstalledChipsResult(BaseModel):
    rows: list[dict] = Field(
        description=(
            "Units installed per group, highest first, e.g. {'chip_type': 'B300', 'units': 1820465}."
        )
    )
    total_units: int = Field(description="Units summed over every group, before limit")
    as_of: str = Field(description="The date the counts are read at")
    message: str | None = Field(
        description="Set when results were cut off by limit, e.g. 'Showing 20 of 45.'"
    )
    notes: str = Field(
        default=(
            "Unit counts are Epoch AI estimates. Counts after today are Epoch's projections. "
            "Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Caveats and attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def query_installed_chips(
    chip_type: Annotated[
        str | None,
        Field(description="Short chip name as query_chips lists it, e.g. 'B200', 'TPU v6e'. Case-insensitive, exact."),
    ] = None,
    dc_name: Annotated[
        str | None,
        Field(
            description=(
                "Full or partial name of one data center, e.g. 'Colossus 1'. Case-insensitive. "
                "Leave out to cover all data centers."
            )
        ),
    ] = None,
    date: Annotated[
        date | None,
        Field(description="Read the counts at this date, e.g. '2027-12-31'. Leave out for today."),
    ] = None,
    group_by: Annotated[
        Literal["chip_type", "data_center"],
        Field(description="One row per chip type, or one row per data center."),
    ] = "chip_type",
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum rows to return.")] = 20,
    ctx: Context = None,
) -> InstalledChipsResult:
    """Count installed chips by type or by data center, e.g. "how many B200s are installed across
    all data centers", "which chip type has the most units installed", "which data centers have
    the most TPU v6e", "how many H100s are at Colossus 1".

    For chip specs (TDP, memory, H100e per chip), use query_chips.
    """
    store: DataStore = ctx.request_context.lifespan_context
    day = pd.Timestamp(date) if date else get_today()

    chips = store.chip_quantities[store.chip_quantities["Date"] <= day]
    if dc_name:
        chips = chips[chips["Data center"] == _match_name(dc_name, store)]
    if chip_type:
        chips = chips[chips["Chip type"].str.lower() == chip_type.lower()]

    # Unit counts are cumulative, so each data center's latest row per chip type is what's installed.
    # Adding every row would double-count.
    installed = (
        chips.sort_values("Date")
        .groupby(["Data center", "Chip type"])["Number of Units"]
        .last()
        .reset_index()
    )
    totals = (
        installed.groupby(GROUP_COLUMNS[group_by])["Number of Units"]
        .sum()
        .sort_values(ascending=False)
    )
    rows = [{group_by: key, "units": int(n)} for key, n in totals.items()]

    message = None
    if len(rows) > limit:
        message = f"Showing {limit} of {len(rows)}. total_units covers all of them."
    return InstalledChipsResult(
        rows=rows[:limit],
        total_units=int(totals.sum()),
        as_of=day.date().isoformat(),
        message=message,
    )
