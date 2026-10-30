"""lookup: find information on a datacenter buildout"""

from datetime import date
from difflib import SequenceMatcher
from typing import Annotated

import pandas as pd
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_server.data.store import DataStore
from mcp_server.server import READ_ONLY, mcp

CONFIDENCE_TAGS = (
    "Tagged #confident, #likely or #speculative; report the tag, and don't state "
    "#likely or #speculative as confirmed."
)


class DataCenterRecord(BaseModel):
    # Current state, from data_centers
    name: str = Field(description="Name of the datacenter")
    owner: str | None = Field(description=f"The owner of the datacenter. {CONFIDENCE_TAGS}")
    users: str | None = Field(description=f"Companies using the data center's compute. {CONFIDENCE_TAGS}")
    project: str | None = Field(description=f"Larger project it belongs to, e.g. Stargate. {CONFIDENCE_TAGS}")
    country: str = Field(description="Country where the data center is located")
    address: str | None = Field(description="Street address")
    current_power_mw: float = Field(description="Current IT power in megawatts. Estimate.")
    current_h100e: float = Field(description="Current compute in H100 equivalents. Estimate.")
    current_capital_cost_usd_billions: float = Field(description="Current total capital cost, 2025 USD billions. Estimate.")
    current_chip_types: str | None = Field(description="Chip types installed now, comma-separated")
    investors: str | None = Field(description="Investors, comma-separated")
    construction_companies: str | None = Field(description="Construction companies, comma-separated")
    energy_companies: str | None = Field(description="Energy companies, comma-separated")

    # Buildout summary, from timelines and chip_quantities
    buildout_start: date = Field(description="Earliest date in the data center's timeline")
    status_as_of: date = Field(description="Latest timeline date that isn't in the future; current figures are as of this date")
    next_milestone_date: date | None = Field(description="Next projected date when IT power changes, or null if none is projected")
    next_milestone_power_mw: float | None = Field(description="IT power in megawatts projected at the next milestone. Estimate.")
    chip_counts: dict[str, int] = Field(description="Installed units per chip type as of status_as_of, e.g. {'B300': 100800}. Estimate.")

    notes: str = Field(
        default=(
            "All figures are Epoch AI estimates (80% intervals: ~1.4x for power, ~1.5x for compute, "
            "~1.6x for cost, ~6 months for dates). Data: Epoch AI, AI Data Centers, CC BY 4.0."
        ),
        description="Caveats and attribution to include when reporting this data",
    )


@mcp.tool(annotations=READ_ONLY)
def lookup(
    dc_name: Annotated[
        str,
        Field(description=(
            "Full or partial name of one of the data centers. e.g. 'OpenAI Stargate Abilene', 'Stargate Abilene', or 'prometheus'. Case insensitive."
        ))
    ],
    ctx: Context
) -> DataCenterRecord:
    """Get the profile of one data center by name: owner, users, location, current IT power,
    compute (H100e), capital cost, chip counts, and buildout status with its next projected milestone.

    Use when the question is about a specific data center. Not for finding data centers by
    owner, country or other attributes (e.g. "which data centers does Microsoft own"): names
    don't reliably show the owner.

    If the name matches several data centers or none, the tool returns an error listing the
    matches or close spellings; call again with one of those names.
    """
    store: DataStore = ctx.request_context.lifespan_context

    names = store.data_centers["Name"]
    mask = names.str.contains(dc_name, case=False, regex=False)
    matches = store.data_centers[mask]

    if len(matches) == 0:
        suggestions = close_names(dc_name, names.tolist())
        if suggestions:
            raise ToolError(
                f"No data center matches '{dc_name}'. "
                f"Did you mean: {', '.join(suggestions)}?"
            )
        raise ToolError(f"No data center matches '{dc_name}'.")

    if len(matches) > 1:
        raise ToolError(
            f"{len(matches)} data centers match '{dc_name}': {', '.join(matches['Name'])}. "
            "Call lookup again with one of these names."
        )

    row = matches.iloc[0]
    name = row["Name"]
    today = pd.Timestamp.today().normalize()

    # Timeline rows after today are Epoch's projections.
    timeline = store.timelines[store.timelines["Data center"] == name].sort_values("Date")
    past = timeline[timeline["Date"] <= today]
    current_power = row["Current power (MW)"]
    upcoming = timeline[(timeline["Date"] > today) & (timeline["IT power (MW)"] != current_power)]
    next_milestone = upcoming.iloc[0] if len(upcoming) else None

    # Unit counts are cumulative, so the latest row per chip type is the installed total.
    chips = store.chip_quantities[
        (store.chip_quantities["Data center"] == name) & (store.chip_quantities["Date"] <= today)
    ]
    chip_counts = chips.sort_values("Date").groupby("Chip type")["Number of Units"].last()

    return DataCenterRecord(
        name=name,
        owner=cell(row, "Owner"),
        users=cell(row, "Users"),
        project=cell(row, "Project"),
        country=cell(row, "Country"),
        address=cell(row, "Address"),
        current_power_mw=current_power,
        current_h100e=cell(row, "Current H100 equivalents"),
        current_capital_cost_usd_billions=cell(row, "Current total capital cost (2025 USD billions)"),
        current_chip_types=cell(row, "Current chip types"),
        investors=cell(row, "Investors"),
        construction_companies=cell(row, "Construction companies"),
        energy_companies=cell(row, "Energy companies"),
        buildout_start=timeline["Date"].min().date(),
        status_as_of=past["Date"].max().date(),
        next_milestone_date=None if next_milestone is None else next_milestone["Date"].date(),
        next_milestone_power_mw=None if next_milestone is None else cell(next_milestone, "IT power (MW)"),
        chip_counts={chip: int(units) for chip, units in chip_counts.items()},
    )


def cell(row: pd.Series, column: str):
    """The row's value in column, or None if the cell is empty (pandas stores those as NaN)."""
    value = row[column]
    return None if pd.isna(value) else value


def close_names(query: str, names: list[str]) -> list[str]:
    """Up to 3 names that look like a misspelling of query, best first."""

    def score(name: str) -> float:
        # Compare against the full name and each word, so 'abeline' finds 'OpenAI Stargate Abilene'.
        q, name = query.lower(), name.lower()
        return max(SequenceMatcher(None, q, part).ratio() for part in [name, *name.split()])

    ranked = sorted(names, key=score, reverse=True)
    return [name for name in ranked if score(name) >= 0.7][:3]