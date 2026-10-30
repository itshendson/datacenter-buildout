"""Build the golden dataset used to evaluate the data center agent.

Every reference answer is computed here from data/data_centers.zip, the same zip the
server loads. When the server downloads a new version, rerun this script to update the answers.

Run from the repo root:  uv run python -m evals.build_golden_dataset

Each line of golden_dataset.jsonl is one case:
    id, category, difficulty, question  -- what the user asks
    data_needed                         -- tables/columns an answer has to read; tool-agnostic,
                                           since the tools don't exist yet
    expected                            -- the reference answer and how to grade it (below)
    rubric                              -- checks for an LLM judge, on top of `expected`
    must_not                            -- wrong answers the case is designed to catch
    data_sha256, as_of                  -- which data the answer came from

`expected.type` is one of:
    number      value (+ unit), within rel_tol; `also_accept` lists other defensible values
    count       exact integer
    text        exact string, case-insensitive
    set         same items, any order
    ranked      same items, same order
    date        same date at the given granularity (day or month)
    fields      several named values, each graded as number (rel_tol) or text
    not_in_dataset  the agent should say the data doesn't cover this; no number to check
    judge       no single right value; graded by the rubric only
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from mcp_server.data.fetch import ZIP_PATH
from mcp_server.data.store import DataStore

EVALS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVALS_DIR / "golden_dataset.jsonl"

# The day the zip was downloaded. Epoch's "Current ..." columns mean "as of this date".
AS_OF = pd.Timestamp("2026-10-28")

# Only absorbs rounding. The agent reads the same CSV we do, so Epoch's ~1.4-1.6x
# estimate intervals don't belong here.
REL_TOL = 0.01

# Added to the rubric of every case that reports figures from the data.
ATTRIBUTION = "Attributes the data to Epoch AI."
ESTIMATE = "Presents the figures as estimates, not exact values."


# --- Expected-answer constructors -------------------------------------------------------


def number(value, unit, also_accept=None, rel_tol=REL_TOL):
    out = {"type": "number", "value": round(float(value), 4), "unit": unit, "rel_tol": rel_tol}
    if also_accept:
        out["also_accept"] = [round(float(v), 4) for v in also_accept]
    return out


def count(value):
    return {"type": "count", "value": int(value)}


def text(value):
    return {"type": "text", "value": value}


def set_of(values):
    return {"type": "set", "value": sorted(values)}


def ranked(values):
    return {"type": "ranked", "value": list(values)}


def date(value, granularity="day"):
    return {"type": "date", "value": pd.Timestamp(value).date().isoformat(), "granularity": granularity}


def fields(rel_tol=REL_TOL, **values):
    clean = {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in values.items()}
    return {"type": "fields", "value": clean, "rel_tol": rel_tol}


NOT_IN_DATASET = {"type": "not_in_dataset"}
JUDGE = {"type": "judge"}


# --- Helpers over the tables ------------------------------------------------------------


def split_tag(value):
    """'Meta #likely' -> ('Meta', 'likely'); 'Google' -> ('Google', None)."""
    name, _, label = str(value).partition(" #")
    return name.strip(), (label.strip() or None)


def timeline_as_of(t, when):
    """Each data center's latest timeline row on or before `when`."""
    return t[t["Date"] <= when].sort_values("Date").groupby("Data center").tail(1)


def chips_as_of(q, when):
    """Latest chip count per (data center, chip type) on or before `when`.

    A row is the installed total of that chip type at that date, not a delivery,
    so rows from different dates must not be added together.
    """
    return q[q["Date"] <= when].sort_values("Date").groupby(["Data center", "Chip type"]).tail(1)


def first_online(t):
    """Date each data center first reports non-zero IT power."""
    return t[t["IT power (MW)"] > 0].groupby("Data center")["Date"].min()


# --- Cases --------------------------------------------------------------------------------


def build_cases(s: DataStore) -> list[dict]:
    dc, t, q, ct = s.data_centers, s.timelines, s.chip_quantities, s.chip_types
    by_name = dc.set_index("Name")
    power, h100e, capex = (
        "Current power (MW)",
        "Current H100 equivalents",
        "Current total capital cost (2025 USD billions)",
    )
    owners = dc["Owner"].dropna().map(split_tag)
    dc = dc.assign(
        owner_name=owners.map(lambda o: o[0]),
        owner_label=owners.map(lambda o: o[1]),
    )
    now = timeline_as_of(t, AS_OF).set_index("Data center")
    chips_now = chips_as_of(q, AS_OF)
    online = dc[dc[power] > 0]
    cases = []

    def add(id, category, difficulty, question, data_needed, expected, rubric=(), must_not=()):
        cases.append(
            {
                "id": id,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "data_needed": list(data_needed),
                "expected": expected,
                "rubric": list(rubric),
                "must_not": list(must_not),
            }
        )

    # Single lookups

    add(
        "lookup-colossus2-it-power", "lookup", "easy",
        "How much IT power does Colossus 2 have right now?",
        ["data_centers: Current power (MW)"],
        number(by_name.at["Colossus 2", power], "MW"),
    )

    add(
        "lookup-abilene-owner-country", "lookup", "easy",
        "Who owns OpenAI Stargate Abilene, and what country is it in?",
        ["data_centers: Owner, Country"],
        fields(owner=split_tag(by_name.at["OpenAI Stargate Abilene", "Owner"])[0],
               country=by_name.at["OpenAI Stargate Abilene", "Country"]),
        must_not=["Names OpenAI as the owner because it's in the site's name."],
    )

    add(
        "lookup-pryor-chips", "lookup", "easy",
        "What chip types are currently installed at Google Pryor (North)?",
        ["data_centers: Current chip types"],
        set_of(by_name.at["Google Pryor (North)", "Current chip types"].split(",")),
    )

    add(
        "lookup-fairwater-atlanta-capex", "lookup", "easy",
        "What's the estimated capital cost of Microsoft Fairwater Atlanta?",
        ["data_centers: Current total capital cost (2025 USD billions)"],
        number(by_name.at["Microsoft Fairwater Atlanta", capex], "2025 USD billions"),
        rubric=["States the figure is in 2025 US dollars."],
    )

    water = t[(t["Data center"] == "Anthropic-Amazon New Carlisle") & t["Water use (MGD)"].notna()]
    water = water[water["Date"] <= AS_OF].sort_values("Date").iloc[-1]
    add(
        "lookup-new-carlisle-water", "lookup", "medium",
        "How much water does Anthropic-Amazon New Carlisle use?",
        ["timelines: Water use (MGD), Date"],
        number(water["Water use (MGD)"], "million gallons per day"),
        rubric=[f"Says the figure is from {water['Date'].date().isoformat()} (the latest record)."],
    )

    add(
        "trap-colossus2-facility-power", "lookup", "medium",
        "What is the total facility power of Colossus 2, including cooling and other overhead?",
        ["timelines: Power (MW), Date"],
        number(now.at["Colossus 2", "Power (MW)"], "MW"),
        must_not=[
            f"Reports {by_name.at['Colossus 2', power]:.0f} MW as the facility power; "
            "that is IT power (the data_centers 'Current power (MW)' column)."
        ],
    )

    # Filters and aggregates

    add(
        "agg-count-all", "aggregate", "easy",
        "How many data centers are in the dataset?",
        ["data_centers: one row per site"],
        count(len(dc)),
    )

    add(
        "agg-china-sites", "aggregate", "easy",
        "Which data centers in the dataset are in China?",
        ["data_centers: Country, Name"],
        set_of(dc.loc[dc["Country"] == "China", "Name"]),
    )

    add(
        "agg-not-yet-online", "aggregate", "easy",
        "How many data centers in the dataset have no IT power online yet?",
        ["data_centers: Current power (MW)"],
        count((dc[power] == 0).sum()),
        rubric=["Treats these as planned or under construction, not as missing data."],
    )

    add(
        "agg-total-h100e", "aggregate", "easy",
        "What's the total compute across all data centers right now, in H100 equivalents?",
        ["data_centers: Current H100 equivalents"],
        number(dc[h100e].sum(), "H100e"),
    )

    meta = dc[dc["owner_name"] == "Meta"]
    meta_conf = meta[meta["owner_label"] == "confident"]
    meta_uncertain = meta[meta["owner_label"] != "confident"]
    add(
        "agg-meta-it-power", "aggregate", "medium",
        "How much IT power does Meta have online today across the data centers it owns?",
        ["data_centers: Owner (with confidence tag), Current power (MW)"],
        number(meta_conf[power].sum(), "MW", also_accept=[meta[power].sum()]),
        rubric=[
            "Flags that Meta's ownership is not confirmed for "
            + ", ".join(f"{r.Name} ({r.owner_label})" for r in meta_uncertain.itertuples())
            + ", and says whether they were included."
        ],
    )

    add(
        "agg-capex-per-mw", "aggregate", "medium",
        "Across data centers that are operational, what's the average capital cost per MW of IT power?",
        ["data_centers: Current total capital cost (2025 USD billions), Current power (MW)"],
        # Epoch models cost as linear in power, so the mean of per-site ratios is the same number.
        number(online[capex].sum() / online[power].sum() * 1000, "2025 USD millions per MW"),
        rubric=["Excludes sites with zero IT power instead of dividing by zero or counting them."],
    )

    by_country = online.groupby("Country")[power].sum().sort_values(ascending=False)
    add(
        "agg-top3-countries-power", "aggregate", "medium",
        "Rank the top 3 countries by total IT power currently online.",
        ["data_centers: Country, Current power (MW)"],
        ranked(by_country.index[:3]),
        rubric=["Gives each country's total: "
                + ", ".join(f"{c} ~{v:,.0f} MW" for c, v in by_country.iloc[:3].items()) + "."],
    )

    # Rankings

    add(
        "rank-top5-it-power", "ranking", "easy",
        "Which 5 data centers have the most IT power right now?",
        ["data_centers: Name, Current power (MW)"],
        ranked(dc.nlargest(5, power)["Name"]),
    )

    add(
        "rank-top3-h100e", "ranking", "easy",
        "Which 3 data centers have the most compute right now?",
        ["data_centers: Name, Current H100 equivalents"],
        ranked(dc.nlargest(3, h100e)["Name"]),
        rubric=["Measures compute in H100 equivalents, not IT power."],
    )

    site_counts = dc["owner_name"].value_counts()
    conf_counts = dc.loc[dc["owner_label"] == "confident", "owner_name"].value_counts()
    top_owner = site_counts.index[0]
    add(
        "rank-owner-most-sites", "ranking", "easy",
        "Which company owns the most data centers in the dataset?",
        ["data_centers: Owner"],
        text(top_owner),
        rubric=[f"Gives {top_owner}'s count as {site_counts.iloc[0]} "
                f"(or {conf_counts[top_owner]} counting only confident ownership)."],
    )

    owner_power = dc.groupby("owner_name")[power].sum().sort_values(ascending=False)
    add(
        "rank-owner-most-power", "ranking", "medium",
        "Which owner has the most IT power online in total?",
        ["data_centers: Owner, Current power (MW)"],
        text(owner_power.index[0]),
        rubric=[f"Gives a total near {owner_power.iloc[0]:,.0f} MW.",
                "Groups an owner's sites together even when their confidence tags differ."],
    )

    # Timelines (fixed dates, so answers don't depend on when the eval runs)

    colossus_2027 = timeline_as_of(t, pd.Timestamp("2027-03-01")).set_index("Data center")
    add(
        "time-colossus2-early-2027", "timeline", "medium",
        "What IT power is Colossus 2 expected to have by March 1, 2027?",
        ["timelines: IT power (MW), Date"],
        number(colossus_2027.at["Colossus 2", "IT power (MW)"], "MW"),
        rubric=["Says this is a projection, not capacity that exists today."],
    )

    first = first_online(t)
    add(
        "time-abilene-first-online", "timeline", "medium",
        "When did OpenAI Stargate Abilene first bring IT power online?",
        ["timelines: IT power (MW), Date"],
        date(first["OpenAI Stargate Abilene"], granularity="month"),
    )

    add(
        "time-first-online-2027", "timeline", "hard",
        "Which data centers are projected to come online for the first time in 2027?",
        ["timelines: Data center, Date, IT power (MW)"],
        set_of(first[first.dt.year == 2027].index),
        rubric=["Says these are projections."],
        must_not=["Includes sites that were already online before 2027 and only expand in 2027."],
    )

    end_2027 = timeline_as_of(t, pd.Timestamp("2027-12-31"))
    add(
        "time-total-power-end-2027", "timeline", "hard",
        "What total IT power does the dataset project to be online across all data centers at the end of 2027?",
        ["timelines: Data center, Date, IT power (MW)"],
        number(end_2027["IT power (MW)"].sum(), "MW"),
        rubric=["Uses each site's latest figure on or before 2027-12-31.", "Says this is a projection."],
        must_not=["Adds up every timeline row, which counts each site several times."],
    )

    end_2024 = timeline_as_of(t, pd.Timestamp("2024-12-31"))["IT power (MW)"].sum()
    end_2025 = timeline_as_of(t, pd.Timestamp("2025-12-31"))["IT power (MW)"].sum()
    add(
        "time-power-added-2025", "timeline", "hard",
        "How much IT power came online across the dataset during 2025?",
        ["timelines: Data center, Date, IT power (MW)"],
        number(end_2025 - end_2024, "MW"),
        rubric=["Computes it as the total at the end of 2025 minus the total at the end of 2024."],
    )

    hyperion = t[t["Data center"] == "Meta Hyperion"].sort_values("Date")
    planned = hyperion[hyperion["IT power (MW)"] > 0].iloc[0]
    add(
        "time-hyperion-status", "timeline", "medium",
        "Is Meta Hyperion operational yet, and what's planned for it?",
        ["data_centers: Current power (MW)", "timelines: IT power (MW), Date"],
        fields(current_it_power_mw=by_name.at["Meta Hyperion", power],
               planned_it_power_mw=planned["IT power (MW)"],
               planned_date=planned["Date"].date().isoformat()),
        rubric=["Says it is not operational yet (0 MW today)."],
    )

    # Chips

    colossus1_h100 = q[(q["Data center"] == "Colossus 1") & (q["Chip type"] == "H100")]
    add(
        "chips-colossus1-h100", "chips", "medium",
        "How many H100s are at Colossus 1?",
        ["chip_quantities: Data center, Chip type, Number of Units, Date"],
        count(chips_now.loc[(chips_now["Data center"] == "Colossus 1")
                            & (chips_now["Chip type"] == "H100"), "Number of Units"].sum()),
        must_not=[f"Adds up the H100 rows from different dates ({colossus1_h100['Number of Units'].sum():,}); "
                  "each row is the installed total at that date."],
    )

    nc = chips_now[(chips_now["Data center"] == "Anthropic-Amazon New Carlisle")
                   & (chips_now["Chip type"] == "Trainium2")].iloc[0]
    add(
        "chips-new-carlisle-trainium2", "chips", "medium",
        "How many Trainium2 chips are at Anthropic-Amazon New Carlisle?",
        ["chip_quantities: Number of Units, Number of Units source, Date"],
        count(nc["Number of Units"]),
        rubric=[f"Says the count is '{nc['Number of Units source']}' per the dataset."],
    )

    b200 = chips_now[chips_now["Chip type"] == "B200"]
    add(
        "chips-b200-total", "chips", "hard",
        "How many B200 chips are installed across all data centers, according to the chip records?",
        ["chip_quantities: Data center, Chip type, Number of Units, Date"],
        number(b200["Number of Units"].sum(), "chips"),
        rubric=[f"Uses each site's latest B200 count on or before {AS_OF.date().isoformat()}, not future rows."],
        must_not=[f"Adds every B200 row ({q.loc[q['Chip type'] == 'B200', 'Number of Units'].sum():,}), "
                  "which double-counts and includes projections."],
    )

    b300 = ct.set_index("Name").loc["B300"]
    add(
        "chips-b300-spec", "chips", "easy",
        "What's the power draw (TDP) of a B300, and how many H100 equivalents is one B300 worth?",
        ["chip_types: TDP (W) (from ML Hardware (linked)), H100e"],
        fields(tdp_w=b300["TDP (W) (from ML Hardware (linked))"], h100e=b300["H100e"]),
    )

    units_by_type = chips_now.groupby("Chip type")["Number of Units"].sum().sort_values(ascending=False)
    add(
        "chips-most-installed-type", "chips", "hard",
        "Which chip type has the most units installed across all data centers today?",
        ["chip_quantities: Chip type, Number of Units, Date"],
        text(units_by_type.index[0]),
        rubric=[f"Gives a count near {units_by_type.iloc[0]:,}."],
    )

    # User affiliations and confidence labels

    anthropic = dc[dc["Users"].fillna("").str.contains("Anthropic")]
    labels = {r.Name: next(split_tag(u)[1] for u in r.Users.split(", ") if u.startswith("Anthropic"))
              for r in anthropic.itertuples()}
    add(
        "affil-anthropic-sites", "affiliation", "medium",
        "Which data centers does Anthropic use?",
        ["data_centers: Users (with confidence tags)"],
        set_of(labels),
        rubric=["Gives each site's confidence label: "
                + "; ".join(f"{n} ({lab})" for n, lab in sorted(labels.items())) + ".",
                "Doesn't present 'likely' or 'speculative' sites as confirmed."],
    )

    denton_users = dict(split_tag(u) for u in by_name.at["CoreWeave Denton TX", "Users"].split(", "))
    add(
        "affil-denton-openai", "affiliation", "easy",
        "Is OpenAI a confirmed user of CoreWeave Denton TX?",
        ["data_centers: Users (with confidence tags)"],
        text(denton_users["OpenAI"]),
        rubric=[f"Says no: the dataset marks OpenAI as a '{denton_users['OpenAI']}' user, not confirmed."],
    )

    add(
        "affil-dayone-owner-missing", "affiliation", "medium",
        "Who owns DayOne Nusajaya?",
        ["data_centers: Owner"],
        NOT_IN_DATASET,
        rubric=["Says the dataset doesn't list an owner for this site.",
                "If it mentions DayOne, makes clear that's inferred from the name, not from the data."],
    )

    # Cooling equipment (catalogs of models, not linked to any data center)

    makers = s.chillers["Manufacturer"].value_counts()
    add(
        "cooling-chiller-maker", "cooling", "easy",
        "Which manufacturer has the most chiller models in the dataset?",
        ["chillers: Manufacturer, Model"],
        text(makers.index[0]),
        rubric=[f"Gives the count as {makers.iloc[0]}."],
    )

    tower = s.cooling_towers.loc[s.cooling_towers["Cooling capacity (central estimate, kW)"].idxmax()]
    add(
        "cooling-biggest-tower", "cooling", "easy",
        "What's the highest-capacity cooling tower in the dataset?",
        ["cooling_towers: Model, Manufacturer, Cooling capacity (central estimate, kW)"],
        fields(model=tower["Model"], manufacturer=tower["Manufacturer"],
               capacity_kw=tower["Cooling capacity (central estimate, kW)"]),
    )

    # Out of scope: the agent should say the data doesn't cover it, not fill the gap

    add(
        "neg-colossus2-chillers", "out_of_scope", "medium",
        "What chillers does Colossus 2 use?",
        ["chillers / cooling_towers: no Data center column"],
        NOT_IN_DATASET,
        rubric=["Explains the cooling tables are equipment catalogs, not linked to specific data centers."],
    )

    energy = by_name.at["Meta Prometheus", "Energy companies"]
    add(
        "neg-prometheus-electricity-price", "out_of_scope", "easy",
        "What does Meta Prometheus pay for electricity?",
        ["none: the dataset has no electricity prices"],
        NOT_IN_DATASET,
        rubric=["Doesn't state or estimate a price."]
        + ([f"May mention the listed energy companies ({energy})."] if isinstance(energy, str) else []),
    )

    stargate = dc.loc[dc["Name"].str.contains("Stargate"), "Name"]
    add(
        "neg-stargate-tokyo", "out_of_scope", "easy",
        "How much IT power does the Stargate Tokyo data center have?",
        ["data_centers: Name"],
        NOT_IN_DATASET,
        rubric=["Says there's no Stargate Tokyo in the dataset.",
                "May list the Stargate sites that are in it: " + ", ".join(sorted(stargate)) + "."],
    )

    msft_capex = dc.loc[dc["owner_name"] == "Microsoft", capex].sum()
    add(
        "neg-microsoft-capex-budget", "out_of_scope", "medium",
        "What is Microsoft's total AI capex budget for 2027?",
        ["none: the dataset is per site, with no company budgets"],
        NOT_IN_DATASET,
        rubric=["Says the dataset doesn't cover company budgets.",
                f"If it offers the summed capital cost of Microsoft-owned sites (~${msft_capex:.1f}B), "
                "labels it as a different measure, not the budget."],
    )

    # Ambiguous: no single right answer

    top_power = dc.loc[dc[power].idxmax()]
    top_compute = dc.loc[dc[h100e].idxmax()]
    add(
        "ambig-biggest-data-center", "ambiguous", "medium",
        "What's the biggest AI data center?",
        ["data_centers: Current power (MW), Current H100 equivalents, Current total capital cost"],
        JUDGE,
        rubric=["Either asks which measure the user means, or says which measure it used.",
                f"By IT power: {top_power['Name']} (~{top_power[power]:,.0f} MW). "
                f"By compute: {top_compute['Name']} (~{top_compute[h100e]:,.0f} H100e)."],
    )

    # Units and multi-step comparisons

    google = dc[dc["owner_name"] == "Google"]
    google_conf = google[google["owner_label"] == "confident"]
    add(
        "unit-google-gw", "aggregate", "medium",
        "How many gigawatts of IT power does Google have online?",
        ["data_centers: Owner, Current power (MW)"],
        number(google[power].sum() / 1000, "GW", also_accept=[google_conf[power].sum() / 1000]),
        rubric=["Gives the answer in GW, not MW.",
                "Mentions any Google sites whose ownership isn't tagged confident, and whether they were counted."],
    )

    msft = dc.loc[dc["owner_name"] == "Microsoft", h100e].sum()
    meta_h = dc.loc[dc["owner_name"] == "Meta", h100e].sum()
    leader = "Meta" if meta_h > msft else "Microsoft"
    add(
        "multi-microsoft-vs-meta-compute", "multi_step", "hard",
        "Compare Microsoft and Meta: which has more compute online today, and by how much?",
        ["data_centers: Owner, Current H100 equivalents"],
        fields(leader=leader, difference_h100e=abs(meta_h - msft), rel_tol=0.05),
        rubric=["Says it compared by owner (so Microsoft-Nebius New Jersey counts for Nebius), "
                "or explains how it decided which sites belong to each company.",
                f"Gives totals near Meta ~{meta_h:,.0f} and Microsoft ~{msft:,.0f} H100e."],
    )

    return cases


def main() -> None:
    sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    store = DataStore(ZIP_PATH)
    cases = build_cases(store)

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for case in cases:
            if case["expected"]["type"] != "not_in_dataset":
                case["rubric"].append(ATTRIBUTION)
            if case["expected"]["type"] in ("number", "fields"):
                case["rubric"].append(ESTIMATE)
            case["data_sha256"] = sha
            case["as_of"] = AS_OF.date().isoformat()
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
