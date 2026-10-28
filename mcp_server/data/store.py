"""Load Epoch AI's data center dataset from the downloaded zip into pandas DataFrames."""

import zipfile
from pathlib import Path

import pandas as pd


class DataStore:
    """Every table in the dataset, parsed once and held in memory.

    Creating a DataStore loads all the tables. The server creates one at startup
    and every tool shares it, so tools should treat the DataFrames as read-only.
    """

    def __init__(self, zip_path: Path):
        with zipfile.ZipFile(zip_path) as zf:

            def read(name: str) -> pd.DataFrame:
                with zf.open(name) as f:
                    return pd.read_csv(f)

            self.data_centers = read("data_centers.csv")
            self.timelines = read("data_center_timelines.csv")
            self.chip_quantities = read("data_center_chip_quantities.csv")
            self.chip_types = read("chip_types.csv")
            self.chillers = read("data_center_chillers.csv")
            self.cooling_towers = read("data_center_cooling_towers.csv")

        self.timelines["Date"] = pd.to_datetime(self.timelines["Date"])
        # Values reach ~1e22, too big for int64, so pandas reads the column as text.
        # A few cells are blank; those become NaN.
        self.timelines["Performance (8-bit OP/s)"] = pd.to_numeric(
            self.timelines["Performance (8-bit OP/s)"], errors="coerce"
        ).astype("float64")
        self.chip_quantities["Date"] = pd.to_datetime(self.chip_quantities["Date"])
        self.chip_quantities["Timeline"] = pd.to_datetime(self.chip_quantities["Timeline"])
        self.chip_types["Release date"] = pd.to_datetime(self.chip_types["Release date"])


if __name__ == "__main__":
    from mcp_server.data.fetch import ZIP_PATH

    store = DataStore(ZIP_PATH)
    for name, df in vars(store).items():
        print(f"{name}: {df.shape}")
