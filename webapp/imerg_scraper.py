"""
Fetch the latest available day of IMERG Late rainfall for the Selangor grid and
write the two one-row CSV files that app.py reads.

This is the V07 version of the 2023 IMERG_scraper.py (kept in legacy_2023/).
The output files and their layout are unchanged so the trained models still work.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from imerg_client import fetch_day, latest_available_day  # noqa: E402

HQ_FILE = os.path.join(HERE, "HQprecipitation Data Test.csv")
CAL_FILE = os.path.join(HERE, "precipitationCal Data Test.csv")


def refresh() -> str:
    day = latest_available_day()
    row = fetch_day(day)
    pd.DataFrame({"time": [row["time"]], "HqPrecips": [row["HqPrecips"]]}).to_csv(HQ_FILE, index=False)
    pd.DataFrame({"time": [row["time"]], "PrecipCals": [row["PrecipCals"]]}).to_csv(CAL_FILE, index=False)
    return row["time"]


if __name__ == "__main__":
    print("fetched", refresh())
else:
    # Imported by app.py when its cached CSVs are stale, same behaviour as 2023.
    refresh()
