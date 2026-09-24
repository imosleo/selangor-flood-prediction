"""
Bulk-download the training rainfall (IMERG Late daily, V07) for a date range and
append it to the two CSV files the notebooks read.

    python scripts/imerg_download.py                      # 2001-01-01 .. 2010-12-31
    python scripts/imerg_download.py --start 2021-12-01 --end 2021-12-31 --out data/dec21

Resumable: dates already present in the CSVs are skipped, so you can stop and
re-run. Dates that fail are listed at the end for a retry pass.

Output files (same names and layout as the 2023 project):
    <out>/HQprecipitation Data.csv     columns: time, HqPrecips
    <out>/precipitationCal Data.csv    columns: time, PrecipCals
where each row holds one day and a 225-value list for the 15 x 15 Selangor grid.

Expect one request per day. 2001 to 2010 is 3,652 files, so a full run takes
one to three hours depending on GES DISC.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imerg_client import EarthdataSession, fetch_day  # noqa: E402

FILES = {
    "HqPrecips": "HQprecipitation Data.csv",
    "PrecipCals": "precipitationCal Data.csv",
}


def existing_dates(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["time"][:10] for row in csv.DictReader(fh)}


def append_row(path: str, column: str, day_key: str, values: list[float]) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["time", column])
        # Stored as the printed Python list, exactly like the 2023 files.
        writer.writerow([f"{day_key} 00:00:00", str(values)])


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2001-01-01")
    ap.add_argument("--end", default="2010-12-31")
    ap.add_argument("--out", default="data", help="directory for the two CSV files")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    os.makedirs(args.out, exist_ok=True)
    paths = {col: os.path.join(args.out, name) for col, name in FILES.items()}

    done = set.intersection(*(existing_dates(p) for p in paths.values())) if paths else set()
    todo = [d for d in daterange(start, end) if d.isoformat() not in done]
    print(f"{len(done)} days already present, {len(todo)} to fetch")

    session = EarthdataSession()
    failed: list[date] = []
    for n, day in enumerate(todo, 1):
        key = day.isoformat()
        try:
            row = fetch_day(day, session)
        except Exception as exc:  # noqa: BLE001
            failed.append(day)
            print(f"[{n}/{len(todo)}] {key} FAILED: {exc}")
            continue
        for col, path in paths.items():
            append_row(path, col, key, row[col])
        print(f"[{n}/{len(todo)}] {key} ok")
        time.sleep(args.sleep)

    if failed:
        print(f"\n{len(failed)} day(s) failed, re-run to retry:")
        for d in failed:
            print("  ", d.isoformat())
        return 1
    print("\nAll days fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
