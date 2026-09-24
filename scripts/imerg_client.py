"""
Shared client for NASA GPM IMERG Late daily rainfall (V07) over the Selangor grid.

The 2023 project used IMERG V06, which NASA has since retired. This module is the
V07 replacement: same 15 x 15 grid over Selangor, same output format, new product
path and variable names.

    V06 name            V07 name
    ----------------    ----------------
    HQprecipitation  -> MWprecipitation   (microwave-only, "high quality")
    precipitationCal -> precipitation     (calibrated multi-satellite)

Authentication
    GES DISC requires an Earthdata Login. Put your credentials in ~/.netrc
    (or %USERPROFILE%\\_netrc on Windows):

        machine urs.earthdata.nasa.gov
            login    YOUR_USERNAME
            password YOUR_PASSWORD

    and approve the "NASA GESDISC DATA ARCHIVE" application in your Earthdata
    profile once. `requests` picks the netrc file up automatically.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta

import requests
import xarray as xr

BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDL.07"
PRODUCT = "3B-DAY-L.MS.MRG.3IMERG"
VERSION = "V07B"

# OPeNDAP index ranges for the Selangor box on the 0.1 degree grid.
# lon index 2806..2820 -> 100.65E .. 102.05E, lat index 925..939 -> 2.55N .. 3.95N
LON_SLICE = "[2806:2820]"
LAT_SLICE = "[925:939]"
GRID_CELLS = 15 * 15

# Output column -> candidate variable names (V07 first, V06 kept so the 2023
# sample file in data/samples still parses).
VARIABLES = {
    "HqPrecips": ["MWprecipitation", "HQprecipitation"],
    "PrecipCals": ["precipitation", "precipitationCal"],
}

# Late Run latency is about 14 hours after the end of the UTC day.
LATENCY = timedelta(days=1, hours=22)


class EarthdataSession(requests.Session):
    """requests.Session that keeps the Authorization header across the
    Earthdata Login redirect and drops it for any other host."""

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            original = requests.utils.urlparse(response.request.url).hostname
            redirect = requests.utils.urlparse(url).hostname
            if (
                original != redirect
                and redirect != self.AUTH_HOST
                and original != self.AUTH_HOST
            ):
                del headers["Authorization"]


def latest_available_day(now: datetime | None = None) -> date:
    """The most recent UTC day whose Late Run daily file should exist."""
    now = now or datetime.utcnow()
    return (now - LATENCY).date()


def build_url(day: date) -> str:
    """OPeNDAP URL that returns only the Selangor box, as NetCDF-4."""
    fname = f"{PRODUCT}.{day:%Y%m%d}-S000000-E235959.{VERSION}.nc4"
    query = (
        f"precipitation[0:0]{LON_SLICE}{LAT_SLICE},"
        f"MWprecipitation[0:0]{LON_SLICE}{LAT_SLICE},"
        f"time,lon{LON_SLICE},lat{LAT_SLICE}"
    )
    return f"{BASE_URL}/{day:%Y}/{day:%m}/{fname}.nc4?{query}"


def download(day: date, dest: str, session: requests.Session | None = None) -> str:
    """Download one day's subset to `dest`. Raises on HTTP errors."""
    session = session or EarthdataSession()
    resp = session.get(build_url(day), timeout=120)
    resp.raise_for_status()
    if not resp.content.startswith(b"\x89HDF") and not resp.content.startswith(b"CDF"):
        raise RuntimeError(
            f"{day}: response is not a NetCDF file (got {resp.content[:60]!r}). "
            "Usually this means the Earthdata login failed or the file does not exist yet."
        )
    with open(dest, "wb") as fh:
        fh.write(resp.content)
    return dest


def extract(nc4_path: str) -> dict:
    """Read a downloaded subset and return {'time': str, 'HqPrecips': [225 floats],
    'PrecipCals': [225 floats]} in the same cell order the 2023 notebooks used."""
    with xr.open_dataset(nc4_path, engine="netcdf4") as ds:
        df = ds.to_dataframe().reset_index()
        # The original loader dropped the duplicate rows created by time_bnds.
        if "nv" in df.columns:
            df = df[df["nv"] != 1.0].reset_index(drop=True)
        out = {"time": str(df["time"].iloc[0])}
        for column, candidates in VARIABLES.items():
            name = next((v for v in candidates if v in df.columns), None)
            if name is None:
                raise KeyError(f"none of {candidates} found in {nc4_path}")
            values = [float(v) for v in df[name].tolist()]
            if len(values) != GRID_CELLS:
                raise ValueError(f"{name}: expected {GRID_CELLS} cells, got {len(values)}")
            out[column] = values
    return out


def fetch_day(day: date, session: requests.Session | None = None) -> dict:
    """Download and extract one day, cleaning up the temporary file."""
    fd, tmp = tempfile.mkstemp(suffix=".nc4")
    os.close(fd)
    try:
        download(day, tmp, session)
        return extract(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
