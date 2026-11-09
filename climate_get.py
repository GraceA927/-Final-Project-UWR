"""
get_climate.py -- request Calculon weather and (optionally) edit it.

Folder layout (same as the FE project template):

    <root>/<tag>/climate/<start>-<end>/<label>/<spec>/
        dtk_15arcmin_air_temperature_daily_revised.bin (+ .json)
        dtk_15arcmin_land_temperature_daily_revised.bin
        dtk_15arcmin_rainfall_daily_revised.bin
        dtk_15arcmin_relative_humidity_daily_revised.bin

<start>/<end> are YYYYDDD (e.g. 2013001-2023365). <spec> describes the edits:
    _noleap                 Feb 29 removed -> every year is 365 days
    _noleap_fill2025        ... and any days after the last available day, up to the
                            end of 2025, filled with the PREVIOUS YEAR's data
                            (same day of year; a filled year is itself copied
                            forward, so 2024 = 2023 and 2025 = 2024 = 2023)
    _noleap_meanT28.95      ... and temperature shifted so the mean is 28.95 C
                            (seasonal cycle kept)
    _noleap_30.0degrees     ... or temperature held constant (no seasonality)
    _<n>shift               rainfall shifted n days (testing)

Why drop Feb 29: the EMOD runs use a flat 365-day calendar. Leap days would
push the seasons a day later every four years and leave the burn-in -> pickup
hand-over (day 18,250 -> day 1) out of phase.

Order of edits: drop leap days -> rain shift -> temperature edit -> fill.
The temperature edit is computed on the REAL data only, before any filling, so
the target mean is not diluted by copied years.

climate_dir() returns the same path without downloading, so the run script and
this file always agree on where the files are.
"""
import os
import glob
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from emodpy_malaria.weather import (WeatherArgs, WeatherRequest, WeatherVariable,
                                    weather_to_csv, csv_to_weather)

DAYS_PER_YEAR = 365

# Upper West Region, Ghana
UPPER_WEST = dict(node_id=1, lat=10.060074, lon=-2.509891)

WEATHER_COLUMNS = {WeatherVariable.AIR_TEMPERATURE: "airtemp",
                   WeatherVariable.LAND_TEMPERATURE: "landtemp",
                   WeatherVariable.RELATIVE_HUMIDITY: "humidity",
                   WeatherVariable.RAINFALL: "rainfall"}

WEATHER_FILENAMES = {
    WeatherVariable.AIR_TEMPERATURE: "dtk_15arcmin_air_temperature_daily_revised.bin",
    WeatherVariable.LAND_TEMPERATURE: "dtk_15arcmin_land_temperature_daily_revised.bin",
    WeatherVariable.RELATIVE_HUMIDITY: "dtk_15arcmin_relative_humidity_daily_revised.bin",
    WeatherVariable.RAINFALL: "dtk_15arcmin_rainfall_daily_revised.bin"}

VARIABLE_COLUMNS = ("airtemp", "landtemp", "humidity", "rainfall")


def validate_weather_dates(start_year, start_day, end_year, end_day):
    start = date(int(start_year), 1, 1) + timedelta(days=int(start_day) - 1)
    end = date(int(end_year), 1, 1) + timedelta(days=int(end_day) - 1)

    if end < start:
        raise ValueError("Weather end date precedes start date.")

    print(f"Weather period: {start} to {end}")


def climate_spec(fix_temp=None, target_mean_temp=None, rain_shift=None,
                 drop_leap_days=True, fill_to_year=None):
    """Sub-folder name describing the edits ('' when the files are unedited)."""
    parts = []
    if drop_leap_days:
        parts.append("noleap")
    if fill_to_year is not None:
        parts.append(f"fill{int(fill_to_year)}")
    if rain_shift is not None:
        parts.append(f"{int(rain_shift)}shift")
    if fix_temp is not None:
        parts.append(f"{float(fix_temp)}degrees")
    if target_mean_temp is not None:
        parts.append(f"meanT{float(target_mean_temp)}")
    return "_" + "_".join(parts) if parts else ""


def climate_dir(tag, start_year, end_year, start_day="001", end_day="365", label="",
                root="inputs", **spec):
    """Folder holding the (edited) weather files for this request."""
    base = os.path.join(root, tag, "climate",
                        f"{start_year}{start_day}-{end_year}{end_day}", label)
    s = climate_spec(**spec)
    return os.path.normpath(os.path.join(base, s) if s else base)


def make_site_file(path):
    """Single-node site CSV (node_id, lat, lon) if no site file was supplied."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame([UPPER_WEST]).to_csv(path, index=False)
        print(f"Created site file: {path}")
    return path


def _node_col(df):
    return "nodes" if "nodes" in df.columns else "node_id"


def _drop_feb29(df, start_year):
    """Remove Feb 29 rows and renumber steps so they stay contiguous."""
    steps = sorted(df["steps"].unique())
    first = int(steps[0])
    d0 = date(int(start_year), 1, 1)
    keep = [s for s in steps
            if not ((d0 + timedelta(days=int(s) - first)).month == 2
                    and (d0 + timedelta(days=int(s) - first)).day == 29)]
    rank = {s: first + i for i, s in enumerate(keep)}
    out = df[df["steps"].isin(keep)].copy()
    out["steps"] = out["steps"].map(rank)
    print(f"  removed {len(steps) - len(keep)} leap day(s); {len(keep)} days remain")
    return out


def _extend_previous_year(df, n_target):
    """
    Make every node's series exactly n_target days long. Missing days at the end
    take the value of the same day one year (365 days) earlier, so a missing
    year, or the missing tail of a partial year, repeats the previous year.
    """
    node_col = _node_col(df)
    var_cols = [c for c in VARIABLE_COLUMNS if c in df.columns]
    pieces = []

    for node, g in df.sort_values([node_col, "steps"]).groupby(node_col):
        g = g.reset_index(drop=True)
        n = len(g)

        if n < DAYS_PER_YEAR:
            raise ValueError(
                f"Node {node}: only {n} days of real data; at least one full "
                f"{DAYS_PER_YEAR}-day year is needed to fill later years from.")

        first = int(g["steps"].iloc[0])
        vals = {c: g[c].tolist() for c in var_cols}

        for i in range(n, n_target):
            for c in var_cols:
                vals[c].append(vals[c][i - DAYS_PER_YEAR])
        for c in var_cols:
            vals[c] = vals[c][:n_target]

        if n_target > n:
            print(f"  node {node}: {n} real days; filled {n_target - n} day(s) "
                  f"with previous-year data -> {n_target} days")
        elif n_target < n:
            print(f"  node {node}: {n} real days trimmed to {n_target}")

        pieces.append(pd.DataFrame({node_col: node,
                                    "steps": np.arange(first, first + n_target),
                                    **vals}))

    return pd.concat(pieces, ignore_index=True)


def _edit_weather(dir1, dir2, start_year, fix_temp, target_mean_temp, rain_shift,
                  drop_leap_days, strip_schema_version, fill_to_year=None):
    """Edit the downloaded weather; returns the number of days per node written."""
    if fix_temp is not None and target_mean_temp is not None:
        raise ValueError("Use fix_temp OR target_mean_temp, not both.")
    if fill_to_year is not None and not drop_leap_days:
        raise ValueError("fill_to_year needs drop_leap_days=True (flat 365-day years).")

    csv_file = os.path.join(dir1, "weather.csv")
    df, wa = weather_to_csv(weather_dir=dir1, csv_file=csv_file)
    node_col = _node_col(df)

    if drop_leap_days:
        df = _drop_feb29(df, start_year)

    if rain_shift is not None:
        # shift RAINFALL only, wrapping around, per node
        df = df.sort_values([node_col, "steps"]).reset_index(drop=True)
        df["rainfall"] = df.groupby(node_col)["rainfall"].transform(
            lambda s: np.roll(s.values, -int(rain_shift)))

    if fix_temp is not None:
        df = df.assign(airtemp=float(fix_temp), landtemp=float(fix_temp))
    elif target_mean_temp is not None:
        offset = float(target_mean_temp) - df["airtemp"].mean()
        df = df.assign(airtemp=df["airtemp"] + offset, landtemp=df["landtemp"] + offset)
        print(f"  temperature shifted {offset:+.2f} C -> mean {df['airtemp'].mean():.2f} C")

    if fill_to_year is not None:
        n_target = (int(fill_to_year) - int(start_year) + 1) * DAYS_PER_YEAR
        df = _extend_previous_year(df, n_target)
        try:
            wa.end_year = int(fill_to_year)       # keep the metadata honest
        except Exception:
            pass

    df.to_csv(csv_file)
    os.makedirs(dir2, exist_ok=True)
    ws2 = csv_to_weather(csv_data=df, attributes=wa, weather_dir=dir2,
                         weather_columns=WEATHER_COLUMNS, weather_file_names=WEATHER_FILENAMES)
    ws2.to_files(dir_path=dir2)

    if strip_schema_version:
        # legacy parser: this Eradication does not want WeatherSchemaVersion
        for f in os.listdir(dir2):
            if f.endswith(".bin.json"):
                p = os.path.join(dir2, f)
                with open(p) as fh:
                    meta = json.load(fh)
                meta.get("Metadata", {}).pop("WeatherSchemaVersion", None)
                with open(p, "w") as fh:
                    json.dump(meta, fh, indent=4)

    return int(len(df) // df[node_col].nunique())


def get_climate(tag="upper_west", label="", start_year="2015", start_day="001",
                end_year="2016", end_day="365", demo_fname="demographics.csv",
                site_file=None, root="inputs", id_reference=None,
                fix_temp=None, rain_shift=None, target_mean_temp=None,
                drop_leap_days=True, strip_schema_version=True,
                fill_to_year=None, expected_days=None):
    """
    Request Calculon weather for [start_year day start_day .. end_year day end_day],
    edit it as asked, and return the folder holding the final files.

    site_file     CSV with node_id, lat, lon (your demographics.csv works). Default:
                  <root>/<tag>/<demo_fname>, created for Upper West if missing.
    id_reference  IdReference written into the weather metadata. It MUST equal the
                  IdReference of the demographics file the run uses.
    fill_to_year  Pad the series up to the end of this year using the previous
                  year's data wherever Calculon has no data (see module docstring).
    expected_days If given, raise when the final series (per node) has a different
                  number of days. Only checked when the files are (re)built.
    """
    start = "".join((str(start_year), start_day))
    end = "".join((str(end_year), end_day))
    validate_weather_dates(start_year, start_day, end_year, end_day)
    site_file = site_file or make_site_file(os.path.join(root, tag, demo_fname))

    base = os.path.join(root, tag, "climate", f"{start}-{end}", label)
    dir1 = os.path.normpath(base)
    final = climate_dir(tag, start_year, end_year, start_day, end_day, label, root,
                        fix_temp=fix_temp, target_mean_temp=target_mean_temp,
                        rain_shift=rain_shift, drop_leap_days=drop_leap_days,
                        fill_to_year=fill_to_year)

    if os.path.isdir(final) and glob.glob(os.path.join(final, "*.bin")):
        print(f"Climate files already exist in {final}; nothing to do.")
        return final

    os.makedirs(dir1, exist_ok=True)

    if glob.glob(os.path.join(dir1, "*.bin")):
        print(f"Calculon files already downloaded in {dir1}; skipping download.")
    else:
        print(f"Requesting Calculon weather {start} .. {end} for {site_file}")
        wa = WeatherArgs(site_file=site_file, start_date=int(start), end_date=int(end),
                         node_column="node_id", id_reference=id_reference or tag)
        wr = WeatherRequest(platform="Calculon")
        wr.generate(weather_args=wa, request_name=f"{tag}_{start}_{end}")
        wr.download(local_dir=dir1)
        print(f"Original files downloaded in: {dir1}")

    if final != dir1:
        n_days = _edit_weather(dir1, final, start_year, fix_temp, target_mean_temp,
                               rain_shift, drop_leap_days, strip_schema_version,
                               fill_to_year=fill_to_year)
        print(f"Revised files written to: {final}  ({n_days} days)")
        if expected_days is not None and n_days != int(expected_days):
            raise ValueError(
                f"{final}: built {n_days} days but {expected_days} were expected. "
                f"Delete that folder after fixing the request.")
    elif expected_days is not None:
        print("  (expected_days not checked: no edits were applied)")

    return final


if __name__ == "__main__":
    # Burn-in decade (real 2003-2012, looped by EMOD) and pickup 2013-2025.
    # Calculon data ends in 2023, so 2024 and 2025 repeat the previous year.
    root = os.path.join("inputs")
    get_climate(tag="upper_west", start_year="2003", end_year="2012", end_day="366",
                root=root, target_mean_temp=28.95, expected_days=3650)
    get_climate(tag="upper_west", start_year="2013", end_year="2023", root=root,
                target_mean_temp=30.60, fill_to_year=2025, expected_days=4745)