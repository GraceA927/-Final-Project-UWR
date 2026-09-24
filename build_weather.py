"""
Build EMOD malaria weather files for the Upper West Region -- SINGLE NODE.

This model represents the entire Upper West Region as ONE EMOD node
(node_id = 1). This script does NOT use the 11-district structure from
the original FE-2026-examples template and does NOT loop over districts.

This pipeline creates TWO weather sets from the project's existing daily
NASA POWER data:

1. BURN-IN WEATHER
   - 50 years: 1963-2012 (18,250 daily records)
   - Built by repeating a 365-day climatology
   - The climatology is DERIVED here from the existing daily EMOD climate
     file (upper_west_daily_EMOD_1981_2025.csv), by averaging each
     calendar day (month, day) across all available years and dropping
     Feb 29. There is no pre-existing separate climatology file in this
     project, so this derivation step is new -- not pulled from an
     existing file.
   - Preserves seasonal rainfall/temperature/humidity patterns.
   - Does NOT preserve year-to-year weather variability (expected/accepted
     for the burn-in).

2. PICKUP WEATHER
   - 2013-2025 inclusive (4,745 daily records after removing leap days:
     2016, 2020, 2024)
   - Uses the actual year-specific daily NASA POWER data already present
     in upper_west_daily_EMOD_1981_2025.csv (no fetch-script change is
     needed: that file already starts in 1981, well before 2013).
   - Preserves both seasonal and interannual weather variation.

LAND TEMPERATURE:
    The existing NASA POWER pipeline in this project does not download a
    separate land-surface-temperature variable. As in the rest of this
    project's scripts, land temperature falls back to air temperature.
    This is called out explicitly (printed at runtime) rather than
    silently assumed.

Prerequisite:
    Run climate_fetch.py first to produce:
        climate/upper_west_daily_EMOD_1981_2025.csv

Output:
    climate/burnin/emod_weather_upper_west/
    climate/pickup/emod_weather_upper_west/
"""

import os
import json
import numpy as np
import pandas as pd

from emodpy_malaria.weather import (
    csv_to_weather,
    WeatherVariable,
    WeatherAttributes,
)

# ============================================================
# SINGLE-NODE MODEL IDENTITY
# ============================================================

NODE_ID = 1
SITE_NAME = "UpperWest_Region"
LATITUDE = 10.060074
LONGITUDE = -2.509891

REGION_KEY = "upper_west"  # matches climate_fetch.py's filename convention

# ============================================================
# CONFIGURATION
# ============================================================

# 50-year burn-in
BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_YEARS = BURNIN_END_YEAR - BURNIN_START_YEAR + 1  # 50

# Historical pickup
PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025

DAYS_PER_YEAR = 365  # 365-day convention; leap days are dropped, not added

DEMOG_ID_REF = "Gridded world grump2.5arcmin"  # must match demographics IdReference

CLIM_DIR = os.path.join(os.path.dirname(__file__), "climate")

# Existing daily EMOD climate file produced by climate_fetch.py.
# Columns: date, year, month, day, airtemp, humidity, rainfall
INPUT_FILE = os.path.join(CLIM_DIR, f"{REGION_KEY}_daily_EMOD_1981_2025.csv")

BURNIN_OUTPUT_DIR = os.path.join(CLIM_DIR, "burnin", f"emod_weather_{REGION_KEY}")
PICKUP_OUTPUT_DIR = os.path.join(CLIM_DIR, "pickup", f"emod_weather_{REGION_KEY}")


# ============================================================
# WEATHER FILE NAMES
# ============================================================

FILE_NAMES = {
    WeatherVariable.AIR_TEMPERATURE: "air_temperature.bin",
    WeatherVariable.RELATIVE_HUMIDITY: "relative_humidity.bin",
    WeatherVariable.RAINFALL: "rainfall.bin",
    WeatherVariable.LAND_TEMPERATURE: "land_temperature.bin",
}


# ============================================================
# HELPERS
# ============================================================

def remove_weather_schema_version(weather_dir):
    """
    Strip WeatherSchemaVersion from generated metadata so this Eradication
    uses the legacy parser -- matches the convention already used by
    build_weather.py elsewhere in this project.
    """
    for filename in os.listdir(weather_dir):
        if not filename.endswith(".bin.json"):
            continue

        path = os.path.join(weather_dir, filename)

        with open(path, "r") as f:
            meta = json.load(f)

        meta.get("Metadata", {}).pop("WeatherSchemaVersion", None)

        with open(path, "w") as f:
            json.dump(meta, f, indent=4)


def create_weather_attributes(start_year, end_year, provenance):
    return WeatherAttributes(
        reference=DEMOG_ID_REF,
        start_year=start_year,
        end_year=end_year,
        start_doy=1,
        lat_min=LATITUDE,
        lat_max=LATITUDE,
        lon_min=LONGITUDE,
        lon_max=LONGITUDE,
        provenance=provenance,
        resolution="daily",
        update_freq="CLIMATE_UPDATE_DAY",
    )


def build_emod_dataframe(airtemp, humidity, rainfall):
    """
    Assemble the EMOD-ready dataframe for a given set of daily arrays.
    Land temperature falls back to air temperature (see module docstring).
    """
    n = len(airtemp)

    assert len(humidity) == n, f"humidity length {len(humidity)} != airtemp length {n}"
    assert len(rainfall) == n, f"rainfall length {len(rainfall)} != airtemp length {n}"

    landtemp = airtemp.copy()

    df = pd.DataFrame({
        "nodes": NODE_ID,
        "steps": np.arange(n),
        "airtemp": airtemp,
        "humidity": humidity,
        "rainfall": rainfall,
        "landtemp": landtemp,
    })

    assert (df["nodes"] == NODE_ID).all(), "node_id must be 1 for every record"
    assert df["nodes"].nunique() == 1, "found more than one node_id -- this is a single-node model"

    # All four variables must have identical length by construction, but
    # verify explicitly per the requested validation checks.
    lengths = {
        "airtemp": len(df["airtemp"]),
        "humidity": len(df["humidity"]),
        "rainfall": len(df["rainfall"]),
        "landtemp": len(df["landtemp"]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Weather variables have mismatched lengths: {lengths}")

    required_cols = ["airtemp", "humidity", "rainfall", "landtemp"]
    missing_counts = df[required_cols].isna().sum()
    if missing_counts.any():
        raise ValueError(f"Missing climate values before writing weather files:\n{missing_counts}")

    return df, n


def write_emod_weather(df, weather_dir, start_year, end_year, provenance):
    os.makedirs(weather_dir, exist_ok=True)

    attrs = create_weather_attributes(start_year, end_year, provenance)

    csv_to_weather(
        csv_data=df,
        node_column="nodes",
        step_column="steps",
        weather_columns={
            WeatherVariable.AIR_TEMPERATURE: "airtemp",
            WeatherVariable.RELATIVE_HUMIDITY: "humidity",
            WeatherVariable.RAINFALL: "rainfall",
            WeatherVariable.LAND_TEMPERATURE: "landtemp",
        },
        attributes=attrs,
        weather_dir=weather_dir,
        weather_file_names=FILE_NAMES,
    )

    remove_weather_schema_version(weather_dir)


# ============================================================
# LOAD EXISTING DAILY CLIMATE DATA
# ============================================================

def load_daily_climate():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            f"Climate file not found:\n{INPUT_FILE}\n\n"
            "Run climate_fetch.py first to produce this file."
        )

    df = pd.read_csv(INPUT_FILE)

    required = ["date", "year", "month", "day", "airtemp", "humidity", "rainfall"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_FILE} is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# BUILD CLIMATOLOGY (derived -- see module docstring)
# ============================================================

def build_climatology(daily_df):
    """
    Derive a 365-day climatology by averaging each (month, day) across all
    available years in daily_df, dropping Feb 29. This project has no
    pre-existing climatology file, so this is computed here rather than
    read from disk.
    """
    df = daily_df[~((daily_df["month"] == 2) & (daily_df["day"] == 29))].copy()

    clim = (
        df.groupby(["month", "day"], as_index=False)[["airtemp", "humidity", "rainfall"]]
        .mean()
        .sort_values(["month", "day"])
        .reset_index(drop=True)
    )

    if len(clim) != 365:
        raise ValueError(
            f"Derived climatology has {len(clim)} days after removing Feb 29; expected 365. "
            "Check the input data for gaps in calendar-day coverage."
        )

    return clim


# ============================================================
# BUILD BURN-IN WEATHER (1963-2012, tiled climatology)
# ============================================================

def build_burnin_weather(clim):
    print(f"\n[1/2] Building {BURNIN_YEARS}-year burn-in weather "
          f"({BURNIN_START_YEAR}-{BURNIN_END_YEAR})...")
    print("      Method: 365-day climatology repeated for all 50 years.")
    print("      Land temperature: falls back to air temperature (no separate "
          "land-temperature variable in the source data).")

    airtemp = np.tile(clim["airtemp"].to_numpy(), BURNIN_YEARS)
    humidity = np.tile(clim["humidity"].to_numpy(), BURNIN_YEARS)
    rainfall = np.tile(clim["rainfall"].to_numpy(), BURNIN_YEARS)

    expected_days = BURNIN_YEARS * DAYS_PER_YEAR  # 18,250
    if len(airtemp) != expected_days:
        raise ValueError(
            f"Burn-in weather contains {len(airtemp)} days; expected {expected_days}."
        )

    df, n = build_emod_dataframe(airtemp, humidity, rainfall)

    print(f"      Coverage: {BURNIN_START_YEAR}-{BURNIN_END_YEAR} "
          f"({BURNIN_YEARS} years, no interannual variability)")
    print(f"      Records:  {n:,} (expected {expected_days:,})")

    provenance = (
        "NASA POWER-derived 365-day climatology (averaged 1981-2025 daily "
        f"data by calendar day) repeated for a {BURNIN_YEARS}-year EMOD "
        f"burn-in; {SITE_NAME}; burn-in period {BURNIN_START_YEAR}-{BURNIN_END_YEAR}; "
        "land temperature = air temperature fallback."
    )

    write_emod_weather(
        df, BURNIN_OUTPUT_DIR,
        start_year=BURNIN_START_YEAR, end_year=BURNIN_END_YEAR,
        provenance=provenance,
    )

    return BURNIN_OUTPUT_DIR, n


# ============================================================
# BUILD PICKUP WEATHER (2013-2025, year-specific)
# ============================================================

def build_pickup_weather(daily_df):
    print(f"\n[2/2] Building historical pickup weather "
          f"({PICKUP_START_YEAR}-{PICKUP_END_YEAR})...")
    print("      Method: actual year-specific daily NASA POWER data "
          "(no climatology, no tiling).")
    print("      Land temperature: falls back to air temperature (no separate "
          "land-temperature variable in the source data).")

    start_date = pd.Timestamp(f"{PICKUP_START_YEAR}-01-01")
    end_date = pd.Timestamp(f"{PICKUP_END_YEAR}-12-31")

    pickup = daily_df[
        (daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)
    ].copy()

    # Remove leap days so the pickup uses the same 365-day-per-year
    # convention as the burn-in. 2013-2025 contains leap days in
    # 2016, 2020, and 2024.
    pickup = pickup[~((pickup["month"] == 2) & (pickup["day"] == 29))].copy()
    pickup = pickup.sort_values("date").reset_index(drop=True)

    pickup_years = PICKUP_END_YEAR - PICKUP_START_YEAR + 1  # 13
    expected_days = pickup_years * DAYS_PER_YEAR  # 4,745

    if len(pickup) != expected_days:
        raise ValueError(
            f"Pickup weather has {len(pickup)} usable days, but "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR} requires {expected_days} "
            f"days after removing leap days.\n"
            f"Check that {INPUT_FILE} actually covers this full period "
            f"with no gaps."
        )

    df, n = build_emod_dataframe(
        pickup["airtemp"].to_numpy(),
        pickup["humidity"].to_numpy(),
        pickup["rainfall"].to_numpy(),
    )

    print(f"      Coverage: {pickup['date'].min().date()} to {pickup['date'].max().date()} "
          f"({pickup_years} years, leap days removed)")
    print(f"      Records:  {n:,} (expected {expected_days:,})")

    provenance = (
        f"NASA POWER daily historical weather; {SITE_NAME}; "
        f"pickup period {PICKUP_START_YEAR}-{PICKUP_END_YEAR}; "
        "leap days removed; land temperature = air temperature fallback."
    )

    write_emod_weather(
        df, PICKUP_OUTPUT_DIR,
        start_year=PICKUP_START_YEAR, end_year=PICKUP_END_YEAR,
        provenance=provenance,
    )

    return PICKUP_OUTPUT_DIR, n


# ============================================================
# MAIN
# ============================================================

def main():
    #print("=" * 80)
    #print("UPPER WEST REGION EMOD WEATHER BUILDER (single node)")
    #print("=" * 80)
    #print(f"Node ID    : {NODE_ID}")
    #print(f"Site name  : {SITE_NAME}")
    #print(f"Lat / Lon  : {LATITUDE}, {LONGITUDE}")
    #print(f"Input file : {INPUT_FILE}")
    #print(f"\nBurn-in : {BURNIN_START_YEAR}-{BURNIN_END_YEAR} ({BURNIN_YEARS} years, tiled climatology)")
    #print(f"Pickup  : {PICKUP_START_YEAR}-{PICKUP_END_YEAR} (year-specific daily data)")

    daily_df = load_daily_climate()

    data_start = daily_df["date"].min().date()
    data_end = daily_df["date"].max().date()
    print(f"\nLoaded {len(daily_df):,} daily records from {data_start} to {data_end}")

    if pd.Timestamp(f"{PICKUP_END_YEAR}-12-31") > daily_df["date"].max():
        raise ValueError(
            f"Input data end at {data_end}, but the pickup period requires "
            f"coverage through {PICKUP_END_YEAR}-12-31. Re-run climate_fetch.py "
            f"with an END date of at least {PICKUP_END_YEAR}1231."
        )

    clim = build_climatology(daily_df)
    print(f"Derived climatology: {len(clim)} days (validated)")

    burnin_dir, burnin_days = build_burnin_weather(clim)
    pickup_dir, pickup_days = build_pickup_weather(daily_df)

    #print("\n" + "=" * 80)
    #print("WEATHER BUILD COMPLETE")
    #print("=" * 80)
    #print(f"Burn-in : {burnin_days:,} days -> {burnin_dir}")
    #print(f"Pickup  : {pickup_days:,} days -> {pickup_dir}")


if __name__ == "__main__":
    main()