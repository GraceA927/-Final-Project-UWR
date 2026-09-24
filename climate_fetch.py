"""
Download daily NASA POWER climate for Upper West Region, Ghana.

Purpose:
    Generate the actual daily climate time series required for the
    EMOD malaria burn-in.

Climate period:
    1981-01-01 to 2011-12-31

NASA POWER daily meteorological data are available from 1981 onward.
"""

import os
import numpy as np
import pandas as pd
import requests

# =============================================================================
# SITE
# =============================================================================

SITES = {
    "upper_west": {"name": "Upper West", "lat": 10.060074, "lon": -2.509891},
}

# =============================================================================
# CLIMATE PERIOD
# =============================================================================

# IMPORTANT:
# NASA POWER daily meteorological data begin in 1981.
#
# Do NOT use 1962 here because POWER does not provide daily meteorological
# observations for that period.

START = "19810101"
END   = "20251231"

# =============================================================================
# OUTPUT
# =============================================================================

OUTDIR = os.path.join(os.path.dirname(__file__), "climate")

# =============================================================================
# PARAMETERS REQUIRED BY EMOD
# =============================================================================

PARAMS = [
    "T2M",          # air temperature, °C
    "RH2M",         # relative humidity, %
    "PRECTOTCORR",  # precipitation, mm/day
]


# =============================================================================
# DOWNLOAD
# =============================================================================

def fetch(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={','.join(PARAMS)}"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={START}"
        f"&end={END}"
        f"&format=JSON"
        f"&time-standard=UTC"
    )

    # print("\nDownloading climate data:")
    # print(url)

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    data = response.json()
    parameters = data["properties"]["parameter"]

    df = pd.DataFrame({p: parameters[p] for p in PARAMS})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")

    # NASA POWER missing value
    df = df.replace(-999, np.nan)

    # Sort chronologically
    df = df.sort_index()

    return df.rename(columns={
        "T2M": "airtemp_C",
        "RH2M": "rh_pct",
        "PRECTOTCORR": "rain_mm",
    })


# =============================================================================
# QUALITY CONTROL
# =============================================================================

def quality_control(df):
    df = df.copy()

    # Remove impossible values
    df.loc[(df["rh_pct"] < 0) | (df["rh_pct"] > 100), "rh_pct"] = np.nan
    df.loc[df["rain_mm"] < 0, "rain_mm"] = np.nan

    # Check missingness
    missing = df.isna().sum()
    # print("\nMissing values:")
    # print(missing)

    # Interpolate only short gaps.
    # Do not blindly fill large missing periods.
    df = df.interpolate(method="time", limit=3)

    return df


# =============================================================================
# EMOD-FRIENDLY DAILY TABLE
# =============================================================================

def make_emod_daily(df):
    d = df.copy()

    d["airtemp"] = d["airtemp_C"]
    d["humidity"] = (d["rh_pct"] / 100.0).clip(lower=0, upper=1)
    d["rainfall"] = d["rain_mm"].clip(lower=0)

    d["year"] = d.index.year
    d["month"] = d.index.month
    d["day"] = d.index.day
    d["date"] = d.index.strftime("%Y-%m-%d")

    return d[["date", "year", "month", "day", "airtemp", "humidity", "rainfall"]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    for region, site in SITES.items():
        #print("\n" + "=" * 70)
        #print(f"Region: {site['name']}")
        #print(f"Period: {START} - {END}")
        #print("=" * 70)

        # Download
        df = fetch(site["lat"], site["lon"])
        # print(f"Downloaded {len(df):,} daily records.")

        # QC
        df = quality_control(df)

        # Save raw climate
        raw_file = os.path.join(OUTDIR, f"{region}_daily_raw_1981_2025.csv")
        df.to_csv(raw_file, index_label="date")


        # Create EMOD-friendly table
        emod_df = make_emod_daily(df)
        emod_file = os.path.join(OUTDIR, f"{region}_daily_EMOD_1981_2025.csv")
        emod_df.to_csv(emod_file, index=False)

        # Summary
        #print(f"\nSaved raw climate:\n  {raw_file}")
        #print(f"\nSaved EMOD climate table:\n  {emod_file}")

        #print("\nClimate summary:")
        #print(f"  Start:       {df.index.min().date()}")
        #print(f"  End:         {df.index.max().date()}")
        #print(f"  Days:        {len(df):,}")
        #print(f"  Mean temp:   {df['airtemp_C'].mean():.2f} °C")
        #print(f"  Mean RH:     {df['rh_pct'].mean():.2f} %")
        #print(f"  Annual rain: {df['rain_mm'].sum() / 31:.1f} mm/year")


if __name__ == "__main__":
    main()