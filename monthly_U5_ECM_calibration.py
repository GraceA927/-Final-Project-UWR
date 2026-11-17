"""
UWR seasonality calibration: rank parameter sets by how well simulated U5
case seasonality matches observed routine malaria case counts.

Simulated data
    PfPR_ClinicalIncidence_monthly_pickup.csv from the pickup analyzer, one row
    per simulation x 30-day reporting block x agebin. Parameter sets are
    (x_Temporary_Larval_Habitat, scale_constant, scale_temp_rain,
    scale_water_veg) plus the IRS tags when they are present in the CSV.
    Simulations are averaged over seeds (Run_Number) before any correlation.

Observed data
    CSV with columns Year, Month (text), Age_Group, Malaria_Cases
    (older schema year/month/conf_u5/Region is also accepted).

What changed relative to the earlier version
    * Observed loader matches the new Year/Month/Age_Group/Malaria_Cases file.
    * Simulated "report_interval" (30-day block) is accepted as well as "month";
      the incomplete 13th interval is dropped. Block k maps to calendar month k
      (block midpoints 15, 45, ... 345 fall in Jan ... Dec of the flat 365-day
      EMOD calendar), which is asserted below.
    * Monthly cases use the block length (30/365 of a year), not 1/12.
    * " U5" suffixes on column names are stripped, as in the PfPR calibration.
    * IRS tags are included in the parameter-set key when present.
    * Multi-year comparison uses the within-year SHAPE (each year divided by its
      own mean). Reported-case levels changed over time (testing/reporting) and
      the simulations carry an IRS trend, so a global min-max correlation mostly
      measures those level changes rather than seasonality.
    * Float keys are rounded before grouping, so seeds always group together.
    * Constant series are guarded against (no NaN correlations).
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")

ANALYZER_DIR = os.path.join(PROJECT_DIR, "experiments", "Selected_pickup13yr_AnalyzerOutput")
PFPR_CSV = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")
OBSERVED_CASES_PATH = os.path.join(INPUT_DIR, "monthly_U5_conf.csv")  # new-format file
OUTPUT_DIR = ANALYZER_DIR

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Under-5 bins from add_reports(): age_bins=[0.25, 5, 15, 25, 100].
U5_AGEBINS = (0.25, 5.0)

HABITAT_VARS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
IRS_VARS = ["irs_kill_scale", "irs_decay_scale", "irs_coverage_scale"]
SEED_VAR = "Run_Number"

BLOCK_DAYS = 30
N_BLOCKS = 12
DAYS_PER_YEAR = 365
MONTH_START = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151,
               7: 181, 8: 212, 9: 243, 10: 273, 11: 304, 12: 334}

CALIBRATION_YEAR = 2015            # must exist in the simulated AND observed data
COMPARE_YEAR_RANGE = (2015, 2025)  # multi-year overlap
TOP_SINGLE_YEAR = 6
TOP_MULTI_YEAR = 3
MIN_MULTI_YEAR_POINTS = 24

OBS_AGE_GROUP = "U5"
OBS_REGION = None                  # set e.g. "Upper West" only if the file has a Region column

MONTH_NAME_TO_NUMBER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

SHORT = {"scale_constant": "const", "scale_temp_rain": "rain",
         "scale_water_veg": "veg", "irs_kill_scale": "kill",
         "irs_decay_scale": "decay", "irs_coverage_scale": "cov"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_corr(a, b):
    """Pearson r, or NaN when either series is constant or too short."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(pearsonr(a, b)[0])


def combo_label(row, keys):
    parts = [f"xTLH={row[keys[0]]:.3g}"]
    parts += [f"{SHORT.get(k, k)}={row[k]:g}" for k in keys[1:]]
    return ", ".join(parts)


def select_combo(df, row, keys):
    mask = np.logical_and.reduce([np.isclose(df[k], row[k]) for k in keys])
    return df[mask]


def year_index(df, value_col, group_cols):
    """Divide each (group, year) series by its own mean; keep only full 12-month years."""
    by = list(group_cols) + ["year"]
    full = df.groupby(by)["month"].transform("nunique") == 12
    df = df[full].copy()
    mean = df.groupby(by)[value_col].transform("mean")
    df["idx"] = df[value_col] / mean.where(mean > 0)
    return df.dropna(subset=["idx"])


def add_date(df):
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df


def peak_stats(sim_clim, obs_clim):
    obs_peak = int(np.argmax(obs_clim)) + 1
    peak = int(np.argmax(sim_clim)) + 1
    diff = (peak - obs_peak + 6) % 12 - 6
    obs_range = np.ptp(obs_clim)
    amp = np.ptp(sim_clim) / obs_range if obs_range > 0 else np.nan
    return peak, diff, amp


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_u5_cases(csv_path):
    """One row per simulation x month: expected U5 cases ('Cases U5'). Returns (df, keys)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.replace(" U5", "") for c in df.columns]

    block_col = ("report_interval" if "report_interval" in df.columns
                 else "month" if "month" in df.columns else None)
    if block_col is None:
        raise ValueError("Simulation CSV needs a report_interval or month column.")

    needed = HABITAT_VARS + [SEED_VAR, "Year", block_col, "agebin", "Cases", "Pop"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing column(s) {missing}. If these are tags, re-run the "
            f"analyzer with them in SWEEP_VARIABLES. Columns present: {list(df.columns)}"
        )

    keys = list(HABITAT_VARS)
    # IRS sweep disabled: IRS tags are not used in the parameter-set key.
    # if all(c in df.columns and df[c].notna().all() for c in IRS_VARS):
    #     keys += IRS_VARS
    #     print(f"IRS tags found; parameter sets keyed on {keys}")

    df = df.rename(columns={"Year": "year", block_col: "month"})

    # Drop the incomplete interval (13th, 5-day tail).
    if "is_complete_interval" in df.columns:
        ok = (df["is_complete_interval"].astype(str).str.strip().str.lower()
              .isin(["true", "1", "t", "yes", "y"]))
        df = df[ok]
    df = df[df["month"].between(1, N_BLOCKS)]

    df = df[np.isin(np.round(df["agebin"].astype(float), 6), np.round(U5_AGEBINS, 6))].copy()
    if df.empty:
        raise ValueError(f"No rows with agebin in {U5_AGEBINS} in {csv_path}.")

    for c in ["Cases", "Pop"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Cases", "Pop"])

    for k in keys:
        df[k] = df[k].astype(float).round(6)
    df["month"] = df["month"].astype(int)

    # Blocks map one-to-one onto calendar months on the flat 365-day calendar.
    mids = [BLOCK_DAYS * (b - 0.5) for b in range(1, N_BLOCKS + 1)]
    mapped = [max(m for m, s in MONTH_START.items() if s <= d) for d in mids]
    assert mapped == list(range(1, 13)), "block -> month mapping is not one-to-one"

    # Annualized incidence per person -> expected cases in one 30-day block.
    df["Cases U5"] = df["Cases"] * df["Pop"] * BLOCK_DAYS / DAYS_PER_YEAR

    u5 = df.groupby(keys + [SEED_VAR, "year", "month"], as_index=False)["Cases U5"].sum()
    return u5, keys


def load_observed_cases(csv_path, age_group=OBS_AGE_GROUP, region=OBS_REGION):
    """Observed monthly cases with integer 'year', 'month' and a 'Cases' column."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        raise ValueError(f"{csv_path}: none of {names} found. Columns: {list(df.columns)}")

    year_c = col("year")
    month_c = col("month")
    case_c = col("Malaria_Cases", "conf_u5")

    if "age_group" in lower:
        df = df[df[lower["age_group"]].astype(str).str.upper() == age_group.upper()]
    if region is not None and "region" in lower:
        df = df[df[lower["region"]] == region]
    if df.empty:
        raise ValueError(f"No observed rows after filtering (age_group={age_group}, region={region}).")

    months = df[month_c]
    if months.dtype == object:
        unmapped = sorted(set(months) - set(MONTH_NAME_TO_NUMBER))
        if unmapped:
            raise ValueError(f"{csv_path}: unrecognised month name(s) {unmapped}.")
        months = months.map(MONTH_NAME_TO_NUMBER)

    out = pd.DataFrame({
        "year": df[year_c].astype(int).to_numpy(),
        "month": months.astype(int).to_numpy(),
        "Cases": df[case_c].astype(float).to_numpy(),
    })
    out = out.groupby(["year", "month"], as_index=False)["Cases"].sum()
    return out


def average_over_seeds(sim_df, keys, extra):
    """Mean over seeds: one row per parameter set x extra columns."""
    return sim_df.groupby(keys + extra, as_index=False)["Cases U5"].mean()


# -----------------------------------------------------------------------------
# 1. Single-year seasonality
# -----------------------------------------------------------------------------

def single_year_seasonality(sim_df, obs_df, keys, year):
    obs_years = sorted(obs_df["year"].unique())
    if year not in obs_years:
        raise ValueError(f"CALIBRATION_YEAR={year} not in observed data. Available: {obs_years}")

    sim_year = sim_df[sim_df["year"] == year]
    if sim_year.empty:
        raise ValueError(f"CALIBRATION_YEAR={year} has no rows in the simulated data.")

    sim_agg = average_over_seeds(sim_year, keys, ["month"])
    obs_year = obs_df[obs_df["year"] == year]
    merged = sim_agg.merge(obs_year, on="month", how="inner")

    rows = []
    for combo, g in merged.groupby(keys):
        if len(g) == 12 and g["Cases"].notna().all():
            g = g.sort_values("month")
            rows.append((*combo, safe_corr(g["Cases U5"], g["Cases"])))

    if not rows:
        print(f"\nNo complete 12-month combinations for {year}: merged has {len(merged)} rows.")
        return None

    ranking = (pd.DataFrame(rows, columns=keys + ["correlation"])
               .dropna(subset=["correlation"])
               .sort_values("correlation", ascending=False)
               .reset_index(drop=True))
    ranking.to_csv(os.path.join(OUTPUT_DIR, f"seasonality_ranking_{year}.csv"), index=False)

    top = ranking.head(TOP_SINGLE_YEAR)
    obs_norm = obs_year.sort_values("month").copy()
    obs_norm["norm"] = obs_norm["Cases"] / obs_norm["Cases"].max()

    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, (_, row) in zip(flat, top.iterrows()):
        sub = select_combo(merged, row, keys).sort_values("month")
        ax.plot(sub["month"], sub["Cases U5"] / sub["Cases U5"].max(),
                label="Simulated", color="blue")
        ax.plot(obs_norm["month"], obs_norm["norm"], label="Observed",
                color="black", linestyle="--")
        ax.set_title(f"Corr: {row['correlation']:.4f}\n{combo_label(row, keys)}", fontsize=9)
        ax.set_xlabel("Month")
        ax.set_ylabel("Max-normalized cases")
        ax.set_xticks(range(1, 13))
        ax.grid(True)
        ax.legend()
    for ax in flat[len(top):]:
        ax.set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"UWR_seasonality_top{len(top)}_{year}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")
    return ranking


# -----------------------------------------------------------------------------
# 2. Multi-year seasonality (within-year shape)
# -----------------------------------------------------------------------------

def multi_year_seasonality(sim_df, obs_df, keys, year_range):
    lo, hi = year_range

    obs = obs_df[obs_df["year"].between(lo, hi)].copy()
    obs = year_index(obs, "Cases", [])
    if obs.empty:
        raise ValueError(f"No complete observed years in {lo}-{hi}.")
    obs = add_date(obs)
    obs_clim = obs.groupby("month")["idx"].mean().reindex(range(1, 13)).to_numpy()
    obs_clim = obs_clim / obs_clim.mean()
    print(f"\nObserved years used: {sorted(obs['year'].unique())}")
    print("Observed seasonal profile (Jan-Dec, mean 1):", np.round(obs_clim, 2),
          f"| peak month {int(np.argmax(obs_clim)) + 1}")

    sim = sim_df[sim_df["year"].between(lo, hi)]
    sim_monthly = average_over_seeds(sim, keys, ["year", "month"])
    sim_monthly = year_index(sim_monthly, "Cases U5", keys)
    sim_monthly = add_date(sim_monthly)

    rows = []
    for combo, g in sim_monthly.groupby(keys):
        m = g.merge(obs[["year", "month", "idx"]], on=["year", "month"],
                    suffixes=("_sim", "_obs"))
        if len(m) < MIN_MULTI_YEAR_POINTS:
            continue
        r_series = safe_corr(m["idx_sim"], m["idx_obs"])

        sim_clim = g.groupby("month")["idx"].mean().reindex(range(1, 13)).to_numpy()
        if not np.isfinite(sim_clim).all():
            continue
        sim_clim = sim_clim / sim_clim.mean()
        r_clim = safe_corr(sim_clim, obs_clim)
        peak, diff, amp = peak_stats(sim_clim, obs_clim)
        rows.append((*combo, r_series, r_clim, peak, diff, amp, len(m)))

    if not rows:
        print("\nNo overlapping year-months between simulated and observed series.")
        return None

    cols = keys + ["correlation", "climatology_r", "sim_peak_month",
                   "peak_diff_months", "amplitude_ratio", "n_months"]
    ranking = (pd.DataFrame(rows, columns=cols)
               .dropna(subset=["correlation"])
               .sort_values("correlation", ascending=False)
               .reset_index(drop=True))
    ranking.to_csv(os.path.join(OUTPUT_DIR, "seasonality_ranking_multiyear.csv"), index=False)

    top = ranking.head(TOP_MULTI_YEAR)
    print(f"\nTop {len(top)} multi-year parameter sets (year-normalized series):")
    for _, row in top.iterrows():
        print(f"  {combo_label(row, keys)} | r={row['correlation']:.3f} | "
              f"clim r={row['climatology_r']:.3f} | peak diff {int(row['peak_diff_months']):+d} mo | "
              f"amp ratio={row['amplitude_ratio']:.2f} | months={int(row['n_months'])}")

    # Time series, year-normalized (1 = that year's mean).
    fig, axes = plt.subplots(len(top), 1, figsize=(14, 4.5 * len(top)),
                             sharex=True, squeeze=False)
    for ax, (_, row) in zip(axes.flatten(), top.iterrows()):
        sub = select_combo(sim_monthly, row, keys).sort_values("date")
        ax.plot(sub["date"], sub["idx"], color="blue", linewidth=2,
                label="Simulated U5 (year mean = 1)")
        ax.scatter(obs["date"], obs["idx"], facecolors="none", edgecolors="black",
                   s=40, linewidths=1.5, label="Observed U5 (year mean = 1)")
        ax.axhline(1, color="0.7", linewidth=0.6)
        ax.set_title(f"U5 seasonality {lo}-{hi}\nr={row['correlation']:.3f}, "
                     f"{combo_label(row, keys)} (months={int(row['n_months'])})", fontsize=10)
        ax.set_ylabel("Relative cases")
        ax.legend()
        ax.grid(True)
    axes.flatten()[-1].set_xlabel("Date")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"UWR_seasonality_top{len(top)}_multiyear.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")

    # Climatology (mean seasonal profile) of the same parameter sets.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, 13)
    ax.plot(x, obs_clim, "k--o", linewidth=2, label="Observed")
    for i, (_, row) in enumerate(top.iterrows()):
        g = select_combo(sim_monthly, row, keys)
        c = g.groupby("month")["idx"].mean().reindex(range(1, 13)).to_numpy()
        ax.plot(x, c / c.mean(), "-o", markersize=3,
                label=f"#{i + 1} (r={row['climatology_r']:.2f})")
    ax.set_xticks(x)
    ax.set_xlabel("Month")
    ax.set_ylabel("Relative cases (mean = 1)")
    ax.set_title(f"Mean seasonal profile {lo}-{hi}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "UWR_seasonality_climatology.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")
    return ranking


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not os.path.isfile(PFPR_CSV):
        raise FileNotFoundError(
            f"PfPR CSV does not exist:\n{PFPR_CSV}\nRun the pickup analyzer first."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_df, keys = load_u5_cases(PFPR_CSV)
    obs_df = load_observed_cases(OBSERVED_CASES_PATH)

    n_combos = sim_df[keys].drop_duplicates().shape[0]
    print(f"Parameter sets: {n_combos}  seeds: {sim_df[SEED_VAR].nunique()}  "
          f"sim years: {int(sim_df['year'].min())}-{int(sim_df['year'].max())}  "
          f"observed years: {int(obs_df['year'].min())}-{int(obs_df['year'].max())}")

    single_year_seasonality(sim_df, obs_df, keys, CALIBRATION_YEAR)
    multi_year_seasonality(sim_df, obs_df, keys, COMPARE_YEAR_RANGE)


if __name__ == "__main__":
    main()