"""
UWR seasonality calibration: rank habitat combinations by how well simulated
U5 case seasonality correlates with observed routine case counts.

Simulated data
    PfPR_ClinicalIncidence_monthly_pickup.csv from MonthlyPfPRAnalyzer
    (uwr_pickup_analyzer.py), one row per simulation x month x agebin.
    Under Option B each pickup simulation inherits its burn-in row's habitat
    combination, so the ranking is over (x_Temporary_Larval_Habitat,
    scale_constant, scale_temp_rain, scale_water_veg), averaged over seeds
    (Run_Number) before any correlation is computed.

Observed data
    inputs/monthly_u5_cases.csv: Year, Month (text, "January"...), Age_Group,
    Malaria_Cases. Months are converted to integers; years start in 2015.

Two analyses
    1. Single year (CALIBRATION_YEAR): correlate the 12-month profile; plot top 6.
    2. Multi-year (COMPARE_YEAR_RANGE): correlate the full monthly series; plot top 3.

Simulated U5 cases
    The analyzer's "Cases" column is "Annual Clinical Incidence by Age Bin":
    a per-person annualized rate, not a count. Monthly expected cases are
    therefore computed as incidence x Pop / 12 per bin, then summed over the
    two U5 bins. (Summing raw incidences, as the earlier version did, would
    weight the 0-3 month bin as heavily as the 3 month-5 year bin.)
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

# Analyzer output folder: <job_directory>/<experiment>_output/
ANALYZER_DIR = os.path.join(PROJECT_DIR, "experiments", "UWR_Pickup_ECM_output")
PFPR_CSV = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")
OBSERVED_CASES_PATH = os.path.join(INPUT_DIR, "monthly_u5_cases.csv")
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
SEED_VAR = "Run_Number"

CALIBRATION_YEAR = 2015            # must exist in the simulated AND observed data
COMPARE_YEAR_RANGE = (2015, 2025)  # multi-year overlap
TOP_SINGLE_YEAR = 6
TOP_MULTI_YEAR = 3

MONTH_NAME_TO_NUMBER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_u5_cases(csv_path):
    """One row per simulation x month: expected U5 cases ('Cases U5')."""
    df = pd.read_csv(csv_path)

    missing = [c for c in HABITAT_VARS + [SEED_VAR, "Year", "month", "agebin", "Cases", "Pop"]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing column(s) {missing}. If these are tags, re-run the "
            f"analyzer with them in SWEEP_VARIABLES. Columns present: {list(df.columns)}"
        )

    df = df[df["agebin"].isin(U5_AGEBINS)].copy()
    if df.empty:
        raise ValueError(f"No rows with agebin in {U5_AGEBINS} in {csv_path}.")

    # Annualized incidence per person -> expected cases in the month.
    df["Cases U5"] = df["Cases"] * df["Pop"] / 12.0

    keys = HABITAT_VARS + [SEED_VAR, "Year", "month"]
    u5 = df.groupby(keys, as_index=False)["Cases U5"].sum()
    return u5.rename(columns={"Year": "year"})


def load_observed_cases(csv_path, age_group="U5"):
    """Observed monthly cases with integer 'year' and 'month' and a 'Cases' column."""
    df = pd.read_csv(csv_path)
    df = df[df["Age_Group"] == age_group].copy()
    if df.empty:
        raise ValueError(f"No rows with Age_Group == '{age_group}' in {csv_path}.")

    unmapped = sorted(set(df["Month"]) - set(MONTH_NAME_TO_NUMBER))
    if unmapped:
        raise ValueError(f"{csv_path}: unrecognised month name(s) {unmapped}.")

    df["month"] = df["Month"].map(MONTH_NAME_TO_NUMBER)
    df["year"] = df["Year"].astype(int)
    df["Cases"] = df["Malaria_Cases"].astype(float)
    return df[["year", "month", "Cases"]].reset_index(drop=True)


def average_over_seeds(sim_df, extra_keys):
    """Mean over seeds: one row per habitat combination x extra_keys."""
    return sim_df.groupby(HABITAT_VARS + extra_keys, as_index=False)["Cases U5"].mean()


def combo_label(values):
    parts = [f"xTLH={values[0]:.3g}"] + [
        f"{name.replace('scale_', '')}={v:g}"
        for name, v in zip(HABITAT_VARS[1:], values[1:])
    ]
    return ", ".join(parts)


def select_combo(df, combo_values):
    mask = np.logical_and.reduce(
        [np.isclose(df[col], val) for col, val in zip(HABITAT_VARS, combo_values)]
    )
    return df[mask]


# -----------------------------------------------------------------------------
# 1. Single-year seasonality
# -----------------------------------------------------------------------------

def single_year_seasonality(sim_df, obs_df, year):
    obs_years = sorted(obs_df["year"].unique())
    if year not in obs_years:
        raise ValueError(f"CALIBRATION_YEAR={year} not in observed data. Available: {obs_years}")

    sim_year = sim_df[sim_df["year"] == year]
    if sim_year.empty:
        raise ValueError(f"CALIBRATION_YEAR={year} has no rows in the simulated data.")

    sim_agg = average_over_seeds(sim_year, ["month"])
    obs_year = obs_df[obs_df["year"] == year]
    merged = sim_agg.merge(obs_year, on="month", how="inner")

    rows = []
    for combo, g in merged.groupby(HABITAT_VARS):
        if len(g) == 12 and g["Cases"].notna().all():
            corr, _ = pearsonr(g["Cases U5"], g["Cases"])
            rows.append((*combo, corr))

    if not rows:
        print(f"\nNo complete 12-month combinations for {year}: merged has {len(merged)} rows "
              "(expect 12 x number of habitat combinations).")
        return None

    ranking = (
        pd.DataFrame(rows, columns=HABITAT_VARS + ["correlation"])
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(os.path.join(OUTPUT_DIR, f"seasonality_ranking_{year}.csv"), index=False)

    top = ranking.head(TOP_SINGLE_YEAR)
    obs_norm = obs_year.sort_values("month").copy()
    obs_norm["norm"] = obs_norm["Cases"] / obs_norm["Cases"].max()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    for ax, (_, row) in zip(axes.flatten(), top.iterrows()):
        subset = select_combo(merged, [row[c] for c in HABITAT_VARS]).sort_values("month")
        ax.plot(subset["month"], subset["Cases U5"] / subset["Cases U5"].max(),
                label="Simulated", color="blue")
        ax.plot(obs_norm["month"], obs_norm["norm"], label="Observed",
                color="black", linestyle="--")
        ax.set_title(f"Corr: {row['correlation']:.4f}\n{combo_label([row[c] for c in HABITAT_VARS])}",
                     fontsize=10)
        ax.set_xlabel("Month")
        ax.set_ylabel("Max normalized cases")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"UWR_seasonality_top6_{year}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")
    return ranking


# -----------------------------------------------------------------------------
# 2. Multi-year seasonality
# -----------------------------------------------------------------------------

def multi_year_seasonality(sim_df, obs_df, year_range):
    lo, hi = year_range

    obs = obs_df[obs_df["year"].between(lo, hi)].copy()
    obs["date"] = pd.to_datetime(obs[["year", "month"]].assign(day=1))
    obs = obs.groupby("date", as_index=False)["Cases"].sum()
    if obs["Cases"].max() == obs["Cases"].min():
        raise ValueError("Observed U5 cases have no variation across the compare year range.")
    obs["norm"] = (obs["Cases"] - obs["Cases"].min()) / (obs["Cases"].max() - obs["Cases"].min())
    obs_series = obs.set_index("date")["norm"].rename("Observed_norm")

    sim = sim_df[sim_df["year"].between(lo, hi)]
    sim_monthly = average_over_seeds(sim, ["year", "month"])
    sim_monthly["date"] = pd.to_datetime(sim_monthly[["year", "month"]].assign(day=1))

    expected_months = len(pd.date_range(f"{lo}-01-01", f"{hi}-12-01", freq="MS"))
    print(f"\nSimulated rows after averaging seeds: {len(sim_monthly)} "
          f"(up to {expected_months} months x number of habitat combinations)")

    rows = []
    for combo, g in sim_monthly.groupby(HABITAT_VARS):
        peak = g["Cases U5"].max()
        if peak <= 0:
            continue
        sim_series = (g.set_index("date")["Cases U5"] / peak).rename("Simulated_norm")
        merged = pd.concat([obs_series, sim_series], axis=1, join="inner").dropna()
        if len(merged) >= 2:
            corr, _ = pearsonr(merged["Observed_norm"], merged["Simulated_norm"])
            rows.append((*combo, corr, len(merged)))

    if not rows:
        print("\nNo overlapping dates between simulated and observed series.")
        return None

    ranking = (
        pd.DataFrame(rows, columns=HABITAT_VARS + ["correlation", "n_months"])
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(os.path.join(OUTPUT_DIR, "seasonality_ranking_multiyear.csv"), index=False)

    top = ranking.head(TOP_MULTI_YEAR)
    print(f"\nTop {TOP_MULTI_YEAR} multi-year combinations:")
    for _, row in top.iterrows():
        print(f"  {combo_label([row[c] for c in HABITAT_VARS])} | "
              f"correlation={row['correlation']:.4f} | months={int(row['n_months'])}")

    fig, axes = plt.subplots(len(top), 1, figsize=(14, 5 * len(top)), sharex=True, squeeze=False)
    for ax, (_, row) in zip(axes.flatten(), top.iterrows()):
        subset = select_combo(sim_monthly, [row[c] for c in HABITAT_VARS]).sort_values("date")
        lo_v, hi_v = subset["Cases U5"].min(), subset["Cases U5"].max()
        # Min-max both series for display (correlation is unaffected by scaling).
        ax.plot(subset["date"], (subset["Cases U5"] - lo_v) / (hi_v - lo_v),
                label="Simulated U5 (min-max)", color="blue", linewidth=2)
        ax.scatter(obs["date"], obs["norm"], facecolors="none", edgecolors="black",
                   s=40, linewidths=1.5, label="Observed U5 (min-max)")
        ax.set_title(f"U5 seasonality ({lo}-{hi})\nCorr: {row['correlation']:.3f}, "
                     f"{combo_label([row[c] for c in HABITAT_VARS])} (months={int(row['n_months'])})",
                     fontsize=10)
        ax.set_ylabel("Normalized cases")
        ax.legend()
        ax.grid(True)

    axes.flatten()[-1].set_xlabel("Date")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "UWR_seasonality_top3_multiyear.png")
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
            f"PfPR CSV does not exist:\n{PFPR_CSV}\nRun uwr_pickup_analyzer.py first."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_df = load_u5_cases(PFPR_CSV)
    obs_df = load_observed_cases(OBSERVED_CASES_PATH)

    n_combos = sim_df[HABITAT_VARS].drop_duplicates().shape[0]
    print(f"Habitat combinations: {n_combos}  seeds: {sim_df[SEED_VAR].nunique()}")

    single_year_seasonality(sim_df, obs_df, CALIBRATION_YEAR)
    multi_year_seasonality(sim_df, obs_df, COMPARE_YEAR_RANGE)


if __name__ == "__main__":
    main()