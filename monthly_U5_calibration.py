# -*- coding: utf-8 -*-
"""UWR_calibration_scripts.py -- SEASONALITY calibration only.

Adapted from FE_Calibration scripts.ipynb for the Upper West Region (UWR)
PICKUP run (run_pickup_UWR.py + UWR_pickup_analyzer.py), 2013-2025.

PREVALENCE calibration (the notebook's beta-binomial section) lives in
UWR_calibrate_prevalence.py instead -- see that file's docstring for why
(pcr_prevalence.csv's real shape doesn't match what this notebook
originally assumed). This file keeps only the seasonality routines, with
the same correlation-ranking LOGIC as the original notebook -- only the
data source and column names are swapped to match what this project's
pipeline actually produces.

WHAT CHANGED VS. THE ORIGINAL NOTEBOOK
  - Simulated data comes from PfPR_ClinicalIncidence_monthly_pickup.csv,
    written by MonthlyPfPRAnalyzer in the UWR pickup analyzer, which
    reports one row per (simulation, month, agebin) with an integer
    'month' column (1-12). load_u5_pfpr() below collapses the two
    under-5 bins (0.25, 5.0 -- see U5_AGEBINS) into single 'Cases U5' /
    'PfPR U5' / 'Pop U5' columns, Pop-weighted, so the correlation logic
    below is otherwise unchanged from the original.
  - Observed data is monthly_u5_cases.csv -- real UWR routine U5 case
    counts, 2015-2025, one row per (Year, Month, Age_Group). ITS Month
    COLUMN IS TEXT ("January", "February", ...), not an integer, and its
    Year range starts in 2015 -- both matter, see BUGS FIXED below.
  - The habitat/xTLH sweep columns are 'scale_temp_rain', 'scale_water_veg'
    (from update_pickup_habitat(), this pickup's OWN habitat mix) and
    'xTLH' -- read per-simulation from each sim's own config.json by
    UWR_pickup_analyzer.py (see that file's docstring for why
    'x_Temporary_Larval_Habitat' is not itself a pickup tag).

BUGS FIXED (from the version that printed "No complete (12-month)
parameter combinations to correlate" with no other error)
  - monthly_u5_cases.csv's 'Month' column is text ("January", ...); the
    simulated data's 'month' column is an integer (1-12). Merging text
    against integers on 'month' matched ZERO rows in every group -- not
    a crash, just silently empty groups, which is why nothing printed
    except the generic "no complete combinations" message.
    MONTH_NAME_TO_NUMBER below converts the observed data's month names
    to integers ONCE, shared by both sections.
  - CALIBRATION_YEAR was 2013, but monthly_u5_cases.csv only starts in
    2015 -- so even with the month fix, that year alone had nothing to
    correlate against. Default is now 2015 (the first year the file
    actually has), and load_observed_cases() now raises a clear error
    naming the years actually available if you pick one that isn't
    there, rather than merging to nothing and printing a generic message.
  - The observed data was being loaded twice (once per section, under
    two separate OBSERVED_CASES_PATH definitions) with an incomplete
    edit halfway between the old ROUTINE_CASES placeholder and the real
    file -- load_observed_cases() below is the one place this happens
    now, reused by both sections.
  - The multi-year section built sim_series directly from sim_trimmed
    (one row per SIMULATION, not per month) and joined it against
    obs_series on 'date' without aggregating first. Every burn-in
    replicate at a given habitat combo shares the same date, so the
    join duplicated each observed row once per replicate -- silently
    inflating the correlation's sample size, feeding raw per-replicate
    noise into pearsonr instead of averaging it out, and producing
    vertical zigzags in the plotted line (multiple simulated points at
    the same x position). sim_monthly now aggregates to one row per
    (date, habitat combo) BEFORE any correlation or join, matching what
    sim_agg already did correctly in the single-year section. A printed
    diagnostic (months=...) now reports how many months actually
    overlapped per combination, so a suspiciously low count is visible
    immediately rather than silently distorting the correlation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

## --------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------
PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")

# Exact pickup experiment directory to calibrate
EXPERIMENT_DIR = os.path.join(PROJECT_DIR,"experiments","outputs",
    "UWR_Pickup_2013_2025_WECM",
)

# PfPR/clinical-incidence output produced by UWR_pickup_analyzer.py
PFPR_CSV = os.path.join(
    EXPERIMENT_DIR,
    "PfPR_ClinicalIncidence_monthly_pickup.csv",
)

# Observed U5 routine case data
OBSERVED_CASES_PATH = os.path.join(
    INPUT_DIR,
    "monthly_u5_cases.csv",
)

# Save calibration plots directly inside this experiment directory
OUTPUT_DIR = EXPERIMENT_DIR

# Fail early with a clear message if the expected file is missing
if not os.path.isdir(EXPERIMENT_DIR):
    raise FileNotFoundError(
        f"Experiment directory does not exist:\n{EXPERIMENT_DIR}"
    )

if not os.path.isfile(PFPR_CSV):
    raise FileNotFoundError(
        f"PfPR CSV does not exist:\n{PFPR_CSV}\n\n"
        f"Check that UWR_pickup_analyzer.py has finished and produced "
        f"PfPR_ClinicalIncidence_monthly_pickup.csv."
    )

print("\nCalibration experiment directory:")
print(f"  {EXPERIMENT_DIR}")

print("\nUsing PfPR CSV:")
print(f"  {PFPR_CSV}")

OBSERVED_CASES_PATH = os.path.join(INPUT_DIR, "monthly_u5_cases.csv")

# Plots save alongside PfPR_ClinicalIncidence_monthly_pickup.csv itself
# (PFPR_CSV's own directory), not PROJECT_DIR -- keeps each experiment's
# calibration outputs next to the data they were generated from.
OUTPUT_DIR = os.path.dirname(PFPR_CSV)

# The two under-5 age bins in run_pickup_UWR.py's add_reports():
# age_bins=[0.25, 5, 15, 25, 100] -> bins are 0-0.25y and 0.25-5y.
U5_AGEBINS = (0.25, 5.0)

MONTH_NAME_TO_NUMBER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def load_u5_pfpr(csv_path):
    """
    Read a MonthlyPfPRAnalyzer CSV (one row per sim/month/agebin) and
    collapse the two under-5 bins into single 'Cases U5' / 'PfPR U5' /
    'Pop U5' columns per simulation/month -- reproducing the shape the
    old site's simulation export gave directly, so the calibration
    logic below is unchanged.
    """
    df = pd.read_csv(csv_path)
    df = df[df["agebin"].isin(U5_AGEBINS)].copy()
    if df.empty:
        raise ValueError(
            f"No rows with agebin in {U5_AGEBINS} in {csv_path}. "
            "Check the 'agebin' column's actual values (print df['agebin'].unique())."
        )

    value_cols = [
        "PfPR", "Cases", "Severe_cases", "New_infections",
        "Anemia", "Mean_density", "Gametocyte_prevalence", "Pop",
    ]
    id_cols = [c for c in df.columns if c not in value_cols + ["agebin"]]

    def _collapse(g):
        pop = g["Pop"].sum()
        out = {
            "Cases U5": g["Cases"].sum(),
            "Severe_cases U5": g["Severe_cases"].sum(),
            "New_infections U5": g["New_infections"].sum(),
            "Pop U5": pop,
        }
        for rate_col, out_name in [
            ("PfPR", "PfPR U5"),
            ("Mean_density", "Mean_density U5"),
            ("Gametocyte_prevalence", "Gametocyte_prevalence U5"),
            ("Anemia", "Anemia U5"),
        ]:
            out[out_name] = float(np.average(g[rate_col], weights=g["Pop"])) if pop > 0 else np.nan
        return pd.Series(out)

    collapsed = (
        df.groupby(id_cols, dropna=False, group_keys=False)
        .apply(_collapse)
        .reset_index()
    )
    # lower-case 'year' to match the original notebook's column naming
    if "Year" in collapsed.columns:
        collapsed = collapsed.rename(columns={"Year": "year"})
    return collapsed


def load_observed_cases(csv_path, age_group="U5"):
    """
    Read monthly_u5_cases.csv (columns: Year, Month, Age_Group,
    Malaria_Cases) and return a tidy frame with integer 'year', integer
    'month' (converted from the text month names via
    MONTH_NAME_TO_NUMBER), and 'Cases' -- the exact column names the
    correlation logic below expects. Filters to age_group ('U5' by
    default), though the real file only has U5 rows anyway.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Age_Group"] == age_group].copy()
    if df.empty:
        raise ValueError(f"No rows with Age_Group == '{age_group}' in {csv_path}.")

    unmapped = sorted(set(df["Month"]) - set(MONTH_NAME_TO_NUMBER))
    if unmapped:
        raise ValueError(
            f"{csv_path}: unrecognised month name(s) {unmapped} -- "
            "add them to MONTH_NAME_TO_NUMBER above."
        )

    df["month"] = df["Month"].map(MONTH_NAME_TO_NUMBER)
    df["year"] = df["Year"].astype(int)
    df["Cases"] = df["Malaria_Cases"].astype(float)
    return df[["year", "month", "Cases"]].reset_index(drop=True)


sim_pfpr = load_u5_pfpr(PFPR_CSV)
obs_cases = load_observed_cases(OBSERVED_CASES_PATH)

# Habitat/xTLH columns actually present depend on how you ran the
# analyzer -- 'xTLH' comes from UWR_pickup_analyzer.py reading each
# simulation's own config.json (see that file's docstring). Fall back
# gracefully if any of these are missing for some reason.
HABITAT_COLS = [c for c in ["scale_temp_rain", "scale_water_veg", "xTLH"] if c in sim_pfpr.columns]
if not HABITAT_COLS:
    raise ValueError(
        "None of scale_temp_rain/scale_water_veg/xTLH are in the PfPR CSV -- "
        "check that you passed sweep_variables when running UWR_pickup_analyzer.py, "
        "and that it recovered xTLH from config.json for this experiment."
    )


################################################################################
# SEASONALITY CALIBRATION -- OPTION 1
################################################################################
#
# Same correlation-ranking logic as the original notebook: for each
# (scale_temp_rain, scale_water_veg, xTLH) combination, correlate the
# simulated U5 case seasonality against observed routine case counts
# for ONE calibration year, then plot the top-6 fits.
#
# CALIBRATION_YEAR must be a year present in BOTH the simulated data
# (2013-2025) AND monthly_u5_cases.csv (2015-2025) -- 2015 is the
# earliest year that satisfies both.
################################################################################

CALIBRATION_YEAR = 2015

available_obs_years = sorted(obs_cases["year"].unique())
if CALIBRATION_YEAR not in available_obs_years:
    raise ValueError(
        f"CALIBRATION_YEAR={CALIBRATION_YEAR} has no rows in {OBSERVED_CASES_PATH}. "
        f"Years actually available: {available_obs_years}."
    )

sim_year = sim_pfpr[sim_pfpr["year"] == CALIBRATION_YEAR]
if sim_year.empty:
    raise ValueError(
        f"CALIBRATION_YEAR={CALIBRATION_YEAR} has no rows in the simulated PfPR CSV -- "
        "check that this year is inside PICKUP_START_YEAR-PICKUP_END_YEAR "
        "(2013-2025) in run_pickup_UWR.py."
    )

obs_year = obs_cases[obs_cases["year"] == CALIBRATION_YEAR]

sim_agg = (
    sim_year.groupby(["month"] + HABITAT_COLS)[["Cases U5"]]
    .mean()
    .reset_index()
)

merged_seasonal = sim_agg.merge(obs_year, on="month", how="inner")

correlations_seasonal = []
for combo, group in merged_seasonal.groupby(HABITAT_COLS):
    if len(group) == 12 and group["Cases"].notna().all():
        corr, _ = pearsonr(group["Cases U5"], group["Cases"])
        correlations_seasonal.append((*np.atleast_1d(combo), corr))

if not correlations_seasonal:
    print(
        f"\nNo complete (12-month) parameter combinations to correlate for "
        f"{CALIBRATION_YEAR} -- merged_seasonal has {len(merged_seasonal)} rows "
        f"(expect 12 x number of habitat/xTLH combinations). Check that "
        f"sim_year and obs_year both actually cover all 12 months."
    )
else:
    top6_seasonal_corr = sorted(correlations_seasonal, key=lambda x: x[-1], reverse=True)[:6]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, row in enumerate(top6_seasonal_corr):
        *combo_vals, corr = row
        mask = np.logical_and.reduce([
            merged_seasonal[col] == val for col, val in zip(HABITAT_COLS, combo_vals)
        ])
        subset = merged_seasonal[mask].copy()

        subset["Cases_U5_max_norm"] = subset["Cases U5"] / subset["Cases U5"].max()
        obs_norm = obs_year.copy()
        obs_norm["Cases_max_norm"] = obs_norm["Cases"] / obs_norm["Cases"].max()

        ax = axes[idx]
        ax.plot(subset["month"], subset["Cases_U5_max_norm"], label="Simulated", color="blue")
        ax.plot(obs_norm["month"], obs_norm["Cases_max_norm"], label="Observed",
                 color="black", linestyle="--")

        label = ", ".join(f"{c}={v:g}" for c, v in zip(HABITAT_COLS, combo_vals))
        ax.set_title(f"Corr: {corr:.4f}\n{label}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Max Normalized Cases")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"UWR_seasonality_top6_{CALIBRATION_YEAR}.png"), dpi=300)
    plt.close(fig)


################################################################################
# SEASONALITY CALIBRATION -- multi-year comparison against the observed
# monthly time series (was the Kumasi comparison in the original
# notebook; here generalised to UWR and to the pickup's own years).
################################################################################
#
# BUG FIXED: this section used to build sim_series by indexing
# sim_trimmed on 'date' WITHOUT first aggregating across simulations --
# sim_trimmed has one row per (simulation, month, habitat combo), so a
# habitat combo run with several burn-in replicates has SEVERAL rows
# sharing the same date. obs_series.join(sim_series, how='inner') then
# duplicated each observed row to match every one of those replicate
# rows at that date, which (a) fed raw per-replicate noise straight
# into pearsonr instead of averaging it out first, (b) silently
# inflated the correlation's sample size well past the true number of
# overlapping months, and (c) produced the vertical zigzags in the
# plotted line, since plot() draws every replicate's value at the same
# x position. sim_monthly below aggregates to one row per (date +
# habitat combo) BEFORE anything touches pearsonr or the join, exactly
# like sim_agg already did in the single-year section above -- so both
# sections now follow the same pattern.
################################################################################

# Overlap of simulated years (2013-2025) and observed years (2015-2025).
COMPARE_YEAR_RANGE = (2015, 2025)

# ---- Observed: one row per date, guaranteed --------------------------
obs_multi = obs_cases[obs_cases["year"].between(*COMPARE_YEAR_RANGE)].copy()
obs_multi["date"] = pd.to_datetime(obs_multi[["year", "month"]].assign(day=1))
obs_multi = obs_multi.groupby("date", as_index=False)["Cases"].sum()

obs_min, obs_max = obs_multi["Cases"].min(), obs_multi["Cases"].max()
if obs_max == obs_min:
    raise ValueError("Observed U5 cases have no variation across the compare year range.")
obs_multi["Cases_norm"] = (obs_multi["Cases"] - obs_min) / (obs_max - obs_min)

obs_series = (
    obs_multi[["date", "Cases_norm"]]
    .rename(columns={"Cases_norm": "Observed_norm"})
    .set_index("date")
)

# ---- Simulated: aggregate across replicates to one row per
# (date, habitat combo) BEFORE any correlation or join -------------------
sim_trimmed = sim_pfpr[sim_pfpr["year"].between(*COMPARE_YEAR_RANGE)].copy()
sim_trimmed["date"] = pd.to_datetime(sim_trimmed[["year", "month"]].assign(day=1))

sim_monthly = (
    sim_trimmed.groupby(["date"] + HABITAT_COLS, as_index=False)["Cases U5"].mean()
)

expected_months = len(pd.date_range(
    f"{COMPARE_YEAR_RANGE[0]}-01-01", f"{COMPARE_YEAR_RANGE[1]}-12-01", freq="MS"
))
print(
    f"\nSimulated monthly rows after aggregation: {len(sim_monthly)} "
    f"(expected up to {expected_months} months x number of habitat/xTLH combos)"
)

correlations = []
for combo, group in sim_monthly.groupby(HABITAT_COLS):
    group = group.sort_values("date").copy()

    sim_max = group["Cases U5"].max()
    if sim_max <= 0:
        continue
    group["Simulated_norm"] = group["Cases U5"] / sim_max

    sim_series = group[["date", "Simulated_norm"]].set_index("date")
    merged = obs_series.join(sim_series, how="inner").dropna()

    # Diagnostic: how many months actually overlapped for this combo --
    # should be close to expected_months, not a small fraction of it.
    if len(merged) >= 2:
        corr, _ = pearsonr(merged["Observed_norm"], merged["Simulated_norm"])
        correlations.append((*np.atleast_1d(combo), corr, len(merged)))

if not correlations:
    print(
        f"\nNo overlapping (date, habitat combo) rows for the "
        f"{COMPARE_YEAR_RANGE[0]}-{COMPARE_YEAR_RANGE[1]} multi-year comparison -- "
        "check that sim_monthly and obs_multi actually share some dates."
    )
else:
    correlations = sorted(correlations, key=lambda x: x[-2], reverse=True)
    top3_corr_avg = correlations[:3]

    print("\nTop 3 multiyear seasonality combinations:")
    for row in top3_corr_avg:
        *combo_vals, corr, n_months = row
        label = ", ".join(f"{c}={v:g}" for c, v in zip(HABITAT_COLS, combo_vals))
        print(f"  {label} | correlation={corr:.4f} | months={n_months}")

    fig, axes = plt.subplots(nrows=len(top3_corr_avg), figsize=(14, 5 * len(top3_corr_avg)), sharex=True)
    axes = np.atleast_1d(axes)

    for idx, row in enumerate(top3_corr_avg):
        *combo_vals, corr, n_months = row
        mask = np.logical_and.reduce([
            sim_monthly[col] == val for col, val in zip(HABITAT_COLS, combo_vals)
        ])
        subset = sim_monthly[mask].sort_values("date").copy()
        sim_max = subset["Cases U5"].max()
        subset["Cases_U5_norm"] = subset["Cases U5"] / sim_max

        ax = axes[idx]
        ax.plot(subset["date"], subset["Cases_U5_norm"], label="Simulated U5 (Norm)",
                 color="blue", linewidth=2)
        ax.scatter(obs_multi["date"], obs_multi["Cases_norm"], facecolors="none",
                    edgecolors="black", s=40, marker="o", label="Observed U5 (Min-Max)", linewidths=1.5)

        label = ", ".join(f"{c}={v:g}" for c, v in zip(HABITAT_COLS, combo_vals))
        ax.set_title(
            f"U5 Seasonality ({COMPARE_YEAR_RANGE[0]}-{COMPARE_YEAR_RANGE[1]})\n"
            f"Corr: {corr:.3f}, {label} (months={n_months})"
        )
        ax.set_ylabel("Normalized Cases")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "UWR_seasonality_top3_multiyear.png"), dpi=300)
    plt.close(fig)
    print(f"\nSaved multiyear seasonality plot to:\n"
          f"{os.path.join(OUTPUT_DIR, 'UWR_seasonality_top3_multiyear.png')}")


################################################################################
# PREVALENCE CALIBRATION -- moved out of this file
################################################################################
#
# The real pcr_prevalence.csv has 'survey_year' plus a 'fieldwork_start'/
# 'fieldwork_end' window -- no 'year'/'month' to merge on directly, and
# the 2011 survey predates the pickup's simulated years (2013-2025)
# entirely. That logic lives in UWR_calibrate_prevalence.py instead, so
# there's one source of truth rather than two paths that could drift
# apart. Run that script directly for prevalence calibration.
################################################################################