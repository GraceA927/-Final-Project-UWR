"""UWR under-five PfPR calibration using year/month-matched DHS observations.

This script reads the output of analyzer_Pickup13yr_PREG.py for the pickup
experiment configured in Site_Setup_Final.py (EXPERIMENT_NAME_PICKUP).

Method
------
1. Read PfPR_ClinicalIncidence_monthly_pickup.csv from the pickup analyzer.
2. Reconstruct PfPR for the DHS age range (age_min_months-age_max_months in
   the DHS file, 6-59 months) by population-weighting the report age bins
   (Site_Setup_Final.REPORT_AGE_BINS) that lie inside it. With bins
   [0.25, 0.5, 5, ...] that is exactly the 0.5-5 year bin. The 5-day tail
   interval of each report year is excluded (it shares the December label).
3. Retain the complete independently swept habitat combination:
   x_Temporary_Larval_Habitat, scale_constant, scale_temp_rain,
   scale_water_veg, and Run_Number.
4. Match DHS observations to simulated output using year + month only.
   The DHS year/month is the midpoint of the fieldwork dates; surveys outside
   the simulated period are excluded.
5. Calculate beta-binomial log-likelihood for each habitat combination/run.
6. Average run-level likelihoods within each full habitat combination.
7. Also calculate an xTLH profile by averaging the full-combination scores
   at each xTLH. This profile is for plotting only; the joint score table
   remains the correct basis for selecting a complete parameter combination.

Temporal caveat
---------------
The EMOD PfPR report uses 30-day reporting intervals, not true calendar months.
The adapted analyzer assigns each interval to the flat-model-calendar month
containing its midpoint. Therefore year/month matching is an approximation
that makes the requested year+month join possible; the resulting PfPR values
are 30-day interval estimates, not exact calendar-month means.

Expected DHS columns: year, month, DHS_n, DHS_pos.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from idmtools_calibra.utilities.ll_calculators import beta_binomial
import manifest
import Site_Setup_Final as SETUP


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

EXPERIMENT_NAME = SETUP.EXPERIMENT_NAME_PICKUP
JOB_DIR = manifest.job_directory
INPUT_DIR = manifest.input_dir

ANALYZER_OUTPUT_DIR = os.path.join(JOB_DIR, f"{EXPERIMENT_NAME}_Output")
SIMULATION_FILE = os.path.join(
    ANALYZER_OUTPUT_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv"
)
DHS_FILE = os.path.join(INPUT_DIR, "pcr_prevalence.csv")

OUTPUT_DIR = os.path.join(ANALYZER_OUTPUT_DIR, "calibration")
os.makedirs(OUTPUT_DIR, exist_ok=True)

XCOL = "x_Temporary_Larval_Habitat"
HABITAT_COLS = list(SETUP.HABITAT_TAGS)
RUN_COL = "Run_Number"
PARAM_COLS = [XCOL] + HABITAT_COLS
GROUP_COLS = PARAM_COLS + [RUN_COL]
TIME_COLS = ["year", "month"]

# The analyzer reports age-bin upper bounds (Site_Setup_Final.REPORT_AGE_BINS).
REPORT_AGE_BINS = sorted(float(a) for a in SETUP.REPORT_AGE_BINS)
# Fallback age range (years) if the DHS file has no age_min/max_months columns.
DEFAULT_AGE_RANGE_YEARS = (0.0, 5.0)


def age_bins_for_range(age_min_years, age_max_years):
    """Report bins (upper bounds) whose [lower, upper) lies inside the survey range."""
    lowers = [0.0] + REPORT_AGE_BINS[:-1]
    tol = 1e-9
    chosen = [up for lo, up in zip(lowers, REPORT_AGE_BINS)
              if lo >= age_min_years - tol and up <= age_max_years + tol]
    if not chosen:
        raise ValueError(
            f"No report age bins {REPORT_AGE_BINS} fit inside the survey age range "
            f"{age_min_years:g}-{age_max_years:g} years. Adjust REPORT_AGE_BINS in Site_Setup_Final.py.")
    lo_cov = lowers[REPORT_AGE_BINS.index(chosen[0])]
    if not (np.isclose(lo_cov, age_min_years) and np.isclose(chosen[-1], age_max_years)):
        warnings.warn(
            f"Report age bins cover {lo_cov:g}-{chosen[-1]:g} years but the survey covers "
            f"{age_min_years:g}-{age_max_years:g} years; PfPR is not an exact age match.")
    return chosen

SCORE_FILE = os.path.join(OUTPUT_DIR, "calibration_scores_by_combination.csv")
RUN_SCORE_FILE = os.path.join(OUTPUT_DIR, "calibration_run_scores.csv")
XTLH_PROFILE_FILE = os.path.join(OUTPUT_DIR, "calibration_xtlh_profile.csv")
SKIPPED_FILE = os.path.join(OUTPUT_DIR, "calibration_skipped_runs.csv")
PLOT_FILE = os.path.join(OUTPUT_DIR, "calibration_year_month_diagnostics.png")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalise_columns(df):
    """Make column matching tolerant of capitalization and whitespace."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {
        "Year": "year", "Month": "month", "Agebin": "agebin",
        "PFPR": "PfPR", "P f P R": "PfPR", "POP": "Pop",
        "run_number": RUN_COL,
    }
    # Only rename an alias when the canonical name is not already present
    # (e.g. analyzer output carries both "Year" and "year").
    df = df.rename(columns={
        k: v for k, v in aliases.items() if k in df.columns and v not in df.columns
    })
    # Guard against any remaining duplicate column names.
    return df.loc[:, ~df.columns.duplicated()]


def numeric_columns(df, columns, source_name):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[columns].isna().any(axis=1)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} {source_name} rows with missing/non-numeric required values.")
        df = df.loc[~bad].copy()
    return df


# ---------------------------------------------------------------------
# Load and prepare simulation data
# ---------------------------------------------------------------------

def load_simulation_data(path, age_range_years=DEFAULT_AGE_RANGE_YEARS):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Simulation file not found: {path}\n"
            "Run the pickup analyzer first and confirm EXPERIMENT_NAME."
        )

    df = normalise_columns(pd.read_csv(path))
    required = TIME_COLS + GROUP_COLS + ["agebin", "PfPR", "Pop"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Simulation CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = numeric_columns(df, required, "simulation")
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    if not df["month"].between(1, 12).all():
        raise ValueError("Simulation month values must be integers from 1 to 12.")
    if not df["PfPR"].between(0, 1).all():
        raise ValueError("Simulation PfPR values must be between 0 and 1.")
    if (df["Pop"] <= 0).any():
        raise ValueError("Simulation Pop values must be greater than zero.")

    # Drop the short tail interval of each report year (12 x 30 days + 5 days):
    # the analyzer labels it December too, which would mix it into December.
    if "is_complete_interval" in df.columns:
        complete = df["is_complete_interval"].astype(str).str.strip().str.lower().eq("true")
        df = df.loc[complete].copy()

    # The report age bins must be the ones Site_Setup_Final.py configured.
    observed_bins = sorted(df["agebin"].dropna().unique())
    if not np.allclose(observed_bins, REPORT_AGE_BINS):
        raise ValueError(
            f"Simulation age bins {observed_bins} differ from "
            f"Site_Setup_Final.REPORT_AGE_BINS {REPORT_AGE_BINS}. Re-run the analyzer.")

    # Keep the age bins matching the survey age range.
    target_bins = age_bins_for_range(*age_range_years)
    print(f"Survey ages {age_range_years[0]:g}-{age_range_years[1]:g} y -> "
          f"report age bins (upper bounds) {target_bins}")
    u5 = df[np.isclose(df["agebin"].to_numpy()[:, None], target_bins).any(axis=1)].copy()

    # Population-weighted PfPR for the combined under-five population.
    u5["PfPR_numerator"] = u5["PfPR"] * u5["Pop"]
    agg_keys = GROUP_COLS + TIME_COLS
    sim = (
        u5.groupby(agg_keys, as_index=False)
        .agg(
            PfPR_numerator=("PfPR_numerator", "sum"),
            Pop=("Pop", "sum"),
            n_age_bins=("agebin", "nunique"),
        )
    )
    sim["PfPR"] = sim["PfPR_numerator"] / sim["Pop"]
    sim["npos"] = np.rint(sim["PfPR"] * sim["Pop"])
    sim = sim.drop(columns="PfPR_numerator")

    # There must be one U5 row per full parameter combination/run/year/month.
    duplicate_key = agg_keys
    duplicates = sim.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        example = sim.loc[duplicates, duplicate_key].drop_duplicates().head(12)
        raise ValueError(
            "Duplicate simulation rows remain after under-five age aggregation. "
            "Check whether the analyzer output contains duplicate report intervals "
            "or duplicated simulation tags. Examples:\n"
            + example.to_string(index=False)
        )

    return sim.sort_values(agg_keys).reset_index(drop=True)


# ---------------------------------------------------------------------
# Load DHS data
# ---------------------------------------------------------------------

def load_dhs_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DHS file not found: {path}")

    df = normalise_columns(pd.read_csv(path))

    required = [
        "survey_year",
        "DHS_n",
        "DHS_pos",
        "fieldwork_start",
        "fieldwork_end",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DHS CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = numeric_columns(
        df, ["survey_year", "DHS_n", "DHS_pos"], "DHS"
    )

    df["survey_year"] = df["survey_year"].astype(int)

    df["fieldwork_start"] = pd.to_datetime(
        df["fieldwork_start"], errors="coerce"
    )
    df["fieldwork_end"] = pd.to_datetime(
        df["fieldwork_end"], errors="coerce"
    )

    if df[["fieldwork_start", "fieldwork_end"]].isna().any().any():
        raise ValueError("Invalid or missing DHS fieldwork dates.")

    if (df["fieldwork_end"] < df["fieldwork_start"]).any():
        raise ValueError("fieldwork_end cannot precede fieldwork_start.")

    # Derive matching year and month from fieldwork midpoint.
    df["fieldwork_midpoint"] = (
        df["fieldwork_start"]
        + (df["fieldwork_end"] - df["fieldwork_start"]) / 2
    )

    df["year"] = df["fieldwork_midpoint"].dt.year.astype(int)
    df["month"] = df["fieldwork_midpoint"].dt.month.astype(int)

    if (df["DHS_n"] <= 0).any():
        raise ValueError("DHS_n must be greater than zero.")

    if ((df["DHS_pos"] < 0) | (df["DHS_pos"] > df["DHS_n"])).any():
        raise ValueError("DHS_pos must be between zero and DHS_n.")

    duplicates = df.duplicated(TIME_COLS, keep=False)

    if duplicates.any():
        raise ValueError(
            "Multiple DHS observations map to the same year/month."
        )

    df["DHS_n"] = df["DHS_n"].astype(float)
    df["DHS_pos"] = df["DHS_pos"].astype(float)

    df["PfPR_observed"] = df["DHS_pos"] / df["DHS_n"]

    if "prevalence" in df.columns:
        df["prevalence"] = pd.to_numeric(
            df["prevalence"], errors="coerce"
        )
        df["count_prevalence_difference"] = (
            df["PfPR_observed"] - df["prevalence"]
        )

    df["date"] = df["fieldwork_midpoint"]

    return df.sort_values(TIME_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------
# Likelihood scoring
# ---------------------------------------------------------------------

def beta_binomial_loglik(observed, simulated):
    result = beta_binomial(
        observed["DHS_n"].to_numpy(dtype=float),
        simulated["Pop"].to_numpy(dtype=float),
        observed["DHS_pos"].to_numpy(dtype=float),
        simulated["npos"].to_numpy(dtype=float),
    )
    values = np.asarray(result, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("beta_binomial returned an empty or non-finite result.")
    # The installed utility may return a scalar or per-observation array.
    return float(values.reshape(-1)[0]) if values.size == 1 else float(values.sum())


def score_all_combinations(sim, dhs):
    run_rows = []
    skipped_rows = []

    for key, run_df in sim.groupby(GROUP_COLS, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        params = dict(zip(GROUP_COLS, key))

        merged = dhs.merge(
            run_df[TIME_COLS + ["Pop", "npos", "PfPR"]],
            on=TIME_COLS,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = merged["_merge"].ne("both")
        if missing.any():
            skipped_rows.append({
                **params,
                "n_dhs_months": len(dhs),
                "n_unmatched_months": int(missing.sum()),
                "missing_year_months": str(
                    merged.loc[missing, TIME_COLS].to_dict("records")
                ),
                "reason": "Incomplete year/month coverage",
            })
            continue

        merged = merged.drop(columns="_merge")
        try:
            ll = beta_binomial_loglik(merged, merged)
        except Exception as exc:
            skipped_rows.append({
                **params,
                "n_dhs_months": len(dhs),
                "n_unmatched_months": 0,
                "missing_year_months": "",
                "reason": f"Likelihood calculation failed: {exc}",
            })
            continue

        run_rows.append({
            **params,
            "log_likelihood": ll,
            "n_dhs_months": len(dhs),
        })

    run_scores = pd.DataFrame(run_rows)
    skipped = pd.DataFrame(skipped_rows)
    if run_scores.empty:
        raise ValueError(
            "No complete habitat-combination/run groups could be scored. "
            "Check year/month overlap and the skipped-run diagnostics."
        )

    # Joint score: each row is one complete habitat-parameter combination.
    combo_scores = (
        run_scores.groupby(PARAM_COLS, as_index=False, dropna=False)
        .agg(
            mean_log_likelihood=("log_likelihood", "mean"),
            sd_log_likelihood=("log_likelihood", "std"),
            n_runs=("log_likelihood", "size"),
        )
        .sort_values("mean_log_likelihood", ascending=False)
        .reset_index(drop=True)
    )

    # xTLH profile is a marginal descriptive summary across the swept values
    # of the other habitat parameters. It is not the joint model-selection table.
    xtlh_profile = (
        combo_scores.groupby(XCOL, as_index=False)
        .agg(
            mean_log_likelihood=("mean_log_likelihood", "mean"),
            sd_across_combinations=("mean_log_likelihood", "std"),
            n_parameter_combinations=("mean_log_likelihood", "size"),
        )
        .sort_values(XCOL)
        .reset_index(drop=True)
    )
    return combo_scores, run_scores, xtlh_profile, skipped


# ---------------------------------------------------------------------
# Diagnostics and plots
# ---------------------------------------------------------------------

def print_diagnostics(sim, dhs, combo_scores, run_scores, xtlh_profile, skipped):
    print("\n" + "=" * 78)
    print("UWR CALIBRATION: YEAR + MONTH MATCHING")
    print("=" * 78)
    print(f"Simulation U5 rows: {len(sim):,}")
    print(f"Unique xTLH values: {sim[XCOL].nunique():,}")
    print(f"Unique habitat combinations: {sim[PARAM_COLS].drop_duplicates().shape[0]:,}")
    print(f"Full combination × Run_Number groups: {sim[GROUP_COLS].drop_duplicates().shape[0]:,}")
    print(f"DHS observations: {len(dhs):,}")
    print(f"Complete run groups scored: {len(run_scores):,}")
    print(f"Incomplete/failed run groups skipped: {len(skipped):,}")

    #print("\nDHS observations used (fieldwork dates not used):")
    #print(dhs[["year", "month", "DHS_n", "DHS_pos", "PfPR_observed"]].to_string(index=False))
    
    print("\nDHS observations used (matching month derived from fieldwork midpoint):")

    print(dhs[[  "survey_year",  "fieldwork_start",  "fieldwork_end",  "fieldwork_midpoint", "year",
            "month","DHS_n","DHS_pos","PfPR_observed", ]].to_string(index=False))

    print("\nHabitat parameter combinations ranked by mean log-likelihood:")
    print(combo_scores.to_string(index=False))

    if not skipped.empty:
        print("\nSkipped groups (first 20):")
        print(skipped.head(20).to_string(index=False))

    print("\nMarginal xTLH profile (mean across habitat combinations):")
    print(xtlh_profile.to_string(index=False))


def plot_results(sim, dhs, combo_scores, xtlh_profile, path):
    # Select the best *joint* parameter combination, then average its runs.
    best = combo_scores.iloc[0]
    mask = np.ones(len(sim), dtype=bool)
    for col in PARAM_COLS:
        mask &= np.isclose(
            pd.to_numeric(sim[col], errors="coerce").to_numpy(dtype=float),
            float(best[col]),
        )
    best_sim = sim.loc[mask].copy()
    best_monthly = (
        best_sim.groupby(TIME_COLS, as_index=False)
        .agg(PfPR_simulated=("PfPR", "mean"))
    )
    best_monthly["date"] = pd.to_datetime(
        dict(year=best_monthly["year"], month=best_monthly["month"], day=1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        best_monthly["date"], best_monthly["PfPR_simulated"],
        linewidth=1.8, label="Simulated U5 PfPR (best fit)"
    )
    axes[0].scatter(
        dhs["date"], dhs["PfPR_observed"],
        s=np.clip(dhs["DHS_n"] / 10.0, 18, 100),
        color="black", label="Observed DHS", zorder=5,
    )
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Under-five PfPR")
    axes[0].set_title("Observed and simulated under-five PfPR")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(
        xtlh_profile[XCOL], xtlh_profile["mean_log_likelihood"],
        "-o", linewidth=1.5, markersize=5,
    )
    best_xtlh = float(best[XCOL])
    profile_best = xtlh_profile.iloc[
        np.argmin(np.abs(xtlh_profile[XCOL].to_numpy(dtype=float) - best_xtlh))
    ]
    axes[1].scatter(
        [best_xtlh], [profile_best["mean_log_likelihood"]],
        s=75, edgecolor="black", zorder=5,
        label="xTLH of best joint combination",
    )
    axes[1].set_xlabel("x_Temporary_Larval_Habitat")
    axes[1].set_ylabel("Mean log-likelihood")
    axes[1].set_title("Marginal mean log-likelihood by xTLH")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Reading DHS data: {DHS_FILE}")
    dhs = load_dhs_data(DHS_FILE)

    # Simulated age range = DHS age range (age_max_months is the last completed
    # month, so 6-59 months -> 0.5 to <5.0 years).
    age_range = DEFAULT_AGE_RANGE_YEARS
    if {"age_min_months", "age_max_months"}.issubset(dhs.columns):
        lo = pd.to_numeric(dhs["age_min_months"], errors="coerce").dropna().unique()
        hi = pd.to_numeric(dhs["age_max_months"], errors="coerce").dropna().unique()
        if len(lo) != 1 or len(hi) != 1:
            raise ValueError(f"DHS surveys use different age ranges ({lo}, {hi}); "
                             "split them or harmonise before calibrating.")
        age_range = (float(lo[0]) / 12.0, (float(hi[0]) + 1.0) / 12.0)

    print(f"Reading analyzer output: {SIMULATION_FILE}")
    sim = load_simulation_data(SIMULATION_FILE, age_range)

    # Keep only DHS surveys whose year/month falls inside the simulated period.
    sim_months = sim[TIME_COLS].drop_duplicates()
    in_sim = dhs.merge(sim_months, on=TIME_COLS, how="left", indicator=True)["_merge"].eq("both").to_numpy()
    if not in_sim.all():
        dropped = dhs.loc[~in_sim, ["survey_year"] + TIME_COLS].to_dict("records")
        print(f"NOTE: excluding {int((~in_sim).sum())} DHS survey(s) outside the simulated period: {dropped}")
        dhs = dhs.loc[in_sim].reset_index(drop=True)
    if dhs.empty:
        raise ValueError("No DHS surveys overlap the simulated year/month range.")

    combo_scores, run_scores, xtlh_profile, skipped = score_all_combinations(sim, dhs)
    print_diagnostics(sim, dhs, combo_scores, run_scores, xtlh_profile, skipped)

    combo_scores.to_csv(SCORE_FILE, index=False)
    run_scores.to_csv(RUN_SCORE_FILE, index=False)
    xtlh_profile.to_csv(XTLH_PROFILE_FILE, index=False)
    if not skipped.empty:
        skipped.to_csv(SKIPPED_FILE, index=False)

    plot_results(sim, dhs, combo_scores, xtlh_profile, PLOT_FILE)

    best = combo_scores.iloc[0]
    print("\n" + "=" * 78)
    print("BEST JOINT HABITAT PARAMETER COMBINATION")
    print("=" * 78)
    for col in PARAM_COLS:
        print(f"{col}: {best[col]}")
    print(f"Mean log-likelihood: {best['mean_log_likelihood']:.6f}")
    print(f"Runs contributing: {int(best['n_runs'])}")
    print("\nSaved calibration outputs:")
    for path in [SCORE_FILE, RUN_SCORE_FILE, XTLH_PROFILE_FILE, PLOT_FILE]:
        print(f"  {path}")
    if not skipped.empty:
        print(f"  {SKIPPED_FILE}")


if __name__ == "__main__":
    main()
