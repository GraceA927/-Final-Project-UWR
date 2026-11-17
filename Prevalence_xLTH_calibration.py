"""
UWR xTLH calibration using the reference run-level likelihood method.

Method
------
1. Keep each EMOD run separate:
   (xTLH, scale_constant, scale_temp_rain, scale_water_veg, Run_Number).
2. Match each DHS survey to the EMOD model YEAR and MONTH of its fieldwork
   midpoint. Simulation rows are labelled with model_year / model_month by the
   pickup analyzer (flat 365-day calendar, 29 Feb dropped from the climate, so
   model months and Gregorian months are the same months). No 30-day blocks and
   no day-of-year arithmetic are used here.
3. Calculate rounded simulated positives within each run:
       npos = round(PfPR * Pop)
4. Score each complete run against all DHS surveys using beta-binomial
   log-likelihood.
5. Average run-level likelihoods across all other habitat parameters and
   seeds for each xTLH.
6. Select the xTLH with the highest mean run-level log-likelihood.

The plot shows mean simulated PfPR trajectories across all other parameter
combinations and seeds at each xTLH. Incomplete runs are excluded from
likelihood scoring and counted in the diagnostics.

Input requirement: PfPR_ClinicalIncidence_monthly_pickup.csv must contain the
model_year and model_month columns written by the updated pickup analyzer.
"""

import os

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from idmtools_calibra.utilities.ll_calculators import beta_binomial


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
DHS_FILE = "pcr_prevalence.csv"

try:
    import manifest
    JOB_DIR = manifest.job_directory
except (ImportError, AttributeError):
    JOB_DIR = os.path.join(PROJECT_DIR, "experiments")

EXPERIMENT_NAME = "Pickup13yrs_Final"     # round-2 habitat-type grid (6ce151ce); spline run: Pickup13yrs_Spline
ANALYZER_DIR = os.path.join(
    JOB_DIR, f"{EXPERIMENT_NAME}_AnalyzerOutput"           # analyzer_Pickup13yr.py output folder
)

SIM_PFPR_CSV = os.path.join(
    ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv"
)
OUTPUT_DIR = ANALYZER_DIR


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
N_MONTHS = 12
DAYS_PER_YEAR = 365

# Fixed-month-length EMOD calendar; must match the analyzer/pickup script.
# Used only to place points on the plots (mid-month).
MONTH_START = {
    1: 0, 2: 31, 3: 59, 4: 90,
    5: 120, 6: 151, 7: 181, 8: 212,
    9: 243, 10: 273, 11: 304, 12: 334,
}
MONTH_LEN = {
    1: 31, 2: 28, 3: 31, 4: 30,
    5: 31, 6: 30, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 12: 31,
}

# DHS children 6-59 months correspond to the 0.5-5 year age bin.
SURVEY_AGEBINS = (5.0,)

XCOL = "x_Temporary_Larval_Habitat"
# Habitat tags of the run (Site_Setup_Final.HABITAT_MODE):
#   spline: spline_lag, spline_power, spline_scale   types: scale_constant, scale_temp_rain, scale_water_veg
HABITAT_MODE = "types"
OTHER_VARS = (["spline_lag", "spline_power", "spline_scale"] if HABITAT_MODE == "spline"
              else ["scale_constant", "scale_temp_rain", "scale_water_veg"])
SEED_VAR = "Run_Number"
RUN_KEYS = [XCOL] + OTHER_VARS + [SEED_VAR]

# IRS calibration grid tags (Site_Setup_Final.IRS_SWEEP). When they are present
# in the simulation CSV, selection is joint over (xTLH, IRS settings). Averaging LL
# across IRS settings would mix settings that eliminate transmission with ones
# that don't.
IRS_VARS = ["irs_kill_scale", "irs_decay_scale", "irs_coverage_scale"]
# Habitat-decay sweep tags (Site_Setup_Final.PICKUP_CONFIG_SWEEP), scored jointly like IRS.
DECAY_VARS = ["Temporary_Habitat_Decay_Factor", "Semipermanent_Habitat_Decay_Rate"]
SEL_KEYS = [XCOL]          # + IRS_VARS when IRS tags exist, + OTHER_VARS with seasonality

# Seasonality from routine U5 confirmed cases (inputs/monthly_U5_conf.csv).
# Only the within-year SHAPE is compared (each year divided by its mean), because
# the level of reported cases depends on testing/reporting, which changed over time.
# With seasonality on, every full parameter set (xTLH + habitat + IRS) is scored
# separately (mean over seeds), and the best set is the highest DHS PfPR LL among
# sets whose seasonal profile correlates r >= SEASON_MIN_R with the observed one.
USE_SEASONALITY = True
OBS_U5_FILE = "monthly_U5_conf.csv"
OBS_POP_COL = "Pop_5"          # U5 population; None = no per-person scaling
# Observed monthly totals are converted to an annualised rate per person,
#   rate = cases / Pop_5 * 365 / days_in_month,
# the same scale as the simulated 'Annual Clinical Incidence' (Cases).
OBS_U5_COL = "conf_u5"         # file columns: Region,year,month,conf_u5,adj1_conf_u5,Pop_5,observed_incidence_per_1000
SEASON_YEARS = (2015, 2025)
SEASON_MIN_R = 0.6             # climatology correlation (Jan-Dec mean profile)
SEASON_MAX_PEAK_DIFF = 1       # |simulated - observed peak month| allowed (months)
# LAG ADJUSTMENT. The simulated profile can be moved EARLIER by L months before it
# is compared with the observed one. "auto" picks, per parameter set, the L in
# 0..SEASON_MAX_LAG with the highest climatology r; an int fixes L for every set;
# 0 switches the adjustment off. The unshifted r (season_r_raw) and the lag used
# (season_lag) are always reported, and the plot draws the raw curve too.
# NOTE: this aligns the comparison only; the model's transmission season is still
# L months later than observed, which matters for intervention timing (IRS in April,
# IG2 on 1 April) in the scenario runs. Report the lag with the results.
SEASON_LAG_MONTHS = 0 if HABITAT_MODE == "spline" else "auto"   # spline run: no shift
SEASON_MAX_LAG = 3
# Simulated U5 = all report age bins with upper edge <= 5 (0-0.25 and 0.25-5),
# combined population-weighted: sum(Cases x Pop) / sum(Pop).
U5_AGEBIN_MAX = 5.0
# Habitat grid swept in the burn-in (Site_Setup_Final.py), xTLH fixed at 37.4.
# The calibration reports the best (constant, temp_rain, water_veg) from this grid.
HABITAT_GRID = ({
    "spline_lag": [1.0, 2.0, 3.0],
    "spline_power": [0.5, 1.0, 1.5],
    "spline_scale": [0.75, 1.25, 2.0],
} if HABITAT_MODE == "spline" else {
    "scale_constant": [0.0, 0.1, 0.25, 0.5],
    "scale_temp_rain": [8.0, 14.0, 20.0],
    "scale_water_veg": [0.1, 0.5, 1.0],
})
# Only the seasonality plot is produced (xTLH is fixed, so the xTLH/PfPR
# selection plot and the IRS grid plot are skipped).
PLOT_ONLY_SEASONALITY = True
N_SEASON_ALTERNATIVES = 3      # runner-up habitat sets drawn faintly on the plot
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

C_BEST = "firebrick"
C_OBS = "black"

PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"
    ],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

# Figure and marker sizes (journal double-column width is about 7.2 in).
FIG_SIZE = (7.2, 3.3)
OBS_DOT_SIZE = 16        # scatter area (points^2) for DHS observations
LL_MARKER_SIZE = 4       # marker size for the likelihood profile
BEST_MARKER_SIZE = 45    # scatter area for the best-fit xTLH marker
SAVE_DPI = 600


# ---------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------

def _parse_dates(series):
    """Parse M/D/YYYY dates; fall back to pandas date parsing."""
    try:
        return pd.to_datetime(series, format="%m/%d/%Y")
    except (ValueError, TypeError):
        return pd.to_datetime(series)


def month_midpoint_date(year, month):
    """Plot position: middle of the model month (flat 365-day calendar)."""
    month = int(month)
    mid_day = MONTH_START[month] + (MONTH_LEN[month] - 1) / 2.0
    return (
        pd.Timestamp(year=int(year), month=1, day=1)
        + pd.to_timedelta(mid_day, unit="D")
    )


def _is_complete(series):
    return (
        series.astype(str).str.strip().str.lower()
        .isin(["true", "1", "t", "yes", "y"])
    )


def load_dhs_data(csv_path):
    """
    Load DHS surveys and assign each one the calendar year and month of its
    fieldwork midpoint. The simulation is matched on (year, month) only.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")

    required = [
        "fieldwork_start", "fieldwork_end", "DHS_n", "DHS_pos"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DHS file is missing columns: {missing}")

    for col in ["fieldwork_start", "fieldwork_end"]:
        df[col] = _parse_dates(df[col])

    midpoint = (
        df["fieldwork_start"]
        + (df["fieldwork_end"] - df["fieldwork_start"]) / 2
    )

    df["date"] = midpoint
    df["year"] = midpoint.dt.year.astype(int)
    df["month"] = midpoint.dt.month.astype(int)

    # Plot position on the same flat calendar as the simulation points.
    df["plot_date"] = [
        month_midpoint_date(y, m) for y, m in zip(df["year"], df["month"])
    ]

    in_range = df["year"].between(
        PICKUP_START_YEAR, PICKUP_END_YEAR
    )
    if not in_range.all():
        print(
            f"Dropping {int((~in_range).sum())} survey(s) outside "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR}."
        )

    df = df.loc[in_range].copy()
    df["DHS_n"] = pd.to_numeric(df["DHS_n"], errors="raise")
    df["DHS_pos"] = pd.to_numeric(df["DHS_pos"], errors="raise")
    df["PfPR"] = df["DHS_pos"] / df["DHS_n"]

    if df.empty:
        raise ValueError("No DHS surveys remain after date filtering.")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Simulation loading
# ---------------------------------------------------------------------

def _model_calendar_cols(df):
    """analyzer_Pickup13yr.py writes the midpoint model year/month as 'year'/'month';
    the PREG analyzer writes 'model_year'/'model_month'. Accept both."""
    if "model_year" not in df.columns and {"year", "month"} <= set(df.columns):
        df = df.rename(columns={"year": "model_year", "month": "model_month"})
    return df


def load_sim_per_run(csv_path):
    """
    Load one row per simulation run x model year x model month.

    The Run_Number and all habitat parameter tags are retained so that
    each simulation run is scored independently, as in the reference code.
    """
    global RUN_KEYS, SEL_KEYS
    df = pd.read_csv(csv_path)
    df.columns = [c.replace(" U5", "") for c in df.columns]
    df = _model_calendar_cols(df)

    sweep_vars = [v for v in IRS_VARS if v in df.columns and df[v].notna().all()]
    sweep_vars += [v for v in DECAY_VARS if v in df.columns and df[v].notna().all()]
    if sweep_vars:
        RUN_KEYS = [XCOL] + OTHER_VARS + sweep_vars + [SEED_VAR]
        SEL_KEYS = [XCOL] + sweep_vars
        print(f"IRS/decay sweep tags found: selecting jointly over {SEL_KEYS}")
    if USE_SEASONALITY:
        # Habitat mix sets seasonality, so score each full parameter set.
        SEL_KEYS = SEL_KEYS[:1] + OTHER_VARS + SEL_KEYS[1:]
        print(f"Seasonality on: scoring each parameter set {SEL_KEYS}")

    missing_tags = [c for c in RUN_KEYS if c not in df.columns]
    if missing_tags:
        raise ValueError(
            f"Missing sweep tag columns {missing_tags}. "
            f"Columns present: {list(df.columns)}"
        )

    required = ["model_year", "model_month", "agebin", "PfPR", "Pop"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Simulation CSV is missing columns: {missing}. "
            "model_year / model_month are written by the updated pickup "
            "analyzer; re-run it on this experiment."
        )

    # Label rows by model year and month; the analyzer's report-year column
    # ('Year') is dropped so it cannot be confused with model_year.
    df = (
        df.drop(columns=["Year"], errors="ignore")
        .rename(columns={"model_year": "year", "model_month": "month"})
    )

    keep = (
        df["month"].between(1, N_MONTHS)
        & df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
        & np.isin(
            np.round(df["agebin"].astype(float), 6),
            np.round(np.asarray(SURVEY_AGEBINS, dtype=float), 6),
        )
    )

    # Exclude partial reporting intervals when the flag exists.
    if "is_complete_interval" in df.columns:
        keep &= _is_complete(df["is_complete_interval"])
    else:
        print("WARNING: no is_complete_interval column; partial intervals "
              "are averaged into their month.")

    df = df.loc[keep].copy()

    if df.empty:
        raise ValueError(
            f"No simulation rows for agebin {SURVEY_AGEBINS} in "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR}. Check the analyzer "
            "age bins and reporting intervals."
        )

    # Ensure numeric inputs.
    for col in ["PfPR", "Pop"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Pop"].gt(0) & df["PfPR"].notna()].copy()
    df["npos_unrounded"] = df["PfPR"] * df["Pop"]

    month_cols = RUN_KEYS + ["year", "month"]
    interval_cols = (
        ["interval_start_day"] if "interval_start_day" in df.columns else []
    )

    # Step 1: within each reporting interval, combine selected age rows
    # (for the usual single 0.5-5y bin this leaves the row unchanged).
    per_interval = (
        df.groupby(month_cols + interval_cols, as_index=False)
        .agg(
            Pop=("Pop", "sum"),
            npos_unrounded=("npos_unrounded", "sum"),
            PfPR=("PfPR", "mean"),
        )
    )

    # Step 2: one row per run x year x month. If several intervals fall in the
    # same model month they are averaged, never summed.
    n_per_month = per_interval.groupby(month_cols).size()
    if (n_per_month > 1).any():
        print(
            f"NOTE: {int((n_per_month > 1).sum())} run-months contain more "
            "than one reporting interval; intervals are averaged."
        )

    run_df = (
        per_interval.groupby(month_cols, as_index=False)
        .agg(
            Pop=("Pop", "mean"),
            npos_unrounded=("npos_unrounded", "mean"),
            PfPR=("PfPR", "mean"),
        )
    )

    # Reference script convention: round simulated positive counts.
    run_df["npos"] = run_df["npos_unrounded"].round(0)
    run_df["sim_PfPR"] = run_df["npos"] / run_df["Pop"]

    # Keep PfPR as the reported prevalence for trajectory plots; likelihood
    # uses the rounded npos and Pop, as in the reference script.
    return run_df


# ---------------------------------------------------------------------
# Likelihood scoring
# ---------------------------------------------------------------------

_ll_mode = None  # "per_survey", "sum" or "mean"


def _detect_ll_mode():
    """
    Detect whether beta_binomial returns per-survey values, a sum, or a mean.
    """
    one = beta_binomial(
        np.array([50.0]), np.array([1000.0]),
        np.array([10.0]), np.array([120.0])
    )
    two = beta_binomial(
        np.array([50.0, 50.0]), np.array([1000.0, 1000.0]),
        np.array([10.0, 10.0]), np.array([120.0, 120.0])
    )

    if np.size(two) > 1:
        return "per_survey"

    one_value = float(np.sum(one))
    two_value = float(two)

    if np.isclose(two_value, 2 * one_value, rtol=1e-6):
        return "sum"
    if np.isclose(two_value, one_value, rtol=1e-6):
        return "mean"

    raise RuntimeError(
        "Cannot determine beta_binomial output mode: "
        f"one={one_value}, two={two_value}."
    )


def mean_log_likelihood(n_obs, n_sim, pos_obs, pos_sim):
    """Return mean beta-binomial log-likelihood across DHS surveys."""
    global _ll_mode

    if _ll_mode is None:
        _ll_mode = _detect_ll_mode()
        print(
            f"beta_binomial returns: {_ll_mode}; "
            "reporting mean log-likelihood across surveys."
        )

    values = beta_binomial(n_obs, n_sim, pos_obs, pos_sim)

    if _ll_mode == "per_survey":
        return float(np.mean(values))
    if _ll_mode == "sum":
        return float(values) / len(n_obs)
    return float(values)


def score_xtlh(sim_df, data_df):
    """
    Score each complete simulation run separately, then average run-level
    log-likelihoods by xTLH across scale_constant, scale_temp_rain,
    scale_water_veg and Run_Number.

    Returns:
        score_df: one row per xTLH, with mean LL and run counts.
        run_scores: one row per individual simulation run.
    """
    rows = []
    n_skipped = 0
    n_surveys = len(data_df)

    for keys, run in sim_df.groupby(RUN_KEYS, sort=False):

        # Each run must contain exactly one row for each DHS year/month.
        try:
            merged = data_df.merge(
                run[
                    ["year", "month", "Pop", "npos", "sim_PfPR"]
                ],
                on=["year", "month"],
                how="left",
                validate="many_to_one",
            )
        except pd.errors.MergeError:
            n_skipped += 1
            continue

        missing = merged[
            ["Pop", "npos", "sim_PfPR"]
        ].isna().any(axis=1)

        n_missing = int(missing.sum())

        # Do not score partial runs: all DHS surveys must be represented.
        if len(merged) != n_surveys or n_missing > 0:
            n_skipped += 1
            continue

        ll = mean_log_likelihood(
            merged["DHS_n"].to_numpy(dtype=float),
            merged["Pop"].to_numpy(dtype=float),
            merged["DHS_pos"].to_numpy(dtype=float),
            merged["npos"].to_numpy(dtype=float),
        )

        rows.append((*keys, ll, n_missing))

    if not rows:
        raise ValueError(
            "No complete simulation run matched every DHS survey. "
            "Check survey year/month and simulation output coverage."
        )

    run_scores = pd.DataFrame(
        rows,
        columns=RUN_KEYS + ["ll", "n_surveys_unmatched"]
    )

    # Reference method: mean run-level likelihood for each xTLH
    # (or each full parameter set when seasonality is on).
    score_df = (
        run_scores.groupby(SEL_KEYS, as_index=False)
        .agg(
            ll=("ll", "mean"),
            n_runs=("ll", "size"),
            sd_ll=("ll", "std"),
        )
        .sort_values("ll", ascending=False)
        .reset_index(drop=True)
    )

    print(
        f"Scored {len(run_scores)} complete runs; "
        f"skipped {n_skipped} incomplete or duplicated runs."
    )

    return score_df, run_scores


def score_parameter_sets(run_scores):
    """
    Mean run-level log-likelihood for each full habitat parameter set
    (xTLH + OTHER_VARS), averaged over seeds (Run_Number). Used only for
    reporting; xTLH selection still uses the mean across all other
    parameters and seeds.
    """
    # SEL_KEYS already contains OTHER_VARS when seasonality is on; avoid
    # duplicate group keys.
    keys = list(dict.fromkeys(SEL_KEYS + OTHER_VARS))
    return (
        run_scores.groupby(keys, as_index=False)
        .agg(
            ll=("ll", "mean"),
            n_runs=("ll", "size"),
        )
        .sort_values("ll", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Diagnostics and plotting
# ---------------------------------------------------------------------

def _add_plot_dates(sim_df):
    """Add a mid-month plot date from model year and month."""
    sim = sim_df.copy()
    sim["date"] = [
        month_midpoint_date(y, m) for y, m in zip(sim["year"], sim["month"])
    ]
    return sim


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def plot_xtlh(sim_df, data_df, score_df, out_png):
    """
    Left: mean PfPR trajectory for each xTLH, averaged across all other
          habitat parameters and seeds. Best xTLH is dark red.
    Right: mean run-level log-likelihood by xTLH.
    """
    # Average trajectories across all other parameters and seeds at each
    # xTLH, matching the reference plotting approach.
    plot_group_cols = ["year", "month", XCOL]

    if score_df["ll"].dropna().empty:
        raise ValueError("All xTLH likelihood scores are missing.")
    best = pick_best(score_df)

    # When selecting over full parameter sets, show xTLH trajectories / LL
    # profile at the best values of the other selected parameters.
    fixed_vars = [v for v in SEL_KEYS if v != XCOL]
    fixed_label = ""
    if fixed_vars:
        sel = np.all([np.isclose(sim_df[v], best[v]) for v in fixed_vars], axis=0)
        sim_df = sim_df[sel]
        sel = np.all([np.isclose(score_df[v], best[v]) for v in fixed_vars], axis=0)
        score_df = score_df[sel]
        short = {"scale_constant": "const", "scale_temp_rain": "rain",
                 "scale_water_veg": "veg", "irs_kill_scale": "IRS kill x",
                 "irs_decay_scale": "decay x", "irs_coverage_scale": "cov x"}
        fixed_label = ", ".join(f"{short.get(v, v)} {best[v]:g}" for v in fixed_vars)

    sim_agg = (
        sim_df.groupby(plot_group_cols, as_index=False)
        .agg(PfPR=("PfPR", "mean"))
    )
    sim_agg = _add_plot_dates(sim_agg)

    obs = data_df.copy()
    obs["date"] = obs["plot_date"]

    with mpl.rc_context(PUB_RC):
        if sns is not None:
            sns.set_style("whitegrid", {"axes.linewidth": 0.5})

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=FIG_SIZE
        )

        # Simulated trajectories: all non-best xTLH values faint.
        for x, group in sim_agg.groupby(XCOL):
            if np.isclose(x, best[XCOL]):
                continue

            group = group.sort_values("date")
            ax1.plot(
                group["date"], group["PfPR"],
                color=C_BEST, alpha=0.2, linewidth=0.6
            )

        # Best xTLH trajectory.
        best_group = sim_agg[
            np.isclose(sim_agg[XCOL], best[XCOL])
        ].sort_values("date")

        ax1.plot(
            best_group["date"], best_group["PfPR"],
            color=C_BEST, linewidth=1.5,
            label=f"Best fit (xTLH = {best[XCOL]:.1f})"
        )

        ax1.scatter(
            obs["date"], obs["PfPR"],
            s=OBS_DOT_SIZE, color=C_OBS, edgecolor="white",
            linewidth=0.4, zorder=5,
            label="Observed (DHS)"
        )

        ax1.set_xlabel("Year")
        ax1.set_ylabel("PfPR (children 6–59 months)")
        ax1.set_title("Observed vs simulated (red = best fit)")
        ax1.legend(frameon=False, loc="best")
        _clean_axes(ax1)

        # Mean run-level log-likelihood by xTLH.
        ordered = score_df.sort_values(XCOL)

        ax2.plot(
            ordered[XCOL], ordered["ll"],
            "-o", color=C_BEST, markersize=LL_MARKER_SIZE,
            linewidth=1.2, label="Mean log-likelihood"
        )

        ax2.scatter(
            [best[XCOL]], [best["ll"]],
            s=BEST_MARKER_SIZE, color=C_BEST, edgecolor="black",
            linewidth=0.7, zorder=5,
            label=(
                f"Best: xTLH = {best[XCOL]:.1f}, "
                f"LL = {best['ll']:.2f}"
            )
        )

        ax2.set_xlabel("Larval habitat multiplier (xTLH)")
        ax2.set_ylabel("Mean log-likelihood")
        ax2.set_title("Larger value = better fit" )
        ax2.legend(frameon=False, loc="best")
        _clean_axes(ax2)

        fig.tight_layout()
        fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {out_png}")


def _norm_by_year(df, value_col, year_col):
    mean = df.groupby(year_col)[value_col].transform("mean")
    return df[value_col] / mean.where(mean > 0)


def load_obs_season_monthly(csv_path):
    """Observed U5 monthly rates by year, each year scaled to mean 1.

    Monthly routine counts -> annualised rate per child, on the same scale as
    EMOD's 'Annual Clinical Incidence' per 30-day block:
        rate = cases / Pop_5 x 365 / days_in_month
    Dividing by days removes the 28-31 day month-length effect; dividing each
    year by its own mean removes level changes (testing/reporting, IRS), so
    only the within-year shape is compared. Returns columns year, m, idx."""
    obs = pd.read_csv(csv_path)
    obs.columns = [c.strip() for c in obs.columns]
    if "Age_Group" in obs.columns:
        obs = obs[obs["Age_Group"].astype(str).str.strip().str.upper() == "U5"]
    obs = obs.rename(columns={"Year": "year", "Month": "month"})
    obs["m"] = obs["month"].astype(str).str.strip().map(
        {m: i + 1 for i, m in enumerate(MONTHS)})
    obs[OBS_U5_COL] = pd.to_numeric(obs[OBS_U5_COL], errors="coerce")
    use_pop = bool(OBS_POP_COL) and OBS_POP_COL in obs.columns
    if use_pop:
        obs[OBS_POP_COL] = pd.to_numeric(obs[OBS_POP_COL], errors="coerce")
    else:
        print("NOTE: no population column; observed rate is annualised "
              "per month but not divided by population.")
    obs = obs[obs["year"].between(*SEASON_YEARS)].dropna(subset=["m", OBS_U5_COL])
    agg = {"cases": (OBS_U5_COL, "sum")}
    if use_pop:
        agg["pop"] = (OBS_POP_COL, "mean")
    obs = obs.groupby(["year", "m"], as_index=False).agg(**agg)
    full = obs.groupby("year")["m"].nunique()
    obs = obs[obs["year"].isin(full[full == 12].index)].copy()
    days = obs["m"].astype(int).map(MONTH_LEN)
    obs["rate"] = obs["cases"] / (obs["pop"] if use_pop else 1.0) * DAYS_PER_YEAR / days
    obs["idx"] = _norm_by_year(obs, "rate", "year")
    return obs[["year", "m", "rate", "idx"]]


def load_obs_season(csv_path):
    """Observed U5 seasonal profile (12 values, mean 1) averaged over SEASON_YEARS."""
    obs = load_obs_season_monthly(csv_path)
    clim = obs.groupby("m")["idx"].mean().reindex(range(1, 13))
    return (clim / clim.mean()).to_numpy()


def _safe_r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _choose_lag(clim, obs_clim):
    """Months to move the simulated profile earlier (see SEASON_LAG_MONTHS)."""
    if SEASON_LAG_MONTHS == "auto":
        rs = [np.corrcoef(np.roll(clim, -L), obs_clim)[0, 1] for L in range(SEASON_MAX_LAG + 1)]
        return int(np.nanargmax(rs))
    return int(SEASON_LAG_MONTHS or 0)


def _shift_months(g, lag):
    """Relabel simulated (year, month) rows `lag` months earlier, so sim month m+lag
    lines up with observed month m (year-by-year comparison)."""
    if not lag:
        return g
    g = g.copy()
    t = g["year"] * 12 + (g["month"] - 1) - int(lag)
    g["year"], g["month"] = t // 12, t % 12 + 1
    return g


def season_scores(csv_path, obs_clim, keys, obs_monthly=None):
    """Seasonal fit of simulated U5 clinical incidence for each parameter set.

    Simulated: 'Cases' is annualised clinical incidence per person in each
    complete 30-day block (block k = model month k). U5 = population-weighted
    mean over the 0-0.25 and 0.25-5 bins. Seed-mean, then each year divided by
    its own mean (same scaling as the observed series).

    Scores (all on the within-year shape):
      season_r          r of the Jan-Dec mean profiles (climatology)
      season_r_years    r over every matched year-month (year-by-year shape)
      season_rmse       RMSE of the climatologies (mean-1 scale)
      sim_peak_month, peak_diff_months, amplitude_ratio (peak-trough range ratio)
      peak_hit_frac     share of years whose simulated peak is within
                        SEASON_MAX_PEAK_DIFF months of that year's observed peak
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.replace(" U5", "") for c in df.columns]
    df = _model_calendar_cols(df)
    for c in ("model_year", "model_month"):
        if c not in df.columns:
            raise ValueError(
                f"Simulation CSV is missing '{c}'; re-run the updated pickup analyzer."
            )
    ab = df["agebin"].astype(float)
    sel = ((ab <= U5_AGEBIN_MAX + 1e-9)
           & df["model_month"].between(1, 12)
           & df["model_year"].between(*SEASON_YEARS))
    if "is_complete_interval" in df.columns:
        sel &= _is_complete(df["is_complete_interval"])
    df = df[sel].copy()
    df["cxp"] = df["Cases"] * df["Pop"]
    seed_keys = keys + [SEED_VAR, "model_year", "model_month"]
    u5 = df.groupby(seed_keys, as_index=False)[["cxp", "Pop"]].sum()
    u5["Cases"] = u5["cxp"] / u5["Pop"].where(u5["Pop"] > 0)
    prof = (u5.groupby(keys + ["model_year", "model_month"], as_index=False)["Cases"].mean()
            .rename(columns={"model_year": "year", "model_month": "month"}))
    prof["idx"] = prof.groupby(keys + ["year"])["Cases"].transform(
        lambda v: v / v.mean() if v.mean() > 0 else np.nan)

    obs_peak = int(np.argmax(obs_clim)) + 1
    obs_year_peak = None
    if obs_monthly is not None:
        ov = obs_monthly.dropna(subset=["idx"])
        obs_year_peak = ov.loc[ov.groupby("year")["idx"].idxmax()].set_index("year")["m"]
    rows = []
    for k, g in prof.groupby(keys):
        k = np.atleast_1d(k)
        clim = g.groupby("month")["idx"].mean().reindex(range(1, 13)).to_numpy()
        if not np.isfinite(clim).all():
            rows.append((*k,) + (np.nan,) * 9)
            continue
        clim = clim / clim.mean()
        r_raw = float(np.corrcoef(clim, obs_clim)[0, 1])
        lag = _choose_lag(clim, obs_clim)
        clim = np.roll(clim, -lag)                  # sim month m+lag compared with obs month m
        peak = int(np.argmax(clim)) + 1
        diff = (peak - obs_peak + 6) % 12 - 6
        amp = (clim.max() - clim.min()) / (obs_clim.max() - obs_clim.min())
        rmse = float(np.sqrt(np.mean((clim - obs_clim) ** 2)))
        r_years = hit = np.nan
        if obs_monthly is not None:
            g = _shift_months(g, lag)
            m = g.merge(obs_monthly, left_on=["year", "month"], right_on=["year", "m"],
                        suffixes=("_sim", "_obs")).dropna(subset=["idx_sim", "idx_obs"])
            r_years = _safe_r(m["idx_sim"], m["idx_obs"])
            gv = g.dropna(subset=["idx"])
            sim_year_peak = gv.loc[gv.groupby("year")["idx"].idxmax()].set_index("year")["month"]
            yrs = sim_year_peak.index.intersection(obs_year_peak.index)
            if len(yrs):
                d = ((sim_year_peak[yrs] - obs_year_peak[yrs] + 6) % 12 - 6).abs()
                hit = float((d <= SEASON_MAX_PEAK_DIFF).mean())
        rows.append((*k, float(np.corrcoef(clim, obs_clim)[0, 1]), r_years, rmse,
                     peak, diff, float(amp), hit, r_raw, lag))
    cols = keys + ["season_r", "season_r_years", "season_rmse", "sim_peak_month",
                   "peak_diff_months", "amplitude_ratio", "peak_hit_frac",
                   "season_r_raw", "season_lag"]
    return pd.DataFrame(rows, columns=cols), prof


def season_ok(score_df):
    """Seasonality filter: climatology r and peak month close to observed."""
    ok = score_df["season_r"] >= SEASON_MIN_R
    ok &= score_df["peak_diff_months"].abs() <= SEASON_MAX_PEAK_DIFF
    return ok


def pick_best(score_df):
    """Highest PfPR LL among sets that pass the seasonality filter
    (season_r >= SEASON_MIN_R and |peak diff| <= SEASON_MAX_PEAK_DIFF);
    falls back to the highest LL if no set passes."""
    valid = score_df.dropna(subset=["ll"])
    if "season_r" in valid.columns:
        ok = valid[season_ok(valid)]
        if ok.empty:
            print(f"\nWARNING: no parameter set passes the seasonality filter "
                  f"(r >= {SEASON_MIN_R}, |peak diff| <= {SEASON_MAX_PEAK_DIFF}); "
                  f"best season_r = {valid['season_r'].max():.2f}. Selecting on PfPR LL only.")
        else:
            print(f"\n{len(ok)} of {len(valid)} parameter sets pass the seasonality filter.")
            valid = ok
    return valid.loc[valid["ll"].idxmax()]


def _clim_for(prof, row, keys, shifted=True):
    sel = np.all([np.isclose(prof[k], row[k]) for k in keys], axis=0)
    clim = prof[sel].groupby("month")["idx"].mean().reindex(range(1, 13)).to_numpy()
    clim = clim / np.nanmean(clim)
    lag = int(row.get("season_lag", 0) or 0) if shifted else 0
    return np.roll(clim, -lag)


def _habitat_label(row):
    short = {"scale_constant": "const", "scale_temp_rain": "rain", "scale_water_veg": "veg",
             "spline_lag": "lag", "spline_power": "power", "spline_scale": "scale"}
    return ", ".join(f"{short.get(k, k)} {row[k]:g}" for k in OTHER_VARS)


def plot_season(prof, obs_clim, best, keys, out_png, score_df=None):
    """Observed vs simulated U5 seasonal profile (each year scaled to mean 1,
    averaged over SEASON_YEARS): best-fit habitat set plus runners-up."""
    x = np.arange(1, 13)
    with mpl.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        if score_df is not None and N_SEASON_ALTERNATIVES > 0:
            pool = score_df.dropna(subset=["season_r"])
            if "season_pass" in pool.columns and pool["season_pass"].any():
                pool = pool[pool["season_pass"]]
            pool = pool.sort_values("ll", ascending=False)
            pool = pool[~np.all([np.isclose(pool[k], best[k]) for k in keys], axis=0)]
            for i, (_, row) in enumerate(pool.head(N_SEASON_ALTERNATIVES).iterrows()):
                ax.plot(x, _clim_for(prof, row, keys), "-", color="0.6", linewidth=0.8,
                        alpha=0.8, label="Runners-up" if i == 0 else None)
        ax.plot(x, obs_clim, "-o", color=C_OBS, markersize=3, linewidth=1.3,
                label="Observed U5 confirmed cases")
        lag = int(best.get("season_lag", 0) or 0)
        if lag:
            ax.plot(x, _clim_for(prof, best, keys, shifted=False), "--", color=C_BEST,
                    linewidth=0.9, alpha=0.6,
                    label=f"Best fit, unshifted (r = {best['season_r_raw']:.2f})")
        shift_txt = f", shifted {lag} mo earlier" if lag else ""
        ax.plot(x, _clim_for(prof, best, keys), "-o", color=C_BEST, markersize=3,
                linewidth=1.3,
                label=f"Best fit{shift_txt}: {_habitat_label(best)}\n(r = {best['season_r']:.2f}, "
                      f"peak {MONTHS[int(best['sim_peak_month']) - 1][:3]})")
        ax.axhline(1, color="0.85", linewidth=0.6)
        ax.set_xticks(x, [m[0] for m in MONTHS])
        ax.set_ylabel("Relative incidence (year mean = 1)")
        ax.set_title(f"U5 seasonality {SEASON_YEARS[0]}-{SEASON_YEARS[1]}, "
                     f"xTLH {best[XCOL]:.3g}")
        ax.legend(frameon=False, loc="upper left", fontsize=6)
        _clean_axes(ax)
        fig.tight_layout()
        fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_png}")


def plot_irs_grid(score_df, out_png):
    """Mean LL over the IRS grid at the best xTLH / habitat: one heatmap
    (killing scale x decay scale) per coverage scale."""
    best = pick_best(score_df)
    fixed = [k for k in SEL_KEYS if k not in IRS_VARS]
    at = score_df[np.all([np.isclose(score_df[k], best[k]) for k in fixed], axis=0)]
    covs = sorted(at["irs_coverage_scale"].unique())
    vmin, vmax = at["ll"].min(), at["ll"].max()

    with mpl.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, len(covs), figsize=(2.2 * len(covs) + 0.8, 2.6),
                                 squeeze=False, constrained_layout=True)
        for ax, c in zip(axes[0], covs):
            grid = (at[np.isclose(at["irs_coverage_scale"], c)]
                    .pivot_table(index="irs_kill_scale", columns="irs_decay_scale",
                                 values="ll"))
            im = ax.imshow(grid.values, origin="lower", cmap="viridis",
                           vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(grid.shape[1]), [f"{v:g}" for v in grid.columns])
            ax.set_yticks(range(grid.shape[0]), [f"{v:g}" for v in grid.index])
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    ax.text(j, i, f"{grid.values[i, j]:.1f}", ha="center", va="center",
                            fontsize=6.5, color="white")
            ax.set_title(f"Coverage x {c:g}")
        axes[0][0].set_ylabel("IRS killing multiplier")
        fig.colorbar(im, ax=axes[0].tolist(), label="Mean log-likelihood", shrink=0.9)
        fig.supxlabel("Decay time-constant multiplier", fontsize=9)
        fig.suptitle(f"IRS grid at best xTLH = {best[XCOL]:.1f}", fontsize=9)
        fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_png}")


def print_diagnostics(sim_df, data_df):
    """Print survey-to-simulation mapping and run coverage diagnostics."""
    print("\nSurvey-to-simulation mapping (matched on year and month):")
    cols = [
        "date", "year", "month",
        "DHS_n", "DHS_pos", "PfPR"
    ]
    print(data_df[cols].to_string(index=False))

    print("\nSimulation model months:")
    print(sorted(sim_df["month"].unique()))

    # Check whether every observed survey year/month occurs in the simulation.
    available = sim_df[["year", "month"]].drop_duplicates()
    check = data_df[["year", "month"]].merge(
        available,
        on=["year", "month"],
        how="left",
        indicator=True,
    )
    unmatched = check[check["_merge"] == "left_only"]

    if len(unmatched):
        print(
            "\nWARNING: DHS survey year/months absent from all simulations:"
        )
        print(
            unmatched[["year", "month"]]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print(
            "\nEvery DHS survey year/month is represented in "
            "the simulation output."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def report_best_habitat(score_df, best):
    """Print and save the best (scale_constant, scale_temp_rain, scale_water_veg)
    from HABITAT_GRID: highest DHS PfPR LL among sets passing the seasonality
    filter; also the best set on seasonality alone (highest climatology r)."""
    sc = score_df.dropna(subset=["season_r"]).copy()
    grid_ok = np.ones(len(sc), bool)
    for k, vals in HABITAT_GRID.items():
        grid_ok &= np.isin(np.round(sc[k].astype(float), 6), np.round(vals, 6))
    missing = len(list(product(*HABITAT_GRID.values()))) - sc[grid_ok][OTHER_VARS].drop_duplicates().shape[0]
    if missing:
        print(f"NOTE: {missing} habitat set(s) of HABITAT_GRID have no seasonality score.")
    by_season = sc.sort_values("season_r", ascending=False).iloc[0]
    out = pd.DataFrame([dict(criterion="selected (seasonality filter, then best PfPR LL)",
                             **{k: best[k] for k in [XCOL] + OTHER_VARS},
                             **{c: best.get(c) for c in ["ll", "season_r", "season_r_raw",
                                "season_lag", "season_r_years",
                                "season_rmse", "sim_peak_month", "peak_diff_months",
                                "amplitude_ratio", "peak_hit_frac"]}),
                        dict(criterion="best seasonality only (highest season_r)",
                             **{k: by_season[k] for k in [XCOL] + OTHER_VARS},
                             **{c: by_season.get(c) for c in ["ll", "season_r", "season_r_raw",
                                "season_lag", "season_r_years",
                                "season_rmse", "sim_peak_month", "peak_diff_months",
                                "amplitude_ratio", "peak_hit_frac"]})])
    out.to_csv(os.path.join(OUTPUT_DIR, "best_fit_habitat.csv"), index=False)
    print("\nBEST-FIT HABITAT (grid: " + "; ".join(f"{k} {v}" for k, v in HABITAT_GRID.items()) + ")")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nSet in Site_Setup_Final.py for the next run:")
    names = {"scale_constant": "SCALE_CONSTANT", "scale_temp_rain": "SCALE_TEMP_RAIN",
             "scale_water_veg": "SCALE_WATER_VEG", "spline_lag": "SPLINE_LAG_MONTHS",
             "spline_power": "SPLINE_POWER", "spline_scale": "SPLINE_SCALE"}
    for k in OTHER_VARS:
        print(f"  {names[k]} = [{best[k]:g}]")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_df = load_sim_per_run(SIM_PFPR_CSV)
    data_df = load_dhs_data(
        os.path.join(INPUT_DIR, DHS_FILE)
    )

    n_runs = sim_df[RUN_KEYS].drop_duplicates().shape[0]

    print(
        f"\nxTLH values: {sim_df[XCOL].nunique()} | "
        f"simulation runs: {n_runs} | "
        f"DHS surveys: {len(data_df)}"
    )

    print_diagnostics(sim_df, data_df)

    score_df, run_scores = score_xtlh(sim_df, data_df)

    prof = obs_clim = None
    if USE_SEASONALITY:
        obs_monthly = load_obs_season_monthly(os.path.join(INPUT_DIR, OBS_U5_FILE))
        obs_clim = load_obs_season(os.path.join(INPUT_DIR, OBS_U5_FILE))
        print("\nObserved U5 seasonal profile (Jan-Dec, mean 1):",
              np.round(obs_clim, 2), f"| peak month {int(np.argmax(obs_clim)) + 1}")
        seas, prof = season_scores(SIM_PFPR_CSV, obs_clim, SEL_KEYS, obs_monthly)
        score_df = score_df.merge(seas, on=SEL_KEYS, how="left")
        score_df["season_pass"] = season_ok(score_df)
        score_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_scores_season.csv"), index=False)
        print("\nSeasonality ranking (best climatology r first, top 15):")
        cols = SEL_KEYS + ["season_r", "season_r_raw", "season_lag", "season_r_years",
                           "season_rmse", "sim_peak_month",
                           "peak_diff_months", "amplitude_ratio", "peak_hit_frac", "ll"]
        print(score_df.sort_values("season_r", ascending=False)[cols].head(15)
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    print(f"\nMean run-level log-likelihood by {SEL_KEYS} (best first, top 20):")
    print(score_df.head(20).to_string(index=False))

    best = pick_best(score_df)

    if USE_SEASONALITY:
        report_best_habitat(score_df, best)

    print("\nBEST-FITTING xTLH")
    print(f"  xTLH: {best[XCOL]:.4g}")
    for var in SEL_KEYS[1:]:
        print(f"  {var}: {best[var]:.4g}")
    print(f"  Mean log-likelihood: {best['ll']:.4f}")
    print(f"  Complete runs contributing: {int(best['n_runs'])}")
    if "season_r" in best.index:
        print(f"  Lag adjustment: simulated profile moved {int(best['season_lag'])} month(s) "
              f"earlier (unshifted r = {best['season_r_raw']:.2f})")
        print(f"  U5 seasonality: r = {best['season_r']:.2f} (year-by-year "
              f"{best['season_r_years']:.2f}), RMSE {best['season_rmse']:.2f}, peak month "
              f"{int(best['sim_peak_month'])} (diff {int(best['peak_diff_months']):+d}), "
              f"amplitude ratio {best['amplitude_ratio']:.2f}, "
              f"peak right in {best['peak_hit_frac']:.0%} of years")

    # Best combination of the other habitat parameters at the best xTLH
    # (mean log-likelihood over seeds).
    param_df = score_parameter_sets(run_scores)
    at_best = param_df[np.all([np.isclose(param_df[k], best[k]) for k in SEL_KEYS], axis=0)]
    best_combo = at_best.iloc[0]

    print("\nBEST-FITTING COMBINATION (at the best xTLH, mean over seeds)")
    print(f"  {XCOL}: {best_combo[XCOL]:.4g}")
    for var in OTHER_VARS:
        print(f"  {var}: {best_combo[var]:.4g}")
    print(f"  Mean log-likelihood: {best_combo['ll']:.4f}")
    print(f"  Seeds contributing: {int(best_combo['n_runs'])}")

    print("\nTop 10 combinations at the best xTLH:")
    print(at_best.head(10).to_string(index=False))


    # Save both the xTLH-level scores and individual run-level scores.
    score_df.to_csv(
        os.path.join(OUTPUT_DIR, "calibration_scores_xTLH.csv"),
        index=False
    )
    run_scores.to_csv(
        os.path.join(OUTPUT_DIR, "calibration_scores_per_run.csv"),
        index=False
    )
    param_df.to_csv(
        os.path.join(OUTPUT_DIR, "calibration_scores_param_sets.csv"),
        index=False
    )

    if prof is not None:
        plot_season(prof, obs_clim, best, SEL_KEYS,
                    os.path.join(OUTPUT_DIR, "selection_U5_seasonality.png"), score_df)

    if PLOT_ONLY_SEASONALITY:
        return

    if all(v in SEL_KEYS for v in IRS_VARS):
        plot_irs_grid(score_df, os.path.join(OUTPUT_DIR, "selection_IRS_grid.png"))

    plot_xtlh(
        sim_df,
        data_df,
        score_df,
        os.path.join(OUTPUT_DIR, "selection.png")
    )


if __name__ == "__main__":
    main()