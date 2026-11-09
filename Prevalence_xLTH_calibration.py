"""
UWR xTLH calibration using the reference run-level likelihood method.

Method
------
1. Keep each EMOD run separate:
   (xTLH, scale_constant, scale_temp_rain, scale_water_veg, Run_Number).
2. Match each DHS survey to the 30-day EMOD reporting block containing its
   fieldwork midpoint, using the flat 365-day model calendar.
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
"""

import os

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

EXPERIMENT_NAME = "Pickup13yrs_xTLHfine"
ANALYZER_DIR = os.path.join(
    JOB_DIR, f"{EXPERIMENT_NAME}_AnalyzerOutput"
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

BLOCK_DAYS = 30
N_BLOCKS = 12
DAYS_PER_YEAR = 365

# Fixed-month-length EMOD calendar; must match the analyzer/pickup script.
MONTH_START = {
    1: 0, 2: 31, 3: 59, 4: 90,
    5: 120, 6: 151, 7: 181, 8: 212,
    9: 243, 10: 273, 11: 304, 12: 334,
}
FEB_28_MODEL_DAY = 58

# DHS children 6-59 months correspond to the 0.5-5 year age bin.
SURVEY_AGEBINS = (5.0,)

XCOL = "x_Temporary_Larval_Habitat"
OTHER_VARS = [
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
SEED_VAR = "Run_Number"
RUN_KEYS = [XCOL] + OTHER_VARS + [SEED_VAR]

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


def load_dhs_data(csv_path):
    """
    Load DHS surveys and map each fieldwork midpoint to the 30-day EMOD
    reporting block containing it.

    This deliberately does not match by Gregorian month: analyzer month/
    report_interval represents a 30-day block, not a calendar month.
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
    df["year"] = midpoint.dt.year

    model_day = (
        midpoint.dt.month.map(MONTH_START)
        + midpoint.dt.day - 1
    )

    leap_day = midpoint.dt.month.eq(2) & midpoint.dt.day.eq(29)
    model_day.loc[leap_day] = FEB_28_MODEL_DAY

    df["model_day"] = model_day.astype(int)

    # Position on the same flat 365-day calendar used by EMOD.
    df["plot_date"] = (
        pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01")
        + pd.to_timedelta(df["model_day"], unit="D")
    )

    df["block"] = (
        df["model_day"] // BLOCK_DAYS + 1
    ).astype(int)

    # The 13th interval is a 5-day tail and is not used.
    late = df["block"] > N_BLOCKS
    if late.any():
        print(
            f"Excluding {int(late.sum())} DHS survey(s) mapped to "
            "the incomplete 13th reporting interval."
        )
        df = df.loc[~late].copy()

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

def load_sim_per_run(csv_path):
    """
    Load one row per simulation run x year x 30-day block.

    The Run_Number and all habitat parameter tags are retained so that
    each simulation run is scored independently, as in the reference code.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.replace(" U5", "") for c in df.columns]

    missing_tags = [c for c in RUN_KEYS if c not in df.columns]
    if missing_tags:
        raise ValueError(
            f"Missing sweep tag columns {missing_tags}. "
            f"Columns present: {list(df.columns)}"
        )

    block_col = (
        "report_interval" if "report_interval" in df.columns
        else "month" if "month" in df.columns
        else None
    )
    if block_col is None:
        raise ValueError(
            "Simulation CSV must contain report_interval or month."
        )

    required = ["Year", block_col, "agebin", "PfPR", "Pop"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Simulation CSV is missing columns: {missing}")

    df = df.rename(columns={"Year": "year", block_col: "block"})

    keep = (
        df["block"].between(1, N_BLOCKS)
        & df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
        & np.isin(
            np.round(df["agebin"].astype(float), 6),
            np.round(np.asarray(SURVEY_AGEBINS, dtype=float), 6),
        )
    )

    # Exclude the incomplete interval 13 explicitly when the flag exists.
    if "is_complete_interval" in df.columns:
        complete = (
            df["is_complete_interval"]
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "t", "yes", "y"])
        )
        keep &= complete

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

    # If multiple selected age rows exist within a run/block, combine them
    # using population-weighted prevalence before calculating positives.
    # For the usual single 0.5-5y bin, this leaves the row unchanged.
    group_cols = RUN_KEYS + ["year", "block"]
    agg = {
        "Pop": "sum",
        "npos_unrounded": "sum",
        "PfPR": "mean",
    }

    df["npos_unrounded"] = df["PfPR"] * df["Pop"]

    if "interval_midpoint_model_time" in df.columns:
        agg["interval_midpoint_model_time"] = "mean"

    run_df = (
        df.groupby(group_cols, as_index=False)
        .agg(agg)
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

        # Each run must contain exactly one row for each DHS year/block.
        try:
            merged = data_df.merge(
                run[
                    ["year", "block", "Pop", "npos", "sim_PfPR"]
                ],
                on=["year", "block"],
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
            "Check survey-to-block mapping and simulation output coverage."
        )

    run_scores = pd.DataFrame(
        rows,
        columns=RUN_KEYS + ["ll", "n_surveys_unmatched"]
    )

    # Reference method: mean run-level likelihood for each xTLH.
    score_df = (
        run_scores.groupby(XCOL, as_index=False)
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
    return (
        run_scores.groupby([XCOL] + OTHER_VARS, as_index=False)
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
    """Add dates using analyzer model-time midpoints or block midpoints."""
    sim = sim_df.copy()
    year_start = pd.to_datetime(
        sim["year"].astype(int).astype(str) + "-01-01"
    )

    if (
        "interval_midpoint_model_time" in sim.columns
        and sim["interval_midpoint_model_time"].notna().all()
    ):
        days = (
            sim["interval_midpoint_model_time"]
            - sim["year"].astype(float)
        ) * DAYS_PER_YEAR
    else:
        days = BLOCK_DAYS * (sim["block"].astype(float) - 0.5)

    sim["date"] = year_start + pd.to_timedelta(days, unit="D")
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
    plot_group_cols = ["year", "block", XCOL]

    agg = {
        "PfPR": "mean",
    }
    if "interval_midpoint_model_time" in sim_df.columns:
        agg["interval_midpoint_model_time"] = "mean"

    sim_agg = (
        sim_df.groupby(plot_group_cols, as_index=False)
        .agg(agg)
    )
    sim_agg = _add_plot_dates(sim_agg)

    valid = score_df.dropna(subset=["ll"])
    if valid.empty:
        raise ValueError("All xTLH likelihood scores are missing.")

    best = valid.loc[valid["ll"].idxmax()]

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
        ax2.set_title("Larger value = better fit")
        ax2.legend(frameon=False, loc="best")
        _clean_axes(ax2)

        fig.tight_layout()
        fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {out_png}")


def print_diagnostics(sim_df, data_df):
    """Print survey-to-simulation mapping and run coverage diagnostics."""
    print("\nSurvey-to-simulation mapping:")
    cols = [
        "date", "model_day", "year", "block",
        "DHS_n", "DHS_pos", "PfPR"
    ]
    print(data_df[cols].to_string(index=False))

    print("\nSimulation reporting blocks:")
    print(sorted(sim_df["block"].unique()))

    # Check whether every observed survey block occurs in the simulation.
    available = sim_df[["year", "block"]].drop_duplicates()
    check = data_df[["year", "block"]].merge(
        available,
        on=["year", "block"],
        how="left",
        indicator=True,
    )
    unmatched = check[check["_merge"] == "left_only"]

    if len(unmatched):
        print(
            "\nWARNING: DHS survey blocks absent from all simulations:"
        )
        print(
            unmatched[["year", "block"]]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print(
            "\nEvery DHS survey year/block is represented in "
            "the simulation output."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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

    print("\nMean run-level log-likelihood by xTLH (best first):")
    print(score_df.to_string(index=False))

    best = score_df.loc[score_df["ll"].idxmax()]

    print("\nBEST-FITTING xTLH")
    print(f"  xTLH: {best[XCOL]:.4g}")
    print(f"  Mean log-likelihood: {best['ll']:.4f}")
    print(f"  Complete runs contributing: {int(best['n_runs'])}")

    # Best combination of the other habitat parameters at the best xTLH
    # (mean log-likelihood over seeds).
    param_df = score_parameter_sets(run_scores)
    at_best = param_df[np.isclose(param_df[XCOL], best[XCOL])]
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

    plot_xtlh(
        sim_df,
        data_df,
        score_df,
        os.path.join(OUTPUT_DIR, "selection.png")
    )


if __name__ == "__main__":
    main()
