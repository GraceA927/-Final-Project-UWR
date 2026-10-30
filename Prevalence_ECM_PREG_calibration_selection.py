"""
UWR prevalence calibration: beta-binomial log-likelihood over habitat
combinations, with the best fit chosen by how well the simulation represents
the survey points.

Aligned to MonthlyPfPRAnalyzer output (PfPR_ClinicalIncidence_monthly_pickup.csv).

Conventions
    - Calendar: EMOD uses a flat 365-day model year with fixed month lengths.
      Survey dates are mapped onto that calendar (Feb 29 -> Feb 28), NOT via
      Gregorian day-of-year, which shifts every date after Feb 29 in leap years.
    - Time matching: the analyzer writes `report_interval` (a 30-day REPORTING
      BLOCK, 1..13 per year) and `is_complete_interval`. Interval 13 is a 5-day
      tail flagged is_complete_interval = False and is dropped here, leaving
      blocks 1..12. (Older analyzer files with a `month` column are still
      accepted.) Each survey is assigned to the block containing its fieldwork
      midpoint: block = model_day // 30 + 1, and joined to the simulation on
      (year, block). Plot dates use the analyzer's interval_midpoint_model_time
      and model-calendar survey dates, so both sit on the same flat calendar.
    - Seeds: infected population (PfPR x Pop) is computed WITHIN each replicate
      first, then averaged. Simulated prevalence used everywhere (likelihood,
      Wilson-interval check, MAE, plots) is mean(npos) / mean(Pop)
      (`sim_PfPR_from_counts`), so it is consistent with the likelihood inputs.
    - Incomplete combinations are skipped: if a habitat combination lacks a
      valid simulated value for ANY survey, it gets no score.
    - Simulated positives are NOT rounded.
    - The log-likelihood is reported as the MEAN over surveys regardless of
      whether beta_binomial returns per-survey values, a sum, or a mean
      (detected once at start-up; see `_ll_mode`).

Accepted fits (not simply the maximum log-likelihood)
    A combination is accepted when at most MAX_MISSES surveys fall outside
    their 95% Wilson interval. Accepted fits are ranked by surveys inside the
    interval, then by mean absolute error. LL_TOLERANCE optionally restricts
    candidates to those near the top log-likelihood (None = all).

Note: these checks improve data alignment and likelihood consistency. They do
not by themselves show that the best-fit parameters are biologically valid.
"""

import os

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from idmtools_calibra.utilities.ll_calculators import beta_binomial

# Paths

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
DHS_FILE = "pcr_prevalence.csv"

# Must equal the job directory used by the analyzer. Taken from manifest when
# it is importable (the analyzer writes to manifest.job_directory), otherwise
# the fallback below.
try:
    import manifest
    JOB_DIR = manifest.job_directory
except (ImportError, AttributeError):
    JOB_DIR = os.path.join(PROJECT_DIR, "experiments")

EXPERIMENT_NAME = "Final_Pickup13yrs"
ANALYZER_DIR = os.path.join(JOB_DIR, f"{EXPERIMENT_NAME}_AnalyzerOutput")

SIM_PFPR_CSV = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")
OUTPUT_DIR = ANALYZER_DIR

# Settings

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025

BLOCK_DAYS = 30
N_BLOCKS = 12

# Cumulative start day of each month in the flat 365-day EMOD calendar
# (equals MONTH_OFFSET in the analyzer and the pickup script).
DAYS_PER_YEAR = 365

MONTH_START = {
    1: 0, 2: 31, 3: 59, 4: 90,
    5: 120, 6: 151, 7: 181, 8: 212,
    9: 243, 10: 273, 11: 304, 12: 334,
}
FEB_28_MODEL_DAY = 58  # Feb 29 is mapped to Feb 28 (keep consistent with climate inputs)

# Survey age range 6-59 months = the 0.5-5 y bin (upper edge 5 years).
SURVEY_AGEBINS = (5.0,)

HABITAT_VARS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
XCOL = "x_Temporary_Larval_Habitat"
SEED_VAR = "Run_Number"
SWEEP_VARIABLES = HABITAT_VARS + [SEED_VAR]

OBS_COLUMNS = ["year", "block", "DHS_n", "DHS_pos"]
TOP_N = 5

# Best-fit selection
LL_TOLERANCE = None  # None = all combinations are candidates; or e.g. 2.0
MAX_MISSES = 1       # surveys allowed outside their CI
CI_Z = 1.96          # 95% interval for the observed prevalence (selection only)


# Loading

def load_u5_pfpr(csv_path):
    """
    Simulated PfPR and Pop for the survey age range, averaged over seeds.
    Returns one row per habitat combination x year x block with:
        Pop                    mean population across replicates
        npos                   mean (PfPR x Pop) across replicates (not rounded)
        PfPR_arith_mean        arithmetic mean of replicate PfPR (diagnostic only)
        sim_PfPR_from_counts   npos / Pop
        PfPR                   = sim_PfPR_from_counts (used for scoring/plots)
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in SWEEP_VARIABLES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing tag column(s) {missing}. Re-run the analyzer "
            f"with these in SWEEP_VARIABLES. Columns present: {list(df.columns)}"
        )

    nan_tags = [c for c in HABITAT_VARS if df[c].isnull().any()]
    if nan_tags:
        raise ValueError(
            f"Calibration tag(s) {nan_tags} contain NaN (tag missing from some "
            "simulations). Those rows cannot be scored."
        )

    # The analyzer writes `month` (1-12, one 30-day block); older files used
    # `report_interval`. Accept either.
    block_col = "report_interval" if "report_interval" in df.columns else "month"

    required = ["Year", block_col, "agebin", "PfPR", "Pop"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_path} is missing columns {missing_cols}. "
            f"Columns present: {list(df.columns)}"
        )

    keep = (
        df[block_col].between(1, N_BLOCKS)
        & df["Year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
        & np.isin(np.round(df["agebin"].astype(float), 6),
                  np.round(np.asarray(SURVEY_AGEBINS, dtype=float), 6))
    )

    # Completeness flag written by the analyzer; drops the 5-day interval 13.
    # Parsed as text so "False" is False.
    if "is_complete_interval" not in df.columns and block_col == "report_interval":
        print("WARNING: report_interval present but is_complete_interval is "
              "missing; the partial 13th interval is excluded only by the "
              "1-12 block filter.")
    if "is_complete_interval" in df.columns:
        keep &= (
            df["is_complete_interval"]
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "t", "yes", "y"])
        )

    df = df[keep].copy()
    if df.empty:
        raise ValueError(
            f"No rows with agebin in {SURVEY_AGEBINS} for "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR} in {csv_path}. The pickup "
            "must be run with age_bins including 0.5 and 5."
        )

    df = df.rename(columns={"Year": "year", block_col: "block"})

    # Combine age bins if more than one is listed (population-weighted rates).
    count_cols = ["New_infections", "Pop"]
    rate_cols = ["PfPR", "Cases", "Severe_cases", "Anemia",
                 "Mean_density", "Gametocyte_prevalence"]
    value_cols = [c for c in count_cols + rate_cols if c in df.columns]
    rate_cols = [c for c in rate_cols if c in df.columns]
    id_cols = [c for c in df.columns if c not in value_cols + ["agebin"]]

    for col in rate_cols:
        df[col] = df[col] * df["Pop"]

    u5 = df.groupby(id_cols, dropna=False)[value_cols].sum().reset_index()
    pop = u5["Pop"].where(u5["Pop"] > 0)
    for col in rate_cols:
        u5[col] = u5[col] / pop

    # Keep only rows where both PfPR and Pop are valid, so mean(Pop) and
    # mean(npos) are taken over the same replicates.
    u5 = u5[u5["Pop"].gt(0) & u5["PfPR"].notna()].copy()

    # Infected population within each replicate, then average.
    u5["simulated_positive_population"] = u5["PfPR"] * u5["Pop"]
    agg = dict(
        Pop=("Pop", "mean"),
        npos=("simulated_positive_population", "mean"),
        PfPR_arith_mean=("PfPR", "mean"),
    )
    if "interval_midpoint_model_time" in u5.columns:
        agg["midpoint_model_time"] = ("interval_midpoint_model_time", "mean")
    avg = (
        u5.groupby(HABITAT_VARS + ["year", "block"], as_index=False)
        .agg(**agg)
    )
    avg["sim_PfPR_from_counts"] = avg["npos"] / avg["Pop"]
    # Downstream scoring/plotting uses the count-based prevalence consistently.
    avg["PfPR"] = avg["sim_PfPR_from_counts"]
    return avg


def _parse_dates(series):
    """Parse M/D/YYYY dates; fall back to pandas' parsing."""
    try:
        return pd.to_datetime(series, format="%m/%d/%Y")
    except (ValueError, TypeError):
        return pd.to_datetime(series)


def load_dhs_data(csv_path):
    """
    Observed surveys, each assigned to the 30-day reporting block (flat
    365-day EMOD calendar) containing the fieldwork midpoint. `date` (the
    midpoint) is kept for plotting.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")

    for col in ["fieldwork_start", "fieldwork_end"]:
        df[col] = _parse_dates(df[col])

    midpoint = (
        df["fieldwork_start"]
        + (df["fieldwork_end"] - df["fieldwork_start"]) / 2
    )

    df["date"] = midpoint
    df["year"] = midpoint.dt.year

    # Fixed-month-length model day, not Gregorian day-of-year.
    model_day = (
        midpoint.dt.month.map(MONTH_START)
        + midpoint.dt.day - 1
    )
    leap_day = midpoint.dt.month.eq(2) & midpoint.dt.day.eq(29)
    model_day.loc[leap_day] = FEB_28_MODEL_DAY
    df["model_day"] = model_day.astype(int)
    # Survey position on the flat model calendar (for plotting).
    df["plot_date"] = (
        pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01")
        + pd.to_timedelta(df["model_day"], unit="D")
    )
    df["block"] = (df["model_day"] // BLOCK_DAYS + 1).astype(int)

    # Surveys in days 360-364 fall in the incomplete 13th interval.
    late = df["block"] > N_BLOCKS
    if late.any():
        print(
            f"\nNOTE: Excluding {int(late.sum())} survey(s) falling "
            "in the incomplete 13th reporting interval."
        )
        df = df.loc[~late].copy()

    in_range = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    if not in_range.all():
        dropped = (
            df.loc[~in_range, "survey_year"].tolist()
            if "survey_year" in df.columns
            else df.loc[~in_range, "year"].tolist()
        )
        print(
            f"\nDropping {int((~in_range).sum())} survey(s) outside "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR} "
            f"(no simulated data): {sorted(dropped)}"
        )

    return df.loc[in_range].reset_index(drop=True)


# Scoring

_ll_mode = None  # "per_survey", "sum" or "mean"; set on first use


def _detect_ll_mode():
    """
    Find out what beta_binomial returns by calling it on one survey and on the
    same survey twice: array -> per-survey values; scalar that doubles -> sum;
    scalar that stays the same -> mean.
    """
    one = beta_binomial(np.array([50.0]), np.array([1000.0]),
                        np.array([10.0]), np.array([120.0]))
    two = beta_binomial(np.array([50.0, 50.0]), np.array([1000.0, 1000.0]),
                        np.array([10.0, 10.0]), np.array([120.0, 120.0]))

    if np.size(two) > 1:
        return "per_survey"

    one, two = float(np.sum(one)), float(two)

    if np.isclose(two, 2 * one, rtol=1e-6):
        return "sum"
    if np.isclose(two, one, rtol=1e-6):
        return "mean"

    raise RuntimeError(
        f"Cannot tell whether beta_binomial returns a sum or a mean "
        f"(one point: {one}, two identical points: {two})."
    )


def mean_log_likelihood(n_obs, n_sim, pos_obs, pos_sim):
    """Mean over surveys of the beta-binomial log-likelihood."""
    global _ll_mode

    if _ll_mode is None:
        _ll_mode = _detect_ll_mode()
        print(f"beta_binomial returns: {_ll_mode}; reporting the mean over surveys.")

    ll = beta_binomial(n_obs, n_sim, pos_obs, pos_sim)

    if _ll_mode == "per_survey":
        return float(np.mean(ll))
    if _ll_mode == "sum":
        return float(ll) / len(n_obs)
    return float(ll)


def wilson_ci(pos, n, z=CI_Z):
    """Wilson score interval for observed prevalence (arrays in, arrays out)."""
    pos = np.asarray(pos, dtype=float)
    n = np.asarray(n, dtype=float)
    p = pos / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def score(sim_df, data_df):
    """
    Per habitat combination: beta-binomial log-likelihood (mean over surveys),
    number of surveys whose CI contains the simulated PfPR (count-based), and
    mean absolute error at the survey points. A combination is skipped if any
    survey has no valid simulated value. Returns (score_df, n_scored, n_skipped).
    """
    obs = data_df[OBS_COLUMNS].reset_index(drop=True)
    n_obs = len(obs)
    obs_p = (obs["DHS_pos"] / obs["DHS_n"]).values
    lo, hi = wilson_ci(obs["DHS_pos"].values, obs["DHS_n"].values)

    rows = []
    n_skipped = 0
    for keys, g in sim_df.groupby(HABITAT_VARS):
        # Each survey must match exactly one simulated (year, block) row;
        # duplicates raise MergeError and the combination is skipped rather
        # than scored on different data.
        try:
            merged = obs.merge(
                g[["year", "block", "Pop", "npos", "sim_PfPR_from_counts"]],
                on=["year", "block"],
                how="left",
                validate="many_to_one",
            )
        except pd.errors.MergeError:
            n_skipped += 1
            continue

        if (
            len(merged) != n_obs
            or merged[["Pop", "npos", "sim_PfPR_from_counts"]].isna().any().any()
        ):
            n_skipped += 1
            continue

        ll = mean_log_likelihood(
            merged["DHS_n"].values,
            merged["Pop"].values,
            merged["DHS_pos"].values,
            merged["npos"].values,
        )
        sim_p = merged["sim_PfPR_from_counts"].values
        n_inside = int(((sim_p >= lo) & (sim_p <= hi)).sum())
        mae = float(np.abs(sim_p - obs_p).mean())
        rows.append((*keys, ll, n_inside, mae))

    n_scored = len(rows)
    print(f"  Scored {n_scored} habitat combination(s); skipped {n_skipped} "
          "with missing/duplicate survey values.")
    if not rows:
        raise ValueError("No habitat combination had simulated values for every survey.")

    score_df = pd.DataFrame(rows, columns=HABITAT_VARS + ["ll", "n_inside_ci", "mae"])
    score_df = score_df.sort_values("ll", ascending=False).reset_index(drop=True)
    return score_df, n_scored, n_skipped


def rank_fit(score_df, n_surveys):
    """
    Rank candidates by surveys inside their CI (more is better), then by mean
    absolute error. Returns (ranked, accepted).
    """
    cand = score_df
    if LL_TOLERANCE is not None:
        cand = score_df[score_df["ll"] >= score_df["ll"].max() - LL_TOLERANCE]
    ranked = cand.sort_values(
        ["n_inside_ci", "mae"], ascending=[False, True]
    ).reset_index(drop=True)
    accepted = ranked[ranked["n_inside_ci"] >= n_surveys - MAX_MISSES].reset_index(drop=True)
    return ranked, accepted


def report_best(sim_df, data_df, best):
    """Print observed vs simulated PfPR at each survey for the selected best fit."""
    g = sim_df
    for var in HABITAT_VARS:
        g = g[np.isclose(g[var], best[var])]
    lo, hi = wilson_ci(data_df["DHS_pos"].values, data_df["DHS_n"].values)
    m = data_df[["year", "block", "DHS_n", "DHS_pos"]].merge(
        g[["year", "block", "sim_PfPR_from_counts"]], on=["year", "block"], how="left"
    )
    out = pd.DataFrame({
        "year": m["year"],
        "block": m["block"],
        "observed": m["DHS_pos"] / m["DHS_n"],
        "ci_low": lo,
        "ci_high": hi,
        "simulated": m["sim_PfPR_from_counts"],
    })
    out["inside_ci"] = (out["simulated"] >= out["ci_low"]) & (out["simulated"] <= out["ci_high"])
    print("\nSelected best fit, survey by survey:")
    print(out.round(3).to_string(index=False))


def print_diagnostics(sim_df, data_df):
    """Confirm observed and simulated data line up."""
    print("\nDiagnostics")
    print("Observed surveys:", len(data_df))
    print("Observed blocks:", sorted(data_df["block"].unique()))
    print("Simulated blocks:", sorted(sim_df["block"].unique()))
    print("Observed years:", sorted(data_df["year"].unique()))
    print("Simulated years:", sorted(sim_df["year"].unique()))

    assert set(sim_df["block"].unique()) <= set(range(1, N_BLOCKS + 1)), \
        "Simulated data contain blocks outside 1-12 (partial interval leaked in)."
    unmatched = data_df.merge(
        sim_df[["year", "block"]].drop_duplicates(),
        on=["year", "block"], how="left", indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    if len(unmatched):
        print("WARNING: surveys with no simulated (year, block) in ANY combination:")
        print(unmatched[["year", "block"]].to_string(index=False))
    else:
        print("All surveys have a simulated (year, block) in at least one combination.")

    # Compare the two replicate-averaging estimates for transparency.
    diff = (sim_df["sim_PfPR_from_counts"] - sim_df["PfPR_arith_mean"]).abs()
    print(f"Max |count-based - arithmetic-mean PfPR|: {diff.max():.4f}")


# Plotting (publication style)

N_NEXT = 4
STAR_SIZE = 400
STAR_EDGE = "white"
SHOW_TITLES = False
SHOW_OBS_CI = False

C_ALL = "grey"
C_NEXT = "coral"
C_BEST = "firebrick"
C_OBS = "black"

PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _add_block_dates(sim):
    """Mid-point date of each 30-day block (vectorised)."""
    sim = sim.copy()
    year_start = pd.to_datetime(sim["year"].astype(int).astype(str) + "-01-01")

    if "midpoint_model_time" in sim.columns and sim["midpoint_model_time"].notna().all():
        # Analyzer's own interval midpoint: model_time = year + day_in_year / 365.
        days = (sim["midpoint_model_time"] - sim["year"].astype(float)) * DAYS_PER_YEAR
    else:
        days = BLOCK_DAYS * (sim["block"].astype(float) - 0.5)

    sim["date"] = year_start + pd.to_timedelta(days, unit="D")
    return sim


def _combo_series(sim_df, row):
    """Simulated PfPR time series (sorted by date) for one habitat combination."""
    g = sim_df
    for var in HABITAT_VARS:
        g = g[np.isclose(g[var], row[var])]
    return g.sort_values("date")


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def _panel_letter(ax, letter):
    ax.text(-0.13, 1.03, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")


def _save(fig, stem):
    path = f"{stem}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)


def _draw_pfpr(ax, sim_df, data_df, ranked, legend_fs=None):
    best = ranked.iloc[0]
    sim = _add_block_dates(sim_df)
    n_combo = sim[HABITAT_VARS].drop_duplicates().shape[0]

    obs = data_df.copy()
    obs["PfPR"] = obs["DHS_pos"] / obs["DHS_n"]
    if "plot_date" in obs.columns:
        obs["date"] = obs["plot_date"]

    ymax = max(sim["PfPR"].max(), obs["PfPR"].max())

    for _, g in sim.groupby(HABITAT_VARS):
        g = g.sort_values("date")
        ax.plot(g["date"], g["PfPR"], color=C_ALL, alpha=0.10, linewidth=0.6,
                rasterized=True, zorder=1)

    for _, row in ranked.iloc[1:1 + N_NEXT].iterrows():
        g = _combo_series(sim, row)
        ax.plot(g["date"], g["PfPR"], color=C_NEXT, alpha=0.7, linewidth=1.0, zorder=2)

    gb = _combo_series(sim, best)
    ax.plot(gb["date"], gb["PfPR"], color=C_BEST, linewidth=1.8, zorder=3)

    if SHOW_OBS_CI:
        lo, hi = wilson_ci(obs["DHS_pos"].values, obs["DHS_n"].values)
        ax.errorbar(obs["date"], obs["PfPR"],
                    yerr=[obs["PfPR"] - lo, hi - obs["PfPR"]],
                    fmt="none", ecolor=C_OBS, elinewidth=1.0, capsize=2.5, zorder=4)
    ax.scatter(obs["date"], obs["PfPR"], s=STAR_SIZE, marker="o", color=C_OBS,
               edgecolor=STAR_EDGE, linewidths=0.8, zorder=5)

    handles = [
        Line2D([], [], color=C_ALL, alpha=0.6, lw=1.2,
               label=f"All parameter combinations (n = {n_combo})"),
        Line2D([], [], color=C_NEXT, lw=1.4, label=f"Next {N_NEXT} best fits"),
        Line2D([], [], color=C_BEST, lw=2.0,
               label=(f"Best fit (xTLH = {best[XCOL]:.1f}, const = {best['scale_constant']:g}, "
                      f"rain = {best['scale_temp_rain']:g}, veg = {best['scale_water_veg']:g})")),
        Line2D([], [], color=C_OBS, marker="o", linestyle="none", markersize=11,
               markeredgecolor=STAR_EDGE, markeredgewidth=0.6,
               label="Observed (DHS)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=legend_fs, handlelength=1.8)

    ax.set_ylim(0, ymax * 1.45)
    ax.set_yticks([t for t in np.arange(0, 1.01, 0.2) if t <= ymax * 1.05])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("PfPR (children 6\u201359 months)")
    if SHOW_TITLES:
        ax.set_title("Observed vs simulated PfPR")
    _clean_axes(ax)


def _draw_loglik(ax, score_df, ranked, legend_fs=None):
    best = ranked.iloc[0]
    top_per_x = score_df.groupby(XCOL)["ll"].max().sort_index()

    ax.scatter(score_df[XCOL], score_df["ll"], s=14, color=C_NEXT, alpha=0.45,
               linewidths=0, rasterized=True, zorder=1)
    ax.plot(top_per_x.index, top_per_x.values, color=C_BEST, marker="o",
            linewidth=1.4, markersize=3.5, zorder=2)
    ax.scatter([best[XCOL]], [best["ll"]], s=STAR_SIZE, marker="o", color=C_BEST,
               edgecolor=STAR_EDGE, linewidths=0.8, zorder=5)

    handles = [
        Line2D([], [], color=C_NEXT, marker="o", linestyle="none", markersize=4,
               alpha=0.8, label="Habitat combinations"),
        Line2D([], [], color=C_BEST, marker="o", markersize=3.5, lw=1.4,
               label="Best combination at each xTLH"),
        Line2D([], [], color=C_BEST, marker="o", linestyle="none", markersize=11,
               markeredgecolor=STAR_EDGE, markeredgewidth=0.6,
               label=f"Selected best fit (ll = {best['ll']:.2f})"),
    ]
    ax.margins(y=0.08)
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=legend_fs)

    ax.set_xlabel("Larval habitat multiplier (xTLH)")
    ax.set_ylabel("Mean log-likelihood per survey")
    if SHOW_TITLES:
        ax.set_title("Larger value = better fit")
    _clean_axes(ax)


def plot_pfpr(sim_df, data_df, ranked, stem):
    with mpl.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        _draw_pfpr(ax, sim_df, data_df, ranked)
        fig.tight_layout()
        _save(fig, stem)


def plot_loglik(score_df, ranked, stem):
    with mpl.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        _draw_loglik(ax, score_df, ranked)
        fig.tight_layout()
        _save(fig, stem)


def plot_combined(sim_df, data_df, score_df, ranked, stem):
    with mpl.rc_context(PUB_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        _draw_pfpr(ax1, sim_df, data_df, ranked, legend_fs=6)
        _draw_loglik(ax2, score_df, ranked, legend_fs=6)
        _panel_letter(ax1, "A")
        _panel_letter(ax2, "B")
        fig.tight_layout(w_pad=2.0)
        _save(fig, stem)


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_df = load_u5_pfpr(SIM_PFPR_CSV)
    data_df = load_dhs_data(os.path.join(INPUT_DIR, DHS_FILE))

    n_combos = sim_df[HABITAT_VARS].drop_duplicates().shape[0]
    print(f"Habitat combinations: {n_combos}  surveys used: {len(data_df)}")

    print("\nSurvey -> simulation time mapping (year, 30-day block, flat 365-day calendar):")
    cols = (["survey_year", "date", "model_day", "year", "block"] if "survey_year" in data_df
            else ["date", "model_day", "year", "block"])
    print(data_df[cols].to_string(index=False))

    print_diagnostics(sim_df, data_df)

    n_surveys = len(data_df)
    score_df, n_scored, n_skipped = score(sim_df, data_df)
    ranked, accepted = rank_fit(score_df, n_surveys)
    print(f"Parameter combinations scored: {n_scored}; skipped: {n_skipped}")

    score_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_scores.csv"), index=False)
    ranked.to_csv(os.path.join(OUTPUT_DIR, "calibration_fit_ranking.csv"), index=False)
    accepted.to_csv(os.path.join(OUTPUT_DIR, "calibration_accepted_fits.csv"), index=False)

    print(f"\nTop {TOP_N} by log-likelihood:")
    print(score_df.head(TOP_N).to_string(index=False))

    print("\nCombinations by number of surveys inside their CI:")
    counts = ranked["n_inside_ci"].value_counts().sort_index(ascending=False)
    for k, v in counts.items():
        print(f"  {int(k)}/{n_surveys}: {v}")
    print(f"Accepted (at most {MAX_MISSES} survey(s) outside CI): {len(accepted)}")

    if len(accepted):
        print(f"\nTop 10 accepted fits (all {len(accepted)} saved to calibration_accepted_fits.csv):")
        print(accepted.head(10).to_string(index=False))
    else:
        print("\nNOTE: no combination meets the acceptance rule. Showing the closest "
              "combinations instead; consider widening the parameter grid (xTLH, "
              "scale_constant) or checking the detection method.")
        print(ranked.head(TOP_N).to_string(index=False))

    best = ranked.iloc[0]
    report_best(sim_df, data_df, best)

    plot_pfpr(sim_df, data_df, ranked, os.path.join(OUTPUT_DIR, "Fig_observed_vs_simulated"))
    plot_loglik(score_df, ranked, os.path.join(OUTPUT_DIR, "Fig_loglik_vs_xTLH"))
    plot_combined(sim_df, data_df, score_df, ranked,
                  os.path.join(OUTPUT_DIR, "Fig_calibration_combined"))


if __name__ == "__main__":
    main()