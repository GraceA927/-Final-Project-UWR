"""
UWR prevalence calibration: beta-binomial log-likelihood over the inherited
habitat combinations (idmtools_calibra score()/plot_output() logic).

Simulated data
    PfPR_ClinicalIncidence_monthly_pickup.csv from MonthlyPfPRAnalyzer
    (uwr_pickup_analyzer.py): one row per simulation x month x agebin.
    Under Option B each pickup simulation inherits its burn-in row's habitat
    combination, so every simulation carries these tags:
        x_Temporary_Larval_Habitat, scale_constant, scale_temp_rain,
        scale_water_veg, Run_Number (seed), burnin_sim_index
    A habitat combination is scored as the mean log-likelihood over its seeds.

Observed data
    inputs/pcr_prevalence.csv. It has no month column, so a representative
    (year, month) is taken from each survey's fieldwork midpoint, and that is
    what the merge joins on.

Assumptions to verify
    - Survey ages (6-59 months) are treated as equivalent to the simulation's
      U5 bins (0-0.25y and 0.25-5y). The simulated U5 aggregate includes 0-6
      month-olds that the survey excludes; adjust U5_AGEBINS if this matters.
    - The fieldwork midpoint month stands in for the month the survey reflects.
    - Surveys outside 2013-2025 (e.g. 2011) are dropped with a printed warning.
"""

import os

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from idmtools_calibra.utilities.ll_calculators import beta_binomial

# -----------------------------------------------------------------------------
# Paths (match run_pickup_UWR.py / uwr_pickup_analyzer.py)
# -----------------------------------------------------------------------------

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
DHS_FILE = "pcr_prevalence.csv"

# Analyzer output folder: <job_directory>/<experiment>_output/
ANALYZER_DIR = os.path.join(PROJECT_DIR, "experiments", "UWR_Pickup_ECM_output")
SIM_PFPR_CSV = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")

# Calibration results are written next to the analyzer outputs.
OUTPUT_DIR = ANALYZER_DIR

# Settings

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025

# Under-5 bins from add_reports(): age_bins=[0.25, 5, 15, 25, 100].
U5_AGEBINS = (0.25, 5.0)

HABITAT_VARS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
SEED_VAR = "Run_Number"
SWEEP_VARIABLES = HABITAT_VARS + [SEED_VAR]

OBS_COLUMNS = ["year", "month", "DHS_n", "DHS_pos"]
TOP_N = 5  # best habitat combinations highlighted in the plot


# Loading

def load_u5_pfpr(csv_path):
    """
    Collapse the two under-5 agebins into one U5 row per simulation x month.
    Rates are population-weighted; counts are summed. Adds `npos`, the
    simulated number of positives (PfPR x Pop, rounded).
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in SWEEP_VARIABLES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing tag column(s) {missing}. Re-run the analyzer "
            f"with these in SWEEP_VARIABLES. Columns present: {list(df.columns)}"
        )

    df = df[df["agebin"].isin(U5_AGEBINS)].copy()
    if df.empty:
        raise ValueError(f"No rows with agebin in {U5_AGEBINS} in {csv_path}.")

    count_cols = ["Cases", "Severe_cases", "New_infections", "Pop"]
    rate_cols = ["PfPR", "Mean_density", "Gametocyte_prevalence", "Anemia"]
    value_cols = count_cols + rate_cols
    id_cols = [c for c in df.columns if c not in value_cols + ["agebin"]]

    for col in rate_cols:
        df[col] = df[col] * df["Pop"]  # population-weighted numerator

    u5 = df.groupby(id_cols, dropna=False)[value_cols].sum().reset_index()

    pop = u5["Pop"].where(u5["Pop"] > 0)
    for col in rate_cols:
        u5[col] = u5[col] / pop

    u5 = u5.rename(columns={"Year": "year"})
    u5["npos"] = (u5["PfPR"] * u5["Pop"]).round(0)
    return u5


def load_dhs_data(csv_path):
    """Observed surveys with a representative (year, month) from the fieldwork midpoint."""
    df = pd.read_csv(csv_path, parse_dates=["fieldwork_start", "fieldwork_end"])
    midpoint = df["fieldwork_start"] + (df["fieldwork_end"] - df["fieldwork_start"]) / 2
    df["year"] = midpoint.dt.year
    df["month"] = midpoint.dt.month

    in_range = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    if not in_range.all():
        print(
            f"\nDropping {int((~in_range).sum())} survey(s) outside "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR} (no simulated data): "
            f"{sorted(df.loc[~in_range, 'survey_year'].tolist())}"
        )
    return df[in_range].reset_index(drop=True)


# Scoring

def score(sim_df, data_df):
    """
    Log-likelihood per simulation (habitat combination x seed), then the mean
    over seeds for each habitat combination. Returns (per_sim_df, score_df),
    with score_df sorted best-first.
    """
    obs = data_df[OBS_COLUMNS]

    # Surveys with no simulated (year, month) at all.
    sim_months = sim_df[["year", "month"]].drop_duplicates()
    unmatched = obs.merge(sim_months, on=["year", "month"], how="left", indicator=True)
    n_unmatched = int((unmatched["_merge"] == "left_only").sum())
    if n_unmatched:
        print(f"  WARNING: {n_unmatched}/{len(obs)} survey row(s) have no simulated "
              "(year, month) and are excluded from every score.")

    comb = obs.merge(sim_df, on=["year", "month"], how="inner")
    comb = comb.dropna(subset=["npos", "Pop"])

    rows = []
    for keys, g in comb.groupby(SWEEP_VARIABLES):
        # beta_binomial indexes its inputs by position (0..n-1), so each
        # group needs a fresh index.
        g = g.reset_index(drop=True)
        ll = beta_binomial(g["DHS_n"], g["Pop"], g["DHS_pos"], g["npos"])
        rows.append((*keys, ll))
    per_sim = pd.DataFrame(rows, columns=SWEEP_VARIABLES + ["ll"])

    score_df = (
        per_sim.groupby(HABITAT_VARS)["ll"]
        .agg(ll="mean", n_seeds="count")
        .reset_index()
        .sort_values("ll", ascending=False)
        .reset_index(drop=True)
    )
    return per_sim, score_df


# Plotting

def _combo_label(row):
    return (f"xTLH={row['x_Temporary_Larval_Habitat']:.3g}, "
            f"const={row['scale_constant']:g}, rain={row['scale_temp_rain']:g}, "
            f"veg={row['scale_water_veg']:g}")


def plot_output(sim_df, data_df, score_df, path):
    # Mean PfPR over seeds, per habitat combination and month.
    agg = (
        sim_df.groupby(HABITAT_VARS + ["year", "month"])["PfPR"].mean().reset_index()
    )
    agg["date"] = pd.to_datetime(
        [f"{y}-{m}-01" for y, m in zip(agg["year"], agg["month"])]
    )
    obs = data_df.copy()
    obs["date"] = pd.to_datetime([f"{y}-{m}-01" for y, m in zip(obs["year"], obs["month"])])
    obs["PfPR"] = obs["DHS_pos"] / obs["DHS_n"]

    top = score_df.head(TOP_N)
    best = top.iloc[0]

    sns.set_style("whitegrid", {"axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all candidates faint, top N highlighted, best in dark red.
    for _, g in agg.groupby(HABITAT_VARS):
        axes[0].plot(g["date"], g["PfPR"], color="#999999", alpha=0.05, linewidth=0.8)

    for rank, (_, row) in enumerate(top.iterrows()):
        g = agg
        for var in HABITAT_VARS:
            g = g[np.isclose(g[var], row[var])]
        is_best = rank == 0
        axes[0].plot(
            g["date"], g["PfPR"],
            color="#CC0000" if is_best else "#FF8888",
            alpha=1.0 if is_best else 0.6,
            linewidth=1.8 if is_best else 1.0,
            label=f"best: {_combo_label(row)}" if is_best else None,
        )

    sizes = 20 + 180 * obs["DHS_n"] / obs["DHS_n"].max()
    axes[0].scatter(obs["date"], obs["PfPR"], s=sizes, color="k", label="Observed", zorder=3)
    axes[0].set_ylabel("PfPR (U5)")
    axes[0].set_title(f"Observed vs simulated (dark red = best; light red = next {TOP_N - 1})")
    axes[0].legend(loc="best", fontsize=8)

    # Right: log-likelihood against xTLH. Each dot is a habitat combination;
    # the line follows the best combination at each xTLH.
    xcol = "x_Temporary_Larval_Habitat"
    axes[1].scatter(score_df[xcol], score_df["ll"], s=14, color="#FF8888", alpha=0.5,
                    label="habitat combinations")
    best_per_x = score_df.groupby(xcol)["ll"].max().reset_index()
    axes[1].plot(best_per_x[xcol], best_per_x["ll"], "-o", color="#CC0000", markersize=4,
                 label="best combination at each xTLH")
    axes[1].scatter([best[xcol]], [best["ll"]], s=100, color="red", zorder=3,
                    label=f"best fit: ll={best['ll']:.3g}")
    axes[1].set_xlabel(xcol)
    axes[1].set_ylabel("mean log-likelihood (over seeds)")
    axes[1].set_title("Larger value = better fit")
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_df = load_u5_pfpr(SIM_PFPR_CSV)
    data_df = load_dhs_data(os.path.join(INPUT_DIR, DHS_FILE))

    n_combos = sim_df[HABITAT_VARS].drop_duplicates().shape[0]
    print(f"Simulations: {sim_df['sim_id'].nunique() if 'sim_id' in sim_df else 'n/a'}  "
          f"habitat combinations: {n_combos}  surveys used: {len(data_df)}")

    per_sim, score_df = score(sim_df, data_df)

    per_sim.to_csv(os.path.join(OUTPUT_DIR, "calibration_ll_per_simulation.csv"), index=False)
    score_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_scores.csv"), index=False)

    print(f"\nTop {TOP_N} habitat combinations:")
    print(score_df.head(TOP_N).to_string(index=False))

    plot_output(sim_df, data_df, score_df, os.path.join(OUTPUT_DIR, "selection.png"))


if __name__ == "__main__":
    main()