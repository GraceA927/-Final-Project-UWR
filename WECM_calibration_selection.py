################################################################################
# UWR Prevalence Calibration -- beta-binomial log-likelihood over xTLH
#
# Adapted from the idmtools_calibra template (score() / plot_output()
# logic and structure kept as-is) for:
#   - Simulated data: PfPR_ClinicalIncidence_monthly_pickup.csv, written by
#     MonthlyPfPRAnalyzer in UWR_pickup_analyzer.py. That file reports one
#     row per (simulation, month, agebin) rather than a pre-summed
#     'PfPR U5' column, and carries the xTLH each simulation actually ran
#     as an 'xTLH' column (read from each sim's own config.json -- see
#     that analyzer's docstring for why it isn't a simulation tag).
#     load_u5_pfpr() below collapses the two under-5 agebins (0.25, 5.0)
#     into single U5 rows, Pop-weighted, and renames 'xTLH' back to
#     'x_Temporary_Larval_Habitat' so the score()/plot_output() logic
#     below is otherwise unchanged from the original template.
#   - Observed data: /shared/home/grace/Final-Project-UWR/inputs/
#     pcr_prevalence.csv (real UWR prevalence surveys), replacing the
#     placeholder fake_DHS_calib.csv. This file has no 'month' column --
#     only survey_year plus a fieldwork_start/fieldwork_end window -- so
#     load_dhs_data() below derives a representative (year, month) from
#     each survey's fieldwork midpoint, which is what the merge in
#     score() actually joins on.
#
# ASSUMPTIONS TO VERIFY
#   - age_min_months/age_max_months in pcr_prevalence.csv (6-59) is
#     treated as equivalent to the simulation's U5 agebins (0.25y-5y,
#     i.e. 3 months-5 years) -- close but not identical at the low end;
#     the survey excludes 0-6 month-olds that the simulation's U5
#     aggregate includes. Adjust U5_AGEBINS below if this matters for
#     your calibration.
#   - A survey's fieldwork midpoint month is a reasonable stand-in for
#     "the month this survey's prevalence reflects." If you'd rather
#     average simulated PfPR across the whole fieldwork window instead of
#     picking one representative month, that's a larger rewrite of
#     score()'s merge step -- flag it if you want that version instead.
#   - Surveys outside the pickup's simulated years (2013-2025) -- the
#     2011 survey here -- are dropped with a printed warning rather than
#     silently producing NaNs that would break beta_binomial.
################################################################################

import os

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from idmtools_calibra.utilities.ll_calculators import beta_binomial

# --------------------------------------------------------------------
# PROJECT PATHS -- match run_pickup_UWR.py / UWR_pickup_analyzer.py
# --------------------------------------------------------------------
PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
DHS_FILE = "pcr_prevalence.csv"

# Point this at the (possibly copied) analyzer output for the experiment
# you want to calibrate against -- see UWR_pickup_analyzer.py's
# "WHERE THIS RUNS VS. CALIBRATION" note for how that CSV gets here.
SIM_PFPR_CSV = os.path.join(
    PROJECT_DIR, "experiments", "outputs", "UWR_Pickup_2013_2025_WECM",
    "PfPR_ClinicalIncidence_monthly_pickup.csv",
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "experiments", "outputs", "UWR_Pickup_2013_2025_WECM")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Must match run_pickup_UWR.py's calendar -- surveys outside this range
# have no simulated data to compare against.
PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025

# The two under-5 age bins in run_pickup_UWR.py's add_reports():
# age_bins=[0.25, 5, 15, 25, 100] -> bins are 0-0.25y and 0.25-5y.
U5_AGEBINS = (0.25, 5.0)

sweep_variables = ['x_Temporary_Larval_Habitat', 'Run_Number']


def load_u5_pfpr(csv_path):
    """
    Read a MonthlyPfPRAnalyzer CSV (one row per sim/month/agebin) and
    collapse the two under-5 bins into single U5 rows per simulation/
    month, Pop-weighted for rate columns and summed for count columns.
    Renames 'Year' -> 'year' and 'xTLH' -> 'x_Temporary_Larval_Habitat'
    so the rest of this script matches the original template's column
    names.
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
            "Cases": g["Cases"].sum(),
            "Severe_cases": g["Severe_cases"].sum(),
            "New_infections": g["New_infections"].sum(),
            "Pop": pop,
        }
        for rate_col in ["PfPR", "Mean_density", "Gametocyte_prevalence", "Anemia"]:
            out[rate_col] = float(np.average(g[rate_col], weights=g["Pop"])) if pop > 0 else np.nan
        return pd.Series(out)

    collapsed = (
        df.groupby(id_cols, dropna=False, group_keys=False)
        .apply(_collapse)
        .reset_index()
    )

    collapsed = collapsed.rename(columns={"Year": "year", "xTLH": "x_Temporary_Larval_Habitat"})

    missing_sweep = [v for v in sweep_variables if v not in collapsed.columns]
    if missing_sweep:
        raise ValueError(
            f"{csv_path} is missing sweep_variables columns {missing_sweep} after "
            "renaming -- check that UWR_pickup_analyzer.py actually recovered xTLH "
            "from config.json for this experiment (it prints a per-experiment xTLH "
            "check at the end of its run)."
        )

    return collapsed


def load_dhs_data(csv_path):
    """
    Read pcr_prevalence.csv and derive a representative (year, month) per
    survey from its fieldwork window's midpoint -- there's no 'month'
    column to merge on directly, unlike the original fake_DHS_calib.csv
    template this script is adapted from. Drops surveys outside the
    pickup's simulated years (with a printed warning) so score() never
    merges against a year with no simulated data.
    """
    df = pd.read_csv(csv_path, parse_dates=["fieldwork_start", "fieldwork_end"])
    midpoint = df["fieldwork_start"] + (df["fieldwork_end"] - df["fieldwork_start"]) / 2
    df["year"] = midpoint.dt.year
    df["month"] = midpoint.dt.month

    out_of_range = df[~df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)]
    if not out_of_range.empty:
        print(
            f"\nDropping {len(out_of_range)} survey(s) outside the pickup's "
            f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR} range (no simulated data to "
            f"compare against): {sorted(out_of_range['survey_year'].tolist())}"
        )
    df = df[df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)].reset_index(drop=True)

    return df


sim_pfpr_df = load_u5_pfpr(SIM_PFPR_CSV)
sim_pfpr_df['npos'] = sim_pfpr_df['PfPR'] * sim_pfpr_df['Pop']
sim_pfpr_df['npos'] = sim_pfpr_df.npos.round(0)

dhs_pfpr_df = load_dhs_data(os.path.join(INPUT_DIR, DHS_FILE))


def score(sim_df, data_df, sweep_variables):
    uniq_df = sim_df.groupby(sweep_variables).size().reset_index(name='Freq')
    ll = []
    for r, row in uniq_df.iterrows():
        mask = pd.Series(True, index=sim_df.index)
        for sv in sweep_variables:
            mask &= sim_df[sv] == row[sv]
        sim_subset_df = sim_df[mask]

        comb_df = data_df.merge(sim_subset_df, on=['year', 'month'], how='left')
        n_unmatched = comb_df['npos'].isna().sum()
        if n_unmatched:
            print(
                f"  WARNING: {n_unmatched}/{len(comb_df)} survey row(s) had no matching "
                f"simulated (year, month) for {dict(zip(sweep_variables, row[sweep_variables]))} "
                "-- dropping those rows from this combination's score."
            )
        comb_df = comb_df.dropna(subset=['npos', 'Pop'])

        ll = ll + [beta_binomial(comb_df.DHS_n, comb_df.Pop, comb_df.DHS_pos, comb_df.npos)]

    uniq_df['ll'] = ll

    score_df = uniq_df.groupby(['x_Temporary_Larval_Habitat'])['ll'].mean().reset_index(name='ll')
    return score_df


def plot_output(sim_df, data_df, score_df, variable):
    sim_df['date'] = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(sim_df.year, sim_df.month)])
    data_df['date'] = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(data_df.year, data_df.month)])
    data_df['PfPR'] = data_df['DHS_pos'] / data_df['DHS_n']
    sns.set_style('whitegrid', {'axes.linewidth': 0.5})
    fig = plt.figure(figsize=(12, 5))
    axes = [fig.add_subplot(1, 2, x + 1) for x in range(2)]

    candidate_val = score_df[variable]
    score_df1 = score_df[score_df.ll == max(score_df.ll)]
    print(score_df1)
    best_val = score_df1[variable].values[0]
    best_ll = score_df1['ll'].values[0]

    for cand in candidate_val:
        plot_df = sim_df[sim_df[variable] == cand]
        is_best = (cand == score_df1[variable].max())
        a = 1 if is_best else 0.15
        label = f'best fit ({variable}={best_val:.3g})' if is_best else None
        axes[0].plot(plot_df['date'], plot_df['PfPR'], color='#FF0000', alpha=a, label=label)

    axes[0].scatter(data_df['date'].values, data_df['PfPR'], data_df['DHS_n'], 'k', label='Observed')
    axes[0].set_ylabel('PfPR')
    axes[0].set_title('Observed vs Simulated (Dark red is the best fit)')
    axes[0].legend(loc='best', fontsize=8)

    axes[1].plot(score_df[variable], score_df['ll'], '-o', color='#FF0000', markersize=5, label='log-likelihood')
    axes[1].scatter(score_df1[variable], score_df1.ll, s=90, color='red',
                     label=f'best fit: {variable}={best_val:.3g}, ll={best_ll:.3g}')
    axes[1].set_ylabel('log-likelihood')
    axes[1].set_xlabel('x_Temporary_Larval_Habitat')
    axes[1].set_title('Mean log-likelihood. Larger value = better fit')
    axes[1].legend(loc='best', fontsize=8)

    fig.savefig(os.path.join(OUTPUT_DIR, 'selection.png'))


if __name__ == "__main__":
    scores = score(sim_pfpr_df, dhs_pfpr_df, sweep_variables)
    print(scores)

    sim_pfpr_agg = sim_pfpr_df.groupby(['year', 'month', 'x_Temporary_Larval_Habitat'])['PfPR'].mean().reset_index(name='PfPR')
    plot_output(sim_pfpr_agg, dhs_pfpr_df, scores, 'x_Temporary_Larval_Habitat')