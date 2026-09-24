import os
import getpass

import pandas as pd
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from idmtools_calibra.utilities.ll_calculators import beta_binomial
import manifest

# Reference burnin_UpperWest.py directly rather than re-typing its parameter
# space here. Importing it runs its module-level setup code (parameter-space
# construction, the "STRICT 500-SIMULATION VALIDATION" print/assert block)
# but NOT general_sim()/experiment submission, since that's guarded behind
# `if __name__ == "__main__":` in that file. This keeps U5_AGEBIN, site_name,
# and the expected-habitat-combination count locked to whatever the burnin
# script actually used, instead of a second hardcoded copy that could
# silently drift out of sync if burnin_UpperWest.py changes later.
import burnin_UpperWest as burnin

user = getpass.getuser()

# -----------------------------------------------------------------------------------
# TODO: update to your actual burnin experiment output directory name (the same
# expt_name used in analyzer_Project_Burnin.py). This is the analyzer's own
# output-folder label, not burnin.py's Experiment.name, so it isn't derived
# from the import above.
# -----------------------------------------------------------------------------------
expt_name = 'UpperWest_Region_burnin50'

jdir = manifest.job_directory
output_dir = os.path.join(jdir, 'UWR_burnin50_outputs')

input_dir = os.path.join(manifest.input_dir)

# -----------------------------------------------------------------------------------
# All 5 swept parameters from burnin_UpperWest.py. Run_Number is a stochastic seed,
# not a calibration target -- score() averages log-likelihood across it, leaving one
# score per unique (x_Temporary_Larval_Habitat, scale_constant, scale_temp_rain,
# scale_water_veg) combination.
# -----------------------------------------------------------------------------------
sweep_variables = [
    'x_Temporary_Larval_Habitat',
    'scale_constant',
    'scale_temp_rain',
    'scale_water_veg',
    'Run_Number'
]
seed_variable = 'Run_Number'
habitat_variables = [v for v in sweep_variables if v != seed_variable]

# Expected number of unique (non-seed) habitat combinations, straight from
# burnin_UpperWest.py's own parameter space -- 5 xTLH x 20 habitat combos = 100.
expected_habitat_combos = len(burnin.xTLH_range) * burnin.n_habs

# -----------------------------------------------------------------------------------
# Load simulated PfPR/incidence data -- produced by MonthlyPfPRAnalyzer in
# analyzer_Project_Burnin.py, NOT the toy example's U5-only file.
# -----------------------------------------------------------------------------------
pfpr_file = os.path.join(output_dir, expt_name, 'PfPR_ClinicalIncidence_monthly_burnin.csv')
sim_pfpr_df = pd.read_csv(pfpr_file)

# The analyzer's file covers ALL age bins in one file (an 'agebin' column), not a
# pre-filtered U5-only file. Filter down to the under-5 bin, taken directly from
# burnin_UpperWest.py's report_age_bins = [0.25, 5, 15, 25, 100] -- index 1 (value
# 5) is the 0.25-5 year bin.
U5_AGEBIN = burnin.report_age_bins[1]

available_agebins = sorted(sim_pfpr_df['agebin'].unique())
print(f"Available age bins in {pfpr_file}: {available_agebins}")
print(f"Using U5_AGEBIN = {U5_AGEBIN} (burnin_UpperWest.py report_age_bins[1])")

sim_pfpr_df = sim_pfpr_df[sim_pfpr_df['agebin'] == U5_AGEBIN].copy()

if sim_pfpr_df.empty:
    raise ValueError(
        f"No rows matched agebin == {U5_AGEBIN}. "
        f"burnin_UpperWest.py's report_age_bins is {burnin.report_age_bins} -- "
        f"if that's changed since this experiment ran, adjust U5_AGEBIN's index above, "
        f"or pick from the available bins printed above."
    )

# 'month' from the analyzer is a running index over the full report period (e.g.
# 1..600 across 50 years), NOT calendar month. Convert to calendar month (1-12) and
# use 'Year' (capitalized by the analyzer) for the merge below.
sim_pfpr_df['year'] = sim_pfpr_df['Year'].astype(int)
sim_pfpr_df['month'] = ((sim_pfpr_df['month'] - 1) % 12) + 1

sim_pfpr_df['npos'] = (sim_pfpr_df['PfPR'] * sim_pfpr_df['Pop']).round(0)

# -----------------------------------------------------------------------------------
# TODO: point this at your actual DHS calibration data, not the toy fake_DHS_calib.csv.
# Expected columns: year, month, DHS_n, DHS_pos.
# -----------------------------------------------------------------------------------
dhs_pfpr_df = pd.read_csv(os.path.join(input_dir, 'fake_DHS_calib.csv'))


def score(sim_df, data_df, sweep_variables, seed_variable='Run_Number'):
    """
    Score every unique combination of sweep_variables (habitat params x seeds)
    against observed DHS PfPR via beta-binomial log-likelihood, then average
    across seed_variable so each row of the returned score_df is one
    calibration-parameter combination (not one seed).
    """

    habitat_vars = [v for v in sweep_variables if v != seed_variable]

    uniq_df = sim_df.groupby(sweep_variables).size().reset_index(name='Freq')

    ll = []
    for _, row in uniq_df.iterrows():
        mask = pd.Series(True, index=sim_df.index)
        for var in sweep_variables:
            mask &= (sim_df[var] == row[var])
        sim_subset_df = sim_df[mask]

        comb_df = data_df.merge(sim_subset_df, on=['year', 'month'], how='left')
        ll.append(beta_binomial(comb_df.DHS_n, comb_df.Pop, comb_df.DHS_pos, comb_df.npos))

    uniq_df['ll'] = ll

    score_df = uniq_df.groupby(habitat_vars)['ll'].mean().reset_index(name='ll')
    return score_df


def plot_output(sim_df, data_df, score_df, habitat_variables):
    """
    Left panel: observed vs. simulated PfPR trajectory for every habitat
    combination (best fit highlighted, rest faint).
    Right panel: mean log-likelihood ranked across all habitat combinations
    (a single x-axis line plot doesn't work once there's more than one
    calibration dimension, so this is a ranked bar chart instead).
    """

    sim_df = sim_df.copy()
    data_df = data_df.copy()

    sim_df['date'] = pd.to_datetime([f'{int(y)}-{int(m)}-01' for y, m in zip(sim_df.year, sim_df.month)])
    data_df['date'] = pd.to_datetime([f'{int(y)}-{int(m)}-01' for y, m in zip(data_df.year, data_df.month)])
    data_df['PfPR'] = data_df['DHS_pos'] / data_df['DHS_n']

    sns.set_style('whitegrid', {'axes.linewidth': 0.5})
    fig = plt.figure(figsize=(14, 5))
    axes = [fig.add_subplot(1, 2, x + 1) for x in range(2)]

    best_row = score_df.loc[score_df['ll'].idxmax()]
    best_ll = best_row['ll']
    best_label = ', '.join(f'{v}={best_row[v]:.3g}' for v in habitat_variables)

    for _, combo in score_df.iterrows():
        mask = pd.Series(True, index=sim_df.index)
        for var in habitat_variables:
            mask &= (sim_df[var] == combo[var])
        plot_df = sim_df[mask].groupby('date', as_index=False)['PfPR'].mean()

        is_best = all(combo[v] == best_row[v] for v in habitat_variables)
        alpha = 1.0 if is_best else 0.10
        label = f'best fit ({best_label})' if is_best else None

        axes[0].plot(plot_df['date'], plot_df['PfPR'], color='#FF0000', alpha=alpha, label=label)

    axes[0].scatter(data_df['date'], data_df['PfPR'], s=data_df['DHS_n'], color='k', label='Observed')
    axes[0].set_ylabel('PfPR')
    axes[0].set_title('Observed vs Simulated (dark red = best fit)')
    axes[0].legend(loc='best', fontsize=7)

    ranked = score_df.sort_values('ll', ascending=False).reset_index(drop=True)
    colors = ['red' if i == 0 else '#FF9999' for i in range(len(ranked))]

    axes[1].bar(range(len(ranked)), ranked['ll'], color=colors)
    axes[1].set_xticks(range(len(ranked)))
    axes[1].set_xticklabels(
        [', '.join(f'{ranked.loc[i, v]:.3g}' for v in habitat_variables) for i in range(len(ranked))],
        rotation=90, fontsize=6
    )
    axes[1].set_ylabel('mean log-likelihood')
    axes[1].set_title(
        f'Ranked habitat combinations ({", ".join(habitat_variables)})\n'
        f'Best: {best_label}, ll={best_ll:.3g}'
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, expt_name, 'selection.png'))
    print(f"Wrote {os.path.join(output_dir, expt_name, 'selection.png')}")


if __name__ == "__main__":

    scores = score(sim_pfpr_df, dhs_pfpr_df, sweep_variables, seed_variable=seed_variable)

    # Sanity check against burnin_UpperWest.py's own parameter space -- catches
    # missing/failed simulations before you pick a "best fit" from incomplete data.
    if len(scores) != expected_habitat_combos:
        print(
            f"\nWARNING: expected {expected_habitat_combos} habitat combinations "
            f"({len(burnin.xTLH_range)} xTLH values x {burnin.n_habs} habitat combos, "
            f"from burnin_UpperWest.py), but found {len(scores)} in the analyzed data. "
            f"Some simulations may be missing or failed to complete."
        )

    print("\nScored habitat combinations (best first):")
    print(scores.sort_values('ll', ascending=False).to_string(index=False))

    sim_pfpr_agg = (
        sim_pfpr_df
        .groupby(['year', 'month'] + habitat_variables)['PfPR']
        .mean()
        .reset_index(name='PfPR')
    )

    plot_output(sim_pfpr_agg, dhs_pfpr_df, scores, habitat_variables)
    print(f"\nDone -- {burnin.site_name}, {expt_name}")