import os
import shutil
import datetime
import pandas as pd
import numpy as np
from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation
import manifest

## For plotting
import matplotlib
matplotlib.use('Agg')  # headless on the cluster -- no display available
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class InsetChartAnalyzer(IAnalyzer):
    """
    Pulls per-timestep InsetChart channels for every simulation in the
    experiment and writes them all to a single combined CSV.

    step is used in the output filename (not just cosmetic) so a burnin run
    and a pickup run against the same working_dir/expt_name don't clobber
    each other's CSV.
    """

    @classmethod
    def monthparser(cls, x):
        if x == 0:
            return 12
        return datetime.datetime.strptime(str(x), '%j').month

    def __init__(self, expt_name, sweep_variables=None, channels=None, working_dir=".",
                 start_year=0, step="burnin"):
        super().__init__(working_dir=working_dir, filenames=["output/InsetChart.json"])

        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.inset_channels = channels or [
            'Statistical Population',
            'New Clinical Cases',
            'Blood Smear Parasite Prevalence',
            'Infectious Vectors'
        ]
        self.expt_name = expt_name
        self.start_year = start_year
        self.step = step

    def map(self, data, simulation: Simulation):
        simdata = pd.DataFrame({
            x: data[self.filenames[0]]['Channels'][x]['Data']
            for x in self.inset_channels
        })

        simdata['Time'] = simdata.index

        # Vectorized date construction (cheap) instead of a row-wise .apply()
        # over 18,250 rows/sim.
        base_date = pd.Timestamp(year=int(self.start_year), month=1, day=1)
        simdata['date'] = base_date + pd.to_timedelta(simdata['Time'], unit='D')

        # Downsample to monthly means. Daily resolution across 500 sims x 50
        # years is ~9.1M rows total -- that's what was exhausting memory
        # before reduce() even started. A burn-in equilibrium check doesn't
        # need daily granularity; this cuts row count ~30x with no loss of
        # the trend you're actually plotting.
        simdata = simdata.set_index('date')
        monthly = simdata[self.inset_channels].resample('MS').mean()
        monthly['date'] = monthly.index
        monthly = monthly.reset_index(drop=True)

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                monthly[sweep_var] = simulation.tags[sweep_var]
            elif sweep_var == 'Run_Number':
                monthly[sweep_var] = 0

        monthly['sim_id'] = str(simulation.uid)

        # Write this simulation's rows straight to disk and return only a
        # small marker (path + row count) instead of the DataFrame itself.
        # idmtools' AnalyzeManager collects every map() return value into
        # one big in-memory dict BEFORE reduce() is ever called -- so a
        # streaming write inside reduce() (below) doesn't prevent the
        # platform from holding all 500 simulations' full DataFrames in
        # memory at once beforehand. Writing per-sim here is what actually
        # avoids that peak. See MonthlyPfPRAnalyzer for the version of this
        # that was needed to fix an actual OOM kill (this one is smaller and
        # survived so far, but the same risk grows with sim count/duration).
        out_dir = os.path.join(self.working_dir, self.expt_name, '_per_sim_insetchart')
        os.makedirs(out_dir, exist_ok=True)
        sim_path = os.path.join(out_dir, f'{simulation.uid}.csv')
        monthly.to_csv(sim_path, index=False)

        return {'sim_id': str(simulation.uid), 'path': sim_path, 'n_rows': len(monthly)}

    def reduce(self, all_data):

        if len(all_data) == 0:
            print("No data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f'All_Age_InsetChart_{self.step}.csv')

        # Stream each simulation's already-on-disk CSV into the combined
        # CSV with plain file I/O rather than re-loading it through pandas
        # -- avoids holding the full combined dataset in memory a second
        # time on top of whatever the per-sim write already avoided.
        n_rows = 0
        n_sims = 0
        first = True

        for sim, result in all_data.items():
            if not result or result.get('n_rows', 0) == 0:
                continue

            with open(result['path'], 'r') as src:
                header = src.readline()
                if first:
                    with open(out_path, 'w') as dst:
                        dst.write(header)
                    first = False
                with open(out_path, 'a') as dst:
                    for line in src:
                        dst.write(line)

            n_rows += result['n_rows']
            n_sims += 1

        print(f"Wrote {n_rows} rows across {n_sims} simulations to {out_path}")

        # Clean up the per-simulation scratch files now that everything is
        # merged into the combined CSV.
        per_sim_dir = os.path.join(out_dir, '_per_sim_insetchart')
        if os.path.isdir(per_sim_dir):
            shutil.rmtree(per_sim_dir, ignore_errors=True)


class MonthlyPfPRAnalyzer(IAnalyzer):
    """
    Pulls the monthly PfPR / clinical-incidence summary report for every
    simulation in the experiment, broken out by age bin, and writes them
    all to a single combined CSV.

    burnin_UpperWest.py's general_sim() calls add_malaria_summary_report()
    once PER CALENDAR YEAR of the 50-year burn-in (filename_suffix=
    f"Monthly_U5_{calendar_year}"), so each simulation produces one report
    file per year rather than one file covering the whole burn-in --
    e.g. output/MalariaSummaryReport_Monthly_U5_1963.json through
    ..._2012.json. This analyzer requests all of those filenames (built
    from report_years) and reassembles them per simulation in map().

    report_years must be passed in explicitly rather than imported from
    burnin_UpperWest.py: importing that module executes its top-level
    path/climate validation as a side effect (raising FileNotFoundError/
    ValueError if run from a different working directory), so it isn't
    safe to import purely to read BURNIN_START_YEAR/BURNIN_END_YEAR.

    filename_template's default assumes add_malaria_summary_report()
    names its output "MalariaSummaryReport_{filename_suffix}.json" --
    matching how the original single-file default filename_suffix=
    "monthly" produced "MalariaSummaryReport_monthly.json". If the
    installed emodpy_malaria version names report files differently,
    check one simulation's output/ directory and override
    filename_template accordingly.
    """

    def __init__(self, expt_name, report_years, sweep_variables=None, working_dir='./',
                 start_year=0, burnin=None, step="burnin",
                 filename_template="output/MalariaSummaryReport_Monthly_U5_{year}.json"):

        self.report_years = list(report_years)
        self.filename_template = filename_template

        filenames = [self.filename_template.format(year=y) for y in self.report_years]
        # Map each requested filename back to the calendar year it covers,
        # so map() can stamp rows with the correct Year without having to
        # parse it back out of the filename string.
        self.year_by_filename = dict(zip(filenames, self.report_years))

        super().__init__(working_dir=working_dir, filenames=filenames)

        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.expt_name = expt_name
        self.start_year = start_year
        self.burnin = burnin
        self.step = step

    # NOTE: no filter() override here on purpose.
    #
    # A previous version tried to skip sims missing this file via
    # simulation.get_path(), but filter() receives a restricted,
    # platform-agnostic proxy of the simulation, and get_path() isn't
    # implemented for the FileSimulation item type SlurmPlatform uses here --
    # calling it raises AttributeError and aborts the whole analyze() run.
    # burnin_UpperWest.py's set_param_fn attaches add_malaria_summary_report()
    # to the task unconditionally, so every one of the 500 sims should produce
    # these files; if a sim is missing one, skipping just that year inside
    # map() (below) is the safe fallback rather than a filter check that
    # crashes every item.

    def map(self, data, simulation: Simulation):
        adf_parts = []

        for fname in self.filenames:
            year_data = data.get(fname)
            if not year_data:
                # This one year's report is missing for this sim (e.g. a
                # truncated job) -- skip just that year rather than the
                # whole simulation.
                continue

            calendar_year = self.year_by_filename[fname]
            age_bins = year_data['Metadata']['Age Bins']

            for age in range(len(age_bins)):

                d = year_data['DataByTimeAndAgeBins']['PfPR by Age Bin'][:-1]
                pfpr = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['Annual Clinical Incidence by Age Bin'][:-1]
                clinical_cases = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['Annual Severe Incidence by Age Bin'][:-1]
                severe_cases = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['New Infections by Age Bin'][:-1]
                new_infections = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['Annual Mild Anemia by Age Bin'][:-1]
                mild_anemia = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['Pf Gametocyte Prevalence by Age Bin'][:-1]
                gamete_prev = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['Mean Log Parasite Density by Age Bin'][:-1]
                dens = [x[age] for x in d]

                d = year_data['DataByTimeAndAgeBins']['Average Population by Age Bin'][:-1]
                pop = [x[age] for x in d]

                simdata = pd.DataFrame({
                    'month': range(1, len(pfpr) + 1),
                    'PfPR': pfpr,
                    'Cases': clinical_cases,
                    'Severe_cases': severe_cases,
                    'New_infections': new_infections,
                    'Anemia': mild_anemia,
                    'Mean_density': dens,
                    'Gametocyte_prevalence': gamete_prev,
                    'Pop': pop
                })

                simdata['agebin'] = age_bins[age]
                # Stamped directly from the file this data came from, not
                # derived from a cumulative month offset -- each report
                # file only covers its own single calendar year.
                simdata['Year'] = calendar_year

                adf_parts.append(simdata)

        if not adf_parts:
            return {'sim_id': str(simulation.uid), 'n_rows': 0}

        # One concat instead of one per (year, agebin) -- with 50 years x 5
        # age bins that was 250 incremental concats per sim, each copying
        # the whole DataFrame built so far (quadratic memory churn).
        adf = pd.concat(adf_parts, ignore_index=True)

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                try:
                    adf[sweep_var] = simulation.tags[sweep_var]
                except Exception:
                    adf[sweep_var] = '-'.join([str(x) for x in simulation.tags[sweep_var]])

        adf['sim_id'] = str(simulation.uid)

        # Write this simulation's rows straight to disk and return only a
        # small marker instead of the DataFrame itself.
        #
        # This is what actually fixes the OOM kill: idmtools' AnalyzeManager
        # collects EVERY map() return value into one big in-memory dict
        # BEFORE reduce() is ever called, so a memory-efficient reduce()
        # (streaming writes, no second concat) doesn't help -- by the time
        # reduce() runs, the platform has already accumulated all 500
        # simulations' full DataFrames. With 50 years x 5 age bins x 12
        # months per sim (~3,000 rows, several columns), that's what
        # exhausted memory and got the process killed right as reduce()
        # started. Writing to disk here means the platform only ever holds
        # a tiny marker per simulation, never the data itself.
        out_dir = os.path.join(self.working_dir, self.expt_name, '_per_sim_pfpr')
        os.makedirs(out_dir, exist_ok=True)
        sim_path = os.path.join(out_dir, f'{simulation.uid}.csv')
        adf.to_csv(sim_path, index=False)

        return {'sim_id': str(simulation.uid), 'path': sim_path, 'n_rows': len(adf)}

    def reduce(self, all_data):

        if len(all_data) == 0:
            print("\nWarning: No data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f'\nSaving outputs to: {out_dir}')

        out_path = os.path.join(out_dir, f'PfPR_ClinicalIncidence_monthly_{self.step}.csv')

        # Stream each simulation's already-on-disk CSV into the combined
        # CSV with plain file I/O rather than re-loading it through pandas
        # -- avoids ever holding the full ~1.5M-row combined dataset (500
        # sims x 50 years x 5 age bins x 12 months) in memory at once.
        n_rows = 0
        n_sims = 0
        first = True

        for sim, result in all_data.items():
            if not result or result.get('n_rows', 0) == 0:
                continue

            with open(result['path'], 'r') as src:
                header = src.readline()
                if first:
                    with open(out_path, 'w') as dst:
                        dst.write(header)
                    first = False
                with open(out_path, 'a') as dst:
                    for line in src:
                        dst.write(line)

            n_rows += result['n_rows']
            n_sims += 1

        print(f"Wrote {n_rows} rows across {n_sims} simulations to {out_path}")

        # Clean up the per-simulation scratch files now that everything is
        # merged into the combined CSV.
        per_sim_dir = os.path.join(out_dir, '_per_sim_pfpr')
        if os.path.isdir(per_sim_dir):
            shutil.rmtree(per_sim_dir, ignore_errors=True)


if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    # Replace with the actual experiment UUID printed when
    # burnin_UpperWest.py's experiment.run(...) call finishes.
    expts = {
        'UpperWest_Region_burnin50': '3b47f3dc-c35e-4de0-89b3-ccef54ab945f'
    }

    jdir = manifest.job_directory
    wdir = os.path.join(jdir, 'UWR_burnin50_outputs')
    os.makedirs(wdir, exist_ok=True)

    # Must match BURNIN_START_YEAR in burnin_UpperWest.py (1963), or every
    # 'date'/'Year' column downstream will be wrong.
    sim_start_year = 1963
    serialize_years = 50
    # Must match burnin_UpperWest.py's NUM_SEEDS. The burn-in sweep uses
    # range(NUM_SEEDS), so run_num=2 means Run_Number values 0 and 1.
    run_num = 5
    step = 'burnin'

    # burnin_UpperWest.py sweeps x_Temporary_Larval_Habitat, scale_constant,
    # scale_temp_rain, scale_water_veg (via update_campaign_multiple_parameters),
    # and Run_Number -- all five tags need to be listed here or map() silently
    # drops them and every simulation's data becomes indistinguishable from
    # every other's once concatenated in reduce().
    sweep_variables = [
        'x_Temporary_Larval_Habitat',
        'scale_constant',
        'scale_temp_rain',
        'scale_water_veg',
        'Run_Number'
    ]

    # Burnin has no campaign-broadcasting interventions, so channels tied to
    # intervention effects aren't meaningful here. These four reflect burnin
    # equilibrium behavior: population size, vector population, transmission
    # intensity, and parasite prevalence.
    channels_inset_chart = [
        'Statistical Population',
        'Adult Vectors',
        'Daily EIR',
        'PCR Parasite Prevalence'
    ]

    # Capture BOTH output files produced by every simulation in the burnin
    # sweep: the per-timestep InsetChart, and the monthly malaria summary
    # report (since burnin_UpperWest.py calls add_malaria_summary_report()
    # for the full 50-year run). partial_analyze_ok=True below means any
    # individual sim that's missing a file is skipped rather than aborting
    # the whole run.
    analyzers_burnin = [
        InsetChartAnalyzer(
            expt_name=list(expts.keys())[0],
            channels=channels_inset_chart,
            start_year=sim_start_year,
            sweep_variables=sweep_variables,
            step=step,
            working_dir=wdir
        ),
        MonthlyPfPRAnalyzer(
            expt_name=list(expts.keys())[0],
            report_years=range(sim_start_year, sim_start_year + serialize_years),
            start_year=sim_start_year,
            sweep_variables=sweep_variables,
            step=step,
            working_dir=wdir
        )
    ]

    with Platform('SLURM_LOCAL', job_directory=jdir) as platform:

        for expt_name, exp_id in expts.items():

            am = AnalyzeManager(
                configuration={},
                ids=[(exp_id, ItemType.EXPERIMENT)],
                analyzers=analyzers_burnin,
                partial_analyze_ok=True
            )
            am.analyze()

    # -------------------------------------------------------------------
    # Read back the InsetChart CSV and confirm what was captured
    # -------------------------------------------------------------------
    expt_name = list(expts.keys())[0]

    df = pd.read_csv(os.path.join(wdir, expt_name, f'All_Age_InsetChart_{step}.csv'))
    df['date'] = pd.to_datetime(df['date'])

    expected_run_numbers = set(range(run_num))
    actual_run_numbers = set(
        pd.to_numeric(df['Run_Number'], errors='raise').dropna().astype(int).unique()
    )
    if actual_run_numbers != expected_run_numbers:
        raise ValueError(
            f'Run_Number mismatch for {expt_name}: analyzer expects '
            f'{sorted(expected_run_numbers)} from run_num={run_num}, but the '
            f'experiment contains {sorted(actual_run_numbers)}. '
            'Use the experiment generated by burnin_UpperWest.py or update run_num.'
        )
    print(f'Run_Number values verified: {sorted(actual_run_numbers)}')

    #print("\n========== FINAL CSV CHECK ==========")
    #print("Rows:", len(df))
    #print("Simulations:", df['sim_id'].nunique() if 'sim_id' in df.columns else "n/a")
    #print("Minimum date:", df['date'].min())
    #print("Maximum date:", df['date'].max())
    #print("Number of years:", df['date'].dt.year.nunique())
    #print("Years:", df['date'].dt.year.min(), "to", df['date'].dt.year.max())
    #print("=====================================\n")

    # -------------------------------------------------------------------
    # Generic multi-panel plotter: one figure, one subplot per channel,
    # one line per distinct value of group_col (all other dimensions
    # averaged away). This replaces overlaying every one of the ~200
    # parameter combinations as individual lines in one crowded plot --
    # that "spaghetti + shaded cloud" look comes from plotting every sweep
    # combo at once. Calling this once per sweep dimension instead gives a
    # clean, readable figure per dimension (Image 2's style), each with
    # only as many lines as that dimension has distinct values.
    # -------------------------------------------------------------------
    def plot_channels_by_group(plot_df, group_col, channels, x_col, title, path, x_is_date=False):
        n_channels = len(channels)
        ncols = 2
        nrows = int(np.ceil(n_channels / ncols))

        fig = plt.figure(figsize=(14, 4 * nrows))
        fig.subplots_adjust(hspace=0.5, wspace=0.25, left=0.08, right=0.85, bottom=0.12, top=0.90)
        fig.suptitle(title, fontsize=16)

        axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n_channels)]
        group_values = sorted(plot_df[group_col].unique())

        for i, channel in enumerate(channels):
            ax = axes[i]

            for gv in group_values:
                gdf = plot_df[plot_df[group_col] == gv].sort_values(x_col)

                if group_col == 'Run_Number':
                    label = f'Run {int(gv)}'
                elif isinstance(gv, (float, np.floating)):
                    label = f'{group_col}={gv:g}'
                else:
                    label = f'{group_col}={gv}'

                ax.plot(gdf[x_col], gdf[channel], linewidth=1.2, label=label)

            ax.set_title(channel, fontsize=13)
            ax.set_ylabel(channel, fontsize=10)

            if x_is_date:
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.tick_params(axis='x', rotation=90)
            else:
                ax.set_xlabel(x_col, fontsize=10)

            ax.grid(True, linestyle='--', alpha=0.3)

        axes[-1].legend(
            title=group_col,
            fontsize=8,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5)
        )

        if x_is_date:
            fig.autofmt_xdate(rotation=90)

        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote chart to {path}')

    # The parameter sweep, excluding Run_Number (the stochastic seed).
    plot_sweep_variables = [x for x in sweep_variables if x != 'Run_Number']

    # By Run_Number: every parameter combo averaged away, one line per seed.
    by_run = df.groupby(['date', 'Run_Number'])[channels_inset_chart].mean().reset_index()
    plot_channels_by_group(
        by_run, 'Run_Number', channels_inset_chart, 'date',
        f'Analyzer: InsetChartAnalyzer (by Run_Number, {sim_start_year}-{sim_start_year + serialize_years - 1} burn-in)',
        os.path.join(wdir, expt_name, f'InsetChart_{step}_byRunNumber.png'),
        x_is_date=True
    )

    # By each individually swept parameter: seeds and the other three
    # parameters averaged away, one line per value of that parameter.
    #
    # Commented out for now -- keeping only the by-Run_Number plot active,
    # matching the simpler reference analyzer script's single-plot style.
    # Left in place (rather than deleted) so it's a one-line uncomment to
    # bring back the per-parameter sensitivity views if/when they're needed
    # again.
    #
    # for param in plot_sweep_variables:
    #     by_param = df.groupby(['date', param])[channels_inset_chart].mean().reset_index()
    #     plot_channels_by_group(
    #         by_param, param, channels_inset_chart, 'date',
    #         f'Analyzer: InsetChartAnalyzer (by {param})',
    #         os.path.join(wdir, expt_name, f'InsetChart_{step}_by_{param}.png'),
    #         x_is_date=True
    #     )

    # -------------------------------------------------------------------
    # Read back the PfPR / clinical-incidence summary-report CSV and plot it
    #
    # MonthlyPfPRAnalyzer writes one row per (sim, month, agebin), so unlike
    # the InsetChart CSV above this one is NOT collapsible to a single
    # sweep-group line per subplot without also choosing what to do with the
    # agebin axis. We average across sweep_variables/sims (same as the
    # InsetChart block) but keep agebin as the plotted series, since burnin
    # equilibrium behavior is usually checked per age group (e.g. U5 vs
    # older bins settling at different PfPR/EIR-driven prevalence).
    # -------------------------------------------------------------------
    summary_csv_path = os.path.join(wdir, expt_name, f'PfPR_ClinicalIncidence_monthly_{step}.csv')

    if not os.path.isfile(summary_csv_path):
        print(f"\nWarning: {summary_csv_path} not found -- skipping summary report plot.")
    else:
        sdf = pd.read_csv(summary_csv_path)

        channels_summary_report = [
            'PfPR',
            'Cases',
            'Severe_cases',
            'New_infections',
            'Anemia',
            'Mean_density',
            'Gametocyte_prevalence',
            'Pop'
        ]

        # Guard against a schema drift (e.g. a column renamed upstream in
        # MonthlyPfPRAnalyzer.map()) silently producing a KeyError deep in
        # groupby() below.
        missing_channels = [c for c in channels_summary_report if c not in sdf.columns]
        if missing_channels:
            raise KeyError(
                f"{summary_csv_path} is missing expected column(s): {missing_channels}. "
                "Check that MonthlyPfPRAnalyzer.map() and channels_summary_report still agree."
            )

        # One set of figures per age bin (PfPR/clinical incidence differ a
        # lot by age, so collapsing agebin away would hide that). Within
        # each age bin: one "by Run_Number" figure (all four swept
        # parameters averaged away, one line per seed -- Image 2's style),
        # plus one figure per individually swept parameter (seeds and the
        # other three parameters averaged away, one line per value of that
        # parameter) -- same breakdown as the InsetChart plots above.
        for agebin in sorted(sdf['agebin'].unique()):
            age_sdf = sdf[sdf['agebin'] == agebin]
            age_label = f'{agebin:g}' if isinstance(agebin, (float, np.floating)) else f'{agebin}'

            by_run = age_sdf.groupby(['Year', 'Run_Number'])[channels_summary_report].mean().reset_index()
            plot_channels_by_group(
                by_run, 'Run_Number', channels_summary_report, 'Year',
                f'Analyzer: MonthlyPfPRAnalyzer (Age {age_label}, by Run_Number)',
                os.path.join(wdir, expt_name, f'PfPR_ClinicalIncidence_{step}_Age{age_label}_byRunNumber.png')
            )

            # By each individually swept parameter: commented out for now,
            # keeping only the by-Run_Number plot active per age bin (see
            # the matching InsetChart comment above for why). Uncomment to
            # bring back the per-parameter sensitivity views.
            #
            for param in plot_sweep_variables:
                by_param = age_sdf.groupby(['Year', param])[channels_summary_report].mean().reset_index()
                plot_channels_by_group(
                    by_param, param, channels_summary_report, 'Year',
                    f'Analyzer: MonthlyPfPRAnalyzer (Age {age_label}, by {param})',
                    os.path.join(wdir, expt_name, f'PfPR_ClinicalIncidence_{step}_Age{age_label}_by_{param}.png')
                )