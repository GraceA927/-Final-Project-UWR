import os
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

        return monthly

    def reduce(self, all_data):

        if len(all_data) == 0:
            print("No data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f'All_Age_InsetChart_{self.step}.csv')

        # Stream each simulation's rows to disk as they're processed instead
        # of pd.concat()-ing all 500 results into one DataFrame in memory --
        # that concat briefly holds the per-sim list AND the combined frame
        # at once, which is its own memory spike on top of what all_data is
        # already holding.
        n_rows = 0
        n_sims = 0
        first = True

        for sim, simdata in all_data.items():
            if simdata is None or len(simdata) == 0:
                continue

            simdata.to_csv(out_path, mode='w' if first else 'a', header=first, index=False)
            first = False

            n_rows += len(simdata)
            n_sims += 1

        print(f"Wrote {n_rows} rows across {n_sims} simulations to {out_path}")


class MonthlyPfPRAnalyzer(IAnalyzer):
    """
    Pulls the monthly PfPR / clinical-incidence summary report
    (MalariaSummaryReport_monthly.json) for every simulation in the
    experiment, broken out by age bin, and writes them all to a single
    combined CSV.

    The burnin script (burnin_UpperWest.py) calls add_malaria_summary_report()
    for the full 50-year run, so this file exists during burnin too -- not
    just during the pickup stage.
    """

    def __init__(self, expt_name, sweep_variables=None, working_dir='./', start_year=0,
                 burnin=None, step="burnin"):

        super().__init__(
            working_dir=working_dir,
            filenames=["output/MalariaSummaryReport_monthly.json"]
        )

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
    # this file; if a sim is missing it, letting map() raise (and
    # partial_analyze_ok=True skip that one item) is the safe fallback rather
    # than a filter check that crashes every item.

    def map(self, data, simulation: Simulation):
        adf = pd.DataFrame()
        fname = self.filenames[0]
        age_bins = data[fname]['Metadata']['Age Bins']

        for age in range(len(age_bins)):

            d = data[fname]['DataByTimeAndAgeBins']['PfPR by Age Bin'][:-1]
            pfpr = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['Annual Clinical Incidence by Age Bin'][:-1]
            clinical_cases = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['Annual Severe Incidence by Age Bin'][:-1]
            severe_cases = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['New Infections by Age Bin'][:-1]
            new_infections = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['Annual Mild Anemia by Age Bin'][:-1]
            mild_anemia = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['Pf Gametocyte Prevalence by Age Bin'][:-1]
            gamete_prev = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['Mean Log Parasite Density by Age Bin'][:-1]
            dens = [x[age] for x in d]

            d = data[fname]['DataByTimeAndAgeBins']['Average Population by Age Bin'][:-1]
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

            adf = pd.concat([adf, simdata])

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                try:
                    adf[sweep_var] = simulation.tags[sweep_var]
                except Exception:
                    adf[sweep_var] = '-'.join([str(x) for x in simulation.tags[sweep_var]])

        adf['sim_id'] = str(simulation.uid)
        adf['Year'] = np.floor((adf['month'] - 1) / 12) + self.start_year

        return adf

    def reduce(self, all_data):

        if len(all_data) == 0:
            print("\nWarning: No data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f'\nSaving outputs to: {out_dir}')

        out_path = os.path.join(out_dir, f'PfPR_ClinicalIncidence_monthly_{self.step}.csv')

        # Same streaming-write approach as InsetChartAnalyzer.reduce() -- avoid
        # holding a second, fully-concatenated copy of all 500 sims' data in
        # memory at once on top of what all_data already holds.
        n_rows = 0
        n_sims = 0
        first = True

        for sim, simdata in all_data.items():
            if simdata is None or len(simdata) == 0:
                continue

            simdata.to_csv(out_path, mode='w' if first else 'a', header=first, index=False)
            first = False

            n_rows += len(simdata)
            n_sims += 1

        print(f"Wrote {n_rows} rows across {n_sims} simulations to {out_path}")


if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    # Replace with the actual experiment UUID printed when
    # burnin_UpperWest.py's experiment.run(...) call finishes.
    expts = {
        'UpperWest_Region_burnin50': 'c522a73f-90c8-4407-a776-1993bd2083a3'
    }

    jdir = manifest.job_directory
    wdir = os.path.join(jdir, 'outputs')
    os.makedirs(wdir, exist_ok=True)

    # Must match sim_start_year in burnin_UpperWest.py (1965), or every
    # 'date'/'Year' column downstream will be wrong.
    sim_start_year = 1963
    serialize_years = 50
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

    # Burnin equilibrium-check channels.
    channels_inset_chart = [
        'Statistical Population',
        'New Clinical Cases',
        'Adult Vectors',
        'Infected'
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

    print("\n========== FINAL CSV CHECK ==========")
    print("Rows:", len(df))
    print("Simulations:", df['sim_id'].nunique() if 'sim_id' in df.columns else "n/a")
    print("Minimum date:", df['date'].min())
    print("Maximum date:", df['date'].max())
    print("Number of years:", df['date'].dt.year.nunique())
    print("Years:", df['date'].dt.year.min(), "to", df['date'].dt.year.max())
    print("=====================================\n")

    # -------------------------------------------------------------------
    # One CSV + one plot PER SWEEP VARIABLE, instead of a single output
    # grouped jointly by all 5 sweep tags at once. Each variable's summary
    # is grouped by [date, that_variable] only -- averaged across every
    # other sweep dimension -- so you can see each parameter's marginal
    # effect on its own.
    # -------------------------------------------------------------------
    n_channels = len(channels_inset_chart)
    ncols = 2
    nrows = int(np.ceil(n_channels / ncols))

    for sweep_var in sweep_variables:

        summary = df.groupby(['date', sweep_var])[channels_inset_chart].mean().reset_index()

        csv_path = os.path.join(wdir, expt_name, f'InsetChart_summary_by_{sweep_var}_{step}.csv')
        summary.to_csv(csv_path, index=False)
        print(f"Wrote {len(summary)} rows to {csv_path}")

        # ---------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------
        fig = plt.figure(f'InsetChart_{sweep_var}', figsize=(14, 8))
        fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.08, right=0.88, bottom=0.12)
        fig.suptitle(f'Analyzer: InsetChartAnalyzer -- grouped by {sweep_var}', fontsize=16)

        axes = [fig.add_subplot(nrows, ncols, x + 1) for x in range(n_channels)]

        for ch, channel in enumerate(channels_inset_chart):
            ax = axes[ch]

            for value, pdf in summary.groupby(sweep_var):
                ax.plot(
                    pdf['date'],
                    pdf[channel],
                    linewidth=1.0,
                    label=str(value)
                )

            ax.set_title(channel, fontsize=14)
            ax.set_ylabel(channel, fontsize=11)

            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=90)
            ax.grid(True, linestyle='--', alpha=0.3)

        axes[-1].legend(
            title=sweep_var,
            fontsize=8,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5)
        )

        fig.autofmt_xdate(rotation=90)

        png_path = os.path.join(wdir, expt_name, f'InsetChart_by_{sweep_var}_{step}.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Wrote chart to {png_path}")