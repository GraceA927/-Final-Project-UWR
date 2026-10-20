################################################################################
# UWR Pickup Analyzer -- for experiments submitted by run_pickup_UWR.py
#
# Adapted from the burn-in analyzer (InsetChartAnalyzer / MonthlyPfPRAnalyzer),
# keeping its memory-safe engineering unchanged: per-simulation results are
# streamed straight to disk in map() and stitched together with plain file
# I/O in reduce(), because idmtools' AnalyzeManager collects every map()
# return value into memory BEFORE reduce() ever runs -- a "smart" reduce()
# alone does not avoid the peak. See the burn-in analyzer's docstrings for
# the fuller explanation; it isn't repeated per class here.
#
# What's different for pickup vs. burn-in:
#   - Calendar: PICKUP_START_YEAR/PICKUP_YEARS (2013-2025), not 1963-2012.
#   - Report years for MalariaSummaryReport_Monthly_U5_{year}.json follow
#     run_pickup_UWR.py's add_reports(), which calls
#     add_malaria_summary_report() once per pickup year (2013 .. 2025) --
#     same filename convention as burn-in, different year range.
#   - Sweep/tag variables: pickup tags come from run_pickup_UWR.py's
#     update_serialize_parameters() (Run_Number, burnin_sim_index, and the
#     burnin_* provenance tags) and update_pickup_habitat() (this pickup's
#     OWN scale_constant/scale_temp_rain/scale_water_veg). The batch's xTLH
#     value is NOT a per-simulation tag -- set_param() in run_pickup_UWR.py
#     doesn't return one, and every simulation in a --batch N experiment
#     shares the same xTLH anyway -- so it's supplied per experiment via
#     `extra_columns` below instead of read from simulation.tags.
#   - New: EventRecorderAnalyzer, for output/ReportEventRecorder.csv, which
#     only exists for pickup (burn-in has no broadcasting interventions).
#     This is what lets you directly check the CM/ITN/IRS/SMC/vaccine
#     event chain -- e.g. whether Received_Vaccine_D2_2024 counts look
#     sane relative to Received_Vaccine_D1_2024 -- straight from the data
#     rather than only from campaign_pickup_preview.json.
#
# USAGE
#   1. Fill in `expts` below with each finished pickup batch's experiment
#      ID (from `idmtools slurm ... status --exp-id ...`, SUCCEEDED) and
#      that batch's xTLH value (printed by run_pickup_UWR.py when it was
#      submitted).
#   2. Run this script. Each batch gets its own output subfolder --
#      results are never overwritten across batches -- so combining
#      multiple batches into one table, if you want that, is a separate
#      pandas concat afterward rather than something this script assumes.
################################################################################

import io
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

# Must match run_pickup_UWR.py's calendar. Not imported from that script on
# purpose -- importing it executes its top-level input/climate validation
# as a side effect (raises FileNotFoundError/ValueError if run from a
# different working directory), so it isn't safe to import purely to read
# a couple of constants.
PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1
DAYS_PER_YEAR = 365


class InsetChartAnalyzer(IAnalyzer):
    """
    Pulls per-timestep InsetChart channels for every simulation in the
    experiment and writes them all to a single combined CSV.

    step is used in the output filename (not just cosmetic) so this
    pickup run and a burn-in run against the same working_dir/expt_name
    don't clobber each other's CSV.
    """

    def __init__(self, expt_name, sweep_variables=None, channels=None, working_dir=".",
                 start_year=PICKUP_START_YEAR, step="pickup", extra_columns=None):
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
        # Static columns stamped onto every row of THIS experiment (e.g.
        # {'xTLH': 100.0, 'batch': 'batch2'}) -- for values that are
        # constant across the whole experiment rather than a per-sim tag.
        self.extra_columns = extra_columns or {}

    def map(self, data, simulation: Simulation):
        simdata = pd.DataFrame({
            x: data[self.filenames[0]]['Channels'][x]['Data']
            for x in self.inset_channels
        })

        simdata['Time'] = simdata.index

        # Vectorized date construction (cheap) instead of a row-wise .apply().
        base_date = pd.Timestamp(year=int(self.start_year), month=1, day=1)
        simdata['date'] = base_date + pd.to_timedelta(simdata['Time'], unit='D')

        # Downsample to monthly means -- 13 years x 2,500 sims at daily
        # resolution is still ~12M rows; monthly cuts that ~30x with no
        # loss of the trend actually being plotted.
        simdata = simdata.set_index('date')
        monthly = simdata[self.inset_channels].resample('MS').mean()
        monthly['date'] = monthly.index
        monthly = monthly.reset_index(drop=True)

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                monthly[sweep_var] = simulation.tags[sweep_var]
            elif sweep_var == 'Run_Number':
                monthly[sweep_var] = 0

        for key, value in self.extra_columns.items():
            monthly[key] = value

        monthly['sim_id'] = str(simulation.uid)

        # Write this simulation's rows straight to disk and return only a
        # small marker -- see module docstring for why this (not a
        # memory-light reduce()) is what actually avoids the OOM peak.
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

        per_sim_dir = os.path.join(out_dir, '_per_sim_insetchart')
        if os.path.isdir(per_sim_dir):
            shutil.rmtree(per_sim_dir, ignore_errors=True)


class MonthlyPfPRAnalyzer(IAnalyzer):
    """
    Pulls the monthly PfPR / clinical-incidence summary report for every
    simulation, broken out by age bin, and writes them all to a single
    combined CSV.

    run_pickup_UWR.py's add_reports() calls add_malaria_summary_report()
    once PER PICKUP YEAR (filename_suffix=f"Monthly_U5_{PICKUP_START_YEAR
    + i}"), same convention as the burn-in script, so this analyzer is
    unchanged from the burn-in version apart from which years it's told
    to look for (report_years, passed in explicitly below).
    """

    def __init__(self, expt_name, report_years, sweep_variables=None, working_dir='./',
                 start_year=PICKUP_START_YEAR, burnin=None, step="pickup",
                 filename_template="output/MalariaSummaryReport_Monthly_U5_{year}.json",
                 extra_columns=None):

        self.report_years = list(report_years)
        self.filename_template = filename_template

        filenames = [self.filename_template.format(year=y) for y in self.report_years]
        self.year_by_filename = dict(zip(filenames, self.report_years))

        super().__init__(working_dir=working_dir, filenames=filenames)

        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.expt_name = expt_name
        self.start_year = start_year
        self.burnin = burnin
        self.step = step
        self.extra_columns = extra_columns or {}

    # NOTE: no filter() override here on purpose -- see the burn-in
    # analyzer for why (get_path() isn't implemented for SlurmPlatform's
    # FileSimulation item type; missing files are skipped inside map()
    # instead, below).

    def map(self, data, simulation: Simulation):
        adf_parts = []

        for fname in self.filenames:
            year_data = data.get(fname)
            if not year_data:
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
                simdata['Year'] = calendar_year

                adf_parts.append(simdata)

        if not adf_parts:
            return {'sim_id': str(simulation.uid), 'n_rows': 0}

        adf = pd.concat(adf_parts, ignore_index=True)

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                try:
                    adf[sweep_var] = simulation.tags[sweep_var]
                except Exception:
                    adf[sweep_var] = '-'.join([str(x) for x in simulation.tags[sweep_var]])

        for key, value in self.extra_columns.items():
            adf[key] = value

        adf['sim_id'] = str(simulation.uid)

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

        per_sim_dir = os.path.join(out_dir, '_per_sim_pfpr')
        if os.path.isdir(per_sim_dir):
            shutil.rmtree(per_sim_dir, ignore_errors=True)


class EventRecorderAnalyzer(IAnalyzer):
    """
    Pulls output/ReportEventRecorder.csv -- the file add_event_recorder()
    in run_pickup_UWR.py writes, with one row per (individual, event) --
    and writes a combined, per-sim/year/event ROW-COUNT summary. This is
    what lets you check the CM/ITN/IRS/SMC/vaccine event chain directly:
    e.g. whether Received_Vaccine_D2_2024 counts are a sane fraction of
    Received_Vaccine_D1_2024, whether Received_SMC counts land where the
    campaign schedule says they should, etc.

    Doesn't exist for the burn-in analyzer because burn-in has no
    broadcasting interventions to record.

    ASSUMPTIONS TO VERIFY against your actual output/ReportEventRecorder.csv:
      - idmtools returns .csv report content already parsed as a pandas
        DataFrame (as it does for .json). If your installed idmtools
        version instead hands back raw text/bytes for CSV files, the
        isinstance check in map() below falls back to pd.read_csv() on
        that content -- but if it comes back some other way entirely,
        this will need a small adjustment.
      - The event-name column is called "Event_Name" and the simulation
        time column is called "Time" (both are EMOD/emodpy_malaria
        defaults) -- check your CSV's header and adjust EVENT_COL/TIME_COL
        below if your installed version names them differently.
    """

    EVENT_COL = "Event_Name"
    TIME_COL = "Time"

    def __init__(self, expt_name, sweep_variables=None, working_dir=".",
                 start_year=PICKUP_START_YEAR, step="pickup",
                 filename="output/ReportEventRecorder.csv", extra_columns=None):
        super().__init__(working_dir=working_dir, filenames=[filename])
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.expt_name = expt_name
        self.start_year = start_year
        self.step = step
        self.extra_columns = extra_columns or {}

    def map(self, data, simulation: Simulation):
        raw = data.get(self.filenames[0])
        if raw is None:
            return {'sim_id': str(simulation.uid), 'n_rows': 0}

        if isinstance(raw, pd.DataFrame):
            edf = raw.copy()
        elif isinstance(raw, (bytes, bytearray)):
            edf = pd.read_csv(io.BytesIO(raw))
        else:
            edf = pd.read_csv(io.StringIO(raw))

        if edf.empty or self.EVENT_COL not in edf.columns:
            return {'sim_id': str(simulation.uid), 'n_rows': 0}

        if self.TIME_COL in edf.columns:
            edf['Year'] = self.start_year + (edf[self.TIME_COL].astype(float) // DAYS_PER_YEAR).astype(int)
        else:
            edf['Year'] = np.nan

        counts = (
            edf.groupby(['Year', self.EVENT_COL]).size()
            .reset_index(name='Count')
            .rename(columns={self.EVENT_COL: 'Event_Name'})
        )

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                counts[sweep_var] = simulation.tags[sweep_var]
            elif sweep_var == 'Run_Number':
                counts[sweep_var] = 0

        for key, value in self.extra_columns.items():
            counts[key] = value

        counts['sim_id'] = str(simulation.uid)

        out_dir = os.path.join(self.working_dir, self.expt_name, '_per_sim_events')
        os.makedirs(out_dir, exist_ok=True)
        sim_path = os.path.join(out_dir, f'{simulation.uid}.csv')
        counts.to_csv(sim_path, index=False)

        return {'sim_id': str(simulation.uid), 'path': sim_path, 'n_rows': len(counts)}

    def reduce(self, all_data):

        if len(all_data) == 0:
            print("\nWarning: No event data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f'EventCounts_{self.step}.csv')

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

        per_sim_dir = os.path.join(out_dir, '_per_sim_events')
        if os.path.isdir(per_sim_dir):
            shutil.rmtree(per_sim_dir, ignore_errors=True)


# -----------------------------------------------------------------------
# Generic multi-panel plotter: one figure, one subplot per channel, one
# line per distinct value of group_col (all other dimensions averaged
# away). Unchanged from the burn-in analyzer.
# -----------------------------------------------------------------------
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
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
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


def plot_event_counts_by_year(event_df, path, title):
    """One line per Event_Name, summed across every sim/tag, by Year --
    a quick sanity check that the intervention chain (CM/SMC/vaccine
    doses, etc.) is firing at plausible relative volumes."""
    totals = event_df.groupby(['Year', 'Event_Name'])['Count'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    for event_name, edf in totals.groupby('Event_Name'):
        edf = edf.sort_values('Year')
        ax.plot(edf['Year'], edf['Count'], marker='o', linewidth=1.2, label=event_name)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total event count (all simulations)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote chart to {path}')


if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    # One entry per finished pickup batch (from run_pickup_UWR.py
    # --batch N). Add the remaining batches here as they finish -- each
    # gets its own output subfolder so results never overwrite each
    # other; concat the resulting CSVs yourself if you want one
    # all-batches table.
    #   name  ->  (experiment_id, that batch's xTLH value)
    # Fill in the xTLH value from the "xTLH: X.XXXX" line
    # run_pickup_UWR.py printed when that batch was submitted.
    expts = {
        'UWR_Pickup_batch_90b2f488': ('90b2f488-c183-4c28-bd70-05eb497e3756', None),  # TODO: fill in this batch's xTLH
        # 'UWR_Pickup_batch0': ('<exp id>', 70.79),
        # 'UWR_Pickup_batch1': ('<exp id>', 84.14),
        # 'UWR_Pickup_batch2': ('<exp id>', 100.0),
        # 'UWR_Pickup_batch3': ('<exp id>', 118.85),
        # 'UWR_Pickup_batch4': ('<exp id>', 141.25),
    }

    jdir = manifest.job_directory
    wdir = os.path.join(jdir, 'UWR_pickup_outputs')
    os.makedirs(wdir, exist_ok=True)

    step = 'pickup'
    report_years = range(PICKUP_START_YEAR, PICKUP_END_YEAR + 1)

    # Tags set by run_pickup_UWR.py's update_serialize_parameters() and
    # update_pickup_habitat(). The batch's own xTLH is NOT one of these --
    # see the module docstring -- it's supplied via extra_columns below.
    sweep_variables = [
        'Run_Number',                          # update_serialize_parameters
        'burnin_sim_index',                    # update_serialize_parameters
        'burnin_x_Temporary_Larval_Habitat',   # burn-in provenance only (Option A)
        'burnin_scale_constant',
        'burnin_scale_temp_rain',
        'burnin_scale_water_veg',
        'scale_constant',                      # THIS pickup's own habitat mix
        'scale_temp_rain',
        'scale_water_veg',
    ]

    channels_inset_chart = [
        'Statistical Population',
        'New Clinical Cases',
        'Blood Smear Parasite Prevalence',
        'Infectious Vectors',
    ]

    for expt_name, (exp_id, xtlh) in expts.items():
        extra_columns = {'batch': expt_name}
        if xtlh is not None:
            extra_columns['xTLH'] = xtlh

        analyzers_pickup = [
            InsetChartAnalyzer(
                expt_name=expt_name,
                channels=channels_inset_chart,
                start_year=PICKUP_START_YEAR,
                sweep_variables=sweep_variables,
                step=step,
                working_dir=wdir,
                extra_columns=extra_columns,
            ),
            MonthlyPfPRAnalyzer(
                expt_name=expt_name,
                report_years=report_years,
                start_year=PICKUP_START_YEAR,
                sweep_variables=sweep_variables,
                step=step,
                working_dir=wdir,
                extra_columns=extra_columns,
            ),
            EventRecorderAnalyzer(
                expt_name=expt_name,
                start_year=PICKUP_START_YEAR,
                sweep_variables=sweep_variables,
                step=step,
                working_dir=wdir,
                extra_columns=extra_columns,
            ),
        ]

        with Platform('SLURM_LOCAL', job_directory=jdir) as platform:
            am = AnalyzeManager(
                configuration={},
                ids=[(exp_id, ItemType.EXPERIMENT)],
                analyzers=analyzers_pickup,
                partial_analyze_ok=True
            )
            am.analyze()

    # -------------------------------------------------------------------
    # Read back each batch's CSVs and make plots
    # -------------------------------------------------------------------
    for expt_name in expts:
        out_dir = os.path.join(wdir, expt_name)

        inset_path = os.path.join(out_dir, f'All_Age_InsetChart_{step}.csv')
        if os.path.isfile(inset_path):
            df = pd.read_csv(inset_path)
            df['date'] = pd.to_datetime(df['date'])

            by_run = df.groupby(['date', 'Run_Number'])[channels_inset_chart].mean().reset_index()
            plot_channels_by_group(
                by_run, 'Run_Number', channels_inset_chart, 'date',
                f'{expt_name}: InsetChartAnalyzer (by Run_Number, {PICKUP_START_YEAR}-{PICKUP_END_YEAR} pickup)',
                os.path.join(out_dir, f'InsetChart_{step}_byRunNumber.png'),
                x_is_date=True,
            )
        else:
            print(f"\n{expt_name}: no {inset_path} -- skipping InsetChart plot.")

        event_path = os.path.join(out_dir, f'EventCounts_{step}.csv')
        if os.path.isfile(event_path):
            edf = pd.read_csv(event_path)
            plot_event_counts_by_year(
                edf,
                os.path.join(out_dir, f'EventCounts_{step}_byYear.png'),
                f'{expt_name}: event counts by year (CM/ITN/IRS/SMC/vaccine)',
            )
        else:
            print(f"\n{expt_name}: no {event_path} -- skipping event-count plot.")

        pfpr_path = os.path.join(out_dir, f'PfPR_ClinicalIncidence_monthly_{step}.csv')
        if not os.path.isfile(pfpr_path):
            print(f"\n{expt_name}: no {pfpr_path} -- skipping PfPR plot.")
            continue
        # PfPR/clinical-incidence plotting by age bin left for you to add
        # here in the same style as plot_channels_by_group() above, once
        # you know which age bins and years you want to compare first --
        # deliberately not auto-generated, since with 5 age bins x 13
        # years x however many batches, an unrequested default set of
        # figures here would likely be the wrong slice to look at first.