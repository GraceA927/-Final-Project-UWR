"""
Analyzer for the Upper West Region burn-in (run_burnin_UWR.py).

Adapted from the example analyzer. It reads three kinds of output from every
simulation in the burn-in experiment and writes combined CSVs and plots.

    InsetChart.json                           -> monthly means of InsetChart channels
    MalariaSummaryReport_Monthly_U5_<yr>.json -> PfPR / incidence by age bin, 1998-2012
    ReportEventRecorder.csv                   -> yearly counts of Pregnant / GaveBirth

Usage
-----
Full burn-in:
    python analyze_burnin_UWR.py

Birth-rate calibration test (run_burnin_UWR.py --test-birth):
    python analyze_burnin_UWR.py --test-birth
  Only InsetChart exists in those runs (reports and the recorder are off).

Re-plot from CSVs already written, without re-reading the simulations:
    python analyze_burnin_UWR.py --plots-only

Before running, paste the experiment ID printed by run_burnin_UWR.py
into EXPTS below.

Calendar
--------
The model uses flat 365-day years with no leap days. Dates here are built from
that calendar (Year + Month from cumulative month lengths), not from real
dates, so there is no drift against the 1963-2012 labels.
"""

import argparse
import os

import numpy as np
import pandas as pd

from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation

import manifest

import matplotlib
matplotlib.use("Agg")   # headless on the cluster
import matplotlib.pyplot as plt


################################################################################
# SETTINGS  --  MUST MATCH run_burnin_UWR.py
################################################################################

BURNIN_START_YEAR = 1963
DAYS_PER_YEAR = 365

REPORT_START_YEAR = 1998
REPORT_END_YEAR = 2012
REPORT_INTERVAL = 30            # days per reporting period
EVENT_LIST = [
    "Births", "DiseaseDeaths", "NonDiseaseDeaths",
    "NewInfectionEvent", "NewClinicalCase",
    "Received_Treatment", "Received_Severe_Treatment",
    "Pregnant", "GaveBirth",
]

# The example analyzer dropped the last entry of every time series ([:-1]).
# Keep that behaviour unless a check of one MalariaSummaryReport file shows it
# is not needed (see "periods kept" printed after the analysis).
DROP_LAST_PERIOD = True

# Upper edge of the age bin to plot as the prevalence-survey group
# (bins are [0.25, 0.5, 5, 15, 25, 100], so the 0.5-5 year bin is "5").
PREVALENCE_BIN_UPPER = 5

# Experiment name -> experiment ID (printed when run_burnin_UWR.py submits).
EXPTS = {
    "UWR_50yrs_With_ECM_PREG": "207f3dfa-3ddb-4810-a524-e9b4513dae5d",
}

# Tags set by the full burn-in sweeps.
SWEEP_VARIABLES_FULL = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
    "Run_Number",
]
# Tags set by the --test-birth sweeps.
SWEEP_VARIABLES_TEST = ["x_Birth", "Run_Number"]

INSET_CHANNELS = [
    "Statistical Population",
    "New Clinical Cases",
    "Adult Vectors",
    "Infected",
]

MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_END = np.cumsum(MONTH_LENGTHS)   # 31, 59, ..., 365


################################################################################
# HELPERS
################################################################################

def day_to_year_month(time_index, start_year):
    """Flat 365-day calendar: simulation day -> (calendar year, month 1-12)."""
    t = np.asarray(time_index, dtype=int)
    year = start_year + t // DAYS_PER_YEAR
    day_of_year = t % DAYS_PER_YEAR
    month = np.searchsorted(MONTH_END, day_of_year, side="right") + 1
    return year, month


def add_sweep_tags(df, simulation, sweep_variables):
    """Copy the simulation's sweep tags onto every row."""
    for sweep_var in sweep_variables:
        if sweep_var in simulation.tags.keys():
            value = simulation.tags[sweep_var]
            try:
                df[sweep_var] = value
            except Exception:
                df[sweep_var] = "-".join(str(x) for x in value)
        elif sweep_var == "Run_Number":
            df[sweep_var] = 0
    df["sim_id"] = str(simulation.uid)
    return df


def stream_to_csv(all_data, out_path):
    """
    Write each simulation's rows to disk as they are processed instead of
    concatenating everything in memory first (1,250 simulations).
    """
    n_rows = 0
    n_sims = 0
    first = True

    for _, simdata in all_data.items():
        if simdata is None or len(simdata) == 0:
            continue
        simdata.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(simdata)
        n_sims += 1

    print(f"Wrote {n_rows} rows across {n_sims} simulations to {out_path}")


def coerce_numeric(df, columns):
    """Tags can be read back as text; convert columns that are fully numeric."""
    for col in columns:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().all():
                df[col] = converted
    return df


################################################################################
# ANALYZER 1: InsetChart
################################################################################

class InsetChartAnalyzer(IAnalyzer):
    """
    Monthly means of InsetChart channels for every simulation.

    Daily data for 1,250 simulations x 50 years is too large to hold in memory,
    so each simulation is reduced to monthly means inside map().
    """

    def __init__(self, expt_name, sweep_variables=None, channels=None,
                 working_dir=".", start_year=BURNIN_START_YEAR, step="burnin"):
        super().__init__(working_dir=working_dir, filenames=["output/InsetChart.json"])
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.inset_channels = channels or INSET_CHANNELS
        self.expt_name = expt_name
        self.start_year = start_year
        self.step = step

    def map(self, data, simulation: Simulation):
        channels = data[self.filenames[0]]["Channels"]
        simdata = pd.DataFrame({x: channels[x]["Data"] for x in self.inset_channels})

        year, month = day_to_year_month(np.arange(len(simdata)), self.start_year)
        simdata["Year"] = year
        simdata["Month"] = month

        monthly = (simdata.groupby(["Year", "Month"], as_index=False)[self.inset_channels]
                   .mean())
        monthly["date"] = pd.to_datetime(
            pd.DataFrame({"year": monthly["Year"], "month": monthly["Month"], "day": 1})
        )

        return add_sweep_tags(monthly, simulation, self.sweep_variables)

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("No InsetChart data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)
        stream_to_csv(all_data, os.path.join(out_dir, f"All_Age_InsetChart_{self.step}.csv"))


################################################################################
# ANALYZER 2: Malaria summary reports (one file per year, 1998-2012)
################################################################################

class MonthlyPfPRAnalyzer(IAnalyzer):
    """
    PfPR, incidence and related outputs by age bin, from the yearly
    MalariaSummaryReport_Monthly_U5_<year>.json files.

    The burn-in writes one file per year, each with 30-day reporting periods,
    so 'period' is a 30-day block, not a calendar month.
    """

    def __init__(self, expt_name, sweep_variables=None, working_dir="./",
                 start_year=BURNIN_START_YEAR, step="burnin",
                 report_start_year=REPORT_START_YEAR, report_end_year=REPORT_END_YEAR):
        self.report_years = list(range(report_start_year, report_end_year + 1))
        super().__init__(
            working_dir=working_dir,
            filenames=[f"output/MalariaSummaryReport_Monthly_U5_{y}.json"
                       for y in self.report_years],
        )
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.expt_name = expt_name
        self.start_year = start_year
        self.step = step

    # Output column -> key in DataByTimeAndAgeBins
    FIELDS = {
        "PfPR": "PfPR by Age Bin",
        "Cases": "Annual Clinical Incidence by Age Bin",
        "Severe_cases": "Annual Severe Incidence by Age Bin",
        "New_infections": "New Infections by Age Bin",
        "Anemia": "Annual Mild Anemia by Age Bin",
        "Mean_density": "Mean Log Parasite Density by Age Bin",
        "Gametocyte_prevalence": "Pf Gametocyte Prevalence by Age Bin",
        "Pop": "Average Population by Age Bin",
    }

    def map(self, data, simulation: Simulation):
        frames = []

        for year, fname in zip(self.report_years, self.filenames):
            report = data.get(fname)
            if report is None:
                continue   # this simulation has no report for that year

            age_edges = report["Metadata"]["Age Bins"]
            by_time = report["DataByTimeAndAgeBins"]

            n_periods = len(by_time["PfPR by Age Bin"])
            if DROP_LAST_PERIOD:
                n_periods -= 1

            for a, upper in enumerate(age_edges):
                lower = 0 if a == 0 else age_edges[a - 1]

                cols = {"period": range(1, n_periods + 1)}
                for out_name, key in self.FIELDS.items():
                    cols[out_name] = [row[a] for row in by_time[key][:n_periods]]

                df = pd.DataFrame(cols)
                df["Year"] = year
                df["agebin_upper"] = upper
                df["agebin"] = f"{lower:g}-{upper:g}"
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        adf = pd.concat(frames, ignore_index=True)

        # Start of each 30-day period on the flat 365-day calendar.
        adf["day_in_year"] = (adf["period"] - 1) * REPORT_INTERVAL
        adf["year_frac"] = adf["Year"] + adf["day_in_year"] / DAYS_PER_YEAR
        adf["sim_day"] = (adf["Year"] - self.start_year) * DAYS_PER_YEAR + adf["day_in_year"]

        return add_sweep_tags(adf, simulation, self.sweep_variables)

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("\nWarning: No summary-report data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nSaving outputs to: {out_dir}")
        stream_to_csv(all_data,
                      os.path.join(out_dir, f"PfPR_ClinicalIncidence_monthly_{self.step}.csv"))


################################################################################
# ANALYZER 3: Event recorder (births)
################################################################################

class BirthEventAnalyzer(IAnalyzer):
    """
    Yearly counts of recorded events (Pregnant, GaveBirth) per simulation,
    from ReportEventRecorder.csv. Raw rows are never kept: the file is large,
    so each simulation is reduced to counts inside map().
    """

    def __init__(self, expt_name, sweep_variables=None, events=None, working_dir=".",
                 start_year=BURNIN_START_YEAR, step="burnin"):
        super().__init__(working_dir=working_dir,
                         filenames=["output/ReportEventRecorder.csv"])
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.events = events or EVENT_LIST
        self.expt_name = expt_name
        self.start_year = start_year
        self.step = step

    def map(self, data, simulation: Simulation):
        ev = data[self.filenames[0]]
        ev.columns = [c.strip() for c in ev.columns]

        ev = ev.loc[ev["Event_Name"].isin(self.events), ["Time", "Event_Name"]].copy()
        if ev.empty:
            return pd.DataFrame()

        ev["Year"] = self.start_year + ev["Time"].astype(int) // DAYS_PER_YEAR
        counts = ev.groupby(["Year", "Event_Name"]).size().reset_index(name="Count")

        return add_sweep_tags(counts, simulation, self.sweep_variables)

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("\nWarning: No event data have been returned... Exiting...")
            return

        out_dir = os.path.join(self.working_dir, self.expt_name)
        os.makedirs(out_dir, exist_ok=True)
        stream_to_csv(all_data, os.path.join(out_dir, f"Event_counts_yearly_{self.step}.csv"))


################################################################################
# PLOTS AND CHECKS (read back the combined CSVs)
################################################################################

def plot_inset_chart(wdir, expt_name, step, sweep_variables, channels):
    """One CSV and one figure per sweep variable (marginal effect, others averaged)."""
    df = pd.read_csv(os.path.join(wdir, expt_name, f"All_Age_InsetChart_{step}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = coerce_numeric(df, sweep_variables)

    print("\n========== INSETCHART CHECK ==========")
    print("Rows:", len(df))
    print("Simulations:", df["sim_id"].nunique())
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("======================================\n")

    n_channels = len(channels)
    ncols = 2
    nrows = int(np.ceil(n_channels / ncols))

    for sweep_var in sweep_variables:
        if sweep_var not in df.columns:
            continue

        summary = df.groupby(["date", sweep_var])[channels].mean().reset_index()
        csv_path = os.path.join(wdir, expt_name, f"InsetChart_summary_by_{sweep_var}_{step}.csv")
        summary.to_csv(csv_path, index=False)
        print(f"Wrote {len(summary)} rows to {csv_path}")

        fig = plt.figure(figsize=(14, 8))
        fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.08, right=0.88, bottom=0.12)
        fig.suptitle(f"InsetChart (monthly mean) -- grouped by {sweep_var}", fontsize=16)
        axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n_channels)]

        for ax, channel in zip(axes, channels):
            for value, pdf in summary.groupby(sweep_var):
                ax.plot(pdf["date"], pdf[channel], linewidth=1.0, label=str(value))
            ax.set_title(channel, fontsize=14)
            ax.set_ylabel(channel, fontsize=11)
            ax.tick_params(axis="x", rotation=90)
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[-1].legend(title=sweep_var, fontsize=8, loc="center left",
                        bbox_to_anchor=(1.02, 0.5))

        png_path = os.path.join(wdir, expt_name, f"InsetChart_by_{sweep_var}_{step}.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote chart to {png_path}")


def plot_pfpr(wdir, expt_name, step, sweep_variables):
    """PfPR and clinical incidence for the prevalence-survey age bin, per sweep variable."""
    path = os.path.join(wdir, expt_name, f"PfPR_ClinicalIncidence_monthly_{step}.csv")
    df = pd.read_csv(path)
    df = coerce_numeric(df, sweep_variables)

    print("\n========== SUMMARY REPORT CHECK ==========")
    print("Rows:", len(df))
    print("Simulations:", df["sim_id"].nunique())
    print("Years:", int(df["Year"].min()), "to", int(df["Year"].max()))
    print("Age bins:", sorted(df["agebin"].unique(), key=lambda s: float(s.split('-')[1])))
    print("Periods kept per report:", sorted(df["period"].unique()))
    print("==========================================\n")

    sub = df[df["agebin_upper"] == PREVALENCE_BIN_UPPER]
    if sub.empty:
        print(f"No rows for age bin with upper edge {PREVALENCE_BIN_UPPER}; skipping PfPR plots.")
        return

    for sweep_var in sweep_variables:
        if sweep_var not in sub.columns:
            continue

        summary = sub.groupby(["year_frac", sweep_var])[["PfPR", "Cases"]].mean().reset_index()
        csv_path = os.path.join(wdir, expt_name, f"PfPR_summary_by_{sweep_var}_{step}.csv")
        summary.to_csv(csv_path, index=False)
        print(f"Wrote {len(summary)} rows to {csv_path}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.subplots_adjust(left=0.07, right=0.85, wspace=0.25)
        fig.suptitle(f"6-59 months (age bin up to {PREVALENCE_BIN_UPPER} y) -- grouped by {sweep_var}",
                     fontsize=14)

        for ax, channel in zip(axes, ["PfPR", "Cases"]):
            for value, pdf in summary.groupby(sweep_var):
                ax.plot(pdf["year_frac"], pdf[channel], linewidth=1.0, label=str(value))
            ax.set_title(channel)
            ax.set_xlabel("Year")
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[-1].legend(title=sweep_var, fontsize=8, loc="center left",
                        bbox_to_anchor=(1.02, 0.5))

        png_path = os.path.join(wdir, expt_name, f"PfPR_by_{sweep_var}_{step}.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote chart to {png_path}")


def birth_rate_check(wdir, expt_name, step, sweep_variables):
    """
    Births per 1,000 population per year, from GaveBirth counts and the mean
    InsetChart population. Written to CSV and the last 15 years are printed.
    """
    ev_path = os.path.join(wdir, expt_name, f"Event_counts_yearly_{step}.csv")
    pop_path = os.path.join(wdir, expt_name, f"All_Age_InsetChart_{step}.csv")
    if not (os.path.exists(ev_path) and os.path.exists(pop_path)):
        print("Birth check skipped: event or InsetChart CSV is missing.")
        return

    ev = pd.read_csv(ev_path)
    births = ev[ev["Event_Name"] == "GaveBirth"][["sim_id", "Year", "Count"]]
    births = births.rename(columns={"Count": "GaveBirth"})

    pop = pd.read_csv(pop_path)
    pop = (pop.groupby(["sim_id", "Year"])["Statistical Population"].mean()
           .reset_index().rename(columns={"Statistical Population": "mean_pop"}))

    merged = births.merge(pop, on=["sim_id", "Year"], how="inner")
    merged["births_per_1000"] = merged["GaveBirth"] / merged["mean_pop"] * 1000

    by_year = merged.groupby("Year")[["GaveBirth", "mean_pop", "births_per_1000"]].mean().reset_index()
    out_path = os.path.join(wdir, expt_name, f"Birth_rate_check_{step}.csv")
    by_year.to_csv(out_path, index=False)

    print("\n========== BIRTH-RATE CHECK (mean over simulations) ==========")
    print(by_year.tail(15).to_string(index=False))
    print(f"\nWrote {out_path}")
    print("Compare births_per_1000 with the crude death rate used in the burn-in.")
    print("==============================================================\n")


################################################################################
# MAIN
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the UWR burn-in experiment.")
    parser.add_argument("--test-birth", action="store_true",
                        help="Analyze a --test-birth experiment (InsetChart only).")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip reading simulations; plot from CSVs already written.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    expt_name, exp_id = list(EXPTS.items())[0]
    if exp_id.startswith("PASTE"):
        raise SystemExit("Set the experiment ID in EXPTS before running.")

    jdir = manifest.job_directory
    wdir = os.path.join(jdir, "outputs")
    os.makedirs(wdir, exist_ok=True)

    if args.test_birth:
        step = "burnin_birthtest"
        sweep_variables = SWEEP_VARIABLES_TEST
    else:
        step = "burnin"
        sweep_variables = SWEEP_VARIABLES_FULL

    if not args.plots_only:
        analyzers = [
            InsetChartAnalyzer(expt_name=expt_name, channels=INSET_CHANNELS,
                               start_year=BURNIN_START_YEAR,
                               sweep_variables=sweep_variables,
                               step=step, working_dir=wdir),
        ]

        if not args.test_birth:
            analyzers += [
                MonthlyPfPRAnalyzer(expt_name=expt_name,
                                    start_year=BURNIN_START_YEAR,
                                    sweep_variables=sweep_variables,
                                    step=step, working_dir=wdir),
                BirthEventAnalyzer(expt_name=expt_name,
                                   start_year=BURNIN_START_YEAR,
                                   sweep_variables=sweep_variables,
                                   step=step, working_dir=wdir),
            ]

        with Platform("SLURM_LOCAL", job_directory=jdir) as platform:
            am = AnalyzeManager(
                configuration={},
                ids=[(exp_id, ItemType.EXPERIMENT)],
                analyzers=analyzers,
                partial_analyze_ok=True,
            )
            am.analyze()

    plot_inset_chart(wdir, expt_name, step, sweep_variables, INSET_CHANNELS)

    if not args.test_birth:
        plot_pfpr(wdir, expt_name, step, sweep_variables)
        birth_rate_check(wdir, expt_name, step, sweep_variables)