"""
UWR Pickup Analyzer, 2013-2025 -- for run_UWR_burnin_pickup.py (RTS,S / IRS version).

Analyzes an existing EMOD pickup experiment. It does not configure interventions
or change the serialized pickup state.

What the pickup run writes (and what this script reads)
  output/MalariaSummaryReport_Monthly_U5_<year>.json   one per year, 30-day intervals, age bins
  output/ReportEventRecorder.csv                        births, pregnancies, treatments and
                                                        every intervention event (Received_IRS,
                                                        Received_SMC, Received_IPTp_D<k>,
                                                        Received_Vaccine_D<k>, ANC/CWC visits ...)
  output/InsetChart.json                                ONLY if Enable_Default_Reporting is 1.
                                                        The pickup sets it to 0, so InsetChart is
                                                        optional here (see INSETCHART NOTE below).

Simulation tags written by the pickup: Run_Number, x_Temporary_Larval_Habitat,
scale_constant, scale_temp_rain, scale_water_veg, burnin_sim_id.

Changes relative to the earlier UWR analyzer
  1. InsetChart is optional; a clear message says why when it is absent.
  2. Event names with a year suffix (ANC_Visit_2017, IPTp_Step2_2024 ...) are merged.
  3. Event plot is grouped by intervention family (one panel each) instead of one crowded axis.
  4. New DoseRatios_pickup.csv: realized Received_Vaccine_D<k> / D<k-1> and
     Received_IPTp_D<k> / D<k-1> per year, for checking the implementation against the CSVs.
  5. Outputs go to <job_directory>/my_outputs (same convention as the burn-in analyzer).
  6. Command line: --exp-id, --name, --work-dir (no need to edit the file for each run).
  7. burnin_sim_id is carried as a column so each pickup run links to its burn-in state.

INSETCHART NOTE: to get daily InsetChart from the pickup, change in run_UWR_burnin_pickup.py
    config.parameters.Enable_Default_Reporting = 1 if test else 0
to
    config.parameters.Enable_Default_Reporting = 1
(adds one InsetChart.json per simulation; the burn-in-style daily plots then appear).

Usage
    python UWR_pickup_analyzer.py --exp-id <pickup experiment id>
    python UWR_pickup_analyzer.py                       (uses EXPERIMENTS below)

Outputs (in <work_dir>/<expt_name>_AnalyzerOutput/)
  All_Age_InsetChart_pickup.csv                (only if InsetChart exists)
  PfPR_ClinicalIncidence_monthly_pickup.csv
  EventCounts_pickup.csv
  DoseRatios_pickup.csv
  InsetChart_pickup_byRunNumber.png            (only if InsetChart exists)
  EventCounts_pickup_byYear.png
  PfPR_pickup_byAgeBin.png
"""

import argparse
import datetime
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation

import manifest

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
DAYS_PER_YEAR = 365
REPORTING_INTERVAL_DAYS = 30
STEP = "pickup"
OUT_SUFFIX = "_AnalyzerOutput"
WORK_SUBDIR = ""

# Offset added to the zero-based day-of-year when building the Gregorian date.
#   0  -> model day 0 = Jan 1 (correct)
#  -1  -> exactly reproduces the original script's timedelta(Day - 1),
#         which places model day 0 on Dec 31 of the previous year.
DATE_DAY_OFFSET = 0

# Flat 365-day EMOD calendar: cumulative starts for Jan-Dec.
MONTH_STARTS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])

EXPERIMENTS = {
    "Pickup13yrs_Insecticide": "c61bb366-d630-46d7-b32e-87dd5745a8ef", #### 4ab4ec74-b636-4b03-8877-b72d6084d1cf #  805fc826-ff02-4b3b-964c-53ed2e5793f3   #edit to pickup ID, or pass --exp-id
}

TAG_ALIASES = {
    "Run_Number": ["Run_Number"],
    "x_Temporary_Larval_Habitat": ["x_Temporary_Larval_Habitat", "xTLH"],
    "scale_constant": ["scale_constant"],
    "scale_temp_rain": ["scale_temp_rain"],
    "scale_water_veg": ["scale_water_veg"],
    "burnin_sim_id": ["burnin_sim_id"],
}
SWEEP_VARIABLES = [
    "Run_Number",
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
    "burnin_sim_id",
]
INSET_CHANNELS = [
    "Statistical Population",
    "New Clinical Cases",
    "Blood Smear Parasite Prevalence",
    "Infectious Vectors",
    "Adult Vectors",
    "Daily EIR",
]
PFPR_METRICS = {
    "PfPR": "PfPR by Age Bin",
    "Cases": "Annual Clinical Incidence by Age Bin",
    "Severe_cases": "Annual Severe Incidence by Age Bin",
    "New_infections": "New Infections by Age Bin",
    "Anemia": "Annual Mild Anemia by Age Bin",
    "Mean_density": "Mean Log Parasite Density by Age Bin",
    "Gametocyte_prevalence": "Pf Gametocyte Prevalence by Age Bin",
    "Pop": "Average Population by Age Bin",
}
EVENT_YEAR_SUFFIX = re.compile(r"_(?:19|20)\d{2}$")

# Intervention families for the event plot (matched on the year-stripped name).
EVENT_FAMILIES = [
    ("Demography", re.compile(r"^(Births|GaveBirth|Pregnant|NonDiseaseDeaths|DiseaseDeaths)$")),
    ("Infection / disease", re.compile(r"^(NewInfectionEvent|NewClinicalCase)$")),
    ("Case management", re.compile(r"^Received_(Severe_)?Treatment$")),
    ("ITN (ANC / CWC / mass)", re.compile(r"^(Bednet_Got_New_One|ANC_Visit|CWC_Visit|Received_ITN.*)$")),
    ("IRS and SMC", re.compile(r"^(Received_IRS|Received_SMC)$")),
    ("IPTp", re.compile(r"^(IPTp_Visit|IPTp_Step\d+|IPTp_Dose\d+|Received_IPTp_D\d+)$")),
    ("RTS,S vaccine", re.compile(r"^(Received_Vaccine_D\d+|Vaccine_D\d+)$")),
]
# Dose series checked in DoseRatios_pickup.csv: (label, regex with dose number group)
DOSE_SERIES = [
    ("Vaccine", re.compile(r"^Received_Vaccine_D(\d+)$")),
    ("IPTp", re.compile(r"^Received_IPTp_D(\d+)$")),
]


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #
def output_dir(work_dir, expt_name):
    return os.path.join(work_dir, f"{expt_name}{OUT_SUFFIX}")


def flat_calendar(n_days, start_year):
    """Map daily rows to the EMOD flat 365-day calendar: (year, month 1-12)."""
    if n_days < 0:
        raise ValueError("n_days must be non-negative")
    day = np.arange(n_days)
    year = int(start_year) + day // DAYS_PER_YEAR
    month = np.searchsorted(MONTH_STARTS, day % DAYS_PER_YEAR, side="right")
    return year, month


def model_day_to_date(year, day_of_year):
    """Gregorian date = date(Year, 1, 1) + timedelta(day_of_year + DATE_DAY_OFFSET)."""
    return [
        datetime.date(int(y), 1, 1) + datetime.timedelta(int(d) + DATE_DAY_OFFSET)
        for y, d in zip(year, day_of_year)
    ]


def model_day_time(model_day, start_year=PICKUP_START_YEAR):
    """Continuous model time (years) for a zero-based day index."""
    return int(start_year) + float(model_day) / DAYS_PER_YEAR


def event_family(name):
    for family, pattern in EVENT_FAMILIES:
        if pattern.match(str(name)):
            return family
    return "Other"


# --------------------------------------------------------------------------- #
# Analyzers
# --------------------------------------------------------------------------- #
class PickupAnalyzer(IAnalyzer):
    """Base class: map() builds a per-simulation frame, reduce() concatenates in memory."""

    OUTPUT_PREFIX = ""
    EMPTY_HINT = ""

    def __init__(
        self, filenames, expt_name, sweep_variables=None, working_dir=".",
        start_year=PICKUP_START_YEAR, step=STEP, extra_columns=None
    ):
        super().__init__(working_dir=working_dir, filenames=filenames)
        self.expt_name = expt_name
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.start_year = start_year
        self.step = step
        self.extra_columns = extra_columns or {}

    @property
    def expt_dir(self):
        return output_dir(self.working_dir, self.expt_name)

    @property
    def out_path(self):
        return os.path.join(self.expt_dir, f"{self.OUTPUT_PREFIX}_{self.step}.csv")

    def build_frame(self, data, simulation):
        raise NotImplementedError

    def map(self, data, simulation: Simulation):
        df = self.build_frame(data, simulation)
        if df is None or df.empty:
            return None
        self._add_metadata(df, simulation)
        return df

    def reduce(self, all_data):
        selected = [df for df in (all_data or {}).values() if df is not None and len(df)]
        if not selected:
            print(f"No {self.OUTPUT_PREFIX} data returned. {self.EMPTY_HINT}".rstrip())
            return

        os.makedirs(self.expt_dir, exist_ok=True)
        adf = pd.concat(selected).reset_index(drop=True)
        adf.to_csv(self.out_path, index=False)
        print(f"Wrote {len(adf)} rows from {len(selected)} simulations to {self.out_path}")

    def _add_metadata(self, df, simulation):
        """Original-script tag handling.

        - tag present            -> written to the frame
        - Run_Number missing     -> 0
        - any other tag missing  -> silently skipped
        """
        tags = simulation.tags or {}
        for column in self.sweep_variables:
            candidates = TAG_ALIASES.get(column, [column])
            key = next((t for t in candidates if t in tags), None)
            if key is not None:
                value = tags[key]
                if isinstance(value, (list, tuple)):
                    value = "-".join(str(x) for x in value)
                df[column] = value
            elif column == "Run_Number":
                df[column] = 0

        for key, value in self.extra_columns.items():
            df[key] = value
        df["sim_id"] = str(simulation.uid)


class InsetChartAnalyzer(PickupAnalyzer):
    """Daily InsetChart channels with Gregorian dates built from model days (optional)."""

    OUTPUT_PREFIX = "All_Age_InsetChart"
    EMPTY_HINT = ("The pickup sets Enable_Default_Reporting = 0, so no InsetChart.json is "
                  "written. This is expected unless you enabled it (see the INSETCHART NOTE "
                  "in this file's docstring).")

    def __init__(self, expt_name, channels=None, **kwargs):
        super().__init__(["output/InsetChart.json"], expt_name, **kwargs)
        self.channels = channels or INSET_CHANNELS

    def build_frame(self, data, simulation):
        inset = data.get(self.filenames[0])
        if not inset or "Channels" not in inset:
            return None
        available = {
            c: inset["Channels"][c]["Data"]
            for c in self.channels if c in inset["Channels"]
        }
        if not available:
            return None

        lengths = {len(v) for v in available.values()}
        if len(lengths) != 1:
            raise ValueError(f"InsetChart channels have unequal lengths: {lengths}")

        simdata = pd.DataFrame(available)
        expected_days = (PICKUP_END_YEAR - self.start_year + 1) * DAYS_PER_YEAR
        if len(simdata) != expected_days:
            print(
                f"WARNING: InsetChart has {len(simdata)} daily rows; expected "
                f"{expected_days} for {self.start_year}-{PICKUP_END_YEAR}"
            )

        simdata["Time"] = simdata.index
        simdata["Day"] = simdata["Time"] % DAYS_PER_YEAR
        year, month = flat_calendar(len(simdata), self.start_year)
        simdata["Year"] = year
        simdata["Month"] = month
        simdata["date"] = model_day_to_date(simdata["Year"], simdata["Day"])

        front = ["date", "Time", "Day", "Year", "Month"]
        return simdata[front + list(available)]


class MonthlyPfPRAnalyzer(PickupAnalyzer):
    """
    Extract MalariaSummaryReport time-by-age data.

    EMOD's reporting_interval is 30 days, so `report_interval` is deliberately
    not labelled as a calendar month. Interval end days come from the report's
    own 'Time Of Report', so partial tail intervals are flagged, not dropped.
    """

    OUTPUT_PREFIX = "PfPR_ClinicalIncidence_monthly"

    def __init__(
        self, expt_name, report_years,
        filename_template="output/MalariaSummaryReport_Monthly_U5_{year}.json",
        **kwargs
    ):
        self.year_by_filename = {
            filename_template.format(year=y): y for y in report_years
        }
        super().__init__(list(self.year_by_filename), expt_name, **kwargs)

    def build_frame(self, data, simulation):
        parts = []
        for filename, report_year in self.year_by_filename.items():
            report = data.get(filename)
            if not report:
                continue

            metadata = report.get("Metadata", {})
            age_bins = metadata.get("Age Bins", [])
            by_time_age = report.get("DataByTimeAndAgeBins", {})
            if not age_bins or not by_time_age:
                print(f"WARNING: missing age/time data in {filename}")
                continue

            interval_days_nominal = int(
                metadata.get("Reporting_Interval", REPORTING_INTERVAL_DAYS)
            )
            if interval_days_nominal <= 0:
                raise ValueError(
                    f"Invalid Reporting_Interval={interval_days_nominal} in {filename}"
                )

            metric_arrays = {
                output_name: by_time_age[source_name]
                for output_name, source_name in PFPR_METRICS.items()
                if source_name in by_time_age
            }
            if "PfPR" not in metric_arrays:
                print(f"WARNING: PfPR metric missing from {filename}")
                continue

            n_available = min(len(rows) for rows in metric_arrays.values())
            if n_available == 0:
                continue

            default_start = (int(report_year) - self.start_year) * DAYS_PER_YEAR
            metadata_start_day = int(metadata.get("Start_Day", default_start))

            by_time = report.get("DataByTime", {})
            raw_end_times = by_time.get("Time Of Report", [])
            if len(raw_end_times) >= n_available:
                end_days = [int(round(float(x))) for x in raw_end_times[:n_available]]
            else:
                end_days = [
                    metadata_start_day + min((i + 1) * interval_days_nominal, DAYS_PER_YEAR)
                    for i in range(n_available)
                ]
                print(
                    f"WARNING: {filename} lacks complete Time Of Report metadata; "
                    "using Start_Day and the nominal interval as a fallback."
                )

            records = []
            previous_end = metadata_start_day
            for interval_idx in range(n_available):
                end_day = end_days[interval_idx]
                start_day = previous_end
                actual_days = end_day - start_day
                previous_end = end_day

                if actual_days <= 0:
                    print(
                        f"WARNING: non-positive interval length ({actual_days}) in "
                        f"{filename}, interval {interval_idx + 1}; skipping."
                    )
                    continue

                midpoint_day = start_day + (actual_days - 1) / 2.0

                for age_idx, age_bin in enumerate(age_bins):
                    rec = {
                        "Year": int(report_year),
                        "report_interval": interval_idx + 1,
                        "interval_start_day": start_day,
                        "interval_end_day_exclusive": end_day,
                        "interval_days": actual_days,
                        "nominal_interval_days": interval_days_nominal,
                        "is_complete_interval": bool(actual_days == interval_days_nominal),
                        "interval_start_model_time": model_day_time(start_day, self.start_year),
                        "interval_end_model_time_exclusive": model_day_time(end_day, self.start_year),
                        "interval_midpoint_model_time": model_day_time(midpoint_day, self.start_year),
                        "agebin": float(age_bin),
                    }
                    for metric, rows in metric_arrays.items():
                        row = rows[interval_idx]
                        rec[metric] = row[age_idx] if age_idx < len(row) else np.nan
                    records.append(rec)

            if records:
                parts.append(pd.DataFrame.from_records(records))

        return pd.concat(parts, ignore_index=True) if parts else None


class EventRecorderAnalyzer(PickupAnalyzer):
    """Count events by pickup year and event name (year suffixes merged)."""

    OUTPUT_PREFIX = "EventCounts"
    EMPTY_HINT = "Check that output/ReportEventRecorder.csv exists in the simulations."

    def __init__(self, expt_name, filename="output/ReportEventRecorder.csv", **kwargs):
        super().__init__([filename], expt_name, **kwargs)

    @staticmethod
    def _to_dataframe(raw):
        if raw is None:
            return None
        if isinstance(raw, pd.DataFrame):
            return raw.copy()
        if isinstance(raw, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(raw))
        if isinstance(raw, str):
            return pd.read_csv(io.StringIO(raw))
        return None

    def build_frame(self, data, simulation):
        edf = self._to_dataframe(data.get(self.filenames[0]))
        if edf is None or edf.empty:
            return None
        edf.columns = [str(c).strip() for c in edf.columns]
        if "Event_Name" not in edf.columns:
            return None
        edf["Event_Name"] = edf["Event_Name"].astype(str).str.strip().str.replace(
            EVENT_YEAR_SUFFIX, "", regex=True
        )
        if "Time" in edf.columns:
            event_time = pd.to_numeric(edf["Time"], errors="coerce")
            edf["Year"] = self.start_year + np.floor(event_time / DAYS_PER_YEAR)
        else:
            edf["Year"] = np.nan
        return edf.groupby(["Year", "Event_Name"], dropna=False).size().reset_index(name="Count")

    def reduce(self, all_data):
        super().reduce(all_data)
        if os.path.isfile(self.out_path):
            totals = pd.read_csv(self.out_path).groupby("Event_Name")["Count"].sum().sort_index()
            print("\nRecorded event totals (all simulations):")
            print(totals.to_string())


# --------------------------------------------------------------------------- #
# Intervention checks
# --------------------------------------------------------------------------- #
def mean_events_per_year(event_df):
    """Mean recorded events per simulation: rows = Year, columns = Event_Name."""
    n_sims = max(event_df["sim_id"].nunique(), 1)
    totals = event_df.dropna(subset=["Year"]).groupby(["Year", "Event_Name"])["Count"].sum()
    return (totals / n_sims).unstack("Event_Name").fillna(0.0)


def dose_ratio_table(event_df):
    """Realized dose k / dose k-1 per year for the vaccine and IPTp series.
    Compare with the conditional uptake the script prints in --dry-run. Ratios are
    approximate: dose delays push some of a year's dose k-1 recipients into the next year."""
    piv = mean_events_per_year(event_df)
    rows = []
    for label, pattern in DOSE_SERIES:
        doses = {}
        for col in piv.columns:
            m = pattern.match(str(col))
            if m:
                doses[int(m.group(1))] = col
        for k in sorted(doses):
            if k == min(doses):
                continue
            prev = doses.get(k - 1)
            if prev is None:
                continue
            for year, row in piv.iterrows():
                denom = row[prev]
                if denom > 0:
                    rows.append({"series": label, "dose": f"D{k}/D{k - 1}", "Year": int(year),
                                 "numerator_mean_per_sim": row[doses[k]],
                                 "denominator_mean_per_sim": denom,
                                 "ratio": row[doses[k]] / denom})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_inset_chart(df, channels, path, title):
    """Single InsetChart figure: one subplot per channel, one line per Run_Number."""
    channels = [c for c in channels if c in df.columns]
    if not channels:
        print("No InsetChart channels available to plot.")
        return

    # Average over any rows sharing date + Run_Number (e.g. other sweep tags).
    by_run = df.groupby(["date", "Run_Number"], as_index=False)[channels].mean()

    ncols = 2
    nrows = int(np.ceil(len(channels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.08, right=0.88, bottom=0.12)
    axes = axes.ravel()
    for ax in axes[len(channels):]:
        ax.set_visible(False)

    for ax, channel in zip(axes, channels):
        for run, rdf in by_run.groupby("Run_Number"):
            rdf = rdf.sort_values("date")
            ax.plot(rdf["date"], rdf[channel], linewidth=1.0, label=f"Run {run}")
        ax.set_title(channel, fontsize=12)
        ax.set_ylabel(channel, fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=90)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].legend(
        title="Run_Number", fontsize=8,
        loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_event_counts(event_df, path, title):
    """Mean recorded events per simulation per year, one panel per intervention family."""
    if event_df.empty:
        return
    piv = mean_events_per_year(event_df)
    if piv.empty:
        return

    families = {}
    for name in piv.columns:
        families.setdefault(event_family(name), []).append(name)
    order = [f for f, _ in EVENT_FAMILIES if f in families] + (["Other"] if "Other" in families else [])

    ncols = 2
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(order):]:
        ax.set_visible(False)

    for ax, family in zip(axes, order):
        for name in sorted(families[family]):
            ax.plot(piv.index, piv[name], marker="o", markersize=3, linewidth=1.1, label=name)
        ax.set_title(family, fontsize=11)
        ax.set_xlabel("Pickup year")
        ax.set_ylabel("Mean events per simulation")
        ax.set_xlim(PICKUP_START_YEAR - 0.5, PICKUP_END_YEAR + 0.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pfpr_by_agebin(df, path, title):
    if df.empty or "PfPR" not in df.columns:
        return
    if "interval_midpoint_model_time" not in df.columns:
        print("Cannot plot PfPR: interval_midpoint_model_time is missing.")
        return

    df = df.copy()
    df["interval_midpoint_model_time"] = pd.to_numeric(
        df["interval_midpoint_model_time"], errors="coerce"
    )
    df = df.dropna(subset=["interval_midpoint_model_time", "agebin"])
    age_bins = sorted(df["agebin"].dropna().unique())
    if not age_bins:
        print("No age bins available to plot.")
        return

    ncols = 2
    nrows = int(np.ceil(len(age_bins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(age_bins):]:
        ax.set_visible(False)

    for ax, age_bin in zip(axes, age_bins):
        subset = df[df["agebin"] == age_bin]
        summary = (
            subset.groupby("interval_midpoint_model_time")["PfPR"]
            .agg(
                q10=lambda x: x.quantile(0.10),
                median="median",
                q90=lambda x: x.quantile(0.90),
            )
            .reset_index()
            .sort_values("interval_midpoint_model_time")
        )
        x = summary["interval_midpoint_model_time"].to_numpy()
        ax.fill_between(x, summary["q10"].to_numpy(), summary["q90"].to_numpy(), alpha=0.25)
        ax.plot(x, summary["median"].to_numpy(), linewidth=1.3)
        ax.set_title(f"Age bin up to {age_bin:g} years")
        ax.set_xlabel("Model time (365-day model calendar)")
        ax.set_ylabel("PfPR")
        ax.set_xlim(PICKUP_START_YEAR, PICKUP_END_YEAR + 1)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_analysis(expt_name, exp_id, job_dir, work_dir):
    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    common = dict(
        expt_name=expt_name,
        start_year=PICKUP_START_YEAR,
        sweep_variables=SWEEP_VARIABLES,
        step=STEP,
        working_dir=work_dir,
        extra_columns={"experiment": expt_name},
    )
    analyzers = [
        InsetChartAnalyzer(channels=INSET_CHANNELS, **common),
        MonthlyPfPRAnalyzer(
            report_years=range(PICKUP_START_YEAR, PICKUP_END_YEAR + 1), **common
        ),
        EventRecorderAnalyzer(**common),
    ]
    print(f"\nAnalyzing {expt_name}; experiment ID: {exp_id}")
    with Platform("SLURM_LOCAL", job_directory=job_dir):
        AnalyzeManager(
            configuration={}, ids=[(exp_id, ItemType.EXPERIMENT)],
            analyzers=analyzers, partial_analyze_ok=True
        ).analyze()


def make_plots(expt_name, work_dir):
    out_dir = output_dir(work_dir, expt_name)

    inset_path = os.path.join(out_dir, f"All_Age_InsetChart_{STEP}.csv")
    if os.path.isfile(inset_path):
        df = pd.read_csv(inset_path)
        df["date"] = pd.to_datetime(df["date"])

        print("\n========== INSETCHART CSV CHECK ==========")
        print("Rows:", len(df))
        print("Minimum date:", df["date"].min())
        print("Maximum date:", df["date"].max())
        print("Years:", df["date"].dt.year.min(), "to", df["date"].dt.year.max())
        print("==========================================\n")

        if "Run_Number" in df.columns:
            plot_inset_chart(
                df, INSET_CHANNELS,
                os.path.join(out_dir, f"InsetChart_{STEP}_byRunNumber.png"),
                f"{expt_name}: InsetChart ({PICKUP_START_YEAR}-{PICKUP_END_YEAR})"
            )
    else:
        print("No InsetChart CSV (expected when Enable_Default_Reporting is 0); "
              "daily InsetChart plot skipped.")

    event_path = os.path.join(out_dir, f"EventCounts_{STEP}.csv")
    if os.path.isfile(event_path):
        event_df = pd.read_csv(event_path)
        plot_event_counts(
            event_df,
            os.path.join(out_dir, f"EventCounts_{STEP}_byYear.png"),
            f"{expt_name}: mean recorded events per simulation, by pickup year"
        )
        ratios = dose_ratio_table(event_df)
        if ratios.empty:
            print("No vaccine/IPTp dose events found for the dose-ratio check.")
        else:
            ratio_path = os.path.join(out_dir, f"DoseRatios_{STEP}.csv")
            ratios.to_csv(ratio_path, index=False)
            print("\nRealized dose ratios (compare with the conditional uptake from --dry-run):")
            print(ratios.round({"numerator_mean_per_sim": 1, "denominator_mean_per_sim": 1,
                                "ratio": 3}).to_string(index=False))
            print(f"Dose ratios: {ratio_path}")

    pfpr_path = os.path.join(out_dir, f"PfPR_ClinicalIncidence_monthly_{STEP}.csv")
    if os.path.isfile(pfpr_path):
        pfpr_df = pd.read_csv(pfpr_path)
        plot_pfpr_by_agebin(
            pfpr_df, os.path.join(out_dir, f"PfPR_{STEP}_byAgeBin.png"),
            f"{expt_name}: PfPR by age bin ({PICKUP_START_YEAR}-{PICKUP_END_YEAR})"
        )
        expected = [
            "x_Temporary_Larval_Habitat", "scale_constant",
            "scale_temp_rain", "scale_water_veg"
        ]
        missing = [c for c in expected if c not in pfpr_df.columns]
        if missing:
            print(f"Note: calibration columns not present (tags absent): {missing}")
        else:
            print("Calibration tags found:")
            print(pfpr_df[expected].drop_duplicates().head(10).to_string(index=False))
        print(f"PfPR and incidence data: {pfpr_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="UWR pickup analyzer")
    ap.add_argument("--exp-id", help="pickup experiment ID (overrides EXPERIMENTS)")
    ap.add_argument("--name", default="Pickup13yrs_setup",
                    help="label for the output folder when --exp-id is used")
    ap.add_argument("--work-dir", default=None,
                    help=f"output root (default <job_directory>/{WORK_SUBDIR})")
    return ap.parse_args()


def main():
    args = parse_args()
    job_dir = manifest.job_directory
    work_dir = args.work_dir or os.path.join(job_dir, WORK_SUBDIR)
    os.makedirs(work_dir, exist_ok=True)

    experiments = {args.name: args.exp_id} if args.exp_id else EXPERIMENTS
    for name, exp_id in experiments.items():
        if not exp_id or not exp_id.strip():
            raise SystemExit(f"Set a valid pickup experiment ID for {name} "
                             f"(--exp-id or the EXPERIMENTS dict).")
        run_analysis(name, exp_id.strip(), job_dir, work_dir)
    for name in experiments:
        make_plots(name, work_dir)
    print(f"\nPickup analysis complete. Results: {work_dir}")


if __name__ == "__main__":
    main()