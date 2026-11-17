"""
UWR Pickup Analyzer, 2013–2025.

Analyzes an existing EMOD pickup experiment. It does not configure interventions
or change the serialized pickup state.

Calendar handling: the EMOD simulation, climate forcing and intervention
schedules all use a flat 365-day model calendar (run_UWR_burnin_pickup.py:
DAYS_PER_YEAR = 365, MONTH_OFFSET = 0,31,59,90,...,334). The analyzer therefore
keeps model-year/model-month labels and does NOT assign Gregorian dates to
model days. This matches the run script because the pickup climate files are
built with 29 Feb dropped (DROP_LEAP_DAYS = True): model day-of-year 60 is
1 March in every year, so model months line up with Gregorian months.

Pickup climate (from the run script): real weather 2013-2023, then 2024 repeats
2023 and 2025 repeats 2024. Model years 2024 and 2025 therefore carry no new
weather information; interpret them as the 2023 climate with updated
interventions.

MalariaSummaryReport intervals are 30-day reporting intervals, not calendar
months. Each interval is given a model_year / model_month from its midpoint
(taken from the report's own Time Of Report), so it can be joined to the monthly
InsetChart output. A short tail interval can share a month with the previous
interval; use is_complete_interval or interval_days when merging.

Day indexing: InsetChart rows are zero-based (row 0 = model day 1 = 1 Jan 2013).

Outputs:
  All_Age_InsetChart_pickup.csv
  PfPR_ClinicalIncidence_monthly_pickup.csv
  EventCounts_pickup.csv
  InsetChart_pickup_byRunNumber.png
  EventCounts_pickup_byYear.png
  PfPR_pickup_byAgeBin.png
"""

import io
import os
import re
import shutil
from abc import abstractmethod

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
PICKUP_DURATION_DAYS = (PICKUP_END_YEAR - PICKUP_START_YEAR + 1) * DAYS_PER_YEAR
REPORTING_INTERVAL_DAYS = 30
STEP = "pickup"
OUT_SUFFIX = "_AnalyzerOutput"

# Flat 365-day EMOD calendar: cumulative starts for Jan–Dec.
# Must equal MONTH_OFFSET in run_UWR_burnin_pickup.py.
MONTH_STARTS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
MONTH_LENGTHS = np.diff(np.append(MONTH_STARTS, DAYS_PER_YEAR))   # 31,28,31,30,...


def _check_calendar_constants():
    """Fail early if the analyzer calendar drifts from the run script's."""
    if int(MONTH_LENGTHS.sum()) != DAYS_PER_YEAR or list(MONTH_LENGTHS) != [
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ]:
        raise ValueError(f"MONTH_STARTS is not a flat 365-day calendar: {MONTH_LENGTHS}")
    if PICKUP_DURATION_DAYS != 4745:
        raise ValueError(
            f"PICKUP_DURATION_DAYS {PICKUP_DURATION_DAYS} != 4745 (run script PICKUP_DAYS)"
        )


_check_calendar_constants()

EXPERIMENTS = {
    #"Pickup13yrs_IRSadj": "c5440c10-ad7a-4d91-8707-89c6221b93fb", # 2024 IRS cov0.90 kill0.40 decay240 (prev febcacc9, a06a970d)  #    5b51f593-9063-4545-a08e-9b55ff4fc556 final: sweep result in CSV (kill x0.4), 2024-25 cov 0.90
    "Pickup13yrs_SMCfix": "a34f3fc2-e5a2-4665-8908-597987ce8759",  # 2024 IRS cov0.90 kill0.40 decay240 (prev febcacc9, a06a970d)  #    5b51f593-9063-4545-a08e-9b55ff4fc556 final: sweep result in CSV (kill x0.4), 2024-25 cov 0.90

}

TAG_ALIASES = {
    "Run_Number": ["Run_Number"],
    "x_Temporary_Larval_Habitat": ["x_Temporary_Larval_Habitat", "xTLH"],
    "scale_constant": ["scale_constant"],
    "scale_temp_rain": ["scale_temp_rain"],
    "scale_water_veg": ["scale_water_veg"],
}
SWEEP_VARIABLES = [
    "Run_Number",
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
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


def output_dir(work_dir, expt_name):
    return os.path.join(work_dir, f"{expt_name}{OUT_SUFFIX}")


def flat_calendar(n_days, start_year):
    """Map daily rows to the EMOD flat 365-day model calendar.

    Row i (zero-based) is model day i + 1. Model month lengths are fixed at
    [31, 28, 31, 30, ...]. This function intentionally does not create
    Gregorian dates or leap days.
    """
    if n_days < 0:
        raise ValueError("n_days must be non-negative")
    day = np.arange(n_days)
    year = int(start_year) + day // DAYS_PER_YEAR
    month = np.searchsorted(MONTH_STARTS, day % DAYS_PER_YEAR, side="right")
    return year, month


def model_year_month(model_day, start_year=PICKUP_START_YEAR):
    """Model year and month (1-12) of a zero-based (possibly fractional) day index."""
    day = int(np.floor(model_day))
    year = int(start_year) + day // DAYS_PER_YEAR
    month = int(np.searchsorted(MONTH_STARTS, day % DAYS_PER_YEAR, side="right"))
    return year, month


def model_day_time(model_day, start_year=PICKUP_START_YEAR):
    """Return continuous model time for a zero-based day index."""
    return int(start_year) + float(model_day) / DAYS_PER_YEAR


class PickupAnalyzer(IAnalyzer):
    """Write per-simulation results separately, then concatenate them safely."""

    OUTPUT_PREFIX = ""
    TMP_DIRNAME = ""

    def __init__(
        self, filenames, expt_name, sweep_variables=None, working_dir=".",
        start_year=PICKUP_START_YEAR, step=STEP, extra_columns=None
    ):
        super().__init__(working_dir=working_dir, filenames=filenames)
        self.expt_name = expt_name
        self.sweep_variables = sweep_variables or SWEEP_VARIABLES
        self.start_year = start_year
        self.step = step
        self.extra_columns = extra_columns or {}
        self._missing_tag_warnings = set()

    @property
    def expt_dir(self):
        return output_dir(self.working_dir, self.expt_name)

    @property
    def tmp_dir(self):
        return os.path.join(self.expt_dir, self.TMP_DIRNAME)

    @property
    def out_path(self):
        return os.path.join(self.expt_dir, f"{self.OUTPUT_PREFIX}_{self.step}.csv")

    @abstractmethod
    def build_frame(self, data, simulation):
        pass

    def map(self, data, simulation: Simulation):
        sim_id = str(simulation.uid)
        df = self.build_frame(data, simulation)
        if df is None or df.empty:
            return {"sim_id": sim_id, "n_rows": 0}

        self._add_metadata(df, simulation)
        os.makedirs(self.tmp_dir, exist_ok=True)
        path = os.path.join(self.tmp_dir, f"{sim_id}.csv")
        df.to_csv(path, index=False)
        return {"sim_id": sim_id, "path": path, "n_rows": len(df)}

    def reduce(self, all_data):
        results = [
            r for r in (all_data or {}).values()
            if r and r.get("n_rows") and r.get("path")
        ]
        if not results:
            print(f"No {self.OUTPUT_PREFIX} data returned.")
            return

        os.makedirs(self.expt_dir, exist_ok=True)
        with open(self.out_path, "w", newline="", encoding="utf-8") as dst:
            for i, result in enumerate(results):
                with open(result["path"], "r", encoding="utf-8") as src:
                    header = src.readline()
                    if i == 0:
                        dst.write(header)
                    shutil.copyfileobj(src, dst)

        print(f"Wrote {sum(r['n_rows'] for r in results)} rows from "
              f"{len(results)} simulations to {self.out_path}")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _add_metadata(self, df, simulation):
        tags = simulation.tags or {}
        for column in self.sweep_variables:
            candidates = TAG_ALIASES.get(column, [column])
            value = next(
                (tags[t] for t in candidates if t in tags and tags[t] is not None),
                None
            )
            if isinstance(value, (list, tuple)):
                value = "-".join(map(str, value))
            if value is None:
                # Do not silently assign all untagged simulations to Run 0.
                value = np.nan
                key = (column, tuple(candidates))
                if key not in self._missing_tag_warnings:
                    print(f"WARNING: no simulation tag for {column}; writing NaN.")
                    self._missing_tag_warnings.add(key)
            df[column] = value

        for key, value in self.extra_columns.items():
            df[key] = value
        df["sim_id"] = str(simulation.uid)


class InsetChartAnalyzer(PickupAnalyzer):
    """Aggregate daily InsetChart channels to model-month means.

    Channels are MEANS of the daily values (e.g. 'New Clinical Cases' is the
    mean daily count). Multiply by days_simulated for a monthly total.
    """

    OUTPUT_PREFIX = "All_Age_InsetChart"
    TMP_DIRNAME = "_per_sim_insetchart"

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

        daily = pd.DataFrame(available)
        expected_days = (PICKUP_END_YEAR - self.start_year + 1) * DAYS_PER_YEAR
        if len(daily) != expected_days:
            print(
                f"WARNING: InsetChart has {len(daily)} daily rows; expected "
                f"{expected_days} for {self.start_year}-{PICKUP_END_YEAR} "
            )

        daily["Year"], daily["Month"] = flat_calendar(len(daily), self.start_year)
        value_columns = list(available)
        daily["_day_present"] = 1
        monthly = daily.groupby(["Year", "Month"], as_index=False).agg(
            **{c: (c, "mean") for c in value_columns},
            days_simulated=("_day_present", "sum"),
        )

        # Calendar check: every complete model month must have its fixed length.
        expected_len = monthly["Month"].astype(int).map(lambda m: int(MONTH_LENGTHS[m - 1]))
        partial = monthly["days_simulated"] != expected_len
        if partial.any() and len(daily) == expected_days:
            print(f"WARNING: {int(partial.sum())} model months have an unexpected "
                  f"number of days (calendar mismatch).")

        monthly["model_month"] = (
            monthly["Year"].astype(int).astype(str) + "-" +
            monthly["Month"].astype(int).astype(str).str.zfill(2)
        )
        # Midpoint of the fixed model month, expressed in model-year units.
        monthly["model_time"] = monthly["Year"] + (monthly["Month"] - 0.5) / 12.0
        return monthly[["model_month", "model_time", "Year", "Month", "days_simulated"] + value_columns]


class MonthlyPfPRAnalyzer(PickupAnalyzer):
    """
    Extract MalariaSummaryReport time-by-age data.

    EMOD's reporting_interval is 30 days, so `report_interval` is deliberately
    not labelled as a calendar month. Each interval is however given a
    model_year / model_month taken from its midpoint (flat 365-day calendar) so
    it can be joined to the monthly InsetChart output. The final interval is excluded only when the
    report contains more intervals than the expected complete 30-day intervals
    for that report-year file; this avoids blindly dropping data.
    """

    OUTPUT_PREFIX = "PfPR_ClinicalIncidence_monthly"
    TMP_DIRNAME = "_per_sim_pfpr"

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

            # Use the report's actual end times. DataByTime['Time Of Report']
            # contains cumulative model-day endpoints (e.g., 2585,...,2920).
            # This correctly identifies the 5-day tail as a partial interval.
            by_time = report.get("DataByTime", {})
            raw_end_times = by_time.get("Time Of Report", [])
            if len(raw_end_times) >= n_available:
                end_days = [int(round(float(x))) for x in raw_end_times[:n_available]]
            else:
                start_day_fallback = int(metadata.get("Start_Day", (int(report_year) - self.start_year) * DAYS_PER_YEAR))
                end_days = [
                    start_day_fallback + min((i + 1) * interval_days_nominal, DAYS_PER_YEAR)
                    for i in range(n_available)
                ]
                print(
                    f"WARNING: {filename} lacks complete Time Of Report metadata; "
                    "using Start_Day and the nominal interval as a fallback."
                )

            metadata_start_day = int(
                metadata.get("Start_Day", (int(report_year) - self.start_year) * DAYS_PER_YEAR)
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

                # end_day is treated as an exclusive boundary. The final
                # simulated day in the interval is end_day - 1.
                midpoint_day = start_day + (actual_days - 1) / 2.0
                mid_year, mid_month = model_year_month(midpoint_day, self.start_year)

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
                        "model_year": mid_year,
                        "model_month": mid_month,
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
    """Count events by pickup (model) year and event name."""

    OUTPUT_PREFIX = "EventCounts"
    TMP_DIRNAME = "_per_sim_events"

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
        if edf is None or edf.empty or "Event_Name" not in edf.columns:
            return None
        edf["Event_Name"] = edf["Event_Name"].astype(str).str.replace(
            EVENT_YEAR_SUFFIX, "", regex=True
        )
        if "Time" in edf.columns:
            # Use the EMOD model year directly. Do not reinterpret this as a
            # Gregorian calendar year after leap years.
            event_time = pd.to_numeric(edf["Time"], errors="coerce")
            edf["Year"] = self.start_year + np.floor(event_time / DAYS_PER_YEAR)
        else:
            edf["Year"] = np.nan
        return edf.groupby(["Year", "Event_Name"], dropna=False).size().reset_index(name="Count")

    def reduce(self, all_data):
        super().reduce(all_data)
        if os.path.isfile(self.out_path):
            totals = pd.read_csv(self.out_path).groupby("Event_Name")["Count"].sum().sort_index()
            print("\nRecorded event totals:")
            print(totals.to_string())


def plot_channels_by_group(df, group_col, channels, x_col, title, path):
    channels = [c for c in channels if c in df.columns]
    if not channels:
        return
    ncols = 2
    nrows = int(np.ceil(len(channels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(channels):]:
        ax.set_visible(False)
    for ax, channel in zip(axes, channels):
        for group, gdf in df.groupby(group_col):
            gdf = gdf.sort_values(x_col)
            ax.plot(gdf[x_col], gdf[channel], linewidth=1.1, label=str(group))
        ax.set_title(channel)
        ax.grid(True, linestyle="--", alpha=0.3)
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", rotation=90)
        else:
            ax.set_xlabel("Model year (365-day model calendar)" if x_col == "model_time" else x_col)
    axes[0].legend(title=group_col, fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_event_counts(event_df, path, title):
    if event_df.empty:
        return
    n_sims = max(event_df["sim_id"].nunique(), 1)
    totals = event_df.groupby(["Year", "Event_Name"], as_index=False)["Count"].sum()
    totals["Mean_per_sim"] = totals["Count"] / n_sims
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, group in totals.groupby("Event_Name"):
        group = group.sort_values("Year")
        ax.plot(group["Year"], group["Mean_per_sim"], marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Pickup year")
    ax.set_ylabel("Mean recorded events per simulation")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
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
        ax.set_xlabel("Model time ")
        ax.set_ylabel("PfPR")
        ax.set_xlim(PICKUP_START_YEAR, PICKUP_END_YEAR + 1)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


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
        if "Run_Number" in df.columns and "model_time" in df.columns:
            channels = [c for c in INSET_CHANNELS if c in df.columns]
            by_run = df.groupby(["model_time", "Run_Number"], as_index=False)[channels].mean()
            plot_channels_by_group(
                by_run, "Run_Number", channels, "model_time",
                f"{expt_name}: monthly InsetChart summaries ({PICKUP_START_YEAR}–{PICKUP_END_YEAR})",
                os.path.join(out_dir, f"InsetChart_{STEP}_byRunNumber.png")
            )

    event_path = os.path.join(out_dir, f"EventCounts_{STEP}.csv")
    if os.path.isfile(event_path):
        plot_event_counts(
            pd.read_csv(event_path),
            os.path.join(out_dir, f"EventCounts_{STEP}_byYear.png"),
            f"{expt_name}: event counts by pickup year"
        )

    pfpr_path = os.path.join(out_dir, f"PfPR_ClinicalIncidence_monthly_{STEP}.csv")
    if os.path.isfile(pfpr_path):
        pfpr_df = pd.read_csv(pfpr_path)
        plot_pfpr_by_agebin(
            pfpr_df, os.path.join(out_dir, f"PfPR_{STEP}_byAgeBin.png"),
            f"{expt_name}: PfPR by age bin ({PICKUP_START_YEAR}–{PICKUP_END_YEAR})"
        )
        expected = [
            "x_Temporary_Larval_Habitat", "scale_constant",
            "scale_temp_rain", "scale_water_veg"
        ]
        missing = [c for c in expected if c not in pfpr_df.columns]
        if missing:
            print(f"WARNING: missing calibration columns: {missing}")
        else:
            print("Calibration tags found:")
            print(pfpr_df[expected].drop_duplicates().head(10).to_string(index=False))
        print(f"PfPR and incidence data: {pfpr_path}")


def main():
    job_dir = manifest.job_directory
    work_dir = job_dir
    os.makedirs(work_dir, exist_ok=True)
    for name, exp_id in EXPERIMENTS.items():
        if not exp_id or not exp_id.strip():
            raise SystemExit(f"Set a valid pickup experiment ID for {name}.")
        run_analysis(name, exp_id.strip(), job_dir, work_dir)
    for name in EXPERIMENTS:
        make_plots(name, work_dir)
    print(f"\nPickup analysis complete. Results: {work_dir}")


if __name__ == "__main__":
    main()