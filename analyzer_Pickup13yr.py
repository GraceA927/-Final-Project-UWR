"""
UWR Pickup Analyzer, 2013-2025.

Analyzes an existing EMOD pickup experiment produced by run_UWR_burnin_pickup.py
(--step pickup). It does not configure interventions or change the serialized state.

Calendar handling: the EMOD simulation, climate forcing and intervention schedules use
a flat 365-day model calendar (leap days dropped). The analyzer keeps model-year /
model-month labels and does NOT assign Gregorian dates to model days.
MalariaSummaryReport intervals are 30-day reporting intervals, not calendar months, and
each report-year file has 12 full intervals plus a 5-day tail (13 x 30 > 365).

Climate: real weather exists for 2013-2023 only. 2024 repeats 2023 and 2025 repeats
2024, so every output frame carries `climate_filled` (True for Year > 2023) and the
PfPR plot shades those years. Do not read 2024-2025 climate-driven variation as real.

Sweep tags written by the run script (pickup):
  x_Temporary_Larval_Habitat, Run_Number, scale_constant, scale_temp_rain,
  scale_water_veg, burnin_sim_id, and irs_kill_scale / irs_decay_scale /
  irs_coverage_scale only when IRS_SWEEP is non-empty (optional here, no warning).

Outputs:
  All_Age_InsetChart_pickup.csv
  PfPR_ClinicalIncidence_monthly_pickup.csv
  EventCounts_pickup.csv                       (with Event_Group column)
  InsetChart_pickup_<habitat>.png              (one per habitat mix; lines = xTLH, seed-mean)
  EventCounts_pickup_byYear.png                (two panels: epi/demography, interventions)
  PfPR_pickup_byAgeBin.png                     (complete 30-day intervals only)
"""

import io
import os
import re
import shutil
from abc import abstractmethod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation

import manifest

# Single source of truth: calendar, report and sweep settings come from the run script,
# so the analyzer cannot drift from what the simulations were built with.
import Site_Setup_Final as SETUP

PICKUP_START_YEAR = SETUP.PICKUP_START_YEAR
PICKUP_END_YEAR = SETUP.PICKUP_END_YEAR
LAST_REAL_CLIMATE_YEAR = min(SETUP.filled_calendar_years("pickup"), default=PICKUP_END_YEAR + 1) - 1
DAYS_PER_YEAR = SETUP.DAYS_PER_YEAR
PICKUP_DURATION_DAYS = SETUP.PICKUP_DAYS
REPORTING_INTERVAL_DAYS = SETUP.REPORT_INTERVAL
REPORT_AGE_BINS = [float(a) for a in SETUP.REPORT_AGE_BINS]
PREGNANCY_AGE_BINS = list(SETUP.PREGNANCY_AGE_BINS)
STEP = "pickup"
OUT_SUFFIX = "_Output"

# ReportEventRecorder "Time" -> zero-based model day offset. Leave 0 if Time runs
# 0..4744; set to 1 if the first recorded day is Time = 1 (then Time 365 is still 2013).
EVENT_TIME_OFFSET = 0

# Flat 365-day EMOD calendar: cumulative starts for Jan-Dec.
MONTH_STARTS = np.asarray(SETUP.MONTH_OFFSET)

# Key must equal SETUP.EXPERIMENT_NAME_PICKUP (checked in main()).
EXPERIMENTS = {
    "Pickup13yrs_Decay": "b14c3478-2022-450f-b156-7fea5b6a19ac",   # habitat-decay sweep (temp 0.05/0.15/0.4 x semi 0.01/0.03/0.1) on burn-in 8694a259
    # "Pickup13yrs_Season": "6ce151ce-66dd-4ba5-9a71-805d592bd343",  # seasonality habitat grid (burn-in 8694a259), xTLH 37.4
    # "Pickup13yrs_IRSadj": "a06a970d-6f81-454b-8507-fbedb236e259",  # burn-in CM ramp + adjusted IRS (2016/2022 weaker, 2024-26 stronger)
    # "Pickup13yrs_IRSfinal": "8061abba-ba56-4db6-8dc5-6382aea5dc0b",  # final: sweep result in CSV (kill x0.4), 2024-25 cov 0.90
}

TAG_ALIASES = {
    "Run_Number": ["Run_Number"],
    "x_Temporary_Larval_Habitat": ["x_Temporary_Larval_Habitat", "xTLH"],
    **{t: [t] for t in SETUP.HABITAT_TAGS},          # habitat tags of the current HABITAT_MODE
}
SWEEP_VARIABLES = [
    "Run_Number",
    "x_Temporary_Larval_Habitat",
    *SETUP.HABITAT_TAGS,
]
HABITAT_COLS = list(SETUP.HABITAT_TAGS)
XTLH_COL = "x_Temporary_Larval_Habitat"
# Written only if present on the simulation (no NaN column, no warning otherwise).
OPTIONAL_TAGS = ["burnin_sim_id", "irs_kill_scale", "irs_decay_scale", "irs_coverage_scale",
                 *SETUP.PICKUP_CONFIG_SWEEP]   # habitat-decay sweep tags

# Everything written to the CSV (channels missing from InsetChart are skipped).
INSET_CHANNELS = [
    "Statistical Population",
    "New Clinical Cases",
    "Blood Smear Parasite Prevalence",
    "Infectious Vectors",
    "Adult Vectors",
    "Daily EIR",
    "Births",
    "Air Temperature",
    "Rainfall",
    "Relative Humidity",
]
INSET_PLOT_CHANNELS = [
    "Statistical Population",
    "New Clinical Cases (monthly total)",
    "Blood Smear Parasite Prevalence",
    "Infectious Vectors",
    "Adult Vectors",
    "Daily EIR",
]
# Flow channels: also written as a monthly SUM (mean of a daily count is hard to read).
SUM_CHANNELS = ["New Clinical Cases", "Births"]

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
# Events broadcast by the pickup interventions (Received_SMC/IRS/IPTp_D*/Vaccine_D*,
# ANC/CWC/IPTp visits, Bednet_Got_New_One); everything else is epi/demography.
INTERVENTION_EVENT_RE = re.compile(r"^(Received_(?!Treatment|Severe_Treatment)|ANC_Visit|CWC_Visit|IPTp_Visit|Bednet_)")
#INTERVENTION_EVENT_RE = re.compile(r"^(Received_(?!Treatment|Severe_Treatment)"r"|ANC_Visit"r"|CWC_Visit"r"|IPTp_Visit"
   # r"|IPTp_D" r"|Vaccine_D" r"|Bednet_)")

def output_dir(work_dir, expt_name):
    return os.path.join(work_dir, f"{expt_name}{OUT_SUFFIX}")


def flat_calendar(n_days, start_year):
    """Map daily rows to the EMOD flat 365-day model calendar.

    Model month lengths are fixed at [31, 28, 31, 30, ...]. This function
    intentionally does not create Gregorian dates or leap days.
    """
    if n_days < 0:
        raise ValueError("n_days must be non-negative")
    day = np.arange(n_days)
    year = int(start_year) + day // DAYS_PER_YEAR
    month = np.searchsorted(MONTH_STARTS, day % DAYS_PER_YEAR, side="right")
    return year, month


def model_day_time(model_day, start_year=PICKUP_START_YEAR):
    """Return continuous model time for a zero-based day index."""
    return int(start_year) + float(model_day) / DAYS_PER_YEAR


def model_day_year_month(model_day, start_year=PICKUP_START_YEAR):
    """Map a model-day offset to flat-calendar year/month (month is 1-12)."""
    day = int(np.floor(float(model_day)))
    year = int(start_year) + day // DAYS_PER_YEAR
    day_in_year = day % DAYS_PER_YEAR
    month = int(np.searchsorted(MONTH_STARTS, day_in_year, side="right"))
    return year, month


def event_group(name):
    return "Intervention" if INTERVENTION_EVENT_RE.match(str(name)) else "Epi/Demography"


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

        for column in OPTIONAL_TAGS:             # only present for some run configurations
            if column in tags and tags[column] is not None:
                df[column] = tags[column]

        if "Year" in df.columns:                 # 2024-2025 climate is a repeat, not data
            df["climate_filled"] = pd.to_numeric(df["Year"], errors="coerce") > LAST_REAL_CLIMATE_YEAR

        for key, value in self.extra_columns.items():
            df[key] = value
        df["sim_id"] = str(simulation.uid)


class InsetChartAnalyzer(PickupAnalyzer):
    """Aggregate daily InsetChart channels to model-month means (plus totals for flows)."""

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
        total_columns = [f"{c} (monthly total)" for c in SUM_CHANNELS if c in available]
        daily["_day_present"] = 1
        agg = {c: (c, "mean") for c in value_columns}
        agg.update({f"{c} (monthly total)": (c, "sum") for c in SUM_CHANNELS if c in available})
        monthly = daily.groupby(["Year", "Month"], as_index=False).agg(
            **agg, days_simulated=("_day_present", "sum"),
        )
        monthly["model_month"] = (
            monthly["Year"].astype(int).astype(str) + "-" +
            monthly["Month"].astype(int).astype(str).str.zfill(2)
        )
        # Midpoint of the fixed model month, expressed in model-year units.
        monthly["model_time"] = monthly["Year"] + (monthly["Month"] - 0.5) / 12.0
        return monthly[["model_month", "model_time", "Year", "Month", "days_simulated"]
                       + value_columns + total_columns]


class MonthlyPfPRAnalyzer(PickupAnalyzer):
    """
    Extract MalariaSummaryReport time-by-age data.

    The run script writes one report per year (suffix Monthly_U5_<year>), 30-day
    interval, max 13 reports -> 12 full intervals + a 5-day tail. `report_interval`
    is NOT a calendar month. Interval boundaries come from the report's own
    'Time Of Report', and `is_complete_interval` flags the tail so it can be excluded
    from plots without dropping data from the CSV.
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

            # Use the report's actual end times (cumulative model-day endpoints).
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

                # end_day is an exclusive boundary; last simulated day is end_day - 1.
                midpoint_day = start_day + (actual_days - 1) / 2.0
                model_year, model_month = model_day_year_month(
                    midpoint_day, self.start_year
                )

                for age_idx, age_bin in enumerate(age_bins):
                    rec = {
                        "Year": int(report_year),
                        # Calibration-ready model-calendar labels. These are
                        # assigned from the 30-day interval midpoint; they are
                        # approximate month labels, not true calendar-month means.
                        "year": model_year,
                        "month": model_month,
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
        # ANC_Visit_2015 -> ANC_Visit, CWC_Visit_2015 -> CWC_Visit, IPTp_Visit_2015 -> IPTp_Visit
        edf["Event_Name"] = edf["Event_Name"].astype(str).str.replace(
            EVENT_YEAR_SUFFIX, "", regex=True
        )
        if "Time" in edf.columns:
            # Model year from model day (flat 365-day calendar), clipped to the pickup window.
            event_time = pd.to_numeric(edf["Time"], errors="coerce")
            year = self.start_year + np.floor((event_time - EVENT_TIME_OFFSET) / DAYS_PER_YEAR)
            edf["Year"] = year.clip(self.start_year, PICKUP_END_YEAR)
        else:
            edf["Year"] = np.nan
        out = edf.groupby(["Year", "Event_Name"], dropna=False).size().reset_index(name="Count")
        out["Event_Group"] = out["Event_Name"].map(event_group)
        return out

    def reduce(self, all_data):
        super().reduce(all_data)
        if os.path.isfile(self.out_path):
            totals = pd.read_csv(self.out_path).groupby("Event_Name")["Count"].sum().sort_index()
            print("\nRecorded event totals (summed over all simulations):")
            print(totals.to_string())


def _fmt_group(g):
    try:
        return f"{float(g):.1f}"
    except (TypeError, ValueError):
        return str(g)


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
            ax.plot(gdf[x_col], gdf[channel], linewidth=1.1, label=_fmt_group(group))
        ax.axvspan(LAST_REAL_CLIMATE_YEAR + 1, PICKUP_END_YEAR + 1, color="grey", alpha=0.12)
        ax.set_title(channel)
        ax.set_xlabel("Model year (365-day model calendar)")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(title=group_col, fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_inset_by_habitat(df, channels, expt_name, out_dir):
    """Seed-mean per xTLH; one figure per habitat mix (avoids averaging across the grid)."""
    channels = [c for c in channels if c in df.columns]
    if not channels or XTLH_COL not in df.columns:
        return
    hab = [c for c in HABITAT_COLS if c in df.columns]
    seed_mean = df.groupby(hab + [XTLH_COL, "model_time"], as_index=False)[channels].mean()
    groups = seed_mean.groupby(hab) if hab else [((), seed_mean)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        tag = "_".join(f"{c.replace('scale_', '').replace('spline_', '')}{float(v):g}"
                       for c, v in zip(hab, key)) or "all"
        plot_channels_by_group(
            sub, XTLH_COL, channels, "model_time",
            f"{expt_name}: monthly InsetChart, seed mean ({tag}); grey = repeated-year climate",
            os.path.join(out_dir, f"InsetChart_{STEP}_{tag}.png"),
        )


def plot_event_counts(event_df, path, title):
    if event_df.empty:
        return
    if "Event_Group" not in event_df.columns:
        event_df = event_df.assign(Event_Group=event_df["Event_Name"].map(event_group))
    n_sims = max(event_df["sim_id"].nunique(), 1)
    totals = event_df.groupby(["Event_Group", "Year", "Event_Name"], as_index=False)["Count"].sum()
    totals["Mean_per_sim"] = totals["Count"] / n_sims
    groups = ["Epi/Demography", "Intervention"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 12), sharex=True)
    for ax, grp in zip(axes, groups):
        sub = totals[totals["Event_Group"] == grp]
        for name, group in sub.groupby("Event_Name"):
            group = group.sort_values("Year")
            ax.plot(group["Year"], group["Mean_per_sim"], marker="o", label=name)
        ax.set_title(grp)
        ax.set_ylabel("Mean recorded events per simulation")
        ax.grid(True, linestyle="--", alpha=0.3)
        if grp == "Epi/Demography":
            ax.set_yscale("log")
        if not sub.empty:
            ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Pickup year")
    fig.suptitle(title)
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
    if "is_complete_interval" in df.columns:      # drop the 5-day tail of each report year
        df = df[df["is_complete_interval"].astype(bool)]
    df["interval_midpoint_model_time"] = pd.to_numeric(
        df["interval_midpoint_model_time"], errors="coerce"
    )
    df = df.dropna(subset=["interval_midpoint_model_time", "agebin"])
    age_bins = sorted(df["agebin"].dropna().unique())
    if not age_bins:
        print("No age bins available to plot.")
        return
    lowers = [0.0] + list(age_bins[:-1])

    ncols = 2
    nrows = int(np.ceil(len(age_bins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(age_bins):]:
        ax.set_visible(False)

    for ax, lo, age_bin in zip(axes, lowers, age_bins):
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
        ax.axvspan(LAST_REAL_CLIMATE_YEAR + 1, PICKUP_END_YEAR + 1, color="grey", alpha=0.12)
        ax.set_title(f"Age {lo:g}-{age_bin:g} years")
        ax.set_xlabel("Model time (365-day calendar)")
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
        if "model_time" in df.columns:
            plot_inset_by_habitat(df, INSET_PLOT_CHANNELS, expt_name, out_dir)

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
            f"{expt_name}: PfPR by age bin, ensemble median and 10-90% "
            f"({PICKUP_START_YEAR}-{PICKUP_END_YEAR}; grey = repeated-year climate)"
        )
        expected = [
            "year", "month", "x_Temporary_Larval_Habitat", "Run_Number",
            *SETUP.HABITAT_TAGS
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