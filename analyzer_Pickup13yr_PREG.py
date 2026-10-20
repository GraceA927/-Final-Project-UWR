
"""
UWR Pickup Analyzer, 2013-2025.

Burn-in (1963-2012) -> pickup (2013-2025, 4,745 days).
Each pickup simulation inherits its serialized burn-in state and calibration
parameters. No independent pickup xTLH/habitat sweep is performed.

Outputs:
    All_Age_InsetChart_pickup.csv
    PfPR_ClinicalIncidence_monthly_pickup.csv
    EventCounts_pickup.csv

All outputs are written to:
    <job_directory>/<experiment name>_output/

The InsetChart plots remain grouped and labelled by Run_Number.
Calibration parameters are copied from simulation.tags into the CSV outputs.
Missing calibration tags are recorded as NaN, not replaced with invented values.
"""

import datetime
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
DAYS_PER_YEAR = 365
STEP = "pickup"
OUT_SUFFIX = "_output"

MONTH_OFFSET = np.cumsum(
    [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
)

EXPERIMENTS = {
    "Pickup_FOR_ECM13yrs_PREG": "30f189b6-59b4-4a6c-9422-7aa5d57978f6",
}

# Output column -> possible names used in simulation.tags.
# Run_Number remains the plotting group. The calibration parameters are
# inherited from burn-in; they are not independently swept during pickup.
TAG_ALIASES = {
    "Run_Number": ["Run_Number"],
    "x_Temporary_Larval_Habitat": [
        "x_Temporary_Larval_Habitat",
        "xTLH",
    ],
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


# -----------------------------------------------------------------------------
# Calendar helper
# -----------------------------------------------------------------------------

def flat_calendar(n_days, start_year):
    """Return model year and month for each day on a flat 365-day calendar."""
    day = np.arange(n_days)
    year = start_year + day // DAYS_PER_YEAR
    month = np.searchsorted(
        MONTH_OFFSET, day % DAYS_PER_YEAR, side="right"
    )
    return year, month


# -----------------------------------------------------------------------------
# Shared analyzer base
# -----------------------------------------------------------------------------

class PickupAnalyzer(IAnalyzer):
    """Build per-simulation DataFrames and combine them into one CSV."""

    OUTPUT_PREFIX = ""
    TMP_DIRNAME = ""

    def __init__(
        self,
        filenames,
        expt_name,
        sweep_variables=None,
        working_dir=".",
        start_year=PICKUP_START_YEAR,
        step=STEP,
        extra_columns=None,
    ):
        super().__init__(working_dir=working_dir, filenames=filenames)
        self.expt_name = expt_name
        self.sweep_variables = (
            sweep_variables if sweep_variables is not None
            else SWEEP_VARIABLES
        )
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
        return os.path.join(
            self.expt_dir, f"{self.OUTPUT_PREFIX}_{self.step}.csv"
        )

    @abstractmethod
    def build_frame(self, data, simulation):
        """Return a DataFrame for a simulation, or None if no data."""

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
            result
            for result in (all_data or {}).values()
            if result and result.get("n_rows")
        ]

        if not results:
            print(f"No {self.OUTPUT_PREFIX} data returned.")
            return

        os.makedirs(self.expt_dir, exist_ok=True)

        n_rows = 0
        with open(self.out_path, "w", newline="") as dst:
            for index, result in enumerate(results):
                with open(result["path"], "r") as src:
                    header = src.readline()
                    if index == 0:
                        dst.write(header)
                    shutil.copyfileobj(src, dst)
                n_rows += result["n_rows"]

        print(
            f"Wrote {n_rows} rows across {len(results)} simulations "
            f"to {self.out_path}"
        )
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _add_metadata(self, df, simulation):
        """Attach canonical metadata columns using actual simulation tags."""
        tags = simulation.tags or {}

        for column in self.sweep_variables:
            candidates = TAG_ALIASES.get(column, [column])
            found = False

            for tag in candidates:
                if tag in tags and tags[tag] is not None:
                    value = tags[tag]

                    if isinstance(value, (list, tuple)):
                        value = "-".join(str(item) for item in value)

                    df[column] = value
                    found = True
                    break

            if not found:
                # Preserve the previous Run_Number fallback. Calibration
                # parameters must never be fabricated if a tag is missing.
                if column == "Run_Number":
                    df[column] = 0
                else:
                    df[column] = np.nan

                warning_key = (column, tuple(candidates))
                if warning_key not in self._missing_tag_warnings:
                    print(
                        f"WARNING: Missing metadata tag for '{column}'. "
                        f"Checked {candidates}. Values will be NaN where "
                        f"the tag is unavailable."
                    )
                    self._missing_tag_warnings.add(warning_key)

        for key, value in self.extra_columns.items():
            df[key] = value

        df["sim_id"] = str(simulation.uid)


# -----------------------------------------------------------------------------
# 1. InsetChart
# -----------------------------------------------------------------------------

class InsetChartAnalyzer(PickupAnalyzer):
    """Aggregate daily InsetChart channels to monthly means."""

    OUTPUT_PREFIX = "All_Age_InsetChart"
    TMP_DIRNAME = "_per_sim_insetchart"

    def __init__(self, expt_name, channels=None, **kwargs):
        super().__init__(
            filenames=["output/InsetChart.json"],
            expt_name=expt_name,
            **kwargs,
        )
        self.channels = channels or INSET_CHANNELS

    def build_frame(self, data, simulation):
        inset = data.get(self.filenames[0])
        if not inset:
            return None

        available = {
            channel: inset["Channels"][channel]["Data"]
            for channel in self.channels
            if channel in inset.get("Channels", {})
        }

        if not available:
            return None

        daily = pd.DataFrame(available)
        daily["Year"], daily["Month"] = flat_calendar(
            len(daily), self.start_year
        )

        monthly = daily.groupby(
            ["Year", "Month"], as_index=False
        ).mean()

        monthly["date"] = pd.to_datetime([
            datetime.date(int(year), int(month), 1)
            for year, month in zip(monthly["Year"], monthly["Month"])
        ])

        return monthly[["date"] + list(available)]


# -----------------------------------------------------------------------------
# 2. Monthly PfPR / clinical incidence
# -----------------------------------------------------------------------------

class MonthlyPfPRAnalyzer(PickupAnalyzer):
    """
    Read yearly MalariaSummaryReport files.

    Reports have 30-day intervals. The month column represents a 30-day
    reporting block, not a calendar month. The final interval is dropped.
    """

    OUTPUT_PREFIX = "PfPR_ClinicalIncidence_monthly"
    TMP_DIRNAME = "_per_sim_pfpr"

    def __init__(
        self,
        expt_name,
        report_years,
        filename_template="output/MalariaSummaryReport_Monthly_U5_{year}.json",
        **kwargs,
    ):
        self.year_by_filename = {
            filename_template.format(year=year): year
            for year in report_years
        }
        super().__init__(
            filenames=list(self.year_by_filename),
            expt_name=expt_name,
            **kwargs,
        )

    def build_frame(self, data, simulation):
        parts = []

        for filename, year in self.year_by_filename.items():
            report = data.get(filename)
            if not report:
                continue

            age_bins = report["Metadata"]["Age Bins"]
            by_time_age = report["DataByTimeAndAgeBins"]

            for age_idx, age_bin in enumerate(age_bins):
                columns = {
                    name: [
                        row[age_idx]
                        for row in by_time_age[key][:-1]
                    ]
                    for name, key in PFPR_METRICS.items()
                    if key in by_time_age
                }

                if "PfPR" not in columns or not columns["PfPR"]:
                    continue

                n_months = len(columns["PfPR"])
                part = pd.DataFrame({
                    "month": range(1, n_months + 1),
                    **columns,
                })
                part["agebin"] = age_bin
                part["Year"] = year
                parts.append(part)

        return pd.concat(parts, ignore_index=True) if parts else None


# -----------------------------------------------------------------------------
# 3. Event recorder
# -----------------------------------------------------------------------------

class EventRecorderAnalyzer(PickupAnalyzer):
    """Summarize yearly event counts, collapsing year-specific event names."""

    OUTPUT_PREFIX = "EventCounts"
    TMP_DIRNAME = "_per_sim_events"
    EVENT_COL = "Event_Name"
    TIME_COL = "Time"

    def __init__(
        self,
        expt_name,
        filename="output/ReportEventRecorder.csv",
        **kwargs,
    ):
        super().__init__(
            filenames=[filename],
            expt_name=expt_name,
            **kwargs,
        )

    def build_frame(self, data, simulation):
        edf = self._to_dataframe(data.get(self.filenames[0]))
        if edf is None or edf.empty:
            return None

        if self.EVENT_COL not in edf.columns:
            print(
                f"WARNING: {self.EVENT_COL} not found in "
                f"{self.filenames[0]}"
            )
            return None

        edf[self.EVENT_COL] = edf[self.EVENT_COL].astype(str).str.replace(
            EVENT_YEAR_SUFFIX, "", regex=True
        )

        if self.TIME_COL in edf.columns:
            edf["Year"] = self.start_year + (
                edf[self.TIME_COL].astype(float) // DAYS_PER_YEAR
            ).astype(int)
        else:
            edf["Year"] = np.nan

        return (
            edf.groupby(["Year", self.EVENT_COL])
            .size()
            .reset_index(name="Count")
        )

    def reduce(self, all_data):
        super().reduce(all_data)

        if os.path.isfile(self.out_path):
            totals = (
                pd.read_csv(self.out_path)
                .groupby(self.EVENT_COL)["Count"]
                .sum()
                .sort_index()
            )
            print(f"\nEvents found in recorder output ({len(totals)}):")
            print(totals.to_string())

    @staticmethod
    def _to_dataframe(raw):
        if raw is None:
            return None
        if isinstance(raw, pd.DataFrame):
            return raw.copy()
        if isinstance(raw, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(raw))
        return pd.read_csv(io.StringIO(raw))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _group_label(group_col, value):
    if group_col == "Run_Number":
        try:
            return f"Run {int(float(value))}"
        except (TypeError, ValueError):
            return f"Run {value}"

    if isinstance(value, (float, np.floating)):
        return f"{group_col}={value:g}"

    return f"{group_col}={value}"


def plot_channels_by_group(
    df, group_col, channels, x_col, title, path, x_is_date=False
):
    channels = [channel for channel in channels if channel in df.columns]
    if not channels:
        print(f"No requested plotting channels available for {title}")
        return

    ncols = 2
    nrows = int(np.ceil(len(channels) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 4 * nrows), squeeze=False
    )
    fig.subplots_adjust(
        hspace=0.5, wspace=0.25, left=0.08, right=0.85,
        bottom=0.12, top=0.90
    )
    fig.suptitle(title, fontsize=16)

    axes = axes.ravel()
    for ax in axes[len(channels):]:
        ax.set_visible(False)

    groups = sorted(df[group_col].dropna().unique())

    for ax, channel in zip(axes, channels):
        for group in groups:
            gdf = df[df[group_col] == group].sort_values(x_col)
            ax.plot(
                gdf[x_col],
                gdf[channel],
                linewidth=1.2,
                label=_group_label(group_col, group),
            )

        ax.set_title(channel, fontsize=13)
        ax.set_ylabel(channel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        if x_is_date:
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", rotation=90)
        else:
            ax.set_xlabel(x_col, fontsize=10)

    axes[0].legend(
        title=group_col,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote chart to {path}")


def event_family(name):
    if name in ("Pregnant", "GaveBirth", "Births"):
        return "Triggers (pregnancies, births)"
    if name.startswith("Received_IPTp"):
        return "IPTp (by dose)"
    if name.startswith("Received_Vaccine"):
        return "Vaccine (by dose)"
    if name.startswith("Received_SMC"):
        return "SMC"
    if "ITN" in name or "Bednet" in name or name.startswith("ANC_Visit"):
        return "Nets / ANC"
    if "Treatment" in name:
        return "Case management"
    return "Other"


def plot_event_counts_by_year(event_df, path, title):
    if event_df.empty:
        print("No event-count rows available to plot.")
        return

    n_sims = max(event_df["sim_id"].nunique(), 1)
    totals = (
        event_df.groupby(["Year", "Event_Name"])["Count"]
        .sum()
        .reset_index()
    )
    totals["Mean_per_sim"] = totals["Count"] / n_sims
    totals["family"] = totals["Event_Name"].map(event_family)

    families = sorted(totals["family"].unique())
    ncols = 2
    nrows = int(np.ceil(len(families) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False
    )
    fig.subplots_adjust(
        hspace=0.45, wspace=0.3, left=0.07, right=0.97,
        bottom=0.08, top=0.92
    )
    fig.suptitle(title, fontsize=15)

    axes = axes.ravel()
    for ax in axes[len(families):]:
        ax.set_visible(False)

    for ax, family in zip(axes, families):
        subset = totals[totals["family"] == family]

        for event_name, edf in subset.groupby("Event_Name"):
            edf = edf.sort_values("Year")
            ax.plot(
                edf["Year"],
                edf["Mean_per_sim"],
                marker="o",
                linewidth=1.2,
                label=event_name,
            )

        ax.set_title(family, fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean events per simulation")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote chart to {path} ({n_sims} simulations)")


def plot_pfpr_by_agebin(pfpr_df, path, title):
    if pfpr_df.empty or "PfPR" not in pfpr_df.columns:
        print("No PfPR data available to plot.")
        return

    df = pfpr_df.copy()
    df["t"] = df["Year"] + (df["month"] - 0.5) / 12.0

    age_bins = sorted(df["agebin"].dropna().unique())
    if not age_bins:
        print("No age bins available to plot.")
        return

    ncols = 2
    nrows = int(np.ceil(len(age_bins) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 4 * nrows), squeeze=False
    )
    fig.subplots_adjust(
        hspace=0.45, wspace=0.25, left=0.07, right=0.97,
        bottom=0.08, top=0.92
    )
    fig.suptitle(title, fontsize=15)

    axes = axes.ravel()
    for ax in axes[len(age_bins):]:
        ax.set_visible(False)

    for ax, age_bin in zip(axes, age_bins):
        subset = df[df["agebin"] == age_bin]
        grouped = subset.groupby("t")["PfPR"]

        q10 = grouped.quantile(0.10)
        q50 = grouped.median()
        q90 = grouped.quantile(0.90)

        ax.fill_between(q50.index, q10.values, q90.values, alpha=0.25)
        ax.plot(q50.index, q50.values, linewidth=1.4)
        ax.set_title(f"Age bin up to {age_bin:g} y", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("PfPR")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote chart to {path}")


# -----------------------------------------------------------------------------
# Analysis and plotting orchestration
# -----------------------------------------------------------------------------

def run_analysis(expt_name, exp_id, job_dir, work_dir):
    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    common = {
        "expt_name": expt_name,
        "start_year": PICKUP_START_YEAR,
        "sweep_variables": SWEEP_VARIABLES,
        "step": STEP,
        "working_dir": work_dir,
        "extra_columns": {"experiment": expt_name},
    }

    analyzers = [
        InsetChartAnalyzer(channels=INSET_CHANNELS, **common),
        MonthlyPfPRAnalyzer(
            report_years=range(
                PICKUP_START_YEAR, PICKUP_END_YEAR + 1
            ),
            **common,
        ),
        EventRecorderAnalyzer(**common),
    ]

    print(
        f"\nAnalyzing pickup experiment: {expt_name}\n"
        f"Experiment ID: {exp_id}"
    )

    with Platform("SLURM_LOCAL", job_directory=job_dir):
        AnalyzeManager(
            configuration={},
            ids=[(exp_id, ItemType.EXPERIMENT)],
            analyzers=analyzers,
            partial_analyze_ok=True,
        ).analyze()


def make_plots(expt_name, work_dir):
    out_dir = output_dir(work_dir, expt_name)
    span = f"{PICKUP_START_YEAR}-{PICKUP_END_YEAR}"

    # Keep InsetChart plots grouped by Run_Number.
    inset_path = os.path.join(
        out_dir, f"All_Age_InsetChart_{STEP}.csv"
    )

    if os.path.isfile(inset_path):
        df = pd.read_csv(inset_path, parse_dates=["date"])

        if "Run_Number" in df.columns:
            channels = [c for c in INSET_CHANNELS if c in df.columns]

            by_run = (
                df.groupby(["date", "Run_Number"], as_index=False)[channels]
                .mean()
            )

            plot_channels_by_group(
                by_run,
                "Run_Number",
                channels,
                "date",
                f"{expt_name}: InsetChart ({span} pickup)",
                os.path.join(
                    out_dir, f"InsetChart_{STEP}_byRunNumber.png"
                ),
                x_is_date=True,
            )
        else:
            print("Run_Number is missing from the InsetChart CSV.")
    else:
        print(f"No InsetChart output found: {inset_path}")

    event_path = os.path.join(out_dir, f"EventCounts_{STEP}.csv")
    if os.path.isfile(event_path):
        plot_event_counts_by_year(
            pd.read_csv(event_path),
            os.path.join(out_dir, f"EventCounts_{STEP}_byYear.png"),
            f"{expt_name}: Event counts by year ({span})",
        )
    else:
        print(f"No event output found: {event_path}")

    pfpr_path = os.path.join(
        out_dir, f"PfPR_ClinicalIncidence_monthly_{STEP}.csv"
    )

    if os.path.isfile(pfpr_path):
        pfpr_df = pd.read_csv(pfpr_path)

        plot_pfpr_by_agebin(
            pfpr_df,
            os.path.join(out_dir, f"PfPR_{STEP}_byAgeBin.png"),
            f"{expt_name}: PfPR by age bin ({span})",
        )

        required = [
            "x_Temporary_Larval_Habitat",
            "scale_constant",
            "scale_temp_rain",
            "scale_water_veg",
        ]
        missing = [col for col in required if col not in pfpr_df.columns]

        if missing:
            print(f"WARNING: PfPR CSV is missing columns: {missing}")
        else:
            print("Calibration columns are present in the PfPR CSV.")
            print(
                pfpr_df[required]
                .drop_duplicates()
                .head(10)
                .to_string(index=False)
            )

        print(f"PfPR/clinical-incidence data available at:\n  {pfpr_path}")
    else:
        print(f"No PfPR output found: {pfpr_path}")


def main():
    job_dir = manifest.job_directory
    work_dir = job_dir
    os.makedirs(work_dir, exist_ok=True)

    missing = [
        name for name, exp_id in EXPERIMENTS.items()
        if not exp_id.strip()
    ]
    if missing:
        raise SystemExit(
            f"Set the pickup experiment ID in EXPERIMENTS for: {missing}. "
            "Use the ID printed when run_pickup_UWR.py submits."
        )

    for expt_name, exp_id in EXPERIMENTS.items():
        run_analysis(expt_name, exp_id.strip(), job_dir, work_dir)

    for expt_name in EXPERIMENTS:
        make_plots(expt_name, work_dir)

    print(f"\nPickup analysis complete. Results: {work_dir}")


if __name__ == "__main__":
    main()
