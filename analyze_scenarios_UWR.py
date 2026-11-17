"""
UWR Scenario Analyzer, 2026-2032 (IRS continuation vs IG2 replacement).

Companion to run_scenarios_UWR.py. Analyzes an existing scenario experiment
(pickup state x scenario sweep). It does not configure interventions.

Net-use assumptions analysed (IG2_USAGE_RUN in run_scenarios_UWR.py)
-------------------------------------------------------------------
  full           WITH age-specific usage, seasonal usage and net retention
                 effective use = c x retention x age multiplier x season multiplier
  no_age_season  WITHOUT age/seasonal usage: coverage and retention only
                 effective use = c x retention
Each assumption gets its own sub folder with estimated clinical incidence and
PfPR, cases averted (with error bars) and effective-use coverage versus malaria
outcomes relative to continued IRS (BAU).

Design
------
  Stage 1  idmtools analyzers write ONE csv per simulation, then concatenate:
             InsetChart           daily -> model-month means (+ sums of counts)
             MalariaSummaryReport time x age-bin PfPR / incidence / population
             ReportEventRecorder  event counts by model year
             ReportEventCounter   Bednet_Using per day -> monthly (optional)
           Every row carries the scenario tags written by update_simulation().
  Stage 2  post-processing (plain pandas, re-runnable with --skip-analysis):
             annual burden per simulation
             PAIRED comparison of each arm with BAU (and NO_IRS): arms share
               weather, serialized pickup state and seed, so the pairing key is
               the pickup state (pickup_sim_index)
             pairing check: years before IRS_SWITCH_YEAR are a common path
             effective use for every tested coverage c and every assumption
             minimum IG2 coverage that maintains or reduces burden
             realized IG2 distribution coverage (sanity check vs c)

Calendar: flat 365-day EMOD model calendar, day 0 = 1 Jan PROJ_START_YEAR (2026).
No Gregorian dates are assigned. Summary-report intervals are 30-day reporting
intervals, not calendar months.

Settings that must agree with run_scenarios_UWR.py are marked "MUST MATCH".

Outputs (in <job_dir>/<experiment>_AnalyzerOutput/):
  All_Age_InsetChart_scenarios.csv
  PfPR_ClinicalIncidence_monthly_scenarios.csv
  EventCounts_scenarios.csv
  NetUse_monthly_scenarios.csv                (if RUN_NET_USE)
  Scenario_AnnualBurden_scenarios.csv
  Scenario_PairedImpact_scenarios.csv         (one row per pair x arm x metric)
  Scenario_ImpactSummary_scenarios.csv        (across pairs)
  Scenario_AnnualImpact_summary_scenarios.csv
  IG2_EffectiveUse_scenarios.csv
  IG2_CoverageResponse_scenarios.csv
  IG2_MinimumCoverage_scenarios.csv
  IG2_TargetCoverage_scenarios.csv
  IG2_RealizedDistribution_scenarios.csv
  PairingCheck_scenarios.csv
  IG2_CoverageResponse_scenarios.png
  WITH_age_season_retention/
  WITHOUT_age_season_coverage_x_retention/
      Estimated_outcomes.csv                  + Estimated_incidence_PfPR.png
      CasesAverted_vs_BAU.csv                 + CasesAverted_vs_BAU.png (error bars)
      CasesAverted_annual_vs_BAU.csv
      EffectiveUse_vs_outcomes.csv            + EffectiveUse_vs_outcomes.png
      IG2_EffectiveUse.csv
      Timeseries_*.png, AnnualImpact_incidence.png, NetUse_monthly.png
  COMPARISON_both_assumptions/
      EffectiveUse_vs_outcomes_both.csv/.png

Usage:
  python analyze_scenarios_UWR.py                  # analyze + post-process + plots
  python analyze_scenarios_UWR.py --skip-analysis  # re-run stage 2 on existing csvs
  python analyze_scenarios_UWR.py --experiment-id <ID> [--experiment-name <NAME>]
"""

import argparse
import io
import os
import re
import shutil
import sys
from abc import abstractmethod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation


# =============================================================================
# SETTINGS
# =============================================================================

# NOTE: this must be the id of the SCENARIO experiment written by
# run_scenarios_UWR.py (printed as "Experiment ID" at submission), NOT the
# PICKUP_EXPERIMENT_ID. Pass --experiment-id to override.
EXPERIMENTS = {
    "UWR_BAU_vs_IG2_netuse_retention": "",
    # previous runs: UWR_BAU_vs_IG2_netuse 2f0c1c1c-ae93-4189-9731-f9fd4fc10450 (full, none);
    # UWR_BAU_vs_IG2_usage 9a45fe29-83ea-43a4-93b7-685f01aabce5 (age/season variants);
    # UWR_BAU_vs_IG2 fb883f01-13a9-4150-a559-f437786bbefa (full use only);
    # Pickup13yrs_Scenario_Final 2dd3b2e9-9f4d-4743-89b9-ab13f4f6c331
}
STEP = "scenarios"
OUT_SUFFIX = "_AnalyzerOutput"

# ---- Calendar (MUST MATCH run_scenarios_UWR.py) -----------------------------
DAYS_PER_YEAR = 365
PROJ_START_YEAR = 2026
PROJ_END_YEAR = 2032
PROJ_DAYS = (PROJ_END_YEAR - PROJ_START_YEAR + 1) * DAYS_PER_YEAR
REPORTING_INTERVAL_DAYS = 30
MONTH_STARTS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])

# ---- Scenario design (MUST MATCH) -------------------------------------------
BASELINE = "BAU"                   # business as usual (run script BAU_ARM_NAME)
LOWER_BOUND = "NO_IRS"
IRS_REFERENCES = []                # extra paired references (none in this experiment)
IRS_SWITCH_YEAR = 2027
IG2_CAMPAIGN_YEARS = [2027, 2030]
IG2_CAMPAIGN_DOY = 91
IG2_DISCARD_K = 3.0
IG2_DISCARD_L = 1095.0
IG2_USE_CURVES = True                  # must match run_scenarios_UWR.py
# Usage assumptions (MUST MATCH IG2_USAGE_MODES in run_scenarios_UWR.py).
# Arms without an ig2_usage tag (older experiments) are treated as "full".
IG2_USAGE_MODES = {
    "full": dict(age=True, season=True, retention=True),
    "none": dict(age=False, season=False, retention=False),
    "no_age": dict(age=False, season=True, retention=True),
    "no_season": dict(age=True, season=False, retention=True),
    "no_retention": dict(age=True, season=True, retention=False),
    "no_age_season": dict(age=False, season=False, retention=True),
}
DEFAULT_IG2_USAGE = "full"

# Sub folders, one per net-use assumption.
USAGE_FOLDERS = {
    "full": "WITH_age_season_retention",
    "no_age_season": "WITHOUT_age_season_coverage_x_retention",
}
USAGE_TITLES = {
    "full": "WITH age-specific usage, seasonal usage and net retention",
    "no_age_season": "WITHOUT age/seasonal usage (coverage x retention only)",
}
COMPARISON_FOLDER = "COMPARISON_both_assumptions"

# Coverage threshold: the IG2 DISTRIBUTION coverage needed to reach each
# EFFECTIVE-USE target (time-averaged over the net cycle), per net-use assumption.
# required distribution coverage = target / (retention x age x season).
# Outcomes vs BAU at that coverage are interpolated between the simulated
# coverages; outside the simulated range they are not estimated.
EFFECTIVE_USE_TARGETS = [0.40, 0.60, 0.80]
TARGET_METRICS = ["incidence_per_1000_py", "pfpr_all_ages", "U5_pfpr", "clinical_cases"]

# ---- Analysis window --------------------------------------------------------
# Years before IRS_SWITCH_YEAR are a common path. Raise ANALYSIS_START_YEAR to
# skip a warm-up (e.g. the start-up gap for women already pregnant on day 1).
ANALYSIS_START_YEAR = IRS_SWITCH_YEAR
ANALYSIS_END_YEAR = PROJ_END_YEAR

# ---- "Maintains or reduces burden" rule -------------------------------------
PRIMARY_METRIC = "incidence_per_1000_py"
# "median": median paired relative change <= margin
# "upper":  UPPER_Q quantile of the paired relative change <= margin
THRESHOLD_RULE = "median"
UPPER_Q = 0.90
NONINFERIORITY_MARGIN = 0.0        # 0.0 = no increase; 0.05 = tolerate +5 %
LOWER_Q, HIGH_Q = 0.10, 0.90       # spread shown in summaries / plots (error bars)

# ---- Usage curves for effective use (same files as the scenario script) -----
PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
NODE_ID = 1
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"
PHASES_ACCEPTED = ("pickup", "projection", "scenario")
USAGE_VALUE_COLUMNS = ["age_usage", "season_usage", "usage", "value", "use",
                       "usage_prob", "probability", "prob"]
USAGE_AGE_COLUMNS = ["age", "age_years", "age_yr"]
USAGE_DAY_COLUMNS = ["day_of_year", "doy", "day"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

# ---- Tags and channels ------------------------------------------------------
TAG_ALIASES = {
    "Run_Number": ["Run_Number"],
    "pickup_sim_index": ["pickup_sim_index"],
    "x_Temporary_Larval_Habitat": ["x_Temporary_Larval_Habitat", "xTLH"],
    "scale_constant": ["scale_constant"],
    "scale_temp_rain": ["scale_temp_rain"],
    "scale_water_veg": ["scale_water_veg"],
    "scenario": ["scenario"],
    "scenario_kind": ["scenario_kind"],
    "ig2_coverage": ["ig2_coverage"],
    "ig2_kill_key": ["ig2_kill_key"],
    "ig2_usage": ["ig2_usage"],
}
SWEEP_VARIABLES = list(TAG_ALIASES)
PAIR_FALLBACK = ["Run_Number", "x_Temporary_Larval_Habitat", "scale_constant",
                 "scale_temp_rain", "scale_water_veg"]
SIM_KEYS = ["sim_id", "pair_id", "scenario", "scenario_kind", "ig2_coverage",
            "ig2_kill_key", "ig2_usage", "strategy", "cm_boost", "Run_Number",
            "pickup_sim_index", "x_Temporary_Larval_Habitat", "scale_constant",
            "scale_temp_rain", "scale_water_veg"]

INSET_CHANNELS = [
    "Statistical Population", "New Clinical Cases", "Blood Smear Parasite Prevalence",
    "Infectious Vectors", "Adult Vectors", "Daily EIR",
]
SUM_CHANNELS = ["New Clinical Cases", "Daily EIR"]   # also written as <name>_sum
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
U5_MAX_AGE_YEARS = 5.0     # age-bin upper edge <= 5 y counts as under five
EVENT_YEAR_SUFFIX = re.compile(r"_(?:19|20)\d{2}$")

# ---- Optional net-use diagnostic (run_scenarios_UWR.py: ADD_NET_USE_COUNTER) -
RUN_NET_USE = True
NET_USE_FILE = "output/ReportEventCounter_NetUse.json"
NET_USE_CHANNELS = ["Bednet_Using"]


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

def output_dir(work_dir, expt_name):
    return os.path.join(work_dir, f"{expt_name}{OUT_SUFFIX}")


def csv_path(out_dir, prefix):
    return os.path.join(out_dir, f"{prefix}_{STEP}.csv")


def flat_calendar(n_days, start_year):
    """Map daily rows to the flat 365-day model calendar (no Gregorian dates)."""
    if n_days < 0:
        raise ValueError("n_days must be non-negative")
    day = np.arange(n_days)
    year = int(start_year) + day // DAYS_PER_YEAR
    month = np.searchsorted(MONTH_STARTS, day % DAYS_PER_YEAR, side="right")
    return year, month


def model_day_time(model_day, start_year=PROJ_START_YEAR):
    return int(start_year) + float(model_day) / DAYS_PER_YEAR


def analysis_years():
    return list(range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1))


# =============================================================================
# STAGE 1: ANALYZERS
# =============================================================================

class ScenarioAnalyzer(IAnalyzer):
    """Write per-simulation results separately, then concatenate them safely."""

    OUTPUT_PREFIX = ""
    TMP_DIRNAME = ""

    def __init__(self, filenames, expt_name, sweep_variables=None, working_dir=".",
                 start_year=PROJ_START_YEAR, step=STEP, extra_columns=None):
        super().__init__(working_dir=working_dir, filenames=filenames)
        self.expt_name = expt_name
        self.sweep_variables = sweep_variables or SWEEP_VARIABLES
        self.start_year = start_year
        self.step = step
        self.extra_columns = extra_columns or {}
        self._missing_tag_warnings = set()
        # Remove leftovers of an interrupted earlier run.
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

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
        results = [r for r in (all_data or {}).values()
                   if r and r.get("n_rows") and r.get("path")]
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
            value = next((tags[t] for t in candidates
                          if t in tags and tags[t] is not None), None)
            if isinstance(value, (list, tuple)):
                value = "-".join(map(str, value))
            if value is None:
                value = np.nan
                key = (column, tuple(candidates))
                if key not in self._missing_tag_warnings:
                    print(f"WARNING: no simulation tag for {column}; writing NaN.")
                    self._missing_tag_warnings.add(key)
            df[column] = value
        for key, value in self.extra_columns.items():
            df[key] = value
        df["sim_id"] = str(simulation.uid)


def _daily_to_monthly(daily, value_columns, sum_columns, start_year):
    daily = daily.copy()
    daily["Year"], daily["Month"] = flat_calendar(len(daily), start_year)
    daily["_n"] = 1
    agg = {c: (c, "mean") for c in value_columns}
    for c in sum_columns:
        agg[f"{c}_sum"] = (c, "sum")
    monthly = daily.groupby(["Year", "Month"], as_index=False).agg(
        **agg, days_simulated=("_n", "sum"))
    monthly["model_month"] = (monthly["Year"].astype(int).astype(str) + "-" +
                              monthly["Month"].astype(int).astype(str).str.zfill(2))
    monthly["model_time"] = monthly["Year"] + (monthly["Month"] - 0.5) / 12.0
    front = ["model_month", "model_time", "Year", "Month", "days_simulated"]
    return monthly[front + [c for c in monthly.columns if c not in front]]


class InsetChartAnalyzer(ScenarioAnalyzer):
    """Daily InsetChart -> model-month means (counts also summed)."""

    OUTPUT_PREFIX = "All_Age_InsetChart"
    TMP_DIRNAME = "_per_sim_insetchart"

    def __init__(self, expt_name, channels=None, **kwargs):
        super().__init__(["output/InsetChart.json"], expt_name, **kwargs)
        self.channels = channels or INSET_CHANNELS

    def build_frame(self, data, simulation):
        inset = data.get(self.filenames[0])
        if not inset or "Channels" not in inset:
            return None
        available = {c: inset["Channels"][c]["Data"]
                     for c in self.channels if c in inset["Channels"]}
        if not available:
            return None
        lengths = {len(v) for v in available.values()}
        if len(lengths) != 1:
            raise ValueError(f"InsetChart channels have unequal lengths: {lengths}")
        daily = pd.DataFrame(available)
        if len(daily) != PROJ_DAYS:
            print(f"WARNING: InsetChart has {len(daily)} daily rows; expected {PROJ_DAYS}.")
        return _daily_to_monthly(daily, list(available),
                                 [c for c in SUM_CHANNELS if c in available],
                                 self.start_year)


class NetUseAnalyzer(ScenarioAnalyzer):
    """Bednet_Using counts (ReportEventCounter). Diagnostic: each net counts
    separately, so this overstates the share of people using a net."""

    OUTPUT_PREFIX = "NetUse_monthly"
    TMP_DIRNAME = "_per_sim_netuse"

    def __init__(self, expt_name, filename=NET_USE_FILE, channels=None, **kwargs):
        super().__init__([filename], expt_name, **kwargs)
        self.channels = channels or NET_USE_CHANNELS

    def build_frame(self, data, simulation):
        raw = data.get(self.filenames[0])
        if not raw or "Channels" not in raw:
            return None
        available = {c: raw["Channels"][c]["Data"]
                     for c in self.channels if c in raw["Channels"]}
        if not available:
            return None
        n = {len(v) for v in available.values()}
        if len(n) != 1:
            raise ValueError(f"Net-use channels have unequal lengths: {n}")
        return _daily_to_monthly(pd.DataFrame(available), list(available),
                                 list(available), self.start_year)


class MonthlyPfPRAnalyzer(ScenarioAnalyzer):
    """
    MalariaSummaryReport time-by-age data, one file per projection year.

    Interval end days come from DataByTime['Time Of Report'] (cumulative model
    days), so the 5-day tail of each 365-day year is flagged as a partial
    interval (is_complete_interval = False) rather than dropped.
    """

    OUTPUT_PREFIX = "PfPR_ClinicalIncidence_monthly"
    TMP_DIRNAME = "_per_sim_pfpr"

    def __init__(self, expt_name, report_years,
                 filename_template="output/MalariaSummaryReport_Monthly_U5_{year}.json",
                 **kwargs):
        self.year_by_filename = {filename_template.format(year=y): y for y in report_years}
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

            nominal = int(metadata.get("Reporting_Interval", REPORTING_INTERVAL_DAYS))
            if nominal <= 0:
                raise ValueError(f"Invalid Reporting_Interval={nominal} in {filename}")

            arrays = {out: by_time_age[src] for out, src in PFPR_METRICS.items()
                      if src in by_time_age}
            if "PfPR" not in arrays:
                print(f"WARNING: PfPR metric missing from {filename}")
                continue
            n_avail = min(len(rows) for rows in arrays.values())
            if n_avail == 0:
                continue

            default_start = (int(report_year) - self.start_year) * DAYS_PER_YEAR
            start_day0 = int(metadata.get("Start_Day", default_start))
            raw_end = report.get("DataByTime", {}).get("Time Of Report", [])
            if len(raw_end) >= n_avail:
                end_days = [int(round(float(x))) for x in raw_end[:n_avail]]
            else:
                end_days = [start_day0 + min((i + 1) * nominal, DAYS_PER_YEAR)
                            for i in range(n_avail)]
                print(f"WARNING: {filename} lacks complete Time Of Report; "
                      "using Start_Day and the nominal interval.")

            records, previous_end = [], start_day0
            for idx in range(n_avail):
                end_day, start_day = end_days[idx], previous_end
                days = end_day - start_day
                previous_end = end_day
                if days <= 0:
                    print(f"WARNING: non-positive interval ({days}) in {filename}, "
                          f"interval {idx + 1}; skipping.")
                    continue
                mid = start_day + (days - 1) / 2.0
                for age_idx, age_bin in enumerate(age_bins):
                    rec = {
                        "Year": int(report_year),
                        "report_interval": idx + 1,
                        "interval_start_day": start_day,
                        "interval_end_day_exclusive": end_day,
                        "interval_days": days,
                        "nominal_interval_days": nominal,
                        "is_complete_interval": bool(days == nominal),
                        "interval_start_model_time": model_day_time(start_day, self.start_year),
                        "interval_midpoint_model_time": model_day_time(mid, self.start_year),
                        "agebin": float(age_bin),
                    }
                    for metric, rows in arrays.items():
                        row = rows[idx]
                        rec[metric] = row[age_idx] if age_idx < len(row) else np.nan
                    records.append(rec)
            if records:
                parts.append(pd.DataFrame.from_records(records))
        return pd.concat(parts, ignore_index=True) if parts else None


class EventRecorderAnalyzer(ScenarioAnalyzer):
    """Count recorded events by model year and event name."""

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
        # Strip per-cohort year suffixes (Received_IPTp_D1_2027 -> Received_IPTp_D1).
        edf["Event_Name"] = edf["Event_Name"].astype(str).str.replace(
            EVENT_YEAR_SUFFIX, "", regex=True)
        if "Time" in edf.columns:
            t = pd.to_numeric(edf["Time"], errors="coerce")
            edf["Year"] = self.start_year + np.floor(t / DAYS_PER_YEAR)
        else:
            edf["Year"] = np.nan
        return (edf.groupby(["Year", "Event_Name"], dropna=False).size()
                .reset_index(name="Count"))

    def reduce(self, all_data):
        super().reduce(all_data)
        if os.path.isfile(self.out_path):
            totals = (pd.read_csv(self.out_path).groupby("Event_Name")["Count"]
                      .sum().sort_index())
            print("\nRecorded event totals (all simulations):")
            print(totals.to_string())


def run_analysis(expt_name, exp_id, job_dir, work_dir):
    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    common = dict(expt_name=expt_name, start_year=PROJ_START_YEAR,
                  sweep_variables=SWEEP_VARIABLES, step=STEP, working_dir=work_dir,
                  extra_columns={"experiment": expt_name})
    analyzers = [
        InsetChartAnalyzer(channels=INSET_CHANNELS, **common),
        MonthlyPfPRAnalyzer(report_years=range(PROJ_START_YEAR, PROJ_END_YEAR + 1), **common),
        EventRecorderAnalyzer(**common),
    ]
    if RUN_NET_USE:
        analyzers.append(NetUseAnalyzer(**common))

    print(f"\nAnalyzing {expt_name}; experiment ID: {exp_id}")
    with Platform("SLURM_LOCAL", job_directory=job_dir):
        AnalyzeManager(configuration={}, ids=[(exp_id, ItemType.EXPERIMENT)],
                       analyzers=analyzers, partial_analyze_ok=True).analyze()


# =============================================================================
# STAGE 2: LOADING AND PAIRING
# =============================================================================

def add_pair_id(df):
    """Arms of the same pickup state share weather, state and seed: that is the pair."""
    if "pickup_sim_index" in df.columns and df["pickup_sim_index"].notna().all():
        df["pair_id"] = df["pickup_sim_index"].astype(int).astype(str)
    else:
        cols = [c for c in PAIR_FALLBACK if c in df.columns]
        print(f"WARNING: pickup_sim_index tag missing; pairing on {cols}.")
        df["pair_id"] = df[cols].astype(str).agg("|".join, axis=1) if cols else "all"
    return df


def load_table(path):
    df = pd.read_csv(path)
    for c in ("scenario", "scenario_kind", "ig2_kill_key", "ig2_usage", "strategy"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "scenario_kind" in df.columns:          # IG2 arms of older runs have no usage tag
        if "ig2_usage" not in df.columns:
            df["ig2_usage"] = ""
        is_ig2 = df["scenario_kind"] == "ig2"
        df.loc[is_ig2 & (df["ig2_usage"] == ""), "ig2_usage"] = DEFAULT_IG2_USAGE
    if "cm_boost" in df.columns:
        df["cm_boost"] = pd.to_numeric(df["cm_boost"], errors="coerce").fillna(0.0)
    if "ig2_coverage" in df.columns:
        df["ig2_coverage"] = pd.to_numeric(df["ig2_coverage"], errors="coerce")
    return add_pair_id(df)


def sim_keys(df):
    return [k for k in SIM_KEYS if k in df.columns]


# =============================================================================
# STAGE 2: BURDEN METRICS
# =============================================================================

def annual_burden(inset):
    """Per simulation and model year. Cases are summed; state channels are
    day-weighted means; incidence is annualised if a year is incomplete."""
    inset = inset.copy()
    keys = sim_keys(inset) + ["Year"]
    mean_cols = [c for c in ("Statistical Population", "Blood Smear Parasite Prevalence",
                             "Infectious Vectors", "Adult Vectors") if c in inset.columns]
    for c in mean_cols:
        inset[f"{c}__w"] = inset[c] * inset["days_simulated"]
    agg = {f"{c}__w": "sum" for c in mean_cols}
    agg["days_simulated"] = "sum"
    for c in SUM_CHANNELS:
        if f"{c}_sum" in inset.columns:
            agg[f"{c}_sum"] = "sum"
    out = inset.groupby(keys, dropna=False, as_index=False).agg(agg)
    for c in mean_cols:
        out[c] = out[f"{c}__w"] / out["days_simulated"]
        out = out.drop(columns=f"{c}__w")
    out = out.rename(columns={
        "New Clinical Cases_sum": "clinical_cases", "Daily EIR_sum": "annual_EIR",
        "Statistical Population": "population",
        "Blood Smear Parasite Prevalence": "pfpr_all_ages"})
    if {"clinical_cases", "population"} <= set(out.columns):
        scale = DAYS_PER_YEAR / out["days_simulated"]
        out["incidence_per_1000_py"] = (out["clinical_cases"] / out["population"]
                                        * 1000.0 * scale)
    out["is_complete_year"] = out["days_simulated"] == DAYS_PER_YEAR
    return out


def window_metrics(annual, years):
    """One row per simulation over the analysis window."""
    a = annual[annual["Year"].isin(years)].copy()
    if a.empty:
        raise ValueError(f"No simulated years inside the analysis window {years}.")
    a["_py"] = a["population"] * a["days_simulated"] / DAYS_PER_YEAR
    a["_pf"] = a["pfpr_all_ages"] * a["days_simulated"]
    keys = sim_keys(a)
    out = a.groupby(keys, dropna=False, as_index=False).agg(
        clinical_cases=("clinical_cases", "sum"), person_years=("_py", "sum"),
        _pf=("_pf", "sum"), days=("days_simulated", "sum"),
        _eir=("annual_EIR", "sum"))
    out["incidence_per_1000_py"] = out["clinical_cases"] / out["person_years"] * 1000.0
    out["pfpr_all_ages"] = out["_pf"] / out["days"]
    out["annual_EIR"] = out["_eir"] / (out["days"] / DAYS_PER_YEAR)
    return out.drop(columns=["_pf", "_eir"])


def u5_window_metrics(pf, years):
    """Under-five clinical incidence and PfPR over the window, population
    weighted across age bins and day-weighted across reporting intervals."""
    need = {"Cases", "Pop", "PfPR", "agebin", "interval_days", "Year"}
    if not need <= set(pf.columns):
        return None
    d = pf[(pf["agebin"] <= U5_MAX_AGE_YEARS) & pf["Year"].isin(years)].copy()
    if d.empty:
        return None
    d["_ci"] = d["Cases"] * d["Pop"]
    d["_pp"] = d["PfPR"] * d["Pop"]
    keys = ["sim_id", "Year", "report_interval", "interval_days"]
    iv = d.groupby(keys, as_index=False).agg(
        _ci=("_ci", "sum"), _pp=("_pp", "sum"), _pop=("Pop", "sum"))
    iv = iv[iv["_pop"] > 0].copy()
    iv["_wi"] = iv["_ci"] / iv["_pop"] * 1000.0 * iv["interval_days"]
    iv["_wp"] = iv["_pp"] / iv["_pop"] * iv["interval_days"]
    out = iv.groupby("sim_id", as_index=False).agg(
        _wi=("_wi", "sum"), _wp=("_wp", "sum"), _d=("interval_days", "sum"))
    out["U5_clinical_incidence_per_1000_py"] = out["_wi"] / out["_d"]
    out["U5_pfpr"] = out["_wp"] / out["_d"]
    return out[["sim_id", "U5_clinical_incidence_per_1000_py", "U5_pfpr"]]


def u5_interval_table(pf):
    """Under-five PfPR and incidence per reporting interval (for time series)."""
    need = {"Cases", "Pop", "PfPR", "agebin", "interval_midpoint_model_time"}
    if not need <= set(pf.columns):
        return None
    d = pf[pf["agebin"] <= U5_MAX_AGE_YEARS].copy()
    if d.empty:
        return None
    d["_ci"] = d["Cases"] * d["Pop"]
    d["_pp"] = d["PfPR"] * d["Pop"]
    keys = sim_keys(d) + ["Year", "report_interval", "interval_midpoint_model_time"]
    iv = d.groupby(keys, dropna=False, as_index=False).agg(
        _ci=("_ci", "sum"), _pp=("_pp", "sum"), _pop=("Pop", "sum"))
    iv = iv[iv["_pop"] > 0].copy()
    iv["U5_pfpr"] = iv["_pp"] / iv["_pop"]
    iv["U5_incidence_per_1000_py"] = iv["_ci"] / iv["_pop"] * 1000.0
    return iv


METRICS_BASE = ["clinical_cases", "incidence_per_1000_py", "pfpr_all_ages", "annual_EIR"]
METRICS_U5 = ["U5_clinical_incidence_per_1000_py", "U5_pfpr"]


# =============================================================================
# STAGE 2: PAIRED IMPACT
# =============================================================================

ARM_KEYS = ["scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key",
            "ig2_usage", "strategy", "cm_boost"]


def _fill_arm_keys(m):
    for k in ARM_KEYS:              # older experiments lack strategy / cm_boost
        if k not in m.columns:
            m[k] = 0.0 if k == "cm_boost" else (m["scenario"] if k == "strategy" else "")
    return m


def _reference_table(df, reference, value_cols, extra_keys=()):
    ref = df[df["scenario"] == reference]
    if ref.empty:
        return None
    keys = ["pair_id", *extra_keys]
    if ref.duplicated(keys).any():
        print(f"WARNING: several {reference} simulations per pair; keeping the first.")
        ref = ref.drop_duplicates(keys)
    return ref[keys + value_cols].rename(columns={c: f"{c}__ref" for c in value_cols})


def paired_impact(win, reference, metrics):
    """arm - reference, per pair, for every arm except the reference itself."""
    ref = _reference_table(win, reference, metrics)
    if ref is None:
        print(f"NOTE: no {reference} simulations; paired impact vs {reference} skipped.")
        return None
    arm = win[win["scenario"] != reference]
    m = _fill_arm_keys(arm.merge(ref, on="pair_id", how="inner"))
    if m.empty:
        print(f"WARNING: no arm could be paired with {reference}.")
        return None
    pieces = []
    for metric in metrics:
        t = m[["pair_id", *ARM_KEYS]].copy()
        t["metric"] = metric
        t["arm_value"] = m[metric]
        t["reference_value"] = m[f"{metric}__ref"]
        t["abs_diff"] = t["arm_value"] - t["reference_value"]
        t["rel_change"] = np.where(t["reference_value"] != 0,
                                   t["abs_diff"] / t["reference_value"], np.nan)
        pieces.append(t)
    out = pd.concat(pieces, ignore_index=True)
    out.insert(0, "reference", reference)
    return out


def _share_not_worse(x):
    x = x.dropna()
    return float((x <= 0).mean()) if len(x) else np.nan


def summarise_impact(paired):
    keys = ["reference", *ARM_KEYS, "metric"]
    return paired.groupby(keys, dropna=False).agg(
        n_pairs=("pair_id", "nunique"),
        median_abs_diff=("abs_diff", "median"),
        mean_abs_diff=("abs_diff", "mean"),
        median_rel_change=("rel_change", "median"),
        mean_rel_change=("rel_change", "mean"),
        q_low_rel_change=("rel_change", lambda x: x.quantile(LOWER_Q)),
        q_high_rel_change=("rel_change", lambda x: x.quantile(HIGH_Q)),
        q_upper_rule=("rel_change", lambda x: x.quantile(UPPER_Q)),
        share_pairs_not_worse=("rel_change", _share_not_worse),
    ).reset_index()


def annual_paired_summary(annual, reference):
    col = "incidence_per_1000_py"
    ref = _reference_table(annual, reference, [col], extra_keys=["Year"])
    if ref is None:
        return None
    arm = annual[annual["scenario"] != reference]
    m = _fill_arm_keys(arm.merge(ref, on=["pair_id", "Year"], how="inner"))
    if m.empty:
        return None
    m["abs_diff"] = m[col] - m[f"{col}__ref"]
    m["rel_change"] = np.where(m[f"{col}__ref"] != 0,
                               m["abs_diff"] / m[f"{col}__ref"], np.nan)
    out = m.groupby([*ARM_KEYS, "Year"], dropna=False).agg(
        n_pairs=("pair_id", "nunique"),
        median_abs_diff=("abs_diff", "median"),
        median_rel_change=("rel_change", "median"),
        q_low_rel_change=("rel_change", lambda x: x.quantile(LOWER_Q)),
        q_high_rel_change=("rel_change", lambda x: x.quantile(HIGH_Q)),
        max_abs_diff=("abs_diff", lambda x: x.abs().max()),
    ).reset_index()
    out.insert(0, "reference", reference)
    out.insert(1, "metric", "incidence_per_1000_py")
    return out


def pairing_check(annual):
    """Years before the switch are a common path: arms should match the baseline."""
    ref = _reference_table(annual, BASELINE, ["clinical_cases"], extra_keys=["Year"])
    if ref is None:
        return None
    early = annual[(annual["Year"] < IRS_SWITCH_YEAR) & (annual["scenario"] != BASELINE)]
    m = early.merge(ref, on=["pair_id", "Year"], how="inner")
    if m.empty:
        return None
    m["abs_diff_cases"] = (m["clinical_cases"] - m["clinical_cases__ref"]).abs()
    out = m.groupby(["scenario", "Year"], as_index=False).agg(
        max_abs_diff_cases=("abs_diff_cases", "max"),
        n_pairs=("pair_id", "nunique"))
    print(f"\nPairing check (years before the switch must match {BASELINE}):")
    worst = out["max_abs_diff_cases"].max()
    if worst > 0:
        print(f"  WARNING: arms differ before 1 Jan {IRS_SWITCH_YEAR} "
              f"(max |difference| in annual cases = {worst:g}). Something in the "
              f"setup is not paired.")
    else:
        print("  OK: all arms identical before the switch.")
    return out


# =============================================================================
# STAGE 2: IG2 EFFECTIVE USE
# =============================================================================

def month_number(value):
    text = str(value).strip().lower()
    if text.replace(".", "").isdigit():
        return int(float(text))
    if text[:3] in MONTH_NAMES:
        return MONTH_NAMES.index(text[:3]) + 1
    raise ValueError(f"Unrecognised month '{value}'.")


def month_mid_doy(month):
    m = month_number(month)
    return float(np.cumsum([0] + MONTH_DAYS)[m - 1]) + (MONTH_DAYS[m - 1] + 1) / 2.0


def cyclic_interp(days, vals, x):
    days, vals = np.asarray(days, float), np.asarray(vals, float)
    xs = np.concatenate([days - DAYS_PER_YEAR, days, days + DAYS_PER_YEAR])
    vs = np.tile(vals, 3)
    order = np.argsort(xs)
    return np.interp(x, xs[order], vs[order])


def read_usage_curve(filename, kind, year, channel="MASS"):
    """MASS usage curve, read the same way run_scenarios_UWR.py reads it.
    kind = 'age' (x = age in years) or 'season' (x = day of year 1..365)."""
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{filename} not found in {INPUT_DIR}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if "node_id" in df.columns:
        df = df[df["node_id"] == NODE_ID]
    if "phase" in df.columns:
        ph = df["phase"].astype(str).str.lower()
        df = df[ph.isin(PHASES_ACCEPTED) | df["phase"].isna()]
    if "channel" in df.columns:
        ch = df["channel"].astype(str).str.strip().str.upper()
        sel = df[ch == channel]
        df = sel if not sel.empty else df[df["channel"].isna()]
    if "year" in df.columns:
        yr = pd.to_numeric(df["year"], errors="coerce")
        dated = yr.dropna()
        earlier = dated[dated <= year]
        if len(earlier):
            df = df[yr == earlier.max()]
        elif yr.isna().any():
            df = df[yr.isna()]
        elif len(dated):
            df = df[yr == dated.min()]
    if df.empty:
        raise ValueError(f"{filename}: no {channel} rows for year {year}.")

    vcol = next((c for c in USAGE_VALUE_COLUMNS if c in df.columns), None)
    if vcol is None:
        raise ValueError(f"{filename}: needs a usage column among {USAGE_VALUE_COLUMNS}")
    vals = pd.to_numeric(df[vcol], errors="coerce")

    if kind == "age":
        col = next((c for c in USAGE_AGE_COLUMNS if c in df.columns), None)
        if col is None:
            raise ValueError(f"{filename}: needs an age column among {USAGE_AGE_COLUMNS}")
        x = pd.to_numeric(df[col], errors="coerce")
    else:
        dcol = next((c for c in USAGE_DAY_COLUMNS if c in df.columns), None)
        if dcol is not None:
            x = pd.to_numeric(df[dcol], errors="coerce")
        elif "season_time" in df.columns:
            x = pd.to_numeric(df["season_time"], errors="coerce") + 1
            keep = x <= DAYS_PER_YEAR
            x, vals = x[keep], vals[keep]
        elif "month" in df.columns:
            x = df["month"].map(month_mid_doy).astype(float)
        else:
            raise ValueError(f"{filename}: needs a day or month column.")
    ok = x.notna() & vals.notna()
    x, vals = x[ok].to_numpy(float), vals[ok].to_numpy(float)
    order = np.argsort(x)
    return x[order], vals[order]


def age_weights_from_reports(pf, years):
    """Mean population per age bin from the simulations' own summary reports.
    Returns (lower_edges, upper_edges, weights) or None."""
    if pf is None or not {"agebin", "Pop", "Year"} <= set(pf.columns):
        return None
    d = pf[pf["Year"].isin(years)]
    pop = d.groupby("agebin")["Pop"].mean().sort_index()
    if pop.empty or pop.sum() <= 0:
        return None
    upper = pop.index.to_numpy(float)
    lower = np.concatenate([[0.0], upper[:-1]])
    return lower, upper, pop.to_numpy(float)


def mean_age_multiplier(ages, vals, weights):
    """Population-weighted mean of the age usage curve. Within each report bin
    ages are taken as uniform (coarse for the wide 25-100 y bin)."""
    if weights is None:
        print("WARNING: no population by age bin; using an unweighted age multiplier "
              "over 0-100 y.")
        grid = np.linspace(0, 100, 1001)
        return float(np.mean(np.interp(grid, ages, vals)))
    lower, upper, w = weights
    per_bin = np.array([np.mean(np.interp(np.linspace(lo, hi, 200), ages, vals))
                        for lo, hi in zip(lower, upper)])
    return float(np.sum(w * per_bin) / np.sum(w))


def campaign_cycle_days():
    if len(IG2_CAMPAIGN_YEARS) > 1:
        return int(IG2_CAMPAIGN_YEARS[1] - IG2_CAMPAIGN_YEARS[0]) * DAYS_PER_YEAR
    return 3 * DAYS_PER_YEAR


def effective_use_factors(age_weights=None, usage=DEFAULT_IG2_USAGE):
    """c-independent factors of: effective use = c x retention x age x season."""
    cycle = campaign_cycle_days()
    t = np.arange(cycle) + 0.5                       # days since distribution
    mode = IG2_USAGE_MODES.get(usage, dict(age=True, season=True, retention=True))
    if mode.get("retention", True):
        survival = np.exp(-(t / IG2_DISCARD_L) ** IG2_DISCARD_K)
    else:
        survival = np.ones_like(t)                   # nets kept for the whole cycle
    retention = float(survival.mean())

    if not IG2_USE_CURVES or not (mode["age"] or mode["season"]):
        return dict(cycle_days=cycle, retention=retention, age_multiplier=1.0,
                    season_multiplier=1.0, season_at_distribution=1.0)

    year = IG2_CAMPAIGN_YEARS[0]
    age_mult, season_mult, season_dist = 1.0, 1.0, 1.0
    if mode["age"]:
        ages, aval = read_usage_curve(ITN_AGE_FILE, "age", year)
        age_mult = mean_age_multiplier(ages, aval, age_weights)
    if mode["season"]:
        days, sval = read_usage_curve(ITN_SEASON_FILE, "season", year)
        doy = ((IG2_CAMPAIGN_DOY - 1 + t) % DAYS_PER_YEAR) + 1
        season_t = cyclic_interp(days, sval, doy)
        # Retention-weighted, so that the product of the three factors is exact.
        season_mult = float((survival * season_t).mean() / retention)
        season_dist = float(cyclic_interp(days, sval, IG2_CAMPAIGN_DOY))
    return dict(cycle_days=cycle, retention=retention, age_multiplier=age_mult,
                season_multiplier=season_mult, season_at_distribution=season_dist)


def effective_use_table(coverages, factors, usage=DEFAULT_IG2_USAGE):
    rows = []
    for c in sorted(set(float(x) for x in coverages)):
        f = factors
        rows.append(dict(
            ig2_usage=usage,
            ig2_coverage=c,
            cycle_days=f["cycle_days"],
            net_retention=f["retention"],
            age_multiplier=f["age_multiplier"],
            season_multiplier=f["season_multiplier"],
            effective_use_time_avg=c * f["retention"] * f["age_multiplier"] * f["season_multiplier"],
            effective_use_at_distribution=c * f["age_multiplier"] * f["season_at_distribution"],
        ))
    return pd.DataFrame(rows)


# =============================================================================
# STAGE 2: MINIMUM COVERAGE
# =============================================================================

def coverage_response(summary, eff):
    s = summary[(summary["reference"] == BASELINE) & (summary["scenario_kind"] == "ig2")
                & (summary["metric"] == PRIMARY_METRIC)].copy()
    if "cm_boost" in s.columns:      # threshold is defined at current case management
        s = s[s["cm_boost"].fillna(0.0) == 0.0]
    if s.empty:
        return None
    stat = "median_rel_change" if THRESHOLD_RULE == "median" else "q_upper_rule"
    s["rule_statistic"] = s[stat]
    s["meets_criterion"] = s["rule_statistic"] <= NONINFERIORITY_MARGIN
    s = s.merge(eff, on=["ig2_usage", "ig2_coverage"], how="left")
    return s.sort_values(["ig2_kill_key", "ig2_usage", "ig2_coverage"]).reset_index(drop=True)


def minimum_coverage(resp):
    """Smallest tested coverage from which every higher tested coverage also
    meets the criterion (robust threshold), and the effective use it implies."""
    rows = []
    for (key, usage), g in resp.groupby(["ig2_kill_key", "ig2_usage"]):
        g = g.sort_values("ig2_coverage").reset_index(drop=True)
        meets = g["meets_criterion"].to_numpy(dtype=bool)
        covs = g["ig2_coverage"].to_numpy(float)
        robust = next((covs[i] for i in range(len(meets)) if meets[i:].all()), np.nan)
        first = float(covs[meets].min()) if meets.any() else np.nan

        note = []
        if np.isnan(robust):
            note.append("no tested coverage reliably meets the criterion; "
                        "extend IG2_COVERAGES upwards")
        elif robust == covs.min():
            note.append("lowest tested coverage already meets it; the true threshold "
                        "may be lower")
        if not np.isnan(robust) and not np.isnan(first) and first != robust:
            note.append(f"criterion is not monotonic in coverage (first met at "
                        f"{first:.2f}, reliably from {robust:.2f})")

        def at(c, col):
            if np.isnan(c):
                return np.nan
            sel = g.loc[np.isclose(g["ig2_coverage"], c), col]
            return float(sel.iloc[0]) if len(sel) else np.nan

        rows.append(dict(
            ig2_kill_key=key, ig2_usage=usage,
            rule=THRESHOLD_RULE, margin=NONINFERIORITY_MARGIN, metric=PRIMARY_METRIC,
            min_coverage_robust=robust, min_coverage_first_met=first,
            effective_use_time_avg_at_min=at(robust, "effective_use_time_avg"),
            effective_use_at_distribution_at_min=at(robust, "effective_use_at_distribution"),
            net_retention=at(robust, "net_retention"),
            age_multiplier=at(robust, "age_multiplier"),
            season_multiplier=at(robust, "season_multiplier"),
            median_rel_change_at_min=at(robust, "median_rel_change"),
            q_low_at_min=at(robust, "q_low_rel_change"),
            q_high_at_min=at(robust, "q_high_rel_change"),
            tested_coverages=", ".join(f"{c:.2f}" for c in covs),
            note="; ".join(note)))
    return pd.DataFrame(rows)


def target_coverage_table(summary, factors):
    """Distribution coverage needed for each effective-use target, with outcomes
    vs BAU interpolated between simulated coverages (NaN outside the tested range)."""
    s = summary[(summary["reference"] == BASELINE) & (summary["scenario_kind"] == "ig2")
                & (summary["metric"].isin(TARGET_METRICS))]
    if "cm_boost" in s.columns:
        s = s[s["cm_boost"].fillna(0.0) == 0.0]
    rows = []
    for (key, usage), g in s.groupby(["ig2_kill_key", "ig2_usage"]):
        f = factors.get(usage)
        if f is None:
            continue
        per_c = f["retention"] * f["age_multiplier"] * f["season_multiplier"]
        for target in EFFECTIVE_USE_TARGETS:
            c_req = target / per_c
            row = dict(ig2_kill_key=key, ig2_usage=usage,
                       target_effective_use=target,
                       required_distribution_coverage=c_req,
                       feasible=bool(c_req <= 1.0))
            notes = []
            if c_req > 1.0:
                notes.append("target not reachable even at 100% distribution coverage")
            for metric, gm in g.groupby("metric"):
                gm = gm.sort_values("ig2_coverage")
                xs = gm["ig2_coverage"].to_numpy(float)
                inside = xs.min() <= c_req <= xs.max()
                for col in ("median_rel_change", "median_abs_diff"):
                    row[f"{metric}__{col}"] = (
                        float(np.interp(c_req, xs, gm[col].to_numpy(float)))
                        if inside else np.nan)
            if c_req <= 1.0 and not g.empty:
                lo, hi = g["ig2_coverage"].min(), g["ig2_coverage"].max()
                if not (lo <= c_req <= hi):
                    notes.append(f"outside tested range {lo:.2f}-{hi:.2f}; "
                                 f"outcomes not estimated")
            row["note"] = "; ".join(notes)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# STAGE 2: REALIZED DISTRIBUTION (sanity check of c)
# =============================================================================

def realized_distribution(events, annual):
    """Received_ITN_IG2 events in each campaign year / mean population of that year."""
    if events is None:
        return None
    ev = events[events["Event_Name"] == "Received_ITN_IG2"].copy()
    ev = ev.dropna(subset=["Year"])
    if ev.empty:
        return None
    ev["Year"] = ev["Year"].astype(int)
    ev = ev[ev["Year"].isin([int(y) for y in IG2_CAMPAIGN_YEARS])]
    if ev.empty:
        return None
    pop = annual[["sim_id", "Year", "population"]].copy()
    pop["Year"] = pop["Year"].astype(int)
    m = ev.merge(pop, on=["sim_id", "Year"], how="left")
    m["realized_coverage"] = m["Count"] / m["population"]
    out = m.groupby(["scenario", "ig2_usage", "ig2_coverage", "Year"], dropna=False).agg(
        n_sims=("sim_id", "nunique"),
        median_received=("Count", "median"),
        median_realized_coverage=("realized_coverage", "median"),
        q_low=("realized_coverage", lambda x: x.quantile(LOWER_Q)),
        q_high=("realized_coverage", lambda x: x.quantile(HIGH_Q))).reset_index()
    out["target_coverage"] = out["ig2_coverage"]
    out["note"] = ("counts divided by the year's mean population; a rough check "
                   "that distribution matches c")
    return out


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _arm_label(scenario, cov=None):
    if scenario == BASELINE:
        return "BAU (continued IRS)"
    if scenario == LOWER_BOUND:
        return "No IRS, no IG2"
    if str(scenario).startswith("IG2_"):
        p = str(scenario).split("_")
        return f"IG2 {p[1]}%" + (f" ({p[2]})" if len(p) > 2 else "")
    if cov is not None and not pd.isna(cov):
        return f"IG2 {int(round(float(cov) * 100))}%"
    return str(scenario)


def _arm_colors(scenarios):
    ig2 = sorted(s for s in scenarios if str(s).startswith("IG2_"))
    cmap = plt.get_cmap("viridis")
    colors = {BASELINE: "black", LOWER_BOUND: "#888888"}
    for i, s in enumerate(ig2):
        colors[s] = cmap(0.15 + 0.7 * i / max(1, len(ig2) - 1))
    return colors


def _scenario_order(names):
    names = set(names)
    ig2 = sorted((s for s in names if str(s).startswith("IG2_")),
                 key=lambda s: (int(str(s).split("_")[1]), str(s)))
    return [s for s in (BASELINE, LOWER_BOUND) if s in names] + ig2


def _spread(df, keys, value):
    return (df.groupby(keys, dropna=False)[value]
            .agg(n_pairs="count", median="median", mean="mean",
                 q_low=lambda x: x.quantile(LOWER_Q),
                 q_high=lambda x: x.quantile(HIGH_Q))
            .reset_index())


def _err(df, scale=1.0):
    lo = (df["median"] - df["q_low"]).clip(lower=0).to_numpy() * scale
    hi = (df["q_high"] - df["median"]).clip(lower=0).to_numpy() * scale
    return np.vstack([lo, hi])


def _vlines(ax):
    ax.axvline(IRS_SWITCH_YEAR, color="#cc4444", ls="--", lw=0.9)
    for y in IG2_CAMPAIGN_YEARS:
        ax.axvline(y + (IG2_CAMPAIGN_DOY - 1) / DAYS_PER_YEAR, color="#4477aa",
                   ls=":", lw=0.9)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# PER-ASSUMPTION OUTPUTS: outcomes, cases averted, effective use
# =============================================================================

def arms_for(win, usage):
    """BAU + NO_IRS + the IG2 arms of ONE net-use assumption."""
    keep = ((win["scenario"] == BASELINE) | (win["scenario"] == LOWER_BOUND) |
            ((win["scenario_kind"] == "ig2") & (win["ig2_usage"] == usage)))
    return win[keep].copy()


def outcome_summary(arms, metrics):
    long = arms.melt(id_vars=["pair_id", "scenario", "ig2_coverage"],
                     value_vars=metrics, var_name="metric", value_name="value")
    return _spread(long, ["scenario", "ig2_coverage", "metric"], "value")


def plot_outcomes(summ, path, title):
    panels = [("incidence_per_1000_py", "Clinical incidence (per 1000 person-years)"),
              ("pfpr_all_ages", "PfPR, all ages (%)"),
              ("U5_clinical_incidence_per_1000_py", "Under-5 clinical incidence (per 1000 py)"),
              ("U5_pfpr", "Under-5 PfPR (%)")]
    panels = [p for p in panels if p[0] in set(summ["metric"])]
    if not panels:
        return
    order = _scenario_order(summ["scenario"])
    colors = _arm_colors(order)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.9), squeeze=False)
    for ax, (m, lab) in zip(axes[0], panels):
        scale = 100.0 if "pfpr" in m.lower() else 1.0
        s = (summ[summ["metric"] == m].drop_duplicates("scenario")
             .set_index("scenario").reindex(order).dropna(subset=["median"]))
        x = np.arange(len(s))
        ax.bar(x, s["median"] * scale, yerr=_err(s, scale), capsize=4,
               color=[colors[i] for i in s.index])
        ax.set_xticks(x)
        ax.set_xticklabels([_arm_label(i, c) for i, c in zip(s.index, s["ig2_coverage"])],
                           rotation=45, ha="right")
        ax.set_ylabel(lab)
    fig.suptitle(f"{title}\nmedian and {int(LOWER_Q * 100)}-{int(HIGH_Q * 100)}th percentile "
                 f"across pickup states, {ANALYSIS_START_YEAR}-{ANALYSIS_END_YEAR}",
                 fontsize=10)
    _save(fig, path)


def cases_averted_pairs(arms):
    p = paired_impact(arms, BASELINE, ["clinical_cases"])
    if p is None:
        return None
    p["cases_averted"] = -p["abs_diff"]          # positive = fewer cases than BAU
    p["pct_averted"] = -p["rel_change"] * 100.0
    return p


def cases_averted_summary(p):
    keys = ["scenario", "ig2_coverage"]
    a = _spread(p, keys, "cases_averted")
    b = _spread(p, keys, "pct_averted")
    return a.merge(b, on=keys, suffixes=("", "_pct"))


def annual_cases_averted(annual_arms):
    a = annual_arms[annual_arms["Year"].isin(analysis_years())]
    ref = _reference_table(a, BASELINE, ["clinical_cases"], extra_keys=["Year"])
    if ref is None:
        return None
    m = a[a["scenario"] != BASELINE].merge(ref, on=["pair_id", "Year"], how="inner")
    m["cases_averted"] = m["clinical_cases__ref"] - m["clinical_cases"]
    return _spread(m, ["scenario", "ig2_coverage", "Year"], "cases_averted")


def plot_cases_averted(total, annual_sum, path, title):
    two = annual_sum is not None and not annual_sum.empty
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(12 if two else 6, 5.0),
                             squeeze=False)
    ax = axes[0][0]
    t = total.sort_values("ig2_coverage")
    colors = _arm_colors(t["scenario"])
    ax.bar(np.arange(len(t)), t["median"], yerr=_err(t), capsize=4,
           color=[colors[s] for s in t["scenario"]])
    ax.set_xticks(np.arange(len(t)))
    ax.set_xticklabels([_arm_label(s, c) for s, c in zip(t["scenario"], t["ig2_coverage"])],
                       rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"Cumulative cases averted vs continued IRS\n"
                  f"{ANALYSIS_START_YEAR}-{ANALYSIS_END_YEAR}")
    if two:
        ax = axes[0][1]
        scen = _scenario_order(annual_sum["scenario"])
        colors = _arm_colors(scen)
        for i, sc in enumerate(scen):
            s = annual_sum[annual_sum["scenario"] == sc].sort_values("Year")
            off = (i - (len(scen) - 1) / 2) * 0.1
            ax.errorbar(s["Year"] + off, s["median"], yerr=_err(s), marker="o", capsize=3,
                        color=colors[sc], label=_arm_label(sc))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual cases averted vs continued IRS")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}\nmedian, {int(LOWER_Q * 100)}-{int(HIGH_Q * 100)}th percentile "
                 f"across pickup states (paired with BAU)", fontsize=10)
    _save(fig, path)


def effuse_outcomes(arms, eff):
    metrics = [m for m in ("incidence_per_1000_py", "pfpr_all_ages", "clinical_cases")
               if m in arms.columns]
    p = paired_impact(arms, BASELINE, metrics)
    if p is None:
        return None
    p = p[p["scenario_kind"] == "ig2"].copy()
    if p.empty:
        return None
    # clinical cases: cases averted (positive = better); others: % change vs BAU.
    p["value"] = np.where(p["metric"] == "clinical_cases", -p["abs_diff"],
                          p["rel_change"] * 100.0)
    s = _spread(p, ["ig2_usage", "ig2_coverage", "metric"], "value")
    return s.merge(eff[["ig2_usage", "ig2_coverage", "net_retention", "age_multiplier",
                        "season_multiplier", "effective_use_time_avg"]],
                   on=["ig2_usage", "ig2_coverage"], how="left")


def plot_effuse(s, path, title):
    panels = [("incidence_per_1000_py",
               "Change in clinical incidence vs continued IRS (%)\n(below 0 = better)"),
              ("pfpr_all_ages",
               "Change in PfPR vs continued IRS (%)\n(below 0 = better)"),
              ("clinical_cases",
               "Cumulative cases averted vs continued IRS\n(above 0 = better)")]
    panels = [p for p in panels if p[0] in set(s["metric"])]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0), squeeze=False)
    for ax, (m, lab) in zip(axes[0], panels):
        for usage, g in s[s["metric"] == m].groupby("ig2_usage"):
            g = g.sort_values("effective_use_time_avg")
            x = g["effective_use_time_avg"].to_numpy() * 100
            ax.errorbar(x, g["median"], yerr=_err(g), marker="o", capsize=4,
                        label=USAGE_TITLES.get(usage, usage))
            for xi, yi, c in zip(x, g["median"], g["ig2_coverage"]):
                ax.annotate(f"c={int(round(c * 100))}%", (xi, yi),
                            textcoords="offset points", xytext=(4, 6), fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Effective-use coverage, time-averaged over the net cycle (%)")
        ax.set_ylabel(lab)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title}\nmedian, {int(LOWER_Q * 100)}-{int(HIGH_Q * 100)}th percentile "
                 f"across pickup states", fontsize=10)
    _save(fig, path)


def monthly_series(df, time_col, value_cols):
    """Median and spread across pairs per arm and time."""
    out = {}
    keys = ["scenario", "ig2_coverage", "ig2_usage", time_col]
    for c in value_cols:
        if c in df.columns:
            out[c] = _spread(df, keys, c)
    return out


def plot_timeseries(ts, time_col, usage, ylabel, path, title, scale=1.0):
    is_ig2 = ts["scenario"].astype(str).str.startswith("IG2_")
    sel = ts[ts["scenario"].isin([BASELINE, LOWER_BOUND]) | (is_ig2 & (ts["ig2_usage"] == usage))]
    if sel.empty:
        return
    colors = _arm_colors(sel["scenario"].unique())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for sc in _scenario_order(sel["scenario"].unique()):
        g = sel[sel["scenario"] == sc].sort_values(time_col)
        ls = "--" if sc == LOWER_BOUND else "-"
        ax.plot(g[time_col], g["median"] * scale, color=colors[sc], ls=ls, lw=1.6)
        ax.fill_between(g[time_col], g["q_low"] * scale, g["q_high"] * scale,
                        color=colors[sc], alpha=0.12, lw=0)
        ax.annotate(_arm_label(sc), (g[time_col].iloc[-1], g["median"].iloc[-1] * scale),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, va="center",
                    color=colors[sc])
    _vlines(ax)
    ax.margins(x=0.12)
    ax.set_xlabel("Model year (flat 365-day calendar)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nmedian and {int(LOWER_Q * 100)}-{int(HIGH_Q * 100)}th percentile "
                 f"across pickup states; red dashed = IRS switch, blue dotted = IG2 campaigns",
                 fontsize=9)
    _save(fig, path)


def plot_annual_impact(ann, usage, path, title):
    is_ig2 = ann["scenario"].astype(str).str.startswith("IG2_")
    sel = ann[(ann["reference"] == BASELINE) &
              (ann["scenario"].eq(LOWER_BOUND) | (is_ig2 & (ann["ig2_usage"] == usage)))]
    if sel.empty:
        return
    colors = _arm_colors(sel["scenario"].unique())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    scen = _scenario_order(sel["scenario"].unique())
    for i, sc in enumerate(scen):
        g = sel[sel["scenario"] == sc].sort_values("Year")
        med = g["median_rel_change"].to_numpy() * 100
        err = np.vstack([(g["median_rel_change"] - g["q_low_rel_change"]).clip(lower=0) * 100,
                         (g["q_high_rel_change"] - g["median_rel_change"]).clip(lower=0) * 100])
        off = (i - (len(scen) - 1) / 2) * 0.08
        ax.errorbar(g["Year"] + off, med, yerr=err, marker="o", capsize=3,
                    color=colors[sc], ls="--" if sc == LOWER_BOUND else "-")
        ax.annotate(_arm_label(sc), (g["Year"].iloc[-1] + off, med[-1]),
                    xytext=(5, 0), textcoords="offset points", fontsize=7, va="center",
                    color=colors[sc])
    ax.axhline(0, color="k", lw=0.8)
    ax.margins(x=0.12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Change in clinical incidence vs continued IRS (%)\n(below 0 = better)")
    ax.set_title(f"{title}\nmedian, {int(LOWER_Q * 100)}-{int(HIGH_Q * 100)}th percentile "
                 f"across pickup states (paired with BAU)", fontsize=9)
    _save(fig, path)


def write_usage_subfolders(out_dir, ctx):
    """One sub folder per net-use assumption, plus a comparison folder."""
    annual, win, eff, ann_summ = ctx["annual"], ctx["win"], ctx["eff"], ctx["ann_summ"]
    metrics = [m for m in METRICS_BASE + METRICS_U5 if m in win.columns]
    both = []
    for usage, folder in USAGE_FOLDERS.items():
        arms = arms_for(win, usage)
        if not ((arms["scenario_kind"] == "ig2") & (arms["ig2_usage"] == usage)).any():
            print(f"NOTE: no IG2 arms with ig2_usage={usage}; {folder} skipped.")
            continue
        d = os.path.join(out_dir, folder)
        os.makedirs(d, exist_ok=True)
        title = USAGE_TITLES[usage]

        # 1. estimated clinical incidence and PfPR
        summ = outcome_summary(arms, metrics)
        summ.to_csv(os.path.join(d, "Estimated_outcomes.csv"), index=False)
        plot_outcomes(summ, os.path.join(d, "Estimated_incidence_PfPR.png"), title)

        # 2. cases averted vs continued IRS, with error bars
        p = cases_averted_pairs(arms)
        if p is not None:
            tot = cases_averted_summary(p)
            tot.to_csv(os.path.join(d, "CasesAverted_vs_BAU.csv"), index=False)
            ann = annual_cases_averted(
                annual[annual["scenario"].isin(arms["scenario"].unique())])
            if ann is not None:
                ann = ann[ann["scenario"].astype(str).str.startswith("IG2_")]
                ann.to_csv(os.path.join(d, "CasesAverted_annual_vs_BAU.csv"), index=False)
            plot_cases_averted(tot[tot["scenario"].astype(str).str.startswith("IG2_")],
                               ann, os.path.join(d, "CasesAverted_vs_BAU.png"), title)

        # 3. effective-use coverage vs outcomes relative to continued IRS
        eo = effuse_outcomes(arms, eff)
        if eo is not None:
            eo.to_csv(os.path.join(d, "EffectiveUse_vs_outcomes.csv"), index=False)
            plot_effuse(eo, os.path.join(d, "EffectiveUse_vs_outcomes.png"), title)
            both.append(eo)
        eff[eff["ig2_usage"] == usage].to_csv(os.path.join(d, "IG2_EffectiveUse.csv"),
                                              index=False)

        # 4. time series, annual impact, net-use diagnostic
        for name, (ts, tcol, ylabel, scale) in ctx["series"].items():
            plot_timeseries(ts, tcol, usage, ylabel,
                            os.path.join(d, f"Timeseries_{name}.png"), title, scale)
        if ann_summ is not None:
            plot_annual_impact(ann_summ, usage, os.path.join(d, "AnnualImpact_incidence.png"),
                               title)
        if ctx.get("netuse") is not None:
            plot_timeseries(ctx["netuse"], "model_time", usage,
                            "Bednet_Using events per day (diagnostic; counts every net)",
                            os.path.join(d, "NetUse_monthly.png"), title)
        print(f"Wrote {d}")

    if len(both) > 1:
        d = os.path.join(out_dir, COMPARISON_FOLDER)
        os.makedirs(d, exist_ok=True)
        allv = pd.concat(both, ignore_index=True)
        allv.to_csv(os.path.join(d, "EffectiveUse_vs_outcomes_both.csv"), index=False)
        plot_effuse(allv, os.path.join(d, "EffectiveUse_vs_outcomes_both.png"),
                    "Effective-use coverage and malaria outcomes relative to continued IRS")
        print(f"Wrote {d}")


def plot_coverage_response(resp, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for usage, g in resp.groupby("ig2_usage"):
        g = g.sort_values("ig2_coverage")
        med = g["rule_statistic"].to_numpy() * 100
        err = np.vstack([(g["median_rel_change"] - g["q_low_rel_change"]).clip(lower=0) * 100,
                         (g["q_high_rel_change"] - g["median_rel_change"]).clip(lower=0) * 100])
        for ax, x, xl in ((axes[0], g["ig2_coverage"] * 100, "IG2 distribution coverage c (%)"),
                          (axes[1], g["effective_use_time_avg"] * 100,
                           "Effective-use coverage, time-averaged (%)")):
            ax.errorbar(x, med, yerr=err, marker="o", capsize=4,
                        label=USAGE_TITLES.get(usage, usage))
            ax.set_xlabel(xl)
    for ax in axes:
        ax.axhline(NONINFERIORITY_MARGIN * 100, color="k", ls="--", lw=0.9)
        ax.set_ylabel(f"Change in {PRIMARY_METRIC} vs continued IRS (%)\n"
                      f"({THRESHOLD_RULE} rule; at or below dashed line = maintains or reduces)")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, path)


# =============================================================================
# STAGE 2: DRIVER
# =============================================================================

def load_inputs(out_dir):
    tables = {}
    for key, prefix, required in (("inset", "All_Age_InsetChart", True),
                                  ("pf", "PfPR_ClinicalIncidence_monthly", True),
                                  ("events", "EventCounts", False),
                                  ("netuse", "NetUse_monthly", False)):
        path = csv_path(out_dir, prefix)
        if os.path.isfile(path):
            tables[key] = load_table(path)
        elif required:
            raise FileNotFoundError(f"{path} not found. Run without --skip-analysis first.")
        else:
            tables[key] = None
    if (tables["inset"]["scenario"] == "").all():
        raise ValueError(
            "No scenario tags in the analyzer output. EXPERIMENTS/--experiment-id must be "
            "the SCENARIO experiment written by run_scenarios_UWR.py, not the pickup "
            "experiment.")
    return tables


def post_process(out_dir):
    t = load_inputs(out_dir)
    inset, pf, events, netuse = t["inset"], t["pf"], t["events"], t["netuse"]
    years = analysis_years()
    present = sorted(inset["scenario"].unique())
    print(f"\nArms found: {present}")
    if BASELINE not in present:
        raise ValueError(f"Baseline arm '{BASELINE}' not found; found {present}.")

    # ---- annual burden and pairing check ----
    annual = annual_burden(inset)
    annual.to_csv(csv_path(out_dir, "Scenario_AnnualBurden"), index=False)
    pc = pairing_check(annual)
    if pc is not None:
        pc.to_csv(csv_path(out_dir, "PairingCheck"), index=False)

    # ---- window metrics (all ages + under five) ----
    win = window_metrics(annual, years)
    u5 = u5_window_metrics(pf, years)
    if u5 is not None:
        win = win.merge(u5, on="sim_id", how="left")
    metrics = [m for m in METRICS_BASE + METRICS_U5 if m in win.columns]
    print(f"Analysis window {years[0]}-{years[-1]}; {win['pair_id'].nunique()} pickup states; "
          f"metrics: {metrics}")

    # ---- paired impact ----
    refs = [BASELINE, LOWER_BOUND] + [r for r in IRS_REFERENCES if r not in (BASELINE, LOWER_BOUND)]
    pieces = [paired_impact(win, r, metrics) for r in refs]
    pieces = [p for p in pieces if p is not None]
    paired = pd.concat(pieces, ignore_index=True)
    paired.to_csv(csv_path(out_dir, "Scenario_PairedImpact"), index=False)
    summary = summarise_impact(paired)
    summary.to_csv(csv_path(out_dir, "Scenario_ImpactSummary"), index=False)

    ann_pieces = [annual_paired_summary(annual, r) for r in refs]
    ann_pieces = [a for a in ann_pieces if a is not None]
    ann_summ = pd.concat(ann_pieces, ignore_index=True) if ann_pieces else None
    if ann_summ is not None:
        ann_summ.to_csv(csv_path(out_dir, "Scenario_AnnualImpact_summary"), index=False)

    # ---- effective use per net-use assumption ----
    ig2_win = win[win["scenario_kind"] == "ig2"]
    usages = sorted(ig2_win["ig2_usage"].unique())
    covs = sorted(ig2_win["ig2_coverage"].dropna().unique())
    age_w = age_weights_from_reports(pf, years)
    factors = {u: effective_use_factors(age_w, u) for u in usages}
    eff = pd.concat([effective_use_table(covs, factors[u], u) for u in usages],
                    ignore_index=True) if usages else pd.DataFrame()
    eff.to_csv(csv_path(out_dir, "IG2_EffectiveUse"), index=False)
    for u in usages:
        f = factors[u]
        print(f"  {u}: retention {f['retention']:.3f}, age x {f['age_multiplier']:.3f}, "
              f"season x {f['season_multiplier']:.3f}")

    # ---- threshold ----
    resp = coverage_response(summary, eff) if not eff.empty else None
    if resp is not None:
        resp.to_csv(csv_path(out_dir, "IG2_CoverageResponse"), index=False)
        minc = minimum_coverage(resp)
        minc.to_csv(csv_path(out_dir, "IG2_MinimumCoverage"), index=False)
        plot_coverage_response(resp, os.path.join(out_dir, "IG2_CoverageResponse_scenarios.png"))
        print("\nMinimum IG2 distribution coverage that maintains or reduces "
              f"{PRIMARY_METRIC} (rule: {THRESHOLD_RULE}, margin {NONINFERIORITY_MARGIN:g}):")
        for r in minc.itertuples():
            print(f"  {r.ig2_kill_key}/{r.ig2_usage}: min c = {r.min_coverage_robust}, "
                  f"effective use at min = {r.effective_use_time_avg_at_min}"
                  f"{'  [' + r.note + ']' if r.note else ''}")
        tgt = target_coverage_table(summary, factors)
        tgt.to_csv(csv_path(out_dir, "IG2_TargetCoverage"), index=False)
    else:
        print("NOTE: no IG2 arms vs BAU; threshold tables skipped.")

    # ---- realized distribution ----
    real = realized_distribution(events, annual)
    if real is not None:
        real.to_csv(csv_path(out_dir, "IG2_RealizedDistribution"), index=False)

    # ---- time series ----
    inset = inset.copy()
    if {"New Clinical Cases_sum", "Statistical Population"} <= set(inset.columns):
        inset["incidence_monthly_per_1000_py"] = (
            inset["New Clinical Cases_sum"] / inset["Statistical Population"]
            * 1000.0 * DAYS_PER_YEAR / inset["days_simulated"])
    series = {}
    s = monthly_series(inset, "model_time", ["Blood Smear Parasite Prevalence"])
    if s:
        series["PfPR_all_ages"] = (s["Blood Smear Parasite Prevalence"], "model_time",
                                   "PfPR, all ages (%)", 100.0)
    s = monthly_series(inset, "model_time", ["incidence_monthly_per_1000_py"])
    if s:
        series["Incidence_all_ages"] = (s["incidence_monthly_per_1000_py"], "model_time",
                                        "Clinical incidence, all ages (per 1000 py)", 1.0)
    u5i = u5_interval_table(pf)
    if u5i is not None:
        s = monthly_series(u5i, "interval_midpoint_model_time", ["U5_pfpr"])
        series["PfPR_under5"] = (s["U5_pfpr"], "interval_midpoint_model_time",
                                 "Under-5 PfPR (%)", 100.0)
        s = monthly_series(u5i, "interval_midpoint_model_time", ["U5_incidence_per_1000_py"])
        series["Incidence_under5"] = (s["U5_incidence_per_1000_py"],
                                      "interval_midpoint_model_time",
                                      "Under-5 clinical incidence (per 1000 py)", 1.0)
    netuse_ts = None
    if netuse is not None and "Bednet_Using" in netuse.columns:
        netuse_ts = monthly_series(netuse, "model_time", ["Bednet_Using"])["Bednet_Using"]
        netuse_ts.to_csv(csv_path(out_dir, "NetUse_summary"), index=False)

    # ---- per-assumption sub folders ----
    write_usage_subfolders(out_dir, dict(annual=annual, win=win, eff=eff,
                                         ann_summ=ann_summ, series=series,
                                         netuse=netuse_ts))
    print(f"\nDone. Outputs in {out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Analyze UWR IRS vs IG2 scenario experiment.")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Skip stage 1 (idmtools) and re-run stage 2 on existing csvs.")
    p.add_argument("--experiment-id", default=None,
                   help="idmtools id of the SCENARIO experiment (overrides EXPERIMENTS).")
    p.add_argument("--experiment-name", default=None,
                   help="Name used for the output folder (default: from EXPERIMENTS).")
    p.add_argument("--job-dir", default=None,
                   help="Job directory (default: manifest.job_directory).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.experiment_id:
        exp_id = args.experiment_id.strip()
        expt_name = args.experiment_name or next(
            (n for n, i in EXPERIMENTS.items() if i == exp_id), f"experiment_{exp_id[:8]}")
    else:
        expt_name, exp_id = next(iter(EXPERIMENTS.items()))
        if args.experiment_name:
            expt_name = args.experiment_name

    job_dir = args.job_dir
    if job_dir is None:
        try:
            import manifest
            job_dir = manifest.job_directory
        except Exception as exc:
            sys.exit(f"Could not read manifest.job_directory ({exc}); pass --job-dir.")

    out_dir = output_dir(job_dir, expt_name)
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_analysis:
        run_analysis(expt_name, exp_id, job_dir, job_dir)

    post_process(out_dir)


if __name__ == "__main__":
    main()