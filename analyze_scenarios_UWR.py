"""
UWR Scenario Analyzer, 2026-2032 (IRS continuation vs IG2 replacement).

Companion to run_scenarios_UWR.py. Analyzes an existing scenario experiment
(pickup state x scenario sweep). It does not configure interventions.

Design (same pattern as the pickup analyzer)
--------------------------------------------
  Stage 1  idmtools analyzers write ONE csv per simulation, then concatenate:
             InsetChart          daily -> model-month means (+ sums of counts)
             MalariaSummaryReport time x age-bin PfPR / incidence / population
             ReportEventRecorder  event counts by model year
             ReportEventCounter   Bednet_Using per day -> monthly (optional)
           Every row carries the scenario tags written by update_simulation():
           scenario, scenario_kind, ig2_coverage, ig2_kill_key, Run_Number,
           pickup_sim_index and the habitat tags.
  Stage 2  post-processing (plain pandas, re-runnable with --skip-analysis):
             annual burden per simulation
             PAIRED comparison of each arm with IRS_CONTINUE (and NO_IRS):
               arms share weather, serialized pickup state and seed, so the
               pairing key is the pickup state (pickup_sim_index)
             pairing check: 2026 is a common path, arms must agree before
               IRS_SWITCH_YEAR
             effective use for every tested coverage c:
               effective use = c x net retention x age multiplier x season multiplier
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
  IG2_RealizedDistribution_scenarios.csv
  PairingCheck_scenarios.csv
  + PNG plots

Usage:
  python analyze_scenarios_UWR.py                  # analyze + post-process + plots
  python analyze_scenarios_UWR.py --skip-analysis  # re-run stage 2 on existing csvs
  python analyze_scenarios_UWR.py --experiment-id <ID>
"""

import argparse
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


# =============================================================================
# SETTINGS
# =============================================================================

EXPERIMENTS = {
    "XTLH_scenarios_IG2": "501cfc20-b86d-464d-9e98-8b2ae8736b03",  #                                  PUT-SCENARIO-EXPERIMENT-ID-HERE
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
BASELINE = "IRS_CONTINUE"
LOWER_BOUND = "NO_IRS"
IRS_SWITCH_YEAR = 2027
IG2_CAMPAIGN_YEARS = [2027, 2030]
IG2_CAMPAIGN_DOY = 91
IG2_DISCARD_K = 3.0
IG2_DISCARD_L = 1095.0
IG2_USE_CURVES = True

# ---- Analysis window --------------------------------------------------------
# Years before IRS_SWITCH_YEAR are a common path. Raise ANALYSIS_START_YEAR to
# skip a warm-up (e.g. the start-up gap for women already pregnant on day 1).
ANALYSIS_START_YEAR = IRS_SWITCH_YEAR
ANALYSIS_END_YEAR = PROJ_END_YEAR

# ---- "Maintains or reduces burden" rule -------------------------------------
# Primary burden metric (column of the window-metric table).
PRIMARY_METRIC = "incidence_per_1000_py"
# "median": median paired relative change <= margin
# "upper":  UPPER_Q quantile of the paired relative change <= margin
THRESHOLD_RULE = "median"
UPPER_Q = 0.90
NONINFERIORITY_MARGIN = 0.0        # 0.0 = no increase; 0.05 = tolerate +5 %
LOWER_Q, HIGH_Q = 0.10, 0.90       # spread shown in summaries / plots

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
}
SWEEP_VARIABLES = list(TAG_ALIASES)
PAIR_FALLBACK = ["Run_Number", "x_Temporary_Larval_Habitat", "scale_constant",
                 "scale_temp_rain", "scale_water_veg"]
SIM_KEYS = ["sim_id", "pair_id", "scenario", "scenario_kind", "ig2_coverage",
            "ig2_kill_key", "Run_Number", "pickup_sim_index",
            "x_Temporary_Larval_Habitat", "scale_constant", "scale_temp_rain",
            "scale_water_veg"]

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
PLOT_EVENTS = ["Received_ITN_IG2", "Received_ITN_ANC", "Received_ITN_CWC",
               "Received_SMC", "Received_Treatment", "NewClinicalCase"]

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
    for c in ("scenario", "scenario_kind", "ig2_kill_key"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "ig2_coverage" in df.columns:
        df["ig2_coverage"] = pd.to_numeric(df["ig2_coverage"], errors="coerce")
    return add_pair_id(df)


def sim_keys(df):
    return [k for k in SIM_KEYS if k in df.columns]


def scenario_cols(df):
    return [c for c in ("scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key")
            if c in df.columns]


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


METRICS_BASE = ["clinical_cases", "incidence_per_1000_py", "pfpr_all_ages", "annual_EIR"]
METRICS_U5 = ["U5_clinical_incidence_per_1000_py", "U5_pfpr"]


# =============================================================================
# STAGE 2: PAIRED IMPACT
# =============================================================================

ARM_KEYS = ["scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key"]


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
    m = arm.merge(ref, on="pair_id", how="inner")
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
    metric = "incidence_per_1000"
    col = "incidence_per_1000_py"
    ref = _reference_table(annual, reference, [col], extra_keys=["Year"])
    if ref is None:
        return None
    arm = annual[annual["scenario"] != reference]
    m = arm.merge(ref, on=["pair_id", "Year"], how="inner")
    m["abs_diff"] = m[col] - m[f"{col}__ref"]
    m["rel_change"] = np.where(m[f"{col}__ref"] != 0,
                               m["abs_diff"] / m[f"{col}__ref"], np.nan)
    out = m.groupby([*ARM_KEYS, "Year"], dropna=False).agg(
        n_pairs=("pair_id", "nunique"),
        median_abs_diff=("abs_diff", "median"),
        median_rel_change=("rel_change", "median"),
        q_low_rel_change=("rel_change", lambda x: x.quantile(LOWER_Q)),
        q_high_rel_change=("rel_change", lambda x: x.quantile(HIGH_Q)),
        max_abs_diff_cases=("abs_diff", lambda x: x.abs().max()),
    ).reset_index()
    out.insert(0, "reference", reference)
    out.insert(1, "metric", metric)
    return out


def pairing_check(annual):
    """2026 is a common path: arms should match the baseline before the switch."""
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
    print("\nPairing check (years before the switch must match "
          f"{BASELINE}):")
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


def effective_use_factors(age_weights=None):
    """Return the c-independent factors of: effective use = c x retention x age x season."""
    cycle = campaign_cycle_days()
    t = np.arange(cycle) + 0.5                       # days since distribution
    survival = np.exp(-(t / IG2_DISCARD_L) ** IG2_DISCARD_K)
    retention = float(survival.mean())

    if not IG2_USE_CURVES:
        return dict(cycle_days=cycle, retention=retention, age_multiplier=1.0,
                    season_multiplier=1.0, season_at_distribution=1.0)

    year = IG2_CAMPAIGN_YEARS[0]
    ages, aval = read_usage_curve(ITN_AGE_FILE, "age", year)
    days, sval = read_usage_curve(ITN_SEASON_FILE, "season", year)
    age_mult = mean_age_multiplier(ages, aval, age_weights)

    doy = ((IG2_CAMPAIGN_DOY - 1 + t) % DAYS_PER_YEAR) + 1
    season_t = cyclic_interp(days, sval, doy)
    # Retention-weighted, so that the product of the three factors is exact.
    season_mult = float((survival * season_t).mean() / retention)
    return dict(cycle_days=cycle, retention=retention, age_multiplier=age_mult,
                season_multiplier=season_mult,
                season_at_distribution=float(cyclic_interp(days, sval, IG2_CAMPAIGN_DOY)))


def effective_use_table(coverages, factors):
    rows = []
    for c in sorted(set(float(x) for x in coverages)):
        f = factors
        rows.append(dict(
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
    if s.empty:
        return None
    stat = "median_rel_change" if THRESHOLD_RULE == "median" else "q_upper_rule"
    s["rule_statistic"] = s[stat]
    s["meets_criterion"] = s["rule_statistic"] <= NONINFERIORITY_MARGIN
    s = s.merge(eff, on="ig2_coverage", how="left")
    return s.sort_values(["ig2_kill_key", "ig2_coverage"]).reset_index(drop=True)


def minimum_coverage(resp):
    rows = []
    for key, g in resp.groupby("ig2_kill_key"):
        g = g.sort_values("ig2_coverage")
        meets = g["meets_criterion"].to_numpy()
        covs = g["ig2_coverage"].to_numpy()
        robust = next((covs[i] for i in range(len(meets)) if meets[i:].all()), np.nan)
        first = covs[meets].min() if meets.any() else np.nan
        note = []
        if np.isnan(robust):
            note.append("no tested coverage reliably meets the criterion; "
                        "extend IG2_COVERAGES upwards")
        elif robust == covs.min():
            note.append("lowest tested coverage already meets it; the true threshold "
                        "may be lower")
        if not np.isnan(robust) and not np.isnan(first) and robust != first:
            note.append("non-monotone response (stochastic noise?); use the robust value")
        pick = g[g["ig2_coverage"] == robust]
        rows.append(dict(
            ig2_kill_key=key, primary_metric=PRIMARY_METRIC, rule=THRESHOLD_RULE,
            upper_q=UPPER_Q if THRESHOLD_RULE == "upper" else np.nan,
            margin=NONINFERIORITY_MARGIN,
            min_coverage_robust=robust, min_coverage_first_meeting=first,
            effective_use_time_avg=float(pick["effective_use_time_avg"].iloc[0]) if len(pick) else np.nan,
            effective_use_at_distribution=float(pick["effective_use_at_distribution"].iloc[0]) if len(pick) else np.nan,
            median_rel_change_at_threshold=float(pick["median_rel_change"].iloc[0]) if len(pick) else np.nan,
            n_pairs=int(g["n_pairs"].max()),
            note="; ".join(note)))
    return pd.DataFrame(rows)


# =============================================================================
# STAGE 2: REALIZED DISTRIBUTION
# =============================================================================

def realized_distribution(events, annual):
    """Received_ITN_IG2 events per person in campaign years, vs the requested c."""
    e = events[events["Event_Name"] == "Received_ITN_IG2"].copy()
    if e.empty:
        return None
    e = e.dropna(subset=["Year"])
    e["Year"] = e["Year"].astype(int)
    per_sim = e.groupby(["sim_id", "Year"], as_index=False)["Count"].sum()
    m = per_sim.merge(annual[["sim_id", "Year", "population"]], on=["sim_id", "Year"], how="left")
    m = m.merge(annual[["sim_id", "scenario", "ig2_coverage"]].drop_duplicates("sim_id"),
                on="sim_id", how="left")
    m["realized_coverage"] = m["Count"] / m["population"]
    out = m.groupby(["scenario", "ig2_coverage", "Year"], as_index=False).agg(
        mean_nets_issued=("Count", "mean"),
        mean_realized_coverage=("realized_coverage", "mean"),
        n_sims=("sim_id", "nunique"))
    out["ratio_to_requested"] = out["mean_realized_coverage"] / out["ig2_coverage"]
    return out


# =============================================================================
# PLOTS
# =============================================================================

def scenario_styles(df):
    """Baseline black, NO_IRS grey dashed, IG2 arms viridis by coverage."""
    info = (df[scenario_cols(df)].drop_duplicates().sort_values(
        ["scenario_kind", "ig2_coverage"]))
    cov = info.loc[info["scenario_kind"] == "ig2", "ig2_coverage"]
    lo, hi = (cov.min(), cov.max()) if len(cov) else (0, 1)
    kills = sorted(info.loc[info["scenario_kind"] == "ig2", "ig2_kill_key"].unique())
    dashes = ["-", ":", "-."]
    styles = {}
    for r in info.itertuples():
        if r.scenario == BASELINE:
            styles[r.scenario] = dict(color="black", linestyle="-", linewidth=2.0)
        elif r.scenario == LOWER_BOUND:
            styles[r.scenario] = dict(color="0.5", linestyle="--", linewidth=1.6)
        else:
            frac = 0.0 if hi == lo else (r.ig2_coverage - lo) / (hi - lo)
            ls = dashes[kills.index(r.ig2_kill_key) % 3] if r.ig2_kill_key in kills else "-"
            styles[r.scenario] = dict(color=plt.cm.viridis(0.05 + 0.85 * frac),
                                      linestyle=ls, linewidth=1.3)
    order = [s for s in [BASELINE, LOWER_BOUND] if s in styles] + \
            [r.scenario for r in info.itertuples() if r.scenario not in (BASELINE, LOWER_BOUND)]
    return styles, order


def _mark_campaigns(ax, x_is_year=True):
    ax.axvline(IRS_SWITCH_YEAR, color="red", linestyle="--", linewidth=0.9, alpha=0.7)
    for y in IG2_CAMPAIGN_YEARS:
        ax.axvline(y + (IG2_CAMPAIGN_DOY - 1) / DAYS_PER_YEAR, color="tab:blue",
                   linestyle=":", linewidth=0.9, alpha=0.7)


def plot_inset_by_scenario(df, channels, path, title, x_col="model_time"):
    channels = [c for c in channels if c in df.columns]
    if not channels:
        return
    styles, order = scenario_styles(df)
    ncols = 2
    nrows = int(np.ceil(len(channels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(channels):]:
        ax.set_visible(False)
    for ax, ch in zip(axes, channels):
        for scen in order:
            g = df[df["scenario"] == scen].groupby(x_col)[ch]
            med = g.median()
            ax.plot(med.index, med.values, label=scen, **styles[scen])
            if scen == BASELINE:
                ax.fill_between(med.index, g.quantile(LOWER_Q).values,
                                g.quantile(HIGH_Q).values, color="black", alpha=0.10)
        _mark_campaigns(ax)
        ax.set_title(ch)
        ax.set_xlabel("Model year (365-day model calendar)")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"{title}\nmedian across pickup states; band = {int(LOWER_Q*100)}-"
                 f"{int(HIGH_Q*100)}% of {BASELINE}; red = IRS switch, blue = IG2 campaigns")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_annual_impact(annual_summary, path, title):
    if annual_summary is None or annual_summary.empty:
        return
    styles, order = scenario_styles(annual_summary)
    fig, ax = plt.subplots(figsize=(11, 6))
    for scen in order:
        g = annual_summary[annual_summary["scenario"] == scen].sort_values("Year")
        if g.empty:
            continue
        ax.plot(g["Year"], g["median_rel_change"] * 100, marker="o", ms=3,
                label=scen, **styles[scen])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(IRS_SWITCH_YEAR, color="red", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Model year")
    ax.set_ylabel(f"Median paired change in annual incidence vs {BASELINE} (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_response(resp, thr, path, title):
    if resp is None or resp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, g in resp.groupby("ig2_kill_key"):
        g = g.sort_values("ig2_coverage")
        y = g["median_rel_change"].to_numpy() * 100
        lo = g["q_low_rel_change"].to_numpy() * 100
        hi = g["q_high_rel_change"].to_numpy() * 100
        ax.errorbar(g["ig2_coverage"], y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    marker="o", capsize=3, label=f"IG2 ({key}); median, {int(LOWER_Q*100)}-{int(HIGH_Q*100)}%")
    ax.axhline(NONINFERIORITY_MARGIN * 100, color="black", linewidth=1,
               label=f"criterion ({THRESHOLD_RULE} <= {NONINFERIORITY_MARGIN*100:g}%)")
    if thr is not None:
        for r in thr.itertuples():
            if not np.isnan(r.min_coverage_robust):
                ax.axvline(r.min_coverage_robust, color="red", linestyle="--", alpha=0.7,
                           label=f"minimum c = {r.min_coverage_robust:.2f} ({r.ig2_kill_key})")
    ax.set_xlabel("IG2 coverage distributed per campaign (c)")
    ax.set_ylabel(f"Paired change in {PRIMARY_METRIC} vs {BASELINE} (%)")
    f = (resp["effective_use_time_avg"] / resp["ig2_coverage"]).dropna()
    if len(f):
        factor = float(f.iloc[0])
        sec = ax.secondary_xaxis("top", functions=(lambda x: x * factor, lambda x: x / factor))
        sec.set_xlabel("Time-averaged effective use (c x retention x age x season)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pfpr_by_agebin(pf, path, title):
    if pf.empty or "PfPR" not in pf.columns:
        return
    styles, order = scenario_styles(pf)
    bins = sorted(pf["agebin"].dropna().unique())
    ncols = 2
    nrows = int(np.ceil(len(bins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(bins):]:
        ax.set_visible(False)
    for ax, b in zip(axes, bins):
        sub = pf[pf["agebin"] == b]
        for scen in order:
            med = (sub[sub["scenario"] == scen]
                   .groupby("interval_midpoint_model_time")["PfPR"].median())
            ax.plot(med.index, med.values, label=scen, **styles[scen])
        _mark_campaigns(ax)
        ax.set_title(f"Age bin up to {b:g} years")
        ax.set_xlabel("Model time")
        ax.set_ylabel("PfPR")
        ax.set_xlim(PROJ_START_YEAR, PROJ_END_YEAR + 1)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_counts(events, path, title):
    names = [e for e in PLOT_EVENTS if e in set(events["Event_Name"])]
    if not names:
        return
    per_sim = (events.dropna(subset=["Year"]).groupby(["sim_id", "scenario", "Year", "Event_Name"],
                                                     as_index=False)["Count"].sum())
    styles, order = scenario_styles(events)
    ncols = 2
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes, names):
        sub = per_sim[per_sim["Event_Name"] == name]
        for scen in order:
            m = sub[sub["scenario"] == scen].groupby("Year")["Count"].mean()
            if len(m):
                ax.plot(m.index, m.values, marker="o", ms=3, label=scen, **styles[scen])
        ax.set_title(name)
        ax.set_xlabel("Model year")
        ax.set_ylabel("Mean events per simulation")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STAGE 2: DRIVER
# =============================================================================

def make_outputs(expt_name, work_dir):
    out_dir = output_dir(work_dir, expt_name)
    inset_path = csv_path(out_dir, "All_Age_InsetChart")
    if not os.path.isfile(inset_path):
        print(f"No InsetChart csv at {inset_path}; nothing to post-process.")
        return

    years = analysis_years()
    inset = load_table(inset_path)
    print(f"\nScenarios found: {sorted(inset['scenario'].unique())}")
    print(f"Pairs (pickup states): {inset['pair_id'].nunique()}; "
          f"simulations: {inset['sim_id'].nunique()}")
    if BASELINE not in set(inset["scenario"]):
        print(f"WARNING: baseline arm {BASELINE} not found; paired outputs will be empty.")

    # ---- annual and window burden -------------------------------------------
    annual = annual_burden(inset)
    annual.to_csv(csv_path(out_dir, "Scenario_AnnualBurden"), index=False)
    incomplete = annual.loc[~annual["is_complete_year"], "sim_id"].nunique()
    if incomplete:
        print(f"WARNING: {incomplete} simulation(s) have incomplete model years "
              f"(partial runs); incidence is annualised.")

    pf_path = csv_path(out_dir, "PfPR_ClinicalIncidence_monthly")
    pf = load_table(pf_path) if os.path.isfile(pf_path) else None

    win = window_metrics(annual, years)
    metrics = list(METRICS_BASE)
    if pf is not None:
        u5 = u5_window_metrics(pf, years)
        if u5 is not None:
            win = win.merge(u5, on="sim_id", how="left")
            metrics += METRICS_U5
    metrics = [m for m in metrics if m in win.columns]
    win.to_csv(csv_path(out_dir, "Scenario_WindowMetrics"), index=False)

    # ---- pairing check and paired impact ------------------------------------
    chk = pairing_check(annual)
    if chk is not None:
        chk.to_csv(csv_path(out_dir, "PairingCheck"), index=False)

    paired = [p for p in (paired_impact(win, BASELINE, metrics),
                          paired_impact(win, LOWER_BOUND, metrics)) if p is not None]
    summary = None
    if paired:
        paired = pd.concat(paired, ignore_index=True)
        paired.to_csv(csv_path(out_dir, "Scenario_PairedImpact"), index=False)
        summary = summarise_impact(paired)
        summary.to_csv(csv_path(out_dir, "Scenario_ImpactSummary"), index=False)
        main_rows = summary[(summary["reference"] == BASELINE) &
                            (summary["metric"] == PRIMARY_METRIC)]
        print(f"\nPaired change in {PRIMARY_METRIC} vs {BASELINE}, "
              f"{years[0]}-{years[-1]} (median [{int(LOWER_Q*100)}-{int(HIGH_Q*100)}%]):")
        for r in main_rows.itertuples():
            print(f"  {r.scenario:<16} {r.median_rel_change*100:+7.1f}% "
                  f"[{r.q_low_rel_change*100:+.1f}, {r.q_high_rel_change*100:+.1f}]  "
                  f"n={r.n_pairs}")

    annual_summary = annual_paired_summary(annual, BASELINE)
    if annual_summary is not None:
        annual_summary.to_csv(csv_path(out_dir, "Scenario_AnnualImpact_summary"), index=False)

    # ---- effective use and threshold ----------------------------------------
    coverages = sorted(inset.loc[inset["scenario_kind"] == "ig2", "ig2_coverage"].dropna().unique())
    thr, resp = None, None
    if coverages:
        weights = age_weights_from_reports(pf, years)
        factors = None
        try:
            factors = effective_use_factors(weights)
        except (FileNotFoundError, ValueError) as exc:
            print(f"WARNING: effective use not computed with usage curves ({exc}); "
                  f"using retention only.")
        if factors is None:
            cycle = campaign_cycle_days()
            t = np.arange(cycle) + 0.5
            ret = float(np.exp(-(t / IG2_DISCARD_L) ** IG2_DISCARD_K).mean())
            factors = dict(cycle_days=cycle, retention=ret, age_multiplier=np.nan,
                           season_multiplier=np.nan, season_at_distribution=np.nan)
        eff = effective_use_table(coverages, factors)
        eff.to_csv(csv_path(out_dir, "IG2_EffectiveUse"), index=False)
        print(f"\nIG2 effective use (cycle {factors['cycle_days']} d): retention "
              f"{factors['retention']:.3f}, age multiplier {factors['age_multiplier']:.3f}, "
              f"season multiplier {factors['season_multiplier']:.3f}")
        print(eff[["ig2_coverage", "effective_use_time_avg",
                   "effective_use_at_distribution"]].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))

        if summary is not None:
            resp = coverage_response(summary, eff)
            if resp is not None:
                resp.to_csv(csv_path(out_dir, "IG2_CoverageResponse"), index=False)
                thr = minimum_coverage(resp)
                thr.to_csv(csv_path(out_dir, "IG2_MinimumCoverage"), index=False)
                print(f"\nMinimum IG2 coverage that maintains or reduces {PRIMARY_METRIC} "
                      f"vs {BASELINE} ({THRESHOLD_RULE} rule, margin {NONINFERIORITY_MARGIN:g}):")
                print(thr.to_string(index=False))

    # ---- events --------------------------------------------------------------
    ev_path = csv_path(out_dir, "EventCounts")
    events = load_table(ev_path) if os.path.isfile(ev_path) else None
    if events is not None:
        real = realized_distribution(events, annual)
        if real is not None:
            real.to_csv(csv_path(out_dir, "IG2_RealizedDistribution"), index=False)
            print("\nRealized IG2 distribution (nets issued per person in campaign years):")
            print(real.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- plots ----------------------------------------------------------------
    plot_inset_by_scenario(inset, [c for c in INSET_CHANNELS if c in inset.columns],
                           os.path.join(out_dir, f"InsetChart_{STEP}_byScenario.png"),
                           f"{expt_name}: monthly InsetChart summaries")
    plot_annual_impact(annual_summary, os.path.join(out_dir, f"AnnualImpact_{STEP}.png"),
                       f"{expt_name}: annual incidence vs {BASELINE}")
    plot_coverage_response(resp, thr, os.path.join(out_dir, f"IG2_CoverageResponse_{STEP}.png"),
                           f"{expt_name}: IG2 coverage response ({years[0]}-{years[-1]})")
    if pf is not None:
        plot_pfpr_by_agebin(pf, os.path.join(out_dir, f"PfPR_{STEP}_byAgeBin.png"),
                            f"{expt_name}: PfPR by age bin and scenario")
    if events is not None:
        plot_event_counts(events, os.path.join(out_dir, f"EventCounts_{STEP}_byYear.png"),
                          f"{expt_name}: events per simulation by model year")
    nu_path = csv_path(out_dir, "NetUse_monthly")
    if os.path.isfile(nu_path):
        nu = load_table(nu_path)
        plot_inset_by_scenario(nu, [c for c in NET_USE_CHANNELS if c in nu.columns],
                               os.path.join(out_dir, f"NetUse_{STEP}_byScenario.png"),
                               f"{expt_name}: Bednet_Using (diagnostic; overstates users)")
    print(f"\nOutputs written to {out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Analyze UWR IRS-vs-IG2 scenarios.")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Skip stage 1 (idmtools) and re-run post-processing on existing csvs.")
    p.add_argument("--experiment-id", default=None,
                   help="Override the experiment ID (applies to the first experiment).")
    return p.parse_args()


def main():
    args = parse_args()
    import manifest
    job_dir = manifest.job_directory
    work_dir = job_dir
    os.makedirs(work_dir, exist_ok=True)

    experiments = dict(EXPERIMENTS)
    if args.experiment_id:
        first = next(iter(experiments))
        experiments[first] = args.experiment_id

    if not args.skip_analysis:
        for name, exp_id in experiments.items():
            if not exp_id or not exp_id.strip() or exp_id.startswith("PUT-"):
                raise SystemExit(f"Set a valid scenario experiment ID for {name}.")
            run_analysis(name, exp_id.strip(), job_dir, work_dir)

    for name in experiments:
        make_outputs(name, work_dir)
    print(f"\nScenario analysis complete. Results: {work_dir}")


if __name__ == "__main__":
    main()