################################################################################
# Upper West Region EMOD malaria model -- PICKUP ANALYZER (2013-2015)
#
# Same structure as the project's reference analyzer: IAnalyzer classes run
# by AnalyzeManager over the experiments in `expts`, then the combined CSVs
# are read back for the comparison, summaries, QC and plots.
#
# ANALYZERS (files read per simulation)
#   InsetChartAnalyzer      output/InsetChart.json
#                           monthly means of the channels below; also the
#                           run length, to check the 1,095-day pickup
#   MonthlyPfPRAnalyzer     output/MalariaSummaryReport_Monthly_U5_{2013..2015}.json
#                           one file per pickup year, 30-day intervals, age bins
#                           [0.25, 5, 15, 25, 100]; every key is checked by name
#   EventCounterAnalyzer    output/ReportEventRecorder.csv
#                           monthly counts of treatment and SMC events
#   FilteredPrevalenceAnalyzer (only if USE_FILTERED_REPORT = True)
#                           output/ReportMalariaFiltered_U5_6to59m.json
#                           dedicated 6-59 month smear/PCR prevalence
#
# After AnalyzeManager, the experiment's simulation list is compared with
# what each analyzer returned, so every simulation that failed or lacked a
# file is listed with a reason (AnalyzeManager's partial_analyze_ok would
# otherwise skip them silently). state-01095.dtk is checked per simulation.
#
# PfPR is never derived from cases. If no prevalence is available the
# analyzer writes its QC report and stops with exit code 2.
#
# USAGE: set the experiment ID in `expts` below, then
#   python analyze_pickup_UWR.py
################################################################################

import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
import pandas as pd
from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation
import manifest

## For plotting
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


################################################################################
# CONFIGURATION -- must match run_pickup_UWR.py
################################################################################

# Set the pickup experiment ID here (name -> ID), as in the reference analyzer
expts = {
    'UpperWest_Region_pickup_2013_2015': '668685e1-e2d8-4a37-8a5f-2e1c12f45b8b'
}

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
OBSERVED_DATA = os.path.join(PROJECT_DIR, "inputs", "pcr_prevalence.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "analysis", "pickup")   # read by calibrate_pickup_UWR.py

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2015
PICKUP_YEARS = list(range(PICKUP_START_YEAR, PICKUP_END_YEAR + 1))
DAYS_PER_YEAR = 365
PICKUP_DAYS = len(PICKUP_YEARS) * DAYS_PER_YEAR          # 1,095
REPORT_INTERVAL_DAYS = 30
STEP = 'pickup'

SUMMARY_REPORT = "output/MalariaSummaryReport_Monthly_U5_{year}.json"
FILTERED_REPORT = "output/ReportMalariaFiltered_U5_6to59m.json"
USE_FILTERED_REPORT = False       # True once the pickup adds add_report_malaria_filtered()
STATE_FILE = "output/state-01095.dtk"

EXPECTED_AGE_BINS = [0.25, 5, 15, 25, 100]
CALIBRATION_AGE_GROUP = "0.25-5"
PFPR_KEY = "PfPR by Age Bin"
SUMMARY_KEYS = {                  # output column -> report key
    "population": "Average Population by Age Bin",
    "pfpr": PFPR_KEY,
    "clinical_incidence": "Annual Clinical Incidence by Age Bin",
    "severe_incidence": "Annual Severe Incidence by Age Bin",
    "new_infections": "New Infections by Age Bin",
}
REQUIRED_SUMMARY_KEYS = ["Average Population by Age Bin", "Annual Clinical Incidence by Age Bin"]

# "auto" = diagnostic-matched metric, filtered report first;
# "summary" = summary-report PfPR first for microscopy observations
PFPR_SOURCE = "auto"

channels_inset_chart = ['Statistical Population', 'New Clinical Cases',
                        'Adult Vectors', 'Infected', 'Daily EIR',
                        'Blood Smear Parasite Prevalence', 'PCR Parasite Prevalence']
EVENTS = ['Received_Treatment', 'Received_Severe_Treatment', 'Received_SMC']

# Simulation tags written by run_pickup_UWR.py -> analyzer column names.
# Pickup and burn-in values are kept apart: the pickup sets its own habitat.
TAG_MAP = {
    "run_number": "Run_Number",
    "burnin_sim_index": "burnin_sim_index",
    "burnin_x_Temporary_Larval_Habitat": "burnin_x_Temporary_Larval_Habitat",
    "burnin_scale_constant": "burnin_scale_constant",
    "burnin_scale_temp_rain": "burnin_scale_temp_rain",
    "burnin_scale_water_veg": "burnin_scale_water_veg",
    "pickup_x_Temporary_Larval_Habitat": "x_Temporary_Larval_Habitat",
    "pickup_scale_constant": "scale_constant",
    "pickup_scale_temp_rain": "scale_temp_rain",
    "pickup_scale_water_veg": "scale_water_veg",
}
REQUIRED_TAGS = ["run_number", "burnin_sim_index", "pickup_x_Temporary_Larval_Habitat",
                 "pickup_scale_constant", "pickup_scale_temp_rain", "pickup_scale_water_veg"]
PICKUP_PARAMS = ["pickup_x_Temporary_Larval_Habitat", "pickup_scale_constant",
                 "pickup_scale_temp_rain", "pickup_scale_water_veg"]

# AGE MATCH (months, [min, max) -- max_age 5 y = up to but not including 60 m)
#   Summary report bin 0.25-5 y      -> 3-60 months
#   Observed surveys, 6-59 months    -> 6-60 months
#   The summary bin therefore includes 3-5-month-olds the surveys exclude;
#   every comparison row records this as an approximate age match.
#   FILTERED_AGE_MONTHS must equal min_age_years/max_age_years of the
#   optional add_report_malaria_filtered() in run_pickup_UWR.py.
SUMMARY_AGE_MONTHS = (3, 60)
FILTERED_AGE_MONTHS = (6, 60)

# DIAGNOSTIC MATCH: which model prevalence may stand in for each survey test.
# Tried in order; the first metric present in the outputs is used.
#   Microscopy -> filtered report's Blood Smear Parasite Prevalence, else the
#                 summary report's PfPR (its positivity follows the config's
#                 Report_Detection_Threshold_* / smear-sensitivity settings,
#                 so it is microscopy-comparable only under those settings)
#   PCR        -> filtered report's PCR Parasite Prevalence only
#   RDT/other  -> no EMOD equivalent in these outputs: not compared
DIAGNOSTIC_METRICS = {
    "microscopy": [
        ("filtered_smear", "ReportMalariaFiltered 'Blood Smear Parasite Prevalence'",
         FILTERED_AGE_MONTHS, "microscopy vs EMOD blood-smear prevalence"),
        ("summary_pfpr", f"MalariaSummaryReport '{PFPR_KEY}'", SUMMARY_AGE_MONTHS,
         "microscopy vs EMOD summary-report PfPR; valid only if the report's "
         "detection settings represent microscopy"),
    ],
    "pcr": [
        ("filtered_pcr", "ReportMalariaFiltered 'PCR Parasite Prevalence'",
         FILTERED_AGE_MONTHS, "PCR vs EMOD PCR prevalence"),
    ],
}

# SURVEY TIMING: by default each observation is compared with the model's
# annual mean. Add a fieldwork window only after checking the survey report,
# e.g. SURVEY_WINDOWS = {"GDHS_2014": (9, 12)}
SURVEY_WINDOWS = {}

# Plot style: neutral greys, one accent colour
ACCENT = "#2F6F8F"
GREY = "#8A8F94"
TEXT = "#2B2B2B"

MONTH_OFFSET = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def month_of_doy(doy):
    return int(np.searchsorted(MONTH_OFFSET, int(doy) - 1, side="right"))


def age_labels(bins):
    edges = [0] + list(bins)
    return [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(bins))]


################################################################################
# ANALYZERS
################################################################################

def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def stamp_tags(df, simulation):
    """Tag columns (pickup_* and burnin_* kept apart) plus simulation_id."""
    for col, tag in TAG_MAP.items():
        df[col] = _num(simulation.tags.get(tag))
    df.insert(0, "simulation_id", str(simulation.uid))
    return df


def write_per_sim(df, working_dir, expt_name, subdir, simulation, **marker):
    """Write one simulation's rows to disk; return only a small marker (memory)."""
    out_dir = os.path.join(working_dir, expt_name, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{simulation.uid}.csv")
    df.to_csv(path, index=False)
    return dict(sim_id=str(simulation.uid), path=path, n_rows=len(df), **marker)


def merge_per_sim(all_data, out_path, per_sim_dir):
    """Stream per-simulation CSVs into one file (columns aligned to the first)."""
    header, n_rows = None, 0
    with open(out_path, "w") as dst:
        for _, result in all_data.items():
            if not result or not result.get("n_rows"):
                continue
            with open(result["path"]) as src:
                sim_header = src.readline()
                if header is None:
                    header = sim_header
                    dst.write(header)
                elif sim_header != header:
                    pd.read_csv(result["path"]).reindex(columns=header.strip().split(",")) \
                        .to_csv(dst, header=False, index=False)
                    n_rows += result["n_rows"]
                    continue
                shutil.copyfileobj(src, dst)
            n_rows += result["n_rows"]
    print(f"Wrote {n_rows} rows to {out_path}")
    shutil.rmtree(per_sim_dir, ignore_errors=True)


def save_markers(all_data, path):
    """Per-simulation flags from map(), used by the QC step."""
    rows = [{k: v for k, v in m.items() if k not in ("path",)} for m in all_data.values() if m]
    pd.DataFrame(rows).to_csv(path, index=False)


class InsetChartAnalyzer(IAnalyzer):
    """Monthly means of InsetChart channels; also records the run length."""

    def __init__(self, expt_name, channels=None, working_dir=".", start_year=0):
        super(InsetChartAnalyzer, self).__init__(working_dir=working_dir,
                                                 filenames=["output/InsetChart.json"])
        self.inset_channels = channels or ['Statistical Population', 'New Clinical Cases']
        self.expt_name = expt_name
        self.start_year = start_year

    def map(self, data, simulation: Simulation):
        channels = data[self.filenames[0]]['Channels']
        present = [c for c in self.inset_channels if c in channels]
        simdata = pd.DataFrame({c: channels[c]['Data'] for c in present})
        n_days = len(simdata)

        simdata['date'] = pd.Timestamp(year=self.start_year, month=1, day=1) + \
            pd.to_timedelta(simdata.index, unit='D')
        monthly = simdata.set_index('date')[present].resample('MS').mean().reset_index()
        monthly = stamp_tags(monthly, simulation)
        return write_per_sim(monthly, self.working_dir, self.expt_name, '_per_sim_insetchart',
                             simulation, n_days=n_days,
                             missing_channels=";".join(c for c in self.inset_channels if c not in channels))

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("No InsetChart data returned.")
            return
        out = os.path.join(self.working_dir, self.expt_name)
        save_markers(all_data, os.path.join(out, "_markers_insetchart.csv"))
        merge_per_sim(all_data, os.path.join(out, f'All_Age_InsetChart_{STEP}.csv'),
                      os.path.join(out, '_per_sim_insetchart'))


class MonthlyPfPRAnalyzer(IAnalyzer):
    """
    Summary reports for 2013-2015, one file per year. Each 30-day interval
    is labelled with its calendar year (from the file) and month (from the
    interval midpoint); the short trailing interval (days 361-365) is dropped.
    """

    def __init__(self, expt_name, report_years, working_dir='./'):
        self.report_years = list(report_years)
        filenames = [SUMMARY_REPORT.format(year=y) for y in self.report_years]
        self.year_by_filename = dict(zip(filenames, self.report_years))
        super(MonthlyPfPRAnalyzer, self).__init__(working_dir=working_dir, filenames=filenames)
        self.expt_name = expt_name

    def map(self, data, simulation: Simulation):
        parts, years_ok, pfpr_flags, issues, age_bins_seen, keys_seen = [], [], [], [], None, None

        for fname in self.filenames:
            year = self.year_by_filename[fname]
            report = data.get(fname)
            if not report:
                issues.append(f"missing {fname}")
                continue
            try:
                by_age = report['DataByTimeAndAgeBins']
                bins = report['Metadata']['Age Bins']
                missing = [k for k in REQUIRED_SUMMARY_KEYS if k not in by_age]
                if missing:
                    raise KeyError(f"keys {missing}")
            except (KeyError, TypeError) as err:
                issues.append(f"malformed {fname}: {err}")
                continue

            age_bins_seen, keys_seen = bins, sorted(by_age.keys())
            pfpr_flags.append(PFPR_KEY in by_age)
            labels = age_labels(bins)
            for k in range(len(by_age['Average Population by Age Bin'])):
                start = k * REPORT_INTERVAL_DAYS + 1
                end = min((k + 1) * REPORT_INTERVAL_DAYS, DAYS_PER_YEAR)
                if end - start + 1 < REPORT_INTERVAL_DAYS:
                    continue
                month = month_of_doy((start + end) // 2)
                for a, label in enumerate(labels):
                    row = {'year': year, 'month': month, 'report_interval': k + 1,
                           'interval_start_doy': start, 'interval_end_doy': end,
                           'pickup_day_start': (year - PICKUP_START_YEAR) * DAYS_PER_YEAR + start,
                           'age_group': label}
                    for col, key in SUMMARY_KEYS.items():
                        row[col] = by_age[key][k][a] if key in by_age else np.nan
                    parts.append(row)
            years_ok.append(year)

        marker = dict(years_found=";".join(map(str, years_ok)),
                      has_all_years=len(years_ok) == len(self.report_years),
                      has_pfpr=bool(pfpr_flags) and all(pfpr_flags),
                      age_bins=json.dumps(age_bins_seen),
                      age_bins_ok=age_bins_seen is not None and
                      [float(b) for b in age_bins_seen] == [float(b) for b in EXPECTED_AGE_BINS],
                      report_keys=json.dumps(keys_seen), issues="; ".join(issues))
        if not parts:
            return dict(sim_id=str(simulation.uid), n_rows=0, **marker)

        adf = pd.DataFrame(parts)
        days = adf['interval_end_doy'] - adf['interval_start_doy'] + 1
        # Derived, not reported: annualised incidence x population x interval length
        adf['clinical_cases_derived'] = adf['clinical_incidence'] * adf['population'] * days / DAYS_PER_YEAR
        adf = stamp_tags(adf, simulation)
        return write_per_sim(adf, self.working_dir, self.expt_name, '_per_sim_pfpr', simulation, **marker)

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("\nWarning: No summary-report data returned.")
            return
        out = os.path.join(self.working_dir, self.expt_name)
        save_markers(all_data, os.path.join(out, "_markers_pfpr.csv"))
        merge_per_sim(all_data, os.path.join(out, 'pickup_calibration_results.csv'),
                      os.path.join(out, '_per_sim_pfpr'))


class EventCounterAnalyzer(IAnalyzer):
    """Monthly counts of campaign events from ReportEventRecorder.csv."""

    def __init__(self, expt_name, events, working_dir='./'):
        super(EventCounterAnalyzer, self).__init__(working_dir=working_dir,
                                                   filenames=["output/ReportEventRecorder.csv"])
        self.events = events
        self.expt_name = expt_name

    def map(self, data, simulation: Simulation):
        raw = data[self.filenames[0]]
        if isinstance(raw, (bytes, bytearray)):
            import io
            raw = pd.read_csv(io.BytesIO(raw))
        raw = raw[raw['Event_Name'].isin(self.events)]
        day = raw['Time'].astype(float).astype(int) + 1
        raw = raw.assign(year=PICKUP_START_YEAR + (day - 1) // DAYS_PER_YEAR,
                         month=((day - 1) % DAYS_PER_YEAR + 1).map(month_of_doy))
        grid = pd.MultiIndex.from_product([PICKUP_YEARS, range(1, 13), self.events],
                                          names=['year', 'month', 'event'])
        counts = raw.groupby(['year', 'month', 'Event_Name']).size()
        counts.index.names = ['year', 'month', 'event']
        counts = counts.reindex(grid, fill_value=0).rename('count').reset_index()
        counts = stamp_tags(counts, simulation)
        return write_per_sim(counts, self.working_dir, self.expt_name, '_per_sim_events', simulation)

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("\nWarning: No event data returned.")
            return
        out = os.path.join(self.working_dir, self.expt_name)
        merge_per_sim(all_data, os.path.join(out, 'pickup_event_counts.csv'),
                      os.path.join(out, '_per_sim_events'))


class FilteredPrevalenceAnalyzer(IAnalyzer):
    """Smear/PCR prevalence from the age-filtered report, per 30-day interval."""

    def __init__(self, expt_name, working_dir='./'):
        super(FilteredPrevalenceAnalyzer, self).__init__(working_dir=working_dir,
                                                         filenames=[FILTERED_REPORT])
        self.expt_name = expt_name

    def map(self, data, simulation: Simulation):
        channels = data[self.filenames[0]]['Channels']
        daily = pd.DataFrame({
            'smear': channels.get('Blood Smear Parasite Prevalence', {}).get('Data'),
            'pcr': channels.get('PCR Parasite Prevalence', {}).get('Data'),
        })
        day = np.arange(1, len(daily) + 1)
        daily['year'] = PICKUP_START_YEAR + (day - 1) // DAYS_PER_YEAR
        doy = (day - 1) % DAYS_PER_YEAR + 1
        daily['report_interval'] = (doy - 1) // REPORT_INTERVAL_DAYS + 1
        daily = daily[daily['report_interval'] <= 12]
        agg = daily.groupby(['year', 'report_interval'])[['smear', 'pcr']].mean().reset_index()
        start = (agg['report_interval'] - 1) * REPORT_INTERVAL_DAYS + 1
        agg['month'] = ((2 * start + REPORT_INTERVAL_DAYS - 1) // 2).map(month_of_doy)
        agg['age_min_months'], agg['age_max_months'] = FILTERED_AGE_MONTHS
        agg = stamp_tags(agg, simulation)
        return write_per_sim(agg, self.working_dir, self.expt_name, '_per_sim_filtered', simulation)

    def reduce(self, all_data):
        if len(all_data) == 0:
            print("\nWarning: No filtered-report data returned.")
            return
        out = os.path.join(self.working_dir, self.expt_name)
        merge_per_sim(all_data, os.path.join(out, 'pickup_filtered_prevalence.csv'),
                      os.path.join(out, '_per_sim_filtered'))


################################################################################
# OBSERVED DATA
################################################################################

def load_observed(path):
    obs = pd.read_csv(path)
    obs.columns = [c.strip() for c in obs.columns]
    required = ["survey_year", "region", "age_min_months", "age_max_months",
                "prevalence", "source", "test_method"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    obs = obs[obs["region"].str.strip().str.lower() == "upper west"].copy()
    obs["source"] = obs["source"].astype(str).str.strip()
    obs["test_method"] = obs["test_method"].astype(str).str.strip()
    obs["in_pickup"] = obs["survey_year"].isin(PICKUP_YEARS)
    return obs


def age_match(model_months, obs_min, obs_max):
    """Observed max is inclusive (59 -> up to 60)."""
    lo, hi = model_months
    exact = lo == int(obs_min) and hi == int(obs_max) + 1
    text = (f"model {lo}-{hi} m vs observed {int(obs_min)}-{int(obs_max)} m "
            f"({'exact' if exact else 'approximate'})")
    return ("exact" if exact else "approximate"), text


################################################################################
# MODEL PREVALENCE METRICS
################################################################################

PARAM_COLS = ["run_number", "burnin_sim_index"] + PICKUP_PARAMS


def sim_params(frame):
    """One row of parameters per simulation (they are constant within a sim)."""
    return frame.groupby("simulation_id")[PARAM_COLS].first().reset_index()


def per_sim_annual(frame, value_cols):
    """
    Annual value per simulation and year = mean of that year's twelve 30-day
    intervals (days 1-360). Grouped by simulation and year only; parameters
    are attached afterwards, so float tag values cannot split or merge groups.
    """
    annual = frame.groupby(["simulation_id", "year"])[value_cols].mean().reset_index()
    return annual.merge(sim_params(frame), on="simulation_id", how="left")


def build_metric_tables(u5, filtered):
    """{metric_id: DataFrame(simulation_id, year, month, value, population, params)}"""
    tables = {}
    if u5["pfpr"].notna().any():
        tables["summary_pfpr"] = u5.rename(columns={"pfpr": "value"})[
            ["simulation_id", "year", "month", "value", "population"] + PARAM_COLS]
    if filtered is not None:
        params = sim_params(u5)
        f = filtered.merge(params, on="simulation_id", how="inner")
        # Population of the filtered age range is not in that report; the
        # summary U5 population is attached for reference only.
        pop = u5.groupby(["simulation_id", "year", "month"])["population"].mean().reset_index()
        f = f.merge(pop, on=["simulation_id", "year", "month"], how="left")
        for metric, col in (("filtered_smear", "smear"), ("filtered_pcr", "pcr")):
            if f[col].notna().any():
                tables[metric] = f.rename(columns={col: "value"})[
                    ["simulation_id", "year", "month", "value", "population"] + PARAM_COLS]
    return tables


def choose_metric(test_method, tables, prefer):
    """First compatible metric for a survey test, or (None, reason)."""
    options = DIAGNOSTIC_METRICS.get(test_method.strip().lower())
    if not options:
        return None, f"no EMOD equivalent for test method '{test_method}' in these outputs"
    if prefer == "summary":
        options = sorted(options, key=lambda o: o[0] != "summary_pfpr")
    for option in options:
        if option[0] in tables:
            return option, None
    return None, (f"test method '{test_method}' needs one of "
                  f"{[o[1] for o in options]}, none present in the outputs")


################################################################################
# OUTPUT TABLES
################################################################################

def build_comparison(tables, obs, prefer, windows):
    """One row per (simulation, in-pickup observation). Replicates kept."""
    rows, status = [], {}
    for o in obs.itertuples():
        key = (int(o.survey_year), o.source)
        if not o.in_pickup:
            status[key] = "not compared: outside the 2013-2015 pickup, kept for later stages"
            continue
        option, reason = choose_metric(o.test_method, tables, prefer)
        if option is None:
            status[key] = f"not compared: {reason}"
            continue
        metric_id, metric_label, model_age, diag_note = option
        table = tables[metric_id]
        year_df = table[table["year"] == o.survey_year]

        m0, m1 = windows.get(o.source, (1, 12))
        period = ("annual mean of twelve 30-day intervals (days 1-360)" if (m0, m1) == (1, 12)
                  else f"months {m0}-{m1} (survey window given with --survey-window)")
        window = year_df[(year_df["month"] >= m0) & (year_df["month"] <= m1)]

        model = window.groupby("simulation_id").agg(
            model_pfpr=("value", "mean"), model_population=("population", "mean")).reset_index()
        annual = year_df.groupby("simulation_id")["value"].mean().rename("model_pfpr_annual_mean")
        model = model.merge(annual, on="simulation_id").merge(sim_params(table), on="simulation_id")

        match, match_text = age_match(model_age, o.age_min_months, o.age_max_months)
        model = model.assign(
            model_year=o.survey_year, model_months=f"{m0}-{m1}", model_period=period,
            model_metric=metric_label,
            model_age_min_months=model_age[0], model_age_max_months=model_age[1],
            observed_year=o.survey_year, observed_pfpr=o.prevalence,
            age_min_months=o.age_min_months, age_max_months=o.age_max_months,
            source=o.source, test_method=o.test_method,
            age_match=match, age_match_note=match_text, diagnostic_note=diag_note)
        model["difference"] = model["model_pfpr"] - o.prevalence
        model["absolute_difference"] = model["difference"].abs()
        model["squared_error"] = model["difference"] ** 2
        rows.append(model)
        status[key] = (f"compared: {metric_label}; {period}; age {match_text}; {diag_note}")
    comparison = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return comparison, status


def build_parameter_summary(u5, comparison, pfpr_col):
    value_cols = ["clinical_incidence"] + ([pfpr_col] if pfpr_col else [])
    annual = per_sim_annual(u5, value_cols)
    agg = dict(n_simulations=("simulation_id", "nunique"),
               clinical_incidence_mean=("clinical_incidence", "mean"),
               clinical_incidence_sd=("clinical_incidence", "std"))
    if pfpr_col:
        agg.update(pfpr_mean=(pfpr_col, "mean"), pfpr_median=(pfpr_col, "median"),
                   pfpr_sd=(pfpr_col, "std"), pfpr_min=(pfpr_col, "min"), pfpr_max=(pfpr_col, "max"))
    summary = annual.groupby(PICKUP_PARAMS + ["year"], dropna=False).agg(**agg).reset_index()

    if not comparison.empty:
        comp = (comparison.groupby(PICKUP_PARAMS + ["observed_year"], dropna=False)
                .agg(comparison_model_pfpr_mean=("model_pfpr", "mean"),
                     comparison_model_months=("model_months", "first"),
                     comparison_model_metric=("model_metric", "first"),
                     observed_pfpr=("observed_pfpr", "first"),
                     observed_source=("source", "first"),
                     age_match=("age_match", "first"),
                     mean_difference=("difference", "mean"),
                     mean_absolute_difference=("absolute_difference", "mean"),
                     rmse=("squared_error", lambda v: float(np.sqrt(v.mean()))))
                .reset_index().rename(columns={"observed_year": "year"}))
        summary = summary.merge(comp, on=PICKUP_PARAMS + ["year"], how="left")
    return summary


################################################################################
# PLOTS
################################################################################

MIX_COLS = ["pickup_scale_constant", "pickup_scale_temp_rain", "pickup_scale_water_veg"]


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C8CCD0")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color="#ECEEF0", linewidth=0.8)
    ax.set_axisbelow(True)


def _mix_label(mix):
    return f"habitat mix {mix[0]:g}/{mix[1]:g}/{mix[2]:g} (constant/temp_rain/water_veg)"


def _mixes(frame):
    return [tuple(m) for m in frame[MIX_COLS].drop_duplicates().sort_values(MIX_COLS).values]


def _mix_mask(frame, mix):
    return np.logical_and.reduce([np.isclose(frame[c].astype(float), v) for c, v in zip(MIX_COLS, mix)])


def _combo_label(row):
    return (f"xTLH {row['pickup_x_Temporary_Larval_Habitat']:.3g} | "
            f"{row['pickup_scale_constant']:g}/{row['pickup_scale_temp_rain']:g}/"
            f"{row['pickup_scale_water_veg']:g}")


def _shades(n):
    return [plt.cm.Blues(v) for v in np.linspace(0.35, 0.9, max(n, 1))]


def plot_outcomes(u5, pfpr_col, path):
    u5 = u5.copy()
    u5["date"] = pd.to_datetime(dict(year=u5["year"], month=u5["month"], day=15))
    panels = [("clinical_incidence", "Clinical incidence, 0.25-5 y", "cases per person-year")]
    if pfpr_col:
        panels.insert(0, (pfpr_col, "PfPR (summary report), 0.25-5 y", "PfPR"))

    combos = u5[PICKUP_PARAMS].drop_duplicates().sort_values(PICKUP_PARAMS).reset_index(drop=True)
    colours = _shades(len(combos))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.6 * len(panels)), squeeze=False)

    for ax, (col, title, ylabel) in zip(axes[:, 0], panels):
        for c, combo in combos.iterrows():
            mask = np.logical_and.reduce([np.isclose(u5[p].astype(float), combo[p]) for p in PICKUP_PARAMS])
            g = u5[mask].groupby("date")[col]
            mean, lo, hi = g.mean(), g.min(), g.max()
            ax.fill_between(mean.index, lo, hi, color=colours[c], alpha=0.12, linewidth=0)
            ax.plot(mean.index, mean.values, color=colours[c], linewidth=1.6, label=_combo_label(combo))
        _style(ax, title, ylabel)
    axes[-1, 0].legend(title="xTLH | habitat mix", fontsize=8, title_fontsize=8, frameon=False,
                       loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Pickup outcomes 2013-2015 (line: mean across replicates, band: min-max)",
                 fontsize=13, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(u5, pfpr_col, path):
    """xTLH sensitivity, one row of panels per fixed habitat mix."""
    value_cols = ["clinical_incidence"] + ([pfpr_col] if pfpr_col else [])
    annual = per_sim_annual(u5, value_cols)
    panels = [("clinical_incidence", "Annual mean clinical incidence, 0.25-5 y", "cases per person-year")]
    if pfpr_col:
        panels.insert(0, (pfpr_col, "Annual mean PfPR (summary report), 0.25-5 y", "PfPR"))

    mixes = _mixes(annual)
    colours = _shades(len(PICKUP_YEARS))
    fig, axes = plt.subplots(len(mixes), len(panels), figsize=(6 * len(panels), 4.3 * len(mixes)),
                             squeeze=False)
    for i, mix in enumerate(mixes):
        mix_df = annual[_mix_mask(annual, mix)]
        for j, (col, title, ylabel) in enumerate(panels):
            ax = axes[i, j]
            for y, year in enumerate(PICKUP_YEARS):
                d = mix_df[mix_df["year"] == year]
                ax.scatter(d["pickup_x_Temporary_Larval_Habitat"], d[col], s=8, alpha=0.25, color=colours[y])
                m = d.groupby("pickup_x_Temporary_Larval_Habitat")[col].mean()
                ax.plot(m.index, m.values, color=colours[y], linewidth=1.8, marker="o",
                        markersize=4, label=str(year))
            ax.set_xscale("log")
            ticks = sorted(annual["pickup_x_Temporary_Larval_Habitat"].unique())
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.3g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("Pickup x_Temporary_Larval_Habitat (log scale)", fontsize=10, color=TEXT)
            _style(ax, f"{title}\n{_mix_label(mix)} fixed", ylabel)
        axes[i, -1].legend(title="Year", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_model(comparison, path):
    for (obs_year, source), comp in comparison.groupby(["observed_year", "source"]):
        o = comp.iloc[0]
        xs = sorted(comp["pickup_x_Temporary_Larval_Habitat"].unique())
        pos = {x: i for i, x in enumerate(xs)}
        mixes = _mixes(comp)
        shades = [TEXT] if len(mixes) == 1 else _shades(len(mixes))

        fig, ax = plt.subplots(figsize=(8.5, 5))
        rng = np.random.default_rng(0)
        for k, mix in enumerate(mixes):
            d = comp[_mix_mask(comp, mix)]
            jitter = rng.uniform(-0.12, 0.12, len(d))
            ax.scatter(d["pickup_x_Temporary_Larval_Habitat"].map(pos) + jitter, d["model_pfpr"],
                       s=9, color=GREY, alpha=0.35, label="Model replicates" if k == 0 else None)
            means = d.groupby("pickup_x_Temporary_Larval_Habitat")["model_pfpr"].mean()
            ax.plot([pos[x] for x in means.index], means.values, color=shades[k], marker="o",
                    linewidth=1.4, markersize=5, label=f"Model mean, {_mix_label(mix)}")

        values = np.append(comp["model_pfpr"].values, o["observed_pfpr"])
        pad = 0.12 * (values.max() - values.min() or 0.1)
        ax.set_ylim(values.min() - pad, values.max() + 2 * pad)
        ax.axhline(o["observed_pfpr"], color=ACCENT, linewidth=2)
        ax.text(-0.4, o["observed_pfpr"] + pad * 0.35,
                f"Observed {o['observed_pfpr']:.1%}  ({source}, {o['test_method']}, "
                f"{int(o['age_min_months'])}-{int(o['age_max_months'])} months)",
                color=ACCENT, fontsize=9, va="bottom")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.3g}" for x in xs])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_xlabel("Pickup x_Temporary_Larval_Habitat", fontsize=10, color=TEXT)
        _style(ax, f"Model vs observed parasite prevalence, {int(obs_year)}", "Prevalence")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        fig.text(0.01, -0.02,
                 f"Model: {o['model_metric']}, {o['model_period']}.\n"
                 f"Age: {o['age_match_note']}.  Diagnostic: {o['diagnostic_note']}.",
                 fontsize=8, color=GREY, ha="left", va="top")
        fig.tight_layout()
        suffix = "" if comparison["source"].nunique() == 1 else f"_{source}"
        fig.savefig(path.replace(".png", f"{suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)



################################################################################
# QC -- compare the experiment's simulations with what the analyzers returned
################################################################################

def _status_name(status):
    return getattr(status, "name", str(status)).upper().replace("ENTITYSTATUS.", "")


def _find_dirs_by_id(job_directory, sim_ids, max_depth=5):
    wanted, found = set(sim_ids), {}
    root_depth = job_directory.rstrip(os.sep).count(os.sep)
    for root, dirs, _ in os.walk(job_directory):
        for d in dirs:
            if d in wanted:
                found[d] = os.path.join(root, d)
        if root.count(os.sep) - root_depth >= max_depth or len(found) == len(wanted):
            dirs[:] = []
    return found


def list_simulations(platform, exp_id, job_directory):
    """Every simulation in the experiment: id, status, tags, state-file check."""
    from idmtools.core import ItemType
    sims = platform.get_children(exp_id, ItemType.EXPERIMENT, force=True)
    rows, dirs = [], {}
    for sim in sims:
        sim_dir = None
        for getter in (lambda: platform.get_directory(sim),
                       lambda: platform.get_directory_by_id(sim.id, ItemType.SIMULATION)):
            try:
                sim_dir = str(getter())
                break
            except Exception:
                continue
        dirs[str(sim.id)] = sim_dir
        tags = dict(sim.tags or {})
        rows.append({"simulation_id": str(sim.id), "status": _status_name(sim.status),
                     "missing_tags": ";".join(c for c in REQUIRED_TAGS if np.isnan(_num(tags.get(TAG_MAP[c]))))})
    missing = [s for s, d in dirs.items() if not d or not os.path.isdir(d)]
    if missing:
        dirs.update(_find_dirs_by_id(job_directory, missing))
    df = pd.DataFrame(rows)
    df["has_state_01095"] = [bool(dirs.get(s)) and os.path.isfile(os.path.join(dirs[s], STATE_FILE))
                             for s in df["simulation_id"]]
    return df


def build_qc_table(sims, out):
    """One row per simulation with every check; `included` = usable for calibration."""
    pf = pd.read_csv(os.path.join(out, "_markers_pfpr.csv")) if \
        os.path.isfile(os.path.join(out, "_markers_pfpr.csv")) else pd.DataFrame(columns=["sim_id"])
    ic = pd.read_csv(os.path.join(out, "_markers_insetchart.csv")) if \
        os.path.isfile(os.path.join(out, "_markers_insetchart.csv")) else pd.DataFrame(columns=["sim_id"])
    qc = sims.merge(pf.rename(columns={"sim_id": "simulation_id"}), on="simulation_id", how="left")
    qc = qc.merge(ic[["sim_id", "n_days"]].rename(columns={"sim_id": "simulation_id"}),
                  on="simulation_id", how="left") if "n_days" in ic else qc.assign(n_days=np.nan)

    for col, default in (("has_all_years", False), ("has_pfpr", False), ("age_bins_ok", True),
                         ("years_found", ""), ("issues", ""), ("age_bins", ""), ("report_keys", None)):
        if col not in qc.columns:
            qc[col] = default
    qc["has_reports"] = qc["has_all_years"].fillna(False).astype(bool)
    qc["has_pfpr"] = qc["has_pfpr"].fillna(False).astype(bool)
    qc["age_bins_ok"] = qc["age_bins_ok"].fillna(True).astype(bool)
    qc["duration_ok"] = qc["n_days"].isna() | (qc["n_days"] == PICKUP_DAYS)

    reasons = []
    for r in qc.itertuples():
        why = []
        if r.status != "SUCCEEDED":
            why.append(f"status {r.status}")
        if r.status == "SUCCEEDED" and not r.has_reports:
            why.append(f"summary reports incomplete (found years: {getattr(r, 'years_found', '') or 'none'})")
        if isinstance(getattr(r, "issues", None), str) and r.issues:
            why.append(r.issues)
        if r.missing_tags:
            why.append(f"missing tags {r.missing_tags}")
        if not r.age_bins_ok:
            why.append(f"unexpected age bins {r.age_bins}")
        if not r.duration_ok:
            why.append(f"InsetChart has {int(r.n_days)} days, expected {PICKUP_DAYS}")
        if r.status == "SUCCEEDED" and pd.isna(r.n_days):
            why.append("InsetChart not returned")
        if not r.has_state_01095:
            why.append(f"missing {STATE_FILE}")
        reasons.append("; ".join(why))
    qc["issues"] = reasons
    qc["included"] = (qc["status"] == "SUCCEEDED") & qc["has_reports"] & (qc["missing_tags"] == "") & qc["age_bins_ok"]
    return qc


def write_qc(path, qc, obs, metrics_found, comparison_status, report_keys):
    dup = [k for k, v in Counter(qc["simulation_id"]).items() if v > 1]
    ok = qc["status"] == "SUCCEEDED"
    lines = [
        "UWR pickup analyzer QC",
        "=" * 50,
        f"Total simulations discovered      {len(qc)}",
        f"Completed simulations             {int(ok.sum())}",
        f"Failed / not completed            {int((~ok).sum())}",
        f"Simulations with required reports {int(qc['has_reports'].sum())}",
        f"Simulations missing reports       {int((ok & ~qc['has_reports']).sum())}",
        f"Simulations with PfPR available   {int(qc['has_pfpr'].sum())}",
        f"Simulations without PfPR          {int((ok & ~qc['has_pfpr']).sum())}",
        f"Simulations with state-01095.dtk  {int(qc['has_state_01095'].sum())}",
        f"Duplicate simulations             {len(dup)}",
        f"Simulations with missing tags     {int((qc['missing_tags'] != '').sum())}",
        f"Unexpected pickup duration        {int((~qc['duration_ok']).sum())}",
        f"Unexpected age bins               {int((~qc['age_bins_ok']).sum())}",
        f"Included in outputs               {int(qc['included'].sum())}",
        "",
        f"Model prevalence metrics found: {metrics_found or 'NONE'}",
        "",
        "Observed prevalence points (Upper West):",
    ]
    for o in obs.itertuples():
        status = comparison_status.get((int(o.survey_year), o.source), "not compared")
        lines.append(f"  {o.survey_year} {o.source:<10} {o.test_method:<11} {o.prevalence:.3f}")
        lines.append(f"      -> {status}")
    if report_keys:
        lines += ["", f"Summary-report keys found: {report_keys}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))


def postprocess(sims, out):
    """QC, comparison, summaries and plots from the analyzers' CSVs."""
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    # ------------------------------------------------------------------
    # QC table: every simulation, every check, every reason
    # ------------------------------------------------------------------
    qc = build_qc_table(sims, out)
    qc.to_csv(os.path.join(out, "pickup_simulation_qc.csv"), index=False)
    qc.loc[~qc["included"], ["simulation_id", "status", "issues"]] \
        .to_csv(os.path.join(out, "pickup_skipped_simulations.csv"), index=False)
    report_keys = next(iter(qc["report_keys"].dropna()), None)

    obs = load_observed(OBSERVED_DATA)
    qc_path = os.path.join(out, "pickup_analyzer_qc.txt")
    results_path = os.path.join(out, "pickup_calibration_results.csv")
    if not os.path.isfile(results_path) or not qc["included"].any():
        write_qc(qc_path, qc, obs, [], {}, report_keys)
        sys.exit(f"No usable simulations. See {os.path.join(out, 'pickup_skipped_simulations.csv')}.")

    # ------------------------------------------------------------------
    # Keep included simulations only; U5 rows for calibration
    # ------------------------------------------------------------------
    data = pd.read_csv(results_path)
    data = data[data["simulation_id"].isin(qc.loc[qc["included"], "simulation_id"])]
    data.to_csv(results_path, index=False)

    if CALIBRATION_AGE_GROUP not in set(data["age_group"]):
        write_qc(qc_path, qc, obs, [], {}, report_keys)
        sys.exit(f"Age group {CALIBRATION_AGE_GROUP} not in the reports "
                 f"(found {sorted(set(data['age_group']))}). Not substituting another age group.")
    u5 = data[data["age_group"] == CALIBRATION_AGE_GROUP].copy()

    filtered = None
    filtered_path = os.path.join(out, "pickup_filtered_prevalence.csv")
    if USE_FILTERED_REPORT and os.path.isfile(filtered_path):
        filtered = pd.read_csv(filtered_path)[["simulation_id", "year", "report_interval",
                                               "month", "smear", "pcr"]]

    # ------------------------------------------------------------------
    # Comparison with observed prevalence (2014 only), summaries, plots
    # ------------------------------------------------------------------
    tables = build_metric_tables(u5, filtered)
    pfpr_col = "pfpr" if "summary_pfpr" in tables else None
    comparison, status = build_comparison(tables, obs, PFPR_SOURCE, SURVEY_WINDOWS)
    if not comparison.empty:
        comparison.to_csv(os.path.join(out, "pickup_pfpr_comparison.csv"), index=False)

    summary = build_parameter_summary(u5, comparison, pfpr_col)
    summary.to_csv(os.path.join(out, "pickup_summary_by_parameter.csv"), index=False)

    plot_outcomes(u5, pfpr_col, os.path.join(out, "plots", "pickup_outcomes_2013_2015.png"))
    plot_sensitivity(u5, pfpr_col, os.path.join(out, "plots", "pickup_parameter_sensitivity.png"))
    if not comparison.empty:
        plot_observed_vs_model(comparison, os.path.join(out, "plots", "pickup_pfpr_observed_vs_model.png"))

    write_qc(qc_path, qc, obs, sorted(tables), status, report_keys)
    for f in ("_markers_pfpr.csv", "_markers_insetchart.csv"):
        if os.path.isfile(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    print(f"\nOutputs written to {out}")
    print(f"Skipped simulations: {int((~qc['included']).sum())} "
          f"(listed in {os.path.join(out, 'pickup_skipped_simulations.csv')})")

    if not tables:
        print("\nERROR: no model prevalence in the pickup outputs. Prevalence was not derived from cases.\n"
              "Add a prevalence reporter to run_pickup_UWR.py, e.g.\n"
              "    add_report_malaria_filtered(task, manifest, start_day=1, end_day=1095,\n"
              "        min_age_years=0.5, max_age_years=5, filename_suffix='U5_6to59m')\n"
              "and set USE_FILTERED_REPORT = True here.")
        return 2
    if comparison.empty:
        print("\nNo observation could be compared -- see the observed-points section of the QC report.")
    return 0


################################################################################
# RUN
################################################################################

if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    expt_name, exp_id = next(iter(expts.items()))
    if not exp_id.strip():
        sys.exit("Set the pickup experiment ID in `expts` at the top of analyze_pickup_UWR.py.")

    jdir = manifest.job_directory
    wdir = os.path.dirname(OUTPUT_DIR)
    out = OUTPUT_DIR
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)

    # AnalyzeManager writes into working_dir/expt_name; point that at OUTPUT_DIR
    analyzers_pickup = [
        InsetChartAnalyzer(expt_name=os.path.basename(out), channels=channels_inset_chart,
                           start_year=PICKUP_START_YEAR, working_dir=wdir),
        MonthlyPfPRAnalyzer(expt_name=os.path.basename(out), report_years=PICKUP_YEARS,
                            working_dir=wdir),
        EventCounterAnalyzer(expt_name=os.path.basename(out), events=EVENTS, working_dir=wdir),
    ]
    if USE_FILTERED_REPORT:
        analyzers_pickup.append(FilteredPrevalenceAnalyzer(expt_name=os.path.basename(out),
                                                           working_dir=wdir))

    with Platform('SLURM_LOCAL', job_directory=jdir) as platform:
        am = AnalyzeManager(configuration={}, ids=[(exp_id.strip(), ItemType.EXPERIMENT)],
                            analyzers=analyzers_pickup, partial_analyze_ok=True)
        am.analyze()
        sims = list_simulations(platform, exp_id.strip(), jdir)

    sys.exit(postprocess(sims, out))