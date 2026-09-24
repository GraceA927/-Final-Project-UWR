################################################################################
# Upper West Region EMOD Malaria Model -- PICKUP 2013-2015
#
# AIM 1: Adapt and calibrate EMOD to the Upper West Region of Ghana.
#
#   1963 ------------------------ 2012 | 2013 ---------- 2015
#          50-year burn-in (CM only)   |  pickup: CM, ITN, IRS, SMC, vaccine
#                                      ^
#                     serialized state (day 18,250) read here
#
# USAGE
#   1. Set BURNIN_EXPERIMENT_ID below.
#   2. Check inputs and preview the campaign without submitting anything:
#          python run_pickup_UWR.py --dry-run
#   3. Submit:
#          python run_pickup_UWR.py
#
# CALENDAR
#   Day 1 = 1 Jan 2013, flat 365-day years (no leap days), 3 years = 1,095
#   days. This matches the pickup rows of intervention_CMBN.csv
#   (2013 -> day 1, 2014 -> day 366, 2015 -> day 731).
#
# INTERVENTION TIMING
#   The `year` column is authoritative. The day within that year comes
#   from day_of_year if present, else start_day, else month (1st of the
#   month), else 1 Jan.
#   The input files count start_day from different base years (CM from
#   2013, ITN and SMC from 2014, IRS from 2015), but in every file
#   start_day mod 365 gives the right day of the stated year:
#     ITN 2014 start_day 165 -> 14 Jun 2014
#     IRS 2015 start_day 91  -> 1 Apr 2015 (matches its month = April)
#     SMC 2015 start_day 605 -> 28 Aug 2015
#   so that is how start_day is read. A month column, when present, is
#   checked against the result. Rows outside 2013-2015 are dropped and
#   counted in the printout.
#
# EMODPY VERSIONS
#   Keyword arguments are checked against the installed emodpy_malaria
#   functions. Any argument your version does not accept is dropped and
#   reported as a WARNING in the dry run -- read those before submitting.
#
# LARVAL HABITAT (Option A)
#   Burn-in states are used purely as starting populations. The pickup
#   sets its own habitat and sweeps it independently:
#     every selected burn-in state x PICKUP_XTLH_RANGE x PICKUP_HABITAT_MIXES
#   The burn-in xTLH / habitat values are NOT carried over; they are only
#   recorded as burnin_* tags for reference. Run_Number is still taken from
#   each burn-in state, as in the reference pickup.
#
#   Serialization_Mask_Node_Read = 16 skips the burn-in larval habitats so
#   the pickup's own xTLH and habitat mix actually take effect. With 0 the
#   burn-in habitats would be read back and the pickup sweep would do
#   nothing. Expect a short re-equilibration at the start of 2013 in any
#   branch whose pickup habitat differs from its burn-in habitat.
#
# INPUTS USED (from /shared/home/grace/Final-Project-UWR/inputs)
#   demographics.csv, vector.csv, intervention_CMBN.csv (phase == pickup),
#   interventions_ITN.csv (+ _ITN_age, _ITN_season), interventions_IRS.csv,
#   interventions_SMC.csv, interventions_malariaVaccine.csv
#   pcr_prevalence.csv is calibration reference data, not an input here.
################################################################################

import argparse
import getpass
import inspect
import json
import os
import pathlib
import sys
from functools import partial

import numpy as np
import pandas as pd

import emodpy_malaria.malaria_config as conf
from emodpy.emod_task import EMODTask
import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emod_api.demographics.PreDefinedDistributions as Distributions
from emod_api.demographics.DemographicsTemplates import CrudeRate
from emodpy_malaria.interventions.treatment_seeking import add_treatment_seeking
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.reporters.builtin import add_malaria_summary_report, add_event_recorder
import emod_api.campaign as camp

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import manifest


################################################################################
# PROJECT / MODEL SETTINGS
################################################################################

SITE_NAME = "UpperWest_Region"
NODE_ID = 1
BIRTH_RATE = 31.0          # must match the burn-in
MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
CLIMATE_ROOT = os.path.join(PROJECT_DIR, "climate", "pickup")
CLIMATE_SUBDIR = None      # e.g. "emod_weather_upper_west"; None = auto-detect

VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"
CM_FILE = "intervention_CMBN.csv"
ITN_FILE = "interventions_ITN.csv"
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"
IRS_FILE = "interventions_IRS.csv"
SMC_FILE = "interventions_SMC.csv"
VACCINE_FILE = "interventions_malariaVaccine.csv"


################################################################################
# CALENDAR
################################################################################

DAYS_PER_YEAR = 365

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_DAYS = (BURNIN_END_YEAR - BURNIN_START_YEAR + 1) * DAYS_PER_YEAR   # 18,250

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2015
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1                    # 3
PICKUP_DAYS = PICKUP_YEARS * DAYS_PER_YEAR                                 # 1,095

MONTH_OFFSET = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])  # flat 365-day year
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def month_number(value):
    """1-12 from a number or a month name ("April", "apr")."""
    text = str(value).strip().lower()
    if text.replace(".", "").isdigit():
        return int(float(text))
    if text[:3] in MONTH_NAMES:
        return MONTH_NAMES.index(text[:3]) + 1
    raise ValueError(f"Unrecognised month '{value}'.")


def month_of_doy(doy):
    return int(np.searchsorted(MONTH_OFFSET, int(doy) - 1, side="right"))


def fmt_day(pickup_day):
    """Pickup day -> '2015-04-01' on the flat 365-day calendar."""
    day = int(pickup_day)
    year = PICKUP_START_YEAR + (day - 1) // DAYS_PER_YEAR
    doy = (day - 1) % DAYS_PER_YEAR + 1
    month = month_of_doy(doy)
    return f"{year}-{month:02d}-{doy - MONTH_OFFSET[month - 1]:02d}"


################################################################################
# BURN-IN SELECTION
################################################################################
#
# Empty = pick up every burn-in simulation. To pick up only the
# best-fitting habitat set, filter on burn-in tags, e.g.
#   BURNIN_FILTER = {"x_Temporary_Larval_Habitat": 100.0, "scale_temp_rain": 10.0}
################################################################################

# Set this directly to the completed 50-year burn-in experiment ID before
# running pickup, e.g. BURNIN_EXPERIMENT_ID = ' '
BURNIN_EXPERIMENT_ID = '8c9e94bd-5053-443c-8d4f-5798a6f3cbe7'

BURNIN_FILTER = {}


################################################################################
# PICKUP PARAMETER SPACE (independent of the burn-in, Option A)
################################################################################
#
# Burn-in grid for reference:
#   SCALE_CONSTANT  = [1.0]
#   SCALE_TEMP_RAIN = [5.0, 10.0, 25.0, 50.0]
#   SCALE_WATER_VEG = [0.0, 0.1, 0.5, 1.0, 2.5]
#   XTLH_RANGE      = np.logspace(1.85, 2.15, 5)
#
# Re-sweep, or set a single calibrated value, e.g.
#   PICKUP_XTLH_RANGE    = [100.0]
#   PICKUP_HABITAT_MIXES = [(1.0, 10.0, 0.5)]
# Each mix is (scale_constant, scale_temp_rain, scale_water_veg).
################################################################################

PICKUP_XTLH_RANGE = list(np.logspace(1.85, 2.15, 5))
PICKUP_HABITAT_MIXES = [(1.0, 10.0, 0.5)]


################################################################################
# INTERVENTION DEFAULTS (used when a column is absent from the CSV)
################################################################################

CM_DRUGS = {
    "AL": ["Artemether", "Lumefantrine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}
SEVERE_TRIGGERS = ["NewSevereCase"]

ITN_DEFAULTS = dict(blocking_initial=0.9, blocking_decay=730,
                    killing_initial=0.6, killing_decay=1460,
                    discard_k=None, discard_l=None)
IRS_DEFAULTS = dict(killing_initial=0.7, killing_box=0, killing_decay=180,
                    repelling_initial=0.0, repelling_box=0, repelling_decay=90,
                    repetitions=1, timesteps_between_repetitions=365)
SMC_DEFAULTS = dict(age_min=0.25, age_max=5.0, drug_code="SPA")
SMC_DRUG_ALIASES = {"SPAQ": "SPA", "SP+AQ": "SPA", "SP-AQ": "SPA"}
VACCINE_DEFAULTS = dict(age_min=0.42, age_max=2.0, initial_effect=0.8,
                        box_duration=0, decay=592, vaccine_type="AcquisitionBlocking")


################################################################################
# CSV HELPERS
################################################################################

def read_clean_csv(path):
    """Read a CSV as text, strip stray whitespace, lower-case headers, parse numbers."""
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.apply(lambda s: s.str.strip()).replace("", np.nan).dropna(how="all")
    for col in df.columns:
        parsed = pd.to_numeric(df[col], errors="coerce")
        if parsed.notna().sum() == df[col].notna().sum():
            df[col] = parsed
    return df.reset_index(drop=True)


def pick(df, names, default=None, label=""):
    """Return the first column in `names` that exists, else a default-filled column."""
    for name in names:
        if name in df.columns:
            return df[name]
    if default is None:
        raise ValueError(f"{label}: needs one of the columns {names}; found {list(df.columns)}")
    return pd.Series([default] * len(df), index=df.index)


def keep_node(df):
    if "node_id" in df.columns:
        df = df[df["node_id"].astype(int) == NODE_ID]
    return df


def keep_pickup_phase(df):
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == "pickup"]
    return df


def add_pickup_day(df, label):
    """Add `pickup_day` (1 = 1 Jan 2013) and drop rows outside the pickup window."""
    df = df.copy()
    if "year" not in df.columns:
        raise ValueError(f"{label}: needs a `year` column; found {list(df.columns)}")
    year = df["year"].astype(int)

    if "day_of_year" in df.columns:
        doy = df["day_of_year"].astype(int)
    elif "start_day" in df.columns:
        doy = (df["start_day"].astype(int) - 1) % DAYS_PER_YEAR + 1
    elif "month" in df.columns:
        doy = df["month"].map(month_number).map(lambda m: MONTH_OFFSET[m - 1] + 1)
    else:
        doy = pd.Series(1, index=df.index)
        in_years = (year >= PICKUP_START_YEAR) & (year <= PICKUP_END_YEAR)
        if in_years.any():
            print(f"  WARNING {label}: no day_of_year/start_day/month column; using 1 Jan of each year.")

    if "month" in df.columns and ("day_of_year" in df.columns or "start_day" in df.columns):
        stated = df["month"].map(month_number)
        derived = doy.map(month_of_doy)
        for i in df.index[stated != derived]:
            print(f"  WARNING {label}: year {year[i]} says month {df.at[i, 'month']} but start_day "
                  f"gives month {derived[i]}; using start_day.")

    df["year"] = year
    df["pickup_day"] = (year - PICKUP_START_YEAR) * DAYS_PER_YEAR + doy.astype(int)
    in_window = (df["pickup_day"] >= 1) & (df["pickup_day"] <= PICKUP_DAYS)
    return df[in_window].copy(), int((~in_window).sum())


def check_fraction(series, label):
    if ((series < 0) | (series > 1)).any():
        raise ValueError(f"{label}: coverage values must be between 0 and 1.")


def curve_from(df, x_col, y_col):
    df = df.sort_values(x_col)
    return {"Times": [float(v) for v in df[x_col]], "Values": [float(v) for v in df[y_col]]}


def load_itn_age_curves():
    """{year: {"Times": ages, "Values": usage}} from interventions_ITN_age.csv."""
    path = os.path.join(INPUT_DIR, ITN_AGE_FILE)
    if not os.path.isfile(path):
        return {}
    df = keep_pickup_phase(keep_node(read_clean_csv(path)))
    if "year" not in df.columns:
        return {None: curve_from(df, "age", "age_usage")}
    return {int(y): curve_from(g, "age", "age_usage") for y, g in df.groupby("year")}


def age_curve_for(curves, year):
    """Curve for this distribution year; else the latest earlier year; else None."""
    if not curves:
        return None
    if None in curves:
        return curves[None]
    earlier = [y for y in curves if y <= year]
    return curves[max(earlier)] if earlier else None


def load_itn_season_curve():
    path = os.path.join(INPUT_DIR, ITN_SEASON_FILE)
    if not os.path.isfile(path):
        return None
    return curve_from(keep_node(read_clean_csv(path)), "season_time", "season_usage")


################################################################################
# INTERVENTION SCHEDULES
################################################################################

def load_case_management():
    df = read_clean_csv(os.path.join(INPUT_DIR, CM_FILE))
    required = ["phase", "duration", "trigger", "age_min", "age_max",
                "coverage", "seek", "rate", "drug", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{CM_FILE} is missing columns: {missing}")

    df = keep_node(keep_pickup_phase(df))
    df["drug"] = df["drug"].str.upper()
    df, dropped = add_pickup_day(df, CM_FILE)

    unknown = sorted(set(df["drug"]) - set(CM_DRUGS))
    if unknown:
        raise ValueError(f"{CM_FILE}: drug code(s) {unknown} not in CM_DRUGS.")

    df["requested_duration"] = df["duration"].astype(int)
    df["duration"] = np.minimum(df["requested_duration"], PICKUP_DAYS - df["pickup_day"] + 1)
    df["effective_coverage"] = df["coverage"].astype(float) * df["seek"].astype(float)
    check_fraction(df["effective_coverage"], f"{CM_FILE} (coverage x seek)")
    df["is_severe"] = df["trigger"].isin(SEVERE_TRIGGERS)
    return df, dropped


def load_generic(filename):
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        return None, 0
    return add_pickup_day(keep_pickup_phase(keep_node(read_clean_csv(path))), filename)


def load_itn():
    df, dropped = load_generic(ITN_FILE)
    if df is None or df.empty:
        return df, dropped
    d = ITN_DEFAULTS
    df["cov"] = pick(df, ["coverage", "cov_all"], label=ITN_FILE).astype(float)
    df["blocking_initial"] = pick(df, ["block_effect", "blocking_initial"], d["blocking_initial"]).astype(float)
    df["blocking_decay"] = pick(df, ["block_decay", "blocking_decay"], d["blocking_decay"]).astype(float)
    df["killing_initial"] = pick(df, ["kill_effect", "killing_initial"], d["killing_initial"]).astype(float)
    df["killing_decay"] = pick(df, ["kill_decay", "killing_decay"], d["killing_decay"]).astype(float)
    df["discard_k"] = pick(df, ["discard_k"], np.nan).astype(float)
    df["discard_l"] = pick(df, ["discard_l"], np.nan).astype(float)
    check_fraction(df["cov"], ITN_FILE)
    return df, dropped


def load_irs():
    df, dropped = load_generic(IRS_FILE)
    if df is None or df.empty:
        return df, dropped
    d = IRS_DEFAULTS
    df["cov"] = pick(df, ["coverage"], label=IRS_FILE).astype(float)
    df["killing_initial"] = pick(df, ["killing_initial_effect"], d["killing_initial"]).astype(float)
    df["killing_box"] = pick(df, ["killing_box_duration"], d["killing_box"]).astype(float)
    df["killing_decay"] = pick(df, ["killing_decay_time_constant"], d["killing_decay"]).astype(float)
    df["repelling_initial"] = pick(df, ["repelling_initial_effect"], d["repelling_initial"]).astype(float)
    df["repelling_box"] = pick(df, ["repelling_box_duration"], d["repelling_box"]).astype(float)
    df["repelling_decay"] = pick(df, ["repelling_decay_time_constant"], d["repelling_decay"]).astype(float)
    df["reps"] = pick(df, ["repetitions"], d["repetitions"]).astype(int)
    df["rep_gap"] = pick(df, ["timesteps_between_repetitions"], d["timesteps_between_repetitions"]).astype(int)
    check_fraction(df["cov"], IRS_FILE)
    return df, dropped


def load_smc():
    df, dropped = load_generic(SMC_FILE)
    if df is None or df.empty:
        return df, dropped
    d = SMC_DEFAULTS
    df["cov"] = pick(df, ["coverage"], label=SMC_FILE).astype(float)
    df["amin"] = pick(df, ["age_min", "agemin"], d["age_min"]).astype(float)
    df["amax"] = pick(df, ["age_max", "agemax"], d["age_max"]).astype(float)
    codes = pick(df, ["drug_code", "drug"], d["drug_code"]).astype(str).str.upper()
    df["drug_code"] = codes.replace(SMC_DRUG_ALIASES)
    df["round"] = pick(df, ["round"], 0).astype(int)
    check_fraction(df["cov"], SMC_FILE)
    return df, dropped


def load_vaccine():
    df, dropped = load_generic(VACCINE_FILE)
    if df is None or df.empty:
        return df, dropped
    d = VACCINE_DEFAULTS
    df["cov"] = pick(df, ["coverage"], label=VACCINE_FILE).astype(float)
    df["amin"] = pick(df, ["age_min", "agemin"], d["age_min"]).astype(float)
    df["amax"] = pick(df, ["age_max", "agemax"], d["age_max"]).astype(float)
    df["initial_effect"] = pick(df, ["initial_effect", "efficacy"], d["initial_effect"]).astype(float)
    df["box_duration"] = pick(df, ["box_duration"], d["box_duration"]).astype(float)
    df["decay"] = pick(df, ["decay", "decay_time"], d["decay"]).astype(float)
    check_fraction(df["cov"], VACCINE_FILE)
    return df, dropped


def load_all_schedules():
    sched, dropped = {}, {}
    sched["cm"], dropped["cm"] = load_case_management()
    sched["itn"], dropped["itn"] = load_itn()
    sched["irs"], dropped["irs"] = load_irs()
    sched["smc"], dropped["smc"] = load_smc()
    sched["vaccine"], dropped["vaccine"] = load_vaccine()
    sched["itn_age"] = load_itn_age_curves()
    sched["itn_season"] = load_itn_season_curve()
    sched["dropped"] = dropped
    return sched


def describe_schedules(sched):
    print(f"\nPickup interventions, {PICKUP_START_YEAR}-{PICKUP_END_YEAR} (day 1 = 1 Jan {PICKUP_START_YEAR}):")

    print("  Case management:")
    for r in sched["cm"].itertuples():
        clip = "" if r.duration == r.requested_duration else f" [clipped from {r.requested_duration}d]"
        print(f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4}) for {r.duration}d  {r.trigger:<16} "
              f"ages {r.age_min:g}-{r.age_max:g}  cov {r.effective_coverage:.3f}  rate {r.rate:g}  {r.drug}{clip}")
    if sched["dropped"]["cm"]:
        print(f"    ({sched['dropped']['cm']} row(s) outside the pickup window ignored)")

    for key, label in [("itn", "ITN"), ("irs", "IRS"), ("smc", "SMC"), ("vaccine", "Vaccine")]:
        df = sched[key]
        if df is None:
            print(f"  {label}: file not found, skipped")
        elif df.empty:
            print(f"  {label}: no rows in {PICKUP_START_YEAR}-{PICKUP_END_YEAR}")
        else:
            print(f"  {label}:")
            for r in df.itertuples():
                if key == "itn":
                    age = age_curve_for(sched["itn_age"], r.year)
                    extra = (f"  block {r.blocking_initial:.3f}/{r.blocking_decay:g}d  "
                             f"kill {r.killing_initial:.3f}/{r.killing_decay:g}d  "
                             f"age curve: {'yes' if age else 'none'}")
                elif key == "irs":
                    extra = f"  kill {r.killing_initial:g}/{r.killing_decay:g}d"
                elif key == "smc":
                    extra = f"  round {r.round}  ages {r.amin:g}-{r.amax:g}  {r.drug_code}"
                else:
                    extra = f"  ages {r.amin:g}-{r.amax:g}"
                print(f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4})  cov {r.cov:.3f}{extra}")
        if sched["dropped"][key]:
            print(f"    ({sched['dropped'][key]} row(s) outside the pickup window ignored)")

    if has_rows(sched["itn"]):
        print(f"  ITN seasonal usage curve: {'yes' if sched['itn_season'] else 'none'}")


SCHEDULES = None   # filled in by main()


################################################################################
# CLIMATE
################################################################################

CLIMATE_FILES = [
    "air_temperature.bin", "air_temperature.bin.json",
    "land_temperature.bin", "land_temperature.bin.json",
    "rainfall.bin", "rainfall.bin.json",
    "relative_humidity.bin", "relative_humidity.bin.json",
]


def resolve_climate_dir():
    """Find the folder that holds the .bin files: CLIMATE_ROOT itself or one subfolder."""
    if CLIMATE_SUBDIR:
        return os.path.join(CLIMATE_ROOT, CLIMATE_SUBDIR)
    if os.path.isfile(os.path.join(CLIMATE_ROOT, "air_temperature.bin")):
        return CLIMATE_ROOT
    if not os.path.isdir(CLIMATE_ROOT):
        raise FileNotFoundError(f"Pickup climate folder does not exist:\n{CLIMATE_ROOT}")
    found = [d for d in sorted(os.listdir(CLIMATE_ROOT))
             if os.path.isfile(os.path.join(CLIMATE_ROOT, d, "air_temperature.bin"))]
    if len(found) == 1:
        return os.path.join(CLIMATE_ROOT, found[0])
    if not found:
        raise FileNotFoundError(f"No air_temperature.bin in {CLIMATE_ROOT} or its subfolders.")
    raise ValueError(f"Several weather folders in {CLIMATE_ROOT}: {found}. Set CLIMATE_SUBDIR.")


def validate_climate(climate_dir):
    for name in CLIMATE_FILES:
        if not os.path.isfile(os.path.join(climate_dir, name)):
            raise FileNotFoundError(f"Missing climate file:\n{os.path.join(climate_dir, name)}")

    for var in ["air_temperature", "land_temperature", "rainfall", "relative_humidity"]:
        with open(os.path.join(climate_dir, f"{var}.bin.json")) as f:
            meta = json.load(f).get("Metadata", {})
        count = meta.get("DatavalueCount")
        years = str(meta.get("OriginalDataYears", ""))
        if count is not None and int(count) < PICKUP_DAYS:
            raise ValueError(
                f"{var}: weather has {count} days but the pickup needs {PICKUP_DAYS}. "
                "EMOD would wrap back to the start of the file."
            )
        if not years.startswith(str(PICKUP_START_YEAR)):
            print(f"  WARNING {var}: OriginalDataYears is '{years}', expected it to start in "
                  f"{PICKUP_START_YEAR}. Day 1 of the weather must be 1 Jan {PICKUP_START_YEAR}.")
    print(f"Climate: {climate_dir}")


################################################################################
# SITE / VECTOR INPUT
################################################################################

DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)
VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)

for p in [INPUT_DIR, DEMOGRAPHICS_PATH, VECTOR_PATH, os.path.join(INPUT_DIR, CM_FILE),
          manifest.eradication_path, manifest.schema_file]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path does not exist:\n{p}")

vector_df = pd.read_csv(VECTOR_PATH)
vector_df.columns = [c.strip() for c in vector_df.columns]


def _read_site_definition():
    site_df = pd.read_csv(DEMOGRAPHICS_PATH)
    site_df.columns = [c.strip() for c in site_df.columns]
    if len(site_df) != 1 or int(site_df["node_id"].iloc[0]) != NODE_ID:
        raise ValueError(f"{DEMOGRAPHICS_FILE} must hold exactly one row with node_id={NODE_ID}.")
    row = site_df.iloc[0]
    return {"name": SITE_NAME, "pop": int(row["pop"]),
            "lat": float(row["lat"]), "lon": float(row["lon"])}


################################################################################
# CONFIG -- must match the burn-in except for duration, climate and serialization
################################################################################

def set_param_fn(config):
    config = conf.set_team_defaults(config, manifest)

    config.parameters.Start_Time = 0
    config.parameters.Simulation_Duration = PICKUP_DAYS

    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", "air_temperature.bin")
    config.parameters.Land_Temperature_Filename = os.path.join("climate", "land_temperature.bin")
    config.parameters.Rainfall_Filename = os.path.join("climate", "rainfall.bin")
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", "relative_humidity.bin")

    config.parameters.Enable_Default_Reporting = 1

    for logger in ["JsonConfigurable", "VectorHabitat", "StandardEventCoordinator",
                   "SusceptibilityMalaria", "default", "Eradication"]:
        config.parameters[f"logLevel_{logger}"] = "ERROR"

    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = int(MEMORY_LIMIT)
    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = int(MEMORY_LIMIT - 500)

    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    # Same species and habitat set-up as the burn-in
    conf.add_species(config, manifest, vector_df["species"].tolist())
    for i in range(len(vector_df)):
        conf.set_species_param(
            config,
            species=vector_df["species"].iloc[i],
            parameter="Habitats",
            value=[
                {"Habitat_Type": "CONSTANT", "Max_Larval_Capacity": 1e7 * vector_df["constant"].iloc[i]},
                {"Habitat_Type": "TEMPORARY_RAINFALL", "Max_Larval_Capacity": 1e7 * vector_df["temp_rain"].iloc[i]},
                {"Habitat_Type": "WATER_VEGETATION", "Max_Larval_Capacity": 1e7 * vector_df["water_veg"].iloc[i]},
            ],
            overwrite=True,
        )

    config.parameters.Run_Number = 0   # replaced per simulation from the burn-in

    # Read the burn-in state; mask 16 = everything EXCEPT larval habitats,
    # which are rebuilt from this config (Option A, see header)
    config.parameters.Serialized_Population_Reading_Type = "READ"
    config.parameters.Serialization_Mask_Node_Read = 16

    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"
    return config


################################################################################
# DEMOGRAPHICS -- identical to the burn-in
################################################################################

def build_demog():
    site = _read_site_definition()
    demog = Demographics.from_template_node(lat=site["lat"], lon=site["lon"],
                                            pop=site["pop"], name=site["name"])
    demog.SetBirthRate(CrudeRate(BIRTH_RATE))
    demog.SetMortalityRate(CrudeRate(BIRTH_RATE))
    demog.SetEquilibriumAgeDistFromBirthAndMortRates(CrudeRate(BIRTH_RATE), CrudeRate(BIRTH_RATE))
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)
    return demog


################################################################################
# CAMPAIGN
################################################################################

def fix_property_restrictions(obj):
    """Replace null property-restriction fields with the empty arrays EMOD requires."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in ["Node_Property_Restrictions", "Property_Restrictions",
                       "Property_Restrictions_Within_Node"]:
                if value is None:
                    obj[key] = []
            else:
                fix_property_restrictions(value)
    elif isinstance(obj, list):
        for item in obj:
            fix_property_restrictions(item)


def add_case_management(campaign, df):
    for (day, duration, drug, severe), group in df.groupby(["pickup_day", "duration", "drug", "is_severe"]):
        if int(duration) <= 0:
            continue
        targets = [
            {"trigger": r.trigger, "coverage": float(r.effective_coverage),
             "agemin": float(r.age_min), "agemax": float(r.age_max), "rate": float(r.rate)}
            for r in group.itertuples()
        ]
        add_treatment_seeking(
            campaign, start_day=int(day), drug=CM_DRUGS[drug], targets=targets,
            duration=int(duration),
            broadcast_event_name="Received_Severe_Treatment" if severe else "Received_Treatment",
        )


_WARNED = set()


def call_supported(fn, campaign, label, **kwargs):
    """Call an emodpy function with only the keyword arguments it accepts; warn about the rest."""
    params = inspect.signature(fn).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        dropped = [k for k in kwargs if k not in params]
        if dropped and label not in _WARNED:
            print(f"  WARNING {label}: {fn.__name__} in this emodpy_malaria does not accept "
                  f"{dropped}; those settings are NOT applied.")
            _WARNED.add(label)
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(campaign, **kwargs)


def add_itns(campaign, df, age_curves, seasonal_dependence):
    # UsageDependentBednet: supports net discard (Weibull expiration), age- and
    # season-dependent usage. Confirmed discard_config works in this install.
    from emodpy_malaria.interventions.usage_dependent_bednet import add_scheduled_usage_dependent_bednet
    for r in df.itertuples():
        kwargs = dict(
            start_day=int(r.pickup_day),
            demographic_coverage=float(r.cov),
            blocking_initial_effect=float(r.blocking_initial),
            blocking_decay_time_constant=float(r.blocking_decay),
            killing_initial_effect=float(r.killing_initial),
            killing_decay_time_constant=float(r.killing_decay),
            age_dependence=age_curve_for(age_curves, r.year),
            seasonal_dependence=seasonal_dependence,
        )
        if not np.isnan(r.discard_k) and not np.isnan(r.discard_l):
            # Net retention: Weibull with shape kappa and scale lambda (days)
            kwargs["discard_config"] = {
                "Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",
                "Expiration_Period_Kappa": float(r.discard_k),
                "Expiration_Period_Lambda": float(r.discard_l),
            }
        call_supported(add_scheduled_usage_dependent_bednet, campaign, "ITN", **kwargs)


def add_irs(campaign, df):
    # The insecticide name is not passed: that needs insecticide resistance
    # set up in the config, which this model does not use.
    from emodpy_malaria.interventions.irs import add_scheduled_irs_housing_modification
    for r in df.itertuples():
        call_supported(
            add_scheduled_irs_housing_modification, campaign, "IRS",
            start_day=int(r.pickup_day),
            demographic_coverage=float(r.cov),
            repetitions=int(r.reps),
            timesteps_between_repetitions=int(r.rep_gap),
            killing_initial_effect=float(r.killing_initial),
            killing_box_duration=float(r.killing_box),
            killing_decay_time_constant=float(r.killing_decay),
            repelling_initial_effect=float(r.repelling_initial),
            repelling_box_duration=float(r.repelling_box),
            repelling_decay_time_constant=float(r.repelling_decay),
        )


def add_smc(campaign, df):
    from emodpy_malaria.interventions.drug_campaign import add_drug_campaign
    for r in df.itertuples():
        call_supported(
            add_drug_campaign, campaign, "SMC",
            campaign_type="SMC",
            drug_code=r.drug_code,
            start_days=[int(r.pickup_day)],
            coverage=float(r.cov),
            repetitions=1,
            target_group={"agemin": float(r.amin), "agemax": float(r.amax)},
            receiving_drugs_event_name="Received_SMC",
        )


def add_vaccine(campaign, df):
    from emodpy_malaria.interventions.vaccine import add_scheduled_vaccine
    for r in df.itertuples():
        call_supported(
            add_scheduled_vaccine, campaign, "Vaccine",
            start_day=int(r.pickup_day),
            demographic_coverage=float(r.cov),
            target_age_min=float(r.amin),
            target_age_max=float(r.amax),
            vaccine_type=VACCINE_DEFAULTS["vaccine_type"],
            vaccine_initial_effect=float(r.initial_effect),
            vaccine_box_duration=float(r.box_duration),
            vaccine_decay_time_constant=float(r.decay),
        )


def has_rows(df):
    return df is not None and not df.empty


def scale_habitat_from_mix(mix):
    """Same construction as the burn-in: mix x vector.csv fraction, per species."""
    scales = {"NodeID": vector_df["node_id"].unique()}
    for sp in vector_df["species"].unique():
        frac = vector_df.loc[vector_df["species"] == sp, "fraction"]
        scales[f"CONSTANT.{sp}"] = [mix[0] * f for f in frac]
        scales[f"TEMPORARY_RAINFALL.{sp}"] = [mix[1] * f for f in frac]
        scales[f"WATER_VEGETATION.{sp}"] = [mix[2] * f for f in frac]
    return pd.DataFrame(scales)


def build_camp(habitat_mix=None):
    camp.set_schema(manifest.schema_file)
    s = SCHEDULES

    mix = habitat_mix if habitat_mix is not None else PICKUP_HABITAT_MIXES[0]
    add_scale_larval_habitats(camp, df=scale_habitat_from_mix(mix), start_day=1, repetitions=-1)

    add_case_management(camp, s["cm"])
    if has_rows(s["itn"]):
        add_itns(camp, s["itn"], s["itn_age"], s["itn_season"])
    if has_rows(s["irs"]):
        add_irs(camp, s["irs"])
    if has_rows(s["smc"]):
        add_smc(camp, s["smc"])
    if has_rows(s["vaccine"]):
        add_vaccine(camp, s["vaccine"])

    fix_property_restrictions(camp.campaign_dict)
    return camp


################################################################################
# SERIALIZATION SWEEP -- one pickup simulation per selected burn-in simulation
################################################################################

HABITAT_TAGS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]


def load_burnin_df(burnin_id, platform):
    from utils_slurm import build_burnin_df
    df = build_burnin_df(burnin_id, platform, BURNIN_DAYS)

    for key, value in BURNIN_FILTER.items():
        if key not in df.columns:
            raise ValueError(f"BURNIN_FILTER key '{key}' is not a burn-in tag. Tags: {list(df.columns)}")
        df = df[np.isclose(df[key].astype(float), float(value))]

    if df.empty:
        raise ValueError(f"No burn-in simulations left for experiment {burnin_id} after BURNIN_FILTER.")
    return df.reset_index(drop=True)


def update_serialize_parameters(simulation, df, x: int):
    row = df.iloc[x]

    filenames = row["Serialized_Population_Filenames"]
    if isinstance(filenames, str):
        filenames = [filenames]

    simulation.task.set_parameter("Serialized_Population_Filenames", list(filenames))
    simulation.task.set_parameter("Serialized_Population_Path",
                                  os.path.join(row["serialized_file_path"], "output"))

    seed = int(row["Run_Number"])
    simulation.task.set_parameter("Run_Number", seed)
    tags = {"Run_Number": seed, "burnin_sim_index": int(x)}

    # Provenance only -- these values are NOT applied to the pickup (Option A)
    for tag in ["x_Temporary_Larval_Habitat"] + HABITAT_TAGS:
        if tag in df.columns:
            tags[f"burnin_{tag}"] = float(row[tag])

    return tags


def update_pickup_habitat(simulation, mix_index):
    mix = PICKUP_HABITAT_MIXES[int(mix_index)]
    simulation.task.create_campaign_from_callback(partial(build_camp, habitat_mix=mix))
    return {"scale_constant": float(mix[0]), "scale_temp_rain": float(mix[1]),
            "scale_water_veg": float(mix[2])}


def set_param(simulation, param, value):
    return simulation.task.set_parameter(param, value)


################################################################################
# REPORTS
################################################################################

def add_reports(task):
    for i in range(PICKUP_YEARS):
        add_malaria_summary_report(
            task, manifest,
            start_day=i * DAYS_PER_YEAR + 1,
            end_day=(i + 1) * DAYS_PER_YEAR,
            reporting_interval=30,
            age_bins=[0.25, 5, 15, 25, 100],
            max_number_reports=13,
            pretty_format=True,
            filename_suffix=f"Monthly_U5_{PICKUP_START_YEAR + i}",
        )

    events = ["Received_Treatment", "Received_Severe_Treatment"]
    if has_rows(SCHEDULES["smc"]):
        events.append("Received_SMC")

    add_event_recorder(task, event_list=events, start_day=1, end_day=PICKUP_DAYS,
                       node_ids=[NODE_ID], min_age_years=0, max_age_years=115)


################################################################################
# MAIN
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="UWR 2013-2015 pickup from a serialized burn-in.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs, print the schedule, write campaign_pickup_preview.json, then stop.")
    return parser.parse_args()


def general_sim(selected_platform, burnin_id, climate_dir):
    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    burnin_df = load_burnin_df(burnin_id, platform)
    n_sims = len(burnin_df) * len(PICKUP_XTLH_RANGE) * len(PICKUP_HABITAT_MIXES)
    print(f"\nBurn-in {burnin_id}: {len(burnin_df)} starting state(s) selected.")
    print(f"Pickup grid: {len(burnin_df)} states x {len(PICKUP_XTLH_RANGE)} xTLH x "
          f"{len(PICKUP_HABITAT_MIXES)} habitat mix(es) = {n_sims} simulations")

    task = EMODTask.from_default2(
        config_path="config.json",
        eradication_path=manifest.eradication_path,
        campaign_builder=build_camp,
        schema_path=manifest.schema_file,
        param_custom_cb=set_param_fn,
        ep4_custom_cb=None,
        demog_builder=build_demog,
        plugin_report=None,
    )

    add_reports(task)
    task.set_sif(manifest.SIF_PATH, platform)
    task.common_assets.add_directory(climate_dir, relative_path="climate")

    builder = SimulationBuilder()
    builder.add_sweep_definition(partial(update_serialize_parameters, df=burnin_df),
                                 range(len(burnin_df)))
    builder.add_sweep_definition(partial(set_param, param="x_Temporary_Larval_Habitat"),
                                 PICKUP_XTLH_RANGE)
    builder.add_sweep_definition(update_pickup_habitat, range(len(PICKUP_HABITAT_MIXES)))

    '''user = getpass.getuser()
    name = f"{user}_{SITE_NAME}_Pickup_{PICKUP_START_YEAR}_{PICKUP_END_YEAR}_{n_sims}sim"
    print(f"\nCreating experiment: {name}")'''
    
    experiment_name = "UWR_AIM1_Pickup_1963_2012_CM"

    experiment = Experiment.from_builder(builder, task, name=experiment_name)
    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")
    return experiment


if __name__ == "__main__":
    args = parse_args()

    SCHEDULES = load_all_schedules()
    describe_schedules(SCHEDULES)

    climate_dir = resolve_climate_dir()
    validate_climate(climate_dir)

    print(f"\nPickup xTLH: {[round(v, 2) for v in PICKUP_XTLH_RANGE]}")
    print(f"Pickup habitat mixes (constant, temp_rain, water_veg): {PICKUP_HABITAT_MIXES}")

    if args.dry_run:
        build_camp()
        camp.save("campaign_pickup_preview.json")
        print("\nDry run OK. Campaign written to campaign_pickup_preview.json. Nothing submitted.")
        sys.exit(0)

    burnin_id = BURNIN_EXPERIMENT_ID.strip()
    if not burnin_id:
        sys.exit("Set BURNIN_EXPERIMENT_ID at the top of the script to the finished burn-in experiment ID.")

    import emod_malaria.bootstrap as dtk
    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)

    general_sim("SLURM_LOCAL", burnin_id, climate_dir)