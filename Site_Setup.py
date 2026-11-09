"""
Upper West Region EMOD malaria model -- BURN-IN (1963-2012) + PICKUP (2013-2025)
One script, one STEP switch.

    1963 ---------- 2012 | 2013 ---------- 2025
      50-year burn-in    |  pickup
      CQ/ASAQ case mgmt  |  AL case mgmt + ANC/CWC ITN, IRS, SMC, IPTp, vaccine
                         ^ serialized state (state-18250.dtk)

STEPS
    setup_climate  Download and build both climate files
    burnin         Submit the 1963-2012 burn-in (writes serialized states)
    test_birth     30-year birth-rate calibration test (InsetChart on, x_Birth sweep)
    pickup         Submit the 2013-2025 pickup (one sim per burn-in state)

USAGE
    python run_UWR_burnin_pickup.py --step setup_climate
    python run_UWR_burnin_pickup.py --step burnin --dry-run
    python run_UWR_burnin_pickup.py --step burnin
    python run_UWR_burnin_pickup.py --step pickup --dry-run      (builds the campaign)
    python run_UWR_burnin_pickup.py --step pickup --test-one
    python run_UWR_burnin_pickup.py --step pickup

CLIMATE
    burn-in : real weather 2003-2012 (leap days dropped = 3,650 days), tiled 5x.
    pickup  : real weather 2013-2023 (4,015 days). 2024 repeats 2023 and 2025
              repeats 2024, so the file has all 4,745 days and EMOD never wraps.
    get_climate() (your get_climate.py) is used only for the download; leap-day
    removal, mean-temperature shift and the fill happen in build_climate().

MUST EQUAL IN BOTH PHASES (not stored in the serialized state): birth model,
birth/death rates, x_Birth, vector behaviour, demographics, Pregnancy property.

BIRTH MODEL: INDIVIDUAL_PREGNANCIES. BirthRate is per possible mother PER DAY.

PICKUP INTERVENTIONS (assumptions are marked "ASSUMPTION"; run --dry-run first,
it builds the whole campaign and fails loudly if an emodpy argument is missing)
    ANC ITN : Pregnant -> +16 wk -> ANC_Visit_<yr> -> ITN if still IsPregnant
    CWC ITN : Births -> +18 mo -> CWC_Visit_<yr> -> ITN, plus one catch-up
    IRS     : scheduled
    SMC     : one scheduled distribution per CSV row (= one round)
    IPTp    : Pregnant -> +16 wk -> IPTp_Visit_<yr> -> dose 1 if still IsPregnant,
              dose k after dose k-1 + 30 d (cumulative coverage -> conditional)
    Vaccine : birth cohorts; Births -> D1 at target age -> D2 ... chained by
              actual receipt of the previous dose
"""

import argparse
import inspect
import json
import os
import pathlib
import re
import sys
from functools import partial
from itertools import product

import numpy as np
import pandas as pd

##### idmtools ###############################################################
from idmtools.builders import SimulationBuilder
from idmtools.core import ItemType
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment
##### emodpy / emod_api ######################################################
from emodpy.emod_task import EMODTask
import emod_api.campaign as camp
import emod_api.interventions.common as common
import emod_api.demographics.PreDefinedDistributions as Distributions
from emod_api.demographics.DemographicsTemplates import CrudeRate, YearlyRate
##### emodpy-malaria #########################################################
import emodpy_malaria.malaria_config as conf
import emodpy_malaria.demographics.MalariaDemographics as Demographics
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.interventions.treatment_seeking import add_treatment_seeking
from emodpy_malaria.reporters.builtin import add_malaria_summary_report, add_event_recorder
from emodpy_malaria.weather import weather_to_csv, csv_to_weather, WeatherVariable

try:
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_scheduled_usage_dependent_bednet, add_triggered_usage_dependent_bednet)
except ImportError:
    add_scheduled_usage_dependent_bednet = add_triggered_usage_dependent_bednet = None
try:
    from emodpy_malaria.interventions.irs import add_scheduled_irs_housing_modification
except ImportError:
    add_scheduled_irs_housing_modification = None
try:
    from emodpy_malaria.interventions.drug_campaign import add_drug_campaign
except ImportError:
    add_drug_campaign = None

import emod_malaria.bootstrap as dtk
from get_climate import get_climate
import manifest


#############################################################################
# OPTIONS
#############################################################################

STEP = "pickup"        # 'setup_climate' | 'burnin' | 'test_birth' | 'pickup'

SITE_NAME = "UpperWest_Region"
NODE_ID = 1
MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"

EXPERIMENT_NAME_BURNIN = "Burnin50yrs_setup"
EXPERIMENT_NAME_PICKUP = "Pickup13yrs_setup"

BURNIN_EXPERIMENT_ID = "cd9b1091-c53b-444e-9106-3d8aecf3d666"  # b466d110-c809-4248-997e-85210e05274c    # <- burn-in experiment the pickup reads states from
BURNIN_FILTER = {}             # optional {tag: value} filter on burn-in sims

SELECTED_PLATFORM = "SLURM_LOCAL"


#############################################################################
# CALENDAR (flat 365-day years, no leap days)
#############################################################################

DAYS_PER_YEAR = 365
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_DAYS = (BURNIN_END_YEAR - BURNIN_START_YEAR + 1) * DAYS_PER_YEAR      # 18,250

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_DAYS = (PICKUP_END_YEAR - PICKUP_START_YEAR + 1) * DAYS_PER_YEAR      # 4,745

PHASE_DAYS = {"burnin": BURNIN_DAYS, "pickup": PICKUP_DAYS}

MONTH_OFFSET = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MID_YEAR_DOY = 183


def month_number(value):
    text = str(value).strip().lower()
    if text.replace(".", "").isdigit():
        return int(float(text))
    if text[:3] in MONTH_NAMES:
        return MONTH_NAMES.index(text[:3]) + 1
    raise ValueError(f"Unrecognised month '{value}'.")


def month_of_doy(doy):
    return int(np.searchsorted(MONTH_OFFSET, int(doy) - 1, side="right"))


def fmt_day(pickup_day):
    day = int(pickup_day)
    year = PICKUP_START_YEAR + (day - 1) // DAYS_PER_YEAR
    doy = (day - 1) % DAYS_PER_YEAR + 1
    month = month_of_doy(doy)
    return f"{year}-{month:02d}-{doy - MONTH_OFFSET[month - 1]:02d}"


def doy_of_day(day):
    return (int(day) - 1) % DAYS_PER_YEAR + 1


def month_mid_doy(month):
    m = month_number(month)
    return float(MONTH_OFFSET[m - 1]) + (MONTH_DAYS[m - 1] + 1) / 2.0


#############################################################################
# CLIMATE
#############################################################################

CLIMATE_TAG = "upper_west_region"
CLIMATE_ROOT = os.path.join(INPUT_DIR, "upper_west", "climate")
SITE_FILE_FOR_WEATHER = os.path.join(CLIMATE_ROOT, "uwr_weather_site.csv")

# real_days = downloaded days after leap days are dropped
# file_days = days in the file EMOD reads (real_days + filled years)
CLIMATE_PERIODS = {
    "burnin": dict(
        sim_start="1963", sim_end="2012", sim_days=18250,
        weather_start="2003", weather_start_day="001",
        weather_end="2012", weather_end_day="366",       # 2012 is a leap year
        real_days=3650, file_days=3650,
    ),
    "pickup": dict(
        sim_start="2013", sim_end="2025", sim_days=4745,
        weather_start="2013", weather_start_day="001",
        weather_end="2023", weather_end_day="365",       # real data stops here
        real_days=4015, file_days=4745,                  # 2024-2025 filled
    ),
}

TARGET_MEAN_TEMPERATURE = {"burnin": 28.95, "pickup": 30.60}   # None = no shift
RAIN_SHIFT = None
DROP_LEAP_DAYS = True

DEMOG_IDREF = "Gridded world grump2.5arcmin"   # must equal demographics IdReference

RAW_KEYS = {"air": "air_temperature", "land": "land_temperature",
            "rain": "rainfall", "hum": "relative_humidity"}


def fill_years(phase):
    p = CLIMATE_PERIODS[phase]
    return (p["file_days"] - p["real_days"]) // DAYS_PER_YEAR


def filled_calendar_years(phase):
    last_real = int(CLIMATE_PERIODS[phase]["weather_end"])
    return [last_real + k for k in range(1, fill_years(phase) + 1)]


def check_calendar():
    b, p = CLIMATE_PERIODS["burnin"], CLIMATE_PERIODS["pickup"]
    if b["sim_days"] != BURNIN_DAYS:
        raise ValueError("burnin sim_days != BURNIN_DAYS")
    if p["sim_days"] != PICKUP_DAYS:
        raise ValueError(f"pickup sim_days {p['sim_days']} != PICKUP_DAYS {PICKUP_DAYS}")
    for phase, c in CLIMATE_PERIODS.items():
        if (c["file_days"] - c["real_days"]) % DAYS_PER_YEAR:
            raise ValueError(f"{phase}: filled period must be whole 365-day years.")
        real_years = int(c["weather_end"]) - int(c["weather_start"]) + 1
        if real_years * DAYS_PER_YEAR != c["real_days"]:
            raise ValueError(f"{phase}: real_days {c['real_days']} != {real_years} years x 365.")
    if b["sim_days"] % b["file_days"]:
        raise ValueError("burn-in weather length must divide the burn-in length.")
    if p["file_days"] != p["sim_days"]:
        raise ValueError("pickup weather file must cover every pickup day exactly.")


def climate_specs(phase):
    parts = []
    if DROP_LEAP_DAYS:
        parts.append("noleap")
    if RAIN_SHIFT is not None:
        parts.append(f"{int(RAIN_SHIFT)}shift")
    target = TARGET_MEAN_TEMPERATURE.get(phase)
    if target is not None:
        parts.append(f"meanT{float(target)}")
    if fill_years(phase) > 0:
        parts.append(f"fill{CLIMATE_PERIODS[phase]['sim_end']}")
    return "_" + "_".join(parts) if parts else ""


def climate_filenames():
    return {
        "air": "dtk_15arcmin_air_temperature_daily_revised.bin",
        "land": "dtk_15arcmin_land_temperature_daily_revised.bin",
        "rain": "dtk_15arcmin_rainfall_daily_revised.bin",
        "hum": "dtk_15arcmin_relative_humidity_daily_revised.bin",
    }


def climate_raw_folder(phase):
    p = CLIMATE_PERIODS[phase]
    return os.path.join(CLIMATE_ROOT, f"{p['weather_start']}{p['weather_start_day']}-"
                                      f"{p['weather_end']}{p['weather_end_day']}")


def climate_folder(phase):
    base, specs = climate_raw_folder(phase), climate_specs(phase)
    return os.path.join(base, specs) if specs else base


def write_weather_site_file():
    """get_climate() needs name, lat, lon, nodes; build that from demographics.csv."""
    site = _read_site_definition()
    os.makedirs(CLIMATE_ROOT, exist_ok=True)
    pd.DataFrame([{"name": SITE_NAME, "lat": site["lat"], "lon": site["lon"],
                   "nodes": NODE_ID}]).to_csv(SITE_FILE_FOR_WEATHER, index=False)
    return SITE_FILE_FOR_WEATHER


def find_raw_files(folder):
    bins = [f for f in os.listdir(folder) if f.endswith(".bin")]
    found = {}
    for k, key in RAW_KEYS.items():
        hits = [f for f in bins if key in f]
        if len(hits) != 1:
            raise FileNotFoundError(f"Expected one '{key}' .bin in {folder}; found {hits}")
        found[k] = hits[0]
    return found


def build_climate(phase):
    """Raw download -> drop 29 Feb -> (rain shift) -> temperature shift -> fill missing years."""
    p = CLIMATE_PERIODS[phase]
    raw_dir, out_dir = climate_raw_folder(phase), climate_folder(phase)
    raw = find_raw_files(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    var = {"air": WeatherVariable.AIR_TEMPERATURE, "land": WeatherVariable.LAND_TEMPERATURE,
           "rain": WeatherVariable.RAINFALL, "hum": WeatherVariable.RELATIVE_HUMIDITY}
    columns = {var["air"]: "airtemp", var["land"]: "landtemp",
               var["hum"]: "humidity", var["rain"]: "rainfall"}

    df, wa = weather_to_csv(
        weather_dir=raw_dir, csv_file=os.path.join(raw_dir, "weather_tmp.csv"),
        weather_file_names={var[k]: raw[k] for k in var})
    node_col = next((c for c in ("nodes", "node", "node_id") if c in df.columns), None)
    split = (lambda d: [g for _, g in d.groupby(node_col)]) if node_col else (lambda d: [d])
    start = pd.Timestamp(year=int(p["weather_start"]), month=1, day=1)

    def prepare(g):
        g = g.sort_values("steps").copy()
        if DROP_LEAP_DAYS:
            dates = start + pd.to_timedelta(g["steps"] - 1, unit="D")
            g = g[~((dates.dt.month == 2) & (dates.dt.day == 29))].copy()
        g["steps"] = np.arange(1, len(g) + 1)
        if RAIN_SHIFT is not None:
            n = g["steps"].max()
            g["steps"] = g["steps"] - int(RAIN_SHIFT)
            g.loc[g["steps"] <= 0, "steps"] += n
            g = g.sort_values("steps")
        if len(g) != p["real_days"]:
            raise ValueError(f"{phase}: {len(g)} real days after processing, "
                             f"expected {p['real_days']}. Check the weather window.")
        return g

    df = pd.concat([prepare(g) for g in split(df)], ignore_index=True)

    target = TARGET_MEAN_TEMPERATURE.get(phase)      # shift uses REAL years only
    if target is not None:
        delta = float(target) - float(df["airtemp"].mean())
        df["airtemp"] += delta
        df["landtemp"] += delta
        print(f"  {phase}: temperature shifted {delta:+.2f} C -> real-period mean "
              f"air {df['airtemp'].mean():.2f}, land {df['landtemp'].mean():.2f}")

    n_fill = fill_years(phase)                       # each filled year = previous year
    if n_fill:
        def extend(g):
            g = g.sort_values("steps")
            last_year = g.tail(DAYS_PER_YEAR)
            parts = [g]
            for k in range(1, n_fill + 1):
                c = last_year.copy()
                c["steps"] = c["steps"] + DAYS_PER_YEAR * k
                parts.append(c)
            return pd.concat(parts)
        df = pd.concat([extend(g) for g in split(df)], ignore_index=True)
        print(f"  {phase}: years {filled_calendar_years(phase)} have no climate data; each "
              f"repeats the previous year (last real year {p['weather_end']}).")

    for g in split(df):
        if len(g) != p["file_days"]:
            raise ValueError(f"{phase}: final length {len(g)} != {p['file_days']}.")

    names = climate_filenames()
    ws = csv_to_weather(csv_data=df, attributes=wa, weather_dir=out_dir,
                        weather_columns=columns,
                        weather_file_names={var[k]: names[k] for k in var})
    ws.to_files(dir_path=out_dir)

    for name in names.values():                      # metadata may keep the raw day count
        meta_path = os.path.join(out_dir, name) + ".json"
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            md = meta.setdefault("Metadata", {})
            changed = False
            if md.get("IdReference") != DEMOG_IDREF:
                md["IdReference"] = DEMOG_IDREF
                changed = True
            if int(md.get("DatavalueCount", -1)) != p["file_days"]:
                md["DatavalueCount"] = p["file_days"]
                changed = True
            if changed:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
    return out_dir


def validate_climate(phase):
    check_calendar()
    if phase not in CLIMATE_PERIODS:
        raise ValueError(f"Unknown climate phase: {phase}")
    folder, p = climate_folder(phase), CLIMATE_PERIODS[phase]
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Climate directory does not exist:\n{folder}\n"
                                f"Run --step setup_climate first.")
    for name in climate_filenames().values():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing climate file:\n{path}\nRun --step setup_climate first.")
        expected = 4 * p["file_days"]                  # float32 x days x 1 node
        if os.path.getsize(path) != expected:
            raise ValueError(f"{name}: {os.path.getsize(path)} bytes, expected "
                             f"{expected} ({p['file_days']} days, 1 node).")
        meta_path = path + ".json"
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                count = json.load(f).get("Metadata", {}).get("DatavalueCount")
            if count is not None and int(count) != p["file_days"]:
                raise ValueError(f"{name}: metadata says {count} days, expected {p['file_days']}.")
    return folder


def setup_climate():
    check_calendar()
    site_file = write_weather_site_file()
    for phase, p in CLIMATE_PERIODS.items():
        raw_dir = climate_raw_folder(phase)
        os.makedirs(raw_dir, exist_ok=True)
        print(f"\n=== {phase}: real weather {p['weather_start']}-{p['weather_end']} ===")
        if any(f.endswith(".bin") for f in os.listdir(raw_dir)):
            print("  raw files already present, skipping download")
        else:
            get_climate(tag=CLIMATE_TAG, label="",
                        start_year=p["weather_start"], start_day=p["weather_start_day"],
                        end_year=p["weather_end"], end_day=p["weather_end_day"],
                        site_file=site_file, node_column="nodes", output_dir=raw_dir)
        build_climate(phase)
        validate_climate(phase)
        print(f"  OK: {climate_folder(phase)}")


#############################################################################
# DEMOGRAPHIC / VECTOR SETTINGS  --  MUST EQUAL IN BURN-IN AND PICKUP
#############################################################################

BIRTH_RATE_DEPENDENCE = "INDIVIDUAL_PREGNANCIES"
CRUDE_DEATH_RATE = 10.9                  # per 1,000 per year (2010 GSS rural Upper West)
PREG_RATE_PER_MOTHER_DAY = 3.62e-4       # per possible mother per day (target 23.1 births/1000/yr)
X_BIRTH = 1.0
MAX_PREG_RATE_PER_MOTHER_DAY = 1.0 / 280.0
STALE_CONFIG_PARAMS = ("Disable_IP_Whitelist",)

APPLY_VECTOR_BEHAVIOUR = True

TEST_YEARS = 30
TEST_SEEDS = 3
X_BIRTH_TEST = [0.8, 0.9, 1.0, 1.1, 1.2]


#############################################################################
# HABITAT GRID  (burn-in sweeps it; pickup inherits it from the burn-in tags)
#############################################################################

SCALE_CONSTANT = [1.0]
SCALE_TEMP_RAIN = [0.5, 1.0, 2.5]
SCALE_WATER_VEG = [0.0, 0.1, 0.5]
XTLH_RANGE = list(np.logspace(0.65, 1.85, 10))
NUM_SEEDS = 5
HABITAT_TAGS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]

habitat_df = pd.DataFrame(list(product(SCALE_CONSTANT, SCALE_TEMP_RAIN, SCALE_WATER_VEG)),
                          columns=HABITAT_TAGS)
N_HABS = len(habitat_df)
EXPECTED_SIMULATIONS = len(XTLH_RANGE) * N_HABS * NUM_SEEDS


#############################################################################
# INTERVENTION INPUT FILES  (a missing file is skipped -- check the names!)
#############################################################################

CM_FILE = "intervention_CMBN.csv"
ITN_ANC = "interventions_ITN_ANC.csv"
ITN_CWC = "interventions_ITN_CWC.csv"
IRS_FILE = "intervention_IRS.csv"
SMC_FILE = "intervention_SMC.csv"
IPTP_FILE = "interventions_IPTp.csv"
VACCINE_FILE = "intervention_malariaVaccine_wide.csv"
ITN_FILE = None                          # mass ITN campaigns disabled
USE_ITN_USAGE_CURVES = True
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"

CM_DRUGS = {
    "AL": ["Artemether", "Lumefantrine"],
    "ASAQ": ["Artesunate", "Amodiaquine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}
SEVERE_TRIGGERS = ["NewSevereCase"]


#############################################################################
# REPORT SETTINGS
#############################################################################

REPORT_INTERVAL = 30
REPORT_MAX_NUMBER = 13
REPORT_AGE_BINS = [0.25, 0.5, 5, 15, 25, 100]
PARASITE_DETECTION_THRESHOLD = 40

ADD_PREGNANCY_STATUS_REPORT = True
PREGNANCY_REPORT_INTERVAL = 15
PREGNANCY_AGE_BINS = [15, 50]            # MUST EQUAL PREGNANCY_AGE_BINS in the analyzer

EVENT_LIST = ["Births", "DiseaseDeaths", "NonDiseaseDeaths", "NewInfectionEvent",
              "NewClinicalCase", "Received_Treatment", "Received_Severe_Treatment",
              "Pregnant", "GaveBirth"]
BURNIN_EVENT_MIN_AGE, BURNIN_EVENT_MAX_AGE = 14, 45
PICKUP_EVENT_MIN_AGE, PICKUP_EVENT_MAX_AGE = 0, 115
RECORD_TRIGGER_EVENTS = True
TRIGGER_EVENTS = ("Pregnant", "GaveBirth", "Births")

WRITE_PICKUP_STATE = True

# RTS,S (EMOD RTSSVaccine). Values from the EMOD RTSSVaccine docs example;
# Boosted_Antibody_Concentration is a calibration choice, not derived from your data.
RTSS_BOOSTED_ANTIBODY = 1200
RTSS_COST_TO_CONSUMER = 1.0
RTSS_MAX_AGE_YEARS = 59 / 12.0


#############################################################################
# PICKUP INTERVENTION CONSTANTS
#############################################################################

ANC_DELAY_DAYS = 16 * 7
ANC_CHECK_PREGNANT_AT_DELIVERY = True
CWC_AGE_MONTHS = 18
CWC_DELAY_DAYS = int(round(CWC_AGE_MONTHS * DAYS_PER_MONTH))

IRS_PARAM_COLUMNS = {
    "killing_initial_effect": "killing_initial_effect",
    "killing_box_duration": "killing_box_duration",
    "killing_decay_time_constant": "killing_decay_time_constant",
    "repelling_initial_effect": "repelling_initial_effect",
    "repelling_box_duration": "repelling_box_duration",
    "repelling_decay_time_constant": "repelling_decay_time_constant",
}

SMC_DRUG_ALIASES = {"SPAQ": "SPA", "SP+AQ": "SPA", "SP-AQ": "SPA"}
SMC_NONCSV_DEFAULTS = dict(age_min=0.25, age_max=5.0, drug_code="SPA")

IPTP_DEFAULTS = dict(drug_code="SP", delay_days=ANC_DELAY_DAYS, gap_days=30)
IPTP_COLUMN_RE = re.compile(r"^iptp(\d+)_coverage$")

VACCINE_EXPECTED_DOSES = ("D1", "D2", "D3", "D4")
# ASSUMPTION: dose ages (months) used unless the CSV has d<k>_age_months columns.
VACCINE_DEFAULT_AGE_MONTHS = {1: 6, 2: 7, 3: 9, 4: 24}
VACCINE_NONCSV_DEFAULTS = dict(initial_effect=0.8, box_duration=0, decay=592,
                               vaccine_type="AcquisitionBlocking")
VACCINE_ISSUES = {}
ALLOW_EXCLUDED_VACCINE_COHORTS = False

USAGE_VALUE_COLUMNS = ["age_usage", "season_usage", "usage", "value", "use",
                       "usage_prob", "probability", "prob"]
USAGE_AGE_COLUMNS = ["age", "age_years", "age_yr"]
USAGE_DAY_COLUMNS = ["day_of_year", "doy", "day"]
USAGE_SEASON_TIME_COLUMNS = ["season_time"]
ITN_USAGE_CHANNELS = ("MASS", "ANC", "CWC")
MAX_USAGE_AGE_YEARS = 125
ITN_USAGE = None
SCHEDULES = None
CM_BURNIN = None
CM_PICKUP = None

PREG_RESTRICTION = {"Pregnancy": "IsPregnant"}


#############################################################################
# INPUT: VECTORS AND SITE
#############################################################################

VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)
DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)


def load_vectors():
    df = pd.read_csv(VECTOR_PATH, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    df["species"] = df["species"].astype(str).str.strip()
    required = ["node_id", "species", "fraction", "constant", "temp_rain", "water_veg"]
    if APPLY_VECTOR_BEHAVIOUR:
        required += ["anthropophily", "indoor_feeding"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{VECTOR_FILE} is missing columns: {missing}")
    if df["node_id"].nunique() != 1 or int(df["node_id"].iloc[0]) != NODE_ID:
        raise ValueError(f"{VECTOR_FILE} must hold only node_id={NODE_ID}.")
    if APPLY_VECTOR_BEHAVIOUR:
        for col in ("anthropophily", "indoor_feeding"):
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
            if ((df[col] < 0) | (df[col] > 1)).any():
                raise ValueError(f"{VECTOR_FILE}: '{col}' must be between 0 and 1.")
    return df


vector_df = load_vectors() if os.path.isfile(VECTOR_PATH) else None


def _read_site_definition():
    site_df = pd.read_csv(DEMOGRAPHICS_PATH)
    site_df.columns = [c.strip() for c in site_df.columns]
    for c in ("node_id", "pop", "lat", "lon"):
        if c not in site_df.columns:
            raise ValueError(f"{DEMOGRAPHICS_FILE} is missing column '{c}'.")
    if len(site_df) != 1 or int(site_df["node_id"].iloc[0]) != NODE_ID:
        raise ValueError(f"{DEMOGRAPHICS_FILE} must hold exactly one row with node_id={NODE_ID}.")
    row = site_df.iloc[0]
    return {"name": SITE_NAME, "pop": int(row["pop"]),
            "lat": float(row["lat"]), "lon": float(row["lon"])}


def validate_inputs(step):
    paths = [INPUT_DIR, DEMOGRAPHICS_PATH, VECTOR_PATH, os.path.join(INPUT_DIR, CM_FILE),
             manifest.eradication_path, manifest.schema_file]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path does not exist:\n{p}")
    if TEST_YEARS * DAYS_PER_YEAR > BURNIN_DAYS:
        raise ValueError("TEST_YEARS cannot exceed the burn-in length.")


#############################################################################
# CSV HELPERS
#############################################################################

def read_clean_csv(path):
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.apply(lambda s: s.str.strip()).replace("", np.nan).dropna(how="all")
    for col in df.columns:
        parsed = pd.to_numeric(df[col], errors="coerce")
        if parsed.notna().sum() == df[col].notna().sum():
            df[col] = parsed
    return df.reset_index(drop=True)


def pick(df, names, default=None, label=""):
    for name in names:
        if name in df.columns:
            return df[name]
    if default is None:
        raise ValueError(f"{label}: needs one of {names}; found {list(df.columns)}")
    return pd.Series([default] * len(df), index=df.index)


def need(df, names, label, kind=float):
    col = next((n for n in names if n in df.columns), None)
    if col is None:
        raise ValueError(f"{label}: needs a column named one of {list(names)}; "
                         f"found {list(df.columns)}. Add it to the CSV.")
    series = df[col]
    if series.isna().any():
        years = (sorted({int(y) for y in df.loc[series.isna(), "year"]})
                 if "year" in df.columns else [])
        where = f" for year(s) {years}" if years else ""
        raise ValueError(f"{label}: column `{col}` is blank{where}.")
    try:
        return series.astype(kind)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: column `{col}` must be {kind.__name__}; "
                         f"found {list(pd.unique(series))[:5]}.")


def parameter_or_default(df, names, default, label, kind=float):
    if any(name in df.columns for name in names):
        return need(df, names, label, kind)
    return pd.Series([default] * len(df), index=df.index, dtype=object).astype(kind)


def keep_node(df):
    if "node_id" in df.columns:
        df = df[df["node_id"].astype(int) == NODE_ID]
    return df


def keep_phase(df, phase):
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == phase]
    return df


def check_fraction(series, label):
    if ((series < 0) | (series > 1)).any():
        raise ValueError(f"{label}: coverage values must be between 0 and 1.")


def has_rows(df):
    return df is not None and not df.empty


def add_pickup_day(df, label, default_day_note=None):
    """Add pickup_day (day 1 = 1 Jan 2013); drop rows outside 2013-2025."""
    df = df.copy()
    if "year" not in df.columns:
        raise ValueError(f"{label}: needs a `year` column; found {list(df.columns)}")
    year = df["year"].astype(int)

    if "day_of_year" in df.columns:
        doy = df["day_of_year"].astype(int)
    elif "start_day" in df.columns:
        raw = df["start_day"].astype(int)
        doy = (raw - 1) % DAYS_PER_YEAR + 1
        implied = PICKUP_START_YEAR + (raw - 1) // DAYS_PER_YEAR
        for i in df.index[(raw > DAYS_PER_YEAR) & (implied != year)]:
            used = (year[i] - PICKUP_START_YEAR) * DAYS_PER_YEAR + doy[i]
            print(f"  WARNING {label}: year {year[i]} but start_day {raw[i]} is "
                  f"{fmt_day(raw[i])}; `year` is authoritative, using {fmt_day(used)}.")
    elif "month" in df.columns:
        doy = df["month"].map(month_number).map(lambda m: MONTH_OFFSET[m - 1] + 1)
    else:
        doy = pd.Series(1, index=df.index)
        if ((year >= PICKUP_START_YEAR) & (year <= PICKUP_END_YEAR)).any():
            print(f"  WARNING {label}: no day_of_year/start_day/month column; "
                  f"{default_day_note or 'using 1 Jan of each year.'}")

    df["year"] = year
    df["pickup_day"] = (year - PICKUP_START_YEAR) * DAYS_PER_YEAR + doy.astype(int)
    in_window = (df["pickup_day"] >= 1) & (df["pickup_day"] <= PICKUP_DAYS)
    return df[in_window].copy(), int((~in_window).sum())


def load_generic(filename, default_day_note=None):
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        return None, 0
    return add_pickup_day(keep_phase(keep_node(read_clean_csv(path)), "pickup"),
                          filename, default_day_note=default_day_note)


#############################################################################
# CASE MANAGEMENT (burn-in and pickup)
#############################################################################
# Burn-in rows are collapsed multi-year blocks: `year` = first year, `duration`
# spans all its years. Start day = (year - 1963) * 365 + 1; CSV start_day is
# ignored. coverage x seek is folded into Demographic_Coverage.

def _load_cm_common(phase):
    df = read_clean_csv(os.path.join(INPUT_DIR, CM_FILE))
    required = ["phase", "duration", "trigger", "age_min", "age_max",
                "coverage", "seek", "rate", "drug", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{CM_FILE} is missing columns: {missing}")
    df = keep_node(keep_phase(df, phase)).copy()
    df["drug"] = df["drug"].astype(str).str.upper()
    unknown = sorted(set(df["drug"]) - set(CM_DRUGS))
    if unknown:
        raise ValueError(f"{CM_FILE}: drug code(s) {unknown} not in CM_DRUGS.")
    return df


def _finish_cm(df, total_days):
    df["requested_duration"] = df["duration"].astype(int)
    df["duration"] = np.minimum(df["requested_duration"], total_days - df["pickup_day"] + 1)
    df["effective_coverage"] = df["coverage"].astype(float) * df["seek"].astype(float)
    check_fraction(df["effective_coverage"], f"{CM_FILE} (coverage x seek)")
    df["is_severe"] = df["trigger"].isin(SEVERE_TRIGGERS)
    return df


def load_case_management_burnin():
    df = _load_cm_common("burnin")
    if df.empty:
        raise ValueError(f"{CM_FILE} contains no rows with phase == 'burnin'.")
    df["year"] = df["year"].astype(int)
    out = df[(df["year"] < BURNIN_START_YEAR) | (df["year"] > BURNIN_END_YEAR)]
    if not out.empty:
        raise ValueError(f"{CM_FILE}: burn-in year(s) {sorted(set(out['year']))} "
                         f"outside {BURNIN_START_YEAR}-{BURNIN_END_YEAR}.")
    df["pickup_day"] = (df["year"] - BURNIN_START_YEAR) * DAYS_PER_YEAR + 1   # phase day
    return _finish_cm(df, BURNIN_DAYS), 0


def load_case_management_pickup():
    df = _load_cm_common("pickup")
    df, dropped = add_pickup_day(df, CM_FILE)
    return _finish_cm(df, PICKUP_DAYS), dropped


def warn_cm_overlaps(df):
    rows = list(df.itertuples())
    found = 0
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a.trigger != b.trigger or a.pickup_day == b.pickup_day:
                continue
            if a.age_min >= b.age_max or b.age_min >= a.age_max:
                continue
            if (a.pickup_day + a.duration <= b.pickup_day
                    or b.pickup_day + b.duration <= a.pickup_day):
                continue
            found += 1
            print(f"  WARNING: overlapping {a.trigger} campaigns: day {a.pickup_day} "
                  f"+{a.duration}d ({a.drug}) and day {b.pickup_day} +{b.duration}d "
                  f"({b.drug}). Coverage will stack.")
    if not found:
        print("  Overlap check: no overlapping case-management campaigns.")


def add_case_management(campaign, df):
    if not has_rows(df):
        return
    for (day, duration, drug, severe), group in df.groupby(
            ["pickup_day", "duration", "drug", "is_severe"]):
        if int(duration) <= 0:
            continue
        targets = [{"trigger": r.trigger, "coverage": float(r.effective_coverage),
                    "agemin": float(r.age_min), "agemax": float(r.age_max),
                    "rate": float(r.rate)} for r in group.itertuples()]
        add_treatment_seeking(
            campaign, start_day=int(day), drug=CM_DRUGS[drug], targets=targets,
            duration=int(duration),
            broadcast_event_name=("Received_Severe_Treatment" if severe
                                  else "Received_Treatment"))


#############################################################################
# PICKUP LOADERS
#############################################################################

def fill_itn_columns(df, label):
    df["cov"] = need(df, ["coverage", "cov_all"], label)
    df["blocking_initial"] = need(df, ["block_effect", "blocking_initial"], label)
    df["blocking_decay"] = need(df, ["block_decay", "blocking_decay"], label)
    df["killing_initial"] = need(df, ["kill_effect", "killing_initial"], label)
    df["killing_decay"] = need(df, ["kill_decay", "killing_decay"], label)
    df["discard_k"] = pick(df, ["discard_k"], np.nan).astype(float)
    df["discard_l"] = pick(df, ["discard_l"], np.nan).astype(float)
    check_fraction(df["cov"], label)
    return df


def load_itn():
    if ITN_FILE is None:
        return None, 0
    df, dropped = load_generic(ITN_FILE)
    if not has_rows(df):
        return df, dropped
    return fill_itn_columns(df, ITN_FILE), dropped


def load_routine_itn(filename, channel):
    """`year` is authoritative; coverage = probability within that year's group."""
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        return None, 0
    df = keep_phase(keep_node(read_clean_csv(path)), "pickup")
    if "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == channel]
    if "year" not in df.columns:
        raise ValueError(f"{filename}: needs a `year` column; found {list(df.columns)}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    in_window = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    dropped = int((~in_window).sum())
    df = df[in_window].reset_index(drop=True)
    if df["year"].duplicated().any():
        raise ValueError(f"{filename}: more than one row per year for channel {channel}.")
    df["year_start_day"] = (df["year"] - PICKUP_START_YEAR) * DAYS_PER_YEAR + 1
    if len(df):
        df = fill_itn_columns(df, filename)
    return df, dropped


def load_irs():
    df, dropped = load_generic(IRS_FILE)
    if not has_rows(df):
        return df, dropped
    df["cov"] = need(df, ["coverage", "cov_all"], IRS_FILE)
    check_fraction(df["cov"], IRS_FILE)
    for internal, col in IRS_PARAM_COLUMNS.items():
        df[internal] = need(df, [col], IRS_FILE)
    return df, dropped


def load_smc():
    """ASSUMPTION: each CSV row is ONE SMC round (year + month/day, coverage)."""
    df, dropped = load_generic(SMC_FILE)
    if not has_rows(df):
        return df, dropped
    df["cov"] = need(df, ["coverage", "cov_all"], SMC_FILE)
    check_fraction(df["cov"], SMC_FILE)
    df["age_min"] = parameter_or_default(df, ["age_min"], SMC_NONCSV_DEFAULTS["age_min"], SMC_FILE)
    df["age_max"] = parameter_or_default(df, ["age_max"], SMC_NONCSV_DEFAULTS["age_max"], SMC_FILE)
    if "drug" in df.columns:
        drug = need(df, ["drug"], SMC_FILE, str).str.upper()
        df["drug_code"] = drug.map(lambda d: SMC_DRUG_ALIASES.get(d, d))
    else:
        df["drug_code"] = SMC_NONCSV_DEFAULTS["drug_code"]
    return df, dropped


def load_iptp():
    """Columns iptp<k>_coverage hold CUMULATIVE coverage (ASSUMPTION); converted to conditional."""
    path = os.path.join(INPUT_DIR, IPTP_FILE)
    if not os.path.isfile(path):
        return None, 0
    df = keep_phase(keep_node(read_clean_csv(path)), "pickup").copy()
    if "year" not in df.columns:
        raise ValueError(f"{IPTP_FILE}: needs a `year` column.")
    df["year"] = df["year"].astype(int)
    in_window = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    dropped = int((~in_window).sum())
    df = df[in_window].reset_index(drop=True)
    if df["year"].duplicated().any():
        raise ValueError(f"{IPTP_FILE}: more than one row per year.")
    doses = sorted(int(m.group(1)) for c in df.columns if (m := IPTP_COLUMN_RE.match(c)))
    if not doses:
        raise ValueError(f"{IPTP_FILE}: needs columns like iptp1_coverage, iptp2_coverage.")
    rows = []
    for _, r in df.iterrows():
        cum = [float(r[f"iptp{k}_coverage"]) for k in doses]
        if any(np.isnan(cum)) or any(c < 0 or c > 1 for c in cum):
            raise ValueError(f"{IPTP_FILE}: year {r['year']} has blank or out-of-range coverage.")
        cond, prev = [], None
        for c in cum:
            if prev is None:
                cond.append(c)
            elif c > prev + 1e-9:
                print(f"  WARNING {IPTP_FILE}: year {r['year']} dose coverage {c:.3f} exceeds previous {prev:.3f}; conditional coverage capped at 1.")
                cond.append(1.0)
            else:
                cond.append(c / prev if prev > 0 else 0.0)
            prev = c
        rows.append({"year": int(r["year"]), "doses": doses, "conditional": cond})
    return rows, dropped


def _dose_col(df, k):
    for c in (f"d{k}", f"d{k}_coverage", f"coverage_d{k}", f"cov_d{k}", f"d{k}_cov"):
        if c in df.columns:
            return c
    return None


def load_vaccine():
    """Cumulative coverage by dose per birth cohort (year = birth year)."""
    VACCINE_ISSUES.clear()
    path = os.path.join(INPUT_DIR, VACCINE_FILE)
    if not os.path.isfile(path):
        return None, 0
    df = keep_phase(keep_node(read_clean_csv(path)), "pickup").copy()
    if "year" not in df.columns:
        raise ValueError(f"{VACCINE_FILE}: needs a `year` column (birth year).")
    df["year"] = df["year"].astype(int)
    in_window = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    dropped = int((~in_window).sum())
    df = df[in_window].reset_index(drop=True)
    if df["year"].duplicated().any():
        raise ValueError(f"{VACCINE_FILE}: more than one row per birth year.")

    cols = {k: _dose_col(df, k) for k in range(1, len(VACCINE_EXPECTED_DOSES) + 1)}
    if cols[1] is None:
        raise ValueError(f"{VACCINE_FILE}: needs a dose-1 coverage column (d1 / d1_coverage).")
    d = VACCINE_NONCSV_DEFAULTS
    df["_effect"] = parameter_or_default(df, ["initial_effect"], d["initial_effect"], VACCINE_FILE)
    df["_box"] = parameter_or_default(df, ["box_duration"], d["box_duration"], VACCINE_FILE)
    df["_decay"] = parameter_or_default(df, ["decay", "decay_time_constant"], d["decay"], VACCINE_FILE)
    df["_vtype"] = (need(df, ["vaccine_type"], VACCINE_FILE, str) if "vaccine_type" in df.columns
                    else d["vaccine_type"])

    cohorts = []
    for _, r in df.iterrows():
        y, issues, cum = int(r["year"]), [], []
        for k, col in cols.items():
            if col is None:
                break
            v = r[col]
            if pd.isna(v):
                if k == 1:
                    issues.append("dose 1 coverage is blank")
                break
            cum.append(float(v))
        if cum and any(c < 0 or c > 1 for c in cum):
            issues.append("coverage outside 0-1")
        if any(cum[i + 1] > cum[i] + 1e-9 for i in range(len(cum) - 1)):
            issues.append("cumulative coverage rises with dose")
        ages = {}
        for k in range(1, len(cum) + 1):
            col = next((c for c in (f"d{k}_age_months", f"age_d{k}_months") if c in df.columns), None)
            ages[k] = float(r[col]) if col and pd.notna(r[col]) else float(VACCINE_DEFAULT_AGE_MONTHS[k])
        if any(ages[k + 1] <= ages[k] for k in range(1, len(cum))):
            issues.append("dose ages must increase")
        if issues:
            VACCINE_ISSUES[y] = issues
            continue
        cond = [cum[0]] + [cum[i] / cum[i - 1] if cum[i - 1] > 0 else 0.0 for i in range(1, len(cum))]
        cohorts.append({"year": y, "cond": cond, "age_days": [int(round(ages[k] * DAYS_PER_MONTH))
                                                               for k in range(1, len(cum) + 1)],
                        "effect": float(r["_effect"]), "box": float(r["_box"]),
                        "decay": float(r["_decay"]), "vtype": str(r["_vtype"])})
    return cohorts, dropped


def cwc_catchup_span(year_start_day):
    """Delivery days of a CWC year that no Births trigger can serve (children born before day 1)."""
    first = int(year_start_day)
    last = min(first + DAYS_PER_YEAR - 1, CWC_DELAY_DAYS)
    return (first, last) if first <= last else None


def cwc_catchup_batches(year_start_day):
    span = cwc_catchup_span(year_start_day)
    return [span] if span else []


# ---- ITN usage curves (age x season) --------------------------------------

def itn_usage_label():
    return "age x season curves" if USE_ITN_USAGE_CURVES else "FLAT"


def _read_usage_table(filename, kind):
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"USE_ITN_USAGE_CURVES is True but {filename} was not found in {INPUT_DIR}.")
    df = keep_phase(keep_node(read_clean_csv(path)), "pickup").copy()
    if df.empty:
        raise ValueError(f"{filename}: no rows for node {NODE_ID} in the pickup phase.")
    value_col = next((c for c in USAGE_VALUE_COLUMNS if c in df.columns), None)
    if value_col is None:
        raise ValueError(f"{filename}: needs a usage column named one of "
                         f"{USAGE_VALUE_COLUMNS}; found {list(df.columns)}")
    usage = pd.to_numeric(df[value_col], errors="coerce")
    if usage.isna().any():
        raise ValueError(f"{filename}: `{value_col}` has blank or non-numeric entries.")
    check_fraction(usage, f"{filename} ({value_col})")
    df["_usage"] = usage.astype(float)

    if kind == "age":
        col = next((c for c in USAGE_AGE_COLUMNS if c in df.columns), None)
        if col is None:
            raise ValueError(f"{filename}: needs an age column named one of "
                             f"{USAGE_AGE_COLUMNS}; found {list(df.columns)}")
        x = pd.to_numeric(df[col], errors="coerce")
        if x.isna().any() or (x < 0).any():
            raise ValueError(f"{filename}: `{col}` must be numeric and >= 0.")
    else:
        day_col = next((c for c in USAGE_DAY_COLUMNS if c in df.columns), None)
        time_col = next((c for c in USAGE_SEASON_TIME_COLUMNS if c in df.columns), None)
        if day_col is not None:
            x = pd.to_numeric(df[day_col], errors="coerce")
        elif time_col is not None:
            x = pd.to_numeric(df[time_col], errors="coerce") + 1
            wrap = x > DAYS_PER_YEAR
            if wrap.any() and (x == 1).any():
                keep = np.logical_not(wrap)
                df = df[keep].copy()
                x = x[keep]
        elif "month" in df.columns:
            x = df["month"].map(month_mid_doy).astype(float)
        else:
            raise ValueError(f"{filename}: needs a day column "
                             f"{USAGE_DAY_COLUMNS + USAGE_SEASON_TIME_COLUMNS} or `month`.")
        if x.isna().any() or (x < 1).any() or (x > DAYS_PER_YEAR).any():
            raise ValueError(f"{filename}: day of year must be numeric, 1-{DAYS_PER_YEAR}.")

    df["_x"] = x.astype(float)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "channel" in df.columns:
        ch = df["channel"]
        df["channel"] = ch.where(ch.isna(), ch.astype(str).str.upper())
        bad = sorted(set(df["channel"].dropna()) - set(ITN_USAGE_CHANNELS))
        if bad:
            raise ValueError(f"{filename}: unknown channel(s) {bad}; "
                             f"use {list(ITN_USAGE_CHANNELS)} or leave blank.")
    return df.reset_index(drop=True)


def load_itn_usage():
    return {"age": _read_usage_table(ITN_AGE_FILE, "age"),
            "season": _read_usage_table(ITN_SEASON_FILE, "season")}


_USAGE_WARNED = set()


def _latest_block(d, year, label=""):
    if "year" not in d.columns:
        return d
    dated = d["year"].dropna()
    earlier = dated[dated <= year]
    if len(earlier):
        return d[d["year"] == earlier.max()]
    blank = d[d["year"].isna()]
    if len(blank):
        return blank
    if len(dated):
        first = float(dated.min())
        if (label, first) not in _USAGE_WARNED:
            print(f"  WARNING ITN {label} usage: earliest curve is for {int(first)}; "
                  f"also used for earlier delivery years.")
            _USAGE_WARNED.add((label, first))
        return d[d["year"] == first]
    return d


def _select_usage_rows(table, channel, year, label):
    if "channel" in table.columns:
        rows = _latest_block(table[table["channel"] == channel], year, label)
        if rows.empty:
            rows = _latest_block(table[table["channel"].isna()], year, label)
    else:
        rows = _latest_block(table, year, label)
    if rows.empty:
        raise ValueError(f"No ITN {label} usage rows for channel {channel}, year {year}.")
    return rows.sort_values("_x")


def usage_age_curve(channel, year):
    rows = _select_usage_rows(ITN_USAGE["age"], channel, int(year), "age")
    ages, vals = rows["_x"].tolist(), rows["_usage"].tolist()
    if len(set(ages)) != len(ages):
        raise ValueError(f"{ITN_AGE_FILE}: duplicate ages for channel {channel}, year {year}.")
    if ages[0] > 0:
        ages.insert(0, 0.0)
        vals.insert(0, vals[0])
    if ages[-1] < MAX_USAGE_AGE_YEARS:
        ages.append(float(MAX_USAGE_AGE_YEARS))
        vals.append(vals[-1])
    return {"Times": ages, "Values": vals}


def _cyclic_interp(days, vals, x):
    xs = np.array(list(days) + [d + DAYS_PER_YEAR for d in days]
                  + [d - DAYS_PER_YEAR for d in days], dtype=float)
    vs = np.array(list(vals) * 3, dtype=float)
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], vs[order]))


def usage_season_curve(channel, year, delivery_doy):
    """Calendar curve rotated to days-since-delivery (EMOD counts from distribution)."""
    rows = _select_usage_rows(ITN_USAGE["season"], channel, int(year), "season")
    days, vals = rows["_x"].tolist(), rows["_usage"].tolist()
    if len(set(days)) != len(days):
        raise ValueError(f"{ITN_SEASON_FILE}: duplicate days for channel {channel}, year {year}.")
    knots = {}
    for d, v in zip(days, vals):
        knots[float((d - delivery_doy) % DAYS_PER_YEAR)] = float(v)
    start_val = _cyclic_interp(days, vals, delivery_doy)
    knots[0.0] = start_val
    times = sorted(knots)
    values = [knots[t] for t in times]
    times.append(float(DAYS_PER_YEAR))               # repeats every 365 days
    values.append(start_val)
    return {"Times": times, "Values": values}


def itn_usage(channel, year, delivery_doy):
    if not USE_ITN_USAGE_CURVES:
        return ({"Times": [0.0, float(MAX_USAGE_AGE_YEARS)], "Values": [1.0, 1.0]},
                {"Times": [0.0, float(DAYS_PER_YEAR)], "Values": [1.0, 1.0]})
    return usage_age_curve(channel, year), usage_season_curve(channel, year, delivery_doy)


#############################################################################
# EMODPY ARGUMENT RESOLUTION (emodpy-malaria argument names differ by version)
#############################################################################

def _kw(fn, spec, required=()):
    """
    spec: {canonical: (value, [candidate argument names])}; value may be a
    function of the chosen name. Missing REQUIRED arguments raise (never silently
    drop a restriction); missing optional ones are reported once.
    """
    if fn is None:
        raise ImportError("An emodpy_malaria intervention function could not be imported.")
    params = inspect.signature(fn).parameters
    out, missing = {}, []
    for canon, (value, aliases) in spec.items():
        name = next((a for a in aliases if a in params), None)
        if name is None:
            if canon in required:
                missing.append(canon)
            else:
                print(f"  NOTE {fn.__name__}: no argument for '{canon}' in this version; skipped.")
        else:
            out[name] = value(name) if callable(value) else value
    if missing:
        raise RuntimeError(f"{fn.__name__}{inspect.signature(fn)}\n  has no argument for "
                           f"{missing}. Add the right name to the alias list in the script.")
    return out


def _call_supported(fn, *args, **kwargs):
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    kept = {k: v for k, v in kwargs.items() if k in params}
    skipped = sorted(set(kwargs) - set(kept))
    if skipped:
        print(f"  NOTE {fn.__name__}: arguments not supported by this version, skipped: {skipped}")
    return fn(*args, **kept)


#############################################################################
# RAW CAMPAIGN EVENTS (stage events, IPTp, vaccine)
#############################################################################
# These use stable EMOD campaign classes directly so delays are CONSTANT and
# events chain exactly. Field names that changed between EMOD versions are
# detected from the schema file.

_SCHEMA_TEXT = None


def schema_text():
    global _SCHEMA_TEXT
    if _SCHEMA_TEXT is None:
        with open(manifest.schema_file) as f:
            _SCHEMA_TEXT = f.read()
    return _SCHEMA_TEXT


def _within_node_key():
    return ("Property_Restrictions_Within_Node"
            if '"Property_Restrictions_Within_Node"' in schema_text() else "Property_Restrictions")


def _restr(d):
    if _within_node_key() == "Property_Restrictions_Within_Node":
        return [d]
    return [f"{k}:{v}" for k, v in d.items()]


def _broadcast(name):
    return {"class": "BroadcastEvent", "Broadcast_Event": name}


def _drug(drug_name):
    return {"class": "AntimalarialDrug", "Drug_Type": drug_name,
            "Dosing_Type": "FullTreatmentCourse", "Cost_To_Consumer": 0}


def _delayed(delay_days, interventions):
    d = {"class": "DelayedIntervention"}
    if '"Delay_Period_Constant"' in schema_text():
        d.update({"Delay_Period_Distribution": "CONSTANT_DISTRIBUTION",
                  "Delay_Period_Constant": float(delay_days)})
    else:
        d.update({"Delay_Distribution": "CONSTANT_DISTRIBUTION", "Delay_Period": float(delay_days)})
    if '"Actual_IndividualIntervention_Configs"' in schema_text():
        d["Actual_IndividualIntervention_Configs"] = interventions
    else:
        d["Actual_IndividualIntervention_Config"] = (
            interventions[0] if len(interventions) == 1
            else {"class": "MultiInterventionDistributor", "Intervention_List": interventions})
    return d


def _multi(interventions):
    return interventions[0] if len(interventions) == 1 else {
        "class": "MultiInterventionDistributor", "Intervention_List": interventions}


def add_triggered_event(campaign, start_day, duration, triggers, actual,
                        coverage=1.0, restriction=None):
    node_iv = {"class": "NodeLevelHealthTriggeredIV",
               "Trigger_Condition_List": list(triggers),
               "Duration": int(duration),
               "Demographic_Coverage": float(coverage),
               "Actual_IndividualIntervention_Config": actual}
    if restriction:
        node_iv[_within_node_key()] = _restr(restriction)
    campaign.campaign_dict["Events"].append({
        "class": "CampaignEvent", "Start_Day": float(start_day),
        "Nodeset_Config": {"class": "NodeSetAll"},
        "Event_Coordinator_Config": {
            "class": "StandardInterventionDistributionEventCoordinator",
            "Intervention_Config": node_iv}})


def add_visit_event(campaign, start_day, duration, trigger, delay_days, visit_event):
    """Stage 1: trigger -> wait -> broadcast visit_event (coverage 1.0)."""
    add_triggered_event(campaign, start_day, duration, [trigger],
                        _delayed(delay_days, [_broadcast(visit_event)]))


def delivery_window(year_start_day, delay_days):
    """Trigger window for deliveries in [year_start_day, +364]; None if wholly before day 1."""
    first = year_start_day - delay_days
    last = year_start_day + DAYS_PER_YEAR - 1 - delay_days
    if last < 1:
        return None
    start = max(1, first)
    return start, last - start + 1, first < 1


#############################################################################
# ITN / IRS / SMC (emodpy functions)
#############################################################################

def _itn_common(row, age, season, name, cov):
    spec = {
        "name": (name, ["intervention_name"]),
        "cov": (float(cov), ["demographic_coverage", "coverage"]),
        "ki": (float(row["killing_initial"]), ["killing_initial_effect"]),
        "kd": (float(row["killing_decay"]), ["killing_decay_time_constant"]),
        "bi": (float(row["blocking_initial"]), ["blocking_initial_effect"]),
        "bd": (float(row["blocking_decay"]), ["blocking_decay_time_constant"]),
        "age": (age, ["age_dependence"]),
        "season": (season, ["seasonal_dependence"]),
    }
    if pd.notna(row.get("discard_k")) and pd.notna(row.get("discard_l")):
        spec["discard"] = ({"Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",
                            "Expiration_Period_Kappa": float(row["discard_k"]),
                            "Expiration_Period_Lambda": float(row["discard_l"])},
                           ["discard_config"])
    return spec


ITN_REQUIRED = ("cov", "ki", "kd", "bi", "bd", "age", "season")


def _bednet_accepts_event():
    names = ("event_name", "broadcast_event", "received_event",
             "bednet_received_event", "intervention_distributed_event")
    params = inspect.signature(add_triggered_usage_dependent_bednet).parameters
    return any(n in params for n in names)


def add_itn_triggered(campaign, row, name, start_day, duration, triggers, cov, channel, year,
                      restriction=None, gender=None, delay=None, event=None):
    age, season = itn_usage(channel, year, MID_YEAR_DOY)
    spec = _itn_common(row, age, season, name, cov)
    spec["start"] = (int(start_day), ["start_day"])
    spec["dur"] = (int(duration), ["listening_duration", "duration"])
    spec["trig"] = (list(triggers), ["trigger_condition_list", "triggers"])
    required = ITN_REQUIRED + ("start", "dur", "trig")
    if restriction:
        spec["rest"] = ([restriction], ["ind_property_restrictions", "property_restrictions",
                                        "individual_property_restrictions"])
        required += ("rest",)
    if gender:
        spec["gender"] = (gender, ["target_gender"])
        required += ("gender",)
    if delay:
        spec["delay"] = (int(delay), ["delay_period_constant", "delay"])
        required += ("delay",)
    if event and _bednet_accepts_event():
        spec["event"] = (event, ["event_name", "broadcast_event", "received_event",
                                 "bednet_received_event", "intervention_distributed_event"])
    add_triggered_usage_dependent_bednet(campaign, **_kw(add_triggered_usage_dependent_bednet,
                                                         spec, required))


def add_itn_scheduled(campaign, row, name, start_day, cov, channel, year, delivery_doy,
                      age_range=None, event=None):
    age, season = itn_usage(channel, year, delivery_doy)
    spec = _itn_common(row, age, season, name, cov)
    spec["start"] = (int(start_day), ["start_day"])
    required = ITN_REQUIRED + ("start",)
    if age_range:
        spec["amin"] = (float(age_range[0]), ["target_age_min", "age_min", "agemin"])
        spec["amax"] = (float(age_range[1]), ["target_age_max", "age_max", "agemax"])
        required += ("amin", "amax")
    if event and _bednet_accepts_event():
        spec["event"] = (event, ["event_name", "broadcast_event", "received_event",
                                 "bednet_received_event", "intervention_distributed_event"])
    add_scheduled_usage_dependent_bednet(campaign, **_kw(add_scheduled_usage_dependent_bednet,
                                                         spec, required))


def add_routine_anc(campaign, df):
    for _, row in df.iterrows():
        y, ys = int(row["year"]), int(row["year_start_day"])
        if ANC_CHECK_PREGNANT_AT_DELIVERY:
            win = delivery_window(ys, ANC_DELAY_DAYS)
            if win is None:
                continue
            start, dur, gap = win
            visit = f"ANC_Visit_{y}"
            add_visit_event(campaign, start, dur, "Pregnant", ANC_DELAY_DAYS, visit)
            add_itn_triggered(campaign, row, f"ANC_ITN_{y}", ys, DAYS_PER_YEAR, [visit],
                              row["cov"], "ANC", y, restriction=PREG_RESTRICTION,
                              event="Received_ITN_ANC")
        else:
            start = max(1, ys - ANC_DELAY_DAYS)
            end = ys + DAYS_PER_YEAR - 1 - ANC_DELAY_DAYS
            if end < 1:
                continue
            add_itn_triggered(campaign, row, f"ANC_ITN_{y}", start, end - start + 1, ["Pregnant"],
                              row["cov"], "ANC", y, gender="Female", delay=ANC_DELAY_DAYS,
                              event="Received_ITN_ANC")


def add_routine_cwc(campaign, df):
    for _, row in df.iterrows():
        y, ys = int(row["year"]), int(row["year_start_day"])
        win = delivery_window(ys, CWC_DELAY_DAYS)
        if win is not None:
            start, dur, _gap = win
            visit = f"CWC_Visit_{y}"
            add_visit_event(campaign, start, dur, "Births", CWC_DELAY_DAYS, visit)
            add_itn_triggered(campaign, row, f"CWC_ITN_{y}", ys, DAYS_PER_YEAR, [visit],
                              row["cov"], "CWC", y, event="Received_ITN_CWC")
        for first, last in cwc_catchup_batches(ys):
            # children born before day 1 who turn CWC_AGE_MONTHS within [first, last]
            lo = max(first - 1, CWC_DELAY_DAYS - (last - first))
            add_itn_scheduled(campaign, row, f"CWC_Catchup_{y}", first, row["cov"], "CWC", y,
                              MID_YEAR_DOY, age_range=(lo / DAYS_PER_YEAR,
                                                       CWC_DELAY_DAYS / DAYS_PER_YEAR),
                              event="Received_ITN_CWC_Catchup")


def add_mass_itn(campaign, df):
    for _, row in df.iterrows():
        add_itn_scheduled(campaign, row, f"MASS_ITN_{int(row['year'])}", int(row["pickup_day"]),
                          row["cov"], "MASS", int(row["year"]), doy_of_day(row["pickup_day"]),
                          event="Received_ITN_MASS")


def add_irs(campaign, df):
    for _, row in df.iterrows():
        spec = {"start": (int(row["pickup_day"]), ["start_day"]),
                "cov": (float(row["cov"]), ["demographic_coverage", "coverage"])}
        for internal in IRS_PARAM_COLUMNS:
            spec[internal] = (float(row[internal]), [internal])
        add_scheduled_irs_housing_modification(
            campaign, **_kw(add_scheduled_irs_housing_modification, spec, tuple(spec)))


def add_smc(campaign, df):
    for _, row in df.iterrows():
        spec = {
            "type": ("SMC", ["campaign_type"]),
            "drug": (str(row["drug_code"]), ["drug_code"]),
            "start": (int(row["pickup_day"]),
                      lambda n: [int(row["pickup_day"])] if n.endswith("s") else int(row["pickup_day"])),
            "cov": (float(row["cov"]), ["coverage", "demographic_coverage"]),
            "target": ({"agemin": float(row["age_min"]), "agemax": float(row["age_max"])},
                       ["target_group"]),
            "event": ("Received_SMC", ["receiving_drugs_event_name", "event_name"]),
        }
        spec["start"] = (spec["start"][1], ["start_days", "start_day"])
        add_drug_campaign(campaign, **_kw(add_drug_campaign, spec,
                                          ("type", "drug", "start", "cov", "target")))


def add_iptp(campaign, rows):
    drug_names = CM_DRUGS[IPTP_DEFAULTS["drug_code"]]
    for r in rows:
        y, ys = r["year"], (r["year"] - PICKUP_START_YEAR) * DAYS_PER_YEAR + 1
        win = delivery_window(ys, IPTP_DEFAULTS["delay_days"])
        if win is None:
            continue
        start, dur, _gap = win
        visit = f"IPTp_Visit_{y}"
        add_visit_event(campaign, start, dur, "Pregnant", IPTP_DEFAULTS["delay_days"], visit)
        for n, (k, cov) in enumerate(zip(r["doses"], r["conditional"])):
            doses = [_drug(d) for d in drug_names] + [_broadcast(f"Received_IPTp_D{k}"),
                                                      _broadcast(f"IPTp_Dose{k}_{y}")]
            if n == 0:
                add_triggered_event(campaign, ys, DAYS_PER_YEAR, [visit], _multi(doses),
                                    coverage=cov, restriction=PREG_RESTRICTION)
            else:
                prev = f"IPTp_Dose{r['doses'][n - 1]}_{y}"
                add_triggered_event(
                    campaign, ys, DAYS_PER_YEAR + IPTP_DEFAULTS["gap_days"] * n, [prev],
                    _delayed(IPTP_DEFAULTS["gap_days"], doses), coverage=cov,
                    restriction=PREG_RESTRICTION)


def add_vaccine(campaign, cohorts):
    """RTS,S via the EMOD RTSSVaccine class (see EMOD docs).
    D1: Births -> wait dose-1 age -> D1. Dk: receipt of D(k-1) of the same
    birth cohort -> wait the age gap -> Dk, coverage = conditional uptake."""
    from emod_api import schema_to_class as s2c
    from emodpy_malaria.interventions.common import add_triggered_campaign_delay_event

    for c in cohorts:
        year = int(c["year"])
        first_day = (year - PICKUP_START_YEAR) * DAYS_PER_YEAR + 1
        for k, cond in enumerate(c["cond"], start=1):
            if cond <= 0:
                break
            iv = s2c.get_class_with_defaults("RTSSVaccine", campaign.schema_path)
            iv.Intervention_Name = f"RTSS_D{k}_{year}"
            iv.Boosted_Antibody_Concentration = RTSS_BOOSTED_ANTIBODY
            iv.Cost_To_Consumer = RTSS_COST_TO_CONSUMER
            announce = common.BroadcastEvent(campaign, Event_Trigger=f"Received_Vaccine_D{k}_{year}")
            if k == 1:
                trigger, start, listen = ["Births"], first_day, DAYS_PER_YEAR
                delay = c["age_days"][0]
            else:
                trigger = [f"Received_Vaccine_D{k - 1}_{year}"]
                start, listen = 1, -1
                delay = max(c["age_days"][k - 1] - c["age_days"][k - 2], 1)
            add_triggered_campaign_delay_event(
                campaign, start_day=int(start), trigger_condition_list=trigger,
                listening_duration=int(listen), delay_period_constant=float(delay),
                demographic_coverage=float(min(cond, 1.0)),
                target_age_min=0, target_age_max=RTSS_MAX_AGE_YEARS,
                individual_intervention=[iv, announce])

#############################################################################
# PICKUP CAMPAIGN
#############################################################################

def load_schedules():
    cm, cm_dropped = load_case_management_pickup()
    out = {"cm": cm}
    out["anc"], d1 = load_routine_itn(ITN_ANC, "ANC")
    out["cwc"], d2 = load_routine_itn(ITN_CWC, "CWC")
    out["mass"], d3 = load_itn()
    out["irs"], d4 = load_irs()
    out["smc"], d5 = load_smc()
    out["iptp"], d6 = load_iptp()
    out["vac"], d7 = load_vaccine()
    dropped = {CM_FILE: cm_dropped, ITN_ANC: d1, ITN_CWC: d2, IRS_FILE: d4,
               SMC_FILE: d5, IPTP_FILE: d6, VACCINE_FILE: d7}
    for name, n in dropped.items():
        if n:
            print(f"  {name}: {n} row(s) outside 2013-2025 dropped.")
    return out


def add_pickup_interventions(campaign):
    s = SCHEDULES
    if has_rows(s["anc"]):
        add_routine_anc(campaign, s["anc"])
    if has_rows(s["cwc"]):
        add_routine_cwc(campaign, s["cwc"])
    if has_rows(s["mass"]):
        add_mass_itn(campaign, s["mass"])
    if has_rows(s["irs"]):
        add_irs(campaign, s["irs"])
    if has_rows(s["smc"]):
        add_smc(campaign, s["smc"])
    if s["iptp"]:
        add_iptp(campaign, s["iptp"])
    if s["vac"]:
        add_vaccine(campaign, s["vac"])


def pickup_recorded_events():
    events = list(EVENT_LIST) + (list(TRIGGER_EVENTS) if RECORD_TRIGGER_EVENTS else [])
    s = SCHEDULES or {}
    events += ["Bednet_Got_New_One", "Received_SMC", "Received_IRS"]
    for key, prefix in (("anc", "ANC_Visit_"), ("cwc", "CWC_Visit_")):
        if has_rows(s.get(key)):
            events += [f"{prefix}{int(y)}" for y in s[key]["year"]]
    if s.get("iptp"):
        events += [f"IPTp_Visit_{r['year']}" for r in s["iptp"]]
        events += [f"Received_IPTp_D{k}" for k in s["iptp"][0]["doses"]]
    if s.get("vac"):
        events += [f"Received_Vaccine_D{k}" for k in range(1, max(len(c["cond"]) for c in s["vac"]) + 1)]
    return list(dict.fromkeys(events))


def pickup_custom_events():
    """Every custom event the pickup campaign broadcasts or listens for."""
    events = set(pickup_recorded_events())
    s = SCHEDULES or {}
    if s.get("iptp"):
        for r in s["iptp"]:
            events |= {f"IPTp_Dose{k}_{r['year']}" for k in r["doses"]}
    if s.get("vac"):
        for c in s["vac"]:
            events |= {f"VaccD{k}_{c['year']}" for k in range(1, len(c["cond"]) + 1)}
    events -= {"Births", "Pregnant", "GaveBirth", "DiseaseDeaths", "NonDiseaseDeaths",
               "NewInfectionEvent", "NewClinicalCase"}      # built-in
    return sorted(events)


def register_custom_events(config, names):
    try:
        current = list(config.parameters["Custom_Individual_Events"])
    except Exception:
        current = []
    config.parameters["Custom_Individual_Events"] = sorted(set(current) | set(names))


#############################################################################
# CONFIG / DEMOGRAPHICS / CAMPAIGNS
#############################################################################

def remove_stale_params(config):
    params = config.parameters
    for name in STALE_CONFIG_PARAMS:
        try:
            params.pop(name, None)
        except (AttributeError, TypeError):
            if hasattr(params, name):
                try:
                    delattr(params, name)
                except AttributeError:
                    pass
    return config


def fix_property_restrictions(obj):
    keys = ["Node_Property_Restrictions", "Property_Restrictions",
            "Property_Restrictions_Within_Node"]
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in keys:
                if value is None:
                    obj[key] = []
            else:
                fix_property_restrictions(value)
    elif isinstance(obj, list):
        for item in obj:
            fix_property_restrictions(item)


def set_param_fn(config, phase, test=False):
    config = conf.set_team_defaults(config, manifest)

    config.parameters.Start_Time = 0          # pickup day 1 = 1 Jan 2013 = weather day 1
    config.parameters.Simulation_Duration = (TEST_YEARS * DAYS_PER_YEAR if test
                                             else PHASE_DAYS[phase])

    names = climate_filenames()
    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", names["air"])
    config.parameters.Land_Temperature_Filename = os.path.join("climate", names["land"])
    config.parameters.Rainfall_Filename = os.path.join("climate", names["rain"])
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", names["hum"])

    config.parameters.Enable_Default_Reporting = 1 if test else 0     # InsetChart only in the test

    for logger in ["JsonConfigurable", "VectorHabitat", "StandardEventCoordinator",
                   "SusceptibilityMalaria", "default", "Eradication"]:
        config.parameters[f"logLevel_{logger}"] = "ERROR"

    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = int(MEMORY_LIMIT)
    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = int(MEMORY_LIMIT - 500)
    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    conf.add_species(config, manifest, vector_df["species"].tolist())
    base = 1e7
    for i in range(len(vector_df)):
        conf.set_species_param(
            config, species=vector_df["species"].iloc[i], parameter="Habitats",
            value=[{"Habitat_Type": "CONSTANT",
                    "Max_Larval_Capacity": base * vector_df["constant"].iloc[i]},
                   {"Habitat_Type": "TEMPORARY_RAINFALL",
                    "Max_Larval_Capacity": base * vector_df["temp_rain"].iloc[i]},
                   {"Habitat_Type": "WATER_VEGETATION",
                    "Max_Larval_Capacity": base * vector_df["water_veg"].iloc[i]}],
            overwrite=True)

    if APPLY_VECTOR_BEHAVIOUR:                  # MUST EQUAL in both phases
        for _, row in vector_df.iterrows():
            sp = str(row["species"]).strip()
            conf.set_species_param(config, species=sp, parameter="Anthropophily",
                                   value=float(row["anthropophily"]), overwrite=True)
            conf.set_species_param(config, species=sp, parameter="Indoor_Feeding_Fraction",
                                   value=float(row["indoor_feeding"]), overwrite=True)

    config.parameters.Run_Number = 0

    if phase == "burnin" and not test:
        config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
        config.parameters.Serialization_Time_Steps = [BURNIN_DAYS]
        config.parameters.Serialization_Mask_Node_Write = 0
        config.parameters.Serialization_Precision = "REDUCED"
    elif phase == "pickup":
        config.parameters.Serialized_Population_Reading_Type = "READ"
        config.parameters.Serialization_Mask_Node_Read = 0
        if WRITE_PICKUP_STATE:
            config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
            config.parameters.Serialization_Time_Steps = [PICKUP_DAYS]
            config.parameters.Serialization_Mask_Node_Write = 0
            config.parameters.Serialization_Precision = "REDUCED"
        register_custom_events(config, pickup_custom_events())

    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE   # MUST EQUAL
    config.parameters.x_Birth = X_BIRTH
    return config


def set_param(simulation, param, value):
    return simulation.task.set_parameter(param, value)


def scale_habitat_from_mix(mix, vdf):
    all_scales = {"NodeID": vdf["node_id"].unique()}
    for sp in vdf["species"].unique():
        sdf = vdf[vdf["species"] == sp]
        all_scales[f"CONSTANT.{sp}"] = [mix[0] * f for f in sdf["fraction"]]
        all_scales[f"TEMPORARY_RAINFALL.{sp}"] = [mix[1] * f for f in sdf["fraction"]]
        all_scales[f"WATER_VEGETATION.{sp}"] = [mix[2] * f for f in sdf["fraction"]]
    return pd.DataFrame(all_scales)


def add_pregnancy_tracking(campaign):
    common.change_individual_property_triggered(
        campaign, new_ip_key="Pregnancy", new_ip_value="IsPregnant",
        target_sex="Female", triggers=["Pregnant"], start_day=1, daily_prob=1,
        revert_in_days=-1, blackout=False)
    common.change_individual_property_triggered(
        campaign, new_ip_key="Pregnancy", new_ip_value="NotPregnant",
        target_sex="Female", triggers=["GaveBirth"], start_day=1, daily_prob=1,
        revert_in_days=-1, blackout=False)
    return campaign


def add_pregnancy_property(demog):
    if hasattr(demog, "AddIndividualPropertyAndHINT"):
        demog.AddIndividualPropertyAndHINT(Property="Pregnancy",
                                           Values=["NotPregnant", "IsPregnant"],
                                           InitialDistribution=[1.0, 0.0])
    else:
        demog.add_individual_property(property="Pregnancy",
                                      values=["NotPregnant", "IsPregnant"],
                                      initial_distribution=[1.0, 0.0])


def _final_demographics_config(config):
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    remove_stale_params(config)
    return config


def build_demog():
    """Identical in both phases."""
    site = _read_site_definition()
    demog = Demographics.from_template_node(lat=site["lat"], lon=site["lon"],
                                            pop=site["pop"], name=site["name"])
    demog.SetBirthRate(YearlyRate(PREG_RATE_PER_MOTHER_DAY * DAYS_PER_YEAR))
    demog.SetMortalityRate(CrudeRate(CRUDE_DEATH_RATE))
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)
    add_pregnancy_property(demog)
    demog.implicits.append(_final_demographics_config)
    return demog


def _mix_from_index(index):
    return (1.0, 1.0, 1.0) if index < 0 else tuple(float(x) for x in habitat_df.iloc[index])


def build_camp_burnin(habitat=-1, scale_start_day=1):
    """Burn-in: pregnancy tracking + habitat scaling + historical case management."""
    camp.set_schema(manifest.schema_file)
    add_pregnancy_tracking(camp)
    add_scale_larval_habitats(camp, df=scale_habitat_from_mix(_mix_from_index(habitat), vector_df),
                              start_day=scale_start_day, repetitions=-1)
    add_case_management(camp, CM_BURNIN)
    fix_property_restrictions(camp.campaign_dict)
    return camp


def build_camp_pickup(mix=(1.0, 1.0, 1.0), scale_start_day=1):
    """Pickup: same habitat mix as the burn-in sim + AL case management + interventions."""
    camp.set_schema(manifest.schema_file)
    add_pregnancy_tracking(camp)
    add_scale_larval_habitats(camp, df=scale_habitat_from_mix(mix, vector_df),
                              start_day=scale_start_day, repetitions=-1)
    add_case_management(camp, CM_PICKUP)
    add_pickup_interventions(camp)
    fix_property_restrictions(camp.campaign_dict)
    return camp


def update_burnin_habitat(simulation, habitat_index, scale_start_day):
    habitat_index, scale_start_day = int(habitat_index), int(scale_start_day)
    simulation.task.create_campaign_from_callback(
        partial(build_camp_burnin, habitat=habitat_index, scale_start_day=scale_start_day))
    simulation.task.set_parameter("Birth_Rate_Dependence", BIRTH_RATE_DEPENDENCE)
    remove_stale_params(simulation.task.config)
    return dict(zip(HABITAT_TAGS, _mix_from_index(habitat_index)))


#############################################################################
# PICKUP <- BURN-IN LINK
#############################################################################

def get_burnin_table(platform):
    if not BURNIN_EXPERIMENT_ID:
        raise ValueError("Set BURNIN_EXPERIMENT_ID before running the pickup.")
    exp = platform.get_item(BURNIN_EXPERIMENT_ID, ItemType.EXPERIMENT, raw=False)
    state_name = f"state-{BURNIN_DAYS:05d}.dtk"
    rows, missing = [], 0
    for sim in exp.simulations:
        tags = sim.tags
        if any(str(tags.get(k)) != str(v) for k, v in BURNIN_FILTER.items()):
            continue
        out_dir = os.path.join(str(platform.get_directory(sim)), "output")
        if not os.path.isfile(os.path.join(out_dir, state_name)):
            missing += 1
            continue
        rows.append({"path": out_dir, "sim_id": str(sim.id),
                     "Run_Number": int(float(tags["Run_Number"])),
                     "xtlh": float(tags["x_Temporary_Larval_Habitat"]),
                     **{k: float(tags[k]) for k in HABITAT_TAGS}})
    if missing:
        print(f"  WARNING: {missing} burn-in sim(s) have no {state_name}; skipped.")
    if not rows:
        raise RuntimeError("No burn-in simulations with a serialized state were found.")
    table = (pd.DataFrame(rows).sort_values(["xtlh"] + HABITAT_TAGS + ["Run_Number"])
             .reset_index(drop=True))
    if len(table) != EXPECTED_SIMULATIONS:
        print(f"  WARNING: {len(table)} burn-in states found, grid expects {EXPECTED_SIMULATIONS}.")
    return table


def update_pickup_from_burnin(simulation, i, table):
    row = table.iloc[int(i)]
    t = simulation.task
    t.set_parameter("Serialized_Population_Path", row["path"])
    t.set_parameter("Serialized_Population_Filenames", [f"state-{BURNIN_DAYS:05d}.dtk"])
    t.set_parameter("Run_Number", int(row["Run_Number"]))
    t.set_parameter("x_Temporary_Larval_Habitat", float(row["xtlh"]))
    mix = tuple(float(row[k]) for k in HABITAT_TAGS)
    t.create_campaign_from_callback(partial(build_camp_pickup, mix=mix, scale_start_day=1))
    t.set_parameter("Birth_Rate_Dependence", BIRTH_RATE_DEPENDENCE)
    register_custom_events(t.config, pickup_custom_events())     # after the campaign is built
    remove_stale_params(t.config)
    return {"x_Temporary_Larval_Habitat": float(row["xtlh"]),
            "Run_Number": int(row["Run_Number"]),
            "burnin_sim_id": row["sim_id"], **dict(zip(HABITAT_TAGS, mix))}


#############################################################################
# REPORTS
#############################################################################

def add_burnin_reports(task):
    add_event_recorder(task, event_list=EVENT_LIST, start_day=1, end_day=BURNIN_DAYS,
                       node_ids=[NODE_ID], min_age_years=BURNIN_EVENT_MIN_AGE,
                       max_age_years=BURNIN_EVENT_MAX_AGE)
    return task


def add_pickup_reports(task):
    add_event_recorder(task, event_list=pickup_recorded_events(), start_day=1,
                       end_day=PICKUP_DAYS, node_ids=[NODE_ID],
                       min_age_years=PICKUP_EVENT_MIN_AGE, max_age_years=PICKUP_EVENT_MAX_AGE)
    for year in range(PICKUP_START_YEAR, PICKUP_END_YEAR + 1):
        start = (year - PICKUP_START_YEAR) * DAYS_PER_YEAR
        _call_supported(
            add_malaria_summary_report, task, manifest,
            start_day=start, end_day=start + DAYS_PER_YEAR,
            reporting_interval=REPORT_INTERVAL, age_bins=REPORT_AGE_BINS,
            max_number_reports=REPORT_MAX_NUMBER, pretty_format=True,
            filename_suffix=f"Monthly_U5_{year}",
            parasite_detection_threshold=PARASITE_DETECTION_THRESHOLD)
    if ADD_PREGNANCY_STATUS_REPORT:
        try:
            from emodpy_malaria.reporters.builtin import add_report_node_demographics
        except ImportError:
            print("  NOTE: add_report_node_demographics not available; pregnancy report skipped.")
        else:
            _call_supported(
                add_report_node_demographics, task, manifest,
                age_bins=PREGNANCY_AGE_BINS, ip_key_to_collect="Pregnancy",
                stratify_by_gender=True, start_day=1, end_day=PICKUP_DAYS,
                reporting_interval=PREGNANCY_REPORT_INTERVAL)
    return task


#############################################################################
# DESCRIBE / SUBMIT
#############################################################################

def describe_settings(step):
    phase = "pickup" if step == "pickup" else "burnin"
    p = CLIMATE_PERIODS[phase]
    site = _read_site_definition()
    print(f"\nSTEP {step}  |  node pop {site['pop']}  |  {PHASE_DAYS[phase]} sim days")
    print(f"  Birth_Rate_Dependence {BIRTH_RATE_DEPENDENCE}; per-mother daily rate "
          f"{PREG_RATE_PER_MOTHER_DAY:g}; x_Birth {X_BIRTH:g}; death {CRUDE_DEATH_RATE:g}/1000/yr")
    print(f"  Vector behaviour applied: {APPLY_VECTOR_BEHAVIOUR}")
    print(f"  Climate: {climate_folder(phase)}")
    print(f"  Real weather {p['weather_start']}-{p['weather_end']} ({p['real_days']} days)"
          + (f"; years {filled_calendar_years(phase)} repeat the previous year"
             if fill_years(phase) else ""))
    if phase == "burnin":
        print(f"  Weather tiled {PHASE_DAYS[phase] // p['file_days']}x over the burn-in")
        print(f"\nCase management ({CM_FILE}, phase 'burnin'):")
        for r in CM_BURNIN.itertuples():
            end_year = BURNIN_START_YEAR + (r.pickup_day + r.duration - 2) // DAYS_PER_YEAR
            span = f"{r.year}" if end_year == r.year else f"{r.year}-{end_year}"
            clip = "" if r.duration == r.requested_duration else f" [clipped from {r.requested_duration}d]"
            print(f"  {span}: day {r.pickup_day} +{r.duration}d {r.trigger} ages "
                  f"{r.age_min:g}-{r.age_max:g} cov {r.effective_coverage:.3f} {r.drug}{clip}")
        warn_cm_overlaps(CM_BURNIN)
        if step == "test_birth":
            print(f"  x_Birth sweep {X_BIRTH_TEST}, {TEST_SEEDS} seeds, {TEST_YEARS} years")
        else:
            print(f"  Grid: {len(XTLH_RANGE)} xTLH x {N_HABS} habitat x {NUM_SEEDS} seeds = "
                  f"{EXPECTED_SIMULATIONS} sims")
    else:
        s = SCHEDULES
        print(f"\nCase management ({CM_FILE}, phase 'pickup'): {len(s['cm'])} row(s)")
        warn_cm_overlaps(s["cm"])
        print(f"ITN usage: {itn_usage_label()}")
        for key, label in (("anc", "ANC ITN"), ("cwc", "CWC ITN"), ("mass", "mass ITN"),
                           ("irs", "IRS"), ("smc", "SMC rounds")):
            df = s[key]
            if has_rows(df):
                yrs = sorted(set(int(y) for y in df["year"]))
                print(f"  {label}: {len(df)} row(s), years {yrs[0]}-{yrs[-1]}")
            else:
                print(f"  {label}: none")
        if s["iptp"]:
            print(f"  IPTp: years {s['iptp'][0]['year']}-{s['iptp'][-1]['year']}, "
                  f"doses {s['iptp'][0]['doses']}")
        else:
            print("  IPTp: none")
        if s["vac"]:
            print(f"  Vaccine: birth cohorts {[c['year'] for c in s['vac']]}")
        else:
            print("  Vaccine: none")
        for y, msgs in sorted(VACCINE_ISSUES.items()):
            print(f"  WARNING vaccine cohort {y} EXCLUDED: {'; '.join(msgs)}")
        for key, delay, what in (("anc", ANC_DELAY_DAYS, "ANC"), ("cwc", CWC_DELAY_DAYS, "CWC")):
            df = s[key]
            if has_rows(df):
                first = int(df["year_start_day"].iloc[0])
                if first - delay < 1:
                    print(f"  NOTE {what}: women/children already in the pipeline at day 1 "
                          f"{'are not reachable by a trigger' if what == 'ANC' else 'get the catch-up delivery'}"
                          f"{' (first ' + str(delay) + ' days of the first delivery year)' if what == 'ANC' else ''}.")


def make_platform():
    return Platform(SELECTED_PLATFORM, job_directory=manifest.job_directory,
                    partition=manifest.partition, time=manifest.sim_time,
                    modules=[manifest.singularity_module],
                    max_running_jobs=manifest.max_running_jobs)


def make_task(phase, test=False):
    return EMODTask.from_default2(
        config_path="config.json", eradication_path=manifest.eradication_path,
        campaign_builder=build_camp_pickup if phase == "pickup" else build_camp_burnin,
        schema_path=manifest.schema_file,
        param_custom_cb=partial(set_param_fn, phase=phase, test=test),
        ep4_custom_cb=None, demog_builder=build_demog, plugin_report=None)


def submit(step, test_one=False):
    phase = "pickup" if step == "pickup" else "burnin"
    test = step == "test_birth"
    platform = make_platform()
    builder = SimulationBuilder()

    if step == "burnin":
        builder.add_sweep_definition(partial(set_param, param="x_Temporary_Larval_Habitat"), XTLH_RANGE)
        builder.add_multiple_parameter_sweep_definition(
            update_burnin_habitat, dict(scale_start_day=[1], habitat_index=list(range(N_HABS))))
        builder.add_sweep_definition(partial(set_param, param="Run_Number"), range(NUM_SEEDS))
        name = EXPERIMENT_NAME_BURNIN
    elif test:
        builder.add_sweep_definition(partial(set_param, param="x_Birth"), X_BIRTH_TEST)
        builder.add_sweep_definition(partial(set_param, param="x_Temporary_Larval_Habitat"),
                                     [XTLH_RANGE[len(XTLH_RANGE) // 2]])
        builder.add_multiple_parameter_sweep_definition(
            update_burnin_habitat, dict(scale_start_day=[1], habitat_index=[N_HABS // 2]))
        builder.add_sweep_definition(partial(set_param, param="Run_Number"), range(TEST_SEEDS))
        name = f"{EXPERIMENT_NAME_BURNIN}_BIRTHTEST"
    else:
        table = get_burnin_table(platform)
        if test_one:
            table = table.iloc[:1].copy()
            print("  --test-one: submitting the first burn-in state only.")
        builder.add_sweep_definition(partial(update_pickup_from_burnin, table=table),
                                     range(len(table)))
        name = EXPERIMENT_NAME_PICKUP

    describe_settings(step)
    task = make_task(phase, test=test)
    if step == "burnin":
        task = add_burnin_reports(task)
    elif step == "pickup":
        task = add_pickup_reports(task)
    task.set_sif(manifest.SIF_PATH, platform)
    task.common_assets.add_directory(validate_climate(phase), relative_path="climate")

    experiment_name = f"{os.getlogin()}_{name}"
    print(f"\nCreating experiment: {experiment_name}")
    experiment = Experiment.from_builder(builder, task, name=experiment_name)
    experiment.run(wait_until_done=False, platform=platform)
    print(f"Experiment submitted. ID: {experiment.id}")
    if step == "burnin":
        print("Put this ID in BURNIN_EXPERIMENT_ID before running the pickup.")
    return experiment


def prepare_inputs(step):
    global CM_BURNIN, SCHEDULES, CM_PICKUP, ITN_USAGE
    validate_inputs(step)
    check_calendar()
    if step in ("burnin", "test_birth"):
        CM_BURNIN, _ = load_case_management_burnin()
    if step == "pickup":
        if USE_ITN_USAGE_CURVES:
            ITN_USAGE = load_itn_usage()
        SCHEDULES = load_schedules()
        CM_PICKUP = SCHEDULES["cm"]


#############################################################################
# MAIN
#############################################################################

def parse_args():
    ap = argparse.ArgumentParser(description="UWR burn-in + pickup")
    ap.add_argument("--step", choices=["setup_climate", "burnin", "test_birth", "pickup"],
                    default=STEP)
    ap.add_argument("--dry-run", action="store_true", help="print settings, submit nothing")
    ap.add_argument("--test-one", action="store_true", help="pickup: first burn-in state only")
    ap.add_argument("--allow-excluded-vaccine-cohorts", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ALLOW_EXCLUDED_VACCINE_COHORTS = args.allow_excluded_vaccine_cohorts

    if args.step == "setup_climate":
        setup_climate()
        raise SystemExit(0)

    prepare_inputs(args.step)

    if args.step == "pickup" and VACCINE_ISSUES and not ALLOW_EXCLUDED_VACCINE_COHORTS \
            and not args.dry_run:
        for y, msgs in sorted(VACCINE_ISSUES.items()):
            print(f"Vaccine cohort {y}: {'; '.join(msgs)}")
        raise SystemExit("Vaccine cohorts failed validation. Fix the CSV or pass "
                         "--allow-excluded-vaccine-cohorts.")

    if args.dry_run:
        describe_settings(args.step)
        validate_climate("pickup" if args.step == "pickup" else "burnin")
        if args.step == "pickup":
            build_camp_pickup()          # exercises every intervention builder
            print(f"  Campaign built: {len(camp.campaign_dict['Events'])} events.")
        print("\nDry run OK. Nothing submitted.")
        raise SystemExit(0)

    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)
    submit(args.step, test_one=args.test_one)