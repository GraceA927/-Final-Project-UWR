"""
Upper West Region EMOD malaria model -- BURN-IN 1963-2012

AIM 1: adapt and calibrate EMOD to the Upper West Region of Ghana.

    1963 ---------- 2012 | 2013 ---------- 2025
      50-year burn-in    |  pickup (run_pickup_UWR.py)
                         ^
              serialized state (state-18250.dtk) written here

Usage
-----
1. Birth-rate calibration test (do this FIRST, and again if you change
   PREG_RATE_PER_MOTHER_DAY, pop in demographics.csv, or the mortality rate):

       python run_burnin_UWR.py --test-birth

   Runs TEST_YEARS years with InsetChart ON, no serialization, one mid-range
   habitat, and sweeps x_Birth x TEST_SEEDS seeds. Read "Statistical
   Population" from each InsetChart.json. Judge the LAST 15 years: choose the
   x_Birth whose population is flattest, multiply it into
   PREG_RATE_PER_MOTHER_DAY, and set the result here AND in the pickup.

2. Preview settings without submitting:

       python run_burnin_UWR.py --dry-run

3. Full burn-in (1250 simulations, writes serialized states):

       python run_burnin_UWR.py

Calendar
--------
Flat 365-day years, no leap days. The climatology has exactly 18,250 days
(50 x 365), so Simulation_Duration is that flat count. Single UWR node.

Birth model (IMPORTANT)
-----------------------
Births use INDIVIDUAL_PREGNANCIES so women conceive, carry for 40 weeks and
give birth, firing the built-in 'Pregnant' and 'GaveBirth' events. The
campaign uses them to keep the individual property "Pregnancy"
(NotPregnant / IsPregnant) current for ANC / IPTp targeting at pickup.

In this mode BirthRate is applied PER POSSIBLE MOTHER (woman aged 14-45 who
is not already pregnant), NOT per head of population. It is therefore NOT a
crude birth rate and must not be set with CrudeRate(). A crude 31/1000 value
passed here gives roughly 5-6x too few births; the population then shrinks
exponentially (there is no feedback in this mode, unlike FIXED_BIRTH_RATE).
Mortality is a flat crude rate for all ages.

Births and deaths both scale with population, so the balance is neutral:
expect small residual drift even after calibration. Check the population on
every pickup output.

What is and is not carried in the serialized state:
  - carried:     individuals (ages, immunity, infections, pregnancies in
                 progress, individual-property values) and vector state
  - NOT carried: config settings such as Birth_Rate_Dependence and x_Birth.
                 The pickup must set the same values itself.

Vector behaviour
----------------
Anthropophily and Indoor_Feeding_Fraction are read per species from
vector.csv (columns 'anthropophily' and 'indoor_feeding') and written to the
config when APPLY_VECTOR_BEHAVIOUR is True. When False, the EMOD / team
defaults are used and those two CSV columns are ignored.
These are config settings, so they are NOT carried in the serialized state:
the pickup must use the same flag and the same values.

Burn-in content
---------------
INCLUDED
  - 50-year tiled climatology
  - Larval habitat parameter sweep (xTLH x habitat mix x seeds)
  - Vector behaviour (Anthropophily, Indoor_Feeding_Fraction per species,
    from vector.csv; controlled by APPLY_VECTOR_BEHAVIOUR)
  - Pregnancy tracking (Pregnancy individual property)
  - Historical case management (phase == "burnin" rows of intervention_CMBN.csv)
  - Event recorder (day 1 to end of burn-in)
NOT INCLUDED
  - Malaria Summary Reports (commented out in add_burnin_reports)
  - IRS, IG2 / ITN, SMC, IPTp, vaccine (all handled at pickup)
  - Reports in --test-birth runs (InsetChart only there)

Case-management schedule and day alignment
------------------------------------------
Burn-in rows are COLLAPSED: consecutive years with identical interventions
are a single row whose 'year' is the FIRST year and whose 'duration' spans
all of them (e.g. year=2004, duration=1460 covers 2004-2007). The schedule is:

    1995       CQ    365 d    (single year)
    2000-2002  CQ   1095 d
    2003       CQ    365 d
    2004-2007  ASAQ 1460 d
    2008       ASAQ  365 d
    2009-2012  ASAQ 1825 d requested, clipped to the end of the burn-in

(CQ for 1995-2003 and ASAQ from 2004, following Ghana's national treatment
policy timeline. The pickup phase uses AL.) The 2008 block is deliberately
one year long so it ends exactly where the 2009+ block begins; otherwise two
campaigns on the same trigger would overlap and stack their coverage.

CM_DAY_SOURCE picks how the start day is set:
    "year" - start_day = (year - 1963) * 365 + 1   (e.g. 2004 - day 14966)
    "csv"  - CSV start_day as-is
The default is "year": the CSV start_day column is then ignored, so its odd
values (1, 1461, 16470, 16791 ...) do not matter. Durations are clipped so
nothing runs past BURNIN_DAYS. This emodpy_malaria version has no 'seek'
target parameter, so coverage x seek is folded into Demographic_Coverage.
All CSV fields are whitespace-stripped. Overlapping campaigns on the same
trigger and age range are reported as warnings at submission time.
"""

import argparse
import json
import os
import pathlib
from functools import partial
from itertools import product

import numpy as np
import pandas as pd

import emodpy_malaria.malaria_config as conf
from emodpy.emod_task import EMODTask
import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emod_api.demographics.PreDefinedDistributions as Distributions
from emod_api.demographics.DemographicsTemplates import (CrudeRate, YearlyRate,)
#from emod_api.demographics.DemographicsTemplates import CrudeRate
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.interventions.treatment_seeking import add_treatment_seeking
import emod_api.campaign as camp
import emod_api.interventions.common as common
from emodpy_malaria.reporters.builtin import add_malaria_summary_report, add_event_recorder
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import emod_malaria.bootstrap as dtk

import manifest


################################################################################
# PROJECT / MODEL SETTINGS
################################################################################

SITE_NAME = "UpperWest_Region"
NODE_ID = 1
MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
CLIMATE_DIRECTORY = os.path.join(PROJECT_DIR, "climate", "burnin", "emod_weather_upper_west")
VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"
CASE_MANAGEMENT_FILE = "intervention_CMBN.csv"

EXPERIMENT_NAME = "UWR_50yrs_Trial"


################################################################################
# DEMOGRAPHIC SETTINGS  --  MUST BE IDENTICAL IN THE PICKUP SCRIPT
################################################################################

# Births via individual pregnancies so 'Pregnant' / 'GaveBirth' events exist.
BIRTH_RATE_DEPENDENCE = "INDIVIDUAL_PREGNANCIES"

# Flat all-age mortality, per 1,000 per year (converted with CrudeRate()).
CRUDE_DEATH_RATE = 10.9  # 2010 GSS standardised rural Upper West rate, per 1,000/year

# Pregnancy rate PER POSSIBLE MOTHER PER DAY (not a crude rate).
# Target: 23.1 births per 1,000 population per year (Upper West Region,
# 2010 Population and Housing Census).
# The earlier value of 5.0e-4 gave about 31.9 births per 1,000 in the
# 1998-2012 burn-in output, so it is scaled by 23.1 / 31.9 = 0.724:
#     5.0e-4 x 0.724 = 3.62e-4   (about 0.132 pregnancies per fertile woman / year)
# Births per 1,000 also depend on the age structure, so confirm with the
# BIRTH-RATE CHECK printed by analyze_burnin_UWR.py (target 23.1).
PREG_RATE_PER_MOTHER_DAY = 3.62e-4

# x_Birth multiplies BirthRate (documented config scalar). Production runs
# leave it at 1.0 and bake the calibrated value into PREG_RATE_PER_MOTHER_DAY.
X_BIRTH = 1.0

# Config parameters the installed schema no longer defines; removed so
# emodpy's schema-based purge does not warn for every simulation.
STALE_CONFIG_PARAMS = ("Disable_IP_Whitelist",)


################################################################################
# VECTOR BEHAVIOUR  --  MUST BE IDENTICAL IN THE PICKUP SCRIPT
################################################################################

# True  - Anthropophily and Indoor_Feeding_Fraction are read per species from
#          vector.csv ('anthropophily', 'indoor_feeding') and written to the
#          config.
# False - EMOD / team defaults are used; the two CSV columns are ignored.
APPLY_VECTOR_BEHAVIOUR = True


################################################################################
# BIRTH-RATE CALIBRATION TEST SETTINGS  (--test-birth)
################################################################################

TEST_MODE = False                         # set by --test-birth
TEST_YEARS = 30                           # judge the last 15 years
TEST_SEEDS = 3
X_BIRTH_TEST = [0.8, 0.9, 1.0, 1.1, 1.2]  # multiplies PREG_RATE_PER_MOTHER_DAY


# BURN-IN CALENDAR

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_YEARS = BURNIN_END_YEAR - BURNIN_START_YEAR + 1   # 50
DAYS_PER_YEAR = 365
BURNIN_DAYS = BURNIN_YEARS * DAYS_PER_YEAR               # 18,250

SIM_START_TIME = 0
NUM_SEEDS = 5


# REPORT SETTINGS

REPORT_START_YEAR = 1998
REPORT_END_YEAR = 2012
REPORT_INTERVAL = 30          # days per reporting period
REPORT_MAX_NUMBER = 13        # reports per yearly report (13 x 30 covers the 365-day year)
# 6-59 months, matching the prevalence surveys.
REPORT_AGE_BINS = [0.25, 0.5, 5, 15, 25, 100]

# Events to monitor across all ages and both sexes.
EVENT_LIST = [
    # Demographic events
    "Births",
    "DiseaseDeaths",
    "NonDiseaseDeaths",

    # Malaria events
    "NewInfectionEvent",
    "NewClinicalCase",

    # Treatment events emitted by your campaigns
    "Received_Treatment",
    "Received_Severe_Treatment",

    # Pregnancy tracking
    "Pregnant",
    "GaveBirth",
]

EVENT_MIN_AGE_YEARS = 14
EVENT_MAX_AGE_YEARS = 45


################################################################################
# CLIMATE SETTINGS
################################################################################

EXPECTED_CLIMATE_DAYS = BURNIN_DAYS
EXPECTED_ORIGINAL_YEARS = f"{BURNIN_START_YEAR}-{BURNIN_END_YEAR}"

## CALIBRATION PARAMETER SPACE
SCALE_CONSTANT = [1.0]
SCALE_TEMP_RAIN = [0.5, 1.0, 2.5, 5.0]
SCALE_WATER_VEG = [0.0, 0.1, 0.5]

#XTLH_RANGE = list(np.logspace(0.65, 1.85, 5))   # 4.5, 9.0, 17.8, 35.5, 70.8
XTLH_RANGE = list(np.round(np.logspace(np.log10(35.5), np.log10(90.0), 5), 3))
habitat_df = pd.DataFrame(
    list(product(SCALE_CONSTANT, SCALE_TEMP_RAIN, SCALE_WATER_VEG)),
    columns=["scale_constant", "scale_temp_rain", "scale_water_veg"],
)
N_HABS = len(habitat_df)
EXPECTED_SIMULATIONS = len(XTLH_RANGE) * N_HABS * NUM_SEEDS


# INPUT VALIDATION

VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)
DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)
CASE_MANAGEMENT_PATH = os.path.join(INPUT_DIR, CASE_MANAGEMENT_FILE)

required_paths = [
    INPUT_DIR,
    VECTOR_PATH,
    DEMOGRAPHICS_PATH,
    CASE_MANAGEMENT_PATH,
    manifest.eradication_path,
    manifest.schema_file,
    CLIMATE_DIRECTORY,
]

for path in required_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path does not exist:\n{path}")

if TEST_YEARS * DAYS_PER_YEAR > BURNIN_DAYS:
    raise ValueError("TEST_YEARS cannot exceed the burn-in length.")

if REPORT_START_YEAR < BURNIN_START_YEAR or REPORT_END_YEAR > BURNIN_END_YEAR:
    raise ValueError("Report years must fall inside the burn-in window.")


################################################################################
# CLIMATE VALIDATION
################################################################################

required_climate_files = [
    "air_temperature.bin", "air_temperature.bin.json",
    "land_temperature.bin", "land_temperature.bin.json",
    "rainfall.bin", "rainfall.bin.json",
    "relative_humidity.bin", "relative_humidity.bin.json",
]

for filename in required_climate_files:
    path = os.path.join(CLIMATE_DIRECTORY, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing climate file:\n{path}")

climate_metadata = {}
for variable in ["air_temperature", "land_temperature", "rainfall", "relative_humidity"]:
    metadata_path = os.path.join(CLIMATE_DIRECTORY, f"{variable}.bin.json")
    with open(metadata_path, "r") as f:
        climate_metadata[variable] = json.load(f)

for variable, metadata in climate_metadata.items():
    original_years = str(metadata.get("Metadata", {}).get("OriginalDataYears", ""))

    if original_years != EXPECTED_ORIGINAL_YEARS:
        raise ValueError(
            "\nClimate period mismatch.\n"
            f"Variable: {variable}\n"
            f"Found: {original_years}\n"
            f"Expected: {EXPECTED_ORIGINAL_YEARS}\n\n"
            f"Climate at {CLIMATE_DIRECTORY} does not match the "
            f"{BURNIN_START_YEAR}-{BURNIN_END_YEAR} burn-in climate. "
            "Re-run build_weather_upperWest.py if this directory has changed."
        )

    data_count = metadata.get("Metadata", {}).get("DatavalueCount")
    if data_count is not None and int(data_count) != EXPECTED_CLIMATE_DAYS:
        raise ValueError(
            f"{variable}: expected {EXPECTED_CLIMATE_DAYS} climate values, "
            f"but metadata reports {data_count}."
        )


################################################################################
# SITE / VECTOR INPUT
################################################################################

def _read_site_csv():
    site_df = pd.read_csv(DEMOGRAPHICS_PATH)
    site_df.columns = [c.strip() for c in site_df.columns]
    required = ["node_id", "pop", "lat", "lon"]
    missing = [c for c in required if c not in site_df.columns]
    if missing:
        raise ValueError(f"{DEMOGRAPHICS_FILE} is missing columns: {missing}")
    return site_df


def _read_site_definition():
    site_df = _read_site_csv()

    if len(site_df) != 1:
        raise ValueError(
            f"The current model is configured as one UWR node. "
            f"{DEMOGRAPHICS_FILE} contains {len(site_df)} rows."
        )

    row = site_df.iloc[0]
    node_id = int(row["node_id"])

    if node_id != NODE_ID:
        raise ValueError(
            f"{DEMOGRAPHICS_FILE} contains node_id={node_id}, "
            f"but the model requires node_id={NODE_ID}."
        )

    return {
        "name": SITE_NAME,
        "node_id": node_id,
        "pop": int(row["pop"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
    }


def get_site_node_ids():
    return _read_site_csv()["node_id"].astype(int).tolist()


# comment="#" ignores any trailing '# ...' text on a line, so stray notes in
# vector.csv cannot create junk columns or break parsing.
vector_df = pd.read_csv(VECTOR_PATH, comment="#", skipinitialspace=True)
vector_df.columns = [c.strip() for c in vector_df.columns]
vector_df = vector_df.dropna(how="all").reset_index(drop=True)
vector_df["species"] = vector_df["species"].astype(str).str.strip()

required_vector_columns = ["node_id", "species", "fraction", "constant", "temp_rain", "water_veg"]
if APPLY_VECTOR_BEHAVIOUR:
    required_vector_columns += ["anthropophily", "indoor_feeding"]

missing_vector_columns = [c for c in required_vector_columns if c not in vector_df.columns]
if missing_vector_columns:
    raise ValueError(f"{VECTOR_FILE} is missing columns: {missing_vector_columns}")

if vector_df["node_id"].nunique() != 1:
    raise ValueError("Multiple vector nodes were detected. This model is configured as a single-node UWR model.")

if int(vector_df["node_id"].iloc[0]) != NODE_ID:
    raise ValueError(
        f"{VECTOR_FILE} contains node_id={int(vector_df['node_id'].iloc[0])}, "
        f"but the model requires node_id={NODE_ID}."
    )

if APPLY_VECTOR_BEHAVIOUR:
    for col in ["anthropophily", "indoor_feeding"]:
        vector_df[col] = pd.to_numeric(vector_df[col], errors="raise").astype(float)
        if ((vector_df[col] < 0) | (vector_df[col] > 1)).any():
            raise ValueError(f"{VECTOR_FILE}: '{col}' must be between 0 and 1.")


################################################################################
# CASE MANAGEMENT INPUT
################################################################################
# Only phase == "burnin" rows are used; "pickup" rows belong to the pickup.
# Burn-in rows are collapsed multi-year blocks (see the docstring): 'year' is
# the first year of a block and 'duration' spans all of its years.

CASE_MANAGEMENT_PHASE = "burnin"

# Which column decides the campaign start day: "year" or "csv" (see docstring).
CM_DAY_SOURCE = "year"

# Drug codes used in the CSV -> EMOD drug names.
#   CQ   : 1995-2003 (burn-in)
#   ASAQ : 2004 onward (burn-in)
#   AL   : pickup phase (kept here so the same lookup can be shared)
# Check the names against the drug list in your installed schema.
DRUG_LOOKUP = {
    "AL": ["Artemether", "Lumefantrine"],
    "ASAQ": ["Artesunate", "Amodiaquine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}

SEVERE_TRIGGERS = ["NewSevereCase"]

required_cm_columns = [
    "node_id", "phase", "start_day", "duration", "trigger",
    "age_min", "age_max", "coverage", "seek", "rate", "drug", "year",
]
cm_int_columns = ["node_id", "start_day", "duration", "year"]
cm_float_columns = ["age_min", "age_max", "coverage", "seek", "rate"]

# Read everything as text, strip stray whitespace, then parse numbers.
cm_df = pd.read_csv(CASE_MANAGEMENT_PATH, dtype=str, skipinitialspace=True)
cm_df.columns = [c.strip() for c in cm_df.columns]

missing_cm_columns = [c for c in required_cm_columns if c not in cm_df.columns]
if missing_cm_columns:
    raise ValueError(f"{CASE_MANAGEMENT_FILE} is missing columns: {missing_cm_columns}")

cm_df = cm_df[required_cm_columns].apply(lambda s: s.str.strip())

if cm_df.isna().any().any() or (cm_df == "").any().any():
    raise ValueError(f"{CASE_MANAGEMENT_FILE} has empty cells. Fill or remove those rows.")

for col in cm_int_columns:
    cm_df[col] = pd.to_numeric(cm_df[col], errors="raise").astype(int)
for col in cm_float_columns:
    cm_df[col] = pd.to_numeric(cm_df[col], errors="raise").astype(float)

cm_df["phase"] = cm_df["phase"].str.lower()
cm_df["drug"] = cm_df["drug"].str.upper()

cm_burnin_df = cm_df[cm_df["phase"] == CASE_MANAGEMENT_PHASE].copy()

if cm_burnin_df.empty:
    raise ValueError(
        f"{CASE_MANAGEMENT_FILE} contains no rows with phase == '{CASE_MANAGEMENT_PHASE}'."
    )

if (cm_burnin_df["node_id"] != NODE_ID).any():
    raise ValueError(
        f"{CASE_MANAGEMENT_FILE} has burn-in rows for node_id(s) "
        f"{sorted(set(cm_burnin_df['node_id']) - {NODE_ID})}; the model uses node_id={NODE_ID}."
    )

unknown_drugs = sorted(set(cm_burnin_df["drug"]) - set(DRUG_LOOKUP))
if unknown_drugs:
    raise ValueError(
        f"{CASE_MANAGEMENT_FILE} uses drug code(s) {unknown_drugs} that are not in "
        f"DRUG_LOOKUP. Add them with their EMOD drug names."
    )

out_of_window = cm_burnin_df[
    (cm_burnin_df["year"] < BURNIN_START_YEAR) | (cm_burnin_df["year"] > BURNIN_END_YEAR)
]
if not out_of_window.empty:
    raise ValueError(
        f"{CASE_MANAGEMENT_FILE} has burn-in rows for year(s) "
        f"{sorted(set(out_of_window['year']))}, outside the "
        f"{BURNIN_START_YEAR}-{BURNIN_END_YEAR} burn-in window."
    )

# --- Campaign start day ---
cm_burnin_df["csv_start_day"] = cm_burnin_df["start_day"]
cm_burnin_df["year_start_day"] = (cm_burnin_df["year"] - BURNIN_START_YEAR) * DAYS_PER_YEAR + 1

if CM_DAY_SOURCE == "year":
    cm_burnin_df["start_day"] = cm_burnin_df["year_start_day"]
elif CM_DAY_SOURCE == "csv":
    cm_burnin_df["start_day"] = cm_burnin_df["csv_start_day"]
else:
    raise ValueError(f"CM_DAY_SOURCE must be 'year' or 'csv', not '{CM_DAY_SOURCE}'.")

if ((cm_burnin_df["start_day"] < 1) | (cm_burnin_df["start_day"] > BURNIN_DAYS)).any():
    raise ValueError(f"A case-management start_day falls outside 1-{BURNIN_DAYS}.")

# --- Clip so nothing runs past the end of the burn-in ---
cm_burnin_df["requested_duration"] = cm_burnin_df["duration"]
cm_burnin_df["duration"] = np.minimum(
    cm_burnin_df["requested_duration"],
    BURNIN_DAYS - cm_burnin_df["start_day"] + 1,
)

# --- Fold seek into coverage ---
cm_burnin_df["effective_coverage"] = cm_burnin_df["coverage"] * cm_burnin_df["seek"]

if ((cm_burnin_df["effective_coverage"] < 0) | (cm_burnin_df["effective_coverage"] > 1)).any():
    raise ValueError(
        f"{CASE_MANAGEMENT_FILE}: coverage x seek falls outside [0, 1] for at least one "
        f"burn-in row. Check the coverage and seek columns."
    )

cm_burnin_df["is_severe"] = cm_burnin_df["trigger"].isin(SEVERE_TRIGGERS)


def warn_cm_overlaps():
    """
    Warn when two campaigns on the same trigger overlap in both time and age.

    Overlapping campaigns each fire on the same event, so their coverages
    stack. Collapsed blocks should be back-to-back, not overlapping.
    """
    rows = list(cm_burnin_df.itertuples())
    reported = set()
    n_found = 0

    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a.trigger != b.trigger or a.start_day == b.start_day:
                continue
            if a.age_min >= b.age_max or b.age_min >= a.age_max:
                continue
            if (a.start_day + a.duration <= b.start_day
                    or b.start_day + b.duration <= a.start_day):
                continue

            key = (a.trigger, a.start_day, b.start_day)
            if key in reported:
                continue
            reported.add(key)
            n_found += 1
            print(
                f"  WARNING: overlapping {a.trigger} campaigns: "
                f"day {a.start_day} +{a.duration}d ({a.drug}) and "
                f"day {b.start_day} +{b.duration}d ({b.drug}). Coverage will stack."
            )

    if n_found == 0:
        print("  Overlap check: no overlapping case-management campaigns.")


def describe_case_management():
    """One line per campaign row, printed at submission time."""
    print(
        f"\nCase management ({CASE_MANAGEMENT_FILE}, phase='{CASE_MANAGEMENT_PHASE}', "
        f"day source='{CM_DAY_SOURCE}'):"
    )
    for row in cm_burnin_df.itertuples():
        clipped = "" if row.duration == row.requested_duration else \
            f"  [clipped from {row.requested_duration}d]"
        end_year = BURNIN_START_YEAR + (row.start_day + row.duration - 2) // DAYS_PER_YEAR
        span = f"{row.year}" if end_year == row.year else f"{row.year}-{end_year}"
        print(
            f"  years {span}: day {row.start_day} for {row.duration}d  "
            f"{row.trigger}  ages {row.age_min:g}-{row.age_max:g}  "
            f"coverage {row.effective_coverage:.3f} (={row.coverage:.3f} x seek {row.seek:.2f})  "
            f"rate {row.rate:g}  {row.drug}{clipped}"
        )

    mismatched = cm_burnin_df[cm_burnin_df["csv_start_day"] != cm_burnin_df["year_start_day"]]
    if not mismatched.empty:
        r = mismatched.iloc[0]
        print(
            f"  NOTE: CSV start_day {r.csv_start_day} vs year-based day {r.year_start_day} "
            f"for year {r.year}. Using '{CM_DAY_SOURCE}'."
        )

    warn_cm_overlaps()


################################################################################
# CONFIG HELPERS
################################################################################

def remove_stale_params(config):
    """
    Remove config parameters the installed schema does not define.

    AddIndividualPropertyAndHINT adds an implicit config function that sets
    Disable_IP_Whitelist for individual properties such as "Pregnancy".
    That runs after set_param_fn, so this must be called AFTER the implicits:
    from the last entry in demog.implicits and from the per-simulation sweep
    callback.
    """
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


################################################################################
# CONFIGURATION CALLBACK
################################################################################

def set_param_fn(config):
    """Configure EMOD for the 1963-2012 Upper West Region transmission burn-in."""

    config = conf.set_team_defaults(config, manifest)

    config.parameters.Start_Time = SIM_START_TIME
    config.parameters.Simulation_Duration = BURNIN_DAYS

    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", "air_temperature.bin")
    config.parameters.Land_Temperature_Filename = os.path.join("climate", "land_temperature.bin")
    config.parameters.Rainfall_Filename = os.path.join("climate", "rainfall.bin")
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", "relative_humidity.bin")

    config.parameters.Enable_Default_Reporting = 0  #  0 No InsetChart in the full burn-in

    for logger in [
        "JsonConfigurable",
        "VectorHabitat",
        "StandardEventCoordinator",
        "SusceptibilityMalaria",
        "default",
        "Eradication",
    ]:
        config.parameters[f"logLevel_{logger}"] = "ERROR"

    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = int(MEMORY_LIMIT)
    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = int(MEMORY_LIMIT - 500)

    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    conf.add_species(config, manifest, vector_df["species"].tolist())

    base_constant = 1e7
    base_other = 1e7

    for i in range(len(vector_df)):
        conf.set_species_param(
            config,
            species=vector_df["species"].iloc[i],
            parameter="Habitats",
            value=[
                {
                    "Habitat_Type": "CONSTANT",
                    "Max_Larval_Capacity": base_constant * vector_df["constant"].iloc[i],
                },
                {
                    "Habitat_Type": "TEMPORARY_RAINFALL",
                    "Max_Larval_Capacity": base_other * vector_df["temp_rain"].iloc[i],
                },
                {
                    "Habitat_Type": "WATER_VEGETATION",
                    "Max_Larval_Capacity": base_other * vector_df["water_veg"].iloc[i],
                },
            ],
            overwrite=True,
        )

    # ---------------------------------------------------------------
    # VECTOR BEHAVIOUR
    # Species-specific values read from vector.csv.
    # These settings must match the pickup script.
    # Runs after conf.add_species() so every species already exists.
    # ---------------------------------------------------------------
    if APPLY_VECTOR_BEHAVIOUR:
        for _, row in vector_df.iterrows():
            species = str(row["species"]).strip()

            conf.set_species_param(
                config,
                species=species,
                parameter="Anthropophily",
                value=float(row["anthropophily"]),
                overwrite=True,
            )

            conf.set_species_param(
                config,
                species=species,
                parameter="Indoor_Feeding_Fraction",
                value=float(row["indoor_feeding"]),
                overwrite=True,
            )

    config.parameters.Run_Number = 0

    #  Serialization: only for the full burn-in
    # The birth-rate test must not set serialization parameters at all.
    # Full burn-in: save state-18250.dtk for the pickup simulations.
    if not TEST_MODE:
        config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
        config.parameters.Serialization_Time_Steps = [BURNIN_DAYS]
        config.parameters.Serialization_Mask_Node_Write = 0
        config.parameters.Serialization_Precision = "REDUCED"

    #  Birth model and scalar
    # Config settings are NOT stored in the serialized state; the pickup must
    # set the same values.
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    config.parameters.x_Birth = X_BIRTH

    #  Birth-rate calibration test overrides ------------------------------
    if TEST_MODE:
        config.parameters.Simulation_Duration = TEST_YEARS * DAYS_PER_YEAR
        config.parameters.Enable_Default_Reporting = 0     #No InsetChart

    return config


################################################################################
# GENERIC PARAMETER SETTER
################################################################################

def set_param(simulation, param, value):
    return simulation.task.set_parameter(param, value)


################################################################################
# HABITAT SCALING
################################################################################

def scale_habitat_from_df(df=habitat_df, index=-1, vdf=None):
    habitat_mix = (1.0, 1.0, 1.0)
    if index >= 0:
        habitat_mix = df.iloc[index].tolist()

    if vdf is None:
        return None

    all_scales = {"NodeID": vdf["node_id"].unique()}

    for sp in vdf["species"].unique():
        species_df = vdf[vdf["species"] == sp]
        all_scales[f"CONSTANT.{sp}"] = [habitat_mix[0] * f for f in species_df["fraction"]]
        all_scales[f"TEMPORARY_RAINFALL.{sp}"] = [habitat_mix[1] * f for f in species_df["fraction"]]
        all_scales[f"WATER_VEGETATION.{sp}"] = [habitat_mix[2] * f for f in species_df["fraction"]]

    return pd.DataFrame(all_scales)


################################################################################
# PREGNANCY TRACKING
################################################################################
# Keeps the individual property "Pregnancy" in step with the built-in events:
#   'Pregnant'  -> IsPregnant
#   'GaveBirth' -> NotPregnant
# The pickup uses an IsPregnant restriction so a woman only receives an ANC
# net if she is still pregnant at the visit.

def add_pregnancy_tracking(campaign):
    common.change_individual_property_triggered(
        campaign,
        new_ip_key="Pregnancy", new_ip_value="IsPregnant",
        target_sex="Female", triggers=["Pregnant"],
        start_day=1, daily_prob=1, revert_in_days=-1, blackout=False,
    )
    common.change_individual_property_triggered(
        campaign,
        new_ip_key="Pregnancy", new_ip_value="NotPregnant",
        target_sex="Female", triggers=["GaveBirth"],
        start_day=1, daily_prob=1, revert_in_days=-1, blackout=False,
    )
    return campaign


################################################################################
# CASE MANAGEMENT CAMPAIGN
################################################################################
# One add_treatment_seeking() call per (start_day, duration, drug, severity)
# block, so all age bands sharing a block are targets of one event.
# Targets carry no 'seek' key -- it is already folded into coverage.
# Each block spans its full (collapsed) duration, which may be several years.

def add_case_management(campaign, node_ids=None):
    group_cols = ["start_day", "duration", "drug", "is_severe"]

    for (start_day, duration, drug, is_severe), group in cm_burnin_df.groupby(group_cols):

        if int(duration) <= 0:
            continue

        targets = [
            {
                "trigger": row.trigger,
                "coverage": float(row.effective_coverage),
                "agemin": float(row.age_min),
                "agemax": float(row.age_max),
                "rate": float(row.rate),
            }
            for row in group.itertuples()
        ]

        add_treatment_seeking(
            campaign,
            start_day=int(start_day),
            drug=DRUG_LOOKUP[drug],
            targets=targets,
            node_ids=node_ids,
            duration=int(duration),
            broadcast_event_name=(
                "Received_Severe_Treatment" if bool(is_severe) else "Received_Treatment"
            ),
        )

    return campaign


################################################################################
# BURN-IN CAMPAIGN
################################################################################

def fix_property_restrictions(obj):
    """Replace null property-restriction fields with the empty arrays EMOD's schema requires."""
    keys = [
        "Node_Property_Restrictions",
        "Property_Restrictions",
        "Property_Restrictions_Within_Node",
    ]

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


def build_camp(habitat=-1, scale_start_day=1):
    """Burn-in campaign: pregnancy tracking + larval-habitat scaling + historical case management."""
    camp.set_schema(manifest.schema_file)

    add_pregnancy_tracking(camp)

    lh_scales = scale_habitat_from_df(df=habitat_df, vdf=vector_df, index=habitat)
    add_scale_larval_habitats(camp, df=lh_scales, start_day=scale_start_day, repetitions=-1)

    add_case_management(camp)

    fix_property_restrictions(camp.campaign_dict)

    return camp


################################################################################
# DEMOGRAPHICS
################################################################################

def add_pregnancy_property(demog):
    """Add the Pregnancy individual property. The pickup must add the same one."""
    if hasattr(demog, "AddIndividualPropertyAndHINT"):
        demog.AddIndividualPropertyAndHINT(
            Property="Pregnancy",
            Values=["NotPregnant", "IsPregnant"],
            InitialDistribution=[1.0, 0.0],
        )
    else:
        # newer emod_api naming
        demog.add_individual_property(
            property="Pregnancy",
            values=["NotPregnant", "IsPregnant"],
            initial_distribution=[1.0, 0.0],
        )


def _final_demographics_config(config):
    """
    Last function in demog.implicits.

    emodpy applies demographics implicits after set_param_fn(). SetBirthRate()
    can overwrite Birth_Rate_Dependence and AddIndividualPropertyAndHINT adds
    Disable_IP_Whitelist, so both are fixed here, after them.
    """
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    remove_stale_params(config)
    return config


def build_demog():
    """Build the single-node UWR demographic object from demographics.csv."""
    site = _read_site_definition()

    demog = Demographics.from_template_node(
        lat=site["lat"],
        lon=site["lon"],
        pop=site["pop"],
        name=site["name"],
    )

    # BirthRate under INDIVIDUAL_PREGNANCIES is a per-possible-mother daily
    # rate. Do NOT pass CrudeRate() here.
    #demog.SetBirthRate(PREG_RATE_PER_MOTHER_DAY)
    demog.SetBirthRate(YearlyRate(PREG_RATE_PER_MOTHER_DAY * DAYS_PER_YEAR))
    demog.SetMortalityRate(CrudeRate(CRUDE_DEATH_RATE))

    # (SetEquilibriumAgeDistFromBirthAndMortRates is deliberately not called:
    # SetAgeDistribution below overrides it.)
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)

    add_pregnancy_property(demog)

    # Must be last so the birth model is not overwritten by the
    # SetBirthRate() implicit.
    demog.implicits.append(_final_demographics_config)

    return demog


################################################################################
# SWEEP CALLBACK: HABITAT
################################################################################

def update_campaign_multiple_parameters(simulation, habitat_index, scale_start_day):
    """Apply one larval-habitat composition from the calibration grid."""
    habitat_index = int(habitat_index)
    scale_start_day = int(scale_start_day)

    build_campaign_partial = partial(build_camp, habitat=habitat_index, scale_start_day=scale_start_day)
    simulation.task.create_campaign_from_callback(build_campaign_partial)

    # This callback runs after the task and every demographics implicit has
    # been built, so it has the final word on these two settings.
    simulation.task.set_parameter("Birth_Rate_Dependence", BIRTH_RATE_DEPENDENCE)
    remove_stale_params(simulation.task.config)

    return {
        "scale_constant": float(habitat_df.iloc[habitat_index, 0]),
        "scale_temp_rain": float(habitat_df.iloc[habitat_index, 1]),
        "scale_water_veg": float(habitat_df.iloc[habitat_index, 2]),
    }


################################################################################
# MAIN SIMULATION
################################################################################

def describe_settings():
    site = _read_site_definition()

    print("\nDemographic settings (must match the pickup):")
    print(f"  node pop (demographics.csv)  : {site['pop']}")
    print(f"  Birth_Rate_Dependence        : {BIRTH_RATE_DEPENDENCE}")
    print(f"  BirthRate per mother per day : {PREG_RATE_PER_MOTHER_DAY:g} "
          f"(~{PREG_RATE_PER_MOTHER_DAY * 365:.3f} pregnancies / woman / year)")
    print(f"  x_Birth                      : {X_BIRTH:g}")
    print(f"  crude death rate             : {CRUDE_DEATH_RATE:g} per 1,000 / year (flat)")

    print("\nVector behaviour (must match the pickup):")
    print(f"  APPLY_VECTOR_BEHAVIOUR       : {APPLY_VECTOR_BEHAVIOUR}")
    if APPLY_VECTOR_BEHAVIOUR:
        for r in vector_df.itertuples():
            print(f"    {r.species:<11} anthropophily {r.anthropophily:.2f}  "
                  f"indoor feeding {r.indoor_feeding:.2f}")
    else:
        print("    EMOD / team defaults used; anthropophily and indoor_feeding "
              "columns are ignored.")

    if TEST_MODE:
        print(f"\n*** BIRTH-RATE TEST: {TEST_YEARS} years, InsetChart ON, no serialization ***")
        print(f"  x_Birth sweep : {X_BIRTH_TEST}")
        print("  effective per-mother daily rate: "
              + ", ".join(f"{PREG_RATE_PER_MOTHER_DAY * x:.2e}" for x in X_BIRTH_TEST))
        print(f"  seeds         : {TEST_SEEDS}   total simulations: "
              f"{len(X_BIRTH_TEST) * TEST_SEEDS}")
        print(f"  habitat index {N_HABS // 2}, "
              f"xTLH {XTLH_RANGE[len(XTLH_RANGE) // 2]:.2f}")
    else:
        print(
            f"\nParameter grid: {len(XTLH_RANGE)} xTLH x {N_HABS} habitat x "
            f"{NUM_SEEDS} seeds = {EXPECTED_SIMULATIONS} simulations"
        )
        print("\nMalaria Summary Reports: DISABLED (commented out in add_burnin_reports)")
        print(f"Event recorder: {EVENT_LIST}")

    describe_case_management()


def add_burnin_reports(task):
    """Add event recorder + one malaria summary report per year (full burn-in only)."""
    if TEST_MODE:
        print("Birth-rate test: event recorder and Malaria Summary Reports disabled.")
        return task

    add_event_recorder(task,
                       event_list=EVENT_LIST,
                       start_day=1,
                       end_day=BURNIN_DAYS,
                       node_ids=[NODE_ID],
                       min_age_years=EVENT_MIN_AGE_YEARS,
                       max_age_years=EVENT_MAX_AGE_YEARS)

    # Malaria Summary Reports are switched off for this run.
    # Uncomment this block to turn them back on.
    # for year in range(REPORT_START_YEAR, REPORT_END_YEAR + 1):
    #     start_day = (year - BURNIN_START_YEAR) * DAYS_PER_YEAR
    #     end_day = start_day + DAYS_PER_YEAR
    #
    #     add_malaria_summary_report(task, manifest,
    #                                start_day=start_day,
    #                                end_day=end_day,
    #                                reporting_interval=REPORT_INTERVAL,
    #                                # 6-59 months, matching the prevalence surveys.
    #                                age_bins=REPORT_AGE_BINS,
    #                                max_number_reports=REPORT_MAX_NUMBER,
    #                                pretty_format=True,
    #                                filename_suffix=f"Monthly_U5_{year}")

    print(
        f"Event recorder configured for day 1-{BURNIN_DAYS}; "
        "Malaria Summary Reports are disabled."
    )
    return task


def general_sim(selected_platform):
    """Create and submit the UWR EMOD burn-in (or the birth-rate test)."""
    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    builder = SimulationBuilder()

    if TEST_MODE:
        # One mid-range habitat / xTLH, sweep only x_Birth and seeds.
        builder.add_sweep_definition(
            partial(set_param, param="x_Birth"), X_BIRTH_TEST
        )
        builder.add_sweep_definition(
            partial(set_param, param="x_Temporary_Larval_Habitat"),
            [XTLH_RANGE[len(XTLH_RANGE) // 2]],
        )
        builder.add_multiple_parameter_sweep_definition(
            update_campaign_multiple_parameters,
            dict(scale_start_day=[1], habitat_index=[N_HABS // 2]),
        )
        builder.add_sweep_definition(
            partial(set_param, param="Run_Number"), range(TEST_SEEDS)
        )
    else:
        # SWEEP 1: x_Temporary_Larval_Habitat
        builder.add_sweep_definition(
            partial(set_param, param="x_Temporary_Larval_Habitat"), XTLH_RANGE
        )

        # SWEEP 2: habitat composition
        builder.add_multiple_parameter_sweep_definition(
            update_campaign_multiple_parameters,
            dict(scale_start_day=[1], habitat_index=list(range(N_HABS))),
        )

        # SWEEP 3: stochastic seeds
        builder.add_sweep_definition(
            partial(set_param, param="Run_Number"), range(NUM_SEEDS)
        )

    describe_settings()

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
    task = add_burnin_reports(task)
    task.set_sif(manifest.SIF_PATH, platform)

    task.common_assets.add_directory(CLIMATE_DIRECTORY, relative_path="climate")

    print(f"\n[{SITE_NAME}] node_ids found in {DEMOGRAPHICS_FILE}: {get_site_node_ids()}")

    # create experiment from builder, prefixed with the login name
    user = os.getlogin()
    experiment_name = f"{user}_{EXPERIMENT_NAME}"
    print(f"Creating experiment: {experiment_name}")

    experiment = Experiment.from_builder(builder, task, name=experiment_name)
    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")

    if TEST_MODE:
        print("\nNext: read 'Statistical Population' from each InsetChart.json, judge "
              "the last 15 years, and pick the flattest x_Birth.")
    else:
        print("\nAfter the run: confirm in one simulation's folder that "
              "config.json has Birth_Rate_Dependence = INDIVIDUAL_PREGNANCIES and "
              "demographics.json has the BirthRate you set.")
        if APPLY_VECTOR_BEHAVIOUR:
            print("Also confirm config.json has the Anthropophily and "
                  "Indoor_Feeding_Fraction values from vector.csv for each species.")

    return experiment


################################################################################
# COMMAND LINE / RUN
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="UWR 1963-2012 burn-in.")

    parser.add_argument(
        "--test-birth",
        action="store_true",
        help=(
            "Run the short birth-rate calibration test (InsetChart on, "
            "x_Birth sweep, no serialization) instead of the full burn-in."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the settings and stop; nothing is submitted.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test_birth:
        TEST_MODE = True
        EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_BIRTHTEST"

    if args.dry_run:
        describe_settings()
        print("\nDry run OK. Nothing submitted.")
        raise SystemExit(0)

    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)

    selected_platform = "SLURM_LOCAL"
    general_sim(selected_platform)