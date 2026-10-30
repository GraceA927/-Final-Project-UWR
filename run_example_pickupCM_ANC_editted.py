"""
Upper West Region EMOD malaria model -- PICKUP 2013-2025

AIM 1: adapt and calibrate EMOD to the Upper West Region of Ghana.

    1963 ---------- 2012 | 2013 ---------- 2025
      50-year burn-in    |  pickup (this script)

              serialized state (state-18250.dtk) read here

Usage
-----
1. Preview settings and write the campaign without submitting:

       python run_pickup_UWR.py --dry-run

2. Submit ONE simulation (first burn-in state) to inspect config.json:

       python run_pickup_UWR.py --test-one

3. Full pickup (one simulation per selected burn-in state, 1:1):

       python run_pickup_UWR.py

Everything marked "MUST EQUAL run_burnin_UWR.py" below has to be identical
in the burn-in script, because config settings are NOT stored in the
serialized state.

Calendar
--------
Flat 365-day years, no leap days. Pickup day 1 = 1 Jan 2013, so the pickup
is 13 x 365 = 4,745 days. Weather day 1 must be 1 Jan 2013.

Burn-in link
------------
Each pickup simulation inherits, from the matching burn-in simulation:
the serialized state, Run_Number, x_Temporary_Larval_Habitat and the habitat
mix (scale_constant / scale_temp_rain / scale_water_veg). The habitat grid is
repeated below so it can be seen and checked against the experiment.

Birth model (IMPORTANT)
-----------------------
Births use INDIVIDUAL_PREGNANCIES so the built-in 'Pregnant' and 'GaveBirth'
events fire; ANC nets, IPTp and the Pregnancy property depend on them.
BirthRate is a per-possible-mother DAILY rate, not a crude rate. These
settings are not carried in the serialized state and must equal the burn-in.

ITN usage
---------
Net use on a night = age value x seasonal value. Age curves are by age in
years; seasonal curves are by calendar day of year and are rotated to each
net's delivery date because EMOD counts the seasonal map from distribution.

Routine CWC nets are delivered in ITN_SEASON_BATCH_DAYS-day batches so the
rotation is correct. Routine ANC nets are delivered as ONE annual window per
delivery year (ANC_ITN_BATCH_DAYS = 365), so every ANC net in a year has its
seasonal curve rotated to mid-year. Batches are only slices of the delivery
year: the CSV coverage is a YEARLY operational coverage and every batch applies
the same value, so nets / eligible over the year equals the CSV value.

Routine ITN coverage and event labels (ANC / CWC)
-------------------------------------------------
interventions_ITN_ANC.csv and interventions_ITN_CWC.csv already hold coverage
among the ELIGIBLE group, so it is applied exactly once to that group, at the
visit / target age (after the delay), never at conception or birth:

    ANC eligible = pregnant women who reach the visit (16 weeks after Pregnant)
    CWC eligible = children alive at the CWC age (18 months after Births)

    ANC: Pregnant -> +16 wk -> ANC_Visit (cov 1) - ANC_Net (cov = CSV,
         still pregnant) -> net
    CWC: Births   -> +18 mo -> CWC_Visit (cov 1) - CWC_Net (cov = CSV) -> net

Each stage broadcasts a batch event for routing and a zero-delay YEAR-level
relay (ANC_Visit_2016, ANC_Net_2016, CWC_Visit_2016, CWC_Net_2016). Only the
year-level events are recorded, so in ReportEventRecorder.csv
    ANC_Net_<yr> / ANC_Visit_<yr>  = yearly ANC coverage
    CWC_Net_<yr> / CWC_Visit_<yr>  = yearly CWC coverage
This needs only emod_api.interventions.common (no bednet broadcast_event).
Bednet_Got_New_One is still recorded as the all-channel total; mass nets and
CWC catch-up nets (scheduled, not triggered) appear only in that total.

Vaccine
-------
The vaccine CSV holds CUMULATIVE coverage by dose within each birth cohort
(year = birth year). The script derives conditional uptake
(D1 = its own coverage, Dk = cum(Dk) / cum(Dk-1)) and chains doses by actual
receipt of the previous dose. Cohorts that fail validation get no campaign,
and submission is refused unless --allow-excluded-vaccine-cohorts is given.
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

import emodpy_malaria.malaria_config as conf
from emodpy.emod_task import EMODTask
import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emod_api.demographics.PreDefinedDistributions as Distributions
from emod_api.demographics.DemographicsTemplates import CrudeRate, YearlyRate
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.interventions.treatment_seeking import add_treatment_seeking
import emod_api.campaign as camp
import emod_api.interventions.common as common
from emodpy_malaria.reporters.builtin import add_malaria_summary_report, add_event_recorder
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import manifest


# =============================================================================
# PROJECT / MODEL SETTINGS
# =============================================================================

SITE_NAME = "UpperWest_Region"
NODE_ID = 1
MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")


CLIMATE_ROOT = os.path.join(PROJECT_DIR, "climate", "pickup")
CLIMATE_SUBDIR = ""

VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"

# ---- CHECK THESE FILE NAMES AGAINST YOUR inputs/ FOLDER ---------------------
# A name that does not exist is skipped (the schedule printout says
# "file not found, skipped"), so a typo silently removes an intervention.
CM_FILE = "intervention_CMBN.csv"
ITN_FILE = "interventions_ITN.csv"
ITN_ANC = "interventions_ITN_ANC.csv"
ITN_CWC = "interventions_ITN_CWC.csv"
IRS_FILE = "intervention_IRS.csv"
SMC_FILE = "intervention_SMC.csv"
IPTP_FILE = "interventions_IPTp.csv"
VACCINE_FILE = "intervention_malariaVaccine.csv"


USE_ITN_USAGE_CURVES = True
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"
ITN_SEASON_BATCH_DAYS = 30
# -----------------------------------------------------------------------------

EXPERIMENT_NAME = "UWR_Pickup_2013_2025"

# From the burn-in experiment. It must be a burn-in run with the habitat grid

BURNIN_EXPERIMENT_ID = "a85e106f-fdd1-4a29-8a97-f571f02e6137"   # aadb65ab-e220-43b0-89ce-3d890db8cbee

BURNIN_FILTER = {}

# These values are inherited directly from each burn-in simulation.
HABITAT_TAGS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]

# Save state at the end of pickup.
WRITE_PICKUP_STATE = True

# Microscopy-like parasite detection threshold.
PARASITE_DETECTION_THRESHOLD = 40


# =============================================================================
# DEMOGRAPHIC SETTINGS
# =============================================================================

BIRTH_RATE_DEPENDENCE = "INDIVIDUAL_PREGNANCIES"

# Flat all-age mortality, per 1,000 per year.
CRUDE_DEATH_RATE = 10.9

# Pregnancy rate PER POSSIBLE MOTHER PER DAY (not a crude rate).
PREG_RATE_PER_MOTHER_DAY = 3.62e-4

X_BIRTH = 1.0

# One pregnancy per 40-week gestation is the most a woman can have.
MAX_PREG_RATE_PER_MOTHER_DAY = 1.0 / 280.0

# Config parameters the installed schema no longer defines.
STALE_CONFIG_PARAMS = ("Disable_IP_Whitelist",)


# =============================================================================
# VECTOR BEHAVIOUR  --  MUST EQUAL WHAT THE BURN-IN DID
# =============================================================================
# The burn-in does not set Anthropophily or Indoor_Feeding_Fraction, so the
# pickup must not either, or vector behaviour would change at 2013. Set True
# ONLY if the same two settings are also added to the burn-in's set_param_fn
# and the burn-in is re-run.
APPLY_VECTOR_BEHAVIOUR = False


# =============================================================================
# HABITAT SWEEP  --  MUST EQUAL run_burnin_UWR.py
# =============================================================================
# The pickup does NOT sweep these. Each simulation inherits its own values
# from the matching burn-in simulation's tags. They are listed here so they
# can be seen, and check_burnin_grid() confirms the burn-in used them.
SCALE_CONSTANT = [1.0]
SCALE_TEMP_RAIN = [0.5, 1.0, 2.5, 5.0]
SCALE_WATER_VEG = [0.0, 0.1, 0.5]
XTLH_RANGE = list(np.logspace(0.65, 1.85, 5))

NUM_SEEDS = 5   # must equal NUM_SEEDS burnin

habitat_df = pd.DataFrame(
    list(product(SCALE_CONSTANT, SCALE_TEMP_RAIN, SCALE_WATER_VEG)),
    columns=["scale_constant", "scale_temp_rain", "scale_water_veg"],
)
N_HABS = len(habitat_df)
# 5 x 12 x 5 = 300 with the grid above
EXPECTED_SIMULATIONS = len(XTLH_RANGE) * N_HABS * NUM_SEEDS


# REPORT SETTINGS

REPORT_INTERVAL = 30          # days per reporting period
REPORT_MAX_NUMBER = 13        # reports per yearly report (13 x 30 covers the 365-day year)
# 6-59 months, matching the prevalence surveys.
REPORT_AGE_BINS = [0.25, 0.5, 5, 15, 25, 100]

# Events to monitor across all ages and both sexes (same list as the burn-in).
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

# The pickup records ALL ages: the vaccine, SMC and CWC events are in children.
EVENT_MIN_AGE_YEARS = 0
EVENT_MAX_AGE_YEARS = 115

# Built-in events that drive ANC, IPTp, CWC and the vaccine. Set to False to
# leave Pregnant / GaveBirth / Births out of the recorder.
RECORD_TRIGGER_EVENTS = True
TRIGGER_EVENTS = ("Pregnant", "GaveBirth", "Births")

# CALENDAR

DAYS_PER_YEAR = 365
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_DAYS = (BURNIN_END_YEAR - BURNIN_START_YEAR + 1) * DAYS_PER_YEAR

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1
PICKUP_DAYS = PICKUP_YEARS * DAYS_PER_YEAR


# ROUTINE ITN TIMING


# Approximately 16 weeks after conception.
ANC_DELAY_DAYS = 16 * 7

# ANC nets are delivered as ONE window per delivery year (365-day batch).
# Coverage is applied once at the visit, so this gives the same expected
# coverage as 30-day batches; only the seasonal-curve rotation changes (every
# ANC net of a year is rotated to mid-year, ~1 July).
# CWC keeps ITN_SEASON_BATCH_DAYS-day batches.
ANC_ITN_BATCH_DAYS = DAYS_PER_YEAR

# If True, ANC nets are only given to women who are STILL pregnant when the
# net is delivered. The check sits on the ANC_Net stage, which runs at the
# visit (16 weeks after Pregnant), not at conception:
#
#   Pregnant -> wait 16 weeks -> ANC_Visit_<year>_<batch>   (eligible, cov 1)
#   ANC_Visit -> ANC_Net_<year>_<batch>  (cov = CSV, Pregnancy=IsPregnant)
#   ANC_Net   -> ITN (cov 1)
#
# If False, no pregnancy check is applied at the ANC_Net stage.
ANC_CHECK_PREGNANT_AT_DELIVERY = True

# Approximately 18 months after birth.
CWC_AGE_MONTHS = 18
CWC_DELAY_DAYS = int(round(CWC_AGE_MONTHS * DAYS_PER_MONTH))


# CALENDAR HELPERS

MONTH_OFFSET = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])

MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]


def month_number(value):
    """Convert a numeric or textual month to 1-12."""
    text = str(value).strip().lower()

    if text.replace(".", "").isdigit():
        return int(float(text))

    if text[:3] in MONTH_NAMES:
        return MONTH_NAMES.index(text[:3]) + 1

    raise ValueError(f"Unrecognised month '{value}'.")


def month_of_doy(doy):
    """Return month number from day of year."""
    return int(np.searchsorted(MONTH_OFFSET, int(doy) - 1, side="right"))


def fmt_day(pickup_day):
    """Convert pickup day to YYYY-MM-DD using the flat 365-day calendar."""
    day = int(pickup_day)
    year = PICKUP_START_YEAR + (day - 1) // DAYS_PER_YEAR
    doy = (day - 1) % DAYS_PER_YEAR + 1
    month = month_of_doy(doy)
    day_in_month = doy - MONTH_OFFSET[month - 1]
    return f"{year}-{month:02d}-{day_in_month:02d}"


# =============================================================================
# INTERVENTION DEFAULTS
# =============================================================================

# Mirrors DRUG_LOOKUP in run_burnin_UWR.py. The pickup rows use AL.
CM_DRUGS = {
    "AL": ["Artemether", "Lumefantrine"],
    "ASAQ": ["Artesunate", "Amodiaquine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}

SEVERE_TRIGGERS = ["NewSevereCase"]


# CSV COLUMNS
# ITN, IRS, SMC and vaccine parameters are read from their CSV files; there
# are no defaults here. A missing column or blank cell is an error.

# IRS: internal name -> CSV column
IRS_PARAM_COLUMNS = {
    "killing_initial": "killing_initial_effect",
    "killing_box": "killing_box_duration",
    "killing_decay": "killing_decay_time_constant",
    "repelling_initial": "repelling_initial_effect",
    "repelling_box": "repelling_box_duration",
    "repelling_decay": "repelling_decay_time_constant",
}


# SMC

SMC_DRUG_ALIASES = {"SPAQ": "SPA", "SP+AQ": "SPA", "SP-AQ": "SPA"}

# Fallbacks are used ONLY when a parameter column is absent from its CSV.
# If the column exists, its row values are authoritative and blanks are errors.
SMC_NONCSV_DEFAULTS = dict(age_min=0.25, age_max=5.0, drug_code="SPA")
VACCINE_NONCSV_DEFAULTS = dict(
    initial_effect=0.8,
    box_duration=0,
    decay=592,
    vaccine_type="AcquisitionBlocking",
)


# IPTp

# Protocol constants (not per-year data, so not read from the CSV).
IPTP_DEFAULTS = dict(
    drug_code="SP",   # Sulfadoxine + Pyrimethamine
    trigger="Pregnant",
    delay_days=16 * 7,   # trigger -> dose 1
    gap_days=30,         # previous dose -> next dose
)

IPTP_COLUMN_RE = re.compile(r"^iptp(\d+)_coverage$")


# VACCINE

VACCINE_CATCHUP_MAX_AGE_MONTHS = 59.0

# Doses expected in the vaccine CSV, in schedule order.
VACCINE_EXPECTED_DOSES = ("D1", "D2", "D3", "D4")

# Age lower bound on the triggered doses. True keeps the existing behaviour
# (min age = target age). If a --test-one run records NO Received_Vaccine_D1
# or D2 events, the age bound is excluding children (EMOD may check it when
# the trigger fires, or exactly at the target age): set this to False to use
# a lower bound of 0 and rely on the trigger delay alone. The upper bound
# (VACCINE_CATCHUP_MAX_AGE_MONTHS) is kept either way.
VACCINE_USE_AGE_FLOOR = True

# Cohorts that failed validation: {year: [messages]}. Filled by load_vaccine().
VACCINE_ISSUES = {}


# CSV HELPERS

def read_clean_csv(path):
    """Read CSV as text, clean headers and parse numeric columns."""
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    df = (
        df.apply(lambda s: s.str.strip())
        .replace("", np.nan)
        .dropna(how="all")
    )

    for col in df.columns:
        parsed = pd.to_numeric(df[col], errors="coerce")
        if parsed.notna().sum() == df[col].notna().sum():
            df[col] = parsed

    return df.reset_index(drop=True)


def pick(df, names, default=None, label=""):
    """
    Return first existing column in names, or a default-filled column.

    Only for things that are not model parameters (for example `round`, or
    the optional discard columns). Model parameters use need().
    """
    for name in names:
        if name in df.columns:
            return df[name]

    if default is None:
        raise ValueError(
            f"{label}: needs one of the columns "
            f"{names}; found {list(df.columns)}"
        )

    return pd.Series([default] * len(df), index=df.index)


def need(df, names, label, kind=float):
    """
    Read a REQUIRED CSV column (the first of `names` that exists).

    Nothing is defaulted: a missing column, a blank cell or a value of the
    wrong type raises an error naming the file, the column and the year(s).
    """
    col = next((n for n in names if n in df.columns), None)

    if col is None:
        raise ValueError(
            f"{label}: needs a column named one of {list(names)}; "
            f"found {list(df.columns)}. Add it to the CSV."
        )

    series = df[col]

    if series.isna().any():
        years = (
            sorted({int(y) for y in df.loc[series.isna(), "year"]})
            if "year" in df.columns else []
        )
        where = f" for year(s) {years}" if years else ""
        raise ValueError(f"{label}: column `{col}` is blank{where}.")

    try:
        return series.astype(kind)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label}: column `{col}` must be {kind.__name__}; "
            f"found {list(pd.unique(series))[:5]}."
        )


def parameter_or_default(df, names, default, label, kind=float):
    """Use a CSV column when present; use a constant only if no alias exists."""
    if any(name in df.columns for name in names):
        return need(df, names, label, kind)
    return pd.Series([default] * len(df), index=df.index, dtype=object).astype(kind)


def keep_node(df):
    if "node_id" in df.columns:
        df = df[df["node_id"].astype(int) == NODE_ID]
    return df


def keep_pickup_phase(df):
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == "pickup"]
    return df


def add_pickup_day(df, label, default_day_note=None):
    """
    Add pickup_day where day 1 = 1 Jan 2013,
    and discard rows outside 2013-2025.
    """
    df = df.copy()

    if "year" not in df.columns:
        raise ValueError(
            f"{label}: needs a `year` column; found {list(df.columns)}"
        )

    year = df["year"].astype(int)

    if "day_of_year" in df.columns:
        doy = df["day_of_year"].astype(int)

    elif "start_day" in df.columns:
        raw = df["start_day"].astype(int)
        doy = (raw - 1) % DAYS_PER_YEAR + 1

        # start_day above 365 is an absolute pickup day. If it falls in a
        # different calendar year from `year`, say so (year still wins).
        implied = PICKUP_START_YEAR + (raw - 1) // DAYS_PER_YEAR

        bad = df.index[(raw > DAYS_PER_YEAR) & (implied != year)]
        offsets = sorted({int(year[i] - implied[i]) for i in bad})

        if len(bad) and len(offsets) == 1:
            i = bad[0]
            used = (year[i] - PICKUP_START_YEAR) * DAYS_PER_YEAR + doy[i]
            k = offsets[0]
            print(
                f"  WARNING {label}: {len(bad)} row(s) have start_day "
                f"{abs(k)} year(s) {'before' if k > 0 else 'after'} `year` "
                f"on a calendar that starts 1 Jan {PICKUP_START_YEAR} "
                f"(e.g. year {year[i]}, start_day {raw[i]} = "
                f"{fmt_day(raw[i])}). If start_day is counted from "
                f"1 Jan {PICKUP_START_YEAR + k} that is expected. `year` is "
                f"authoritative, so the day-of-year is used "
                f"(e.g. {fmt_day(used)})."
            )
        else:
            for i in bad:
                used = (year[i] - PICKUP_START_YEAR) * DAYS_PER_YEAR + doy[i]
                print(
                    f"  WARNING {label}: year {year[i]} but start_day "
                    f"{raw[i]} is {fmt_day(raw[i])} on the pickup calendar. "
                    f"`year` is authoritative, so day-of-year {doy[i]} of "
                    f"{year[i]} ({fmt_day(used)}) is used. Check the "
                    f"start_day."
                )

    elif "month" in df.columns:
        doy = df["month"].map(month_number).map(lambda m: MONTH_OFFSET[m - 1] + 1)

    else:
        doy = pd.Series(1, index=df.index)

        if ((year >= PICKUP_START_YEAR) & (year <= PICKUP_END_YEAR)).any():
            note = default_day_note or "using 1 Jan of each year."
            print(
                f"  WARNING {label}: "
                f"no day_of_year/start_day/month column; {note}"
            )

    # Check month consistency when possible.
    if "month" in df.columns and (
        "day_of_year" in df.columns or "start_day" in df.columns
    ):
        stated = df["month"].map(month_number)
        derived = doy.map(month_of_doy)

        for i in df.index[stated != derived]:
            print(
                f"  WARNING {label}: "
                f"year {year[i]} says month {df.at[i, 'month']} but "
                f"start_day gives month {derived[i]}; using start_day."
            )

    df["year"] = year
    df["pickup_day"] = (year - PICKUP_START_YEAR) * DAYS_PER_YEAR + doy.astype(int)

    in_window = (df["pickup_day"] >= 1) & (df["pickup_day"] <= PICKUP_DAYS)

    return df[in_window].copy(), int((~in_window).sum())


def check_fraction(series, label):
    if ((series < 0) | (series > 1)).any():
        raise ValueError(f"{label}: coverage values must be between 0 and 1.")


def has_rows(df):
    return df is not None and not df.empty


# INTERVENTION LOADING

def load_case_management():
    df = read_clean_csv(os.path.join(INPUT_DIR, CM_FILE))

    required = [
        "phase", "duration", "trigger", "age_min", "age_max",
        "coverage", "seek", "rate", "drug", "year",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{CM_FILE} is missing columns: {missing}")

    df = keep_node(keep_pickup_phase(df))
    df["drug"] = df["drug"].astype(str).str.upper()

    df, dropped = add_pickup_day(df, CM_FILE)

    unknown = sorted(set(df["drug"]) - set(CM_DRUGS))
    if unknown:
        raise ValueError(f"{CM_FILE}: drug code(s) {unknown} not in CM_DRUGS.")

    df["requested_duration"] = df["duration"].astype(int)
    df["duration"] = np.minimum(
        df["requested_duration"],
        PICKUP_DAYS - df["pickup_day"] + 1,
    )

    df["effective_coverage"] = df["coverage"].astype(float) * df["seek"].astype(float)
    check_fraction(df["effective_coverage"], f"{CM_FILE} (coverage x seek)")

    df["is_severe"] = df["trigger"].isin(SEVERE_TRIGGERS)

    return df, dropped


def load_generic(filename, default_day_note=None):
    path = os.path.join(INPUT_DIR, filename)

    if not os.path.isfile(path):
        return None, 0

    return add_pickup_day(
        keep_pickup_phase(keep_node(read_clean_csv(path))),
        filename,
        default_day_note=default_day_note,
    )


def fill_itn_columns(df, label):
    """
    Read ITN effect parameters (blocking, killing, discard) from the CSV.

    Age- and season-dependent USE is handled separately; see
    "ITN USAGE CURVES" below.
    """
    df["cov"] = need(df, ["coverage", "cov_all"], label)

    df["blocking_initial"] = need(
        df, ["block_effect", "blocking_initial"], label
    )
    df["blocking_decay"] = need(df, ["block_decay", "blocking_decay"], label)
    df["killing_initial"] = need(df, ["kill_effect", "killing_initial"], label)
    df["killing_decay"] = need(df, ["kill_decay", "killing_decay"], label)

    # Without discard columns (or with blank cells) nets are not discarded.
    df["discard_k"] = pick(df, ["discard_k"], np.nan).astype(float)
    df["discard_l"] = pick(df, ["discard_l"], np.nan).astype(float)

    check_fraction(df["cov"], label)

    return df


def load_itn():
    # Mass campaigns disabled (ITN_FILE is None): nothing to load.
    if ITN_FILE is None:
        return None, 0

    df, dropped = load_generic(ITN_FILE)

    if not has_rows(df):
        return df, dropped

    return fill_itn_columns(df, ITN_FILE), dropped


def load_routine_itn(filename, channel):
    """
    Load routine ANC/CWC ITNs.

    `coverage` is the YEARLY operational coverage among the ELIGIBLE group
    (ANC: pregnant women reaching the visit; CWC: children at the CWC age).
    It is applied once to that group; no registration column is used.

    IMPORTANT:
    The `year` column is authoritative. Any start_day column is
    deliberately ignored.

    Thus if the first row in the file is 2016, routine ITN rollout
    begins in 2016. This function does not infer an earlier rollout.
    """
    path = os.path.join(INPUT_DIR, filename)

    if not os.path.isfile(path):
        return None, 0

    df = keep_pickup_phase(keep_node(read_clean_csv(path)))

    if "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == channel]

    if "year" not in df.columns:
        raise ValueError(
            f"{filename}: needs a `year` column; found {list(df.columns)}"
        )

    df = df.copy()
    df["year"] = df["year"].astype(int)

    in_window = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    dropped = int((~in_window).sum())
    df = df[in_window].reset_index(drop=True)

    if df["year"].duplicated().any():
        raise ValueError(
            f"{filename}: more than one row per year for channel {channel}."
        )

    # First day of each delivery year.
    df["year_start_day"] = (df["year"] - PICKUP_START_YEAR) * DAYS_PER_YEAR + 1

    if len(df):
        df = fill_itn_columns(df, filename)

    return df, dropped


# Scheduled CWC catch-up is delivered in batches of this many days.
CATCHUP_BATCH_DAYS = 30


def cwc_catchup_span(year_start_day):
    """
    Delivery days of a CWC year that no Births trigger can serve.

    A Births event on day >= 1 delivers on day >= 1 + CWC_DELAY_DAYS, so
    deliveries on days 1..CWC_DELAY_DAYS are for children already born.
    Returns (first_day, last_day) or None.
    """
    first = int(year_start_day)
    last = min(first + DAYS_PER_YEAR - 1, CWC_DELAY_DAYS)

    return (first, last) if first <= last else None


def cwc_catchup_batches(year_start_day):
    """Split the catch-up span into (batch_start, batch_end) day pairs."""
    span = cwc_catchup_span(year_start_day)

    if span is None:
        return []

    first, last = span

    return [
        (b, min(b + CATCHUP_BATCH_DAYS - 1, last))
        for b in range(first, last + 1, CATCHUP_BATCH_DAYS)
    ]


# ITN USAGE CURVES (age and season)
#
# Net use on a night = age value x seasonal value (see module docstring).
#
#   age_dependence       {"Times": [age in years], "Values": [0-1]}
#   seasonal_dependence  {"Times": [days since net delivery], "Values": [0-1]}
#
# EMOD's WaningEffectMapLinearSeasonal measures Times from the day the net was
# distributed and repeats every 365 days, so the calendar curve from the CSV is
# rotated to each net's delivery date here.

# Loaded in the main block (like SCHEDULES).
ITN_USAGE = None

ITN_USAGE_CHANNELS = ("MASS", "ANC", "CWC")

USAGE_VALUE_COLUMNS = [
    "age_usage", "season_usage",
    "usage", "value", "use", "usage_prob", "probability", "prob",
]
USAGE_AGE_COLUMNS = ["age", "age_years", "age_yr"]
# Calendar day of year, 1-365 (1 = 1 Jan).
USAGE_DAY_COLUMNS = ["day_of_year", "doy", "day"]

# Days since 1 Jan, 0-based (0 = 1 Jan, 90 = 1 Apr, 182 = 1 Jul).
USAGE_SEASON_TIME_COLUMNS = ["season_time"]

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

MAX_USAGE_AGE_YEARS = 125


def doy_of_day(day):
    """Day of year (1-365) of a pickup day (day 1 = 1 Jan)."""
    return (int(day) - 1) % DAYS_PER_YEAR + 1


def month_mid_doy(month):
    """Mid-month day of year for a month number or name."""
    m = month_number(month)
    return float(MONTH_OFFSET[m - 1]) + (MONTH_DAYS[m - 1] + 1) / 2.0


def itn_usage_label():
    return "age x season curves" if USE_ITN_USAGE_CURVES else "FLAT"


def _read_usage_table(filename, kind):
    """Read and validate an ITN usage file. kind is 'age' or 'season'."""
    path = os.path.join(INPUT_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"USE_ITN_USAGE_CURVES is True but {filename} was not found in "
            f"{INPUT_DIR}."
        )

    df = keep_pickup_phase(keep_node(read_clean_csv(path))).copy()

    if df.empty:
        raise ValueError(
            f"{filename}: no rows for node {NODE_ID} in the pickup phase."
        )

    value_col = next((c for c in USAGE_VALUE_COLUMNS if c in df.columns), None)

    if value_col is None:
        raise ValueError(
            f"{filename}: needs a usage column named one of "
            f"{USAGE_VALUE_COLUMNS}; found {list(df.columns)}"
        )

    usage = pd.to_numeric(df[value_col], errors="coerce")

    if usage.isna().any():
        raise ValueError(
            f"{filename}: `{value_col}` has blank or non-numeric entries."
        )

    check_fraction(usage, f"{filename} ({value_col})")
    df["_usage"] = usage.astype(float)

    if kind == "age":
        col = next((c for c in USAGE_AGE_COLUMNS if c in df.columns), None)

        if col is None:
            raise ValueError(
                f"{filename}: needs an age column (years) named one of "
                f"{USAGE_AGE_COLUMNS}; found {list(df.columns)}"
            )

        x = pd.to_numeric(df[col], errors="coerce")

        if x.isna().any() or (x < 0).any():
            raise ValueError(f"{filename}: `{col}` must be numeric and >= 0.")

    else:
        day_col = next((c for c in USAGE_DAY_COLUMNS if c in df.columns), None)
        time_col = next(
            (c for c in USAGE_SEASON_TIME_COLUMNS if c in df.columns), None
        )

        if day_col is not None:
            x = pd.to_numeric(df[day_col], errors="coerce")
        elif time_col is not None:
            # 0-based days since 1 Jan -> 1-based day of year.
            x = pd.to_numeric(df[time_col], errors="coerce") + 1

            # A closing row at season_time = 365 repeats season_time = 0.
            wrap = x > DAYS_PER_YEAR
            if wrap.any() and (x == 1).any():
                keep = np.logical_not(wrap)
                df = df[keep].copy()
                usage = usage[keep]
                x = x[keep]
        elif "month" in df.columns:
            x = df["month"].map(month_mid_doy).astype(float)
        else:
            raise ValueError(
                f"{filename}: needs a day column named one of "
                f"{USAGE_DAY_COLUMNS + USAGE_SEASON_TIME_COLUMNS}, "
                f"or a `month` column; "
                f"found {list(df.columns)}"
            )

        if x.isna().any() or (x < 1).any() or (x > DAYS_PER_YEAR).any():
            raise ValueError(
                f"{filename}: day of year must be numeric, 1-{DAYS_PER_YEAR}."
            )

    df["_x"] = x.astype(float)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "channel" in df.columns:
        ch = df["channel"]
        df["channel"] = ch.where(ch.isna(), ch.astype(str).str.upper())

        bad = sorted(set(df["channel"].dropna()) - set(ITN_USAGE_CHANNELS))

        if bad:
            raise ValueError(
                f"{filename}: unknown channel(s) {bad}; "
                f"use {list(ITN_USAGE_CHANNELS)} or leave blank."
            )

    return df.reset_index(drop=True)


def load_itn_usage():
    return {
        "age": _read_usage_table(ITN_AGE_FILE, "age"),
        "season": _read_usage_table(ITN_SEASON_FILE, "season"),
    }


_USAGE_WARNED = set()


def _latest_block(d, year, label=""):
    """
    Rows of the latest dated block <= year; blank-year rows if none.

    If every block is dated later than `year` (for example curves start in
    2014 but nets are delivered in 2013), the earliest block is used and a
    WARNING is printed once.
    """
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
            print(
                f"  WARNING ITN {label} usage: the earliest curve is for "
                f"{int(first)}; it is also used for earlier delivery years."
            )
            _USAGE_WARNED.add((label, first))

        return d[d["year"] == first]

    return d


def _select_usage_rows(table, channel, year, label):
    """
    Rows that apply to a channel and delivery year.

    Channel-specific rows win. If none apply to this year (for example a
    channel curve that only starts in 2016), the blank-channel rows are used.
    Within a group, the latest `year` <= delivery year wins; blank-year rows
    apply when no dated block does.
    """
    if "channel" in table.columns:
        rows = _latest_block(table[table["channel"] == channel], year, label)

        if rows.empty:
            rows = _latest_block(table[table["channel"].isna()], year, label)
    else:
        rows = _latest_block(table, year, label)

    if rows.empty:
        raise ValueError(
            f"No ITN {label} usage rows for channel {channel}, year {year}."
        )

    return rows.sort_values("_x")


def usage_age_curve(channel, year):
    """age_dependence dict for nets of this channel delivered in `year`."""
    rows = _select_usage_rows(ITN_USAGE["age"], channel, int(year), "age")
    ages = rows["_x"].tolist()
    vals = rows["_usage"].tolist()

    if len(set(ages)) != len(ages):
        raise ValueError(
            f"{ITN_AGE_FILE}: duplicate ages for channel {channel}, year {year}."
        )

    # Make the curve defined from birth to the maximum age (125 y), holding
    # the first / last value, so nothing depends on EMOD's end behaviour.
    if ages[0] > 0:
        ages.insert(0, 0.0)
        vals.insert(0, vals[0])

    if ages[-1] < MAX_USAGE_AGE_YEARS:
        ages.append(float(MAX_USAGE_AGE_YEARS))
        vals.append(vals[-1])

    return {"Times": ages, "Values": vals}


def _cyclic_interp(days, vals, x):
    """Linear interpolation on a 365-day circle."""
    xs = np.array(
        list(days)
        + [d + DAYS_PER_YEAR for d in days]
        + [d - DAYS_PER_YEAR for d in days],
        dtype=float,
    )
    vs = np.array(list(vals) * 3, dtype=float)
    order = np.argsort(xs)

    return float(np.interp(x, xs[order], vs[order]))


def usage_season_curve(channel, year, delivery_doy):
    """
    seasonal_dependence dict for a net delivered on `delivery_doy`.

    The CSV curve is by calendar day of year. EMOD's seasonal map counts days
    SINCE DELIVERY, so calendar day c maps to time (c - delivery_doy) mod 365.
    The map is anchored at 0 and 365 with the same (cyclically interpolated)
    value so it wraps smoothly.
    """
    rows = _select_usage_rows(ITN_USAGE["season"], channel, int(year), "season")
    days = rows["_x"].tolist()
    vals = rows["_usage"].tolist()

    if len(set(days)) != len(days):
        raise ValueError(
            f"{ITN_SEASON_FILE}: duplicate days for channel {channel}, "
            f"year {year}."
        )

    start = float(delivery_doy)

    shifted = {
        round((d - start) % DAYS_PER_YEAR, 6): v for d, v in zip(days, vals)
    }
    shifted.pop(float(DAYS_PER_YEAR), None)

    v0 = _cyclic_interp(days, vals, start)
    shifted[0.0] = v0

    times = sorted(shifted)
    values = [shifted[t] for t in times]

    times.append(float(DAYS_PER_YEAR))
    values.append(v0)

    return {"Times": times, "Values": values}


def itn_usage_kwargs(channel, year, delivery_day):
    """
    age_dependence / seasonal_dependence kwargs for one bednet call.

    delivery_day is the pickup day the net is (or, for batched routine nets,
    is centred to be) delivered.
    """
    if not USE_ITN_USAGE_CURVES:
        return dict(age_dependence=None, seasonal_dependence=None)

    return dict(
        age_dependence=usage_age_curve(channel, year),
        seasonal_dependence=usage_season_curve(
            channel, year, doy_of_day(delivery_day)
        ),
    )


def validate_itn_usage(sched):
    """Fail early if any ITN row has no usage curve."""
    if not USE_ITN_USAGE_CURVES:
        return

    for key, channel in (("itn", "MASS"), ("itn_anc", "ANC"), ("itn_cwc", "CWC")):
        df = sched[key]

        if not has_rows(df):
            continue

        for y in sorted({int(v) for v in df["year"]}):
            usage_age_curve(channel, y)
            usage_season_curve(channel, y, 1)


def describe_itn_usage(sched):
    print("\nITN usage configuration:")

    if not USE_ITN_USAGE_CURVES:
        print("  FLAT: age_dependence = None, seasonal_dependence = None")
        return

    print(f"  age curves:    {ITN_AGE_FILE}")
    print(f"  season curves: {ITN_SEASON_FILE}")
    print("  net use = age value x seasonal value")
    print(
        f"  seasonal curve rotated to each delivery date; routine CWC nets "
        f"delivered in {ITN_SEASON_BATCH_DAYS}-day batches, routine ANC nets "
        f"in {ANC_ITN_BATCH_DAYS}-day (annual) batches"
    )

    for key, channel in (("itn", "MASS"), ("itn_anc", "ANC"), ("itn_cwc", "CWC")):
        df = sched[key]

        if not has_rows(df):
            continue

        print(f"  {channel}:")
        last = None

        for y in sorted({int(v) for v in df["year"]}):
            age_rows = _select_usage_rows(ITN_USAGE["age"], channel, y, "age")
            sea_rows = _select_usage_rows(ITN_USAGE["season"], channel, y, "season")

            ages = list(zip(age_rows["_x"], age_rows["_usage"]))
            seas = list(zip(sea_rows["_x"], sea_rows["_usage"]))

            if (ages, seas) == last:
                continue

            last = (ages, seas)

            print(
                f"    from {y}: age  "
                + "  ".join(f"{a:g}y:{v:.2f}" for a, v in ages)
            )
            print(
                f"    {' ' * len(f'from {y}:')} season  "
                + "  ".join(f"doy{d:g}:{v:.2f}" for d, v in seas)
            )
            print(
                f"    {' ' * len(f'from {y}:')} peak combined use "
                f"{max(v for _, v in ages) * max(v for _, v in seas):.2f}"
            )


# ROUTINE ITN EVENT NAMES AND BATCHING

def routine_event_name(kind, year, batch):
    """
    Batch-level routing event (internal, not recorded).
    kind: ANC_Visit | ANC_Net | CWC_Visit | CWC_Net -> e.g. ANC_Net_2016_04
    """
    return f"{kind}_{int(year)}_{int(batch):02d}"


def yearly_event_name(kind, year):
    """
    Year-level event (recorded). The batch events are relayed to this name
    with coverage 1 and no delay, so a yearly count is the sum over batches.
    kind: ANC_Visit | ANC_Net | CWC_Visit | CWC_Net -> e.g. ANC_Net_2016
    """
    return f"{kind}_{int(year)}"


def routine_batch_size(channel):
    """
    Batch length (days) for a routine-net channel.

    ANC is delivered as one annual window (ANC_ITN_BATCH_DAYS). CWC uses
    ITN_SEASON_BATCH_DAYS when usage curves are on, else one annual window.
    """
    if channel == "ANC":
        return ANC_ITN_BATCH_DAYS

    return ITN_SEASON_BATCH_DAYS if USE_ITN_USAGE_CURVES else DAYS_PER_YEAR


def routine_delivery_batches(year_start_day, delay, size=None):
    """
    Split a routine-net delivery year into trigger windows.

    Returns [(batch, start_day, listening_duration, mid_delivery_day)].

    Nets in a batch are delivered on days d0..d1 of the pickup, so the
    trigger window is d0-delay .. d1-delay. mid_delivery_day is used to
    rotate the seasonal curve. `size` is the batch length in days; use
    routine_batch_size(channel) to get the right value for a channel. If
    omitted, the CWC/default size is used.

    The batches partition the delivery year with no gaps or overlaps, so
    every eligible person reaching the visit in the year falls in exactly
    one batch and the yearly coverage is preserved.
    """
    if size is None:
        size = routine_batch_size("CWC")

    first = int(year_start_day)
    last = first + DAYS_PER_YEAR - 1

    batches = []
    k = 0

    for d0 in range(first, last + 1, size):
        d1 = min(d0 + size - 1, last)
        k += 1

        window_start = d0 - delay
        window_end = d1 - delay + 1     # exclusive

        if window_end <= 1:
            continue

        start = max(1, window_start)

        batches.append((k, start, window_end - start, (d0 + d1) // 2))

    return batches


# IRS

def normalise_product(name):
    parts = [
        " ".join(p.replace("-", " ").replace("_", " ").lower().split())
        for p in str(name).split("+")
    ]
    return " + ".join(sorted(p for p in parts if p))


def load_irs():
    df, dropped = load_generic(IRS_FILE)

    if not has_rows(df):
        return df, dropped

    df["cov"] = need(df, ["coverage"], IRS_FILE)
    df["reps"] = need(df, ["repetitions"], IRS_FILE, int)
    df["rep_gap"] = need(df, ["timesteps_between_repetitions"], IRS_FILE, int)

    # Every effect parameter comes from this row's own CSV columns.
    for internal, col in IRS_PARAM_COLUMNS.items():
        df[internal] = need(df, [col], IRS_FILE)

    if "insecticide" not in df.columns:
        df["insecticide"] = ""

    df["insecticide"] = df["insecticide"].fillna("").astype(str)

    # Keep each source row as one IRS campaign.
    # Semicolon-separated formulations were co-applied to the same
    # households, so do NOT split the campaign or divide its coverage.
    for _, r in df.iterrows():
        products = [p.strip() for p in r["insecticide"].split(";") if p.strip()]
        if len(products) > 1:
            print(
                f"  INFO {IRS_FILE}: {int(r['year'])} lists "
                f"{len(products)} co-applied formulations; "
                f"keeping one campaign at coverage {r['cov']:.3f}."
            )

    df["iv_name"] = [
        "IRS_"
        + (
            normalise_product(p).replace(";", "_and_").replace(" + ", "_plus_").replace(" ", "_")
            or "default"
        )
        for p in df["insecticide"]
    ]

    check_fraction(df["cov"], IRS_FILE)

    year_total = df.groupby("year")["cov"].sum()

    if (year_total > 1).any():
        raise ValueError(
            f"{IRS_FILE}: product coverages add up to more than 1 "
            f"in year(s) {list(year_total[year_total > 1].index)}."
        )

    return df, dropped


# SMC

def load_smc():
    df, dropped = load_generic(SMC_FILE)

    if not has_rows(df):
        return df, dropped

    df["cov"] = need(df, ["coverage"], SMC_FILE)
    # Current intervention_SMC.csv defines coverage/timing only. Keep age bounds
    # and drug code as constants unless the CSV explicitly supplies those columns.
    df["amin"] = parameter_or_default(
        df, ["age_min", "agemin"], SMC_NONCSV_DEFAULTS["age_min"], SMC_FILE
    )
    df["amax"] = parameter_or_default(
        df, ["age_max", "agemax"], SMC_NONCSV_DEFAULTS["age_max"], SMC_FILE
    )

    codes = parameter_or_default(
        df, ["drug_code", "drug"], SMC_NONCSV_DEFAULTS["drug_code"], SMC_FILE, str
    ).str.upper()
    df["drug_code"] = codes.replace(SMC_DRUG_ALIASES)

    df["round"] = pick(df, ["round"], 0).astype(int)

    check_fraction(df["cov"], SMC_FILE)

    return df, dropped


# IPTp

def iptp_dose_columns(df):
    """Return [(dose_number, column)] for iptpN_coverage columns, in order."""
    found = []

    for c in df.columns:
        m = IPTP_COLUMN_RE.match(c)

        if m:
            found.append((int(m.group(1)), c))

    return sorted(found)


def iptp_dose_plan(row, dose_cols, year):
    """
    Convert cumulative dose coverage into per-dose conditional coverage.

    Returns [(dose, cumulative, conditional)]. Stops at the first blank
    dose. Conditional coverage is capped at 1 (with a WARNING) when a
    later dose is reported higher than the one before it.
    """
    plan = []
    achieved = None

    for n, col in dose_cols:
        cum = row[col]

        if pd.isna(cum):
            print(
                f"  WARNING {IPTP_FILE}: {year} {col} is blank; "
                f"dose {n} onward not delivered that year."
            )
            break

        cum = float(cum)

        if achieved is None:
            cond = cum

        elif achieved <= 0:
            cond = 0.0

        else:
            cond = cum / achieved

            if cond > 1:
                print(
                    f"  WARNING {IPTP_FILE}: {year} {col} ({cum:.4f}) "
                    f"exceeds the previous dose ({achieved:.4f}); "
                    f"conditional coverage capped at 1."
                )
                cond = 1.0

        achieved = cond if achieved is None else achieved * cond

        plan.append((n, cum, cond))

    return plan


def iptp_trigger_window(year_start_day):
    """
    (start_day, listening_duration) of the Pregnant-trigger window whose
    dose 1 falls in the CSV year, or None. Same shift as routine ANC.
    """
    window_start = int(year_start_day) - IPTP_DEFAULTS["delay_days"]
    window_end = window_start + DAYS_PER_YEAR

    if window_end <= 1:
        return None

    start = max(1, window_start)

    return start, int(window_end - start)


def load_iptp():
    """
    Load per-dose IPTp coverage. The `year` column is authoritative
    (year of dose 1 delivery); any day columns are ignored.
    """
    path = os.path.join(INPUT_DIR, IPTP_FILE)

    if not os.path.isfile(path):
        return None, 0

    df = keep_pickup_phase(keep_node(read_clean_csv(path)))

    if "year" not in df.columns:
        raise ValueError(
            f"{IPTP_FILE}: needs a `year` column; found {list(df.columns)}"
        )

    dose_cols = iptp_dose_columns(df)

    if not dose_cols:
        raise ValueError(
            f"{IPTP_FILE}: needs iptp1_coverage, iptp2_coverage, ... "
            f"columns; found {list(df.columns)}"
        )

    if [n for n, _ in dose_cols] != list(range(1, len(dose_cols) + 1)):
        raise ValueError(
            f"{IPTP_FILE}: dose columns must be numbered 1..N without "
            f"gaps; found {[c for _, c in dose_cols]}"
        )

    df = df.copy()
    df["year"] = df["year"].astype(int)

    in_window = df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
    dropped = int((~in_window).sum())
    df = df[in_window].reset_index(drop=True)

    if df["year"].duplicated().any():
        raise ValueError(f"{IPTP_FILE}: more than one row per year.")

    for _, col in dose_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        check_fraction(df[col].dropna(), f"{IPTP_FILE} {col}")

    df["year_start_day"] = (df["year"] - PICKUP_START_YEAR) * DAYS_PER_YEAR + 1

    df["doses"] = [
        iptp_dose_plan(row, dose_cols, int(row["year"]))
        for _, row in df.iterrows()
    ]

    return df, dropped


# VACCINE

def vaccine_expected_age_months(dose, year):
    """GHS schedule: D4 at 24 months before 2023, 18 months from 2023."""
    if dose == "D4":
        return 24 if int(year) < 2023 else 18

    return {"D1": 6, "D2": 7, "D3": 9}[dose]


def validate_vaccine_rows(df):
    """
    Validate vaccine coverage BEFORE deriving conditional uptake.

    Returns (df, issues). `df` gets a `cond_cov` column: the conditional
    uptake (D1 = its own coverage, Dk = coverage(Dk) / coverage(Dk-1) of the
    same year), or NaN for a cohort that failed validation. Supplied values
    are never clipped or altered. issues = {year: [messages]}.
    """
    for col in ("dose", "coverage", "target_age_months"):
        if col not in df.columns:
            raise ValueError(
                f"{VACCINE_FILE}: needs a `{col}` column; "
                f"found {list(df.columns)}"
            )

    df = df.copy()
    df["dose"] = df["dose"].astype(str).str.strip().str.upper()
    df["cov_num"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["age_num"] = pd.to_numeric(df["target_age_months"], errors="coerce")
    df["cond_cov"] = np.nan

    issues = {}

    def flag(year, message):
        issues.setdefault(int(year), []).append(message)

    for year, g in df.groupby(df["year"].astype(int)):
        counts = g["dose"].value_counts()

        for dose, n in counts.items():
            if dose not in VACCINE_EXPECTED_DOSES:
                flag(
                    year,
                    f"unexpected dose '{dose}' "
                    f"(expected {list(VACCINE_EXPECTED_DOSES)})",
                )
            elif n > 1:
                flag(
                    year,
                    f"{dose} appears {n} times for node {NODE_ID}, pickup; "
                    f"each year/dose must occur once",
                )

        for dose in VACCINE_EXPECTED_DOSES:
            if dose not in counts.index:
                flag(year, f"{dose} is missing")

        usable = {}

        for r in g.itertuples():
            if r.dose not in VACCINE_EXPECTED_DOSES:
                continue

            if pd.isna(r.cov_num):
                flag(year, f"{r.dose}: coverage '{r.coverage}' is not numeric")
                continue

            if not 0 <= r.cov_num <= 1:
                flag(year, f"{r.dose}: coverage {r.cov_num:g} is outside 0-1")

            want = vaccine_expected_age_months(r.dose, year)

            if pd.isna(r.age_num) or not np.isclose(r.age_num, want):
                flag(
                    year,
                    f"{r.dose}: target_age_months is {r.age_num:g} "
                    f"but the schedule needs {want}",
                )

            usable[r.dose] = float(r.cov_num)

        # Cumulative coverage must be non-increasing from D1 to D4.
        for a, b in zip(VACCINE_EXPECTED_DOSES, VACCINE_EXPECTED_DOSES[1:]):
            if a in usable and b in usable and usable[b] > usable[a] + 1e-12:
                implied = usable[b] / usable[a] if usable[a] > 0 else float("inf")

                flag(
                    year,
                    f"{b} coverage ({usable[b]:g}) exceeds {a} coverage "
                    f"({usable[a]:g}): implied conditional uptake "
                    f"{implied:.3f} > 1. The dose definitions, the "
                    f"denominator, or the {a} value need confirmation. "
                    f"Values are NOT clipped or changed.",
                )

        # Conditional uptake, only for cohorts that passed every check.
        if year not in issues:
            prev = None

            for dose in VACCINE_EXPECTED_DOSES:
                c = usable[dose]

                if prev is None:
                    cond = c
                else:
                    cond = c / prev if prev > 0 else 0.0

                mask = (df["year"].astype(int) == year) & (df["dose"] == dose)
                df.loc[mask, "cond_cov"] = min(cond, 1.0)   # float noise only
                prev = c

    # Raw and derived values for every record.
    order = {d: i for i, d in enumerate(VACCINE_EXPECTED_DOSES)}
    shown = df.assign(dose_order=df["dose"].map(order).fillna(99)).sort_values(
        ["year", "dose_order"]
    )

    print(f"\nVaccine coverage check ({VACCINE_FILE}):")
    print("  year dose  target_age  cumulative_cov  conditional_uptake")

    for r in shown.itertuples():
        cond = (
            f"{r.cond_cov:.6f}"
            if pd.notna(r.cond_cov)
            else "NOT DERIVED (cohort failed validation)"
        )

        print(
            f"  {int(r.year)} {r.dose:<4} {r.age_num:>6g} mo  "
            f"{r.cov_num:>14.6f}  {cond}"
        )

    if issues:
        print(
            f"\n  ERROR {VACCINE_FILE}: cohort(s) {sorted(issues)} FAILED "
            f"validation and get NO vaccine campaign:"
        )

        for y in sorted(issues):
            for msg in issues[y]:
                print(f"    {y}: {msg}")

    return df.drop(columns=["cov_num", "age_num"]), issues


def load_vaccine():
    df, dropped = load_generic(
        VACCINE_FILE,
        default_day_note=(
            "using 1 Jan of each year as the start of that year's rolling, "
            "birth-triggered D1 eligibility window."
        ),
    )

    VACCINE_ISSUES.clear()

    if not has_rows(df):
        return df, dropped

    # Validate first; cohorts that fail are excluded (never repaired).
    df, issues = validate_vaccine_rows(df)
    VACCINE_ISSUES.update(issues)

    df = df[~df["year"].astype(int).isin(list(issues))].reset_index(drop=True)

    if df.empty:
        return df, dropped

    df["cov"] = need(df, ["coverage"], VACCINE_FILE)

    df["target_age_months"] = df["target_age_months"].astype(float)
    df["amin"] = (
        df["target_age_months"] / 12.0 if VACCINE_USE_AGE_FLOOR else 0.0
    )
    df["amax"] = VACCINE_CATCHUP_MAX_AGE_MONTHS / 12.0

    # The current vaccine CSV defines dose, year, target age and coverage.
    # Effect/type parameters remain constants unless explicitly supplied in CSV.
    df["initial_effect"] = parameter_or_default(
        df, ["initial_effect", "efficacy"],
        VACCINE_NONCSV_DEFAULTS["initial_effect"], VACCINE_FILE
    )
    df["box_duration"] = parameter_or_default(
        df, ["box_duration"], VACCINE_NONCSV_DEFAULTS["box_duration"], VACCINE_FILE
    )
    df["decay"] = parameter_or_default(
        df, ["decay", "decay_time"],
        VACCINE_NONCSV_DEFAULTS["decay"], VACCINE_FILE
    )
    df["vaccine_type"] = parameter_or_default(
        df, ["vaccine_type"], VACCINE_NONCSV_DEFAULTS["vaccine_type"],
        VACCINE_FILE, str
    )

    df = df.sort_values(["year", "target_age_months"]).reset_index(drop=True)

    df["prev_dose"] = None
    df["delay_days"] = 0.0

    for _, group in df.groupby("year"):
        idx = list(group.index)

        for pos in range(1, len(idx)):
            i = idx[pos]
            prev_i = idx[pos - 1]

            df.at[i, "prev_dose"] = df.at[prev_i, "dose"]

            gap_months = (
                df.at[i, "target_age_months"] - df.at[prev_i, "target_age_months"]
            )

            df.at[i, "delay_days"] = max(gap_months * DAYS_PER_MONTH, 1.0)

    return df, dropped


# LOAD ALL SCHEDULES

def load_all_schedules():
    sched = {}
    dropped = {}

    sched["cm"], dropped["cm"] = load_case_management()
    sched["itn"], dropped["itn"] = load_itn()
    sched["itn_anc"], dropped["itn_anc"] = load_routine_itn(ITN_ANC, "ANC")
    sched["itn_cwc"], dropped["itn_cwc"] = load_routine_itn(ITN_CWC, "CWC")
    sched["irs"], dropped["irs"] = load_irs()
    sched["smc"], dropped["smc"] = load_smc()
    sched["iptp"], dropped["iptp"] = load_iptp()
    sched["vaccine"], dropped["vaccine"] = load_vaccine()

    sched["dropped"] = dropped

    return sched


# SCHEDULE DESCRIPTION

def describe_schedules(sched):
    print(
        f"\nPickup interventions, {PICKUP_START_YEAR}-{PICKUP_END_YEAR} "
        f"(day 1 = 1 Jan {PICKUP_START_YEAR}):"
    )

    # ---- Case management ----------------------------------------------------
    print("  Case management:")

    if not has_rows(sched["cm"]):
        print("    none")
    else:
        for r in sched["cm"].itertuples():
            clip = (
                ""
                if r.duration == r.requested_duration
                else f" [clipped from {r.requested_duration}d]"
            )

            print(
                f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4}) "
                f"for {r.duration}d  {r.trigger:<16} "
                f"ages {r.age_min:g}-{r.age_max:g}  "
                f"cov {r.effective_coverage:.3f}  "
                f"rate {r.rate:g}  {r.drug}{clip}"
            )

    if sched["dropped"]["cm"]:
        print(
            f"    ({sched['dropped']['cm']} row(s) outside the "
            f"pickup window ignored)"
        )

    # ---- Other interventions ------------------------------------------------
    for key, label in [
        ("itn", "ITN"),
        ("itn_anc", "ITN-ANC"),
        ("itn_cwc", "ITN-CWC"),
        ("irs", "IRS"),
        ("smc", "SMC"),
        ("iptp", "IPTp"),
        ("vaccine", "Vaccine"),
    ]:
        df = sched[key]

        if df is None:
            if key == "itn" and ITN_FILE is None:
                print(f"  {label}: mass campaigns not used (ITN_FILE disabled)")
            else:
                print(f"  {label}: file not found, skipped")
            continue

        if df.empty:
            if key == "vaccine" and VACCINE_ISSUES:
                print(
                    f"  {label}: no valid cohorts (excluded: "
                    f"{sorted(VACCINE_ISSUES)}; see ERROR above)"
                )
            else:
                print(
                    f"  {label}: no rows in {PICKUP_START_YEAR}-{PICKUP_END_YEAR}"
                )
            continue

        print(f"  {label}:")

        for r in df.itertuples():
            # IPTp (per-dose coverage, no single `cov`)
            if key == "iptp":
                delivery_end = min(
                    r.year_start_day + DAYS_PER_YEAR - 1, PICKUP_DAYS
                )

                doses = "  ".join(
                    f"D{n} cum {cum:.3f} (cond {cond:.3f})"
                    for n, cum, cond in r.doses
                ) or "no doses"

                print(
                    f"    year {r.year}  dose 1 delivered "
                    f"{fmt_day(r.year_start_day)} .. {fmt_day(delivery_end)}  "
                    f"{doses}"
                )
                continue

            cov_label = f"cov {r.cov:.3f}"

            # Mass ITN
            if key == "itn":
                print(
                    f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4})  "
                    f"{cov_label}  "
                    f"block {r.blocking_initial:.3f}/{r.blocking_decay:g}d  "
                    f"kill {r.killing_initial:.3f}/{r.killing_decay:g}d  "
                    f"usage: {itn_usage_label()}"
                )
                continue

            # Routine ANC / CWC
            if key in ("itn_anc", "itn_cwc"):
                delay = ANC_DELAY_DAYS if key == "itn_anc" else CWC_DELAY_DAYS

                delivery_end = min(
                    r.year_start_day + DAYS_PER_YEAR - 1,
                    PICKUP_DAYS,
                )

                print(
                    f"    year {r.year}  nets delivered "
                    f"{fmt_day(r.year_start_day)} .. {fmt_day(delivery_end)}  "
                    f"yearly {cov_label} of eligible  (trigger {delay}d earlier)  "
                    f"block {r.blocking_initial:.2f}/{r.blocking_decay:g}d  "
                    f"kill {r.killing_initial:.2f}/{r.killing_decay:g}d  "
                    f"usage: {itn_usage_label()}"
                )

                channel = "ANC" if key == "itn_anc" else "CWC"
                n_batches = len(
                    routine_delivery_batches(
                        r.year_start_day, delay,
                        size=routine_batch_size(channel),
                    )
                )
                print(
                    f"      delivery windows: {n_batches} "
                    f"({routine_batch_size(channel)}-day batch)"
                )

                if key == "itn_cwc":
                    span = cwc_catchup_span(r.year_start_day)

                    if span:
                        n = len(cwc_catchup_batches(r.year_start_day))
                        print(
                            f"      catch-up for children already born: "
                            f"{n} scheduled batch(es), "
                            f"{fmt_day(span[0])} .. {fmt_day(span[1])}"
                        )

                continue

            # IRS
            if key == "irs":
                extra = (
                    f"{r.insecticide or '(none)'}: "
                    f"kill {r.killing_initial:.2f}, "
                    f"decay {r.killing_decay:g}d"
                )

            # SMC
            elif key == "smc":
                extra = (
                    f"round {r.round}  "
                    f"ages {r.amin:g}-{r.amax:g}  {r.drug_code}"
                )

            # Vaccine: D1
            elif pd.isna(r.prev_dose) or r.prev_dose is None:
                extra = (
                    f"dose {r.dose} ({int(r.year)})  "
                    f"exactly {r.target_age_months:.0f}mo from birth  "
                    f"[age band {r.amin:.3f}-{r.amax:.3f}y, "
                    f"safety bound only]"
                )

            # Vaccine: D2+
            else:
                extra = (
                    f"dose {r.dose} ({int(r.year)})  "
                    f"exactly {r.delay_days:.0f}d after "
                    f"{r.prev_dose} ({int(r.year)})  "
                    f"[age band {r.amin:.3f}-{r.amax:.3f}y, "
                    f"safety bound only]  "
                    f"(conditional on receiving {r.prev_dose})"
                )

                cov_label = (
                    f"cond cov {r.cond_cov:.6f} (cum {r.cov:.6f})"
                )

            print(
                f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4})  "
                f"{cov_label}  {extra}"
            )

        if sched["dropped"][key]:
            print(
                f"    ({sched['dropped'][key]} row(s) outside the "
                f"pickup window ignored)"
            )


def describe_vaccine_timing(df):
    """
    Print, per birth cohort, the births window, each dose's trigger, delay,
    conditional coverage, delivery window, and the share of the cohort that
    can reach the dose before the pickup ends.
    """
    print("\nVaccine cohort timing (flat 365-day calendar):")

    if not has_rows(df):
        print("  no valid vaccine cohorts")
        return

    for year, g in df.groupby("year"):
        g = g.sort_values("target_age_months")

        first = int(g["pickup_day"].iloc[0])
        last = first + DAYS_PER_YEAR - 1

        print(
            f"  {int(year)} cohort: births day {first}-{last} "
            f"({fmt_day(first)} .. {fmt_day(last)})"
        )

        cum = 0.0

        for r in g.itertuples():
            if pd.isna(r.prev_dose) or r.prev_dose is None:
                step = r.target_age_months * DAYS_PER_MONTH
                cum = step
                trigger = "Births"
            else:
                step = r.delay_days
                cum += step
                trigger = f"Received_Vaccine_{r.prev_dose}_{int(year)}"

            reachable = min(
                max((PICKUP_DAYS - cum) - first + 1, 0), DAYS_PER_YEAR
            )

            print(
                f"    {r.dose}: trigger {trigger}, delay {step:.1f}d, "
                f"cond cov {r.cond_cov:.6f}, delivered on days "
                f"{first + cum:.0f}-{last + cum:.0f}, cohort share that "
                f"can reach it before day {PICKUP_DAYS}: "
                f"{reachable / DAYS_PER_YEAR:.0%}"
            )


# Filled in main block.
SCHEDULES = None


# HABITAT GRID (inherited from the burn-in)

def describe_habitat_grid():
    """Print the expected burn-in grid (works in --dry-run, no platform needed)."""
    print("\nHabitat grid inherited from the burn-in (must equal run_burnin_UWR.py):")
    print(f"  x_Temporary_Larval_Habitat ({len(XTLH_RANGE)}): "
          + ", ".join(f"{v:.2f}" for v in XTLH_RANGE))
    print(f"  scale_constant             ({len(SCALE_CONSTANT)}): {SCALE_CONSTANT}")
    print(f"  scale_temp_rain            ({len(SCALE_TEMP_RAIN)}): {SCALE_TEMP_RAIN}")
    print(f"  scale_water_veg            ({len(SCALE_WATER_VEG)}): {SCALE_WATER_VEG}")
    print(f"  habitat combinations: {N_HABS}   seeds: {NUM_SEEDS}   "
          f"expected burn-in simulations: {EXPECTED_SIMULATIONS}")


def check_burnin_grid(burnin_df):
    """
    Compare the burn-in tags actually read from the experiment with the grid
    above. Prints what was found; WARNING-only so it never blocks a submission.
    """
    print("\nBurn-in habitat tags found in the experiment:")

    def unique_vals(col):
        return sorted({float(v) for v in burnin_df[col]})

    checks = [
        ("x_Temporary_Larval_Habitat", XTLH_RANGE),
        ("scale_constant", SCALE_CONSTANT),
        ("scale_temp_rain", SCALE_TEMP_RAIN),
        ("scale_water_veg", SCALE_WATER_VEG),
    ]

    problems = 0

    for col, expected in checks:
        found = unique_vals(col)
        print(f"  {col:<28}: " + ", ".join(f"{v:g}" for v in found))

        unexpected = [v for v in found
                      if not any(np.isclose(v, e, rtol=1e-3) for e in expected)]
        if unexpected:
            problems += 1
            print(f"    WARNING: {col} value(s) {[f'{v:g}' for v in unexpected]} "
                  f"are not in this script's grid.")

        if not BURNIN_FILTER:
            absent = [e for e in expected
                      if not any(np.isclose(e, v, rtol=1e-3) for v in found)]
            if absent:
                problems += 1
                print(f"    WARNING: grid value(s) {[f'{v:g}' for v in absent]} "
                      f"were not found in the burn-in experiment.")

    n = len(burnin_df)

    if BURNIN_FILTER:
        print(f"  simulations selected: {n} (BURNIN_FILTER active)")
    elif n != EXPECTED_SIMULATIONS:
        problems += 1
        print(f"  WARNING: {n} burn-in simulations found, grid expects "
              f"{EXPECTED_SIMULATIONS}. Some may have failed or not finished.")
    else:
        print(f"  simulations: {n} (matches expected {EXPECTED_SIMULATIONS})")

    if problems == 0:
        print("  OK: burn-in tags match this script's grid.")
    else:
        print("  The burn-in experiment does not match this script's grid. "
              "Check BURNIN_EXPERIMENT_ID and the values above.")


# CLIMATE

CLIMATE_VARS = [
    "air_temperature",
    "land_temperature",
    "rainfall",
    "relative_humidity",
]

CLIMATE_FILES = [f"{v}.bin{ext}" for v in CLIMATE_VARS for ext in ("", ".json")]


def resolve_climate_dir():
    if CLIMATE_SUBDIR:
        return os.path.join(CLIMATE_ROOT, CLIMATE_SUBDIR)

    if os.path.isfile(os.path.join(CLIMATE_ROOT, "air_temperature.bin")):
        return CLIMATE_ROOT

    if not os.path.isdir(CLIMATE_ROOT):
        raise FileNotFoundError(
            f"Pickup climate folder does not exist:\n{CLIMATE_ROOT}"
        )

    found = [
        d
        for d in sorted(os.listdir(CLIMATE_ROOT))
        if os.path.isfile(os.path.join(CLIMATE_ROOT, d, "air_temperature.bin"))
    ]

    if len(found) == 1:
        return os.path.join(CLIMATE_ROOT, found[0])

    if not found:
        raise FileNotFoundError(
            f"No air_temperature.bin in {CLIMATE_ROOT} or its subfolders."
        )

    raise ValueError(
        f"Several weather folders in {CLIMATE_ROOT}: {found}. "
        f"Set CLIMATE_SUBDIR."
    )


def validate_climate(climate_dir):
    for name in CLIMATE_FILES:
        path = os.path.join(climate_dir, name)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing climate file:\n{path}")

    for var in CLIMATE_VARS:
        metadata_path = os.path.join(climate_dir, f"{var}.bin.json")

        with open(metadata_path) as f:
            meta = json.load(f).get("Metadata", {})

        count = meta.get("DatavalueCount")
        years = str(meta.get("OriginalDataYears", ""))

        if count is not None and int(count) < PICKUP_DAYS:
            raise ValueError(
                f"{var}: weather has {count} days but pickup needs "
                f"{PICKUP_DAYS}. EMOD would wrap back to the start "
                f"of the file."
            )

        if not years.startswith(str(PICKUP_START_YEAR)):
            print(
                f"  WARNING {var}: OriginalDataYears is '{years}', "
                f"expected it to start in {PICKUP_START_YEAR}. "
                f"Day 1 of the weather must be 1 Jan {PICKUP_START_YEAR}."
            )

    print(f"Climate: {climate_dir}")


# SITE / VECTOR INPUT

DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)
VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)

for _p in [
    INPUT_DIR,
    DEMOGRAPHICS_PATH,
    VECTOR_PATH,
    os.path.join(INPUT_DIR, CM_FILE),
    manifest.eradication_path,
    manifest.schema_file,
]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"Required path does not exist:\n{_p}")


vector_df = pd.read_csv(VECTOR_PATH)
vector_df.columns = [c.strip() for c in vector_df.columns]

if APPLY_VECTOR_BEHAVIOUR:
    _missing = [c for c in ("anthropophily", "indoor_feeding")
                if c not in vector_df.columns]
    if _missing:
        raise ValueError(
            f"APPLY_VECTOR_BEHAVIOUR is True but {VECTOR_FILE} lacks "
            f"column(s) {_missing}."
        )


def _read_site_definition():
    site_df = pd.read_csv(DEMOGRAPHICS_PATH)
    site_df.columns = [c.strip() for c in site_df.columns]

    if len(site_df) != 1 or int(site_df["node_id"].iloc[0]) != NODE_ID:
        raise ValueError(
            f"{DEMOGRAPHICS_FILE} must hold exactly one row "
            f"with node_id={NODE_ID}."
        )

    row = site_df.iloc[0]

    return {
        "name": SITE_NAME,
        "pop": int(row["pop"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
    }


# CONFIGURATION

def remove_stale_params(config):
    """
    Remove config parameters that the installed schema does not define.

    emod_api's AddIndividualPropertyAndHINT adds an implicit config that
    sets Disable_IP_Whitelist = 1 for any individual property not on its
    whitelist (such as "Pregnancy"). Because that implicit config runs after
    set_param_fn, this must be called AFTER the implicits have run: from the
    last entry in demog.implicits and from the per-simulation sweep callback.
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


def set_param_fn(config):
    config = conf.set_team_defaults(config, manifest)

    config.parameters.Start_Time = 0
    config.parameters.Simulation_Duration = PICKUP_DAYS

    # ---- Climate ------------------------------------------------------------
    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", "air_temperature.bin")
    config.parameters.Land_Temperature_Filename = os.path.join("climate", "land_temperature.bin")
    config.parameters.Rainfall_Filename = os.path.join("climate", "rainfall.bin")
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", "relative_humidity.bin")

    # InsetChart ON in the pickup (the burn-in leaves it off).
    config.parameters.Enable_Default_Reporting = 1

    # ---- Logging ------------------------------------------------------------
    for logger in [
        "JsonConfigurable",
        "VectorHabitat",
        "StandardEventCoordinator",
        "SusceptibilityMalaria",
        "default",
        "Eradication",
    ]:
        config.parameters[f"logLevel_{logger}"] = "ERROR"

    # ---- Memory -------------------------------------------------------------
    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = int(MEMORY_LIMIT)
    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = int(MEMORY_LIMIT - 500)

    # ---- Migration ----------------------------------------------------------
    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    # ---- Species and habitats -----------------------------------------------
    # Must match the burn-in: the serialized state carries the habitat types.
    conf.add_species(config, manifest, vector_df["species"].tolist())

    for i in range(len(vector_df)):
        species = vector_df["species"].iloc[i]

        # Only when the burn-in did the same (see APPLY_VECTOR_BEHAVIOUR).
        if APPLY_VECTOR_BEHAVIOUR:
            conf.set_species_param(
                config,
                species=species,
                parameter="Anthropophily",
                value=float(vector_df["anthropophily"].iloc[i]),
                overwrite=True,
            )

            conf.set_species_param(
                config,
                species=species,
                parameter="Indoor_Feeding_Fraction",
                value=float(vector_df["indoor_feeding"].iloc[i]),
                overwrite=True,
            )

        conf.set_species_param(
            config,
            species=species,
            parameter="Habitats",
            value=[
                {
                    "Habitat_Type": "CONSTANT",
                    "Max_Larval_Capacity": 1e7 * vector_df["constant"].iloc[i],
                },
                {
                    "Habitat_Type": "TEMPORARY_RAINFALL",
                    "Max_Larval_Capacity": 1e7 * vector_df["temp_rain"].iloc[i],
                },
                {
                    "Habitat_Type": "WATER_VEGETATION",
                    "Max_Larval_Capacity": 1e7 * vector_df["water_veg"].iloc[i],
                },
            ],
            overwrite=True,
        )

    # ---- Placeholders -------------------------------------------------------
    # Overwritten for each simulation using the burn-in row.
    config.parameters.Run_Number = 0
    config.parameters.x_Temporary_Larval_Habitat = 1

    # ---- Serialization (read) -----------------------------------------------
    config.parameters.Serialized_Population_Reading_Type = "READ"

    # Read the complete node state.
    config.parameters.Serialization_Mask_Node_Read = 0

    # ---- Serialization (write, end of pickup) -------------------------------
    if WRITE_PICKUP_STATE:
        config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
        config.parameters.Serialization_Time_Steps = [PICKUP_DAYS]
        config.parameters.Serialization_Mask_Node_Write = 0
        config.parameters.Serialization_Precision = "REDUCED"

    # ---- Birth model and scalar ---------------------------------------------
    # Config settings are NOT stored in the serialized state, so these must
    # equal the burn-in's. NOTE: demographics implicits run after this
    # function and can change the birth model. build_demog() appends a final
    # implicit function that forces it back, and update_serialize_parameters()
    # sets it again per simulation (see the module docstring).
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    config.parameters.x_Birth = X_BIRTH

    return config


# DEMOGRAPHICS

def validate_demographic_settings():
    """
    Sanity-check the birth / death settings before anything is submitted.

    BirthRate under INDIVIDUAL_PREGNANCIES is a per-possible-mother DAILY
    rate. A woman carries at most one pregnancy per 40-week gestation, so
    rates above 1 / 280 per day cannot be realised, and a crude per-1,000
    value (0.031) pasted here would be far too large, not too small.
    """
    if "PREGNANC" not in BIRTH_RATE_DEPENDENCE:
        # Reported separately in the main block.
        return

    effective = PREG_RATE_PER_MOTHER_DAY * X_BIRTH

    if PREG_RATE_PER_MOTHER_DAY <= 0 or X_BIRTH <= 0:
        raise ValueError(
            "PREG_RATE_PER_MOTHER_DAY and X_BIRTH must both be > 0 "
            f"(got {PREG_RATE_PER_MOTHER_DAY:g} and {X_BIRTH:g})."
        )

    if effective > MAX_PREG_RATE_PER_MOTHER_DAY:
        raise ValueError(
            f"PREG_RATE_PER_MOTHER_DAY x X_BIRTH = {effective:g} per day exceeds "
            f"the maximum of one pregnancy per 40 weeks "
            f"({MAX_PREG_RATE_PER_MOTHER_DAY:.2e}). BirthRate here is a "
            f"per-mother daily rate, not a crude rate; do not use CrudeRate()."
        )

    if CRUDE_DEATH_RATE <= 0:
        raise ValueError("CRUDE_DEATH_RATE must be > 0 (per 1,000 per year).")


def describe_demographics():
    site = _read_site_definition()

    print("\nDemographic settings (must match the burn-in):")
    print(f"  node pop (demographics.csv)  : {site['pop']}  "
          f"(initial value only; the population comes from the serialized state)")
    print(f"  Birth_Rate_Dependence        : {BIRTH_RATE_DEPENDENCE}")
    print(f"  BirthRate per mother per day : {PREG_RATE_PER_MOTHER_DAY:g} "
          f"(~{PREG_RATE_PER_MOTHER_DAY * 365:.3f} pregnancies / woman / year)")
    print(f"  x_Birth                      : {X_BIRTH:g}")
    print(f"  crude death rate             : {CRUDE_DEATH_RATE:g} per 1,000 / year (flat)")
    print(f"  vector behaviour from CSV    : "
          f"{'ON' if APPLY_VECTOR_BEHAVIOUR else 'OFF (matches the burn-in)'}")


def add_pregnancy_property(demog):
    """Add the same Pregnancy individual property used in the burn-in."""
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

    emodpy runs demographics.implicits in list order after set_param_fn
    (task.config = fn(task.config)). Appended last, this runs after:
      - the birth-rate implicit added by SetBirthRate, and
      - the Disable_IP_Whitelist implicit added by AddIndividualPropertyAndHINT.
    """
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    config.parameters.x_Birth = X_BIRTH
    remove_stale_params(config)
    return config


def build_demog():
    """Same demographics as the burn-in: single node, per-mother BirthRate."""
    site = _read_site_definition()

    demog = Demographics.from_template_node(
        lat=site["lat"],
        lon=site["lon"],
        pop=site["pop"],
        name=site["name"],
    )

    # BirthRate under INDIVIDUAL_PREGNANCIES is a per-possible-mother DAILY
    # rate. Do NOT pass CrudeRate() here.
    demog.SetBirthRate(YearlyRate(PREG_RATE_PER_MOTHER_DAY * DAYS_PER_YEAR))
    demog.SetMortalityRate(CrudeRate(CRUDE_DEATH_RATE))

    # SetEquilibriumAgeDistFromBirthAndMortRates is deliberately NOT called
    # (the burn-in does not call it; SetAgeDistribution overrides it anyway,
    # and the age structure comes from the serialized state at pickup).
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)

    add_pregnancy_property(demog)

    # demog.implicits is the documented list of config functions that emodpy
    # applies, in order, after set_param_fn. Append last so ours runs last.
    demog.implicits.append(_final_demographics_config)

    return demog


# CAMPAIGN HELPERS

def fix_property_restrictions(obj):
    """Replace null property restriction fields with []."""
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


_WARNED = set()


def call_supported(fn, campaign, label, **kwargs):
    """Call an emodpy function with only supported keyword arguments."""
    params = inspect.signature(fn).parameters

    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        dropped = [k for k in kwargs if k not in params]

        if dropped and label not in _WARNED:
            print(
                f"  WARNING {label}: {fn.__name__} in this emodpy_malaria "
                f"does not accept {dropped}; those settings are NOT applied."
            )
            _WARNED.add(label)

        kwargs = {k: v for k, v in kwargs.items() if k in params}

    return fn(campaign, **kwargs)


def require_supported(fn, label, names):
    """
    Raise if fn lacks arguments that must NOT be silently dropped
    (dropping them would change who is treated, or when).
    """
    params = inspect.signature(fn).parameters

    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return

    missing = [n for n in names if n not in params]

    if missing:
        raise ValueError(
            f"{label}: {fn.__name__} in this emodpy_malaria does not "
            f"accept {missing}, which this intervention requires."
        )


def weibull_discard_config(r):
    """
    Return a Weibull discard config if both discard parameters are set
    and valid.

    EMOD requires Kappa and Lambda > 0, so 0 (or negative) values are
    treated as "no discard distribution given" and skipped with a warning.
    """
    if np.isnan(r.discard_k) or np.isnan(r.discard_l):
        return None

    if r.discard_k <= 0 or r.discard_l <= 0:
        key = ("discard", float(r.discard_k), float(r.discard_l))

        if key not in _WARNED:
            print(
                f"  WARNING ITN: discard_k={r.discard_k:g}, "
                f"discard_l={r.discard_l:g} must both be > 0; no discard "
                f"distribution applied for those rows."
            )
            _WARNED.add(key)

        return None

    return {
        "Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",
        "Expiration_Period_Kappa": float(r.discard_k),
        "Expiration_Period_Lambda": float(r.discard_l),
    }


# PREGNANCY TRACKING

def add_pregnancy_tracking(campaign):
    """
    Keep Pregnancy property current.

        Pregnant  -> IsPregnant
        GaveBirth -> NotPregnant
    """
    common.change_individual_property_triggered(
        campaign,
        new_ip_key="Pregnancy",
        new_ip_value="IsPregnant",
        target_sex="Female",
        triggers=["Pregnant"],
        start_day=1,
        daily_prob=1,
        revert_in_days=-1,
        blackout=False,
    )

    common.change_individual_property_triggered(
        campaign,
        new_ip_key="Pregnancy",
        new_ip_value="NotPregnant",
        target_sex="Female",
        triggers=["GaveBirth"],
        start_day=1,
        daily_prob=1,
        revert_in_days=-1,
        blackout=False,
    )

    return campaign


# CASE MANAGEMENT

def add_case_management(campaign, df):
    if not has_rows(df):
        return

    for (day, duration, drug, severe), group in df.groupby(
        ["pickup_day", "duration", "drug", "is_severe"]
    ):
        if int(duration) <= 0:
            continue

        targets = [
            {
                "trigger": r.trigger,
                "coverage": float(r.effective_coverage),
                "agemin": float(r.age_min),
                "agemax": float(r.age_max),
                "rate": float(r.rate),
            }
            for r in group.itertuples()
        ]

        add_treatment_seeking(
            campaign,
            start_day=int(day),
            drug=CM_DRUGS[drug],
            targets=targets,
            duration=int(duration),
            broadcast_event_name=(
                "Received_Severe_Treatment" if severe else "Received_Treatment"
            ),
        )


# MASS ITN

def add_itns(campaign, df):
    """
    Add scheduled mass LLIN campaigns.

    Net use is age- and season-dependent (see "ITN USAGE CURVES"). The
    seasonal curve is rotated to this campaign's start day, because EMOD
    counts it from the day the net is distributed.
    """
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_scheduled_usage_dependent_bednet,
    )

    if not has_rows(df):
        return

    for r in df.itertuples():
        kwargs = dict(
            start_day=int(r.pickup_day),
            demographic_coverage=float(r.cov),
            blocking_initial_effect=float(r.blocking_initial),
            blocking_decay_time_constant=float(r.blocking_decay),
            killing_initial_effect=float(r.killing_initial),
            killing_decay_time_constant=float(r.killing_decay),
            **itn_usage_kwargs("MASS", int(r.year), r.pickup_day),
        )

        discard = weibull_discard_config(r)
        if discard is not None:
            kwargs["discard_config"] = discard

        call_supported(
            add_scheduled_usage_dependent_bednet,
            campaign,
            "ITN",
            **kwargs,
        )


# ROUTINE ANC / CWC ITN

def stage_restriction_kind():
    """How the trigger helper takes an individual-property restriction."""
    fn = getattr(common, "triggered_campaign_event_with_optional_delay", None)

    if fn is None:
        return None

    params = inspect.signature(fn).parameters

    if "property_restrictions" in params:
        return "new"

    if "ind_property_restrictions" in params:
        return "old"

    return None


def add_stage_event(campaign, start, listen, delay, event_name, trigger,
                    coverage, target_sex, restriction=None):
    """
    One routing stage: on `trigger`, after `delay` days (0 = same day), with
    probability `coverage`, broadcast `event_name`.

    Uses emod_api.interventions.common only, so it does not depend on the
    emodpy_malaria version (no bednet broadcast_event needed).
    `restriction` is "Property:Value" (for example "Pregnancy:IsPregnant").
    """
    fn = getattr(common, "triggered_campaign_event_with_optional_delay", None)

    if fn is None or not hasattr(common, "BroadcastEvent"):
        raise ValueError(
            "Routine ITN delivery needs emod_api.interventions.common."
            "triggered_campaign_event_with_optional_delay and BroadcastEvent."
        )

    params = inspect.signature(fn).parameters

    kwargs = dict(
        start_day=int(start),
        triggers=[trigger],
        intervention=common.BroadcastEvent(campaign, event_name),
        duration=int(listen),
        coverage=float(coverage),
        target_sex=target_sex,
    )

    if delay:
        kwargs["delay"] = {"Delay_Period_Constant": float(delay)}

    if "blackout" in params:
        kwargs["blackout"] = False

    if restriction:
        kind = stage_restriction_kind()
        key, value = restriction.split(":")

        if kind == "new":
            kwargs["property_restrictions"] = [restriction]
        elif kind == "old":
            kwargs["ind_property_restrictions"] = [{key: value}]
        else:
            raise ValueError(
                f"{fn.__name__} cannot take a property restriction."
            )

    campaign.add(fn(campaign, **kwargs))


def add_routine_itns(campaign, df, channel):
    """
    Routine LLIN delivery. CSV coverage is the YEARLY operational coverage
    among the ELIGIBLE group, so it is applied once, at the visit / target
    age, after the delay.

        ANC: Pregnant -> +16 wk -> ANC_Visit (cov 1: women who reach the visit)
             ANC_Visit -> ANC_Net (cov = CSV, still pregnant) -> net
        CWC: Births   -> +18 mo -> CWC_Visit (cov 1: children at CWC age)
             CWC_Visit -> CWC_Net (cov = CSV) -> net

    ANC delivery years are ONE window (ANC_ITN_BATCH_DAYS = 365, seasonal
    curve rotated to mid-year); CWC years are ITN_SEASON_BATCH_DAYS-day
    batches so the seasonal usage curve follows the delivery date. Batches
    are slices of the year and all use the same yearly coverage.

    Each stage is relayed (cov 1, no delay) to a year-level event, and only
    those year-level events are recorded:

        ANC_Net_<yr> / ANC_Visit_<yr> = yearly ANC coverage
        CWC_Net_<yr> / CWC_Visit_<yr> = yearly CWC coverage
    """
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_triggered_usage_dependent_bednet,
    )

    if not has_rows(df):
        return

    require_supported(
        add_triggered_usage_dependent_bednet,
        f"ITN-{channel}",
        ["trigger_condition_list", "listening_duration"],
    )

    net_extra = {}
    stage_restriction = None

    if channel == "ANC":
        trigger, delay, sex = "Pregnant", ANC_DELAY_DAYS, "Female"
        visit_kind, net_kind = "ANC_Visit", "ANC_Net"

        if ANC_CHECK_PREGNANT_AT_DELIVERY:
            if stage_restriction_kind() is not None:
                stage_restriction = "Pregnancy:IsPregnant"
            else:
                # The trigger helper cannot take a restriction, so apply the
                # pregnancy check on the bednet itself. ANC_Net_<yr> is then
                # counted BEFORE the check.
                if "ANC_restriction" not in _WARNED:
                    print(
                        "  WARNING ITN-ANC: the trigger helper has no "
                        "property restriction argument; the pregnancy check "
                        "is applied on the bednet, so ANC_Net_<year> events "
                        "are counted before the check "
                        "(Bednet_Got_New_One is exact)."
                    )
                    _WARNED.add("ANC_restriction")

                params = inspect.signature(
                    add_triggered_usage_dependent_bednet
                ).parameters

                if "property_restrictions" in params:
                    net_extra = dict(
                        property_restrictions=["Pregnancy:IsPregnant"]
                    )
                else:
                    net_extra = dict(
                        ind_property_restrictions=[{"Pregnancy": "IsPregnant"}]
                    )

    elif channel == "CWC":
        trigger, delay, sex = "Births", CWC_DELAY_DAYS, "All"
        visit_kind, net_kind = "CWC_Visit", "CWC_Net"

    else:
        raise ValueError(f"Unknown routine ITN channel: {channel}")

    batch_size = routine_batch_size(channel)

    for r in df.itertuples():
        year = int(r.year)

        # CSV year = delivery year. Listen earlier by the trigger-to-delivery
        # delay, one window per delivery batch.
        for batch, start, listen, mid_day in routine_delivery_batches(
            r.year_start_day, delay, size=batch_size
        ):
            visit_event = routine_event_name(visit_kind, year, batch)
            net_event = routine_event_name(net_kind, year, batch)

            # Stage 1: reach the visit / target age (eligible group, cov 1).
            add_stage_event(campaign, start, listen, delay, visit_event,
                            trigger, 1.0, sex)

            # Relay: year-level label for the eligible group.
            add_stage_event(campaign, 1, -1, 0,
                            yearly_event_name(visit_kind, year),
                            visit_event, 1.0, sex)

            # Stage 2: yearly operational coverage, applied once to the
            # eligible group (ANC: only if still pregnant).
            add_stage_event(campaign, 1, -1, 0, net_event, visit_event,
                            r.cov, sex, restriction=stage_restriction)

            # Relay: year-level label for those who get a net.
            add_stage_event(campaign, 1, -1, 0,
                            yearly_event_name(net_kind, year),
                            net_event, 1.0, sex)

            # Stage 3: deliver the net (coverage already applied).
            kwargs = dict(
                intervention_name=f"ITN_{channel}",
                start_day=1,
                demographic_coverage=1.0,
                trigger_condition_list=[net_event],
                listening_duration=-1,
                blocking_initial_effect=float(r.blocking_initial),
                blocking_decay_time_constant=float(r.blocking_decay),
                killing_initial_effect=float(r.killing_initial),
                killing_decay_time_constant=float(r.killing_decay),
                **itn_usage_kwargs(channel, year, mid_day),
                **net_extra,
            )

            discard = weibull_discard_config(r)
            if discard is not None:
                kwargs["discard_config"] = discard

            call_supported(
                add_triggered_usage_dependent_bednet,
                campaign,
                f"ITN-{channel}",
                **kwargs,
            )


def add_cwc_catchup(campaign, df):
    """
    CWC deliveries that a Births trigger cannot serve (children already
    born at day 1).

    For each CWC year, deliveries on days 1..CWC_DELAY_DAYS are made in
    scheduled batches. A batch ending on day E covers children who turned
    CWC_DELAY_DAYS old during the batch, so at E their age lies in

        [CWC_DELAY_DAYS / 365, CWC_DELAY_DAYS / 365 + batch_days / 365).

    Same intervention name, coverage and net parameters as the triggered
    CWC nets; the CSV coverage is applied directly to these eligible
    children. Each batch is a scheduled event, so its delivery day (E) is
    known and the seasonal usage curve is rotated to it exactly.

    These nets are scheduled, not triggered, so they carry no CWC_Net event;
    they appear only in Bednet_Got_New_One.
    """
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_scheduled_usage_dependent_bednet,
    )

    if not has_rows(df):
        return

    batches_by_row = [
        (r, cwc_catchup_batches(r.year_start_day)) for r in df.itertuples()
    ]

    if not any(b for _, b in batches_by_row):
        return

    # Dropping an age argument would give nets to everyone.
    require_supported(
        add_scheduled_usage_dependent_bednet,
        "ITN-CWC catch-up",
        ["target_age_min", "target_age_max"],
    )

    age_min = CWC_DELAY_DAYS / DAYS_PER_YEAR

    for r, batches in batches_by_row:
        for b_start, b_end in batches:
            kwargs = dict(
                intervention_name="ITN_CWC",
                start_day=int(b_end),
                demographic_coverage=float(r.cov),
                target_age_min=age_min,
                target_age_max=age_min + (b_end - b_start + 1) / DAYS_PER_YEAR,
                blocking_initial_effect=float(r.blocking_initial),
                blocking_decay_time_constant=float(r.blocking_decay),
                killing_initial_effect=float(r.killing_initial),
                killing_decay_time_constant=float(r.killing_decay),
                **itn_usage_kwargs("CWC", int(r.year), b_end),
            )

            discard = weibull_discard_config(r)
            if discard is not None:
                kwargs["discard_config"] = discard

            call_supported(
                add_scheduled_usage_dependent_bednet,
                campaign,
                "ITN-CWC catch-up",
                **kwargs,
            )


# IRS

def add_irs(campaign, df):
    from emodpy_malaria.interventions.irs import (
        add_scheduled_irs_housing_modification,
    )

    if not has_rows(df):
        return

    for r in df.itertuples():
        call_supported(
            add_scheduled_irs_housing_modification,
            campaign,
            "IRS",
            intervention_name=r.iv_name,
            start_day=int(r.pickup_day),
            demographic_coverage=float(r.cov),
            repetitions=int(r.reps),
            timesteps_between_repetitions=int(r.rep_gap),
            killing_initial_effect=float(r.killing_initial),
            killing_box_duration=int(r.killing_box),
            killing_decay_time_constant=float(r.killing_decay),
            repelling_initial_effect=float(r.repelling_initial),
            repelling_box_duration=int(r.repelling_box),
            repelling_decay_time_constant=float(r.repelling_decay),
            insecticide="",
        )


# SMC

def add_smc(campaign, df):
    from emodpy_malaria.interventions.drug_campaign import add_drug_campaign

    if not has_rows(df):
        return

    for r in df.itertuples():
        call_supported(
            add_drug_campaign,
            campaign,
            "SMC",
            campaign_type="SMC",
            drug_code=r.drug_code,
            start_days=[int(r.pickup_day)],
            coverage=float(r.cov),
            repetitions=1,
            target_group={"agemin": float(r.amin), "agemax": float(r.amax)},
            receiving_drugs_event_name="Received_SMC",
        )


# IPTp

def add_iptp(campaign, df):
    """
    Pregnancy-triggered, dose-chained IPTp.

        Pregnant -> 16 weeks -> D1 (cov = iptp1)
        D(k-1) received -> 30 days -> Dk (cov = iptpK / iptp(K-1))

    Dose events are year-specific so each year's chain stays separate.
    """
    from emodpy_malaria.interventions.drug_campaign import add_drug_campaign

    if not has_rows(df):
        return

    # Dropping any of these would turn a triggered campaign into an
    # immediate one.
    require_supported(
        add_drug_campaign,
        "IPTp",
        [
            "triggered_campaign_delay",
            "trigger_condition_list",
            "listening_duration",
        ],
    )

    d = IPTP_DEFAULTS

    for r in df.itertuples():
        year = int(r.year)

        for n, cum, cond in r.doses:
            event = f"Received_IPTp_D{n}_{year}"

            if cond <= 0:
                continue

            if n == 1:
                window = iptp_trigger_window(r.year_start_day)

                if window is None:
                    break

                start, listen = window
                trigger = [d["trigger"]]
                delay = d["delay_days"]

            else:
                start = 1
                listen = -1
                trigger = [f"Received_IPTp_D{n - 1}_{year}"]
                delay = d["gap_days"]

            call_supported(
                add_drug_campaign,
                campaign,
                "IPTp",
                campaign_type="MDA",
                drug_code=d["drug_code"],
                start_days=[int(start)],
                repetitions=1,
                coverage=float(cond),
                triggered_campaign_delay=delay,
                trigger_condition_list=trigger,
                listening_duration=int(listen),
                receiving_drugs_event_name=event,
            )


# VACCINE

def add_vaccine(campaign, df):
    """
    RTS,S malaria vaccine implementation.

    GHS schedule:
        D1: 6 months
        D2: 7 months
        D3: 9 months
        D4: 24 months before 2023, 18 months from 2023 onward

    The CSV holds CUMULATIVE coverage by dose within each birth cohort
    (year = birth year). load_vaccine() validates it and derives the
    conditional uptake `cond_cov` (see "Vaccine" in the module docstring);
    only cohorts that passed validation reach this function.

    D1:
        Births (during the cohort year) -> wait until target age
        -> D1, demographic coverage = D1 coverage.

    D2+:
        Actual receipt of the previous dose of the SAME cohort year
        -> wait (target age - previous target age) -> next dose,
        demographic coverage = cond_cov = cum(Dk) / cum(Dk-1).

    Event names carry the cohort year, so each year's coverage applies only
    to the children born in that year. No coverage is assumed for years that
    are not in the CSV, and doses falling after the last pickup day
    (day 4,745) are not delivered. There is no catch-up for children born
    before the first cohort year.
    """
    from emodpy_malaria.interventions.vaccine import add_triggered_vaccine

    if not has_rows(df):
        return

    for r in df.itertuples():
        year = int(r.year)

        # Nobody receives this dose (and so nobody receives later ones).
        if r.cond_cov <= 0:
            continue

        common_kwargs = dict(
            intervention_name=f"Vaccine_{r.dose}_{year}",
            target_age_min=float(r.amin),
            target_age_max=float(r.amax),
            vaccine_type=str(r.vaccine_type),
            vaccine_initial_effect=float(r.initial_effect),
            vaccine_box_duration=float(r.box_duration),
            vaccine_decay_time_constant=float(r.decay),
            broadcast_event=f"Received_Vaccine_{r.dose}_{year}",
        )

        # D1: Birth -> wait until the GHS target age -> D1
        #   (D1 target age = 6 months: Birth + 6 months -> D1)
        if pd.isna(r.prev_dose) or r.prev_dose is None:
            call_supported(
                add_triggered_vaccine,
                campaign,
                "Vaccine",
                trigger_condition_list=["Births"],
                delay_period_constant=(
                    float(r.target_age_months) * DAYS_PER_MONTH
                ),
                start_day=int(r.pickup_day),
                listening_duration=DAYS_PER_YEAR,
                demographic_coverage=float(r.cond_cov),
                **common_kwargs,
            )

        # D2+: previous dose RECEIVED -> wait until next target age -> next dose.
        #   D2: D1 at 6 months  + 1 month  = 7 months
        #   D3: D2 at 7 months  + 2 months = 9 months
        #   D4: 2022 and earlier: D3 + 15 months = 24 months
        #       2023 onward:      D3 + 9 months  = 18 months
        # Coverage = conditional uptake: cum(Dk) / cum(Dk-1) of this cohort.
        else:
            call_supported(
                add_triggered_vaccine,
                campaign,
                "Vaccine",
                trigger_condition_list=[
                    f"Received_Vaccine_{r.prev_dose}_{year}"
                ],
                delay_period_constant=float(r.delay_days),
                demographic_coverage=float(r.cond_cov),
                start_day=1,
                listening_duration=-1,
                **common_kwargs,
            )


# HABITAT SCALING

def scale_habitat_from_mix(mix):
    """Apply burn-in habitat mix to vector species fractions."""
    scales = {"NodeID": vector_df["node_id"].unique()}

    for sp in vector_df["species"].unique():
        frac = vector_df.loc[vector_df["species"] == sp, "fraction"]

        scales[f"CONSTANT.{sp}"] = [mix[0] * f for f in frac]
        scales[f"TEMPORARY_RAINFALL.{sp}"] = [mix[1] * f for f in frac]
        scales[f"WATER_VEGETATION.{sp}"] = [mix[2] * f for f in frac]

    return pd.DataFrame(scales)


# CAMPAIGN BUILDER

def build_camp(habitat_mix=(1.0, 1.0, 1.0)):
    """
    Build the complete pickup campaign.

    In an actual simulation, habitat_mix is replaced by
    scale_constant / scale_temp_rain / scale_water_veg
    from that simulation's burn-in row.
    """
    camp.set_schema(manifest.schema_file)

    s = SCHEDULES

    # Pregnancy tracking
    add_pregnancy_tracking(camp)

    # Larval habitat
    add_scale_larval_habitats(
        camp,
        df=scale_habitat_from_mix(habitat_mix),
        start_day=1,
        repetitions=-1,
    )

    # Case management
    add_case_management(camp, s["cm"])

    # Mass ITN (age- and season-dependent usage)
    if has_rows(s["itn"]):
        add_itns(camp, s["itn"])

    # Routine ANC ITN (one annual window per delivery year)
    if has_rows(s["itn_anc"]):
        add_routine_itns(camp, s["itn_anc"], "ANC")

    # Routine CWC ITN (ITN_SEASON_BATCH_DAYS-day batches) + catch-up
    if has_rows(s["itn_cwc"]):
        add_routine_itns(camp, s["itn_cwc"], "CWC")
        add_cwc_catchup(camp, s["itn_cwc"])

    # IRS
    if has_rows(s["irs"]):
        add_irs(camp, s["irs"])

    # SMC
    if has_rows(s["smc"]):
        add_smc(camp, s["smc"])

    # IPTp
    if has_rows(s["iptp"]):
        add_iptp(camp, s["iptp"])

    # Vaccine (validated cohorts only)
    if has_rows(s["vaccine"]):
        add_vaccine(camp, s["vaccine"])

    # Clean property restrictions
    fix_property_restrictions(camp.campaign_dict)

    return camp


# SERIALIZATION SWEEP

def load_burnin_df(burnin_id, platform):
    from utils_slurm import build_burnin_df

    df = build_burnin_df(burnin_id, platform, BURNIN_DAYS)

    required = ["x_Temporary_Larval_Habitat"] + HABITAT_TAGS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Burn-in experiment {burnin_id} is missing tag column(s) "
            f"{missing}. Tags actually present: {list(df.columns)}"
        )

    for key, value in BURNIN_FILTER.items():
        if key not in df.columns:
            raise ValueError(
                f"BURNIN_FILTER key '{key}' is not a burn-in tag. "
                f"Tags: {list(df.columns)}"
            )

        df = df[np.isclose(df[key].astype(float), float(value))]

    if df.empty:
        raise ValueError(
            f"No burn-in simulations left for experiment "
            f"{burnin_id} after BURNIN_FILTER."
        )

    return df.reset_index(drop=True)


def _find_key(obj, key):
    """First value stored under `key` anywhere in a nested dict / list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, key)
            if found is not None:
                return found

    return None


def _first_json(paths):
    """Load the first readable JSON file in `paths`; (path, data) or (None, None)."""
    for path in paths:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return path, json.load(f)
            except (OSError, ValueError):
                continue

    return None, None


def check_burnin_consistency(burnin_df):
    """
    Compare the first burn-in simulation's own birth settings with this
    script's. Best effort and WARNING-only: file layouts differ between
    platforms, and a missing file must never block a submission.

    Config settings are not stored in the serialized state, so a mismatch
    here means the pickup would run with different demographics from the
    ones the population was spun up under.
    """
    print("\nBurn-in demographic consistency check (first burn-in simulation):")

    try:
        sim_dir = str(burnin_df.iloc[0]["serialized_file_path"])
    except (KeyError, IndexError):
        print("  skipped: no serialized_file_path in the burn-in table.")
        return

    cfg_path, cfg = _first_json([os.path.join(sim_dir, "config.json")])

    if cfg is None:
        print(f"  skipped: no readable config.json in {sim_dir}")
    else:
        params = cfg.get("parameters", cfg) if isinstance(cfg, dict) else {}

        dep = params.get("Birth_Rate_Dependence")
        xb = params.get("x_Birth")

        if dep is None:
            print("  WARNING: burn-in config.json has no Birth_Rate_Dependence.")
        elif dep != BIRTH_RATE_DEPENDENCE:
            print(
                f"  WARNING: burn-in Birth_Rate_Dependence = {dep}, "
                f"pickup uses {BIRTH_RATE_DEPENDENCE}."
            )
        else:
            print(f"  Birth_Rate_Dependence matches ({dep}).")

        if xb is None:
            print("  note: burn-in config.json does not list x_Birth "
                  "(EMOD default 1).")
        elif not np.isclose(float(xb), X_BIRTH):
            print(f"  WARNING: burn-in x_Birth = {xb}, pickup uses {X_BIRTH:g}.")
        else:
            print(f"  x_Birth matches ({float(xb):g}).")

    demog_path, demog = _first_json([
        os.path.join(sim_dir, "demographics.json"),
        os.path.join(sim_dir, "Assets", "demographics.json"),
        os.path.join(os.path.dirname(sim_dir.rstrip(os.sep)), "Assets",
                     "demographics.json"),
    ])

    if demog is None:
        print("  note: demographics.json not found next to the burn-in "
              "simulation; BirthRate not compared (check it by hand).")
        return

    birth_rate = _find_key(demog, "BirthRate")

    if birth_rate is None:
        print(f"  note: no BirthRate in {demog_path}; not compared.")
    elif not np.isclose(float(birth_rate), PREG_RATE_PER_MOTHER_DAY, rtol=1e-6):
        print(
            f"  WARNING: burn-in BirthRate = {float(birth_rate):.6g}, "
            f"pickup PREG_RATE_PER_MOTHER_DAY = {PREG_RATE_PER_MOTHER_DAY:.6g}. "
            f"Copy the burn-in value into this script."
        )
    else:
        print(f"  BirthRate matches ({float(birth_rate):.6g} per mother per day).")


def update_serialize_parameters(simulation, df, x: int):
    """
    Point one pickup simulation at one burn-in state.

    Inherits: serialized state, Run_Number, xTLH, habitat mix.
    """
    row = df.iloc[x]

    # ---- Serialized state ---------------------------------------------------
    filenames = row["Serialized_Population_Filenames"]

    if isinstance(filenames, str):
        filenames = [filenames]

    simulation.task.set_parameter("Serialized_Population_Filenames", list(filenames))

    simulation.task.set_parameter(
        "Serialized_Population_Path",
        os.path.join(row["serialized_file_path"], "output"),
    )

    # ---- Seed ---------------------------------------------------------------
    seed = int(row["Run_Number"])
    simulation.task.set_parameter("Run_Number", seed)

    # ---- Birth model and stale parameters -----------------------------------
    # This callback runs after the task (and all demographics implicits) has
    # been built, and EMODTask.set_parameter writes straight into task.config,
    # so this is the final word on the birth settings.
    simulation.task.set_parameter("Birth_Rate_Dependence", BIRTH_RATE_DEPENDENCE)
    simulation.task.set_parameter("x_Birth", X_BIRTH)

    # Drop parameters the schema does not define (Disable_IP_Whitelist) before
    # the config is finalized at commissioning, which is when emodpy prints the
    # "schema-based param purge" warning.
    remove_stale_params(simulation.task.config)

    # ---- Habitat parameters -------------------------------------------------
    xtlh = float(row["x_Temporary_Larval_Habitat"])
    scale_constant = float(row["scale_constant"])
    scale_temp_rain = float(row["scale_temp_rain"])
    scale_water_veg = float(row["scale_water_veg"])

    simulation.task.set_parameter("x_Temporary_Larval_Habitat", xtlh)

    # Rebuild campaign with this simulation's own habitat mix.
    simulation.task.create_campaign_from_callback(
        partial(
            build_camp,
            habitat_mix=(scale_constant, scale_temp_rain, scale_water_veg),
        )
    )

    return {
        "Run_Number": seed,
        "burnin_sim_index": int(x),
        "x_Temporary_Larval_Habitat": xtlh,
        "scale_constant": scale_constant,
        "scale_temp_rain": scale_temp_rain,
        "scale_water_veg": scale_water_veg,
    }


# REPORTS

def recorder_events():
    """
    Events written to ReportEventRecorder.csv (also printed in the dry run).

    Starts from the burn-in's EVENT_LIST and adds the events that the pickup
    campaigns broadcast. Duplicates are removed, order is kept.
    """
    events = list(EVENT_LIST)

    if not RECORD_TRIGGER_EVENTS:
        events = [e for e in events if e not in TRIGGER_EVENTS]

    # Every net of every channel (mass, ANC, CWC, CWC catch-up): total check.
    events.append("Bednet_Got_New_One")

    # Routine ITN, YEAR-level events only. The batch events are used for
    # routing and are not recorded.
    for key, kinds in (
        ("itn_anc", ("ANC_Visit", "ANC_Net")),
        ("itn_cwc", ("CWC_Visit", "CWC_Net")),
    ):
        if has_rows(SCHEDULES[key]):
            for y in sorted({int(v) for v in SCHEDULES[key]["year"]}):
                events.extend(yearly_event_name(k, y) for k in kinds)

    if has_rows(SCHEDULES["smc"]):
        events.append("Received_SMC")

    if has_rows(SCHEDULES["iptp"]):
        events.extend(
            f"Received_IPTp_D{n}_{int(y)}"
            for y, doses in zip(
                SCHEDULES["iptp"]["year"], SCHEDULES["iptp"]["doses"]
            )
            for n, _, cond in doses
            if cond > 0
        )

    if has_rows(SCHEDULES["vaccine"]):
        v = SCHEDULES["vaccine"]

        events.extend(
            sorted(
                f"Received_Vaccine_{d}_{int(y)}"
                for d, y in zip(v["dose"], v["year"])
            )
        )

    return list(dict.fromkeys(events))


def add_reports(task):
    """Add the event recorder and one malaria summary report per year."""
    # ---- Event recorder -----------------------------------------------------
    events = recorder_events()

    add_event_recorder(
        task,
        event_list=events,
        start_day=1,
        end_day=PICKUP_DAYS,
        node_ids=[NODE_ID],
        min_age_years=EVENT_MIN_AGE_YEARS,
        max_age_years=EVENT_MAX_AGE_YEARS,
    )

    # ---- Annual summary reports with 30-day reporting periods ---------------
    # Same day convention as the burn-in's report block and the example:
    # start_day = 365 * year, end_day = start_day + 365.
    for i in range(PICKUP_YEARS):
        start_day = i * DAYS_PER_YEAR
        end_day = start_day + DAYS_PER_YEAR

        call_supported(
            add_malaria_summary_report,
            task,
            "Summary report",
            manifest=manifest,
            start_day=start_day,
            end_day=end_day,
            reporting_interval=REPORT_INTERVAL,
            # Upper edges in years. 0.5 and 5 make the 0.5-5 y bin equal to
            # 6-59 months, matching the prevalence surveys.
            age_bins=REPORT_AGE_BINS,
            max_number_reports=REPORT_MAX_NUMBER,
            parasite_detection_threshold=PARASITE_DETECTION_THRESHOLD,
            pretty_format=True,
            filename_suffix=f"Monthly_U5_{PICKUP_START_YEAR + i}",
        )

    return task


# EXPERIMENT NAME

def full_experiment_name():
    """Experiment name prefixed with the login name (as in the burn-in)."""
    return f"{os.getlogin()}_{EXPERIMENT_NAME}"


# COMMAND LINE

def parse_args():
    parser = argparse.ArgumentParser(description="UWR 2013-2025 pickup.")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check inputs, print schedule, write "
            "campaign_pickup_preview.json, then stop."
        ),
    )

    parser.add_argument(
        "--test-one",
        action="store_true",
        help=(
            "Submit only the first burn-in state, under an experiment name "
            "ending in _TEST, so config.json can be inspected before the "
            "full submission."
        ),
    )

    parser.add_argument(
        "--allow-excluded-vaccine-cohorts",
        action="store_true",
        help=(
            "Submit even though one or more vaccine cohorts failed "
            "validation. Those cohorts get NO vaccine campaign. Without "
            "this flag, submission is refused while any cohort is excluded."
        ),
    )

    return parser.parse_args()


# SUBMISSION

def submit_pickup(platform, burnin_df, climate_dir):
    """Submit exactly one pickup simulation for each selected burn-in simulation."""
    n_sims = len(burnin_df)

    print(
        f"\nSimulations in this submission: {n_sims} "
        f"(= selected burn-in states, 1:1)"
    )

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

    builder.add_sweep_definition(
        partial(update_serialize_parameters, df=burnin_df),
        range(n_sims),
    )

    experiment_name = full_experiment_name()

    print("\nSubmitting under experiment name:")
    print(f"  {experiment_name}")

    experiment = Experiment.from_builder(builder, task, name=experiment_name)

    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")
    print(f"Name: {experiment_name}")
    print(f"Simulations: {n_sims}")

    print(
        "\nAfter the run: confirm in one simulation's folder that config.json "
        f"has Birth_Rate_Dependence = {BIRTH_RATE_DEPENDENCE} and x_Birth = "
        f"{X_BIRTH:g}, that demographics.json has BirthRate = "
        f"{PREG_RATE_PER_MOTHER_DAY:g}, and that 'Statistical Population' in "
        f"InsetChart.json changes smoothly from the burn-in's final value."
    )

    return experiment


# GENERAL SIMULATION

def general_sim(selected_platform, burnin_id, climate_dir, test_one=False):
    global EXPERIMENT_NAME

    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    burnin_df = load_burnin_df(burnin_id, platform)

    limit = int(os.environ.get("PICKUP_LIMIT", "0"))
    if limit > 0:
        burnin_df = burnin_df.head(limit).reset_index(drop=True)
        EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_MEMTEST{limit}"
        print(f"\n*** MEMORY TEST: only {limit} simulations ***")

    if test_one:
        burnin_df = burnin_df.head(1).reset_index(drop=True)

        if not EXPERIMENT_NAME.endswith("_TEST"):
            EXPERIMENT_NAME = EXPERIMENT_NAME + "_TEST"

        print("\n*** TEST MODE: submitting 1 simulation only ***")

    print(f"\nBurn-in experiment: {burnin_id}")
    print(f"Burn-in states selected: {len(burnin_df)}")
    print(f"Experiment name: {full_experiment_name()}")

    check_burnin_consistency(burnin_df)
    check_burnin_grid(burnin_df)

    return submit_pickup(platform, burnin_df, climate_dir)


# MAIN

if __name__ == "__main__":
    args = parse_args()

    # Load and print all intervention schedules.
    SCHEDULES = load_all_schedules()

    # ITN usage curves (age and season); stops here if a file is missing.
    if USE_ITN_USAGE_CURVES:
        ITN_USAGE = load_itn_usage()
        validate_itn_usage(SCHEDULES)

    describe_schedules(SCHEDULES)

    # Demographic settings (must match the burn-in).
    describe_demographics()
    validate_demographic_settings()

    # Pregnancy model check.
    if "PREGNANC" not in BIRTH_RATE_DEPENDENCE:
        msg = (
            f"BIRTH_RATE_DEPENDENCE = {BIRTH_RATE_DEPENDENCE}: "
            f"the 'Pregnant' and 'GaveBirth' events never fire, so ANC nets "
            f"would deliver nothing. It must match the burn-in "
            f"(INDIVIDUAL_PREGNANCIES)."
        )

        if args.dry_run:
            print(f"\n  WARNING: {msg}")
        else:
            sys.exit(msg)

    # Vaccine cohorts that failed validation get no campaign. Refuse to
    # submit unless the user explicitly accepts that.
    if VACCINE_ISSUES:
        msg = (
            f"Vaccine cohort(s) {sorted(VACCINE_ISSUES)} failed validation "
            f"(see ERROR above) and would get NO vaccine campaign. Correct "
            f"{VACCINE_FILE}, or pass --allow-excluded-vaccine-cohorts to "
            f"accept this."
        )

        if args.dry_run:
            print(f"\n  WARNING: {msg}")
        elif args.allow_excluded_vaccine_cohorts:
            print(f"\n  NOTE: proceeding with excluded vaccine cohort(s) "
                  f"{sorted(VACCINE_ISSUES)} (flag given).")
        else:
            sys.exit(msg)

    # Climate.
    climate_dir = resolve_climate_dir()
    validate_climate(climate_dir)

    # Dry run.
    if args.dry_run:
        describe_habitat_grid()
        describe_itn_usage(SCHEDULES)
        describe_vaccine_timing(SCHEDULES["vaccine"])

        print("\nRoutine ITN timing:")
        print(f"  ANC: Pregnant -> {ANC_DELAY_DAYS} days -> ANC_Visit "
              f"(eligible) -> ANC_Net (yearly coverage, still pregnant) -> ITN "
              f"({ANC_ITN_BATCH_DAYS}-day batch, one window per year)")
        print(f"  CWC: Births -> {CWC_DELAY_DAYS} days -> CWC_Visit "
              f"(eligible) -> CWC_Net (yearly coverage) -> ITN "
              f"({routine_batch_size('CWC')}-day batches)")
        print("  Coverage = yearly CSV value applied once to the eligible "
              "group; batches only rotate the seasonal usage curve.")
        print("  Recorded: <ANC|CWC>_<Visit|Net>_<year>; "
              "coverage = Net / Visit per year.")

        print("\nReports:")
        print(f"  Malaria Summary Reports: {PICKUP_START_YEAR}-{PICKUP_END_YEAR}, "
              f"one per year ({REPORT_MAX_NUMBER} x {REPORT_INTERVAL}d), "
              f"age bins {REPORT_AGE_BINS}")
        print(f"  Event recorder ages {EVENT_MIN_AGE_YEARS}-{EVENT_MAX_AGE_YEARS}")

        print(f"\nExperiment name: {full_experiment_name()}")

        print("\nNOTE: dry-run preview uses a PLACEHOLDER habitat mix (1.0, 1.0, 1.0).")
        print("The real pickup uses each burn-in row's own habitat mix.")

        build_camp(habitat_mix=(1.0, 1.0, 1.0))
        camp.save("campaign_pickup_preview.json")

        events = recorder_events()
        names = sorted({re.sub(r"_(?:19|20)\d{2}(?:_\d{2})?$", "", e) for e in events})
        print(f"\nEvent recorder will record {len(events)} events:")
        for name in names:
            n = sum(re.sub(r"_(?:19|20)\d{2}(?:_\d{2})?$", "", e) == name for e in events)
            print(f"  {name}" + (f"  (x{n} years)" if n > 1 else ""))

        print("\nDry run OK. Campaign written to campaign_pickup_preview.json.")
        print("Nothing submitted.")

        sys.exit(0)

    # Require burn-in experiment ID.
    burnin_id = BURNIN_EXPERIMENT_ID.strip()

    if not burnin_id:
        sys.exit(
            "Set BURNIN_EXPERIMENT_ID at the top of the script to "
            "the finished burn-in experiment ID."
        )

    # EMOD malaria bootstrap.
    import emod_malaria.bootstrap as dtk

    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)

    # Submit.
    general_sim("SLURM_LOCAL", burnin_id, climate_dir, test_one=args.test_one)