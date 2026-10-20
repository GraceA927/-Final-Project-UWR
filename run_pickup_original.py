"""
Upper West Region EMOD malaria model -- PICKUP 2013-2025

Adapts and calibrates EMOD to the Upper West Region of Ghana.

    1963 ---------- 2012 | 2013 ---------- 2025
      50-year burn-in    |  pickup: CM, ITN (routine ANC/CWC only),
                         |  IRS, SMC, vaccine
                         ^
              serialized state (day 18,250) read here

Usage
-----
1. Set BURNIN_EXPERIMENT_ID below once the burn-in has finished.

2. Preview inputs and campaign without submitting:
       python run_pickup_UWR.py --dry-run

3. Submit ONE test simulation (first burn-in state) under a *_TEST
   experiment name, then inspect its config.json:
       python run_pickup_UWR.py --test-one

       grep -o '"Birth_Rate_Dependence"[^,]*' <sim_dir>/config.json
       grep -c 'Disable_IP_Whitelist' <sim_dir>/config.json     # expect 0

4. Submit pickup (all burn-in states):
       python run_pickup_UWR.py


Calendar
--------
Day 1 = 1 Jan 2013 using a flat 365-day calendar.

2013 -> day 1
2014 -> day 366
...
2025 -> day 4,381
31 Dec 2025 -> day 4,745


Intervention timing
-------------------
The `year` column is authoritative.

For general intervention files:
    day_of_year -> used if present
    start_day   -> used if day_of_year is absent
    month       -> used if both above are absent
    otherwise   -> 1 Jan of the specified year

Rows outside 2013-2025 are dropped.

Routine ANC/CWC ITN files use the `year` column ONLY.
Their `start_day` column, if present, is ignored.


Birth model and pregnancy
-------------------------
The burn-in uses INDIVIDUAL_PREGNANCIES and an individual property:

    Pregnancy = NotPregnant / IsPregnant

The pickup uses the same BIRTH_RATE_DEPENDENCE and adds the same
individual property.

The campaign keeps the property current through:

    Pregnant  -> IsPregnant
    GaveBirth -> NotPregnant

NOTE (emod_api / emodpy documentation): the config is built in this order

    1. set_param_fn (param_custom_cb)
    2. each function in demographics.implicits, in list order
       (task.config = fn(task.config))
    3. the per-simulation sweep callback (update_serialize_parameters)

demog.SetBirthRate() appends an implicit function that sets the birth
model, and AddIndividualPropertyAndHINT("Pregnancy", ...) appends one that
sets Disable_IP_Whitelist = 1. Both run AFTER set_param_fn, so values set
there can be overwritten. This script therefore:

    1. appends a final function to demog.implicits (build_demog) that
       forces BIRTH_RATE_DEPENDENCE and removes Disable_IP_Whitelist, and
    2. repeats both in update_serialize_parameters using
       task.set_parameter / config.parameters.pop, which run after every
       implicit config.

Disable_IP_Whitelist is not in the current schema, so leaving it in makes
emodpy print "schema-based param purge" warnings for every simulation.

Verify with --test-one and the grep shown under Usage.


Routine LLIN delivery
---------------------
Two independent routine ITN channels are used:

    interventions_ITN_ANC.csv
        Pregnant women through ANC.

    interventions_ITN_CWC.csv
        Children through CWC at approximately 18 months.

ANC:
    Pregnant event
        -> wait 16 weeks
        -> deliver ITN to women still IsPregnant (ANC_CHECK_PREGNANT_AT_DELIVERY).

CWC:
    Births event
        -> wait 18 months
        -> deliver ITN.

The listening window is shifted backward by the delivery delay so that
nets are delivered during the CSV year.

The ANC and CWC interventions have separate intervention names:

    ITN_ANC
    ITN_CWC

so that routine nets can stack with mass-campaign nets.

ITN usage (age and season)
--------------------------
UsageDependentBednet separates net OWNERSHIP from net USE:

    ownership   demographic coverage (who receives a net)
    attrition   discard_config (nets are lost / discarded over time)
    use         age_dependence x seasonal_dependence

On any night, the probability that a net holder uses the net is

    age value (at the holder's age)  x  seasonal value

so the two curves are MULTIPLIED. Choose them so that their product
reproduces the observed use, not so that each reaches the observed level.

Curves are read from INPUT_DIR (both required while USE_ITN_USAGE_CURVES
is True; the script stops if either is missing, instead of silently
falling back to flat 100% use):

    interventions_ITN_age.csv
        node_id, phase, year, age, age_usage
        age         age in years
        age_usage   probability of use, 0-1
    interventions_ITN_season.csv
        node_id, season_time, season_usage
        season_time    days since 1 Jan, 0-based (0 = 1 Jan, 90 = 1 Apr,
                       182 = 1 Jul). A closing row at 365 that repeats 0 is
                       ignored. (`day_of_year`, 1-365, or `month` also work.)
        season_usage   probability of use, 0-1

Optional columns in both files:
    year      curve applies to nets delivered in that year or later (until a
              later year is given); blank = every year. Delivery years before
              the earliest dated curve use that earliest curve (WARNING).
    channel   Mass / ANC / CWC; blank = every channel
    node_id, phase   filtered as in the other input files
Value-column aliases also accepted: usage, value, use, usage_prob, probability,
prob. The age curve is held at its last value up to 125 years.

How EMOD interprets the curves (emodpy_malaria / EMOD documentation):

    age_dependence       {"Times": ages in years, "Values": 0-1}
                         linear interpolation on the holder's age.
    seasonal_dependence  {"Times": days, "Values": 0-1}
                         WaningEffectMapLinearSeasonal. Times are days SINCE
                         THE NET WAS DISTRIBUTED, repeating every 365 days.
                         They are NOT calendar days.

Because of that last point, a calendar-based seasonal curve is rotated to each
net's delivery date before it is passed to EMOD:

    mass campaigns and CWC catch-up   delivery date is known: exact.
    routine ANC/CWC nets              different people receive nets on
                                      different days, so each delivery year is
                                      split into ITN_SEASON_BATCH_DAYS-day
                                      batches (one triggered intervention
                                      each, rotated to the batch midpoint).
                                      Seasonal phase error is at most half a
                                      batch.

Age dependence applies to the net holder's age at any time, so it needs no
rotation. Pregnancy status cannot be a separate usage group in
UsageDependentBednet; use the `channel` column to give ANC nets their own
curves (they reach adult women).


Routine CWC catch-up
--------------------
A Births-triggered net cannot reach children who are already born when
the pickup starts. Without a fix, CWC nets due in the first
CWC_DELAY_DAYS (548) days of the pickup would never be delivered, so
2013 would get no CWC nets and 2014 only the part after day 548.

Those deliveries are therefore added as scheduled monthly batches: each
batch targets children who have just turned 18 months (age window equal
to the batch length), using the same coverage and net parameters as the
triggered nets. Births from day 1 onward use the normal trigger.


IPTp
----
interventions_IPTp.csv gives cumulative coverage by dose per year:

    iptp1_coverage, iptp2_coverage, iptp3_coverage, iptp4_coverage

(share of pregnancies receiving at least that many doses). Doses are
chained, SP (drug code 'SP'):

    Pregnant event
        -> wait 16 weeks
        -> dose 1  (coverage = iptp1)
        -> 30 days after the previous dose, dose k
           (coverage = iptpK / iptp(K-1), applied only to women who
            received dose K-1)

Like the routine ITNs, `year` is the year in which dose 1 is delivered
(the trigger window is shifted back by the 16-week delay). Each dose
broadcasts Received_IPTp_D<k>_<year>. A blank dose coverage means that
dose (and later ones) is not delivered that year. If a later dose has
higher coverage than the one before it, the conditional coverage is
capped at 1 and a WARNING is printed.


Larval habitat
--------------
Each burn-in simulation carries its own calibrated:

    x_Temporary_Larval_Habitat
    scale_constant
    scale_temp_rain
    scale_water_veg

Pickup applies those four values directly.

There is no independent habitat sweep at pickup.

One burn-in state -> one pickup simulation.

Serialization_Mask_Node_Read = 0 reads everything from the serialized
state, including accumulated larval habitat capacities.


Inputs
------
INPUT_DIR must contain:

    demographics.csv
    vector.csv
    intervention_CMBN.csv
    interventions_ITN_ANC.csv
    interventions_ITN_CWC.csv
    intervention_IRS.csv
    intervention_SMC.csv
    interventions_IPTp.csv
    intervention_malariaVaccine.csv

(interventions_ITN.csv, mass campaigns, is not used: no programme records.)

interventions_ITN_age.csv     (ITN usage by age)
interventions_ITN_season.csv  (ITN usage by season)


Emodpy compatibility
--------------------
Keyword arguments are checked against the installed emodpy_malaria
functions.

Unsupported arguments are dropped and reported as WARNINGs during
the dry run.
"""

import argparse
import inspect
import json
import os
import pathlib
import re
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
import emod_api.interventions.common as common

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import manifest


# =============================================================================
# PROJECT / MODEL SETTINGS
# =============================================================================

SITE_NAME = "UpperWest_Region"
EXPERIMENT_NAME = "Pickup_FOR_ECM13yrs_PREG"
NODE_ID = 1

# Must match burn-in.
BIRTH_RATE = 31.0
BIRTH_RATE_DEPENDENCE = "INDIVIDUAL_PREGNANCIES"

MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
CLIMATE_ROOT = os.path.join(PROJECT_DIR, "climate", "pickup")
CLIMATE_SUBDIR = None

VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"
CM_FILE = "intervention_CMBN.csv"

# ITN files
# Mass ITN campaigns are NOT used: the malaria control programme has no
# mass-campaign records, so routine ANC / CWC nets are the only ITN delivery.
# To re-enable, restore the line below and remove "ITN_FILE = None".
# ITN_FILE = "interventions_ITN.csv"      # mass campaigns
ITN_FILE = None
ITN_ANC = "interventions_ITN_ANC.csv"   # routine, antenatal care
ITN_CWC = "interventions_ITN_CWC.csv"   # routine, child welfare clinic

# ITN USE (as opposed to ownership) varies by age and season. Both files are
# required while USE_ITN_USAGE_CURVES is True; see the module docstring for
# their layout.
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"

USE_ITN_USAGE_CURVES = True

# EMOD measures the seasonal curve from the day each net is distributed.
# Routine (triggered) nets reach different people on different days, so each
# delivery year is split into batches of this many days, and the seasonal
# curve is rotated to each batch's midpoint. Smaller = better phase accuracy,
# more campaign events. (Ignored when USE_ITN_USAGE_CURVES is False.)
ITN_SEASON_BATCH_DAYS = 92

IRS_FILE = "intervention_IRS.csv"
SMC_FILE = "intervention_SMC.csv"
IPTP_FILE = "interventions_IPTp.csv"
VACCINE_FILE = "intervention_malariaVaccine.csv"

# Config parameters that the installed schema no longer defines. They are
# removed in set_param_fn so emodpy's schema-based purge does not print a
# warning for every simulation.
STALE_CONFIG_PARAMS = ("Disable_IP_Whitelist",)


# =============================================================================
# BURN-IN EXPERIMENT
# =============================================================================

# Set this after the new pregnancy-enabled burn-in finishes.
BURNIN_EXPERIMENT_ID = "8451120b-182c-4934-b7df-49bc0f17ec62"

# Optional subset of burn-in simulations. Empty means all.
# Example:
# BURNIN_FILTER = {
#     "x_Temporary_Larval_Habitat": 35.0,
#     "scale_temp_rain": 10.0,
# }
BURNIN_FILTER = {}

# These values are inherited directly from each burn-in simulation.
HABITAT_TAGS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]

# Save state at the end of pickup.
WRITE_PICKUP_STATE = True

# Microscopy-like parasite detection threshold.
PARASITE_DETECTION_THRESHOLD = 40

# Also record the built-in events that drive ANC, IPTp, CWC and the vaccine
# (Pregnant, GaveBirth, Births), so the event counts show whether those
# triggers fire. Adds one row per pregnancy / birth to each simulation's
# ReportEventRecorder.csv.
RECORD_TRIGGER_EVENTS = True


# =============================================================================
# CALENDAR
# =============================================================================

DAYS_PER_YEAR = 365
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_DAYS = (BURNIN_END_YEAR - BURNIN_START_YEAR + 1) * DAYS_PER_YEAR

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1
PICKUP_DAYS = PICKUP_YEARS * DAYS_PER_YEAR


# =============================================================================
# ROUTINE ITN TIMING
# =============================================================================

# Approximately 16 weeks after conception.
ANC_DELAY_DAYS = 16 * 7

# If True, ANC nets are only given to women who are STILL pregnant at
# delivery. This is done in two stages:
#
#   Pregnant -> wait 16 weeks -> broadcast ANC_Visit_<year>_<batch>
#   ANC_Visit_<year>_<batch> -> ITN, restricted to Pregnancy=IsPregnant
#
# so the check runs at delivery time. (A restriction on the first trigger
# would be checked at conception, before the delay, and could race with the
# campaign event that sets Pregnancy=IsPregnant.)
# If False, a single triggered ITN is used with Female targeting only.
ANC_CHECK_PREGNANT_AT_DELIVERY = True

# Approximately 18 months after birth.
CWC_AGE_MONTHS = 18
CWC_DELAY_DAYS = int(round(CWC_AGE_MONTHS * DAYS_PER_MONTH))


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

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

CM_DRUGS = {
    "AL": ["Artemether", "Lumefantrine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}

SEVERE_TRIGGERS = ["NewSevereCase"]

ITN_DEFAULTS = dict(
    blocking_initial=0.9,
    blocking_decay=730,
    killing_initial=0.6,
    killing_decay=1460,
    discard_k=None,
    discard_l=None,
)

IRS_DEFAULTS = dict(
    killing_initial=0.7,
    killing_box=0,
    killing_decay=180,
    repelling_initial=0.0,
    repelling_box=0,
    repelling_decay=90,
    repetitions=1,
    timesteps_between_repetitions=365,
)


# =============================================================================
# IRS PRODUCT PARAMETERS
# =============================================================================

IRS_PRODUCTS = {
    "pirimiphos methyl": dict(
        killing_initial=1.0, killing_box=0, killing_decay=180,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
    "propoxur": dict(
        killing_initial=1.0, killing_box=0, killing_decay=60,
        repelling_initial=0.0, repelling_box=0, repelling_decay=60,
    ),
    "clothianidin": dict(
        killing_initial=1.0, killing_box=0, killing_decay=240,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
    "clothianidin + deltamethrin": dict(
        killing_initial=1.0, killing_box=0, killing_decay=240,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
    "broflanilide": dict(
        killing_initial=1.0, killing_box=0, killing_decay=270,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
}

IRS_PARAM_COLUMNS = {
    "killing_initial": "killing_initial_effect",
    "killing_box": "killing_box_duration",
    "killing_decay": "killing_decay_time_constant",
    "repelling_initial": "repelling_initial_effect",
    "repelling_box": "repelling_box_duration",
    "repelling_decay": "repelling_decay_time_constant",
}


# =============================================================================
# SMC
# =============================================================================

SMC_DEFAULTS = dict(age_min=0.25, age_max=5.0, drug_code="SPA")

SMC_DRUG_ALIASES = {"SPAQ": "SPA", "SP+AQ": "SPA", "SP-AQ": "SPA"}


# =============================================================================
# IPTp
# =============================================================================

IPTP_DEFAULTS = dict(
    drug_code="SP",   # Sulfadoxine + Pyrimethamine
    trigger="Pregnant",
    delay_days=16 * 7,   # trigger -> dose 1
    gap_days=30,         # previous dose -> next dose
)

IPTP_COLUMN_RE = re.compile(r"^iptp(\d+)_coverage$")



# =============================================================================
# VACCINE
# =============================================================================

VACCINE_DEFAULTS = dict(
    age_min=0.42,
    age_max=2.0,
    initial_effect=0.8,
    box_duration=0,
    decay=592,
    vaccine_type="AcquisitionBlocking",
)

VACCINE_CATCHUP_MAX_AGE_MONTHS = 59.0


# =============================================================================
# CSV HELPERS
# =============================================================================

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
    """Return first existing column in names, or a default-filled column."""
    for name in names:
        if name in df.columns:
            return df[name]

    if default is None:
        raise ValueError(
            f"{label}: needs one of the columns "
            f"{names}; found {list(df.columns)}"
        )

    return pd.Series([default] * len(df), index=df.index)


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
        doy = (df["start_day"].astype(int) - 1) % DAYS_PER_YEAR + 1

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


# =============================================================================
# INTERVENTION LOADING
# =============================================================================

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
    Fill ITN effect parameters (blocking, killing, discard).

    Age- and season-dependent USE is handled separately; see
    "ITN USAGE CURVES" below.
    """
    d = ITN_DEFAULTS

    df["cov"] = pick(df, ["coverage", "cov_all"], label=label).astype(float)

    df["blocking_initial"] = pick(
        df, ["block_effect", "blocking_initial"], d["blocking_initial"]
    ).astype(float)

    df["blocking_decay"] = pick(
        df, ["block_decay", "blocking_decay"], d["blocking_decay"]
    ).astype(float)

    df["killing_initial"] = pick(
        df, ["kill_effect", "killing_initial"], d["killing_initial"]
    ).astype(float)

    df["killing_decay"] = pick(
        df, ["kill_decay", "killing_decay"], d["killing_decay"]
    ).astype(float)

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


# =============================================================================
# ITN USAGE CURVES (age and season)
# =============================================================================
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
                df = df[~wrap].copy()
                usage = usage[~wrap]
                x = x[~wrap]
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
        f"  seasonal curve rotated to each delivery date; routine nets "
        f"delivered in {ITN_SEASON_BATCH_DAYS}-day batches"
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


def anc_visit_event_name(year, batch):
    return f"ANC_Visit_{int(year)}_{int(batch):02d}"


def routine_delivery_batches(year_start_day, delay):
    """
    Split a routine-net delivery year into trigger windows.

    Returns [(batch, start_day, listening_duration, mid_delivery_day)].

    Nets in a batch are delivered on days d0..d1 of the pickup, so the
    trigger window is d0-delay .. d1-delay. mid_delivery_day is used to
    rotate the seasonal curve. Without usage curves the whole year is one
    batch (one trigger window, as before).
    """
    size = ITN_SEASON_BATCH_DAYS if USE_ITN_USAGE_CURVES else DAYS_PER_YEAR

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


# =============================================================================
# IRS
# =============================================================================

def normalise_product(name):
    parts = [
        " ".join(p.replace("-", " ").replace("_", " ").lower().split())
        for p in str(name).split("+")
    ]
    return " + ".join(sorted(p for p in parts if p))


def product_params(name):
    key = normalise_product(name)

    if key in IRS_PRODUCTS:
        return IRS_PRODUCTS[key], "table"

    parts = key.split(" + ")
    missing = [p for p in parts if p not in IRS_PRODUCTS]

    if missing:
        raise ValueError(
            f"{IRS_FILE}: insecticide '{name}' has no entry for "
            f"{missing} in IRS_PRODUCTS."
        )

    if len(parts) == 1:
        return IRS_PRODUCTS[parts[0]], "table"

    comps = [IRS_PRODUCTS[p] for p in parts]

    combined = dict(
        killing_initial=1 - np.prod([1 - c["killing_initial"] for c in comps]),
        killing_box=max(c["killing_box"] for c in comps),
        killing_decay=max(c["killing_decay"] for c in comps),
        repelling_initial=1 - np.prod([1 - c["repelling_initial"] for c in comps]),
        repelling_box=max(c["repelling_box"] for c in comps),
        repelling_decay=max(c["repelling_decay"] for c in comps),
    )

    return combined, "combined from components"


def load_irs():
    df, dropped = load_generic(IRS_FILE)

    if not has_rows(df):
        return df, dropped

    d = IRS_DEFAULTS

    df["cov"] = pick(df, ["coverage"], label=IRS_FILE).astype(float)
    df["reps"] = pick(df, ["repetitions"], d["repetitions"]).astype(int)
    df["rep_gap"] = pick(
        df, ["timesteps_between_repetitions"], d["timesteps_between_repetitions"]
    ).astype(int)

    if "insecticide" not in df.columns:
        df["insecticide"] = ""

    rows = []

    for _, r in df.iterrows():
        products = [
            p.strip() for p in str(r["insecticide"]).split(";") if p.strip()
        ]

        if len(products) > 1:
            print(
                f"  WARNING {IRS_FILE}: {int(r['year'])} lists "
                f"{len(products)} products in one row; "
                f"splitting coverage {r['cov']:.3f} equally."
            )

        for prod in products or [""]:
            r2 = r.copy()
            r2["insecticide"] = prod
            r2["cov"] = r["cov"] / max(len(products), 1)
            rows.append(r2)

    df = pd.DataFrame(rows).reset_index(drop=True)

    sources = []

    for i, r in df.iterrows():
        if r["insecticide"]:
            params, source = product_params(r["insecticide"])
        else:
            params = d
            source = "IRS_DEFAULTS (no insecticide given)"

        overridden = []

        for internal, col in IRS_PARAM_COLUMNS.items():
            if col in df.columns and pd.notna(r[col]):
                df.at[i, internal] = float(r[col])

                if not np.isclose(float(r[col]), params[internal]):
                    overridden.append(col)
            else:
                df.at[i, internal] = float(params[internal])

        sources.append(
            source + (f"; CSV overrides {overridden}" if overridden else "")
        )

    df["param_source"] = sources

    df["iv_name"] = [
        "IRS_"
        + (
            normalise_product(p).replace(" + ", "_plus_").replace(" ", "_")
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


# =============================================================================
# SMC
# =============================================================================

def load_smc():
    df, dropped = load_generic(SMC_FILE)

    if not has_rows(df):
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


# =============================================================================
# IPTp
# =============================================================================

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


# =============================================================================
# VACCINE
# =============================================================================

def load_vaccine():
    df, dropped = load_generic(
        VACCINE_FILE,
        default_day_note=(
            "using 1 Jan of each year as the start of that year's rolling, "
            "birth-triggered D1 eligibility window."
        ),
    )

    if not has_rows(df):
        return df, dropped

    d = VACCINE_DEFAULTS

    df["cov"] = pick(df, ["coverage"], label=VACCINE_FILE).astype(float)
    df["dose"] = pick(df, ["dose"], "D1").astype(str).str.upper()

    if "target_age_months" in df.columns:
        df["target_age_months"] = df["target_age_months"].astype(float)
        df["amin"] = df["target_age_months"] / 12.0
        df["amax"] = VACCINE_CATCHUP_MAX_AGE_MONTHS / 12.0
    else:
        df["target_age_months"] = np.nan
        df["amin"] = pick(df, ["age_min", "agemin"], d["age_min"]).astype(float)
        df["amax"] = pick(df, ["age_max", "agemax"], d["age_max"]).astype(float)

    df["initial_effect"] = pick(
        df, ["initial_effect", "efficacy"], d["initial_effect"]
    ).astype(float)

    df["box_duration"] = pick(df, ["box_duration"], d["box_duration"]).astype(float)
    df["decay"] = pick(df, ["decay", "decay_time"], d["decay"]).astype(float)

    check_fraction(df["cov"], VACCINE_FILE)

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


# =============================================================================
# LOAD ALL SCHEDULES
# =============================================================================

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


# =============================================================================
# SCHEDULE DESCRIPTION
# =============================================================================

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
                    f"{cov_label}  (trigger {delay}d earlier)  "
                    f"block {r.blocking_initial:.2f}/{r.blocking_decay:g}d  "
                    f"kill {r.killing_initial:.2f}/{r.killing_decay:g}d  "
                    f"usage: {itn_usage_label()}"
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
                    f"decay {r.killing_decay:g}d  "
                    f"[{r.param_source}]"
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
                    f"(full continuation: 100% of prior-dose recipients)"
                )

                cov_label = f"cov n/a (CSV {r.cov:.3f} not applied)"

            print(
                f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4})  "
                f"{cov_label}  {extra}"
            )

        if sched["dropped"][key]:
            print(
                f"    ({sched['dropped'][key]} row(s) outside the "
                f"pickup window ignored)"
            )


# Filled in main block.
SCHEDULES = None


# =============================================================================
# CLIMATE
# =============================================================================

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


# =============================================================================
# SITE / VECTOR INPUT
# =============================================================================

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


# =============================================================================
# CONFIGURATION
# =============================================================================

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
    conf.add_species(config, manifest, vector_df["species"].tolist())

    for i in range(len(vector_df)):
        conf.set_species_param(
            config,
            species=vector_df["species"].iloc[i],
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

    # ---- Pregnancy model ----------------------------------------------------
    # NOTE: demographics implicits run after this function and can change
    # the birth model. build_demog() appends a final implicit function that
    # forces it back, and update_serialize_parameters() sets it again per
    # simulation (see the module docstring).
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE

    return config


# =============================================================================
# DEMOGRAPHICS
# =============================================================================

def add_pregnancy_property(demog):
    """Add the same Pregnancy individual property used in the burn-in."""
    if hasattr(demog, "AddIndividualPropertyAndHINT"):
        demog.AddIndividualPropertyAndHINT(
            Property="Pregnancy",
            Values=["NotPregnant", "IsPregnant"],
            InitialDistribution=[1.0, 0.0],
        )
    else:
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
    remove_stale_params(config)
    return config


def build_demog():
    site = _read_site_definition()

    demog = Demographics.from_template_node(
        lat=site["lat"],
        lon=site["lon"],
        pop=site["pop"],
        name=site["name"],
    )

    demog.SetBirthRate(CrudeRate(BIRTH_RATE))
    demog.SetMortalityRate(CrudeRate(BIRTH_RATE))
    demog.SetEquilibriumAgeDistFromBirthAndMortRates(
        CrudeRate(BIRTH_RATE),
        CrudeRate(BIRTH_RATE),
    )
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)

    add_pregnancy_property(demog)

    # demog.implicits is the documented list of config functions that emodpy
    # applies, in order, after set_param_fn. Append last so ours runs last.
    demog.implicits.append(_final_demographics_config)

    return demog


# =============================================================================
# CAMPAIGN HELPERS
# =============================================================================

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


# =============================================================================
# PREGNANCY TRACKING
# =============================================================================

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


# =============================================================================
# CASE MANAGEMENT
# =============================================================================

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


# =============================================================================
# MASS ITN
# =============================================================================

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


# =============================================================================
# ROUTINE ANC / CWC ITN
# =============================================================================

def add_anc_visit_event(campaign, start, listen, delay, event_name):
    """
    ANC stage 1: after `delay` days, women who triggered on Pregnant
    broadcast `event_name`. The ITN (stage 2) listens for that event, so
    its pregnancy restriction is evaluated at delivery time.
    """
    if not (
        hasattr(common, "triggered_campaign_event_with_optional_delay")
        and hasattr(common, "BroadcastEvent")
    ):
        raise ValueError(
            "ANC two-stage delivery needs emod_api.interventions.common."
            "triggered_campaign_event_with_optional_delay and BroadcastEvent; "
            "set ANC_CHECK_PREGNANT_AT_DELIVERY = False to use a single stage."
        )

    fn = common.triggered_campaign_event_with_optional_delay

    kwargs = dict(
        start_day=int(start),
        triggers=["Pregnant"],
        intervention=common.BroadcastEvent(campaign, event_name),
        delay={"Delay_Period_Constant": float(delay)},
        duration=int(listen),
        coverage=1.0,
        target_sex="Female",
    )

    if "blackout" in inspect.signature(fn).parameters:
        kwargs["blackout"] = False

    campaign.add(fn(campaign, **kwargs))


def add_routine_itns(campaign, df, channel):
    """
    Routine LLIN delivery.

        ANC: Pregnant -> wait 16 weeks -> ITN if still pregnant
         (two stages; see ANC_CHECK_PREGNANT_AT_DELIVERY).
        CWC: Births   -> wait 18 months -> deliver ITN.

    Each delivery year is split into ITN_SEASON_BATCH_DAYS-day batches (one
    triggered intervention each) so that the seasonal usage curve can be
    rotated to the batch's delivery date. See "ITN USAGE CURVES".
    """
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_triggered_usage_dependent_bednet,
    )

    if not has_rows(df):
        return

    # Argument names differ between emodpy_malaria versions. Dropping the
    # delay would deliver nets at the trigger instead of 16 weeks / 18
    # months later, so find the right names and fail loudly if absent.
    bednet_params = inspect.signature(
        add_triggered_usage_dependent_bednet
    ).parameters

    delay_key = (
        "delay_period_constant"
        if "delay_period_constant" in bednet_params
        else "triggered_campaign_delay"
    )

    newer_restrictions = "property_restrictions" in bednet_params

    two_stage = channel == "ANC" and ANC_CHECK_PREGNANT_AT_DELIVERY

    require_supported(
        add_triggered_usage_dependent_bednet,
        f"ITN-{channel}",
        ["trigger_condition_list", "listening_duration"]
        + ([] if two_stage else [delay_key]),
    )

    if channel == "ANC":
        trigger = ["Pregnant"]
        delay = ANC_DELAY_DAYS
        if not ANC_CHECK_PREGNANT_AT_DELIVERY:
            restriction = {}
        elif newer_restrictions:
            restriction = dict(property_restrictions=["Pregnancy:IsPregnant"])
        else:
            restriction = dict(
                ind_property_restrictions=[{"Pregnancy": "IsPregnant"}]
            )

        extra = dict(target_gender="Female", **restriction)

    elif channel == "CWC":
        trigger = ["Births"]
        delay = CWC_DELAY_DAYS
        extra = {}

    else:
        raise ValueError(f"Unknown routine ITN channel: {channel}")

    for r in df.itertuples():
        year = int(r.year)

        # CSV year = delivery year. Begin listening earlier by the required
        # trigger-to-delivery delay, one window per delivery batch.
        for batch, start, listen, mid_day in routine_delivery_batches(
            r.year_start_day, delay
        ):
            if two_stage:
                visit_event = anc_visit_event_name(year, batch)

                add_anc_visit_event(campaign, start, listen, delay, visit_event)

                # Stage 2: deliver when the visit event is heard (no delay).
                net_trigger = [visit_event]
                net_start = 1
                net_listen = -1
                net_delay = {}

            else:
                net_trigger = trigger
                net_start = start
                net_listen = listen
                net_delay = {delay_key: float(delay)}

            kwargs = dict(
                intervention_name=f"ITN_{channel}",
                start_day=int(net_start),
                demographic_coverage=float(r.cov),
                trigger_condition_list=net_trigger,
                listening_duration=int(net_listen),
                **net_delay,
                blocking_initial_effect=float(r.blocking_initial),
                blocking_decay_time_constant=float(r.blocking_decay),
                killing_initial_effect=float(r.killing_initial),
                killing_decay_time_constant=float(r.killing_decay),
                broadcast_event=f"Received_ITN_{channel}",
                **itn_usage_kwargs(channel, year, mid_day),
                **extra,
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
    CWC nets. Each batch is a scheduled event, so its delivery day (E) is
    known and the seasonal usage curve is rotated to it exactly.
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
                broadcast_event="Received_ITN_CWC",
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


# =============================================================================
# IRS
# =============================================================================

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
            killing_box_duration=float(r.killing_box),
            killing_decay_time_constant=float(r.killing_decay),
            repelling_initial_effect=float(r.repelling_initial),
            repelling_box_duration=float(r.repelling_box),
            repelling_decay_time_constant=float(r.repelling_decay),
        )


# =============================================================================
# SMC
# =============================================================================

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


# =============================================================================
# IPTp
# =============================================================================

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


# =============================================================================
# VACCINE
# =============================================================================

def add_vaccine(campaign, df):
    """
    RTS,S malaria vaccine implementation.

    GHS schedule:
        D1: 6 months
        D2: 7 months
        D3: 9 months
        D4: 24 months before 2023, 18 months from 2023 onward

    D1:
        Birth -> wait until target age -> D1

    D2+:
        Actual receipt of previous dose -> wait until next target age
        -> next dose

    CSV coverage controls D1.
    Continuation doses are conditional on receipt of the previous dose,
    so demographic coverage is 1.0 for D2+.
    """
    from emodpy_malaria.interventions.vaccine import add_triggered_vaccine

    if not has_rows(df):
        return

    for r in df.itertuples():
        year = int(r.year)

        common_kwargs = dict(
            intervention_name=f"Vaccine_{r.dose}_{year}",
            target_age_min=float(r.amin),
            target_age_max=float(r.amax),
            vaccine_type=VACCINE_DEFAULTS["vaccine_type"],
            vaccine_initial_effect=float(r.initial_effect),
            vaccine_box_duration=float(r.box_duration),
            vaccine_decay_time_constant=float(r.decay),
            broadcast_event=f"Received_Vaccine_{r.dose}_{year}",
        )

        # ---------------------------------------------------------------------
        # D1
        #
        # Birth -> wait until the GHS target age -> D1
        #
        # Example:
        #   D1 target age = 6 months
        #   Birth + 6 months -> D1
        # ---------------------------------------------------------------------
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
                demographic_coverage=float(r.cov),
                **common_kwargs,
            )

        # D2+
        #
        # Previous dose RECEIVED -> wait until next target age -> next dose.
        #
        # D2:
        #   D1 received at 6 months
        #   + 1 month
        #   = D2 at 7 months
        #
        # D3:
        #   D2 received at 7 months
        #   + 2 months
        #   = D3 at 9 months
        #
        # D4:
        #   2022 and earlier: D3 + 15 months = 24 months
        #   2023 onward:      D3 + 9 months  = 18 months
        #
        # Coverage = 1 because receipt of the previous dose is already
        # the gate for entering the continuation chain.
        else:
            call_supported(
                add_triggered_vaccine,
                campaign,
                "Vaccine",
                trigger_condition_list=[
                    f"Received_Vaccine_{r.prev_dose}_{year}"
                ],
                delay_period_constant=float(r.delay_days),
                demographic_coverage=1.0,
                start_day=1,
                listening_duration=-1,
                **common_kwargs,
            )


# =============================================================================
# HABITAT SCALING
# =============================================================================

def scale_habitat_from_mix(mix):
    """Apply burn-in habitat mix to vector species fractions."""
    scales = {"NodeID": vector_df["node_id"].unique()}

    for sp in vector_df["species"].unique():
        frac = vector_df.loc[vector_df["species"] == sp, "fraction"]

        scales[f"CONSTANT.{sp}"] = [mix[0] * f for f in frac]
        scales[f"TEMPORARY_RAINFALL.{sp}"] = [mix[1] * f for f in frac]
        scales[f"WATER_VEGETATION.{sp}"] = [mix[2] * f for f in frac]

    return pd.DataFrame(scales)


# =============================================================================
# CAMPAIGN BUILDER
# =============================================================================

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

    # Routine ANC ITN
    if has_rows(s["itn_anc"]):
        add_routine_itns(camp, s["itn_anc"], "ANC")

    # Routine CWC ITN
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

    # Vaccine
    if has_rows(s["vaccine"]):
        add_vaccine(camp, s["vaccine"])

    # Clean property restrictions
    fix_property_restrictions(camp.campaign_dict)

    return camp


# =============================================================================
# SERIALIZATION SWEEP
# =============================================================================

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
    # so this is the final word on Birth_Rate_Dependence.
    simulation.task.set_parameter("Birth_Rate_Dependence", BIRTH_RATE_DEPENDENCE)

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


# =============================================================================
# REPORTS
# =============================================================================

def recorder_events():
    """Events written to ReportEventRecorder.csv (also printed in the dry run)."""
    events = ["Received_Treatment", "Received_Severe_Treatment"]

    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_triggered_usage_dependent_bednet,
    )

    if "broadcast_event" in inspect.signature(
        add_triggered_usage_dependent_bednet
    ).parameters:
        if has_rows(SCHEDULES["itn_anc"]):
            events.append("Received_ITN_ANC")

        if has_rows(SCHEDULES["itn_cwc"]):
            events.append("Received_ITN_CWC")

    else:
        # This emodpy_malaria version cannot broadcast a named event for
        # routine nets. Bednets always broadcast this built-in event instead
        # (all channels: mass, ANC, CWC).
        events.append("Bednet_Got_New_One")

    # ANC stage-1 events (the visit, before the pregnancy check at delivery).
    if ANC_CHECK_PREGNANT_AT_DELIVERY and has_rows(SCHEDULES["itn_anc"]):
        for r in SCHEDULES["itn_anc"].itertuples():
            events.extend(
                anc_visit_event_name(r.year, batch)
                for batch, _, _, _ in routine_delivery_batches(
                    r.year_start_day, ANC_DELAY_DAYS
                )
            )

    if RECORD_TRIGGER_EVENTS:
        events.extend(["Pregnant", "GaveBirth", "Births"])

    if has_rows(SCHEDULES["smc"]):
        events.append("Received_SMC")

    if has_rows(SCHEDULES["iptp"]):
        events.extend(
            f"Received_IPTp_D{n}_{int(y)}"
            for y, doses in zip(
                SCHEDULES["iptp"]["year"], SCHEDULES["iptp"]["doses"]
            )
            for n, _, _ in doses
        )

    if has_rows(SCHEDULES["vaccine"]):
        v = SCHEDULES["vaccine"]

        events.extend(
            sorted(
                f"Received_Vaccine_{d}_{int(y)}"
                for d, y in zip(v["dose"], v["year"])
            )
        )

    return events


def add_reports(task):
    # ---- Annual summary reports with monthly reporting interval -------------
    for i in range(PICKUP_YEARS):
        call_supported(
            add_malaria_summary_report,
            task,
            "Summary report",
            manifest=manifest,
            start_day=i * DAYS_PER_YEAR + 1,
            end_day=(i + 1) * DAYS_PER_YEAR,
            reporting_interval=30,
            # Upper edges in years. 0.5 and 5 make the 0.5-5 y bin equal to
            # 6-59 months, matching the prevalence surveys.
            age_bins=[0.25, 0.5, 5, 15, 25, 100],
            max_number_reports=13,
            parasite_detection_threshold=PARASITE_DETECTION_THRESHOLD,
            pretty_format=True,
            filename_suffix=f"Monthly_U5_{PICKUP_START_YEAR + i}",
        )

    # ---- Event recorder -----------------------------------------------------
    events = recorder_events()

    add_event_recorder(
        task,
        event_list=events,
        start_day=1,
        end_day=PICKUP_DAYS,
        node_ids=[NODE_ID],
        min_age_years=0,
        max_age_years=115,
    )


# =============================================================================
# COMMAND LINE
# =============================================================================

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

    return parser.parse_args()


# =============================================================================
# SUBMISSION
# =============================================================================

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

    print("\nSubmitting under experiment name:")
    print(f"  {EXPERIMENT_NAME}")

    experiment = Experiment.from_builder(builder, task, name=EXPERIMENT_NAME)

    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")
    print(f"Name: {EXPERIMENT_NAME}")
    print(f"Simulations: {n_sims}")

    return experiment


# =============================================================================
# GENERAL SIMULATION
# =============================================================================

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
    print(f"Experiment name: {EXPERIMENT_NAME}")

    return submit_pickup(platform, burnin_df, climate_dir)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    # Load and print all intervention schedules.
    SCHEDULES = load_all_schedules()

    # ITN usage curves (age and season); stops here if a file is missing.
    if USE_ITN_USAGE_CURVES:
        ITN_USAGE = load_itn_usage()
        validate_itn_usage(SCHEDULES)

    describe_schedules(SCHEDULES)

    # Pregnancy model check.
    if "PREGNANC" not in BIRTH_RATE_DEPENDENCE:
        msg = (
            f"BIRTH_RATE_DEPENDENCE = {BIRTH_RATE_DEPENDENCE}: "
            f"'Pregnant'  'GaveBirth' events never fire, so ANC nets "
            f"would deliver nothing. It must match the burn-in "
            f"(INDIVIDUAL_PREGNANCIES)."
        )

        if args.dry_run:
            print(f"\n  WARNING: {msg}")
        else:
            sys.exit(msg)

    # Climate.
    climate_dir = resolve_climate_dir()
    validate_climate(climate_dir)

    # Dry run.
    if args.dry_run:
        describe_itn_usage(SCHEDULES)

        print("\nRoutine ITN timing:")
        print(f"  ANC: Pregnant  {ANC_DELAY_DAYS} days  ITN")
        print(f"  CWC: Births  {CWC_DELAY_DAYS} days  ITN")

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