"""
Upper West Region EMOD malaria model -- SCENARIOS (IRS vs IG2 replacement)

AIM 2: estimate the epidemiological impact and the IG2 coverage that must be
distributed (and the effective use it gives) to replace IRS in the Upper West
Region of Ghana.

    1963 ---- 2012 | 2013 ---- 2025 | 2026 ---------- 2032
      burn-in      |    pickup      |  scenarios (this script)

               serialized state (state-04745.dtk) read here

Usage
-----
1. Preview scenarios, write two campaign previews, submit nothing:

       python run_scenarios_UWR.py --dry-run

2. Submit ONE pickup state x ALL scenarios (9 simulations with the defaults:
   2 IRS coverage arms + 7 IG2 arms):

       python run_scenarios_UWR.py --test-one

3. Full run (every selected pickup state x every scenario):

       python run_scenarios_UWR.py

Scenarios (same weather, same states, same seeds)
--------------------------------------------------------------------------------
  2026 (common path): every arm runs IRS as in 2025 (the REFERENCE_YEAR rows
                 are repeated unless the IRS CSV has its own 2026 rows).
  From 1 Jan IRS_SWITCH_YEAR (2027) the arms diverge:
  IRS_<c>        IRS continues at TOTAL annual coverage c (IRS_COVERAGES,
                 default 0.95 and 1.00); product settings come from the IRS
                 CSV rows, only the coverage is replaced. No IG2 nets.
  IG2_<c>_<key>  IRS removed; IG2 nets distributed at coverage c (0.5 ... 1.0),
                 issued on IG2_CAMPAIGN_YEARS (2027, then every 3 years)
  BAU            Business as usual: IRS continues at the CSV coverage and all
                 routine programmes continue (INCLUDE_IRS_BASELINE_ARM).
                 This is the reference arm for every paired comparison.
  NO_IRS         IRS stopped, no replacement (INCLUDE_NO_IRS_ARM).

  Case management (CM_BOOSTS): every arm above is run twice,
    <arm>          current case management (CSV coverage)
    <arm>_CM5      effective CM coverage (coverage x seek) + 5 percentage
                   points from 1 Jan CM_BOOST_START_YEAR (e.g. 0.60 -> 0.65),
                   capped at CM_MAX_COVERAGE.
  Default design: (BAU, NO_IRS, IRS_60/80/90, IG2_60/80/90) x 2 CM = 16 arms.
  Plots for the four policy questions: plot_scenarios_publication.py.
  Because 2026 is identical, the arms should give the same 2026 results; a
  difference before 1 Jan 2027 means something in the setup is not paired.

  NOTE: IRS coverage is the share of eligible structures sprayed; IG2
  coverage is the share of the population receiving a net. They are
  comparable scenario levels, NOT equivalent measures of intervention
  coverage. Report them as separate curves.

What "coverage" and "effective use" mean here
---------------------------------------------
c (the number in the IG2 scenario name) is the demographic coverage
DISTRIBUTED in each IG2 campaign. IG2_USE_CURVES is True, so IG2 nets carry
the MASS age and seasonal usage curves, and the share of people actually
using an IG2 net is

    effective use = c x net retention x age multiplier x season multiplier

  * net retention: Weibull discard (k = 3, lambda = 1095 d, mean about 980 d);
    the mean fraction still held over a 3-year cycle is printed by --dry-run
    (about 0.81). After a re-issue, earlier nets still held by people who were
    not re-issued add to this (a new ITN_IG2 replaces the old one).
  * age multiplier: population-weighted mean of the MASS age-usage curve.
  * season multiplier: MASS seasonal-usage curve at the calendar day.

analyze_scenarios_UWR.py computes the effective use for every tested c and for
the minimum coverage that maintains or reduces burden. Optional net-use
counters (ADD_NET_USE_COUNTER) record Bednet_Using events as a diagnostic; they
count each net separately, so they overstate the share of people using a net.

Everything marked "MUST EQUAL THE PICKUP" must be identical to
Site_Setup_Final.py (the script that ran the pickup), because config settings
are NOT stored in the serialized state. The continuing programmes (case
management, ANC/CWC nets, SMC, IPTp, RTS,S vaccine, IRS) are built by calling
Site_Setup_Final.py's own functions (PICKUP_SCRIPT), so they are identical to
the pickup by construction.

Pickup custom events (IMPORTANT)
--------------------------------
The serialized state holds delayed interventions that are still waiting to
broadcast pickup-era events (ANC_Visit_2025, CWC_Visit_2025, the pickup's
MDA_Now_<id> events, Received_* events ...). EMOD refuses to load the state
unless EVERY one of those names is in the scenario config's
Custom_Individual_Events ("Unknown event = 'ANC_Visit_2025'").

emodpy rebuilds Custom_Individual_Events from the events the CAMPAIGN
registers when the campaign is built, which overwrites anything merged in
set_param_fn. So this script:

  1. reads Custom_Individual_Events from the pickup config.json files,
  2. registers every pickup event that the scenario campaign does not already
     use through a dormant, never-firing BroadcastEvent campaign event
     (register_pickup_events), so emodpy puts it in the config itself,
  3. checks the built campaign before submission (verify_pickup_events),
  4. as a second safety layer, merges the pickup events into the config again
     after emodpy's pre_creation step (PATCH_CONFIG_EVENTS).

Check afterwards: the generated config.json of any scenario simulation must
list ANC_Visit_2025, CWC_Visit_2025 and the pickup's MDA_Now_* ids in
Custom_Individual_Events.

MDA_Now_<id> names of the scenario's own drug campaigns
-------------------------------------------------------
emodpy_malaria names the internal event of each drug campaign
MDA_Now_<random 1..10000>. This script replaces the random draw by a counter
that skips the pickup's MDA_Now ids. The counter is reset at the start of EVERY
campaign build (reset_mda_ids): ids only need to be unique inside one campaign,
and without the reset the 10,000 ids are used up after a few hundred builds
("Exhausted IDs 1..10000"). Every arm therefore gets identical ids.

Calendar
--------
Flat 365-day years. Projection day 1 = 1 Jan PROJECTION_START_YEAR. Weather
day 1 must be that date and the weather must cover the whole projection.

Continuing the other programmes
-------------------------------
Case management, routine ANC / CWC nets, IPTp, SMC and the vaccine are held
the same in every arm. Each input CSV needs rows for the projection years; any
projection year missing from a file is filled by repeating REFERENCE_YEAR's
rows ("rolled forward", reported by --dry-run).

Known start-up gap (identical in every arm, so the paired difference is
barely affected): women already pregnant on day 1 get no ANC net / IPTp dose 1
for their first ~16 weeks, because their Pregnant event happened before the
state was written. Children already born on day 1 ARE served by the CWC
catch-up. RTS,S (built as in Site_Setup_Final.py) listens for Births from day 1,
so children born in the second half of 2025 (D1 due in 2026) miss the 2026 D1
round; this is the same start-up behaviour the pickup had in 2013. Use ANALYSIS_START_YEAR in the analysis script to skip warm-up.
"""

# UWR scenario script, organized using the configuration / builder / execution
# structure of the example script. The original scenario definitions, CSV
# schedules, serialized pickup-state pairing, and IRS-vs-IG2 design are retained.
# Drug-campaign internal event IDs are allocated uniquely within each campaign
# to prevent MDA_Now event-name collisions between SMC/IPTp campaigns.


import argparse
import inspect
import json
import os
import pathlib
import random
import re
import sys
import types
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

# Weather for the projection: the SAME files the pickup was run and calibrated
# with (ERA5 2013-2023, no-leap, 2024-2025 filled; 4745 days), so the climate
# is maintained from calibration into the scenarios. Start_Time = 0, so
# projection day 1 (1 Jan 2026) reads file day 1: 2026-2032 replays the
# observed 2013-2019 weather, identically in every arm.
CLIMATE_ROOT = os.path.join(INPUT_DIR, "upper_west", "climate", "2013001-2023365")
CLIMATE_SUBDIR = "_noleap_meanT30.6_fill2025"
CLIMATE_FILE_TEMPLATE = "dtk_15arcmin_{var}_daily_revised.bin"   # as in the pickup config

VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"

# ---- Input files (same files as the pickup; add projection-year rows) -------
CM_FILE = "intervention_CMBN.csv"
ITN_ANC = "interventions_ITN_ANC.csv"
ITN_CWC = "interventions_ITN_CWC.csv"
IRS_FILE = "intervention_IRS.csv"
SMC_FILE = "intervention_SMC.csv"
IPTP_FILE = "interventions_IPTp.csv"
VACCINE_FILE = "intervention_malariaVaccine.csv"

# Usage curves of the EXISTING routine nets (ANC / CWC channels).
USE_ITN_USAGE_CURVES = True
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"

EXPERIMENT_NAME = "Pickup13yrs_Scenario_Final"

# ---- Continuing programmes are built by the PICKUP script itself -------------
# The pickup experiment (PICKUP_EXPERIMENT_ID) was run with Site_Setup_Final.py.
# Case management, ANC/CWC nets, SMC, IPTp, RTS,S vaccine and IRS are built
# with that script's own functions (same EMOD classes, coverage conversions and
# event chains), its calendar moved to 2026-2032. Its input files are read
# from INPUT_DIR; projection years missing from a file repeat REFERENCE_YEAR.
PICKUP_SCRIPT = "Site_Setup_Final"
PROJECTION_INPUT_DIR = os.path.join(INPUT_DIR, "_projection_inputs")   # generated

# Finished PICKUP experiment (it wrote state-04745.dtk).
PICKUP_EXPERIMENT_ID = "48201665-bccb-4fb0-acc7-b398c2569ab5"
PICKUP_FILTER = {}

# After calibration: CSV of the parameter sets to carry forward, with columns
#   x_Temporary_Larval_Habitat, scale_constant, scale_temp_rain, scale_water_veg
# (one row per accepted set). None = use every pickup simulation.
PARAMETER_SET_FILE = None

HABITAT_TAGS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]

WRITE_END_STATE = False           # serialize at the end of the projection
PARASITE_DETECTION_THRESHOLD = 40


# =============================================================================
# PICKUP CUSTOM EVENTS (needed to deserialize the pickup state)
# =============================================================================

# Filled by submit_scenarios() from the pickup config.json files.
PICKUP_CUSTOM_EVENTS = []

# Pickup event families that must be known to the scenario run. Everything
# else in the pickup list is an emodpy / EMOD default that the scenario config
# already contains.
PICKUP_EVENT_FAMILY_RE = re.compile(r"^(ANC_Visit_|CWC_Visit_|MDA_Now_|Received_)")

# Second safety layer: merge the pickup events into the config again after
# emodpy's pre_creation step has rebuilt Custom_Individual_Events.
PATCH_CONFIG_EVENTS = True

# Dormant registration events start long after the simulation ends, so they
# can never fire.
DORMANT_EVENT_START_DAY = 100000


# =============================================================================
# SCENARIO SETTINGS
# =============================================================================

# IG2 demographic coverage DISTRIBUTED in each campaign (c). Effective use is
# c x net retention x age multiplier x season multiplier (see the docstring).
# WHO 60,80 and 90% coverage targets
IG2_COVERAGES = [0.60, 0.80,0.90]
# Add e.g. [0.35, 0.45, 0.55, 0.65, 0.75, 0.85] to pin down the threshold.
EXTRA_IG2_COVERAGES = []

# IRS continuation at a fixed TOTAL annual coverage from IRS_SWITCH_YEAR on.
# Product settings (Clothianidin, kill decay 180 d, 1 round on 4 April) come
# from the IRS CSV rows; only the coverage is replaced. No IG2 nets in these
# arms. IRS coverage (share of eligible structures sprayed) and IG2 coverage
# (share of the population receiving a net) are comparable scenario levels,
# NOT equivalent measures of intervention coverage.
IRS_COVERAGES = [0.60, 0.80,0.90]

# Optional reference arms, OFF for this run (set True to add them back).
INCLUDE_IRS_BASELINE_ARM = True    # BAU: IRS at the CSV coverage (reference arm)
INCLUDE_NO_IRS_ARM = True          # NO_IRS: IRS stopped, no IG2 replacement
BAU_ARM_NAME = "BAU"               # MUST MATCH BASELINE in analyze_scenarios_UWR.py

# Case-management variants. Every arm is run once per entry. 0.0 = current CM;
# 0.05 = effective CM coverage (coverage x seek) raised by 5 percentage points
# from 1 Jan CM_BOOST_START_YEAR (2026 stays a common path in every arm).
CM_BOOSTS = [0.0, 0.05]
CM_BOOST_START_YEAR = 2027
CM_BOOST_SEVERE = False            # True: also raise severe-case treatment
CM_MAX_COVERAGE = 1.0

# IG2 parameters from the study. discard_k / discard_l and IG2_CAMPAIGN_DOY
# must also be set in analyze_scenarios_UWR.py (effective-use calculation).
IG2 = dict(
    blocking_initial=0.932,
    blocking_decay=730.0,
    killing_initial=0.987,
    discard_k=3.0,
    discard_l=1095.0,
)

# EMOD takes ONE killing decay per net: run one as the main analysis and the
# other as a sensitivity run (put both in IG2_KILL_RUN to run both).
IG2_KILL_DECAY = {"pyr": 1420.0, "cfp": 1580.0}
IG2_KILL_RUN = ["cfp"]

# COMMON PATH: until IRS_SWITCH_YEAR every arm (including the IG2 arms) runs
# IRS exactly as in the reference year (2026 repeats 2025 unless the IRS CSV
# has its own 2026 rows). From 1 Jan IRS_SWITCH_YEAR the arms diverge:
# IRS_<c> arms keep IRS at coverage c, IG2 arms replace IRS with IG2
# (and the optional NO_IRS arm drops IRS).
IRS_SWITCH_YEAR = 2027

# Mass IG2 distributions (on or after IRS_SWITCH_YEAR). Mean net life is
# ~980 d, so re-issue every 3 years. Two full cycles from 2027 need
# PROJECTION_END_YEAR = 2032.
IG2_CAMPAIGN_YEARS = [2027, 2030]
IG2_CAMPAIGN_DOY = 91         # day of year of the distribution (91 = 1 Apr)
# True (used here): IG2 gets the MASS age and seasonal usage curves, so
# effective use = c x retention x age multiplier x season multiplier.
# False: usage multiplier = 1, so effective use at distribution = c.
# Independent of USE_ITN_USAGE_CURVES (which only concerns routine ANC/CWC nets).
IG2_USE_CURVES = False

# Count net-use events per day (all nets). Diagnostic only: each net is counted
# separately, so this overstates the share of people using a net.
ADD_NET_USE_COUNTER = True
NET_USE_EVENTS = ["Bednet_Using"]


# =============================================================================
# CALENDAR
# =============================================================================

DAYS_PER_YEAR = 365
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_DAYS = (PICKUP_END_YEAR - PICKUP_START_YEAR + 1) * DAYS_PER_YEAR   # 4745

PROJECTION_START_YEAR = PICKUP_END_YEAR + 1
PROJECTION_END_YEAR = 2032
PROJ_START = PROJECTION_START_YEAR
PROJ_END = PROJECTION_END_YEAR
PROJ_YEARS = PROJ_END - PROJ_START + 1
PROJ_DAYS = PROJ_YEARS * DAYS_PER_YEAR

# Projection years missing from an input CSV repeat this year's rows.
REFERENCE_YEAR = PICKUP_END_YEAR

# Rows tagged with any of these phases are read.
PHASES_ACCEPTED = ("pickup", "projection", "scenario")

MONTH_OFFSET = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# =============================================================================
# DEMOGRAPHICS  --  MUST EQUAL THE PICKUP
# =============================================================================

BIRTH_RATE_DEPENDENCE = "INDIVIDUAL_PREGNANCIES"
CRUDE_DEATH_RATE = 10.9
PREG_RATE_PER_MOTHER_DAY = 3.62e-4
X_BIRTH = 1.0
MAX_PREG_RATE_PER_MOTHER_DAY = 1.0 / 280.0
STALE_CONFIG_PARAMS = ("Disable_IP_Whitelist",)

# MUST EQUAL THE PICKUP (it used True).
APPLY_VECTOR_BEHAVIOUR = True


# =============================================================================
# REPORT SETTINGS
# =============================================================================

REPORT_INTERVAL = 30
REPORT_MAX_NUMBER = 13
# Upper edges in years; 0.5-5 y bin = 6-59 months.
REPORT_AGE_BINS = [0.25, 0.5, 5, 15, 25, 100]

EVENT_LIST = [
    "Births", "DiseaseDeaths", "NonDiseaseDeaths",
    "NewInfectionEvent", "NewClinicalCase",
    "Received_Treatment", "Received_Severe_Treatment",
    "Pregnant", "GaveBirth",
]
EVENT_MIN_AGE_YEARS = 0
EVENT_MAX_AGE_YEARS = 115           # same as the pickup (PICKUP_EVENT_MAX_AGE)


# =============================================================================
# INTERVENTION DEFAULTS
# =============================================================================

CM_DRUGS = {
    "AL": ["Artemether", "Lumefantrine"],
    "ASAQ": ["Artesunate", "Amodiaquine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}
SEVERE_TRIGGERS = ["NewSevereCase"]

IRS_PARAM_COLUMNS = {
    "killing_initial": "killing_initial_effect",
    "killing_box": "killing_box_duration",
    "killing_decay": "killing_decay_time_constant",
    "repelling_initial": "repelling_initial_effect",
    "repelling_box": "repelling_box_duration",
    "repelling_decay": "repelling_decay_time_constant",
}

SMC_DRUG_ALIASES = {"SPAQ": "SPA", "SP+AQ": "SPA", "SP-AQ": "SPA"}
SMC_NONCSV_DEFAULTS = dict(age_min=0.25, age_max=5.0, drug_code="SPA")
VACCINE_NONCSV_DEFAULTS = dict(
    initial_effect=0.8, box_duration=0, decay=592,
    vaccine_type="AcquisitionBlocking",
)

# Routine ITN timing.
ANC_DELAY_DAYS = 16 * 7
ANC_CHECK_PREGNANT_AT_DELIVERY = True
CWC_AGE_MONTHS = 18
CWC_DELAY_DAYS = int(round(CWC_AGE_MONTHS * DAYS_PER_MONTH))

IPTP_DEFAULTS = dict(drug_code="SP", trigger="Pregnant",
                     delay_days=16 * 7, gap_days=30)
IPTP_COLUMN_RE = re.compile(r"^iptp(\d+)_coverage$")

VACCINE_CATCHUP_MAX_AGE_MONTHS = 59.0
VACCINE_EXPECTED_DOSES = ("D1", "D2", "D3", "D4")
VACCINE_USE_AGE_FLOOR = True
VACCINE_ISSUES = {}

ITN_USAGE_CHANNELS = ("MASS", "ANC", "CWC")
USAGE_VALUE_COLUMNS = ["age_usage", "season_usage", "usage", "value", "use",
                       "usage_prob", "probability", "prob"]
USAGE_AGE_COLUMNS = ["age", "age_years", "age_yr"]
USAGE_DAY_COLUMNS = ["day_of_year", "doy", "day"]
USAGE_SEASON_TIME_COLUMNS = ["season_time"]
MAX_USAGE_AGE_YEARS = 125

SCHEDULES = None
SCENARIOS = None
ITN_USAGE = None
_WARNED = set()
_USAGE_WARNED = set()


# =============================================================================
# CALENDAR HELPERS
# =============================================================================

def month_number(value):
    text = str(value).strip().lower()
    if text.replace(".", "").isdigit():
        return int(float(text))
    if text[:3] in MONTH_NAMES:
        return MONTH_NAMES.index(text[:3]) + 1
    raise ValueError(f"Unrecognised month '{value}'.")


def month_of_doy(doy):
    return int(np.searchsorted(MONTH_OFFSET, int(doy) - 1, side="right"))


def fmt_day(day):
    day = int(day)
    year = PROJ_START + (day - 1) // DAYS_PER_YEAR
    doy = (day - 1) % DAYS_PER_YEAR + 1
    month = month_of_doy(doy)
    return f"{year}-{month:02d}-{doy - MONTH_OFFSET[month - 1]:02d}"


def doy_of_day(day):
    return (int(day) - 1) % DAYS_PER_YEAR + 1


def month_mid_doy(month):
    m = month_number(month)
    return float(MONTH_OFFSET[m - 1]) + (MONTH_DAYS[m - 1] + 1) / 2.0


# =============================================================================
# CSV HELPERS
# =============================================================================

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
    """Read a REQUIRED column. Nothing is defaulted."""
    col = next((n for n in names if n in df.columns), None)
    if col is None:
        raise ValueError(f"{label}: needs a column named one of {list(names)}; "
                         f"found {list(df.columns)}.")
    series = df[col]
    if series.isna().any():
        years = (sorted({int(y) for y in df.loc[series.isna(), "year"]})
                 if "year" in df.columns else [])
        raise ValueError(f"{label}: column `{col}` is blank for year(s) {years}.")
    try:
        return series.astype(kind)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: column `{col}` must be {kind.__name__}.")


def parameter_or_default(df, names, default, label, kind=float):
    if any(n in df.columns for n in names):
        return need(df, names, label, kind)
    return pd.Series([default] * len(df), index=df.index, dtype=object).astype(kind)


def keep_node(df):
    if "node_id" in df.columns:
        df = df[df["node_id"].astype(int) == NODE_ID]
    return df


def keep_phase(df):
    if "phase" in df.columns:
        ph = df["phase"].astype(str).str.lower()
        df = df[ph.isin(PHASES_ACCEPTED) | df["phase"].isna()]
    return df


def check_fraction(series, label):
    if ((series < 0) | (series > 1)).any():
        raise ValueError(f"{label}: coverage values must be between 0 and 1.")


def has_rows(df):
    return df is not None and not df.empty


def load_year_table(filename, channel=None, carry_back=0):
    """
    Read a per-year input CSV for the projection.

    Keeps rows for years PROJ_START - carry_back .. PROJ_END. Every year
    PROJ_START..PROJ_END that has no rows is filled by repeating the rows of
    REFERENCE_YEAR (column `rolled_forward` = True). Returns None if the file
    does not exist. `start_day` is reduced to day of year.
    """
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        return None

    df = keep_phase(keep_node(read_clean_csv(path)))

    if channel and "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == channel]

    if "year" not in df.columns:
        raise ValueError(f"{filename}: needs a `year` column; found {list(df.columns)}")

    df = df.copy()
    df["year"] = df["year"].astype(int)

    if "start_day" in df.columns:
        df["start_day"] = (df["start_day"].astype(int) - 1) % DAYS_PER_YEAR + 1

    df["rolled_forward"] = False
    reference = df[df["year"] == REFERENCE_YEAR]
    pieces = [df[df["year"].between(PROJ_START - carry_back, PROJ_END)]]

    for y in range(PROJ_START, PROJ_END + 1):
        if (df["year"] == y).any():
            continue
        if reference.empty:
            raise ValueError(
                f"{filename}: no rows for {y} and none for REFERENCE_YEAR "
                f"{REFERENCE_YEAR} to repeat. Add {y} rows to the file."
            )
        r = reference.copy()
        r["year"] = y
        r["rolled_forward"] = True
        pieces.append(r)

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values("year", kind="stable").reset_index(drop=True)


def add_day(df, label, drop_outside=True):
    """Add `doy` and `proj_day` (day 1 = 1 Jan PROJ_START); drop rows outside."""
    df = df.copy()
    year = df["year"].astype(int)

    if "day_of_year" in df.columns:
        doy = df["day_of_year"].astype(int)
    elif "start_day" in df.columns:
        doy = df["start_day"].astype(int)
    elif "month" in df.columns:
        doy = df["month"].map(month_number).map(lambda m: MONTH_OFFSET[m - 1] + 1)
    else:
        doy = pd.Series(1, index=df.index)
        print(f"  WARNING {label}: no day_of_year/start_day/month column; "
              f"using 1 Jan of each year.")

    df["doy"] = doy.astype(int)
    df["proj_day"] = (year - PROJ_START) * DAYS_PER_YEAR + df["doy"]

    if not drop_outside:
        return df, 0

    inside = (df["proj_day"] >= 1) & (df["proj_day"] <= PROJ_DAYS)
    return df[inside].copy(), int((~inside).sum())


# =============================================================================
# INTERVENTION LOADING
# =============================================================================

def load_case_management():
    df = load_year_table(CM_FILE)
    if df is None:
        raise FileNotFoundError(f"{CM_FILE} not found in {INPUT_DIR}")

    required = ["duration", "trigger", "age_min", "age_max",
                "coverage", "seek", "rate", "drug", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{CM_FILE} is missing columns: {missing}")

    df["drug"] = df["drug"].astype(str).str.upper()
    df, dropped = add_day(df, CM_FILE)

    unknown = sorted(set(df["drug"]) - set(CM_DRUGS))
    if unknown:
        raise ValueError(f"{CM_FILE}: drug code(s) {unknown} not in CM_DRUGS.")

    df["requested_duration"] = df["duration"].astype(int)
    df["duration"] = np.minimum(df["requested_duration"],
                                PROJ_DAYS - df["proj_day"] + 1)
    df["effective_coverage"] = df["coverage"].astype(float) * df["seek"].astype(float)
    check_fraction(df["effective_coverage"], f"{CM_FILE} (coverage x seek)")
    df["is_severe"] = df["trigger"].isin(SEVERE_TRIGGERS)
    return df, dropped


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


def load_routine_itn(filename, channel):
    """Routine ANC/CWC nets: ONE row per delivery year (year is authoritative)."""
    df = load_year_table(filename, channel=channel)
    if df is None:
        return None, 0
    if df["year"].duplicated().any():
        raise ValueError(f"{filename}: more than one row per year for {channel}.")
    df["year_start_day"] = (df["year"] - PROJ_START) * DAYS_PER_YEAR + 1
    if len(df):
        df = fill_itn_columns(df, filename)
    return df, 0


def normalise_product(name):
    parts = [" ".join(p.replace("-", " ").replace("_", " ").lower().split())
             for p in str(name).split("+")]
    return " + ".join(sorted(p for p in parts if p))


def load_irs():
    df = load_year_table(IRS_FILE)
    if df is None:
        return None, 0

    df, dropped = add_day(df, IRS_FILE)
    if df.empty:
        return df, dropped

    df["cov"] = need(df, ["coverage"], IRS_FILE)
    df["reps"] = need(df, ["repetitions"], IRS_FILE, int)
    df["rep_gap"] = need(df, ["timesteps_between_repetitions"], IRS_FILE, int)

    for internal, col in IRS_PARAM_COLUMNS.items():
        df[internal] = need(df, [col], IRS_FILE)

    check_fraction(df["cov"], IRS_FILE)

    year_total = df.groupby("year")["cov"].sum()
    if (year_total > 1).any():
        raise ValueError(f"{IRS_FILE}: coverages add up to more than 1 in "
                         f"year(s) {list(year_total[year_total > 1].index)}.")

    df["iv_name"] = [f"IRS_{int(y)}_{k}" for k, y in
                     zip(df.groupby("year").cumcount(), df["year"])]
    return df, dropped


def load_smc():
    df = load_year_table(SMC_FILE)
    if df is None:
        return None, 0

    df, dropped = add_day(df, SMC_FILE)
    if df.empty:
        return df, dropped

    df["cov"] = need(df, ["coverage"], SMC_FILE)
    df["amin"] = parameter_or_default(df, ["age_min", "agemin"],
                                      SMC_NONCSV_DEFAULTS["age_min"], SMC_FILE)
    df["amax"] = parameter_or_default(df, ["age_max", "agemax"],
                                      SMC_NONCSV_DEFAULTS["age_max"], SMC_FILE)
    codes = parameter_or_default(df, ["drug_code", "drug"],
                                 SMC_NONCSV_DEFAULTS["drug_code"], SMC_FILE, str).str.upper()
    df["drug_code"] = codes.replace(SMC_DRUG_ALIASES)
    df["round"] = pick(df, ["round"], 0).astype(int)
    check_fraction(df["cov"], SMC_FILE)
    return df, dropped


def iptp_dose_columns(df):
    found = []
    for c in df.columns:
        m = IPTP_COLUMN_RE.match(c)
        if m:
            found.append((int(m.group(1)), c))
    return sorted(found)


def iptp_dose_plan(row, dose_cols, year):
    """Cumulative dose coverage -> [(dose, cumulative, conditional)]."""
    plan = []
    for n, col in dose_cols:
        cum = row[col]
        if pd.isna(cum):
            print(f"  WARNING {IPTP_FILE}: {year} {col} is blank; "
                  f"dose {n} onward not delivered that year.")
            break
        cum = float(cum)
        if not plan:
            cond = cum
        else:
            prev = plan[-1][1]
            if prev <= 0:
                cond = 0.0
            else:
                cond = cum / prev
                if cond > 1:
                    print(f"  WARNING {IPTP_FILE}: {year} {col} ({cum:.4f}) exceeds "
                          f"previous dose ({prev:.4f}); capped at 1.")
                    cond = 1.0
        plan.append((n, cum, cond))
    return plan


def iptp_trigger_window(year_start_day):
    window_start = int(year_start_day) - IPTP_DEFAULTS["delay_days"]
    window_end = window_start + DAYS_PER_YEAR
    if window_end <= 1:
        return None
    start = max(1, window_start)
    return start, int(window_end - start)


def load_iptp():
    # carry_back=1: 2025 dose-chain listeners are needed for pending doses.
    df = load_year_table(IPTP_FILE, carry_back=1)
    if df is None:
        return None, 0

    dose_cols = iptp_dose_columns(df)
    if not dose_cols:
        raise ValueError(f"{IPTP_FILE}: needs iptp1_coverage, iptp2_coverage, ...")
    if [n for n, _ in dose_cols] != list(range(1, len(dose_cols) + 1)):
        raise ValueError(f"{IPTP_FILE}: dose columns must be numbered 1..N.")
    if df["year"].duplicated().any():
        raise ValueError(f"{IPTP_FILE}: more than one row per year.")

    for _, col in dose_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        check_fraction(df[col].dropna(), f"{IPTP_FILE} {col}")

    df["year_start_day"] = (df["year"] - PROJ_START) * DAYS_PER_YEAR + 1
    df["doses"] = [iptp_dose_plan(row, dose_cols, int(row["year"]))
                   for _, row in df.iterrows()]
    return df, 0


def vaccine_expected_age_months(dose, year):
    """GHS schedule: D4 at 24 months before 2023, 18 months from 2023."""
    if dose == "D4":
        return 24 if int(year) < 2023 else 18
    return {"D1": 6, "D2": 7, "D3": 9}[dose]


def validate_vaccine_rows(df):
    """Validate cumulative coverage and derive conditional uptake `cond_cov`."""
    for col in ("dose", "coverage", "target_age_months"):
        if col not in df.columns:
            raise ValueError(f"{VACCINE_FILE}: needs a `{col}` column; "
                             f"found {list(df.columns)}")

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
                flag(year, f"unexpected dose '{dose}'")
            elif n > 1:
                flag(year, f"{dose} appears {n} times")
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
                flag(year, f"{r.dose}: coverage {r.cov_num:g} outside 0-1")
            want = vaccine_expected_age_months(r.dose, year)
            if pd.isna(r.age_num) or not np.isclose(r.age_num, want):
                flag(year, f"{r.dose}: target_age_months {r.age_num:g}, schedule needs {want}")
            usable[r.dose] = float(r.cov_num)

        for a, b in zip(VACCINE_EXPECTED_DOSES, VACCINE_EXPECTED_DOSES[1:]):
            if a in usable and b in usable and usable[b] > usable[a] + 1e-12:
                flag(year, f"{b} coverage ({usable[b]:g}) exceeds {a} ({usable[a]:g})")

        if year not in issues:
            prev = None
            for dose in VACCINE_EXPECTED_DOSES:
                c = usable[dose]
                cond = c if prev is None else (c / prev if prev > 0 else 0.0)
                mask = (df["year"].astype(int) == year) & (df["dose"] == dose)
                df.loc[mask, "cond_cov"] = min(cond, 1.0)
                prev = c

    if issues:
        print(f"\n  ERROR {VACCINE_FILE}: cohort(s) {sorted(issues)} FAILED "
              f"validation and get NO vaccine campaign:")
        for y in sorted(issues):
            for msg in issues[y]:
                print(f"    {y}: {msg}")

    return df.drop(columns=["cov_num", "age_num"]), issues


def load_vaccine():
    # carry_back=2: cohorts 2024 and 2025 still have doses due after day 1.
    df = load_year_table(VACCINE_FILE, carry_back=2)
    VACCINE_ISSUES.clear()
    if df is None:
        return None, 0

    df, _ = add_day(df, VACCINE_FILE, drop_outside=False)
    df, issues = validate_vaccine_rows(df)
    VACCINE_ISSUES.update(issues)
    df = df[~df["year"].astype(int).isin(list(issues))].reset_index(drop=True)
    if df.empty:
        return df, 0

    df["cov"] = need(df, ["coverage"], VACCINE_FILE)
    df["target_age_months"] = df["target_age_months"].astype(float)
    df["amin"] = df["target_age_months"] / 12.0 if VACCINE_USE_AGE_FLOOR else 0.0
    df["amax"] = VACCINE_CATCHUP_MAX_AGE_MONTHS / 12.0

    df["initial_effect"] = parameter_or_default(
        df, ["initial_effect", "efficacy"], VACCINE_NONCSV_DEFAULTS["initial_effect"], VACCINE_FILE)
    df["box_duration"] = parameter_or_default(
        df, ["box_duration"], VACCINE_NONCSV_DEFAULTS["box_duration"], VACCINE_FILE)
    df["decay"] = parameter_or_default(
        df, ["decay", "decay_time"], VACCINE_NONCSV_DEFAULTS["decay"], VACCINE_FILE)
    df["vaccine_type"] = parameter_or_default(
        df, ["vaccine_type"], VACCINE_NONCSV_DEFAULTS["vaccine_type"], VACCINE_FILE, str)

    df = df.sort_values(["year", "target_age_months"]).reset_index(drop=True)
    df["prev_dose"] = None
    df["delay_days"] = 0.0

    for _, group in df.groupby("year"):
        idx = list(group.index)
        for pos in range(1, len(idx)):
            i, prev_i = idx[pos], idx[pos - 1]
            df.at[i, "prev_dose"] = df.at[prev_i, "dose"]
            gap = df.at[i, "target_age_months"] - df.at[prev_i, "target_age_months"]
            df.at[i, "delay_days"] = max(gap * DAYS_PER_MONTH, 1.0)

    return df, 0


def load_all_schedules():
    sched, dropped = {}, {}
    sched["cm"], dropped["cm"] = load_case_management()
    sched["itn_anc"], dropped["itn_anc"] = load_routine_itn(ITN_ANC, "ANC")
    sched["itn_cwc"], dropped["itn_cwc"] = load_routine_itn(ITN_CWC, "CWC")
    sched["irs"], dropped["irs"] = load_irs()
    sched["smc"], dropped["smc"] = load_smc()
    sched["iptp"], dropped["iptp"] = load_iptp()
    sched["vaccine"], dropped["vaccine"] = load_vaccine()
    sched["dropped"] = dropped
    return sched


def describe_schedules(sched):
    print(f"\nProjection inputs, {PROJ_START}-{PROJ_END} (day 1 = 1 Jan {PROJ_START}, "
          f"{PROJ_DAYS} days):")
    for key, label in [("cm", "Case management"), ("itn_anc", "ITN-ANC"),
                       ("itn_cwc", "ITN-CWC"), ("irs", "IRS (product settings)"),
                       ("smc", "SMC"), ("iptp", "IPTp"), ("vaccine", "Vaccine")]:
        df = sched[key]
        if df is None:
            print(f"  {label}: file not found, skipped")
            continue
        if df.empty:
            print(f"  {label}: no rows")
            continue
        years = sorted({int(y) for y in df["year"]})
        rolled = int(df["rolled_forward"].sum()) if "rolled_forward" in df.columns else 0
        note = f"; {rolled} row(s) repeated from {REFERENCE_YEAR}" if rolled else ""
        print(f"  {label}: {len(df)} row(s), years {years[0]}-{years[-1]}{note}")


# =============================================================================
# ITN USAGE CURVES (routine ANC/CWC nets, and IG2 when IG2_USE_CURVES)
# =============================================================================

def _read_usage_table(filename, kind):
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ITN usage curves are needed but {filename} "
                                f"was not found in {INPUT_DIR}.")

    df = keep_phase(keep_node(read_clean_csv(path))).copy()
    if df.empty:
        raise ValueError(f"{filename}: no usable rows for node {NODE_ID}.")

    value_col = next((c for c in USAGE_VALUE_COLUMNS if c in df.columns), None)
    if value_col is None:
        raise ValueError(f"{filename}: needs a usage column among {USAGE_VALUE_COLUMNS}")

    usage = pd.to_numeric(df[value_col], errors="coerce")
    if usage.isna().any():
        raise ValueError(f"{filename}: `{value_col}` has blank or non-numeric entries.")
    check_fraction(usage, f"{filename} ({value_col})")
    df["_usage"] = usage.astype(float)

    if kind == "age":
        col = next((c for c in USAGE_AGE_COLUMNS if c in df.columns), None)
        if col is None:
            raise ValueError(f"{filename}: needs an age column among {USAGE_AGE_COLUMNS}")
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
                df, usage, x = df[keep].copy(), usage[keep], x[keep]
        elif "month" in df.columns:
            x = df["month"].map(month_mid_doy).astype(float)
        else:
            raise ValueError(f"{filename}: needs a day column or a `month` column.")
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
            raise ValueError(f"{filename}: unknown channel(s) {bad}.")
    return df.reset_index(drop=True)


def load_itn_usage():
    return {"age": _read_usage_table(ITN_AGE_FILE, "age"),
            "season": _read_usage_table(ITN_SEASON_FILE, "season")}


def _latest_block(d, year, label=""):
    """Rows of the latest dated block <= year; blank-year rows if none."""
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
            print(f"  WARNING ITN {label} usage: earliest curve is {int(first)}; "
                  f"also used for earlier years.")
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
        raise ValueError(f"{ITN_AGE_FILE}: duplicate ages for {channel}, {year}.")
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
    """Calendar curve rotated so Times = days since delivery."""
    rows = _select_usage_rows(ITN_USAGE["season"], channel, int(year), "season")
    days, vals = rows["_x"].tolist(), rows["_usage"].tolist()
    if len(set(days)) != len(days):
        raise ValueError(f"{ITN_SEASON_FILE}: duplicate days for {channel}, {year}.")

    start = float(delivery_doy)
    shifted = {round((d - start) % DAYS_PER_YEAR, 6): v for d, v in zip(days, vals)}
    shifted.pop(float(DAYS_PER_YEAR), None)
    v0 = _cyclic_interp(days, vals, start)
    shifted[0.0] = v0

    times = sorted(shifted)
    values = [shifted[t] for t in times]
    times.append(float(DAYS_PER_YEAR))
    values.append(v0)
    return {"Times": times, "Values": values}


def itn_usage_kwargs(channel, year, delivery_day):
    """Usage curves of the routine ANC/CWC nets (switched by USE_ITN_USAGE_CURVES)."""
    if not USE_ITN_USAGE_CURVES:
        return dict(age_dependence=None, seasonal_dependence=None)
    return dict(age_dependence=usage_age_curve(channel, year),
                seasonal_dependence=usage_season_curve(channel, year, doy_of_day(delivery_day)))


def ig2_usage_kwargs(year, delivery_day):
    """Usage curves of IG2 (MASS channel), switched by IG2_USE_CURVES only."""
    if not IG2_USE_CURVES:
        return dict(age_dependence=None, seasonal_dependence=None)
    return dict(age_dependence=usage_age_curve("MASS", year),
                seasonal_dependence=usage_season_curve("MASS", year, doy_of_day(delivery_day)))


def validate_itn_usage(sched):
    if USE_ITN_USAGE_CURVES:
        for key, channel in (("itn_anc", "ANC"), ("itn_cwc", "CWC")):
            df = sched[key]
            if not has_rows(df):
                continue
            for y in sorted({int(v) for v in df["year"]}):
                usage_age_curve(channel, y)
                usage_season_curve(channel, y, 1)
    if IG2_USE_CURVES:
        for y in IG2_CAMPAIGN_YEARS:
            usage_age_curve("MASS", y)
            usage_season_curve("MASS", y, IG2_CAMPAIGN_DOY)


# =============================================================================
# ROUTINE ITN WINDOWS
# =============================================================================

def anc_visit_event_name(year):
    return f"ANC_Visit_{int(year)}"


def cwc_visit_event_name(year):
    return f"CWC_Visit_{int(year)}"


def routine_year_window(year_start_day, delay):
    """(start_day, listening_duration, mid_delivery_day) or None."""
    d0 = int(year_start_day)
    d1 = d0 + DAYS_PER_YEAR - 1
    window_start = d0 - delay
    window_end = d1 - delay + 1
    if window_end <= 1:
        return None
    start = max(1, window_start)
    return start, int(window_end - start), (d0 + d1) // 2


def anc_year_window(year_start_day):
    return routine_year_window(year_start_day, ANC_DELAY_DAYS)


def cwc_year_window(year_start_day):
    return routine_year_window(year_start_day, CWC_DELAY_DAYS)


def cwc_catchup_span(year_start_day):
    """Delivery days of a CWC year that no Births trigger can serve."""
    first = int(year_start_day)
    last = min(first + DAYS_PER_YEAR - 1, CWC_DELAY_DAYS)
    return (first, last) if first <= last else None


# =============================================================================
# VECTOR / SITE INPUT
# =============================================================================

DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)
VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)

for _p in [INPUT_DIR, DEMOGRAPHICS_PATH, VECTOR_PATH,
           os.path.join(INPUT_DIR, CM_FILE),
           manifest.eradication_path, manifest.schema_file]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"Required path does not exist:\n{_p}")

vector_df = pd.read_csv(VECTOR_PATH)
vector_df.columns = [c.strip() for c in vector_df.columns]

if APPLY_VECTOR_BEHAVIOUR:
    _missing = [c for c in ("anthropophily", "indoor_feeding") if c not in vector_df.columns]
    if _missing:
        raise ValueError(f"APPLY_VECTOR_BEHAVIOUR is True but {VECTOR_FILE} "
                         f"lacks column(s) {_missing}.")


def _read_site_definition():
    site_df = pd.read_csv(DEMOGRAPHICS_PATH)
    site_df.columns = [c.strip() for c in site_df.columns]
    if len(site_df) != 1 or int(site_df["node_id"].iloc[0]) != NODE_ID:
        raise ValueError(f"{DEMOGRAPHICS_FILE} must hold exactly one row with node_id={NODE_ID}.")
    row = site_df.iloc[0]
    return {"name": SITE_NAME, "pop": int(row["pop"]),
            "lat": float(row["lat"]), "lon": float(row["lon"])}


# =============================================================================
# CLIMATE
# =============================================================================

CLIMATE_VARS = ["air_temperature", "land_temperature", "rainfall", "relative_humidity"]
def climate_file(var):
    return CLIMATE_FILE_TEMPLATE.format(var=var)


CLIMATE_FILES = [climate_file(v) + ext for v in CLIMATE_VARS for ext in ("", ".json")]


def resolve_climate_dir():
    if CLIMATE_SUBDIR:
        return os.path.join(CLIMATE_ROOT, CLIMATE_SUBDIR)
    if os.path.isfile(os.path.join(CLIMATE_ROOT, climate_file("air_temperature"))):
        return CLIMATE_ROOT
    if not os.path.isdir(CLIMATE_ROOT):
        raise FileNotFoundError(f"Projection climate folder does not exist:\n{CLIMATE_ROOT}")
    found = [d for d in sorted(os.listdir(CLIMATE_ROOT))
             if os.path.isfile(os.path.join(CLIMATE_ROOT, d, climate_file("air_temperature")))]
    if len(found) == 1:
        return os.path.join(CLIMATE_ROOT, found[0])
    if not found:
        raise FileNotFoundError(f"No {climate_file('air_temperature')} in {CLIMATE_ROOT} "
                                f"or its subfolders.")
    raise ValueError(f"Several weather folders in {CLIMATE_ROOT}: {found}. Set CLIMATE_SUBDIR.")


def validate_climate(climate_dir):
    for name in CLIMATE_FILES:
        path = os.path.join(climate_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing climate file:\n{path}")

    for var in CLIMATE_VARS:
        with open(os.path.join(climate_dir, climate_file(var) + ".json")) as f:
            meta = json.load(f).get("Metadata", {})
        count = meta.get("DatavalueCount")
        if count is not None and int(count) < PROJ_DAYS:
            raise ValueError(f"{var}: weather has {count} days but the projection "
                             f"needs {PROJ_DAYS}; EMOD would wrap to the start of the file.")
    print(f"Climate: {climate_dir}")
    print(f"  files {CLIMATE_FILE_TEMPLATE.format(var='<var>')} (same as the pickup); "
          f"{PROJ_START}-{PROJ_END} uses file days 1-{PROJ_DAYS}; "
          f"the SAME weather is used in every arm")


# =============================================================================
# CONFIGURATION
# =============================================================================

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


def merge_pickup_events_into_config(config, label=""):
    """
    Union the pickup's custom events into config.parameters.Custom_Individual_Events.
    Returns the number of events that had to be added.
    """
    existing = list(getattr(config.parameters, "Custom_Individual_Events", []) or [])
    projection = projection_custom_events() if SF is not None else []
    merged = sorted(set(existing) | set(PICKUP_CUSTOM_EVENTS) | set(projection))
    added = len(merged) - len(set(existing))
    config.parameters.Custom_Individual_Events = merged
    if label:
        print(f"[EVENT CHECK] {label}: {len(existing)} existing + {added} pickup "
              f"event(s) added = {len(merged)}")
    return added


def set_param_fn(config):
    config = conf.set_team_defaults(config, manifest)

    # First attempt at carrying the pickup events. emodpy may rebuild
    # Custom_Individual_Events from the campaign later, so the campaign
    # registration (register_pickup_events) and the post-pre_creation merge
    # (PATCH_CONFIG_EVENTS) are the layers that actually guarantee it.
    if PICKUP_CUSTOM_EVENTS:
        merge_pickup_events_into_config(config, label="set_param_fn")

    config.parameters.Start_Time = 0
    config.parameters.Simulation_Duration = PROJ_DAYS

    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", climate_file("air_temperature"))
    config.parameters.Land_Temperature_Filename = os.path.join("climate", climate_file("land_temperature"))
    config.parameters.Rainfall_Filename = os.path.join("climate", climate_file("rainfall"))
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", climate_file("relative_humidity"))

    config.parameters.Enable_Default_Reporting = 1   # InsetChart (daily population)

    for logger in ["JsonConfigurable", "VectorHabitat", "StandardEventCoordinator",
                   "SusceptibilityMalaria", "default", "Eradication"]:
        config.parameters[f"logLevel_{logger}"] = "ERROR"

    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = int(MEMORY_LIMIT)
    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = int(MEMORY_LIMIT - 500)

    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    # MUST EQUAL THE PICKUP: species, habitats, vector behaviour.
    conf.add_species(config, manifest, vector_df["species"].tolist())

    for i in range(len(vector_df)):
        species = vector_df["species"].iloc[i]

        if APPLY_VECTOR_BEHAVIOUR:
            conf.set_species_param(config, species=species, parameter="Anthropophily",
                                   value=float(vector_df["anthropophily"].iloc[i]),
                                   overwrite=True)
            conf.set_species_param(config, species=species, parameter="Indoor_Feeding_Fraction",
                                   value=float(vector_df["indoor_feeding"].iloc[i]),
                                   overwrite=True)

        conf.set_species_param(
            config, species=species, parameter="Habitats",
            value=[
                {"Habitat_Type": "CONSTANT",
                 "Max_Larval_Capacity": 1e7 * vector_df["constant"].iloc[i]},
                {"Habitat_Type": "TEMPORARY_RAINFALL",
                 "Max_Larval_Capacity": 1e7 * vector_df["temp_rain"].iloc[i]},
                {"Habitat_Type": "WATER_VEGETATION",
                 "Max_Larval_Capacity": 1e7 * vector_df["water_veg"].iloc[i]},
            ],
            overwrite=True,
        )

    # Placeholders, overwritten per simulation from the pickup row.
    config.parameters.Run_Number = 0
    config.parameters.x_Temporary_Larval_Habitat = 1

    config.parameters.Serialized_Population_Reading_Type = "READ"
    config.parameters.Serialization_Mask_Node_Read = 0

    if WRITE_END_STATE:
        config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
        config.parameters.Serialization_Time_Steps = [PROJ_DAYS]
        config.parameters.Serialization_Mask_Node_Write = 0
        config.parameters.Serialization_Precision = "REDUCED"

    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    config.parameters.x_Birth = X_BIRTH
    return config


# =============================================================================
# DEMOGRAPHICS  (same as the pickup)
# =============================================================================

def validate_demographic_settings():
    if "PREGNANC" not in BIRTH_RATE_DEPENDENCE:
        return
    effective = PREG_RATE_PER_MOTHER_DAY * X_BIRTH
    if PREG_RATE_PER_MOTHER_DAY <= 0 or X_BIRTH <= 0:
        raise ValueError("PREG_RATE_PER_MOTHER_DAY and X_BIRTH must both be > 0.")
    if effective > MAX_PREG_RATE_PER_MOTHER_DAY:
        raise ValueError(f"PREG_RATE_PER_MOTHER_DAY x X_BIRTH = {effective:g} exceeds one "
                         f"pregnancy per 40 weeks ({MAX_PREG_RATE_PER_MOTHER_DAY:.2e}).")
    if CRUDE_DEATH_RATE <= 0:
        raise ValueError("CRUDE_DEATH_RATE must be > 0.")


def add_pregnancy_property(demog):
    if hasattr(demog, "AddIndividualPropertyAndHINT"):
        demog.AddIndividualPropertyAndHINT(
            Property="Pregnancy", Values=["NotPregnant", "IsPregnant"],
            InitialDistribution=[1.0, 0.0])
    else:
        demog.add_individual_property(
            property="Pregnancy", values=["NotPregnant", "IsPregnant"],
            initial_distribution=[1.0, 0.0])


def _final_demographics_config(config):
    """Last demographics implicit: forces the birth model back and drops stale params."""
    config.parameters.Birth_Rate_Dependence = BIRTH_RATE_DEPENDENCE
    config.parameters.x_Birth = X_BIRTH
    remove_stale_params(config)
    return config


def build_demog():
    site = _read_site_definition()
    demog = Demographics.from_template_node(lat=site["lat"], lon=site["lon"],
                                            pop=site["pop"], name=site["name"])
    demog.SetBirthRate(YearlyRate(PREG_RATE_PER_MOTHER_DAY * DAYS_PER_YEAR))
    demog.SetMortalityRate(CrudeRate(CRUDE_DEATH_RATE))
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)
    add_pregnancy_property(demog)
    demog.implicits.append(_final_demographics_config)
    return demog


# =============================================================================
# CAMPAIGN HELPERS
# =============================================================================

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


def call_supported(fn, campaign, label, **kwargs):
    """Call an emodpy function with only the keyword arguments it supports."""
    params = inspect.signature(fn).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        # broadcast_event is only used to name events for reporting; the
        # installed emodpy_malaria does not support it and nothing needs it.
        dropped = [k for k in kwargs if k not in params and k != "broadcast_event"]
        if dropped and label not in _WARNED:
            print(f"  WARNING {label}: {fn.__name__} does not accept {dropped}; "
                  f"those settings are NOT applied.")
            _WARNED.add(label)
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(campaign, **kwargs)


# -----------------------------------------------------------------------------
# Collision-safe names for emodpy_malaria drug-campaign helper events
# -----------------------------------------------------------------------------
# Some emodpy_malaria versions create internal events named MDA_Now_<integer>
# using random.randint(1, 10000). If two drug campaigns of ONE campaign file
# receive the same name, their listeners can cross-trigger. While
# add_drug_campaign() runs, random.randint(1, 10000) is replaced by a counter
# that skips ids already used in this campaign and the MDA_Now ids of the
# pickup (the serialized state may still hold pending listeners for them).
#
# The counter MUST be reset at the start of every campaign build
# (reset_mda_ids, called in build_camp). Ids only have to be unique inside one
# campaign; without the reset they are used up across the preflight builds and
# the per-simulation builds ("Exhausted IDs 1..10000").
_MDA_USED_IDS = set()
_MDA_NEXT_ID = 1
_MDA_PICKUP_IDS = set()


def reset_mda_ids():
    """Start of every campaign build: restart the counter, cache the pickup's ids."""
    global _MDA_NEXT_ID, _MDA_PICKUP_IDS
    _MDA_USED_IDS.clear()
    _MDA_NEXT_ID = 1
    _MDA_PICKUP_IDS = {
        int(m.group(1)) for e in (PICKUP_CUSTOM_EVENTS or []) if isinstance(e, str)
        for m in [re.fullmatch(r"MDA_Now_(\d+)", e)] if m
    }


def call_drug_campaign_collision_safe(fn, campaign, label, **kwargs):
    original_randint = random.randint

    def unique_randint(a, b):
        global _MDA_NEXT_ID
        if a == 1 and b == 10000:
            while _MDA_NEXT_ID in _MDA_USED_IDS or _MDA_NEXT_ID in _MDA_PICKUP_IDS:
                _MDA_NEXT_ID += 1
            if _MDA_NEXT_ID > 10000:
                raise RuntimeError(
                    "Exhausted IDs 1..10000 within ONE campaign "
                    f"({len(_MDA_USED_IDS)} used, {len(_MDA_PICKUP_IDS)} reserved "
                    f"by the pickup).")
            value = _MDA_NEXT_ID
            _MDA_USED_IDS.add(value)
            _MDA_NEXT_ID += 1
            return value
        return original_randint(a, b)

    random.randint = unique_randint
    try:
        return call_supported(fn, campaign, label, **kwargs)
    finally:
        random.randint = original_randint


def require_supported(fn, label, names):
    """Raise if fn lacks arguments that must NOT be silently dropped."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError(f"{label}: {fn.__name__} does not accept {missing}, "
                         f"which this intervention requires.")


def weibull_discard_config(r):
    """Weibull discard for a row with discard_k / discard_l, or None."""
    if np.isnan(r.discard_k) or np.isnan(r.discard_l):
        return None
    if r.discard_k <= 0 or r.discard_l <= 0:
        return None
    return {"Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",
            "Expiration_Period_Kappa": float(r.discard_k),
            "Expiration_Period_Lambda": float(r.discard_l)}


def ig2_discard_config():
    return {"Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",
            "Expiration_Period_Kappa": float(IG2["discard_k"]),
            "Expiration_Period_Lambda": float(IG2["discard_l"])}


# =============================================================================
# PICKUP EVENT REGISTRATION
# =============================================================================

def pickup_family_events():
    """Pickup events that the scenario run must know about."""
    return sorted(e for e in (PICKUP_CUSTOM_EVENTS or [])
                  if isinstance(e, str) and PICKUP_EVENT_FAMILY_RE.match(e))


def add_dormant_event(campaign, event_name):
    """
    Register `event_name` as a custom event through a campaign event that can
    never fire (it starts long after the simulation ends). emodpy collects
    custom events from the campaign, so the name lands in the config's
    Custom_Individual_Events, which EMOD needs to deserialize the pickup state.
    """
    if not (hasattr(common, "triggered_campaign_event_with_optional_delay")
            and hasattr(common, "BroadcastEvent")):
        raise ValueError("Registering pickup events needs emod_api.interventions.common."
                         "triggered_campaign_event_with_optional_delay and BroadcastEvent.")
    fn = common.triggered_campaign_event_with_optional_delay
    kwargs = dict(start_day=int(DORMANT_EVENT_START_DAY), triggers=["Births"],
                  intervention=common.BroadcastEvent(campaign, event_name),
                  delay={"Delay_Period_Constant": 0.0},
                  duration=1, coverage=1.0)
    if "blackout" in inspect.signature(fn).parameters:
        kwargs["blackout"] = False
    campaign.add(fn(campaign, **kwargs))


def register_pickup_events(campaign):
    """
    Make sure every pickup event is registered by the scenario campaign.
    Events the scenario already uses (as a trigger or broadcast name) are
    skipped; the rest get a dormant registration event.
    Returns the list of events registered this way.
    """
    events = sorted(set(pickup_family_events()) |
                    (set(projection_custom_events()) if SF is not None else set()))
    if not events:
        return []

    text = json.dumps(campaign.campaign_dict)
    added = []
    for ev in events:
        if f'"{ev}"' in text:
            continue
        add_dormant_event(campaign, ev)
        added.append(ev)
    return added


def verify_pickup_events(campaign):
    """Raise if any pickup event family member is absent from the built campaign."""
    text = json.dumps(campaign.campaign_dict)
    missing = [ev for ev in pickup_family_events() if f'"{ev}"' not in text]
    if missing:
        raise RuntimeError(
            f"{len(missing)} pickup event(s) are not registered by the scenario "
            f"campaign and EMOD could not load the pickup state: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}")
    return len(pickup_family_events())


def install_event_patch(simulation):
    """
    Second safety layer. After emodpy's pre_creation has rebuilt the config,
    union the pickup events into Custom_Individual_Events again.
    """
    if not PATCH_CONFIG_EVENTS:
        return

    task = simulation.task
    original = type(task).pre_creation

    def pre_creation_with_events(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            merge_pickup_events_into_config(self.config)
        except Exception as exc:  # never block a submission on the safety layer
            print(f"  WARNING: could not merge pickup events after pre_creation: {exc}")
        return result

    task.pre_creation = types.MethodType(pre_creation_with_events, task)


# =============================================================================
# PREGNANCY TRACKING, CASE MANAGEMENT (same as the pickup)
# =============================================================================

def add_pregnancy_tracking(campaign):
    for value, trig in (("IsPregnant", "Pregnant"), ("NotPregnant", "GaveBirth")):
        common.change_individual_property_triggered(
            campaign, new_ip_key="Pregnancy", new_ip_value=value,
            target_sex="Female", triggers=[trig], start_day=1, daily_prob=1,
            revert_in_days=-1, blackout=False)
    return campaign


def add_case_management(campaign, df):
    if not has_rows(df):
        return
    for (day, duration, drug, severe), group in df.groupby(
            ["proj_day", "duration", "drug", "is_severe"]):
        if int(duration) <= 0:
            continue
        targets = [{"trigger": r.trigger, "coverage": float(r.effective_coverage),
                    "agemin": float(r.age_min), "agemax": float(r.age_max),
                    "rate": float(r.rate)} for r in group.itertuples()]
        add_treatment_seeking(
            campaign, start_day=int(day), drug=CM_DRUGS[drug], targets=targets,
            duration=int(duration),
            broadcast_event_name="Received_Severe_Treatment" if severe else "Received_Treatment")


# =============================================================================
# ROUTINE ANC / CWC ITN (same logic as the pickup)
# =============================================================================

def add_visit_event(campaign, start, listen, delay, event_name, trigger, target_sex=None):
    if not (hasattr(common, "triggered_campaign_event_with_optional_delay")
            and hasattr(common, "BroadcastEvent")):
        raise ValueError("Two-stage routine nets need emod_api.interventions.common."
                         "triggered_campaign_event_with_optional_delay and BroadcastEvent.")
    fn = common.triggered_campaign_event_with_optional_delay
    kwargs = dict(start_day=int(start), triggers=[trigger],
                  intervention=common.BroadcastEvent(campaign, event_name),
                  delay={"Delay_Period_Constant": float(delay)},
                  duration=int(listen), coverage=1.0)
    if target_sex:
        kwargs["target_sex"] = target_sex
    if "blackout" in inspect.signature(fn).parameters:
        kwargs["blackout"] = False
    campaign.add(fn(campaign, **kwargs))


def add_routine_itns(campaign, df, channel):
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_triggered_usage_dependent_bednet)

    if not has_rows(df):
        return

    bednet_params = inspect.signature(add_triggered_usage_dependent_bednet).parameters
    delay_key = ("delay_period_constant" if "delay_period_constant" in bednet_params
                 else "triggered_campaign_delay")
    newer_restrictions = "property_restrictions" in bednet_params
    two_stage = channel == "CWC" or (channel == "ANC" and ANC_CHECK_PREGNANT_AT_DELIVERY)

    require_supported(add_triggered_usage_dependent_bednet, f"ITN-{channel}",
                      ["trigger_condition_list", "listening_duration"]
                      + ([] if two_stage else [delay_key]))

    if channel == "ANC":
        trigger, delay = "Pregnant", ANC_DELAY_DAYS
        visit_name, window_fn, visit_sex = anc_visit_event_name, anc_year_window, "Female"
        if not ANC_CHECK_PREGNANT_AT_DELIVERY:
            restriction = {}
        elif newer_restrictions:
            restriction = dict(property_restrictions=["Pregnancy:IsPregnant"])
        else:
            restriction = dict(ind_property_restrictions=[{"Pregnancy": "IsPregnant"}])
        extra = dict(target_gender="Female", **restriction)
    elif channel == "CWC":
        trigger, delay = "Births", CWC_DELAY_DAYS
        visit_name, window_fn, visit_sex, extra = cwc_visit_event_name, cwc_year_window, None, {}
    else:
        raise ValueError(f"Unknown routine ITN channel: {channel}")

    for r in df.itertuples():
        year = int(r.year)
        window = window_fn(r.year_start_day)
        if window is None:
            continue
        start, listen, mid_day = window

        if two_stage:
            visit_event = visit_name(year)
            add_visit_event(campaign, start, listen, delay, visit_event, trigger,
                            target_sex=visit_sex)
            net_trigger, net_start, net_listen, net_delay = [visit_event], 1, -1, {}
        else:
            net_trigger, net_start, net_listen = [trigger], start, listen
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

        call_supported(add_triggered_usage_dependent_bednet, campaign,
                       f"ITN-{channel}", **kwargs)


def add_cwc_catchup(campaign, df):
    """CWC nets for children already born on day 1 (one delivery per year-span)."""
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_scheduled_usage_dependent_bednet)

    if not has_rows(df):
        return
    spans = [(r, cwc_catchup_span(r.year_start_day)) for r in df.itertuples()]
    if not any(s for _, s in spans):
        return

    require_supported(add_scheduled_usage_dependent_bednet, "ITN-CWC catch-up",
                      ["target_age_min", "target_age_max"])

    age_min = CWC_DELAY_DAYS / DAYS_PER_YEAR

    for r, span in spans:
        if not span:
            continue
        b_start, b_end = span
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
            broadcast_event="Received_ITN_CWC_Catchup",
            **itn_usage_kwargs("CWC", int(r.year), b_end),
        )
        discard = weibull_discard_config(r)
        if discard is not None:
            kwargs["discard_config"] = discard
        call_supported(add_scheduled_usage_dependent_bednet, campaign,
                       "ITN-CWC catch-up", **kwargs)


# =============================================================================
# IRS (continuation arms) AND IG2 (replacement arms)
# =============================================================================

def add_irs(campaign, df):
    from emodpy_malaria.interventions.irs import add_scheduled_irs_housing_modification

    if not has_rows(df):
        return
    for r in df.itertuples():
        call_supported(
            add_scheduled_irs_housing_modification, campaign, "IRS",
            intervention_name=r.iv_name,
            start_day=int(r.proj_day),
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


def rescale_irs_coverage(df, target):
    """
    Set the TOTAL IRS coverage of EVERY year to `target`, keeping the CSV's
    split between rows (products / rounds) where the CSV total is > 0 and
    splitting equally otherwise. A year whose CSV coverage is 0 (e.g. a 2027
    row of 0.0) therefore still gets `target`: IRS_<c> means IRS at c in every
    year from IRS_SWITCH_YEAR. target=None returns the rows unchanged (BAU).
    """
    if not has_rows(df) or target is None:
        return df
    df = df.copy()
    total = df.groupby("year")["cov"].transform("sum")
    n_rows = df.groupby("year")["cov"].transform("size")
    share = np.where(total > 0, df["cov"] / total.where(total > 0, 1.0), 1.0 / n_rows)
    df["cov"] = (float(target) * share).clip(0.0, 1.0)
    return df


def ig2_campaign_days():
    days = []
    for y in IG2_CAMPAIGN_YEARS:
        d = (int(y) - PROJ_START) * DAYS_PER_YEAR + int(IG2_CAMPAIGN_DOY)
        if 1 <= d <= PROJ_DAYS:
            days.append(d)
        else:
            print(f"  WARNING: IG2 campaign year {y} is outside the projection; skipped.")
    return days


def add_ig2(campaign, coverage, kill_decay):
    """
    Scheduled IG2 mass distribution at demographic coverage `coverage` (= c).
    The same coverage is used at every re-issue. With IG2_USE_CURVES the nets
    carry the MASS age/season usage curves, so effective use is lower than c
    (see the docstring and analyze_scenarios_UWR.py).
    """
    from emodpy_malaria.interventions.usage_dependent_bednet import (
        add_scheduled_usage_dependent_bednet)

    require_supported(add_scheduled_usage_dependent_bednet, "IG2",
                      ["demographic_coverage", "killing_decay_time_constant",
                       "blocking_decay_time_constant"])

    for day in ig2_campaign_days():
        year = PROJ_START + (day - 1) // DAYS_PER_YEAR
        kwargs = dict(
            intervention_name="ITN_IG2",
            start_day=int(day),
            demographic_coverage=float(coverage),
            blocking_initial_effect=float(IG2["blocking_initial"]),
            blocking_decay_time_constant=float(IG2["blocking_decay"]),
            killing_initial_effect=float(IG2["killing_initial"]),
            killing_decay_time_constant=float(kill_decay),
            discard_config=ig2_discard_config(),
            broadcast_event="Received_ITN_IG2",
        )
        kwargs.update(ig2_usage_kwargs(year, day))

        call_supported(add_scheduled_usage_dependent_bednet, campaign, "IG2", **kwargs)


def mean_retention(cycle_days):
    """Mean fraction of delivered nets still held over `cycle_days` (Weibull)."""
    t = np.linspace(0.0, float(cycle_days), 2001)
    s = np.exp(-(t / IG2["discard_l"]) ** IG2["discard_k"])
    return float(np.mean((s[:-1] + s[1:]) / 2.0))


# =============================================================================
# SMC, IPTp, VACCINE (same logic as the pickup)
# =============================================================================

def add_smc(campaign, df):
    from emodpy_malaria.interventions.drug_campaign import add_drug_campaign

    if not has_rows(df):
        return
    for r in df.itertuples():
        call_drug_campaign_collision_safe(
            add_drug_campaign, campaign, "SMC",
            campaign_type="SMC", drug_code=r.drug_code,
            start_days=[int(r.proj_day)], coverage=float(r.cov), repetitions=1,
            target_group={"agemin": float(r.amin), "agemax": float(r.amax)},
            receiving_drugs_event_name="Received_SMC")


def add_iptp(campaign, df):
    """
    Pregnancy-triggered, dose-chained IPTp.
    Years before PROJ_START only add the D2+ listeners, so doses still pending
    from the pickup keep chaining.
    """
    from emodpy_malaria.interventions.drug_campaign import add_drug_campaign

    if not has_rows(df):
        return

    require_supported(add_drug_campaign, "IPTp",
                      ["triggered_campaign_delay", "trigger_condition_list",
                       "listening_duration"])
    d = IPTP_DEFAULTS

    for r in df.itertuples():
        year = int(r.year)
        carry = year < PROJ_START

        for n, cum, cond in r.doses:
            if cond <= 0:
                continue
            event = f"Received_IPTp_D{n}_{year}"

            if n == 1:
                if carry:
                    continue
                window = iptp_trigger_window(r.year_start_day)
                if window is None:
                    break
                start, listen = window
                trigger, delay = [d["trigger"]], d["delay_days"]
            else:
                start, listen = 1, -1
                trigger, delay = [f"Received_IPTp_D{n - 1}_{year}"], d["gap_days"]

            call_drug_campaign_collision_safe(
                add_drug_campaign, campaign, "IPTp",
                campaign_type="MDA", drug_code=d["drug_code"],
                start_days=[int(start)], repetitions=1, coverage=float(cond),
                triggered_campaign_delay=delay, trigger_condition_list=trigger,
                listening_duration=int(listen), receiving_drugs_event_name=event)


def add_vaccine(campaign, df):
    """
    RTS,S, birth-triggered and dose-chained (see the pickup for the logic).
    Cohorts born before PROJ_START only add D2+ listeners, so doses still
    pending from the pickup keep chaining.
    """
    from emodpy_malaria.interventions.vaccine import add_triggered_vaccine

    if not has_rows(df):
        return

    for r in df.itertuples():
        year = int(r.year)
        carry = year < PROJ_START

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

        if pd.isna(r.prev_dose) or r.prev_dose is None:
            if carry:
                continue
            call_supported(
                add_triggered_vaccine, campaign, "Vaccine",
                trigger_condition_list=["Births"],
                delay_period_constant=float(r.target_age_months) * DAYS_PER_MONTH,
                start_day=int(r.proj_day),
                listening_duration=DAYS_PER_YEAR,
                demographic_coverage=float(r.cond_cov),
                **common_kwargs)
        else:
            call_supported(
                add_triggered_vaccine, campaign, "Vaccine",
                trigger_condition_list=[f"Received_Vaccine_{r.prev_dose}_{year}"],
                delay_period_constant=float(r.delay_days),
                demographic_coverage=float(r.cond_cov),
                start_day=1, listening_duration=-1,
                **common_kwargs)


# =============================================================================
# HABITAT SCALING
# =============================================================================

def scale_habitat_from_mix(mix):
    scales = {"NodeID": vector_df["node_id"].unique()}
    for sp in vector_df["species"].unique():
        frac = vector_df.loc[vector_df["species"] == sp, "fraction"]
        scales[f"CONSTANT.{sp}"] = [mix[0] * f for f in frac]
        scales[f"TEMPORARY_RAINFALL.{sp}"] = [mix[1] * f for f in frac]
        scales[f"WATER_VEGETATION.{sp}"] = [mix[2] * f for f in frac]
    return pd.DataFrame(scales)


# =============================================================================
# SCENARIOS
# =============================================================================

def scenario_label(c):
    return f"{int(round(c * 100))}"


def cm_suffix(boost):
    """'' for current CM, '_CM5' for +5 percentage points."""
    return "" if not boost else f"_CM{int(round(boost * 100))}"


def boost_case_management(df, boost):
    """
    Raise effective CM coverage (coverage x seek) by `boost` (absolute, e.g.
    0.05 = +5 percentage points) for rows of year >= CM_BOOST_START_YEAR.
    Severe-case rows are changed only if CM_BOOST_SEVERE. Capped at CM_MAX_COVERAGE.
    """
    if not boost or not has_rows(df):
        return df
    df = df.copy()
    rows = df["year"].astype(int) >= CM_BOOST_START_YEAR
    if not CM_BOOST_SEVERE:
        rows &= ~df["is_severe"]
    df.loc[rows, "effective_coverage"] = np.minimum(
        df.loc[rows, "effective_coverage"].astype(float) + float(boost), CM_MAX_COVERAGE)
    return df


def build_scenarios():
    scenarios = []

    # Optional reference arms (off by default).
    if INCLUDE_IRS_BASELINE_ARM:
        scenarios.append(dict(name=BAU_ARM_NAME, kind="irs", coverage=0.0,
                              irs_coverage=None, kill_key=""))

    if INCLUDE_NO_IRS_ARM:
        scenarios.append(dict(name="NO_IRS", kind="none", coverage=0.0,
                              irs_coverage=None, kill_key=""))

    # IRS continuation at a fixed total annual coverage, no IG2 nets.
    for c in sorted({round(float(c), 4) for c in IRS_COVERAGES}):
        if not 0 < c <= 1:
            raise ValueError(f"IRS coverage {c} must be in (0, 1].")
        scenarios.append(dict(name=f"IRS_{scenario_label(c)}", kind="irs",
                              coverage=0.0, irs_coverage=c, kill_key=""))

    # IG2 replacement arms (IRS removed from IRS_SWITCH_YEAR).
    coverages = sorted({round(float(c), 4) for c in IG2_COVERAGES + EXTRA_IG2_COVERAGES})
    for key in IG2_KILL_RUN:
        if key not in IG2_KILL_DECAY:
            raise ValueError(f"IG2_KILL_RUN key '{key}' not in IG2_KILL_DECAY.")
        for c in coverages:
            if not 0 < c <= 1:
                raise ValueError(f"IG2 coverage {c} must be in (0, 1].")
            scenarios.append(dict(name=f"IG2_{scenario_label(c)}_{key}", kind="ig2",
                                  coverage=c, irs_coverage=None, kill_key=key))

    if not scenarios:
        raise ValueError("No scenarios selected: check IRS_COVERAGES, IG2_COVERAGES "
                         "and the INCLUDE_* flags.")

    # Every arm x every case-management level.
    boosts = sorted({round(float(b), 4) for b in CM_BOOSTS})
    if not boosts or any(not 0 <= b < 1 for b in boosts):
        raise ValueError(f"CM_BOOSTS {CM_BOOSTS} must be in [0, 1).")
    scenarios = [dict(s, strategy=s["name"], cm_boost=b, name=s["name"] + cm_suffix(b))
                 for b in boosts for s in scenarios]
    names = [s["name"] for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate scenario names: {names}")
    return scenarios


def describe_scenarios(scenarios):
    print("\nScenarios (identical weather, states, seeds and other programmes):")
    for s in scenarios:
        if s["kind"] == "irs":
            if s.get("irs_coverage") is None:
                what = (f"IRS continues after {IRS_SWITCH_YEAR} at the CSV coverage "
                        f"({REFERENCE_YEAR} rows repeated if the CSV has no projection rows)")
            else:
                what = (f"IRS continues after {IRS_SWITCH_YEAR} at total annual coverage "
                        f"{s['irs_coverage']:.2f}; no IG2 nets")
        elif s["kind"] == "none":
            what = f"no IRS and no IG2 from 1 Jan {IRS_SWITCH_YEAR} (lower bound)"
        else:
            what = (f"IRS removed from 1 Jan {IRS_SWITCH_YEAR}; IG2 distributed at coverage "
                    f"{s['coverage']:.2f} on "
                    f"{', '.join(fmt_day(d) for d in ig2_campaign_days())}; "
                    f"kill decay {IG2_KILL_DECAY[s['kill_key']]:g} d")
        if s.get("cm_boost"):
            what += (f"; CM +{s['cm_boost']*100:g} pp from {CM_BOOST_START_YEAR}"
                     f"{' (incl. severe)' if CM_BOOST_SEVERE else ''}")
        print(f"  {s['name']:<16} {what}")

    print("\n  NOTE: IRS coverage (share of structures sprayed) and IG2 coverage "
          "(share of people receiving a net) are comparable scenario levels, "
          "not equivalent measures.")

    cycle = (int(IG2_CAMPAIGN_YEARS[1] - IG2_CAMPAIGN_YEARS[0]) * DAYS_PER_YEAR
             if len(IG2_CAMPAIGN_YEARS) > 1 else 3 * DAYS_PER_YEAR)
    r = mean_retention(cycle)
    if IG2_USE_CURVES:
        print("\nIG2 usage: MASS age/season curves.")
        print("  effective use = c x net retention x age multiplier x season multiplier")
        print("  (analyze_scenarios_UWR.py reports it for every c and for the threshold)")
    else:
        print("\nIG2 usage: multiplier = 1 (effective use at distribution = c).")
    print(f"  net discard Weibull k={IG2['discard_k']:g}, lambda={IG2['discard_l']:g} d")
    print(f"  mean fraction of nets still held over a {cycle}-day cycle: {r:.2f}")
    print(f"  => time-averaged held coverage is roughly c x {r:.2f} "
          f"(before overlap with re-issue)")


# =============================================================================
# CONTINUING PROGRAMMES FROM THE PICKUP SCRIPT (Site_Setup_Final.py)
# =============================================================================

SF = None                 # the imported pickup script, calendar moved to the projection
ROLLED_FORWARD = {}       # file -> projection years filled from REFERENCE_YEAR


def write_projection_inputs(sf):
    """
    Write the pickup script's input files for PROJ_START..PROJ_END into
    PROJECTION_INPUT_DIR. A year that has its own rows (any phase in
    PHASES_ACCEPTED) keeps them; a missing year repeats REFERENCE_YEAR's rows.
    Rows are labelled phase='pickup' (what the pickup loaders read) and
    start_day is reduced to day of year.
    """
    import shutil
    os.makedirs(PROJECTION_INPUT_DIR, exist_ok=True)
    ROLLED_FORWARD.clear()
    files = [sf.CM_FILE, sf.ITN_ANC, sf.ITN_CWC, sf.IRS_FILE, sf.SMC_FILE,
             sf.IPTP_FILE, sf.VACCINE_FILE] + ([sf.ITN_FILE] if sf.ITN_FILE else [])
    for name in files:
        src = os.path.join(INPUT_DIR, name)
        dst = os.path.join(PROJECTION_INPUT_DIR, name)
        if not os.path.isfile(src):
            if os.path.exists(dst):
                os.remove(dst)
            continue
        df = sf.keep_node(sf.read_clean_csv(src)).copy()
        if "phase" in df.columns:
            df = df[df["phase"].astype(str).str.lower().isin(PHASES_ACCEPTED)]
        df["year"] = df["year"].astype(int)
        pieces, rolled = [], []
        for y in range(PROJ_START, PROJ_END + 1):
            own = df[df["year"] == y]
            if own.empty:
                own = df[df["year"] == REFERENCE_YEAR].copy()
                if own.empty:
                    raise ValueError(f"{name}: no rows for {y} and none for REFERENCE_YEAR "
                                     f"{REFERENCE_YEAR} to repeat.")
                own["year"] = y
                rolled.append(y)
            pieces.append(own)
        out = pd.concat(pieces, ignore_index=True)
        if "phase" in out.columns:
            out["phase"] = "pickup"
        if "start_day" in out.columns:
            out["start_day"] = (out["start_day"].astype(int) - 1) % DAYS_PER_YEAR + 1
        out.to_csv(dst, index=False)
        ROLLED_FORWARD[name] = rolled
    for name in (sf.ITN_AGE_FILE, sf.ITN_SEASON_FILE):
        if name and os.path.isfile(os.path.join(INPUT_DIR, name)):
            shutil.copy(os.path.join(INPUT_DIR, name), os.path.join(PROJECTION_INPUT_DIR, name))


def load_pickup_script():
    """Import the pickup script, move its calendar to the projection, load schedules."""
    import importlib
    sf = importlib.import_module(PICKUP_SCRIPT)
    write_projection_inputs(sf)
    sf.PICKUP_START_YEAR = PROJ_START
    sf.PICKUP_END_YEAR = PROJ_END
    sf.PICKUP_DAYS = PROJ_DAYS
    sf.PHASE_DAYS = dict(sf.PHASE_DAYS, pickup=PROJ_DAYS)
    sf.INPUT_DIR = PROJECTION_INPUT_DIR
    if sf.USE_ITN_USAGE_CURVES:
        sf.ITN_USAGE = sf.load_itn_usage()
    sf.SCHEDULES = sf.load_schedules()
    sf.CM_PICKUP, _ = sf.load_case_management_pickup()
    VACCINE_ISSUES.clear()
    VACCINE_ISSUES.update(sf.VACCINE_ISSUES)
    return sf


def load_all_schedules():
    """Projection schedules, loaded by the pickup script's own loaders."""
    global SF
    SF = load_pickup_script()
    s = SF.SCHEDULES
    return {"cm": SF.CM_PICKUP, "itn_anc": s["anc"], "itn_cwc": s["cwc"], "mass": s["mass"],
            "irs": s["irs"], "smc": s["smc"], "iptp": s["iptp"], "vaccine": s["vac"]}


def describe_schedules(sched):
    print(f"\nProjection inputs, {PROJ_START}-{PROJ_END} (day 1 = 1 Jan {PROJ_START}, "
          f"{PROJ_DAYS} days), built with {PICKUP_SCRIPT}.py (same code as the pickup):")
    files = {"cm": SF.CM_FILE, "itn_anc": SF.ITN_ANC, "itn_cwc": SF.ITN_CWC,
             "irs": SF.IRS_FILE, "smc": SF.SMC_FILE, "iptp": SF.IPTP_FILE,
             "vaccine": SF.VACCINE_FILE}
    for key, label in [("cm", "Case management"), ("itn_anc", "ITN-ANC"),
                       ("itn_cwc", "ITN-CWC"), ("irs", "IRS"), ("smc", "SMC"),
                       ("iptp", "IPTp"), ("vaccine", "Vaccine (RTS,S)")]:
        v = sched[key]
        if v is None or len(v) == 0:
            print(f"  {label}: none")
            continue
        years = sorted({int(r["year"]) for r in v} if isinstance(v, list)
                       else {int(y) for y in v["year"]})
        rolled = ROLLED_FORWARD.get(files[key], [])
        note = f"; {len(rolled)} year(s) repeat {REFERENCE_YEAR}" if rolled else ""
        print(f"  {label:<16} {len(v):>3} row(s)/cohort(s), years {years[0]}-{years[-1]}{note}")
    irs = sched["irs"]
    if has_rows(irs):
        print("  IRS coverage by year (CSV / BAU arm): " +
              ", ".join(f"{int(r.year)}={r.cov:.3f}" for r in irs.itertuples()))
        zero = sorted(int(y) for y in irs.loc[(irs["year"] >= IRS_SWITCH_YEAR)
                                              & (irs["cov"] <= 0), "year"])
        if zero:
            print(f"  NOTE: IRS coverage is 0 in {zero} in {IRS_FILE}: the BAU arm has no "
                  f"IRS in those years; IRS_<c> arms still get c (rescale_irs_coverage).")
    print(f"  Generated input files: {PROJECTION_INPUT_DIR}")


def projection_custom_events():
    """Custom events the projection campaign broadcasts or listens for."""
    events = set(SF.pickup_custom_events()) | {"Received_ITN_IG2"}
    return sorted(events)


# =============================================================================
# CAMPAIGN BUILDER
# =============================================================================

def build_camp(habitat_mix=(1.0, 1.0, 1.0), scenario=None):
    """
    Complete campaign for one scenario. `scenario` is a dict from
    build_scenarios(); None means IRS continuing at the CSV coverage.
    """
    if scenario is None:
        scenario = dict(name=BAU_ARM_NAME, kind="irs", coverage=0.0,
                        irs_coverage=None, kill_key="", strategy=BAU_ARM_NAME,
                        cm_boost=0.0)

    if hasattr(camp, "reset"):
        camp.reset()
    camp.set_schema(manifest.schema_file)
    reset_mda_ids()      # MDA_Now ids are unique within ONE campaign only

    s = SCHEDULES
    sf = SF                       # pickup script (Site_Setup_Final.py) builders

    sf.add_pregnancy_tracking(camp)
    add_scale_larval_habitats(camp, df=scale_habitat_from_mix(habitat_mix),
                              start_day=1, repetitions=-1)

    # Case management: identical in every arm except the +CM_BOOSTS variants,
    # which differ only from CM_BOOST_START_YEAR.
    sf.add_case_management(camp, boost_case_management(s["cm"], scenario.get("cm_boost", 0.0)))

    # Held identical in every arm, built exactly as in the pickup.
    if has_rows(s["itn_anc"]):
        sf.add_routine_anc(camp, s["itn_anc"])
    if has_rows(s["itn_cwc"]):
        sf.add_routine_cwc(camp, s["itn_cwc"])
    if has_rows(s["mass"]):
        sf.add_mass_itn(camp, s["mass"])
    if has_rows(s["smc"]):
        sf.add_smc(camp, s["smc"])
    if s["iptp"]:
        sf.add_iptp(camp, s["iptp"])
    if s["vaccine"]:
        sf.add_vaccine(camp, s["vaccine"])

    # IRS (pickup's raw IRSHousingModification + Received_IRS). Common path:
    # rows before IRS_SWITCH_YEAR are in EVERY arm, never rescaled.
    if not has_rows(s["irs"]):
        raise ValueError("No IRS rows: check IRS_FILE and REFERENCE_YEAR.")

    sf.add_irs(camp, s["irs"][s["irs"]["year"] < IRS_SWITCH_YEAR])

    # Scenario-specific part, from 1 Jan IRS_SWITCH_YEAR.
    if scenario["kind"] == "irs":
        irs_after = s["irs"][s["irs"]["year"] >= IRS_SWITCH_YEAR]
        sf.add_irs(camp, rescale_irs_coverage(irs_after, scenario.get("irs_coverage")))
    elif scenario["kind"] == "ig2":
        add_ig2(camp, scenario["coverage"], IG2_KILL_DECAY[scenario["kill_key"]])
    elif scenario["kind"] != "none":
        raise ValueError(f"Unknown scenario kind {scenario['kind']}")

    # Register every pickup event the campaign does not already use, so that
    # emodpy writes them into Custom_Individual_Events (needed to read the
    # serialized pickup state). Must run after all other interventions.
    register_pickup_events(camp)

    fix_property_restrictions(camp.campaign_dict)
    return camp


# =============================================================================
# PICKUP STATES
# =============================================================================

def load_pickup_df(pickup_id, platform):
    """
    One row per pickup simulation: serialized state, Run_Number, xTLH and
    habitat tags. Uses the pickup's last serialized day (state-04745.dtk).
    """
    from utils_slurm import build_burnin_df

    df = build_burnin_df(pickup_id, platform, PICKUP_DAYS)

    # Validate required parameter tags
    required = ["Run_Number", "x_Temporary_Larval_Habitat"] + HABITAT_TAGS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Pickup experiment {pickup_id} lacks tag column(s) {missing}. "
            f"Present: {list(df.columns)}"
        )

    # Apply pickup filters
    for key, value in PICKUP_FILTER.items():
        if key not in df.columns:
            raise ValueError(f"PICKUP_FILTER key '{key}' is not a tag.")

        df = df[np.isclose(df[key].astype(float), float(value))]

    if df.empty:
        raise ValueError("No pickup simulations left after PICKUP_FILTER.")

    # Select calibrated parameter sets
    return select_parameter_sets(df.reset_index(drop=True))


def select_parameter_sets(df):
    """Keep only the calibrated parameter sets listed in PARAMETER_SET_FILE."""
    if not PARAMETER_SET_FILE:
        return df

    path = os.path.join(INPUT_DIR, PARAMETER_SET_FILE)
    sel = read_clean_csv(path)
    cols = {"x_Temporary_Larval_Habitat": "x_temporary_larval_habitat",
            "scale_constant": "scale_constant",
            "scale_temp_rain": "scale_temp_rain",
            "scale_water_veg": "scale_water_veg"}

    missing = [lc for lc in cols.values() if lc not in sel.columns]
    if missing:
        raise ValueError(f"{PARAMETER_SET_FILE}: missing column(s) {missing}")

    mask = np.zeros(len(df), dtype=bool)
    for _, r in sel.iterrows():
        m = np.ones(len(df), dtype=bool)
        for c, lc in cols.items():
            m &= np.isclose(df[c].astype(float).to_numpy(), float(r[lc]), rtol=1e-3)
        mask |= m

    out = df[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No pickup simulation matches any set in {PARAMETER_SET_FILE}.")

    print(f"\nParameter sets: {len(sel)} listed in {PARAMETER_SET_FILE}; "
          f"{len(out)} pickup simulations selected.")
    return out


def _find_key(obj, key):
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
    for path in paths:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return path, json.load(f)
            except (OSError, ValueError):
                continue
    return None, None


def check_pickup_consistency(pickup_df):
    """Best-effort, WARNING-only comparison of the pickup's birth settings."""
    print("\nPickup demographic consistency check (first pickup simulation):")
    try:
        sim_dir = str(pickup_df.iloc[0]["serialized_file_path"])
    except (KeyError, IndexError):
        print("  skipped: no serialized_file_path in the pickup table.")
        return

    _, cfg = _first_json([os.path.join(sim_dir, "config.json"),
                          os.path.join(os.path.dirname(sim_dir.rstrip(os.sep)), "config.json")])
    if cfg is None:
        print(f"  skipped: no readable config.json in {sim_dir}")
    else:
        params = cfg.get("parameters", cfg) if isinstance(cfg, dict) else {}
        dep, xb = params.get("Birth_Rate_Dependence"), params.get("x_Birth")
        if dep is not None and dep != BIRTH_RATE_DEPENDENCE:
            print(f"  WARNING: pickup Birth_Rate_Dependence = {dep}, this script {BIRTH_RATE_DEPENDENCE}.")
        elif dep is not None:
            print(f"  Birth_Rate_Dependence matches ({dep}).")
        if xb is not None and not np.isclose(float(xb), X_BIRTH):
            print(f"  WARNING: pickup x_Birth = {xb}, this script {X_BIRTH:g}.")

    _, demog = _first_json([
        os.path.join(sim_dir, "demographics.json"),
        os.path.join(sim_dir, "Assets", "demographics.json"),
        os.path.join(os.path.dirname(sim_dir.rstrip(os.sep)), "Assets", "demographics.json"),
    ])
    if demog is None:
        print("  note: demographics.json not found; BirthRate not compared.")
        return
    birth_rate = _find_key(demog, "BirthRate")
    if birth_rate is not None and not np.isclose(float(birth_rate), PREG_RATE_PER_MOTHER_DAY, rtol=1e-6):
        print(f"  WARNING: pickup BirthRate = {float(birth_rate):.6g}, "
              f"this script {PREG_RATE_PER_MOTHER_DAY:.6g}.")
    elif birth_rate is not None:
        print(f"  BirthRate matches ({float(birth_rate):.6g}).")


def load_pickup_events(pickup_df):
    """
    Union of Custom_Individual_Events over the pickup simulations' config.json
    files. The serialized state may wait on any of these names.
    """
    pickup_dirs = (pickup_df["serialized_file_path"].dropna().astype(str).unique())
    if len(pickup_dirs) == 0:
        raise RuntimeError("No pickup directories found in pickup_df['serialized_file_path'].")

    events, checked = set(), 0
    for pickup_dir in pickup_dirs:
        pickup_dir = os.path.abspath(pickup_dir)

        # The pickup simulation config is normally beside output/.
        candidates = [os.path.join(pickup_dir, "config.json"),
                      os.path.join(os.path.dirname(pickup_dir), "config.json")]
        config_path = next((p for p in candidates if os.path.isfile(p)), None)
        if config_path is None:
            raise FileNotFoundError(
                "Could not find config.json for pickup directory:\n"
                f"  {pickup_dir}\nChecked:\n" + "\n".join(f"  {p}" for p in candidates))

        with open(config_path, "r", encoding="utf-8") as f:
            pickup_config = json.load(f)

        params = pickup_config.get("parameters", pickup_config)
        found = params.get("Custom_Individual_Events", [])
        if not isinstance(found, list):
            raise TypeError(f"Custom_Individual_Events is not a list in:\n{config_path}")

        events.update(e for e in found if isinstance(e, str) and e.strip())
        checked += 1

    if not events:
        raise RuntimeError("No pickup custom events found. Aborting submission.")

    return sorted(events), checked


def update_simulation(simulation, pair, df):
    """
    Point one scenario simulation at one pickup state and one scenario.
    pair = (pickup row index, scenario index).

    Inherits from the pickup simulation: serialized state, Run_Number, xTLH and
    habitat mix, so arms of the same pickup row are paired.
    """
    x, k = pair
    row = df.iloc[x]
    scenario = SCENARIOS[k]

    filenames = row["Serialized_Population_Filenames"]
    if isinstance(filenames, str):
        filenames = [filenames]

    simulation.task.set_parameter("Serialized_Population_Filenames", list(filenames))
    simulation.task.set_parameter("Serialized_Population_Path",
                                  os.path.join(row["serialized_file_path"], "output"))

    seed = int(row["Run_Number"])
    simulation.task.set_parameter("Run_Number", seed)

    simulation.task.set_parameter("Birth_Rate_Dependence", BIRTH_RATE_DEPENDENCE)
    simulation.task.set_parameter("x_Birth", X_BIRTH)
    remove_stale_params(simulation.task.config)

    xtlh = float(row["x_Temporary_Larval_Habitat"])
    sc_const = float(row["scale_constant"])
    sc_rain = float(row["scale_temp_rain"])
    sc_veg = float(row["scale_water_veg"])

    simulation.task.set_parameter("x_Temporary_Larval_Habitat", xtlh)

    simulation.task.create_campaign_from_callback(
        partial(build_camp, habitat_mix=(sc_const, sc_rain, sc_veg), scenario=scenario))

    # Safety layer: keep the pickup events in the config after pre_creation.
    install_event_patch(simulation)

    irs_cov = scenario.get("irs_coverage")
    return {
        "Run_Number": seed,
        "pickup_sim_index": int(x),
        "x_Temporary_Larval_Habitat": xtlh,
        "scale_constant": sc_const,
        "scale_temp_rain": sc_rain,
        "scale_water_veg": sc_veg,
        "scenario": scenario["name"],
        "scenario_kind": scenario["kind"],
        "ig2_coverage": float(scenario["coverage"]),
        "irs_coverage": float(irs_cov) if irs_cov is not None else np.nan,
        "ig2_kill_key": scenario["kill_key"],
        "strategy": scenario.get("strategy", scenario["name"]),
        "cm_boost": float(scenario.get("cm_boost", 0.0)),
        "projection_start_year": PROJ_START,
    }


# =============================================================================
# REPORTS
# =============================================================================

def recorder_events():
    """The pickup's recorded events (Site_Setup_Final.pickup_recorded_events) + IG2."""
    events = list(SF.pickup_recorded_events())
    if SF._bednet_accepts_event():
        events += ["Received_ITN_ANC", "Received_ITN_CWC", "Received_ITN_CWC_Catchup",
                   "Received_ITN_IG2"]
    return list(dict.fromkeys(events))


def add_net_use_counter(task):
    """Add ReportEventCounter for Bednet_Using events."""
    from emodpy_malaria.reporters import builtin

    event_list = list(NET_USE_EVENTS)

    builtin.add_report_event_counter(
        task,
        manifest,
        event_trigger_list=event_list,
        start_day=1,
        end_day=PROJ_DAYS,
        node_ids=[NODE_ID],
        min_age_years=0,
        max_age_years=EVENT_MAX_AGE_YEARS,
        filename_suffix="NetUse",
    )

    print(f"  Added ReportEventCounter for net-use events: {event_list}")


def add_reports(task):
    add_event_recorder(task, event_list=recorder_events(), start_day=1,
                       end_day=PROJ_DAYS, node_ids=[NODE_ID],
                       min_age_years=EVENT_MIN_AGE_YEARS, max_age_years=EVENT_MAX_AGE_YEARS)

    if ADD_NET_USE_COUNTER:
        add_net_use_counter(task)

    for i in range(PROJ_YEARS):
        start_day = i * DAYS_PER_YEAR
        call_supported(
            add_malaria_summary_report, task, "Summary report",
            manifest=manifest,
            start_day=start_day,
            end_day=start_day + DAYS_PER_YEAR,
            reporting_interval=REPORT_INTERVAL,
            age_bins=REPORT_AGE_BINS,
            max_number_reports=REPORT_MAX_NUMBER,
            parasite_detection_threshold=PARASITE_DETECTION_THRESHOLD,
            pretty_format=True,
            filename_suffix=f"Monthly_U5_{PROJ_START + i}",
        )
    return task


def full_experiment_name():
    return f"{os.getlogin()}_{EXPERIMENT_NAME}"


# =============================================================================
# COMMAND LINE / SUBMISSION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="UWR IRS-coverage and IG2 replacement scenarios.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs, print scenarios, write two campaign previews, stop.")
    parser.add_argument("--test-one", action="store_true",
                        help="Submit the first pickup state x ALL scenarios (name ends _TEST).")
    parser.add_argument("--allow-excluded-vaccine-cohorts", action="store_true",
                        help="Submit even though some vaccine cohorts failed validation.")
    return parser.parse_args()


def submit_scenarios(platform, pickup_df, climate_dir):
    global PICKUP_CUSTOM_EVENTS

    n_states = len(pickup_df)
    n_scen = len(SCENARIOS)
    n_sims = n_states * n_scen

    print(f"\nSimulations: {n_states} pickup states x {n_scen} scenarios = {n_sims}")

    # ---------------------------------------------------------
    # 1. Load custom events from the pickup simulation configs
    # ---------------------------------------------------------
    PICKUP_CUSTOM_EVENTS, config_files_checked = load_pickup_events(pickup_df)
    pickup_events = set(PICKUP_CUSTOM_EVENTS)

    print(f"\n[EVENT CHECK] Read {config_files_checked} pickup configs.")
    print(f"[EVENT CHECK] Unique pickup events: {len(PICKUP_CUSTOM_EVENTS)} "
          f"({len(pickup_family_events())} in the families that must be registered)")

    for event in ("ANC_Visit_2025", "CWC_Visit_2025", "MDA_Now_1624"):
        print(f"[EVENT CHECK] {event}: {'FOUND' if event in pickup_events else 'NOT FOUND'}")

    if not {"ANC_Visit_2025", "CWC_Visit_2025"}.issubset(pickup_events):
        raise RuntimeError(
            "The pickup event registry is missing expected historical "
            "events. Check the pickup paths before submitting scenarios."
        )

    # ---------------------------------------------------------
    # 2. Preflight: the built campaign must register every pickup event
    # ---------------------------------------------------------
    for sc in SCENARIOS:
        built = build_camp(habitat_mix=(1.0, 1.0, 1.0), scenario=sc)
        n_ok = verify_pickup_events(built)
    print(f"[EVENT CHECK] Every scenario campaign registers all {n_ok} required pickup events.")
    print(f"[EVENT CHECK] MDA_Now ids used inside one campaign: {len(_MDA_USED_IDS)} "
          f"(counter restarts for every build)")

    # ---------------------------------------------------------
    # 3. Build the scenario task AFTER loading pickup events
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 4. Create the pickup-state x scenario sweep
    # ---------------------------------------------------------
    builder = SimulationBuilder()
    pairs = list(product(range(n_states), range(n_scen)))
    builder.add_sweep_definition(partial(update_simulation, df=pickup_df), pairs)

    experiment_name = full_experiment_name()
    print(f"\nSubmitting under experiment name:\n  {experiment_name}")

    # ---------------------------------------------------------
    # 5. Submit the experiment
    # ---------------------------------------------------------
    experiment = Experiment.from_builder(builder, task, name=experiment_name)
    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")
    print(f"Simulations: {n_sims}")
    print("\nBefore trusting the run, open one simulation's config.json and confirm "
          "Custom_Individual_Events lists ANC_Visit_2025, CWC_Visit_2025 and the "
          "pickup's MDA_Now_* ids.")

    return experiment


def general_sim(selected_platform, pickup_id, climate_dir, test_one=False):
    global EXPERIMENT_NAME

    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    pickup_df = load_pickup_df(pickup_id, platform)

    if test_one:
        pickup_df = pickup_df.head(1).reset_index(drop=True)
        if not EXPERIMENT_NAME.endswith("_TEST"):
            EXPERIMENT_NAME += "_TEST"
        print("\n*** TEST MODE: first pickup state x all scenarios ***")

    print(f"\nPickup experiment: {pickup_id}")
    print(f"Pickup states selected: {len(pickup_df)}")
    print(f"Experiment name: {full_experiment_name()}")

    check_pickup_consistency(pickup_df)
    return submit_scenarios(platform, pickup_df, climate_dir)


def campaign_counts(c):
    """Counts of IRS / IG2 / other net objects in a built campaign (preview check)."""
    text = json.dumps(c.campaign_dict)
    return {"IRS events": text.count('"class": "IRSHousingModification"'),
            "IG2 nets": text.count('"Intervention_Name": "ITN_IG2"')}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    SCHEDULES = load_all_schedules()
    SCENARIOS = build_scenarios()

    if USE_ITN_USAGE_CURVES or IG2_USE_CURVES:
        ITN_USAGE = load_itn_usage()
        validate_itn_usage(SCHEDULES)

    if not PROJ_START < IRS_SWITCH_YEAR <= PROJ_END:
        sys.exit(f"IRS_SWITCH_YEAR {IRS_SWITCH_YEAR} must be after {PROJ_START} "
                 f"and no later than {PROJ_END}.")
    early = [y for y in IG2_CAMPAIGN_YEARS if y < IRS_SWITCH_YEAR]
    if early:
        sys.exit(f"IG2_CAMPAIGN_YEARS {early} are before IRS_SWITCH_YEAR "
                 f"{IRS_SWITCH_YEAR}; IG2 must not start before the switch.")

    describe_schedules(SCHEDULES)
    describe_scenarios(SCENARIOS)
    validate_demographic_settings()

    print("\nDemographic settings (must match the pickup):")
    print(f"  Birth_Rate_Dependence : {BIRTH_RATE_DEPENDENCE}")
    print(f"  BirthRate per mother/d: {PREG_RATE_PER_MOTHER_DAY:g}   x_Birth: {X_BIRTH:g}")
    print(f"  crude death rate      : {CRUDE_DEATH_RATE:g} per 1,000 / year")
    print(f"  vector behaviour      : {'ON' if APPLY_VECTOR_BEHAVIOUR else 'OFF'}")

    if "PREGNANC" not in BIRTH_RATE_DEPENDENCE:
        msg = (f"BIRTH_RATE_DEPENDENCE = {BIRTH_RATE_DEPENDENCE}: 'Pregnant' and "
               f"'GaveBirth' never fire. It must match the pickup.")
        if args.dry_run:
            print(f"\n  WARNING: {msg}")
        else:
            sys.exit(msg)

    if VACCINE_ISSUES:
        msg = (f"Vaccine cohort(s) {sorted(VACCINE_ISSUES)} failed validation and would "
               f"get NO vaccine campaign. Correct {VACCINE_FILE}, or pass "
               f"--allow-excluded-vaccine-cohorts.")
        if args.dry_run:
            print(f"\n  WARNING: {msg}")
        elif args.allow_excluded_vaccine_cohorts:
            print(f"\n  NOTE: proceeding with excluded vaccine cohort(s) {sorted(VACCINE_ISSUES)}.")
        else:
            sys.exit(msg)

    if not has_rows(SCHEDULES["irs"]):
        msg = (f"No IRS rows available ({IRS_FILE}, REFERENCE_YEAR {REFERENCE_YEAR}): "
               f"the IRS arms and the common 2026 IRS path cannot be built.")
        if args.dry_run:
            print(f"\n  WARNING: {msg}")
        else:
            sys.exit(msg)

    climate_dir = resolve_climate_dir()
    validate_climate(climate_dir)

    if args.dry_run:
        print("\nReports:")
        print(f"  Malaria Summary Reports {PROJ_START}-{PROJ_END}, one per year "
              f"({REPORT_MAX_NUMBER} x {REPORT_INTERVAL}d), age bins {REPORT_AGE_BINS}")
        print(f"  Event recorder ages {EVENT_MIN_AGE_YEARS}-{EVENT_MAX_AGE_YEARS}")
        print(f"  Net-use counter: {'ON' if ADD_NET_USE_COUNTER else 'OFF'}")

        print("\nPreview check (habitat mix placeholder 1.0, 1.0, 1.0):")
        previews = ([s for s in SCENARIOS if s["kind"] == "irs"][:1]
                    + [s for s in SCENARIOS if s["kind"] == "ig2"][:1])
        for sc in previews:
            build_camp(habitat_mix=(1.0, 1.0, 1.0), scenario=sc)
            out = f"campaign_preview_{sc['name']}.json"
            camp.save(out)
            print(f"  {sc['name']:<14} {campaign_counts(camp)}  -> {out}")

        # The drug-campaign ids restart for every build, so building all arms in a
        # row must never exhaust them (this was the failure in the full run).
        for sc in SCENARIOS:
            build_camp(habitat_mix=(1.0, 1.0, 1.0), scenario=sc)
        print(f"\nBuilt all {len(SCENARIOS)} scenario campaigns in a row; "
              f"MDA_Now ids used in the last one: {len(_MDA_USED_IDS)}.")

        events = recorder_events()
        print(f"\nEvent recorder will record {len(events)} events.")
        print("Pickup event registration is skipped in a dry run (the pickup configs "
              "are only read at submission).")
        print(f"\nExpected: IRS_<c> arms have IRS in every year and no IG2 nets. The IG2 "
              f"preview has IRS only for years before {IRS_SWITCH_YEAR} (fewer IRS "
              f"events than the IRS_<c> arms, not zero) and IG2 nets > 0.")
        print("Dry run OK. Nothing submitted.")
        sys.exit(0)

    pickup_id = PICKUP_EXPERIMENT_ID.strip()
    if not pickup_id or pickup_id.startswith("PUT-"):
        sys.exit("Set PICKUP_EXPERIMENT_ID at the top of the script to the finished "
                 "pickup experiment ID.")

    import emod_malaria.bootstrap as dtk

    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)

    general_sim("SLURM_LOCAL", pickup_id, climate_dir, test_one=args.test_one)