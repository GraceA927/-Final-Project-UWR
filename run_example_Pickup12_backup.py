################################################################################
# Upper West Region EMOD Malaria Model -- PICKUP 2013-2025
#
# AIM 1: Adapt and calibrate EMOD to the Upper West Region of Ghana.
#
#   1963 ------------------------ 2012 | 2013 ---------- 2025
#          50-year burn-in (CM only)   |  pickup: CM, ITN, IRS, SMC, vaccine
#                                      ^
#                     serialized state (day 18,250) read here
#
# USAGE
#   1. Set BURNIN_EXPERIMENT_ID below.
#   2. Check inputs and preview the campaign without submitting anything:
#          python run_pickup_UWR.py --dry-run
#   3. Submit one xTLH batch (0-4):
#          python run_pickup_UWR.py --batch 0
#      Omit --batch to submit all 5 batches back-to-back.
#
# CALENDAR
#   Day 1 = 1 Jan 2013, flat 365-day years (no leap days),
#   13 years = 4,745 days.
#
#   2013 -> day 1
#   2014 -> day 366
#   2015 -> day 731
#   ...
#   2025 -> day 4,381
#   31 Dec 2025 -> day 4,745
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
#   checked against the result. Rows outside 2013-2025 are dropped and
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
#     every selected burn-in state x (this batch's xTLH) x PICKUP_HABITAT_MIXES
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
#   demographics.csv, vector.csv, intervention_CM.csv (phase == pickup),
#   interventions_ITN.csv (+ _ITN_age, _ITN_season), interventions_IRS.csv,
#   interventions_SMC.csv, interventions_malariaVaccine.csv
#   pcr_prevalence.csv is calibration reference data, not an input here.
################################################################################

import argparse
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
EXPERIMENT_NAME = "UWR_AIM1_Pickup_2013_2025_WECM"
NODE_ID = 1
BIRTH_RATE = 31.0          # must match the burn-in
MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
CLIMATE_ROOT = os.path.join(PROJECT_DIR, "climate", "pickup")
CLIMATE_SUBDIR = None      # e.g. "emod_weather_upper_west"; None = auto-detect

VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"
#CM_FILE = "intervention_CMBN.csv" # this has earlier CM
CM_FILE = "intervention_CM.csv"
ITN_FILE = "interventions_ITN.csv"
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"
IRS_FILE = "intervention_IRS.csv"
SMC_FILE = "intervention_SMC.csv"
VACCINE_FILE = "intervention_malariaVaccine.csv"


################################################################################
# CALENDAR
################################################################################

DAYS_PER_YEAR = 365
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_DAYS = (BURNIN_END_YEAR - BURNIN_START_YEAR + 1) * DAYS_PER_YEAR   # 18,250

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1                    # 13
PICKUP_DAYS = PICKUP_YEARS * DAYS_PER_YEAR                                # 4,745

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
BURNIN_EXPERIMENT_ID = 'c522a73f-90c8-4407-a776-1993bd2083a3'

BURNIN_FILTER = {
    # Pinned to ONE burn-in combination so this sweep only varies the
    # PICKUP's own habitat mix below, not which burn-in state feeds it.
    # 100.0 is one of the burn-in's own XTLH_RANGE values
    # (10^1.85..10^2.15, 5 points) -- adjust if this exact value isn't
    # actually in your burn-in's tags (check BURNIN_FILTER's own error
    # message, which lists the real tag columns, if this doesn't match).
    "scale_temp_rain": 10.0,
    "scale_water_veg": 0.5,
    "x_Temporary_Larval_Habitat": 100.0,
}


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

# HABITAT-MIX SWEEP (cheap): xTLH pinned to a single value so this run
# isolates the effect of the mix RATIO on seasonal timing, not amplitude.
# Re-widen PICKUP_XTLH_RANGE back to a real sweep once a mix's shape
# looks right and you're tuning amplitude/level instead.
PICKUP_XTLH_RANGE = [100.0]

# Every seasonality fit so far (mix (1.0, 10.0, 0.5): temp_rain 20x
# water_veg) scored a NEGATIVE correlation and barely moved across
# xTLH -- consistent with TEMPORARY_RAINFALL (fast-draining, fires on
# any rain event) dominating over WATER_VEGETATION (persistent, slow
# to build), producing an early-year case peak instead of the observed
# Sep-Nov peak. This grid shifts that ratio from the current baseline
# toward water_veg-dominant, to test whether that's really the cause
# before spending a full run on it. Values are a first-pass grid, not
# calibrated -- narrow around whichever ratio comes out on top.
PICKUP_HABITAT_MIXES = [
    (1.0, 10.0, 0.5),   # current baseline -- temp_rain-dominant (for comparison)
    (1.0, 5.0, 1.0),    # 5:1 temp_rain:water_veg
    (1.0, 2.0, 2.0),    # roughly balanced
    (1.0, 1.0, 5.0),    # water_veg-dominant (5:1)
    (1.0, 0.5, 10.0),   # strongly water_veg-dominant (20:1)
]

# Write the end-of-pickup (day 4,745 = 31 Dec 2025) state for scenario runs.
# Mask 0 on write saves everything, including this pickup's larval habitats.
WRITE_PICKUP_STATE = True

# Microscopy-like detection threshold for the summary reports (parasites/uL)
PARASITE_DETECTION_THRESHOLD = 40


################################################################################
# INTERVENTION DEFAULTS (used when a column is absent from the CSV)
################################################################################

CM_DRUGS = {
    "AL": ["Artemether", "Lumefantrine"],
    "SP": ["Sulfadoxine", "Pyrimethamine"],
    "CQ": ["Chloroquine"],
}
SEVERE_TRIGGERS = ["NewSevereCase"]

ITN_DEFAULTS = dict(
    blocking_initial=0.9, blocking_decay=730,
    killing_initial=0.6, killing_decay=1460,
    discard_k=None, discard_l=None,
)
IRS_DEFAULTS = dict(
    killing_initial=0.7, killing_box=0, killing_decay=180,
    repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    repetitions=1, timesteps_between_repetitions=365,
)

# --------------------------------------------------------------------------
# IRS PRODUCTS
# --------------------------------------------------------------------------
# Resistance is NOT modelled genetically, so the insecticide name does
# nothing in EMOD by itself. Each product is represented by its own
# effect and waning (exponential decay; effect halves after ~0.69 x decay).
#
# These are STARTING VALUES from typical residual-efficacy ranges, not
# calibrated estimates -- replace them with values from your literature
# review or local bioassays.
#
# How the `insecticide` column is read:
#   "A; B"  -> different products in different districts that year. The
#              row's coverage is split equally between them (warning shown);
#              better: give each product its own row with its own coverage.
#   "A + B" -> one combined application. Uses the entry below if the
#              combination is listed (e.g. Fludora Fusion); otherwise the
#              components are combined: killing = 1 - (1-k1)(1-k2), and the
#              longer box/decay of the two.
# Explicit killing_*/repelling_* values in the CSV row override this table;
# leave those cells blank to use the product values.
# --------------------------------------------------------------------------
IRS_PRODUCTS = {
    # Actellic 300CS (organophosphate), ~4-6+ months residual
    "pirimiphos methyl": dict(
        killing_initial=1.0, killing_box=0, killing_decay=180,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
    # Carbamate, short residual ~2-3 months
    "propoxur": dict(
        killing_initial=1.0, killing_box=0, killing_decay=60,
        repelling_initial=0.0, repelling_box=0, repelling_decay=60,
    ),
    # SumiShield 50WG (neonicotinoid), slow-acting, long residual ~6-8+ months
    "clothianidin": dict(
        killing_initial=1.0, killing_box=0, killing_decay=240,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
    # Fludora Fusion (co-formulated); pyrethroid part largely lost to resistance
    "clothianidin + deltamethrin": dict(
        killing_initial=1.0, killing_box=0, killing_decay=240,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
    # VECTRON T500 (meta-diamide), ~6-12 months residual
    "broflanilide": dict(
        killing_initial=1.0, killing_box=0, killing_decay=270,
        repelling_initial=0.0, repelling_box=0, repelling_decay=90,
    ),
}

IRS_PARAM_COLUMNS = {   # internal name -> CSV column
    "killing_initial": "killing_initial_effect",
    "killing_box": "killing_box_duration",
    "killing_decay": "killing_decay_time_constant",
    "repelling_initial": "repelling_initial_effect",
    "repelling_box": "repelling_box_duration",
    "repelling_decay": "repelling_decay_time_constant",
}


def normalise_product(name):
    """'Pirimiphos-methyl' -> 'pirimiphos methyl'; combos sorted: 'a + b'."""
    parts = [
        " ".join(p.replace("-", " ").replace("_", " ").lower().split())
        for p in str(name).split("+")
    ]
    return " + ".join(sorted(p for p in parts if p))


def product_params(name):
    """Effect parameters for one product or '+' combination."""
    key = normalise_product(name)
    if key in IRS_PRODUCTS:
        return IRS_PRODUCTS[key], "table"

    parts = key.split(" + ")
    missing = [p for p in parts if p not in IRS_PRODUCTS]
    if missing:
        raise ValueError(
            f"{IRS_FILE}: insecticide '{name}' has no entry for {missing} in IRS_PRODUCTS. "
            "Add it with its killing/repelling values."
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


SMC_DEFAULTS = dict(age_min=0.25, age_max=5.0, drug_code="SPA")
SMC_DRUG_ALIASES = {"SPAQ": "SPA", "SP+AQ": "SPA", "SP-AQ": "SPA"}
VACCINE_DEFAULTS = dict(
    age_min=0.42, age_max=2.0, initial_effect=0.8,
    box_duration=0, decay=592, vaccine_type="AcquisitionBlocking",
)
# Catch-up ceiling: a child who misses a dose's original target age should
# stay eligible for it later, not be excluded outright. WHO guidance allows
# catch-up vaccination for eligible children up to 5 years old, so every
# dose's upper age bound is this ceiling rather than a narrow window around
# its target age. Each dose's LOWER bound stays at its own target age (the
# clinical minimum age for that dose); the required minimum interval since
# the previous dose is enforced separately, by the trigger+delay chain in
# add_vaccine(), which is timed off the previous dose's ACTUAL receipt --
# so a late (caught-up) dose pushes the next one later too, rather than
# the series restarting or a dose being skipped.
VACCINE_CATCHUP_MAX_AGE_MONTHS = 59.0   # WHO catch-up ceiling: under 5 years


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


def add_pickup_day(df, label, default_day_note=None):
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
            note = default_day_note or "using 1 Jan of each year."
            print(f"  WARNING {label}: no day_of_year/start_day/month column; {note}")

    if "month" in df.columns and ("day_of_year" in df.columns or "start_day" in df.columns):
        stated = df["month"].map(month_number)
        derived = doy.map(month_of_doy)
        for i in df.index[stated != derived]:
            print(
                f"  WARNING {label}: year {year[i]} says month {df.at[i, 'month']} but start_day "
                f"gives month {derived[i]}; using start_day."
            )

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
    required = [
        "phase", "duration", "trigger", "age_min", "age_max",
        "coverage", "seek", "rate", "drug", "year",
    ]
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


def load_generic(filename, default_day_note=None):
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        return None, 0
    return add_pickup_day(
        keep_pickup_phase(keep_node(read_clean_csv(path))), filename, default_day_note=default_day_note
    )


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
    df["reps"] = pick(df, ["repetitions"], d["repetitions"]).astype(int)
    df["rep_gap"] = pick(df, ["timesteps_between_repetitions"], d["timesteps_between_repetitions"]).astype(int)

    if "insecticide" not in df.columns:
        df["insecticide"] = ""

    # "A; B" in one row -> one row per product, coverage split equally
    rows = []
    for _, r in df.iterrows():
        products = [p.strip() for p in str(r["insecticide"]).split(";") if p.strip()]
        if len(products) > 1:
            print(
                f"  WARNING {IRS_FILE}: {int(r['year'])} lists {len(products)} products in one row; "
                f"splitting coverage {r['cov']:.3f} equally. Give each product its own row "
                "with its own coverage if you know the split."
            )
        for prod in products or [""]:
            r2 = r.copy()
            r2["insecticide"] = prod
            r2["cov"] = r["cov"] / max(len(products), 1)
            rows.append(r2)
    df = pd.DataFrame(rows).reset_index(drop=True)

    # Effect parameters: explicit CSV value > product table > IRS_DEFAULTS
    sources = []
    for i, r in df.iterrows():
        if r["insecticide"]:
            params, source = product_params(r["insecticide"])
        else:
            params, source = d, "IRS_DEFAULTS (no insecticide given)"
        overridden = []
        for internal, col in IRS_PARAM_COLUMNS.items():
            if col in df.columns and pd.notna(r[col]):
                df.at[i, internal] = float(r[col])
                if not np.isclose(float(r[col]), params[internal]):
                    overridden.append(col)
            else:
                df.at[i, internal] = float(params[internal])
        sources.append(source + (f"; CSV overrides {overridden}" if overridden else ""))
    df["param_source"] = sources
    df["iv_name"] = [
        "IRS_" + (normalise_product(p).replace(" + ", "_plus_").replace(" ", "_") or "default")
        for p in df["insecticide"]
    ]

    check_fraction(df["cov"], IRS_FILE)
    year_total = df.groupby("year")["cov"].sum()
    if (year_total > 1).any():
        raise ValueError(
            f"{IRS_FILE}: product coverages add up to more than 1 in year(s) "
            f"{list(year_total[year_total > 1].index)}."
        )
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
    df, dropped = load_generic(
        VACCINE_FILE,
        default_day_note=(
            "using 1 Jan of each year as the start of that year's rolling, "
            "birth-triggered D1 eligibility window; every child is assumed "
            "to receive each dose exactly at its target age (D2+ fire the "
            "exact required interval after the previous dose's actual "
            "receipt), not on this or any other fixed calendar date."
        ),
    )
    if df is None or df.empty:
        return df, dropped
    d = VACCINE_DEFAULTS
    df["cov"] = pick(df, ["coverage"], label=VACCINE_FILE).astype(float)
    df["dose"] = pick(df, ["dose"], "D1").astype(str).str.upper()

    if "target_age_months" in df.columns:
        # Each dose's minimum age is its own target age (e.g. D1 at 6
        # months, D2 at 7, D3 at 9, a booster D4 at 18) -- but the upper
        # bound is the program-wide catch-up ceiling, not a narrow window,
        # so a child who misses the original target age stays eligible for
        # that dose later instead of being excluded (see VACCINE_CATCHUP_
        # MAX_AGE_MONTHS above).
        df["target_age_months"] = df["target_age_months"].astype(float)
        df["amin"] = df["target_age_months"] / 12.0
        df["amax"] = VACCINE_CATCHUP_MAX_AGE_MONTHS / 12.0
    else:
        df["target_age_months"] = np.nan
        df["amin"] = pick(df, ["age_min", "agemin"], d["age_min"]).astype(float)
        df["amax"] = pick(df, ["age_max", "agemax"], d["age_max"]).astype(float)

    df["initial_effect"] = pick(df, ["initial_effect", "efficacy"], d["initial_effect"]).astype(float)
    df["box_duration"] = pick(df, ["box_duration"], d["box_duration"]).astype(float)
    df["decay"] = pick(df, ["decay", "decay_time"], d["decay"]).astype(float)
    check_fraction(df["cov"], VACCINE_FILE)

    # Model the WHO/Ghana schedule as a true multi-dose series: a child is
    # only eligible for dose N after receiving dose N-1. The CSV's yearly
    # `coverage` governs only who STARTS the series (D1) -- once a child
    # has D1, they are guaranteed to complete D2, D3 and D4 too. Delays
    # (e.g. a late D1) push the rest of that child's series later with it
    # -- a booster nominally at 18 months can land at 24 for a child who
    # started late -- but no dose is ever skipped once the series begins.
    # `delay_days` is the minimum required interval to the next dose,
    # measured from THIS dose's target age gap.
    df = df.sort_values(["year", "target_age_months"]).reset_index(drop=True)
    df["prev_dose"] = None
    df["delay_days"] = 0.0
    for _, group in df.groupby("year"):
        idx = list(group.index)
        for pos in range(1, len(idx)):
            i, prev_i = idx[pos], idx[pos - 1]
            df.at[i, "prev_dose"] = df.at[prev_i, "dose"]
            gap_months = df.at[i, "target_age_months"] - df.at[prev_i, "target_age_months"]
            df.at[i, "delay_days"] = max(gap_months * (DAYS_PER_YEAR / 12.0), 1.0)

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
        print(
            f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4}) for {r.duration}d  {r.trigger:<16} "
            f"ages {r.age_min:g}-{r.age_max:g}  cov {r.effective_coverage:.3f}  rate {r.rate:g}  {r.drug}{clip}"
        )
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
                    extra = (
                        f"  block {r.blocking_initial:.3f}/{r.blocking_decay:g}d  "
                        f"kill {r.killing_initial:.3f}/{r.killing_decay:g}d  "
                        f"age curve: {'yes' if age else 'none'}"
                    )
                    cov_label = f"cov {r.cov:.3f}"
                elif key == "irs":
                    extra = (
                        f"  {r.insecticide or '(none)'}: kill {r.killing_initial:.2f}, "
                        f"decay {r.killing_decay:g}d  [{r.param_source}]"
                    )
                    cov_label = f"cov {r.cov:.3f}"
                elif key == "smc":
                    extra = f"  round {r.round}  ages {r.amin:g}-{r.amax:g}  {r.drug_code}"
                    cov_label = f"cov {r.cov:.3f}"
                else:
                    if pd.isna(r.prev_dose) or r.prev_dose is None:
                        extra = (
                            f"  dose {r.dose} ({int(r.year)})  exactly {r.target_age_months:.0f}mo from birth  "
                            f"[age band {r.amin:.3f}-{r.amax:.3f}y, safety bound only]"
                        )
                        cov_label = f"cov {r.cov:.3f}"   # applied: this is D1's actual uptake fraction
                    else:
                        extra = (
                            f"  dose {r.dose} ({int(r.year)})  exactly {r.delay_days:.0f}d after "
                            f"{r.prev_dose} ({int(r.year)})  [age band {r.amin:.3f}-{r.amax:.3f}y, safety bound only]  "
                            "(full continuation: 100% of prior-dose recipients)"
                        )
                        cov_label = f"cov n/a (CSV {r.cov:.3f} not applied)"
                print(f"    {fmt_day(r.pickup_day)} (day {r.pickup_day:>4})  {cov_label}{extra}")
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
    found = [
        d for d in sorted(os.listdir(CLIMATE_ROOT))
        if os.path.isfile(os.path.join(CLIMATE_ROOT, d, "air_temperature.bin"))
    ]
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
            print(
                f"  WARNING {var}: OriginalDataYears is '{years}', expected it to start in "
                f"{PICKUP_START_YEAR}. Day 1 of the weather must be 1 Jan {PICKUP_START_YEAR}."
            )
    print(f"Climate: {climate_dir}")


################################################################################
# SITE / VECTOR INPUT
################################################################################

DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)
VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)

for p in [
    INPUT_DIR, DEMOGRAPHICS_PATH, VECTOR_PATH, os.path.join(INPUT_DIR, CM_FILE),
    manifest.eradication_path, manifest.schema_file,
]:
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
    return {"name": SITE_NAME, "pop": int(row["pop"]), "lat": float(row["lat"]), "lon": float(row["lon"])}


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

    # Save the end-of-2025 state (day 4,745) so calibration follow-ups and
    # scenario runs can start from 1 Jan 2026 (writes output/state-04745.dtk
    # per simulation)
    if WRITE_PICKUP_STATE:
        config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
        config.parameters.Serialization_Time_Steps = [PICKUP_DAYS]
        config.parameters.Serialization_Mask_Node_Write = 0
        config.parameters.Serialization_Precision = "REDUCED"

    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"
    return config


################################################################################
# DEMOGRAPHICS -- identical to the burn-in
################################################################################

def build_demog():
    site = _read_site_definition()
    demog = Demographics.from_template_node(lat=site["lat"], lon=site["lon"], pop=site["pop"], name=site["name"])
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
            if key in ["Node_Property_Restrictions", "Property_Restrictions", "Property_Restrictions_Within_Node"]:
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
            print(
                f"  WARNING {label}: {fn.__name__} in this emodpy_malaria does not accept "
                f"{dropped}; those settings are NOT applied."
            )
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
    # The insecticide name is not passed to EMOD: that needs insecticide
    # resistance set up in the config, which this model does not use. Each
    # product gets its own Intervention_Name so products sprayed in the same
    # round do not overwrite each other, while a re-spray with the same
    # product the next year replaces the old, decayed one.
    from emodpy_malaria.interventions.irs import add_scheduled_irs_housing_modification
    for r in df.itertuples():
        call_supported(
            add_scheduled_irs_housing_modification, campaign, "IRS",
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
    # SIMPLIFYING ASSUMPTION for this run: every child receives every dose
    # exactly at its expected age -- nobody arrives late, so there is no
    # catch-up to model. D1 is BIRTH-TRIGGERED every year: a child becomes
    # eligible exactly `target_age_months` after their own birth, at that
    # year's coverage. D2+ stay triggered off the ACTUAL receipt of the
    # previous dose, delayed by the exact required interval -- since D1
    # always lands exactly on time under this assumption, the whole chain
    # (D1 -> D2 -> D3 -> D4) lands exactly on its nominal target ages too.
    #
    # Consequence worth knowing: because there's no catch-up mechanism,
    # only children actually BORN during a year that has vaccine data
    # (2024, 2025, ...) can ever enter the series -- a child already alive
    # and past D1's target age before the vaccine's introduction year has
    # no way to receive it "exactly on time" and is simply never
    # vaccinated in this model. That's an intended trade-off of the
    # exact-timing assumption, not an oversight -- say the word if you'd
    # rather bring back the introduction-year catch-up sweep for that
    # population (at the cost of those children not being exactly on time).
    #
    # Every dose's trigger/broadcast event name is qualified by YEAR
    # (Received_Vaccine_D1_2024, _2025, ...) so 2024's and 2025's cohorts
    # can't cross-trigger each other's parameters.
    from emodpy_malaria.interventions.vaccine import add_triggered_vaccine

    for r in df.itertuples():
        year = int(r.year)
        this_event = f"Received_Vaccine_{r.dose}_{year}"
        common = dict(
            intervention_name=f"Vaccine_{r.dose}_{year}",
            target_age_min=float(r.amin),
            target_age_max=float(r.amax),
            vaccine_type=VACCINE_DEFAULTS["vaccine_type"],
            vaccine_initial_effect=float(r.initial_effect),
            vaccine_box_duration=float(r.box_duration),
            vaccine_decay_time_constant=float(r.decay),
            broadcast_event=this_event,
        )

        if pd.isna(r.prev_dose) or r.prev_dose is None:
            call_supported(
                add_triggered_vaccine, campaign, "Vaccine",
                trigger_condition_list=["Births"],
                delay_period_constant=float(r.target_age_months) * DAYS_PER_MONTH,
                start_day=int(r.pickup_day),
                listening_duration=DAYS_PER_YEAR,
                demographic_coverage=float(r.cov),
                **common,
            )
        else:
            # Full continuation: any child who received the previous dose
            # is guaranteed this one too, exactly on schedule -- dropout
            # only happens if a child never starts the series at all.
            call_supported(
                add_triggered_vaccine, campaign, "Vaccine",
                trigger_condition_list=[f"Received_Vaccine_{r.prev_dose}_{year}"],
                delay_period_constant=float(r.delay_days),
                demographic_coverage=1.0,
                start_day=1,
                listening_duration=-1,
                **common,
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
    simulation.task.set_parameter("Serialized_Population_Path", os.path.join(row["serialized_file_path"], "output"))

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
    return {
        "scale_constant": float(mix[0]),
        "scale_temp_rain": float(mix[1]),
        "scale_water_veg": float(mix[2]),
    }


def set_param(simulation, param, value):
    return simulation.task.set_parameter(param, value)


################################################################################
# REPORTS
################################################################################

def add_reports(task):
    # parasite_detection_threshold=40 parasites/uL makes "PfPR by Age Bin"
    # microscopy-like, matching the survey test (as in the SMC-Ghana project).
    for i in range(PICKUP_YEARS):
        call_supported(
            add_malaria_summary_report, task, "Summary report",
            manifest=manifest,
            start_day=i * DAYS_PER_YEAR + 1,
            end_day=(i + 1) * DAYS_PER_YEAR,
            reporting_interval=30,
            age_bins=[0.25, 5, 15, 25, 100],
            max_number_reports=13,
            parasite_detection_threshold=PARASITE_DETECTION_THRESHOLD,
            pretty_format=True,
            filename_suffix=f"Monthly_U5_{PICKUP_START_YEAR + i}",
        )

    events = ["Received_Treatment", "Received_Severe_Treatment"]
    if has_rows(SCHEDULES["smc"]):
        events.append("Received_SMC")
    if has_rows(SCHEDULES["vaccine"]):
        v = SCHEDULES["vaccine"]
        events.extend(sorted(f"Received_Vaccine_{d}_{int(y)}" for d, y in zip(v["dose"], v["year"])))

    add_event_recorder(task, event_list=events, start_day=1, end_day=PICKUP_DAYS,
                        node_ids=[NODE_ID], min_age_years=0, max_age_years=115)


################################################################################
# CLI / EXPERIMENT SUBMISSION
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="UWR 2013-2025.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check inputs, print the schedule, write campaign_pickup_preview.json, then stop.",
    )
    parser.add_argument(
        "--batch", type=int, choices=range(len(PICKUP_XTLH_RANGE)),
        help=(
            f"Pickup xTLH batch to submit alone: 0 to {len(PICKUP_XTLH_RANGE) - 1}. "
            f"Omit this to submit all {len(PICKUP_XTLH_RANGE)} batches together in one run, "
            "all under the same overall experiment name."
        ),
    )
    return parser.parse_args()


def build_and_submit_batch(platform, burnin_df, climate_dir, batch_index):
    """Build the EMOD task/campaign for one xTLH batch and submit it."""
    if batch_index < 0 or batch_index >= len(PICKUP_XTLH_RANGE):
        raise ValueError(f"batch_index must be between 0 and {len(PICKUP_XTLH_RANGE) - 1}")

    # One xTLH value per batch
    batch_xTLH = [PICKUP_XTLH_RANGE[batch_index]]
    n_sims = len(burnin_df) * len(batch_xTLH) * len(PICKUP_HABITAT_MIXES)

    print(f"\nPickup batch: {batch_index + 1}/{len(PICKUP_XTLH_RANGE)}")
    print(f"xTLH: {batch_xTLH[0]:.4f}")
    print(f"Simulations in this submission: {n_sims}")

    # ------------------------------------------------------------------
    # EMOD TASK
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SIMULATION BUILDER
    # ------------------------------------------------------------------
    builder = SimulationBuilder()

    # All selected burn-in starting states
    builder.add_sweep_definition(partial(update_serialize_parameters, df=burnin_df), range(len(burnin_df)))

    # ONLY this batch's xTLH value (was PICKUP_XTLH_RANGE, which submitted
    # all 5 values in every batch)
    builder.add_sweep_definition(partial(set_param, param="x_Temporary_Larval_Habitat"), batch_xTLH)

    # Pickup habitat mix(es)
    builder.add_sweep_definition(update_pickup_habitat, range(len(PICKUP_HABITAT_MIXES)))

    # ------------------------------------------------------------------
    # SUBMIT
    # ------------------------------------------------------------------
    print(f"\nSubmitting batch {batch_index + 1} under overall name:")
    print(f"  {EXPERIMENT_NAME}")

    experiment = Experiment.from_builder(builder, task, name=EXPERIMENT_NAME)
    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")
    print(f"Overall name: {EXPERIMENT_NAME}")
    print(f"Batch: {batch_index + 1}/{len(PICKUP_XTLH_RANGE)}")
    print(f"xTLH: {batch_xTLH[0]:.4f}")
    print(f"Simulations: {n_sims}")

    return experiment


def general_sim(selected_platform, burnin_id, climate_dir, batch_indices):
    """Submit one batch (batch_indices = [N]) or all of them (the full range),
    all as separate Slurm submissions sharing the same EXPERIMENT_NAME."""
    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    # ------------------------------------------------------------------
    # LOAD ALL BURN-IN STARTING STATES (once, shared across every batch)
    # ------------------------------------------------------------------
    burnin_df = load_burnin_df(burnin_id, platform)

    print(f"\nBurn-in experiment: {burnin_id}")
    print(f"Burn-in states selected: {len(burnin_df)}")
    print(f"Overall experiment name: {EXPERIMENT_NAME}")
    print(f"Batches to submit this run: {[b + 1 for b in batch_indices]} of {len(PICKUP_XTLH_RANGE)}")

    experiments = []
    for batch_index in batch_indices:
        experiments.append(build_and_submit_batch(platform, burnin_df, climate_dir, batch_index))

    if len(experiments) > 1:
        print(f"\nAll {len(experiments)} batches submitted together under: {EXPERIMENT_NAME}")
        for exp, b in zip(experiments, batch_indices):
            print(f"  Batch {b + 1}: Experiment ID {exp.id}")

    return experiments


if __name__ == "__main__":
    args = parse_args()

    SCHEDULES = load_all_schedules()
    describe_schedules(SCHEDULES)

    climate_dir = resolve_climate_dir()
    validate_climate(climate_dir)

    print(f"\nPickup xTLH: {[round(v, 2) for v in PICKUP_XTLH_RANGE]}")
    print(f"Pickup habitat mixes (constant, temp_rain, water_veg): {PICKUP_HABITAT_MIXES}")

    # --------------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------------
    if args.dry_run:
        build_camp()
        camp.save("campaign_pickup_preview.json")
        print("\nDry run OK. Campaign written to campaign_pickup_preview.json. Nothing submitted.")
        sys.exit(0)

    # --------------------------------------------------------------
    # BURN-IN CHECK
    # --------------------------------------------------------------
    burnin_id = BURNIN_EXPERIMENT_ID.strip()
    if not burnin_id:
        sys.exit("Set BURNIN_EXPERIMENT_ID at the top of the script to the finished burn-in experiment ID.")

    # --------------------------------------------------------------
    # BATCH SELECTION
    # --------------------------------------------------------------
    # --batch N submits just that one batch (e.g. to resubmit a single
    # failed one separately). Omitting --batch submits all batches
    # together, back-to-back, under the same EXPERIMENT_NAME.
    if args.batch is None:
        batch_indices = list(range(len(PICKUP_XTLH_RANGE)))
        print(f"\nNo --batch given: submitting all {len(batch_indices)} xTLH batches together.")
    else:
        batch_indices = [args.batch]

    # --------------------------------------------------------------
    # EMOD BOOTSTRAP
    # --------------------------------------------------------------
    import emod_malaria.bootstrap as dtk
    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)

    # --------------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------------
    general_sim("SLURM_LOCAL", burnin_id, climate_dir, batch_indices)