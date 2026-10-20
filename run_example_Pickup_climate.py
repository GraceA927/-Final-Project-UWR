"""
Upper West Region EMOD malaria model -- PICKUP 2013-2025 (climate + habitat only)

Adapts and calibrates EMOD to the Upper West Region of Ghana.

    1963 ---------- 2012 | 2013 ---------- 2025
      50-year burn-in    |  pickup: climate + larval habitat only
                         ^
            serialized state (day 18,250) read here

This pickup applies NO interventions (no CM, ITN, IRS, SMC or vaccine). The
only things driving the 2013-2025 trajectory are the observed climate and the
larval habitat scaling inherited from each burn-in simulation.

Usage
    1. Set BURNIN_EXPERIMENT_ID below once the burn-in has finished.
    2. Preview inputs and campaign without submitting:
           python run_pickup_UWR.py --dry-run
    3. Submit pickup (one simulation per selected burn-in state):
           python run_pickup_UWR.py

Calendar
    Day 1 = 1 Jan 2013, flat 365-day years (no leap days); 13 years = 4,745 days.
    2013 -> day 1, 2014 -> day 366, ..., 2025 -> day 4,381, 31 Dec 2025 -> day 4,745.

emodpy versions
    Keyword arguments are checked against the installed emodpy_malaria
    functions. Unsupported arguments are dropped and reported as WARNINGs in
    the dry run -- read them before submitting.

Larval habitat
    Each burn-in simulation carries its own calibrated
    (x_Temporary_Larval_Habitat, scale_constant, scale_temp_rain,
    scale_water_veg). Pickup applies those four values directly, so there is
    no independent habitat sweep at pickup: one burn-in state -> one pickup
    simulation. Serialization_Mask_Node_Read = 0 reads everything from the
    serialized state, including accumulated larval habitat capacities, and the
    pickup's add_scale_larval_habitats() event keeps applying the same mix.

    Run_Number is taken from each burn-in row, never swept at pickup.

Inputs (INPUT_DIR)
    demographics.csv, vector.csv, and the pickup climate .bin files.
"""

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
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.reporters.builtin import add_malaria_summary_report
import emod_api.campaign as camp

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import manifest


# PROJECT / MODEL SETTINGS

SITE_NAME = "UpperWest_Region"
EXPERIMENT_NAME = "Pickup_Climate_Habitat_13yrs"
NODE_ID = 1
BIRTH_RATE = 31.0          # must match the burn-in
MEMORY_LIMIT = 10000

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
CLIMATE_ROOT = os.path.join(PROJECT_DIR, "climate", "pickup")
CLIMATE_SUBDIR = None      # "emod_weather_upper_west"; None = auto-detect

VECTOR_FILE = "vector.csv"
DEMOGRAPHICS_FILE = "demographics.csv"

# Completed 50-year burn-in experiment ID.
BURNIN_EXPERIMENT_ID = "c522a73f-90c8-4407-a776-1993bd2083a3"

# Optional subset of burn-in sims, filtered on burn-in tags, e.g.
#   {"x_Temporary_Larval_Habitat": 35.0, "scale_temp_rain": 10.0}
# Empty = pick up every burn-in simulation.
BURNIN_FILTER = {}

# Burn-in tag columns applied directly to each pickup simulation.
HABITAT_TAGS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]

# Save the end-of-pickup state (day 4,745 = 31 Dec 2025) for scenario runs.
WRITE_PICKUP_STATE = True

# Microscopy-like detection threshold for summary reports (parasites/uL).
PARASITE_DETECTION_THRESHOLD = 40


# CALENDAR

DAYS_PER_YEAR = 365

BURNIN_START_YEAR = 1963
BURNIN_END_YEAR = 2012
BURNIN_DAYS = (BURNIN_END_YEAR - BURNIN_START_YEAR + 1) * DAYS_PER_YEAR   # 18,250

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1                    # 13
PICKUP_DAYS = PICKUP_YEARS * DAYS_PER_YEAR                                # 4,745


# CLIMATE

CLIMATE_VARS = ["air_temperature", "land_temperature", "rainfall", "relative_humidity"]
CLIMATE_FILES = [f"{v}.bin{ext}" for v in CLIMATE_VARS for ext in ("", ".json")]


def resolve_climate_dir():
    """Folder holding the .bin files: CLIMATE_ROOT itself or one subfolder."""
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

    for var in CLIMATE_VARS:
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


# SITE / VECTOR INPUT

DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, DEMOGRAPHICS_FILE)
VECTOR_PATH = os.path.join(INPUT_DIR, VECTOR_FILE)

for _p in [
    INPUT_DIR, DEMOGRAPHICS_PATH, VECTOR_PATH,
    manifest.eradication_path, manifest.schema_file,
]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"Required path does not exist:\n{_p}")

vector_df = pd.read_csv(VECTOR_PATH)
vector_df.columns = [c.strip() for c in vector_df.columns]


def _read_site_definition():
    site_df = pd.read_csv(DEMOGRAPHICS_PATH)
    site_df.columns = [c.strip() for c in site_df.columns]
    if len(site_df) != 1 or int(site_df["node_id"].iloc[0]) != NODE_ID:
        raise ValueError(f"{DEMOGRAPHICS_FILE} must hold exactly one row with node_id={NODE_ID}.")
    row = site_df.iloc[0]
    return {
        "name": SITE_NAME,
        "pop": int(row["pop"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
    }


# =============================================================================
# CONFIG -- must match the burn-in except duration, climate and serialization
# =============================================================================

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
                {"Habitat_Type": "CONSTANT",
                 "Max_Larval_Capacity": 1e7 * vector_df["constant"].iloc[i]},
                {"Habitat_Type": "TEMPORARY_RAINFALL",
                 "Max_Larval_Capacity": 1e7 * vector_df["temp_rain"].iloc[i]},
                {"Habitat_Type": "WATER_VEGETATION",
                 "Max_Larval_Capacity": 1e7 * vector_df["water_veg"].iloc[i]},
            ],
            overwrite=True,
        )

    # Placeholders, replaced per simulation from the burn-in row
    config.parameters.Run_Number = 0
    config.parameters.x_Temporary_Larval_Habitat = 1

    # Mask 0 reads everything from the serialized state, including larval
    # habitat capacities built up under that burn-in's own xTLH/mix.
    config.parameters.Serialized_Population_Reading_Type = "READ"
    config.parameters.Serialization_Mask_Node_Read = 0

    # Save end-of-pickup state (day 4,745 = 31 Dec 2025) for scenario runs
    # (writes output/state-04745.dtk per simulation).
    if WRITE_PICKUP_STATE:
        config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
        config.parameters.Serialization_Time_Steps = [PICKUP_DAYS]
        config.parameters.Serialization_Mask_Node_Write = 0
        config.parameters.Serialization_Precision = "REDUCED"

    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"
    return config


# DEMOGRAPHICS -- identical to the burn-in

def build_demog():
    site = _read_site_definition()
    demog = Demographics.from_template_node(
        lat=site["lat"], lon=site["lon"], pop=site["pop"], name=site["name"]
    )
    demog.SetBirthRate(CrudeRate(BIRTH_RATE))
    demog.SetMortalityRate(CrudeRate(BIRTH_RATE))
    demog.SetEquilibriumAgeDistFromBirthAndMortRates(CrudeRate(BIRTH_RATE), CrudeRate(BIRTH_RATE))
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)
    return demog


# CAMPAIGN -- larval habitat scaling only

def fix_property_restrictions(obj):
    """Replace null property-restriction fields with the empty arrays EMOD requires."""
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


_WARNED = set()


def call_supported(fn, target, label, **kwargs):
    """Call an emodpy function with only the kwargs it accepts; warn about the rest."""
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
    return fn(target, **kwargs)


def scale_habitat_from_mix(mix):
    """Mix x vector.csv species fraction, per habitat type (same as burn-in)."""
    scales = {"NodeID": vector_df["node_id"].unique()}
    for sp in vector_df["species"].unique():
        frac = vector_df.loc[vector_df["species"] == sp, "fraction"]
        scales[f"CONSTANT.{sp}"] = [mix[0] * f for f in frac]
        scales[f"TEMPORARY_RAINFALL.{sp}"] = [mix[1] * f for f in frac]
        scales[f"WATER_VEGETATION.{sp}"] = [mix[2] * f for f in frac]
    return pd.DataFrame(scales)


def build_camp(habitat_mix=(1.0, 1.0, 1.0)):
    """
    Build the pickup campaign: larval habitat scaling only.

    In a real run `habitat_mix` is the (scale_constant, scale_temp_rain,
    scale_water_veg) tuple from that simulation's own burn-in row; the default
    is only a placeholder used by EMODTask setup and the dry-run preview.
    """
    camp.set_schema(manifest.schema_file)

    add_scale_larval_habitats(
        camp,
        df=scale_habitat_from_mix(habitat_mix),
        start_day=1,
        repetitions=-1,
    )

    fix_property_restrictions(camp.campaign_dict)
    return camp


# SERIALIZATION SWEEP -- one pickup simulation per selected burn-in simulation

def load_burnin_df(burnin_id, platform):
    from utils_slurm import build_burnin_df
    df = build_burnin_df(burnin_id, platform, BURNIN_DAYS)

    required = ["x_Temporary_Larval_Habitat"] + HABITAT_TAGS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Burn-in experiment {burnin_id} is missing tag column(s) {missing}. "
            f"Tags actually present: {list(df.columns)}"
        )

    for key, value in BURNIN_FILTER.items():
        if key not in df.columns:
            raise ValueError(f"BURNIN_FILTER key '{key}' is not a burn-in tag. Tags: {list(df.columns)}")
        df = df[np.isclose(df[key].astype(float), float(value))]

    if df.empty:
        raise ValueError(f"No burn-in simulations left for experiment {burnin_id} after BURNIN_FILTER.")
    return df.reset_index(drop=True)


def update_serialize_parameters(simulation, df, x: int):
    """Point one pickup simulation at burn-in row `x` and inherit its seed,
    xTLH and habitat mix."""
    row = df.iloc[x]

    filenames = row["Serialized_Population_Filenames"]
    if isinstance(filenames, str):
        filenames = [filenames]

    simulation.task.set_parameter("Serialized_Population_Filenames", list(filenames))
    simulation.task.set_parameter(
        "Serialized_Population_Path",
        os.path.join(row["serialized_file_path"], "output"),
    )

    seed = int(row["Run_Number"])
    simulation.task.set_parameter("Run_Number", seed)

    xtlh = float(row["x_Temporary_Larval_Habitat"])
    scale_constant = float(row["scale_constant"])
    scale_temp_rain = float(row["scale_temp_rain"])
    scale_water_veg = float(row["scale_water_veg"])

    simulation.task.set_parameter("x_Temporary_Larval_Habitat", xtlh)
    simulation.task.create_campaign_from_callback(
        partial(build_camp, habitat_mix=(scale_constant, scale_temp_rain, scale_water_veg))
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

def add_reports(task):
    # parasite_detection_threshold=40 parasites/uL makes "PfPR by Age Bin"
    # microscopy-like, matching the survey test.
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


# CLI / SUBMISSION

def parse_args():
    parser = argparse.ArgumentParser(description="UWR 2013-2025 pickup (climate + habitat only).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check inputs, write campaign_pickup_preview.json, then stop.",
    )
    return parser.parse_args()


def submit_pickup(platform, burnin_df, climate_dir):
    """Submit one pickup simulation per selected burn-in row."""
    n_sims = len(burnin_df)
    print(f"\nSimulations in this submission: {n_sims} (= selected burn-in states, 1:1)")

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

    print(f"\nSubmitting under experiment name:\n  {EXPERIMENT_NAME}")
    experiment = Experiment.from_builder(builder, task, name=EXPERIMENT_NAME)
    experiment.run(wait_until_done=False, platform=platform)

    print("\nExperiment submitted successfully.")
    print(f"Experiment ID: {experiment.id}")
    print(f"Name: {EXPERIMENT_NAME}")
    print(f"Simulations: {n_sims}")
    return experiment


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
    print(f"\nBurn-in experiment: {burnin_id}")
    print(f"Burn-in states selected: {len(burnin_df)}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    return submit_pickup(platform, burnin_df, climate_dir)


if __name__ == "__main__":
    args = parse_args()

    climate_dir = resolve_climate_dir()
    validate_climate(climate_dir)

    if args.dry_run:
        print(
            "\nNOTE: dry-run preview uses a PLACEHOLDER habitat mix (1.0, 1.0, 1.0); "
            "the real pickup uses each burn-in row's own mix. No interventions are "
            "included in this pickup."
        )
        build_camp(habitat_mix=(1.0, 1.0, 1.0))
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