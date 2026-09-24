import os
import json
import glob
import calendar
import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

# idmtools
from idmtools.assets import Asset, AssetCollection
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

# emodpy
from emodpy.emod_task import EMODTask
import emod_api.config.default_from_schema_no_validation as dfs
import emod_api.campaign as camp

# emodpy-malaria
import emodpy_malaria.demographics.MalariaDemographics as Demographics
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.interventions.irs import add_scheduled_irs_housing_modification
from emodpy_malaria.reporters.builtin import *

import manifest

################################################################################
# PROJECT INPUTS
#
# These live alongside manifest.py (manifest.ROOT_DIR) / in manifest.input_dir.
################################################################################

VECTOR_FILE = os.path.join(manifest.input_dir, "vector.csv")
CLIMATE_DIR = os.path.join(manifest.ROOT_DIR, "climate")

DEMOGRAPHICS_FILE = os.path.join(manifest.input_dir, "demographics.csv")
IRS_FILE = os.path.join(manifest.input_dir, "IRS.csv")
ITN_FILE = os.path.join(manifest.input_dir, "ITN.csv")
SMC_FILE = os.path.join(manifest.input_dir, "SMC.csv")
CASE_MANAGEMENT_FILE = os.path.join(manifest.input_dir, "case_management.csv")
ITN_AGE_FILE = os.path.join(manifest.input_dir, "ITN_age.csv")
ITN_SEASON_FILE = os.path.join(manifest.input_dir, "ITN_season.csv")

# Every insecticide referenced in IRS.csv / ITN.csv must be listed here, or
# EMOD raises: GeneralConfigurationException: Constrained String
# (Insecticide_Name) with specified value '...' invalid. Possible values are:
#
# These must match the "insecticide" column in IRS.csv EXACTLY (spelling,
# case, and any "+" in a combination product name), since EMOD does a plain
# string match against whatever is registered via add_insecticide_resistance.
INSECTICIDE_NAMES = [
    "Pirimiphos-methyl",
    "Clothianidin",
    "Clothianidin+Deltamethrin",
    "Broflanilide",
]

################################################################################
# PICKUP / BURN-IN SETTINGS
################################################################################

# The simulation's day 0 = Jan 1 of sim_start_year. IRS.csv's own start_day
# column (91, 456, ...) was pre-computed against a 2015 day-0 - it is NOT
# used directly (see calendar_day_of_simulation() / add_irs_from_csv below),
# so sim_start_year can be set independently of that column.
sim_start_year = 2013

# IRS.csv's last round is April 2024, so the pickup window needs to run at
# least through 2024, plus a buffer year so each round's killing/repelling
# decay (up to a ~180-day time constant) has room to resolve inside the sim.
pickup_years = 13
serialize_years = 50  # must match the burn-in simulation's serialization time
burnin_exp_id = 'be3b964f-ff72-40e0-b6d7-8cf70a953619'  # burn-in experiment ID

population_size = 500  # fixed population for every pickup simulation (not swept)
num_seeds = 5  # stochastic replicate seeds swept per burn-in state

pickup_days = pickup_years * 365
burnin_state_filename = f"state-{serialize_years * 365}.dtk"


def load_vector():
    """
    Reads vector.csv (species / larval habitat fractions for the single
    UWR regional node).
    """
    df = pd.read_csv(VECTOR_FILE)

    node_ids = df["node_id"].unique()
    if len(node_ids) != 1:
        raise ValueError(f"This UWR model is configured as a single regional node. Found node IDs: {node_ids}")

    return df


def set_param_fn(config):
    """
    This function is a callback that is passed to emod-api.config to set config parameters, including the malaria defaults.
    """
    import emodpy_malaria.malaria_config as conf

    config = conf.set_team_defaults(config, manifest)

    species_list = load_vector()["species"].unique().tolist()
    conf.add_species(config, manifest, species_list)

    for insecticide_name in INSECTICIDE_NAMES:
        for species in species_list:
            conf.add_insecticide_resistance(
                config,
                manifest,
                insecticide_name=insecticide_name,
                species=species,
            )

    config.parameters.Climate_Model = "CLIMATE_BY_DATA"

    # Defaults to 0 (off) in the installed Eradication build's schema - turn on so
    # InsetChart.json gets written to each simulation's output folder.
    config.parameters.Enable_Default_Reporting = 1
    config.parameters.Air_Temperature_Filename = os.path.join('climate', 'air_temperature.bin')
    config.parameters.Land_Temperature_Filename = os.path.join('climate', 'land_temperature.bin')
    config.parameters.Rainfall_Filename = os.path.join('climate', 'rainfall.bin')
    config.parameters.Relative_Humidity_Filename = os.path.join('climate', 'relative_humidity.bin')

    config.parameters.Simulation_Duration = pickup_days
    config.parameters.Start_Time = 0
    config.parameters.Run_Number = 0

    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"
    config.parameters.Enable_Demographics_Reporting = 0

    config.parameters.Enable_Interventions = 1

    # We are READING a burn-in serialized population here, not writing new ones.
    config.parameters.Serialized_Population_Reading_Type = "READ"
    config.parameters.Serialization_Mask_Node_Read = 0
    config.parameters.Serialized_Population_Filenames = [burnin_state_filename]

    return config


def calendar_day_of_simulation(year, month_name, day_of_month=1, base_year=None):
    """
    Converts a (year, month name) pair from an intervention CSV into an
    EMOD Start_Day: the number of days since the simulation's day 0
    (Jan 1 of base_year / sim_start_year).

    Per EMOD's campaign model, every intervention's Start_Day is an
    integer offset from the simulation's own Start_Time - it has no
    built-in notion of a calendar date. This is computed fresh from
    year/month here (rather than trusting a pre-baked start_day column in
    the CSV) so the schedule stays correct however sim_start_year is set,
    without needing to regenerate the CSV.
    """
    if base_year is None:
        base_year = sim_start_year

    month_number = list(calendar.month_name).index(month_name)

    event_date = datetime.date(int(year), month_number, int(day_of_month))
    epoch = datetime.date(int(base_year), 1, 1)

    return (event_date - epoch).days


def add_irs_from_csv(campaign, csv_file, phase="pickup"):
    """
    Adds one scheduled IRS round per row of IRS.csv.

    The row's Start_Day is (re)computed from its year/month columns via
    calendar_day_of_simulation() rather than trusting the CSV's own
    start_day column, which was pre-computed for a different sim_start_year
    (2015) than the one this pickup now uses (2013). Everything else in
    the row is passed straight through to
    add_scheduled_irs_housing_modification, whose parameter names already
    match the CSV's column names one-for-one.
    """
    df = pd.read_csv(csv_file)

    if phase is not None and "phase" in df.columns:
        df = df[df["phase"] == phase]

    for _, row in df.iterrows():
        start_day = calendar_day_of_simulation(row["year"], row["month"])

        add_scheduled_irs_housing_modification(
            campaign,
            start_day=start_day,
            demographic_coverage=float(row["coverage"]),
            node_ids=[int(row["node_id"])],
            repetitions=int(row["repetitions"]),
            timesteps_between_repetitions=int(row["timesteps_between_repetitions"]),
            insecticide=row["insecticide"],
            killing_initial_effect=float(row["killing_initial_effect"]),
            killing_box_duration=int(row["killing_box_duration"]),
            killing_decay_time_constant=float(row["killing_decay_time_constant"]),
            repelling_initial_effect=float(row["repelling_initial_effect"]),
            repelling_box_duration=int(row["repelling_box_duration"]),
            repelling_decay_time_constant=float(row["repelling_decay_time_constant"]),
        )

    return campaign


def add_historical_interventions(campaign, start_year=sim_start_year, end_year=sim_start_year + pickup_years - 1):
    """
    Add the historical UWR intervention schedule.

    IMPORTANT:
    ITN.csv/SMC.csv/case_management.csv still need their own
    intervention-builder calls below, following the add_irs_from_csv
    pattern above once you confirm their column layouts. Observed data
    (monthly_cases.csv, pcr_prevalence_allAge.csv) does NOT belong here.
    """

    add_irs_from_csv(campaign=campaign, csv_file=IRS_FILE)

    # add_itn_from_csv(campaign=campaign, csv_file=ITN_FILE, age_file=ITN_AGE_FILE, season_file=ITN_SEASON_FILE, start_year=start_year, end_year=end_year)
    # add_smc_from_csv(campaign=campaign, csv_file=SMC_FILE, start_year=start_year, end_year=end_year)
    # add_case_management_from_csv(campaign=campaign, csv_file=CASE_MANAGEMENT_FILE, start_year=start_year, end_year=end_year)

    return campaign


def fix_property_restrictions(obj):
    """
    EMOD campaign objects can sometimes contain null restriction fields.
    Convert null restrictions to empty lists recursively.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("Property_Restrictions", "Property_Restrictions_Within_Node"):
                if value is None:
                    obj[key] = []
            else:
                fix_property_restrictions(value)
    elif isinstance(obj, list):
        for item in obj:
            fix_property_restrictions(item)


def build_pickup_campaign(scale_constant, scale_temp_rain, scale_water_veg):
    """
    This function builds the per-simulation pickup campaign input file for the DTK using emod_api:
    the same larval habitat scaling used by the matching burn-in state, plus the historical
    intervention schedule.

    add_scale_larval_habitats() takes a DataFrame whose columns are EMOD
    habitat-type names (CONSTANT, TEMPORARY_RAINFALL, WATER_VEGETATION, ...)
    holding a *scale factor* to apply on top of whatever those habitats are
    already configured to (not a new absolute habitat value) - see
    emodpy_malaria.interventions.scale_larval_habitats.add_scale_larval_habitats
    for the exact column conventions (optional NodeID / Start_Day columns,
    optional ".species" suffix for per-species factors).
    """
    camp.set_schema(manifest.schema_file)

    vector_df = load_vector()
    node_id = int(vector_df["node_id"].iloc[0])

    habitat_scale_df = pd.DataFrame(
        {
            "NodeID": [node_id],
            "CONSTANT": [scale_constant],
            "TEMPORARY_RAINFALL": [scale_temp_rain],
            "WATER_VEGETATION": [scale_water_veg],
        }
    )

    campaign = add_scale_larval_habitats(campaign=camp, df=habitat_scale_df, start_day=1, repetitions=-1)
    campaign = add_historical_interventions(campaign)

    fix_property_restrictions(campaign.campaign_dict)

    return campaign


def build_camp():
    """
    This function builds a campaign input file for the DTK using emod_api.

    This is only used to establish the schema for EMODTask.from_default2; the
    real per-simulation campaign is built by build_pickup_campaign, applied
    through create_campaign_from_callback in update_serialize_parameters below.
    """
    camp.set_schema(manifest.schema_file)

    return camp


def build_demog():
    """
    This function builds a demographics input file for the DTK using emod_api,
    reading the project's demographics.csv and holding population fixed at
    population_size for every pickup simulation.
    """
    demo_df = pd.read_csv(DEMOGRAPHICS_FILE)

    node_ids = demo_df["node_id"].unique()
    if len(node_ids) != 1:
        raise ValueError(f"UWR model must contain exactly one node. Found: {node_ids}")

    node_id = int(node_ids[0])
    lat = float(demo_df.iloc[0]["latitude"])
    lon = float(demo_df.iloc[0]["longitude"])

    demog = Demographics.from_template_node(lat=lat, lon=lon, pop=population_size, forced_id=node_id)
    demog.SetBirthRate(31.0)  # crude birth rate per 1,000 population per year

    return demog


def set_param(simulation, param, value):
    """
    Set specific parameter value
    Args:
        simulation: idmtools Simulation
        param: parameter
        value: new value
    Returns:
        dict
    """
    return simulation.task.set_parameter(param, value)


def get_burnin_df():
    """
    Reads every serialized burn-in state's metadata.json under the burn-in
    experiment directory and returns a DataFrame with one row per state -
    used the same way as utils_slurm.build_burnin_df, extended with the
    scale_constant/scale_temp_rain/scale_water_veg calibration tags this
    UWR pickup needs to reproduce each burn-in's exact habitat.
    """
    if burnin_exp_id == 'PUT_YOUR_SUCCESSFUL_BURNIN_EXPERIMENT_ID_HERE':
        raise ValueError("\nSet burnin_exp_id to the successful 50-year burn-in experiment ID before running pickup.")

    pattern = os.path.join(manifest.job_directory, f"e_*_{burnin_exp_id}")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"\nCould not find burn-in experiment directory.\nExpected pattern:\n{pattern}")
    if len(matches) > 1:
        raise RuntimeError("More than one burn-in experiment matched:\n" + "\n".join(matches))

    burnin_dir = Path(matches[0])
    print("\nBurn-in experiment:")
    print(burnin_dir)

    required_tags = ["x_Temporary_Larval_Habitat", "scale_constant", "scale_temp_rain", "scale_water_veg", "Run_Number"]
    rows = []

    for metadata_file in burnin_dir.glob("**/metadata.json"):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception:
            continue

        tags = metadata.get("tags", {})

        state_file = metadata_file.parent / "output" / burnin_state_filename
        if not state_file.exists():
            continue

        missing = [tag for tag in required_tags if tag not in tags]
        if missing:
            print("\nSkipping simulation because tags are missing:")
            print(metadata_file)
            print("Missing:", missing)
            continue

        rows.append(
            {
                "serialized_file_path": str(metadata_file.parent),
                "Serialized_Population_Filenames": [burnin_state_filename],
                "Run_Number": int(tags["Run_Number"]),
                "x_Temporary_Larval_Habitat": float(tags["x_Temporary_Larval_Habitat"]),
                "scale_constant": float(tags["scale_constant"]),
                "scale_temp_rain": float(tags["scale_temp_rain"]),
                "scale_water_veg": float(tags["scale_water_veg"]),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No serialized burn-in simulations were found.")

    df = df.sort_values(
        ["x_Temporary_Larval_Habitat", "scale_temp_rain", "scale_water_veg", "Run_Number"]
    ).reset_index(drop=True)

    print(f"\nFound {len(df)} serialized burn-in simulations.")

    return df


def update_serialize_parameters(simulation, df, x: int):
    """
    Applies one burn-in state's serialized population, habitat calibration,
    and Run_Number to one pickup simulation - same interface as
    utils_slurm.build_burnin_df / update_serialize_parameters in
    run_example.py, extended with the extra habitat-scaling tags the UWR
    pickup campaign needs.
    """
    row = df.iloc[x]

    simulation.task.set_parameter("Serialized_Population_Filenames", row["Serialized_Population_Filenames"])
    simulation.task.set_parameter("Serialized_Population_Path", os.path.join(row["serialized_file_path"], "output"))
    simulation.task.set_parameter("Serialized_Population_Reading_Type", "READ")

    # Preserve the exact larval habitat parameterization that generated this state.
    simulation.task.set_parameter("x_Temporary_Larval_Habitat", row["x_Temporary_Larval_Habitat"])

    # Default to the burn-in's own seed; the seed sweep added in general_sim()
    # below (range(num_seeds)) is applied on top, so each burn-in state is
    # actually run with num_seeds stochastic replicates.
    simulation.task.set_parameter("Run_Number", int(row["Run_Number"]))

    simulation.task.create_campaign_from_callback(
        partial(
            build_pickup_campaign,
            scale_constant=row["scale_constant"],
            scale_temp_rain=row["scale_temp_rain"],
            scale_water_veg=row["scale_water_veg"],
        )
    )

    return {
        "Run_Number": int(row["Run_Number"]),
        "x_Temporary_Larval_Habitat": row["x_Temporary_Larval_Habitat"],
        "scale_constant": row["scale_constant"],
        "scale_temp_rain": row["scale_temp_rain"],
        "scale_water_veg": row["scale_water_veg"],
    }


def general_sim(selected_platform):
    """
    This function is designed to be a parameterized version of the sequence of things we do
    every time we run an emod experiment.
    """

    # Set platform and associated values, such as the maximum number of jobs to run at one time.
    # All cluster-specific settings live in manifest.py so users only edit that file.
    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    # create EMODTask
    print("Creating EMODTask (from files)...")

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

    # set the singularity image to be used when running this experiment
    task.set_sif(manifest.SIF_PATH, platform)
    task.common_assets.add_directory(CLIMATE_DIR, relative_path="climate")

    task.common_assets.add_asset(VECTOR_FILE)
    task.common_assets.add_asset(DEMOGRAPHICS_FILE)
    task.common_assets.add_asset(IRS_FILE)
    task.common_assets.add_asset(ITN_FILE)
    task.common_assets.add_asset(SMC_FILE)
    task.common_assets.add_asset(CASE_MANAGEMENT_FILE)

    if os.path.exists(ITN_AGE_FILE):
        task.common_assets.add_asset(ITN_AGE_FILE)
    if os.path.exists(ITN_SEASON_FILE):
        task.common_assets.add_asset(ITN_SEASON_FILE)

    builder = SimulationBuilder()
    burnin_df = get_burnin_df()

    builder.add_sweep_definition(partial(update_serialize_parameters, df=burnin_df), range(len(burnin_df.index)))
    builder.add_sweep_definition(partial(set_param, param='Run_Number'), range(num_seeds))
    ## reports are still located here

    # create experiment from builders
    user = os.getlogin()
    experiment = Experiment.from_builder(builder, task, name=f'{user}_UWR_pickup_2013_2022')

    add_malaria_summary_report(
        task,
        manifest,
        start_day=0,
        end_day=pickup_days,
        reporting_interval=30,
        age_bins=[0, 1, 5, 15, 30, 50, 125],
        max_number_reports=int(pickup_days / 30) + 1,
        pretty_format=True,
        filename_suffix='Monthly_UWR_pickup',
    )

    # The last step is to call run() on the ExperimentManager to run the simulations.
    experiment.run(wait_until_done=False, platform=platform)

    print(f"\nExperiment ID: {experiment.uid}")
    print(f"Pickup period: {sim_start_year}-{sim_start_year + pickup_years - 1}")
    print(f"Population size: {population_size}")
    print(f"Seeds per burn-in state: {num_seeds}")

    # Check result
    '''if not experiment.succeeded:
        print(f"Experiment {experiment.uid} failed.\n")
        exit()

    print(f"Experiment {experiment.uid} succeeded.")'''


if __name__ == "__main__":
    import emod_malaria.bootstrap as dtk
    import pathlib

    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    os.chmod(manifest.eradication_path, 0o755)

    selected_platform = "SLURM_LOCAL"
    general_sim(selected_platform)