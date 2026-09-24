#########################################################################################
# Upper West Region EMOD Malaria Model
#
# AIM 1:
# Adapt and calibrate the EMOD malaria transmission model to the Upper West Region
# of Ghana.
#
# This experiment performs a 50-year climate-driven burn-in and parameter sweep
# for calibration against observed malaria transmission patterns.
#
# EXACTLY 500 SIMULATIONS:
#
#   5 xTLH values
#   x 20 habitat combinations
#   x 5 stochastic seeds
#   = 500 simulations
#
# The burn-in does NOT contain IRS, IG2 nets, SMC, or case-management interventions.
#
# Outputs are intended to support:
#   1. transmission-equilibrium assessment
#   2. calibration against PfPR and clinical incidence
#   3. selection of calibrated parameter set(s)
#   4. generation of serialized populations for Aim 2
#
#########################################################################################


#########################################################################################
# IMPORT MODULES
#########################################################################################

import os
import pathlib
import getpass
import pandas as pd
import numpy as np

from functools import partial
from itertools import product

# ---------------------------------------------------------------------------------------
# idmtools
# ---------------------------------------------------------------------------------------

from idmtools.assets import Asset, AssetCollection
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

# ---------------------------------------------------------------------------------------
# emodpy
# ---------------------------------------------------------------------------------------

from emodpy.emod_task import EMODTask
import emod_api.campaign as camp

# ---------------------------------------------------------------------------------------
# emodpy-malaria
# ---------------------------------------------------------------------------------------

import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emod_api.demographics.PreDefinedDistributions as Distributions

from emod_api.demographics.DemographicsTemplates import CrudeRate
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
from emodpy_malaria.reporters.builtin import add_malaria_summary_report

import emod_malaria.bootstrap as dtk

# ---------------------------------------------------------------------------------------
# Project modules
# ---------------------------------------------------------------------------------------

import manifest


#########################################################################################
# EXPERIMENT SETTINGS
#########################################################################################

# Simulation duration
sim_years = 50

# Calendar year assigned to beginning of burn-in.
# This is a bookkeeping/reference year; EMOD itself advances by simulation days.
sim_start_year = 1962

# Stochastic replication
# IMPORTANT: DO NOT REDUCE THIS. Five independent stochastic seeds are retained
# for every parameter combination.
num_seeds = 5

# Serialization
serialize_years = 50

# Demography
# birth_rate is the crude birth rate, expressed as births per 1,000 population
# per year. CrudeRate() (used in build_demog() below) converts this to EMOD's
# required per-day, per-individual rate internally -- do NOT pre-multiply this
# by 1000 before passing it to CrudeRate(), or the resulting rate will be
# 1000x too large and EMOD will reject it as out of range.
birth_rate = 31.0

# Single-node population used by this calibration model.
#
# IMPORTANT:
# Replace this with the population value selected for your UWR
# single-node representation if 1000 is only a placeholder.
population_size = 1000

# EMOD memory protection
memory_limit = 10000

# Site
site_name = "UpperWest_Region"

# Vector input
vector_file = "vector.csv"


#########################################################################################
# CALIBRATION PARAMETER SPACE
#########################################################################################

# ---------------------------------------------------------------------------------------
# Larval habitat composition
#
# These are calibration dimensions.
#
# scale_constant:
#   scaling of constant larval habitat
#
# scale_temp_rain:
#   scaling of temporary rainfall-dependent habitat
#
# scale_water_veg:
#   scaling of water-vegetation habitat
# ---------------------------------------------------------------------------------------

scale_constant = [1.0]
scale_temp_rain = [5.0, 10.0, 25.0, 50.0]
scale_water_veg = [0.0, 0.1, 0.5, 1.0, 2.5]

# ---------------------------------------------------------------------------------------
# x_Temporary_Larval_Habitat
#
# Five values are used for the trial calibration.
#
# 5 xTLH values x 20 habitat combinations x 5 seeds = EXACTLY 500 simulations
# ---------------------------------------------------------------------------------------

xTLH_range = list(np.logspace(1.85, 2.15, 5))

# Create habitat combinations
habitat_df = pd.DataFrame(
    list(product(scale_constant, scale_temp_rain, scale_water_veg)),
    columns=["scale_constant", "scale_temp_rain", "scale_water_veg"]
)

n_habs = len(habitat_df)


#########################################################################################
# STRICT 500-SIMULATION VALIDATION
#########################################################################################

expected_simulations = len(xTLH_range) * n_habs * num_seeds


#########################################################################################
# REPORTING SETTINGS
#########################################################################################

# Malaria summary report (add_malaria_summary_report)
report_start_day = 1
report_end_day = sim_years * 365
report_interval_days = 30
report_age_bins = [0.25, 5, 15, 25, 100]
report_max_number_reports = sim_years * 13
report_filename_suffix = "monthly"


#########################################################################################
# CONFIGURATION CALLBACK
#########################################################################################

def set_param_fn(config):
    """
    Configure EMOD for the 50-year Upper West Region calibration burn-in.
    """

    import emodpy_malaria.malaria_config as conf

    # Standard malaria configuration
    config = conf.set_team_defaults(config, manifest)

    # -------------------------------------------------------------------------
    # CLIMATE
    # -------------------------------------------------------------------------
    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", "air_temperature.bin")
    config.parameters.Land_Temperature_Filename = os.path.join("climate", "land_temperature.bin")
    config.parameters.Rainfall_Filename = os.path.join("climate", "rainfall.bin")
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", "relative_humidity.bin")

    # -------------------------------------------------------------------------
    # REPORTING
    #
    # Default reporting is retained because the calibration stage requires
    # transmission time-series outputs.
    # -------------------------------------------------------------------------
    config.parameters.Enable_Default_Reporting = 1

    # -------------------------------------------------------------------------
    # LOGGING
    #
    # Keep logging low during the 500-run sweep to reduce unnecessary output.
    # -------------------------------------------------------------------------
    config.parameters["logLevel_JsonConfigurable"] = "ERROR"
    config.parameters["logLevel_VectorHabitat"] = "ERROR"
    config.parameters["logLevel_StandardEventCoordinator"] = "ERROR"
    config.parameters["logLevel_SusceptibilityMalaria"] = "ERROR"
    config.parameters["logLevel_default"] = "ERROR"
    config.parameters["logLevel_Eradication"] = "ERROR"

    # -------------------------------------------------------------------------
    # MEMORY PROTECTION
    # -------------------------------------------------------------------------
    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = int(memory_limit)
    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = int(memory_limit - 500)

    # -------------------------------------------------------------------------
    # MIGRATION
    #
    # The calibration model is a single-node UWR representation.
    # -------------------------------------------------------------------------
    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Enable_Local_Migration = 0

    # -------------------------------------------------------------------------
    # VECTOR SPECIES AND HABITATS
    # -------------------------------------------------------------------------
    vector_path = os.path.join(manifest.input_dir, vector_file)
    vdf = pd.read_csv(vector_path)

    required_columns = ["node_id", "species", "fraction", "constant", "temp_rain", "water_veg"]
    missing_columns = [col for col in required_columns if col not in vdf.columns]

    if missing_columns:
        raise ValueError(f"{vector_file} is missing required columns: {missing_columns}")

    # Single node check
    unique_nodes = vdf["node_id"].unique()

    if len(unique_nodes) != 1:
        raise ValueError(
            f"{vector_file} contains {len(unique_nodes)} node IDs. "
            f"The Upper West Region calibration burn-in requires exactly one node."
        )

    # Species
    species = vdf["species"].tolist()
    conf.add_species(config, manifest, species)

    # -------------------------------------------------------------------------
    # BASE LARVAL HABITAT CAPACITY
    # -------------------------------------------------------------------------
    base_constant = 1e7
    base_other = 1e7

    for i in range(len(vdf)):
        conf.set_species_param(
            config,
            species=vdf["species"].iloc[i],
            parameter="Habitats",
            value=[
                {
                    "Habitat_Type": "CONSTANT",
                    "Max_Larval_Capacity": base_constant * vdf["constant"].iloc[i]
                },
                {
                    "Habitat_Type": "TEMPORARY_RAINFALL",
                    "Max_Larval_Capacity": base_other * vdf["temp_rain"].iloc[i]
                },
                {
                    "Habitat_Type": "WATER_VEGETATION",
                    "Max_Larval_Capacity": base_other * vdf["water_veg"].iloc[i]
                }
            ],
            overwrite=True
        )

    # -------------------------------------------------------------------------
    # SIMULATION DURATION
    # -------------------------------------------------------------------------
    config.parameters.Simulation_Duration = sim_years * 365

    # Run_Number is overwritten by the SimulationBuilder seed sweep.
    config.parameters.Run_Number = 0

    # -------------------------------------------------------------------------
    # SERIALIZATION
    #
    # Save the population state at the end of year 50.
    # This becomes the starting state for the Aim 2 pickup simulations.
    # -------------------------------------------------------------------------
    config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
    config.parameters.Serialization_Time_Steps = [365 * serialize_years]
    config.parameters.Serialization_Mask_Node_Write = 0
    config.parameters.Serialization_Precision = "REDUCED"

    # BIRTH RATE
    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"

    return config


#########################################################################################
# GENERIC PARAMETER SETTER
#########################################################################################

def set_param(simulation, param, value):
    """
    Generic EMOD simulation parameter setter.
    """
    return simulation.task.set_parameter(param, value)


#########################################################################################
# HABITAT SCALING
#########################################################################################

def scale_habitat_from_df(df=habitat_df, index=-1, vdf=None):
    """
    Construct larval habitat scaling factors for each mosquito species.
    """

    # Default: no additional scaling
    habitat_mix = (1.0, 1.0, 1.0)

    if index >= 0:
        habitat_mix = df.iloc[index].tolist()

    if vdf is None:
        return None

    all_scales = {"NodeID": vdf["node_id"].unique()}

    species = vdf["species"].unique()

    for sp in species:
        species_df = vdf[vdf["species"] == sp]

        all_scales[f"CONSTANT.{sp}"] = [habitat_mix[0] * f for f in species_df["fraction"]]
        all_scales[f"TEMPORARY_RAINFALL.{sp}"] = [habitat_mix[1] * f for f in species_df["fraction"]]
        all_scales[f"WATER_VEGETATION.{sp}"] = [habitat_mix[2] * f for f in species_df["fraction"]]

    return pd.DataFrame(all_scales)


#########################################################################################
# BURN-IN CAMPAIGN
#########################################################################################

def fix_property_restrictions(obj):
    """
    Recursively replace null property-restriction fields with empty lists.

    add_scale_larval_habitats() (emod_api) writes Node_Property_Restrictions
    (and its siblings) as JSON null, but EMOD's schema requires these fields
    to be arrays -- a null here raises JsonTypeConfigurationException
    ("'Node_Property_Restrictions' is NOT an ARRAY in <campaign.json>") for
    every single simulation at node-population time, since the campaign shape
    is the same across the whole sweep. Must be called on camp.campaign_dict
    before build_camp() returns.
    """
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in [
                "Node_Property_Restrictions",
                "Property_Restrictions",
                "Property_Restrictions_Within_Node"
            ]:
                if value is None:
                    obj[key] = []
            else:
                fix_property_restrictions(value)
    elif isinstance(obj, list):
        for item in obj:
            fix_property_restrictions(item)


def build_camp(habitat=-1, scale_start_day=1):
    """
    Build the calibration burn-in campaign.

    IMPORTANT:
    There are NO:
        - IRS interventions
        - IG2 nets
        - SMC
        - case management

    The burn-in contains only larval-habitat scaling.
    This preserves the baseline transmission state required for calibration.
    """

    camp.set_schema(manifest.schema_file)

    vector_df = pd.read_csv(os.path.join(manifest.input_dir, vector_file))

    lh_scales = scale_habitat_from_df(df=habitat_df, vdf=vector_df, index=habitat)

    # Apply scaling for entire simulation
    add_scale_larval_habitats(camp, df=lh_scales, start_day=scale_start_day, repetitions=-1)

    # Required -- see fix_property_restrictions() docstring above.
    fix_property_restrictions(camp.campaign_dict)

    return camp


#########################################################################################
# DEMOGRAPHICS
#########################################################################################

def _read_site_definition():
    """
    Read the Upper West Region site definition.
    """

    demo_file = os.path.join(manifest.input_dir, "demographics.csv")
    site_df = pd.read_csv(demo_file)
    site_df.columns = [c.strip() for c in site_df.columns]

    row = site_df.iloc[0]

    return {
        "name": str(row["name"]).strip(),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "nodes": int(row["nodes"])
    }


def build_demog():
    """
    Build the single-node Upper West Region demographic structure.
    """

    site = _read_site_definition()

    # Single-node requirement
    if site["nodes"] != 1:
        raise ValueError(
            f"demographics.csv specifies {site['nodes']} nodes. "
            f"This calibration model requires exactly one node."
        )

    # Create node
    demog = Demographics.from_template_node(
        lat=site["lat"],
        lon=site["lon"],
        pop=population_size,
        name=site["name"]
    )

    # Vital dynamics.
    #
    # FIXED: birth_rate is a crude rate (per 1,000 population per year) and
    # CrudeRate() already performs the /1000/365 conversion to EMOD's required
    # daily per-individual rate internally. The birth-rate call previously
    # passed `birth_rate * 1000`, which after CrudeRate()'s own division
    # produced a rate 1000x too large (0.0849315, vs. EMOD's 0.005 ceiling) --
    # this was the direct cause of every simulation in the 500-run sweep
    # failing with ConfigurationRangeException. Passing birth_rate directly,
    # consistent with the mortality-rate and age-distribution calls below,
    # gives the correct ~0.0000849/day rate.
    demog.SetBirthRate(CrudeRate(birth_rate))
    demog.SetMortalityRate(CrudeRate(birth_rate))
    demog.SetEquilibriumAgeDistFromBirthAndMortRates(CrudeRate(birth_rate), CrudeRate(birth_rate))

    # Sub-Saharan African age distribution
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)

    return demog


#########################################################################################
# NODE IDs
#########################################################################################

def get_site_node_ids():
    """
    Return node IDs for the Upper West Region model.
    """

    site = _read_site_definition()
    return list(range(1, site["nodes"] + 1))


#########################################################################################
# HABITAT PARAMETER METADATA
#########################################################################################

def update_campaign_multiple_parameters(simulation, habitat_index, scale_start_day):
    """
    Apply one habitat parameter combination to the simulation.

    The returned dictionary is retained as simulation metadata by idmtools,
    allowing the calibration results to be linked back to the parameter set.
    """

    habitat_index = int(habitat_index)
    scale_start_day = int(scale_start_day)

    # Build selected campaign
    build_campaign_partial = partial(
        build_camp,
        habitat=habitat_index,
        scale_start_day=scale_start_day
    )

    simulation.task.create_campaign_from_callback(build_campaign_partial)

    # Return calibration parameters
    return {
        "scale_constant": float(habitat_df.iloc[habitat_index, 0]),
        "scale_temp_rain": float(habitat_df.iloc[habitat_index, 1]),
        "scale_water_veg": float(habitat_df.iloc[habitat_index, 2])
    }


#########################################################################################
# MAIN SIMULATION
#########################################################################################

def general_sim(selected_platform):
    """
    Create  Upper West Region calibration experiment.
    """

    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs
    )

    # -------------------------------------------------------------------------
    # Simulation builder
    # -------------------------------------------------------------------------
    builder = SimulationBuilder()

    # SWEEP 1: x_Temporary_Larval_Habitat
    builder.add_sweep_definition(
        partial(set_param, param="x_Temporary_Larval_Habitat"),
        xTLH_range
    )

    # SWEEP 2: habitat composition
    # 20 combinations: 1 constant x 4 temporary-rain values x 5 water-vegetation values
    builder.add_multiple_parameter_sweep_definition(
        update_campaign_multiple_parameters,
        dict(
            scale_start_day=[1],
            habitat_index=list(range(n_habs))
        )
    )

    # SWEEP 3: stochastic seeds (0,1,2,3,4 = five independent seeds)
    builder.add_sweep_definition(
        partial(set_param, param="Run_Number"),
        range(num_seeds)
    )

    # -------------------------------------------------------------------------
    # Confirm expected Cartesian product
    # -------------------------------------------------------------------------
    print(f"Parameter grid: {len(xTLH_range)} xTLH x {n_habs} habitat x {num_seeds} seeds "
      f"= {expected_simulations} simulations")
    # -------------------------------------------------------------------------
    # Create EMOD task
    # -------------------------------------------------------------------------
    task = EMODTask.from_default2(
        config_path="config.json",
        eradication_path=manifest.eradication_path,
        campaign_builder=build_camp,
        schema_path=manifest.schema_file,
        param_custom_cb=set_param_fn,
        ep4_custom_cb=None,
        demog_builder=build_demog,
        plugin_report=None
    )

    # -------------------------------------------------------------------------
    # Malaria summary report
    #
    # Monthly PfPR / clinical-incidence summary, broken out by the age bins
    # below, over the full 50-year burn-in. This is what Aim 1's calibration
    # (against observed PfPR and clinical incidence) is compared against.
    # -------------------------------------------------------------------------
    add_malaria_summary_report(
        task,
        manifest,
        start_day=report_start_day,
        end_day=report_end_day,
        reporting_interval=report_interval_days,
        age_bins=report_age_bins,
        max_number_reports=report_max_number_reports,
        filename_suffix=report_filename_suffix,
        pretty_format=True
    )

    # Singularity image
    task.set_sif(manifest.SIF_PATH, platform)

    # -------------------------------------------------------------------------
    # Climate assets
    # -------------------------------------------------------------------------
    climate_directory = os.path.join(manifest.input_dir, "emod_weather_upper_west")

    if not os.path.isdir(climate_directory):
        raise FileNotFoundError(f"Climate directory not found: {climate_directory}")

    task.common_assets.add_directory(climate_directory, relative_path="climate")

    # -------------------------------------------------------------------------
    # Confirm node IDs
    # -------------------------------------------------------------------------
    node_ids = get_site_node_ids()
    print(f"[{site_name}] node_ids found in demographics.csv: {node_ids}")

    # -------------------------------------------------------------------------
    # Experiment name
    # -------------------------------------------------------------------------
    # FIXED: os.getlogin() can raise OSError ("Inappropriate ioctl for device")
    # when there's no controlling terminal -- a known failure mode for jobs
    # submitted via SLURM batch. getpass.getuser() is the robust equivalent.
    user = getpass.getuser()

    experiment_name = f"{user}_{site_name}_Project_burnin{serialize_years}"

    print(f"\nCreating experiment: {experiment_name}")

    # -------------------------------------------------------------------------
    # Create experiment
    # -------------------------------------------------------------------------
    experiment = Experiment.from_builder(builder, task, name=experiment_name)

    # Run
    experiment.run(wait_until_done=False, platform=platform)


#########################################################################################
# RUN
#########################################################################################

if __name__ == "__main__":

    # EMOD bootstrap
    dtk.setup(pathlib.Path(manifest.eradication_path).parent)

    # Ensure executable permissions
    os.chmod(manifest.eradication_path, 0o755)

    # Platform
    selected_platform = "SLURM_LOCAL"

    # Launch
    general_sim(selected_platform)