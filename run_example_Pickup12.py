#########################################################################################
# Upper West Region EMOD Malaria Model
#
# AIM 1:
# Adapt and calibrate the EMOD malaria transmission model to the Upper West Region
# of Ghana.
#
# PICKUP:
# Run a 2013-2025 historical pickup from the 50-year burn-in serialized population,
# then serialize the pickup's OWN end-state so Aim 2 counterfactual scenarios
# (IRS vs IG2) can branch from a single calibrated starting point.
#
# IG2 is intentionally NOT included in the historical pickup -- it belongs to
# Aim 2 counterfactual simulations after calibration.
#
# Climate:
#   Historical NASA POWER climate for 2013-2025
#   One regional node (node_id = 1)
#
# STATUS -- this version, checked against run_example_pickup10.py (the one
# script in this project actually built around this exact burn-in/pickup
# pattern):
#
#   CONFIRMED against run_example_pickup10.py's ACTIVE code path (not its
#   commented-out alternative):
#     - Demographics: Demographics.from_template_node(lat=, lon=, pop=,
#       name=), matching burnin_UpperWest.py, which is proven to run.
#       The previous version's Demographics.from_csv(...) is UNCONFIRMED
#       and has been replaced.
#     - Case management: cm._get_events(camp, start_day=, drug=[...],
#       targets=[...], broadcast_event_name=...). The previous version's
#       cm.add_treatment_seeking(...) call was claimed "confirmed against a
#       working reference example" -- but that claim was wrong:
#       run_example_pickup10.py has add_treatment_seeking commented OUT and
#       _get_events as the active call. Fixed to match what's actually
#       active in that reference.
#     - Task/platform construction, update_serialize_parameters,
#       build_burnin_df usage -- unchanged, already matched the reference.
#
#   STILL UNCONFIRMED -- run_example_pickup10.py does not use IRS, ITN, or
#   SMC at all, so none of the following have any working reference to
#   check against in this project:
#     - IRS: `from emodpy_malaria.interventions import insecticide` just
#       CONFIRMED BROKEN -- ImportError, that module does not exist in
#       this venv under that name. Import is now deferred into
#       add_irs_campaign() so it doesn't crash the whole script at load
#       time; it raises a clear, actionable ImportError with the
#       verification command to run, at the point IRS is actually needed.
#     - ITN: bednet.add_uncomplicated_bednet()'s age_dependence /
#       seasonal_dependence keyword names -- unverified. Import deferred
#       the same way.
#     - SMC: add_drug_campaign()'s exact kwargs -- unverified. Import
#       deferred the same way.
#
#   RECOMMENDATION: run with ENABLE_IRS = ENABLE_ITN = ENABLE_SMC = False
#   first to confirm the core pickup mechanics (serialization read from
#   burn-in, config, demographics, CM) work end-to-end, THEN enable one
#   intervention at a time after running its verification command below --
#   rather than debugging all three at once if something fails.
#
#   The malaria vaccine remains fully unimplemented (add_vaccine_campaign
#   raises NotImplementedError) -- no reference or verification attempted
#   yet.
#
#   The calibration comparison against prevalence_df (MICS/GDHS/GMIS
#   survey points) is also not yet implemented -- this script produces
#   *a* pickup simulation, not yet *the calibrated* one Aim 1/2.1 require.
#########################################################################################

import os
import pandas as pd
import numpy as np

from functools import partial

import emodpy_malaria.malaria_config as conf
from emodpy.emod_task import EMODTask
import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emod_api.demographics.PreDefinedDistributions as Distributions
import emod_api.campaign as camp

import emodpy_malaria.interventions.treatment_seeking as cm
from emodpy_malaria.reporters.builtin import add_malaria_summary_report

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import manifest


#########################################################################################
# WHICH HISTORICAL INTERVENTIONS TO INCLUDE
#
# See RECOMMENDATION above -- run with all three False first to confirm the
# core pickup mechanics, then enable one at a time.
#########################################################################################

ENABLE_IRS = True  # confirmed: irs.add_scheduled_irs_housing_modification
ENABLE_ITN = True  # confirmed empirically: age/seasonal Times/Values, and discard_config Weibull Kappa/Lambda
ENABLE_SMC = True  # confirmed: add_drug_campaign signature matches exactly


#########################################################################################
# PROJECT PATHS
#########################################################################################

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments")
CLIMATE_DIR = os.path.join(PROJECT_DIR, "climate", "pickup", "emod_weather_upper_west")

DEMOGRAPHICS_FILE = os.path.join(INPUT_DIR, "demographics.csv")
VECTOR_FILE = os.path.join(INPUT_DIR, "vector.csv")
CM_FILE = os.path.join(INPUT_DIR, "interventions_CM.csv")
IRS_FILE = os.path.join(INPUT_DIR, "interventions_IRS.csv")
ITN_FILE = os.path.join(INPUT_DIR, "interventions_ITN.csv")
ITN_AGE_FILE = os.path.join(INPUT_DIR, "interventions_ITN_age.csv")
ITN_SEASON_FILE = os.path.join(INPUT_DIR, "interventions_ITN_season.csv")
SMC_FILE = os.path.join(INPUT_DIR, "interventions_SMC.csv")
VACCINE_FILE = os.path.join(INPUT_DIR, "interventions_malariaVaccine.csv")
PREVALENCE_FILE = os.path.join(INPUT_DIR, "pcr_prevalence.csv")


#########################################################################################
# MODEL SETTINGS
#########################################################################################

SITE_NAME = "UpperWest_Region"
NODE_ID = 1
POPULATION_SIZE = 1000
BIRTH_RATE = 31.0
MEMORY_LIMIT = 10000


#########################################################################################
# PICKUP SETTINGS
#########################################################################################

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
PICKUP_YEARS = PICKUP_END_YEAR - PICKUP_START_YEAR + 1
DAYS_PER_YEAR = 365
PICKUP_DAYS = PICKUP_YEARS * DAYS_PER_YEAR

# NOTE: 1 seed per burn-in state is a known blocker for Aim 2.2's "cases
# averted" outcome, which needs enough stochastic replicates per scenario
# to distinguish a real effect from noise. Left as-is here since seed count
# is an Aim-2 script decision, not a pickup-correctness fix.
NUM_PICKUP_SEEDS = 1

# DESIGN DECISION (confirmed): pickup sweeps x_Temporary_Larval_Habitat
# independently, matching run_example_pickup10.py's pattern -- it does NOT
# inherit each burn-in state's own calibrated xTLH/habitat values. This
# means pickup's sweep is a CROSS-PRODUCT with the burn-in-state sweep:
# every serialized burn-in state gets combined with every value below,
# regardless of what xTLH that state actually equilibrated under.
#
# Same range as burnin_UpperWest.py's XTLH_RANGE, for consistency with this
# project's actual calibration search space (not the reference script's
# arbitrary np.logspace(-0.5, 1, 5) teaching values).
#XTLH_RANGE = list(np.logspace(1.85, 2.15, 5))
# CALIBRATION PARAMETER SPACE

SCALE_CONSTANT = [1.0]
SCALE_TEMP_RAIN = [5.0, 10.0, 25.0, 50.0]
SCALE_WATER_VEG = [0.0, 0.1, 0.5, 1.0, 2.5]
XTLH_RANGE = list(np.logspace(1.85, 2.15, 5))

habitat_df = pd.DataFrame(
    list(product(SCALE_CONSTANT, SCALE_TEMP_RAIN, SCALE_WATER_VEG)),
    columns=["scale_constant", "scale_temp_rain", "scale_water_veg"],
)
N_HABS = len(habitat_df)
EXPECTED_SIMULATIONS = len(XTLH_RANGE) * N_HABS * num_seeds



#########################################################################################
# BURN-IN SETTINGS
#########################################################################################

# Set this directly to the completed 50-year burn-in experiment ID before
# running pickup, e.g. BURNIN_EXPERIMENT_ID = '3b47f3dc-c35e-4de0-89b3-ccef54ab945f'
BURNIN_EXPERIMENT_ID = ' 3b47f3dc-c35e-4de0-89b3-ccef54ab945f'

SERIALIZE_YEARS = 50
SERIALIZATION_DAY = SERIALIZE_YEARS * DAYS_PER_YEAR  # 18,250 -- burn-in's own serialization day


#########################################################################################
# EXPERIMENT
#########################################################################################

EXPERIMENT_NAME = "UWR_Historical_Pickup_2013_2025"


#########################################################################################
# VALIDATION
#########################################################################################

def validate_inputs():
    required_files = [DEMOGRAPHICS_FILE, VECTOR_FILE, CM_FILE]

    if ENABLE_IRS:
        required_files.append(IRS_FILE)
    if ENABLE_ITN:
        required_files += [ITN_FILE, ITN_AGE_FILE, ITN_SEASON_FILE]
    if ENABLE_SMC:
        required_files.append(SMC_FILE)

    for filename in required_files:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Required input file not found:\n{filename}")

    if not os.path.isdir(CLIMATE_DIR):
        raise FileNotFoundError(f"Pickup climate directory not found:\n{CLIMATE_DIR}")

    climate_files = ["air_temperature.bin", "land_temperature.bin", "rainfall.bin", "relative_humidity.bin"]
    for filename in climate_files:
        climate_file = os.path.join(CLIMATE_DIR, filename)
        if not os.path.isfile(climate_file):
            raise FileNotFoundError(f"Required pickup climate file not found:\n{climate_file}")

        metadata_file = climate_file + ".json"
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(f"Required climate metadata not found:\n{metadata_file}")


#########################################################################################
# FILTER HISTORICAL PERIOD
#########################################################################################

def filter_historical_data(df):
    if "year" not in df.columns:
        return df.copy()

    return df[(df["year"] >= PICKUP_START_YEAR) & (df["year"] <= PICKUP_END_YEAR)].copy()


#########################################################################################
# SITE / DEMOGRAPHICS
#
# CONFIRMED against burnin_UpperWest.py (proven to run) and the ACTIVE code
# path in run_example_pickup10.py: Demographics.from_template_node(), not
# from_csv(). Reuses the exact site-reading logic already proven in the
# burn-in script, since demographics.csv's schema (node_id,pop,lat,lon) is
# the same file for both burn-in and pickup.
#########################################################################################

def _read_site_definition():
    site_df = pd.read_csv(DEMOGRAPHICS_FILE)
    site_df.columns = [c.strip() for c in site_df.columns]
    required = ["node_id", "pop", "lat", "lon"]
    missing = [c for c in required if c not in site_df.columns]

    if missing:
        raise ValueError(f"demographics.csv is missing columns: {missing}")

    if len(site_df) != 1:
        raise ValueError(f"This model is configured as one UWR node. demographics.csv contains {len(site_df)} rows.")

    row = site_df.iloc[0]
    node_id = int(row["node_id"])

    if node_id != NODE_ID:
        raise ValueError(f"demographics.csv contains node_id={node_id}, but the model requires node_id={NODE_ID}.")

    return {
        "name": SITE_NAME,
        "node_id": node_id,
        "pop": int(row["pop"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
    }


def build_demographics():
    site = _read_site_definition()

    demog = Demographics.from_template_node(
        lat=site["lat"], lon=site["lon"], pop=site["pop"], name=site["name"]
    )
    demog.SetEquilibriumVitalDynamics()
    demog.SetAgeDistribution(Distributions.AgeDistribution_SSAfrica)

    return demog


#########################################################################################
# BUILD CONFIGURATION
#########################################################################################

def set_param_fn(config):
    config = conf.set_team_defaults(config, manifest)

    config.parameters.Simulation_Duration = PICKUP_DAYS
    config.parameters.Run_Number = 0
    config.parameters.Enable_Default_Reporting = 1

    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = os.path.join("climate", "air_temperature.bin")
    config.parameters.Land_Temperature_Filename = os.path.join("climate", "land_temperature.bin")
    config.parameters.Rainfall_Filename = os.path.join("climate", "rainfall.bin")
    config.parameters.Relative_Humidity_Filename = os.path.join("climate", "relative_humidity.bin")

    config.parameters.Enable_Local_Migration = 0
    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"

    # --- VECTOR SPECIES ---
    # FIXED: this was missing entirely -- the pickup simulation had a vector
    # population with no species parameters loaded, causing:
    #   "There is a vector population in the simulation but no vector
    #    species parameters could be loaded from the configuration."
    # Matches burnin_UpperWest.py's approach exactly: species list comes
    # from vector.csv, added via conf.add_species(config, manifest, species).
    vector_df = pd.read_csv(VECTOR_FILE)
    species = vector_df["species"].tolist()
    conf.add_species(config, manifest, species)

    # --- SERIALIZED POPULATION: READ from burn-in ---
    config.parameters.Serialized_Population_Reading_Type = "READ"
    config.parameters.Serialization_Mask_Node_Read = 0
    config.parameters.Serialization_Time_Steps = [SERIALIZATION_DAY]
    config.parameters.Serialization_Precision = "REDUCED"

    # --- SERIALIZED POPULATION: WRITE at end of pickup ---
    # Mirrors burnin_UpperWest.py's own end-of-run serialization -- without
    # this, no calibrated end-state exists for Aim 2 to branch from.
    config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
    config.parameters.Serialization_Time_Steps = config.parameters.Serialization_Time_Steps + [PICKUP_DAYS]
    config.parameters.Serialization_Mask_Node_Write = 0

    return config


#########################################################################################
# CASE MANAGEMENT
#
# CONFIRMED against run_example_pickup10.py's ACTIVE code path (not its
# commented-out add_treatment_seeking alternative): cm._get_events(camp,
# start_day=, drug=[...], targets=[...], broadcast_event_name=...).
#
# DRUG_NAME_MAP: interventions_CM.csv stores drug as a short code ("AL");
# the confirmed call takes full component names.
#########################################################################################

DRUG_NAME_MAP = {
    "AL": ["Artemether", "Lumefantrine"],
}


def add_case_management(camp, cm_df):
    """
    Apply one treatment-seeking campaign event per row of interventions_CM.csv,
    via cm._get_events() -- the call shape confirmed active (not commented
    out) in run_example_pickup10.py. Each row's own start_day/duration
    defines its active window; rows are not assumed contiguous.
    """
    cm_df = filter_historical_data(cm_df)

    for _, row in cm_df.iterrows():
        start_day = int(row["start_day"])
        age_min = float(row["age_min"])
        age_max = float(row["age_max"])
        coverage = float(row["coverage"])
        rate = float(row["rate"])
        trigger = str(row["trigger"])
        drug_code = str(row["drug"])

        drug = DRUG_NAME_MAP.get(drug_code)
        if drug is None:
            raise ValueError(
                f"Unrecognized CM drug code '{drug_code}' in interventions_CM.csv -- "
                "add it to DRUG_NAME_MAP with its full component names."
            )

        cm._get_events(
            camp,
            start_day=start_day,
            drug=drug,
            targets=[
                {
                    "trigger": trigger,
                    "coverage": coverage,
                    "agemin": age_min,
                    "agemax": age_max,
                    "rate": rate,
                }
            ],
            broadcast_event_name="Received_Treatment",
        )


#########################################################################################
# IRS
#
# CONFIRMED: irs.add_scheduled_irs_housing_modification takes killing/
# repelling effect parameters directly -- no separate irs_configuration
# step needed. Signature verified against this project's installed
# emodpy_malaria.
#########################################################################################

def add_irs_campaign(camp, irs_df):
    from emodpy_malaria.interventions import irs

    irs_df = filter_historical_data(irs_df)

    for _, row in irs_df.iterrows():
        irs.add_scheduled_irs_housing_modification(
            camp,
            start_day=int(row["start_day"]),
            demographic_coverage=float(row["coverage"]),
            repetitions=int(row["repetitions"]),
            timesteps_between_repetitions=int(row["timesteps_between_repetitions"]),
            killing_initial_effect=float(row["killing_initial_effect"]),
            killing_box_duration=int(row["killing_box_duration"]),
            killing_decay_time_constant=float(row["killing_decay_time_constant"]),
            repelling_initial_effect=float(row["repelling_initial_effect"]),
            repelling_box_duration=int(row["repelling_box_duration"]),
            repelling_decay_time_constant=float(row["repelling_decay_time_constant"]),
            insecticide=str(row["insecticide"]),
        )


#########################################################################################
# ITN
#
# CONFIRMED empirically (by calling the function and inspecting the
# resulting campaign JSON): age_dependence and seasonal_dependence both use
# {"Times": [...], "Values": [...]} -- verified this produces
# WaningEffectMapLinearAge / WaningEffectMapLinearSeasonal blocks with the
# exact values passed in.
#
# Real CSV columns (confirmed via head):
#   interventions_ITN.csv:        start_day, coverage, kill_effect,
#                                  kill_decay, block_effect, block_decay,
#                                  discard_k, discard_l
#   interventions_ITN_age.csv:    year, age, age_usage
#   interventions_ITN_season.csv: season_time, season_usage (not
#                                  year-specific -- one shared schedule)
#
# STILL UNVERIFIED at time of writing this comment -- discard_config's key
# names for a Weibull distribution (discard_k/discard_l strongly suggest
# Weibull shape/scale, and WEIBULL_DISTRIBUTION is confirmed present in this
# project's schema). CONFIRMED since: calling the function with
# {'Expiration_Period_Distribution': 'WEIBULL_DISTRIBUTION',
#  'Expiration_Period_Kappa': 2.3, 'Expiration_Period_Lambda': 912.5}
# and inspecting the resulting campaign JSON showed both values landing
# exactly as passed in. All three dict-shaped ITN parameters
# (age_dependence, seasonal_dependence, discard_config) are now empirically
# verified, not guessed.
#########################################################################################

def add_itn_campaign(camp, itn_df, itn_age_df, itn_season_df):
    from emodpy_malaria.interventions import usage_dependent_bednet as bednet

    itn_df = filter_historical_data(itn_df)
    itn_age_df = filter_historical_data(itn_age_df)

    # Seasonal usage is one shared schedule (not year-specific) -- built once.
    seasonal_dependence = {
        "Times": itn_season_df["season_time"].astype(float).tolist(),
        "Values": itn_season_df["season_usage"].astype(float).tolist(),
    }

    for _, row in itn_df.iterrows():
        year = int(row["year"])

        year_age_df = itn_age_df[itn_age_df["year"] == year].sort_values("age")
        age_dependence = {
            "Times": year_age_df["age"].astype(float).tolist(),
            "Values": year_age_df["age_usage"].astype(float).tolist(),
        } if len(year_age_df) > 0 else None

        # Confirmed empirically -- see comment block above.
        discard_config = {
            "Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",
            "Expiration_Period_Kappa": float(row["discard_k"]),
            "Expiration_Period_Lambda": float(row["discard_l"]),
        }

        bednet.add_scheduled_usage_dependent_bednet(
            camp,
            start_day=int(row["start_day"]),
            demographic_coverage=float(row["coverage"]),
            killing_initial_effect=float(row["kill_effect"]),
            killing_decay_time_constant=float(row["kill_decay"]),
            blocking_initial_effect=float(row["block_effect"]),
            blocking_decay_time_constant=float(row["block_decay"]),
            discard_config=discard_config,
            age_dependence=age_dependence,
            seasonal_dependence=seasonal_dependence,
        )


#########################################################################################
# SMC -- import deferred, UNVERIFIED
#
# No reference example in this project uses add_drug_campaign. Before
# enabling ENABLE_SMC, run:
#
#     python -c "
#     import inspect
#     from emodpy_malaria.interventions.drug_campaign import add_drug_campaign
#     print(inspect.signature(add_drug_campaign))
#     "
#
# and adjust drug_code / target_group / any renamed kwargs below if they
# don't match.
#########################################################################################

def add_smc_campaign(camp, smc_df):
    try:
        from emodpy_malaria.interventions.drug_campaign import add_drug_campaign
    except ImportError as e:
        raise ImportError(
            "SMC (drug_campaign) module import failed (see add_smc_campaign's "
            "comment block for the verification command to run before "
            "enabling ENABLE_SMC)."
        ) from e

    smc_df = filter_historical_data(smc_df)

    for _, row in smc_df.iterrows():
        add_drug_campaign(
            camp,
            campaign_type='SMC',
            drug_code='SPA',
            start_days=[int(row["start_day"])],
            coverage=float(row["coverage"]),
            target_group={'agemin': 0.5, 'agemax': 5},  # WHO SMC target: children 6-59 months
            repetitions=1,
            tsteps_btwn_repetitions=-1,
        )


#########################################################################################
# MALARIA VACCINE -- NOT YET WIRED
#########################################################################################

def add_vaccine_campaign(camp, vaccine_df):
    """
    interventions_malariaVaccine.csv contains real D1-D4 coverage for
    2024-2025, but no vaccine intervention call has been verified against
    this project's installed emodpy_malaria version.
    """
    raise NotImplementedError(
        "add_vaccine_campaign is not yet implemented -- run the same "
        "pkgutil listing as the IRS/ITN/SMC verification commands above to "
        "find the vaccine submodule, then inspect.signature() the right "
        "function before wiring this in."
    )


#########################################################################################
# BUILD CAMPAIGN
#########################################################################################

def build_campaign():
    camp.set_schema(manifest.schema_file)

    cm_df = pd.read_csv(CM_FILE)
    add_case_management(camp, cm_df)

    if ENABLE_IRS:
        irs_df = pd.read_csv(IRS_FILE)
        add_irs_campaign(camp, irs_df)

    if ENABLE_ITN:
        itn_df = pd.read_csv(ITN_FILE)
        itn_age_df = pd.read_csv(ITN_AGE_FILE)
        itn_season_df = pd.read_csv(ITN_SEASON_FILE)
        add_itn_campaign(camp, itn_df, itn_age_df, itn_season_df)

    if ENABLE_SMC:
        smc_df = pd.read_csv(SMC_FILE)
        add_smc_campaign(camp, smc_df)

    # IMPORTANT: NO IG2 is added to this historical pickup.
    # IG2 belongs to Aim 2 counterfactual simulations after calibration.

    return camp


#########################################################################################
# CREATE EMOD TASK
#
# Matches burnin_UpperWest.py's proven EMODTask.from_default2(...) call
# exactly.
#########################################################################################

def create_task():
    task = EMODTask.from_default2(
        config_path="config.json",
        eradication_path=manifest.eradication_path,
        campaign_builder=build_campaign,
        schema_path=manifest.schema_file,
        param_custom_cb=set_param_fn,
        ep4_custom_cb=None,
        demog_builder=build_demographics,
        plugin_report=None,
    )

    task.common_assets.add_directory(CLIMATE_DIR, relative_path="climate")
    # NOTE: vector.csv is NOT added as a common asset here, matching both
    # burnin_UpperWest.py and run_example_pickup10.py -- it's only read via
    # pandas.read_csv() at build time on the head node, never needed inside
    # the simulation container at runtime. AssetCollection also has no
    # add_file() method (it uses add_asset() with an Asset object), which is
    # why this line failed.

    return task


#########################################################################################
# GENERIC PARAMETER SETTER
#########################################################################################

def set_param(simulation, param, value):
    return simulation.task.set_parameter(param, value)


#########################################################################################
# LOCATE BURN-IN SERIALIZED STATES
#########################################################################################

def get_burnin_df(burnin_exp_id, platform):
    if not burnin_exp_id:
        raise ValueError("BURNIN_EXPERIMENT_ID is empty. Set the completed 50-year burn-in experiment ID before running pickup.")

    from utils_slurm import build_burnin_df

    burnin_df = build_burnin_df(burnin_exp_id, platform, SERIALIZATION_DAY)

    if burnin_df is None or len(burnin_df.index) == 0:
        raise RuntimeError("No serialized burn-in states were found.")

    required_columns = ["serialized_file_path", "Serialized_Population_Filenames", "Run_Number"]
    missing = [col for col in required_columns if col not in burnin_df.columns]
    if missing:
        raise RuntimeError("Burn-in dataframe is missing required columns: " + ", ".join(missing))

    return burnin_df


#########################################################################################
# UPDATE SERIALIZATION PARAMETERS
#########################################################################################

def update_serialize_parameters(simulation, df, x):
    path = df["serialized_file_path"][x]
    seed = int(df["Run_Number"][x])
    filenames = df["Serialized_Population_Filenames"][x]

    if isinstance(filenames, str):
        filenames = [filenames]

    simulation.task.set_parameter("Serialized_Population_Filenames", filenames)
    simulation.task.set_parameter("Serialized_Population_Path", os.path.join(path, "output"))

    # Preserve the stochastic identity of the burn-in realization.
    simulation.task.set_parameter("Run_Number", seed)

    return {
        "Run_Number": seed,
        "Serialized_Population_Path": os.path.join(path, "output"),
        "Serialized_Population_Filenames": filenames,
    }


#########################################################################################
# BUILD EXPERIMENT
#########################################################################################

def general_sim(selected_platform):
    validate_inputs()

    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    task = create_task()
    task.set_sif(manifest.SIF_PATH, platform)

    builder = SimulationBuilder()

    # DESIGN DECISION (confirmed): independent xTLH sweep, cross-producing
    # with the burn-in-state sweep below -- matches run_example_pickup10.py's
    # pattern. See XTLH_RANGE's comment above for why per-state inheritance
    # was NOT used instead.
    builder.add_sweep_definition(partial(set_param, param="x_Temporary_Larval_Habitat"), XTLH_RANGE)

    # One pickup per serialized burn-in state, crossed with every XTLH_RANGE
    # value above. Do NOT add an independent Run_Number sweep here -- the
    # pickup Run_Number is inherited from the burn-in so each historical
    # pickup starts from its corresponding immune/transmission history.
    burnin_df = get_burnin_df(BURNIN_EXPERIMENT_ID, platform)

    builder.add_sweep_definition(
        partial(update_serialize_parameters, df=burnin_df),
        range(len(burnin_df.index)),
    )

    total_pickup_sims = len(burnin_df.index) * len(XTLH_RANGE)
    print(
        f"Pickup sweep: {len(burnin_df.index)} burn-in states x "
        f"{len(XTLH_RANGE)} xTLH values = {total_pickup_sims} total pickup simulations"
    )

    # Historical pickup reports -- one per calendar year, matching the
    # burn-in's per-year report structure.
    for i in range(PICKUP_YEARS):
        calendar_year = PICKUP_START_YEAR + i

        add_malaria_summary_report(
            task,
            manifest,
            start_day=1 + (DAYS_PER_YEAR * i),
            end_day=DAYS_PER_YEAR + (DAYS_PER_YEAR * i),
            reporting_interval=30,
            age_bins=[0.5, 5, 15, 25, 100],
            max_number_reports=13,
            filename_suffix=f"Monthly_U5_{calendar_year}",
            pretty_format=True,
        )

    experiment = Experiment.from_builder(builder, task, name=EXPERIMENT_NAME)

    experiment.tags = {
        "Study": "UWR Malaria",
        "Aim": "Aim 1",
        "Stage": "Historical Pickup",
        "Site": SITE_NAME,
        "StartYear": str(PICKUP_START_YEAR),
        "EndYear": str(PICKUP_END_YEAR),
        "BurninYears": str(SERIALIZE_YEARS),
        "Climate": "NASA POWER historical 2013-2025",
        "NodeCount": "1",
        "IG2": "Excluded",
        "IRS": "Enabled" if ENABLE_IRS else "Disabled (unverified module)",
        "ITN": "Enabled" if ENABLE_ITN else "Disabled (unverified module)",
        "SMC": "Enabled" if ENABLE_SMC else "Disabled (unverified module)",
        "Vaccine": "Not yet wired",
        "SerializesEndState": "True",
    }

    experiment.run(wait_until_done=False, platform=platform)

    print("\n==============================================================")
    print("UWR HISTORICAL PICKUP SUBMITTED")
    print("==============================================================")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Pickup period: {PICKUP_START_YEAR}-{PICKUP_END_YEAR}")
    print(f"Pickup years: {PICKUP_YEARS}")
    print(f"Burn-in years: {SERIALIZE_YEARS}")
    print(f"Burn-in experiment: {BURNIN_EXPERIMENT_ID}")
    print(f"Climate: {CLIMATE_DIR}")
    print(f"Serialized states read: {len(burnin_df.index)}")
    print(f"xTLH sweep values: {len(XTLH_RANGE)}")
    print(f"Total pickup simulations: {total_pickup_sims}")
    print(f"Pickup end-state serialized at day: {PICKUP_DAYS}")
    print(f"IRS: {'ENABLED' if ENABLE_IRS else 'disabled (unverified)'}")
    print(f"ITN: {'ENABLED' if ENABLE_ITN else 'disabled (unverified)'}")
    print(f"SMC: {'ENABLED' if ENABLE_SMC else 'disabled (unverified)'}")
    print("Vaccine: NOT YET WIRED")
    print(f"Experiment ID: {experiment.id}")
    print("==============================================================\n")

    return experiment


#########################################################################################
# MAIN
#########################################################################################

if __name__ == "__main__":
    selected_platform = "SLURM_LOCAL"
    general_sim(selected_platform)