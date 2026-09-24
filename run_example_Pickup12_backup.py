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
# Historical interventions included:
#   - Case management:       2013-2022
#   - ITNs:                  2014, 2016, 2022 (with per-year age-usage and
#                             seasonal-usage patterns)
#   - IRS:                   2015-2024
#   - SMC:                   2015-2022  [wired via add_drug_campaign -- unverified in
#                             this venv, see SMC block below]
#   - Malaria vaccine:       2024-2025  [NOT YET WIRED -- see vaccine block below]
#
# IG2 is intentionally NOT included in the historical pickup.
#
# Climate:
#   Historical NASA POWER climate for 2013-2025
#   One regional node (node_id = 1)
#
# STATUS:
#   This version fixes:
#     1. End-of-pickup serialization (previously "NONE", which blocked Aim 2
#        entirely -- there was no calibrated end-state for any counterfactual
#        scenario to read from).
#     2. ITN age-usage/seasonal-usage data, previously loaded but never wired
#        into the ITN campaign call.
#     3. A set of plumbing imports/API calls (idmtools.entities.Experiment,
#        emodpy_malaria demographics/manifest imports, EMODTask construction,
#        Platform construction, manifest.schema_* naming, task.set_sif) that
#        were guessed rather than copied from burnin_UpperWest.py, and one of
#        which (the Experiment import) has been CONFIRMED to crash at import
#        time: `cannot import name 'Experiment' from 'idmtools.entities'`.
#        All of these are now byte-for-byte aligned with burnin_UpperWest.py's
#        equivalents, since that script is the one proven to actually run in
#        this project's venv.
#     4. build_campaign()'s `camp = conf.Campaign()` -- conf is
#        emodpy_malaria.malaria_config, which has no Campaign() class. Both
#        burnin_UpperWest.py and an independent reference example in this
#        project use `import emod_api.campaign as camp` at module level and
#        pass that module directly; fixed to match.
#     5. add_case_management()'s call shape, confirmed wrong against the
#        same reference example: drug is a top-level list of full component
#        names (not the CSV's "AL" code per-target), targets don't carry
#        'seek'/'drug' keys, and node_ids/ind_property_restrictions/
#        broadcast_event_name are required. See DRUG_NAME_MAP and
#        add_case_management's docstring.
#     6. SMC is now wired via add_drug_campaign (see caveat below) -- was
#        previously unimplemented.
#     7. The per-year report filename_suffix was "_U5_{year}"; both
#        burnin_UpperWest.py and the reference example above use
#        "Monthly_U5_{year}" -- fixed to match, so pickup output can
#        eventually be read by the same analyzer built for burn-in.
#
#   SMC's add_drug_campaign call is a best-effort implementation, NOT
#   confirmed against this project's installed emodpy_malaria version the
#   way case management now is -- no working reference example was
#   available for SMC specifically. See add_smc_campaign's comment block
#   for the verification command to run before your next submission.
#
#   The malaria vaccine is still NOT applied to the campaign -- no
#   reference example or CSV-driven implementation has been done for it
#   yet. See add_vaccine_campaign's docstring for the verification steps
#   needed before filling it in.
#
#   STILL UNVERIFIED (no reference example exists to check these against,
#   and the track record above means "probably fine" is not a safe
#   assumption -- verify with inspect.signature() before your next run
#   rather than finding out from another crash):
#     - from emodpy_malaria.interventions import insecticide (IRS call shape)
#     - Demographics.from_csv(input_file=..., id_ref=...) and its exact
#       accepted kwargs
#     - bednet.add_uncomplicated_bednet()'s age_dependence / seasonal_dependence
#       keyword names (see add_itn_campaign's docstring below)
#     - add_drug_campaign()'s exact kwargs for SMC (see add_smc_campaign)
#
#   The calibration comparison against prevalence_df (MICS/GDHS/GMIS
#   survey points) is also not yet implemented -- this script produces
#   *a* pickup simulation, not yet *the calibrated* one Aim 2.1 requires.
#########################################################################################

import os
import pathlib
import getpass
import pandas as pd

from functools import partial

import emodpy_malaria.malaria_config as conf
from emodpy.emod_task import EMODTask
import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emod_api.campaign as camp

from emodpy_malaria.interventions import treatment_seeking as cm
from emodpy_malaria.interventions import insecticide
from emodpy_malaria.interventions import usage_dependent_bednet as bednet
from emodpy_malaria.interventions.drug_campaign import add_drug_campaign
from emodpy_malaria.reporters.builtin import add_malaria_summary_report

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

import manifest


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
# is an Aim-2 script decision, not a pickup-correctness fix -- flagging so
# it isn't forgotten.
NUM_PICKUP_SEEDS = 1


#########################################################################################
# BURN-IN SETTINGS
#########################################################################################

# Set this to the completed 50-year burn-in experiment ID (or export
# BURNIN_EXPERIMENT_ID before running).
BURNIN_EXPERIMENT_ID = os.environ.get("BURNIN_EXPERIMENT_ID", "")

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
    required_files = [
        DEMOGRAPHICS_FILE, VECTOR_FILE, CM_FILE, IRS_FILE, ITN_FILE,
        ITN_AGE_FILE, ITN_SEASON_FILE, SMC_FILE, VACCINE_FILE, PREVALENCE_FILE,
    ]

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
# LOAD INTERVENTION DATA
#########################################################################################

def load_intervention_data():
    cm_df = pd.read_csv(CM_FILE)
    irs_df = pd.read_csv(IRS_FILE)
    itn_df = pd.read_csv(ITN_FILE)
    itn_age_df = pd.read_csv(ITN_AGE_FILE)
    itn_season_df = pd.read_csv(ITN_SEASON_FILE)
    smc_df = pd.read_csv(SMC_FILE)
    vaccine_df = pd.read_csv(VACCINE_FILE)
    prevalence_df = pd.read_csv(PREVALENCE_FILE)

    return cm_df, irs_df, itn_df, itn_age_df, itn_season_df, smc_df, vaccine_df, prevalence_df


#########################################################################################
# FILTER HISTORICAL PERIOD
#########################################################################################

def filter_historical_data(df):
    if "year" not in df.columns:
        return df.copy()

    return df[(df["year"] >= PICKUP_START_YEAR) & (df["year"] <= PICKUP_END_YEAR)].copy()


#########################################################################################
# BUILD DEMOGRAPHICS
#########################################################################################

def build_demographics():
    # UNVERIFIED: from_csv()/id_ref kwarg names have not been checked against
    # this project's installed emodpy_malaria.demographics.MalariaDemographics.
    # Run:
    #   python -c "
    #   import inspect
    #   import emodpy_malaria.demographics.MalariaDemographics as Demographics
    #   print(inspect.signature(Demographics.from_csv))
    #   "
    # before your next run, given the import-path track record on this script.
    return Demographics.from_csv(
        input_file=DEMOGRAPHICS_FILE,
        id_ref="Gridded world grump2.5arcmin",
    )


#########################################################################################
# BUILD CONFIGURATION
#########################################################################################

def set_param_fn(config):
    config = conf.set_team_defaults(config, manifest)

    config.parameters.Simulation_Duration = PICKUP_DAYS
    config.parameters.Run_Number = 0

    config.parameters.Climate_Model = "CLIMATE_BY_DATA"
    config.parameters.Air_Temperature_Filename = "air_temperature.bin"
    config.parameters.Land_Temperature_Filename = "land_temperature.bin"
    config.parameters.Rainfall_Filename = "rainfall.bin"
    config.parameters.Relative_Humidity_Filename = "relative_humidity.bin"

    config.parameters.Enable_Local_Migration = 0
    config.parameters.Migration_Model = "NO_MIGRATION"
    config.parameters.Birth_Rate_Dependence = "FIXED_BIRTH_RATE"

    # --- SERIALIZED POPULATION: READ from burn-in ---
    config.parameters.Serialized_Population_Reading_Type = "READ"
    config.parameters.Serialization_Mask_Node_Read = 0
    config.parameters.Serialization_Time_Steps = [SERIALIZATION_DAY]
    config.parameters.Serialization_Precision = "REDUCED"

    # --- SERIALIZED POPULATION: WRITE at end of pickup ---
    # This mirrors exactly what burnin_UpperWest.py does at the end of its
    # own run -- without this, no calibrated end-state exists for Aim 2 to
    # branch from.
    config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
    config.parameters.Serialization_Time_Steps = config.parameters.Serialization_Time_Steps + [PICKUP_DAYS]
    config.parameters.Serialization_Mask_Node_Write = 0

    return config


#########################################################################################
# ADD VECTOR SPECIES
#########################################################################################

def add_vector_species(config):
    vector_df = pd.read_csv(VECTOR_FILE)

    if vector_df["node_id"].nunique() != 1:
        raise ValueError("This UWR model is configured as a single regional node. vector.csv must contain only one node_id.")

    return config


#########################################################################################
# CASE MANAGEMENT
#
# DRUG_NAME_MAP: interventions_CM.csv stores drug as a short code ("AL");
# cm.add_treatment_seeking() -- confirmed against a working reference
# example in this project -- takes a top-level `drug` list of full
# component names, not a per-target code. Extend this map if a new code
# ever appears in the CSV; the loop below raises rather than guessing.
#########################################################################################

DRUG_NAME_MAP = {
    "AL": ["Artemether", "Lumefantrine"],
}


def add_case_management(camp, cm_df):
    """
    Apply one treatment-seeking campaign event per row of interventions_CM.csv.
    Each row's own start_day/duration defines its active window -- rows are
    NOT assumed contiguous, so gaps or overlaps in the CSV are applied exactly
    as given, not inferred.

    Call shape confirmed against a working reference example in this project:
    positional `camp`, `drug` passed once as a list of full component names,
    `node_ids`/`ind_property_restrictions` as empty lists (apply everywhere,
    no property targeting), `broadcast_event_name` set, and targets carrying
    only trigger/coverage/agemin/agemax/rate -- no per-target 'seek' or
    'drug' keys (those aren't part of the confirmed working call shape).

    The CSV's 'seek' column is currently always 1.0 in this project's data
    and is not applied here, since the confirmed reference call doesn't take
    a per-target seek value. If seek ever varies in future CM data, this
    needs revisiting once a seek-bearing call shape is confirmed.

    `duration` is still passed through, since each CSV row is a calendar-
    year window that needs bounding -- but the reference example that
    confirmed the rest of this call shape didn't exercise `duration`, so
    it's the one part of this call still unconfirmed. If case management
    doesn't transition cleanly between calendar years, verify with:

        python -c "
        import inspect
        from emodpy_malaria.interventions import treatment_seeking as cm
        print(inspect.signature(cm.add_treatment_seeking))
        "
    """
    cm_df = filter_historical_data(cm_df)

    for _, row in cm_df.iterrows():
        start_day = int(row["start_day"])
        duration = int(row["duration"])
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

        cm.add_treatment_seeking(
            camp,
            start_day=start_day,
            duration=duration,
            drug=drug,
            node_ids=[],
            ind_property_restrictions=[],
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
#########################################################################################

def add_irs_campaign(camp, irs_df):
    """
    Apply one IRS round per row of interventions_IRS.csv, using that row's own
    start_day, repetitions, and timesteps_between_repetitions -- each round's
    timing comes entirely from the CSV, not computed here.
    """
    irs_df = filter_historical_data(irs_df)

    for _, row in irs_df.iterrows():
        insecticide.add_irs(
            campaign=camp,
            start_day=int(row["start_day"]),
            coverage=float(row["coverage"]),
            killing_initial_effect=float(row["killing_initial_effect"]),
            killing_box_duration=int(row["killing_box_duration"]),
            killing_decay_time_constant=float(row["killing_decay_time_constant"]),
            repelling_initial_effect=float(row["repelling_initial_effect"]),
            repelling_box_duration=int(row["repelling_box_duration"]),
            repelling_decay_time_constant=float(row["repelling_decay_time_constant"]),
            repetitions=int(row["repetitions"]),
            timesteps_between_repetitions=int(row["timesteps_between_repetitions"]),
            insecticide_name=str(row["insecticide"]),
        )


#########################################################################################
# ITN
#########################################################################################

def add_itn_campaign(camp, itn_df, itn_age_df, itn_season_df):
    """
    Apply one ITN distribution per row of interventions_ITN.csv (2014, 2016,
    2022), each carrying that distribution year's own age-usage pattern
    (interventions_ITN_age.csv) and the project's seasonal usage pattern
    (interventions_ITN_season.csv).

    age_usage and seasonal_usage are wired into
    bednet.add_uncomplicated_bednet() as age_dependence / seasonal_dependence.

    VERIFY BEFORE RUNNING: this project's installed usage_dependent_bednet
    module has not been checked for the exact keyword names it expects for
    age- and season-dependent usage. Run:

        python -c "
        import inspect
        from emodpy_malaria.interventions import usage_dependent_bednet as bednet
        print(inspect.signature(bednet.add_uncomplicated_bednet))
        "

    and adjust age_dependence / seasonal_dependence below (or replace them
    with whatever the real parameter names/shapes are) if they don't match.
    """
    itn_df = filter_historical_data(itn_df)
    itn_age_df = filter_historical_data(itn_age_df)

    # Seasonal usage doesn't vary by distribution year in the source CSV
    # (5 rows total, not one set per year) -- built once here.
    seasonal_dependence = [
        {"month": int(row["month"]), "usage": float(row["seasonal_usage"])}
        for _, row in itn_season_df.iterrows()
    ] if "month" in itn_season_df.columns and "seasonal_usage" in itn_season_df.columns else None

    for _, row in itn_df.iterrows():
        year = int(row["year"])

        year_age_df = itn_age_df[itn_age_df["year"] == year]
        age_dependence = [
            {"age": float(age_row["age"]), "usage": float(age_row["age_usage"])}
            for _, age_row in year_age_df.iterrows()
        ]

        bednet.add_uncomplicated_bednet(
            campaign=camp,
            start_day=int(row["start_day"]),
            coverage=float(row["coverage"]),
            killing_initial_effect=float(row["kill_effect"]),
            killing_decay_time_constant=float(row["kill_decay"]),
            blocking_initial_effect=float(row["block_effect"]),
            blocking_decay_time_constant=float(row["block_decay"]),
            discard_k=float(row["discard_k"]),
            discard_l=float(row["discard_l"]),
            age_dependence=age_dependence if age_dependence else None,
            seasonal_dependence=seasonal_dependence,
        )


#########################################################################################
# SMC
#
# UNVERIFIED IN THIS VENV: unlike case management above, no working
# reference example for SMC has been available to confirm this call shape
# against. add_drug_campaign is emodpy_malaria's standard SMC/MDA delivery
# function across most published versions, and campaign_type='SMC' with
# drug_code='SPA' (sulfadoxine-pyrimethamine + amodiaquine) is the standard
# WHO SMC regimen -- but the exact argument names in THIS project's
# installed version have not been checked. Before your next run:
#
#     python -c "
#     import inspect
#     from emodpy_malaria.interventions.drug_campaign import add_drug_campaign
#     print(inspect.signature(add_drug_campaign))
#     "
#
# and adjust drug_code / target_group / any renamed kwargs below if they
# don't match. Guessing wrong here risks calibrating against an SMC history
# that doesn't match what was actually delivered.
#########################################################################################

def add_smc_campaign(camp, smc_df):
    """
    interventions_SMC.csv has an irregular number of rounds per year (2 in
    2016, 4 in most other years) at irregular spacing -- not a fixed
    interval -- so each row is applied as its own single-round call
    (repetitions=1) rather than using add_drug_campaign's own repeat-
    interval mechanism, which assumes evenly spaced rounds.
    """
    smc_df = filter_historical_data(smc_df)

    for _, row in smc_df.iterrows():
        start_day = int(row["start_day"])
        coverage = float(row["coverage"])

        add_drug_campaign(
            camp,
            campaign_type='SMC',
            drug_code='SPA',
            start_days=[start_day],
            coverage=coverage,
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

    Before filling this in, run the same emodpy_malaria.interventions
    submodule listing as in add_smc_campaign's docstring, then look for the
    vaccine-related submodule/function (e.g. something exposing
    SimpleVaccine / MultiEffectVaccine per-dose scheduling) and check its
    signature with inspect.signature before calling it here.
    """
    vaccine_df = filter_historical_data(vaccine_df)  # noqa: F841 -- kept for when this is wired in
    raise NotImplementedError(
        "add_vaccine_campaign is not yet implemented -- see function "
        "docstring for the verification steps needed before wiring in the "
        "actual EMOD vaccine intervention call."
    )


#########################################################################################
# BUILD CAMPAIGN
#########################################################################################

def build_campaign():
    camp.set_schema(manifest.schema_file)

    (cm_df, irs_df, itn_df, itn_age_df, itn_season_df,
     smc_df, vaccine_df, prevalence_df) = load_intervention_data()

    add_case_management(camp, cm_df)
    add_irs_campaign(camp, irs_df)
    add_itn_campaign(camp, itn_df, itn_age_df, itn_season_df)
    add_smc_campaign(camp, smc_df)

    # NOT YET CALLED -- see add_vaccine_campaign docstring. Calling it now
    # will raise NotImplementedError by design, rather than silently
    # omitting the vaccine from the campaign.
    # add_vaccine_campaign(camp, vaccine_df)

    # IMPORTANT: NO IG2 is added to this historical pickup.
    # IG2 belongs to Aim 2 counterfactual simulations after calibration.

    return camp


#########################################################################################
# CREATE EMOD TASK
#########################################################################################

def create_task():
    # Mirrors burnin_UpperWest.py's proven EMODTask.from_default2(...) call
    # exactly, rather than the previous conf.get_config_from_template(...) +
    # EMODTask.from_default(config=...) combination, which was unverified
    # and used a manifest.schema_path attribute that doesn't match burn-in's
    # confirmed manifest.schema_file.
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
    task.common_assets.add_file(VECTOR_FILE, relative_path="vector.csv")

    return task


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

    # Mirrors burnin_UpperWest.py's explicit Platform(...) construction
    # exactly, rather than an unverified manifest.get_platform() call.
    platform = Platform(
        selected_platform,
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )

    task = create_task()

    # Required for the job to actually run in the SLURM container -- present
    # in burnin_UpperWest.py, was missing here entirely.
    task.set_sif(manifest.SIF_PATH, platform)

    builder = SimulationBuilder()

    # One pickup per serialized burn-in state. Do NOT add an independent
    # Run_Number sweep here -- the pickup Run_Number is inherited from the
    # burn-in so each historical pickup starts from its corresponding
    # immune/transmission history.
    burnin_df = get_burnin_df(BURNIN_EXPERIMENT_ID, platform)

    builder.add_sweep_definition(
        partial(update_serialize_parameters, df=burnin_df),
        range(len(burnin_df.index)),
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
        "SMC": "Wired (unverified API)",
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
    print(f"Pickup end-state serialized at day: {PICKUP_DAYS}")
    print("IG2: NOT INCLUDED")
    print("SMC: WIRED (add_drug_campaign call unverified in this venv -- see add_smc_campaign)")
    print("Vaccine: NOT YET WIRED (see add_vaccine_campaign)")
    print(f"Experiment ID: {experiment.id}")
    print("==============================================================\n")

    return experiment


#########################################################################################
# MAIN
#########################################################################################

if __name__ == "__main__":
    selected_platform = "SLURM_LOCAL"
    general_sim(selected_platform)