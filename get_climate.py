import os
from emodpy_malaria.weather import (
    WeatherArgs,
    WeatherRequest,
    WeatherVariable,
    weather_to_csv,
    csv_to_weather,
)

import manifest

# =============================================================================
# UPPER WEST REGION EMOD CLIMATE (COMPS / Calculon)
# =============================================================================
#
# Generate daily EMOD weather files for the Upper West Region via IDM's
# COMPS-hosted weather service (Calculon environment).
#
# Target modelling period:
#
#     1959-01-01 ------------------------------- 2025-12-31
#          |                                         |
#          |<------ 50-year burn-in ------>|         |
#          |                               |         |
#        2009                          historical   RTS,S
#                                      period
#
# Output is written to the project's existing emod_weather_upper_west/
# directory (project root), matching what run_example_Burnin.py already
# reads from -- NOT the stock example's simulation_inputs/<tag>/ layout.
# =============================================================================

TAG = "upper_west_region"
LABEL = "historical_1959_2025"

START_YEAR = "1959"
START_DAY = "001"
END_YEAR = "2025"
END_DAY = "365"

# Your project's real demographics file: columns are name, lat, lon, nodes.
# (The stock example used a separate uwr_sites.csv with a node_id column --
# this project doesn't have that file, so we point at demographics.csv and
# override node_column to match its actual column name.)
SITE_FILE = os.path.join(manifest.input_dir, "demographics.csv")
NODE_COLUMN = "nodes"

# Matches the flat layout your burn-in script (run_example_Burnin.py) reads
# from: manifest.CURRENT_DIR/emod_weather_upper_west/*.bin(.json)
OUTPUT_DIR = os.path.join(manifest.CURRENT_DIR, "emod_weather_upper_west")


def get_climate(
    tag=TAG,
    label=LABEL,
    start_year=START_YEAR,
    start_day=START_DAY,
    end_year=END_YEAR,
    end_day=END_DAY,
    site_file=SITE_FILE,
    node_column=NODE_COLUMN,
    output_dir=OUTPUT_DIR,
    fix_temp=None,
    rain_shift=None,
):
    # -------------------------------------------------------------------------
    # DATE RANGE
    # -------------------------------------------------------------------------

    start = "".join((start_year, start_day))
    end = "".join((end_year, end_day))

    # -------------------------------------------------------------------------
    # SITE FILE CHECK
    # -------------------------------------------------------------------------

    if not os.path.isfile(site_file):
        raise FileNotFoundError(f"Demographics/site file not found:\n{site_file}")

    # -------------------------------------------------------------------------
    # EXISTING OUTPUT CHECK
    # -------------------------------------------------------------------------

    if os.path.exists(output_dir) and fix_temp is None and rain_shift is None:
        existing = sorted(os.listdir(output_dir))
        if existing:
            print("Output directory already has files. Please check before overwriting:")
            print(output_dir)
            print("Existing files:", existing)
            return

    print("=" * 80)
    print("UPPER WEST REGION EMOD CLIMATE (COMPS / Calculon)")
    print("=" * 80)
    print(f"Site file : {site_file}")
    print(f"Node col  : {node_column}")
    print(f"Start     : {start_year}-{start_day}")
    print(f"End       : {end_year}-{end_day}")
    print(f"Output    : {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, "weather.csv")

    # -------------------------------------------------------------------------
    # WEATHER REQUEST
    # -------------------------------------------------------------------------

    wa = WeatherArgs(
        site_file=site_file,
        start_date=int(start),
        end_date=int(end),
        node_column=node_column,
        id_reference=tag,
    )

    wr = WeatherRequest(platform="Calculon")

    print("\nRequesting weather from COMPS weather service...")
    wr.generate(weather_args=wa, request_name=tag)
    print("Weather request generated.")

    wr.download(local_dir=output_dir)

    print("\nOriginal weather files downloaded to:")
    print(output_dir)

    # -------------------------------------------------------------------------
    # SHOW WHAT WAS ACTUALLY DOWNLOADED
    #
    # COMPS weather downloads commonly use names like
    # "dtk_15arcmin_air_temperature_daily.bin", NOT the plain
    # "air_temperature.bin" that run_example_Burnin.py expects.
    # Check this list against FILE_NAMES in your burn-in script and
    # rename/symlink or update FILE_NAMES there if they don't match.
    # -------------------------------------------------------------------------

    downloaded = sorted(
        f for f in os.listdir(output_dir) if f.endswith(".bin") or f.endswith(".bin.json")
    )

    print("\nFiles present in output directory:")
    for f in downloaded:
        print(f"  {f}")

    if not any(f in ("air_temperature.bin", "land_temperature.bin", "rainfall.bin", "relative_humidity.bin")
               for f in downloaded):
        print(
            "\nNOTE: none of these match the plain names "
            "(air_temperature.bin, land_temperature.bin, rainfall.bin, "
            "relative_humidity.bin) that run_example_Burnin.py's FILE_NAMES "
            "expects. You will need to rename these or update FILE_NAMES "
            "in the burn-in script to match."
        )

    # -------------------------------------------------------------------------
    # OPTIONAL WEATHER MODIFICATION
    # -------------------------------------------------------------------------

    if rain_shift is not None or fix_temp is not None:
        print("\nApplying requested weather modifications...")

        df, wa = weather_to_csv(weather_dir=output_dir, csv_file=csv_file)

        specs = ""

        if rain_shift is not None:
            rain_shift = int(rain_shift)
            max_step = df["steps"].max()
            specs = "_".join((specs, str(rain_shift), "shift"))

            df["steps"] = df["steps"] - rain_shift
            df.loc[df["steps"] <= 0, "steps"] = df.loc[df["steps"] <= 0, "steps"] + max_step

        if fix_temp is not None:
            fix_temp = float(fix_temp)
            specs = "_".join((specs, str(fix_temp), "degrees"))
            df = df.assign(airtemp=fix_temp, landtemp=fix_temp)

        df.to_csv(csv_file, index=False)

        weather_columns = {
            WeatherVariable.AIR_TEMPERATURE: "airtemp",
            WeatherVariable.LAND_TEMPERATURE: "landtemp",
            WeatherVariable.RELATIVE_HUMIDITY: "humidity",
            WeatherVariable.RAINFALL: "rainfall",
        }

        weather_filenames = {
            WeatherVariable.AIR_TEMPERATURE: "air_temperature.bin",
            WeatherVariable.LAND_TEMPERATURE: "land_temperature.bin",
            WeatherVariable.RELATIVE_HUMIDITY: "relative_humidity.bin",
            WeatherVariable.RAINFALL: "rainfall.bin",
        }

        revised_dir = os.path.join(output_dir, specs.strip("_"))
        os.makedirs(revised_dir, exist_ok=True)

        ws2 = csv_to_weather(
            csv_data=df,
            attributes=wa,
            weather_dir=revised_dir,
            weather_columns=weather_columns,
            weather_file_names=weather_filenames,
        )

        ws2.to_files(dir_path=revised_dir)

        print("\nRevised EMOD weather files written to:")
        print(revised_dir)


if __name__ == "__main__":
    get_climate()