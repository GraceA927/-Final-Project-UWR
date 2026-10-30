"""
Build the projection weather (day 1 = 1 Jan 2026, 2,555 days) from the pickup
weather (day 1 = 1 Jan 2013, 4,745 days, flat 365-day years).

The last BLOCK_YEARS years of the pickup weather are repeated to fill the
projection. Every scenario arm uses these same files. This is an ASSUMPTION:
future weather = a repeat of recent observed years. State it in the methods.

    python make_projection_climate.py
"""

import json
import os

import numpy as np

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
SRC = os.path.join(PROJECT_DIR, "climate", "pickup")
DST = os.path.join(PROJECT_DIR, "climate", "projection")

PROJ_START_YEAR = 2026
PROJ_END_YEAR = 2032
N_DAYS = (PROJ_END_YEAR - PROJ_START_YEAR + 1) * 365   # 2555

# Last N years of the pickup weather to repeat. 7 = 2019-2025 as 2026-2032.
# 5 = 2021-2025 repeated (2026-2030), then 2021-2022 for 2031-2032.
BLOCK_YEARS = 7

VARS = ["air_temperature", "land_temperature", "rainfall", "relative_humidity"]


def source_dir():
    if os.path.isfile(os.path.join(SRC, "air_temperature.bin")):
        return SRC
    subs = [d for d in sorted(os.listdir(SRC))
            if os.path.isfile(os.path.join(SRC, d, "air_temperature.bin"))]
    if len(subs) != 1:
        raise SystemExit(f"Cannot pick a weather folder in {SRC}: {subs}")
    return os.path.join(SRC, subs[0])


def main():
    src = source_dir()
    os.makedirs(DST, exist_ok=True)
    print(f"Source: {src}\nTarget: {DST}")

    for var in VARS:
        with open(os.path.join(src, f"{var}.bin.json")) as f:
            meta = json.load(f)
        m = meta.get("Metadata", {})

        data = np.fromfile(os.path.join(src, f"{var}.bin"), dtype="<f4")

        node_count = int(m.get("NodeCount", 1))
        if node_count != 1:
            raise SystemExit(f"{var}: NodeCount={node_count}; this script handles 1 node.")
        if int(m.get("DatavalueCount", len(data))) != len(data):
            raise SystemExit(f"{var}: .bin has {len(data)} values but the json says "
                             f"{m.get('DatavalueCount')}. Not a single-node float32 file?")

        block_days = BLOCK_YEARS * 365
        if len(data) < block_days:
            raise SystemExit(f"{var}: only {len(data)} days, need {block_days}.")

        block = data[-block_days:]
        out = np.tile(block, int(np.ceil(N_DAYS / len(block))))[:N_DAYS].astype("<f4")
        out.tofile(os.path.join(DST, f"{var}.bin"))

        m["DatavalueCount"] = int(N_DAYS)
        m["OriginalDataYears"] = f"{PROJ_START_YEAR}-{PROJ_END_YEAR}"
        m["DataProvenance"] = (f"Repeat of the last {BLOCK_YEARS} pickup years "
                               f"({len(data) // 365 - BLOCK_YEARS + 2013}-{len(data) // 365 + 2012})")
        meta["Metadata"] = m
        with open(os.path.join(DST, f"{var}.bin.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"  {var:<18} {len(data)} -> {len(out)} days   "
              f"mean source {data.mean():.3f}  mean projection {out.mean():.3f}")

    print("\nDone. Day 1 of the projection weather = 1 Jan "
          f"{PROJ_START_YEAR}. Re-run run_scenarios_UWR.py --dry-run.")


if __name__ == "__main__":
    main()