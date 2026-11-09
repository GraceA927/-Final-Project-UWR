"""
Rank habitat mixes from `Site_Setup_UWR.py --step test_season` by how well the
TIMING of simulated clinical cases matches monthly_U5_conf.csv.

Reads <experiment dir>/<sim>/output/InsetChart.json + tags.json directly, drops
the first SPINUP_YEARS, folds the rest into 12 x 30-day blocks and correlates
each run's mean seasonal shape with the observed mean shape (2015-2025).

Notes
- Test runs are burn-ins (no SMC/IRS/ITN), all ages: this screens the
  transmission season. The pickup's SMC (Jul-Oct, U5) will flatten the U5
  peak somewhat, so prefer mixes with r clearly > 0.6.
- The PfPR level is not equilibrated after 10 years; it is shown only as a
  check that transmission did not die out.

Usage: python check_season_test.py [experiment_dir]
       (default: newest experiments/*SEASONTEST* directory)
"""

import glob
import json
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
OBS_FILE = os.path.join(PROJECT_DIR, "inputs", "monthly_U5_conf.csv")
SPINUP_YEARS = 4
N_BLOCKS = 12
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
TAGS = ["x_Temporary_Larval_Habitat", "scale_constant", "scale_temp_rain",
        "scale_water_veg"]


def observed_profile():
    o = pd.read_csv(OBS_FILE)
    o["block"] = o["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    o["share"] = o["adj1_conf_u5"] / o.groupby("year")["adj1_conf_u5"].transform("sum")
    return o.groupby("block")["share"].mean()


def block_profile(daily, spinup_days):
    x = np.asarray(daily, dtype=float)[spinup_days:]
    n_years = len(x) // 365
    x = x[: n_years * 365].reshape(n_years, 365)
    blocks = np.minimum(np.arange(365) // 30, N_BLOCKS - 1)
    per_year = np.stack([np.bincount(blocks, weights=row, minlength=N_BLOCKS)
                         for row in x])
    per_year = per_year / per_year.sum(axis=1, keepdims=True)
    return per_year.mean(axis=0)


def main():
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        dirs = sorted(glob.glob(os.path.join(PROJECT_DIR, "experiments", "*SEASONTEST*")),
                      key=os.path.getmtime)
        if not dirs:
            raise SystemExit("No experiments/*SEASONTEST* directory found.")
        exp_dir = dirs[-1]
    print(f"Experiment directory: {exp_dir}")

    obs = observed_profile().to_numpy()
    rows, profiles, missing = [], {}, 0
    for sim in sorted(glob.glob(os.path.join(exp_dir, "*", ""))):
        ic_path = os.path.join(sim, "output", "InsetChart.json")
        tag_path = os.path.join(sim, "tags.json")
        if not (os.path.isfile(ic_path) and os.path.isfile(tag_path)):
            missing += 1
            continue
        tags = json.load(open(tag_path))
        tags = tags.get("tags", tags)
        ch = json.load(open(ic_path))["Channels"]
        spin = SPINUP_YEARS * 365
        cases = block_profile(ch["New Clinical Cases"]["Data"], spin)
        vec = block_profile(ch["Adult Vectors"]["Data"], spin)
        key = tuple(float(tags[t]) for t in TAGS)
        profiles[key] = cases
        rows.append({
            **dict(zip(TAGS, key)),
            "season_r": np.corrcoef(obs, cases)[0, 1],
            "peak_cases": int(np.argmax(cases) + 1),
            "peak_vectors": int(np.argmax(vec) + 1),
            "pfpr_last_year": float(np.mean(
                ch["Blood Smear Parasite Prevalence"]["Data"][-365:])),
        })
    if not rows:
        raise SystemExit(f"No finished simulations in {exp_dir} ({missing} not ready).")

    df = pd.DataFrame(rows).sort_values("season_r", ascending=False)
    print(f"\n{len(df)} runs read, {missing} not finished.  Observed peak month: "
          f"{int(np.argmax(obs) + 1)}")
    print(df.round(3).to_string(index=False))
    df.to_csv(os.path.join(exp_dir, "season_test_ranking.csv"), index=False)

    # Shape plot: observed vs the best and the old constant-dominated mix.
    with mpl.rc_context({"font.size": 8, "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        m = np.arange(1, N_BLOCKS + 1)
        for key, prof in profiles.items():
            ax.plot(m, prof, color="firebrick", alpha=0.12, linewidth=0.6)
        best = df.iloc[0]
        bkey = tuple(best[t] for t in TAGS)
        ax.plot(m, profiles[bkey], color="firebrick", linewidth=1.6,
                label=(f"Best: const={bkey[1]:g}, rain={bkey[2]:g}, veg={bkey[3]:g}, "
                       f"xTLH={bkey[0]:g} (r={best['season_r']:.2f})"))
        ax.plot(m, obs, "-o", color="black", markersize=3, linewidth=1.2,
                label="Observed confirmed U5 (mean share)")
        ax.set_xticks(m)
        ax.set_xticklabels([s[0] for s in MONTHS])
        ax.set_ylabel("Share of annual cases")
        ax.set_xlabel("Month (30-day block)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        out = os.path.join(exp_dir, "season_test_profiles.png")
        fig.savefig(out, dpi=300)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
