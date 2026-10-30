"""
Aim 2 analysis of the IRS-to-IG2 scenarios, matched to run_scenarios_UWR.py.

AIM 2  Estimate the epidemiological impact and the IG2 effective-use coverage
       required to replace IRS in the Upper West Region of Ghana.
  2.1  Using the calibrated model from Aim 1, compare continued IRS with
       replacement of IRS by IG2 at 50, 60, 80 and 90% EFFECTIVE-USE coverage.
  2.2  Estimate clinical malaria incidence, P. falciparum parasite prevalence
       and cases averted under each replacement scenario relative to continued
       IRS, and the minimum IG2 effective-use coverage that maintains or
       reduces malaria burden.

Usage
-----
    python analyze_scenarios_UWR.py --experiment-id <scenario experiment ID>

This script produces Aim 2 outputs only (no monthly view, no net-use
diagnostic, no IRS-effect-recovered table).

Effective use versus distributed coverage
-----------------------------------------
The scenario script varies c, the coverage DISTRIBUTED in each IG2 campaign.
Effective use (share of people using an IG2 net) is lower than c:

    use(day) = sum over rounds i of  c (1-c)^j x R(day - d_i) x A_i x S_i(doy)

  R = Weibull net retention, (1-c)^j = chance the round-i net was not replaced
  by one of the j later rounds, A_i = population-weighted mean of the MASS
  age-usage curve, S_i = MASS seasonal-usage curve.

"Effective-use coverage" in this script = the mean of use(day) over the analysis
window (--use-metric use_window, the default). Alternatives: use_first30d_round1
(best case, new nets) or use_mean_cycle1.

All Aim 2 results are expressed on the EFFECTIVE-USE axis:
  * every simulated c is mapped to its effective use;
  * outcomes at the 50/60/80/90% targets are read by interpolating between the
    simulated arms, pickup state by pickup state, on that axis. There is no
    extrapolation: a target outside the simulated range is labelled NOT
    SIMULATED (with the c to run) or UNREACHABLE (cannot be met even at c = 1);
  * the minimum effective use that maintains or reduces burden is found
    directly on the effective-use axis, with a bootstrap over pickup states,
    and is also given as the equivalent distributed c.

Sign convention: every difference is  IRS_CONTINUE minus scenario, so a
POSITIVE value is a benefit (cases averted, PfPR reduction, incidence reduction).

Files written (suffix _aim2, in <experiment name>_Aim2Output/)
-------------------------------------------------------------
  sim_index_aim2.csv, scenario_results_aim2.csv      index and extracted sums
  aim2_effective_use_by_coverage_aim2.csv            c -> effective use (all arms)
  aim2_use_to_coverage_lookup_aim2.csv               c -> use grid (0.01 steps)
  aim2_scenario_outcomes_aim2.csv                    incidence, PfPR, cases, every arm
  aim2_outcomes_by_year_aim2.csv                     the same by year
  aim2_benefit_vs_irs_aim2.csv                       averted vs IRS, every arm
  aim2_benefit_vs_irs_by_year_aim2.csv               the same by year
  aim2_effective_use_targets_aim2.csv                50/60/80/90% targets
  aim2_minimum_effective_use_aim2.csv                minimum effective use
  aim2_outcomes_vs_effective_use_aim2.png/pdf        figure
  aim2_benefit_vs_effective_use_aim2.png/pdf         figure
  run_info_aim2.json

Useful variations
-----------------
  --reuse-index / --reuse-results   skip idmtools / extraction
  --exp-dir PATH                    scan an experiment folder instead of idmtools
  --targets-use 0.5 0.6 0.8 0.9     effective-use targets
  --use-metric use_window|use_first30d_round1|use_mean_cycle1
  --margin-pct 5                    accept IG2 up to 5% worse than IRS (decide BEFORE looking)
  --analysis-start-year 2028        sensitivity run ignoring residual 2026-27 effects
  --no-ig2-curves                   scenario ran with IG2_USE_CURVES = False

Checks that run automatically
-----------------------------
  tags present; one simulation per (scenario, pickup state); failed simulations
  dropped; a pickup state with any incomplete arm is dropped from ALL arms;
  pairing check (before --switch-year every arm must equal IRS_CONTINUE);
  report keys and age bins verified on the first file.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Must match run_scenarios_UWR.py ----------------------------------------
PROJ_START_YEAR = 2026                      # PROJECTION_START_YEAR
PROJ_END_YEAR = 2032                        # PROJECTION_END_YEAR
DAYS_PER_YEAR = 365
REPORT_AGE_EDGES = [0.0, 0.25, 0.5, 5.0, 15.0, 25.0, 100.0]   # 0 + REPORT_AGE_BINS
N_AGE_BINS = len(REPORT_AGE_EDGES) - 1
U5_BINS = [0, 1, 2]          # 0-3 m, 3-6 m, 6-59 m
PFPR_U5_BIN = 2              # 0.5-5 y = 6-59 months
PERIOD_DAYS = 30             # REPORT_INTERVAL
PERIODS_USED = 12            # period 13 is partial; dropped in every arm
FILE_PATTERN = "MalariaSummaryReport_Monthly_U5_{year}.json"

# ---- IG2 net parameters (must match IG2 / IG2_* in run_scenarios_UWR.py) -----
IG2_DISCARD_K = 3.0          # IG2["discard_k"]
IG2_DISCARD_L = 1095.0       # IG2["discard_l"]
IG2_CAMPAIGN_DOY = 91        # IG2_CAMPAIGN_DOY
IG2_USE_CHANNEL = "MASS"
NODE_ID = 1
INPUT_DIR = "/shared/home/grace/Final-Project-UWR/inputs"   # --input-dir overrides
ITN_AGE_FILE = "interventions_ITN_age.csv"
ITN_SEASON_FILE = "interventions_ITN_season.csv"
PHASES_ACCEPTED = ("pickup", "projection", "scenario")

# ---- Aim 2 settings ----------------------------------------------------------
DEFAULT_TARGETS = [0.50, 0.60, 0.80, 0.90]
USE_METRICS = ["use_window", "use_first30d_round1", "use_mean_cycle1"]

# Tags written by update_simulation() in run_scenarios_UWR.py
TAG_COLUMNS = ["pickup_sim_index", "Run_Number", "scenario", "scenario_kind",
               "ig2_coverage", "ig2_kill_key"]
SIM_KEYS = ["scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key",
            "pickup_sim_index", "Run_Number"]
METRIC_COLS = ["cases_all", "cases_u5", "incidence_all_per1000py",
               "incidence_u5_per1000py", "pfpr_u5", "pfpr_all"]
THRESHOLD_METRICS = ["cases_all", "cases_u5", "pfpr_u5"]
OUTCOME_LABELS = {
    "incidence_all_per1000py": "Clinical incidence, all ages (per 1,000 PY)",
    "incidence_u5_per1000py": "Clinical incidence, under 5 (per 1,000 PY)",
    "pfpr_u5": "PfPR, 6-59 months",
    "pfpr_all": "PfPR, all ages",
    "cases_all": "Clinical cases, all ages (window total)",
    "cases_u5": "Clinical cases, under 5 (window total)",
}
CASE_COLS = ("cases_all", "cases_u5")
SUCCESS_WORDS = ("SUCCEEDED", "SUCCESS", "COMPLETED")

STEP = "aim2"                      # tag added to every output file name
OUT_SUFFIX = "_Aim2Output"         # output folder = <experiment name><OUT_SUFFIX>

# name -> idmtools experiment ID of the SCENARIO experiment (written by
# run_scenarios_UWR.py, NOT the pickup experiment).
EXPERIMENTS = {
    "XTLH_scenarios_IG2": "501cfc20-b86d-464d-9e98-8b2ae8736b03",
}


def out_file(outdir, base):
    """<outdir>/<stem>_<STEP>.<ext>, e.g. aim2_scenario_outcomes_aim2.csv."""
    stem, _, ext = base.rpartition(".")
    return os.path.join(str(outdir), f"{stem}_{STEP}.{ext}")


# =============================================================================
# 1. Simulation index from the idmtools experiment
# =============================================================================

def get_platform():
    """Same platform settings as general_sim() in run_scenarios_UWR.py."""
    import manifest
    from idmtools.core.platform_factory import Platform
    return Platform(
        "SLURM_LOCAL",
        job_directory=manifest.job_directory,
        partition=manifest.partition,
        time=manifest.sim_time,
        modules=[manifest.singularity_module],
        max_running_jobs=manifest.max_running_jobs,
    )


def simulations_from_platform(experiment_id):
    from idmtools.core import ItemType

    platform = get_platform()
    exp = platform.get_item(experiment_id, ItemType.EXPERIMENT, raw=False)
    print(f"Experiment: {getattr(exp, 'name', '?')}  ({experiment_id})")

    sims = list(getattr(exp, "simulations", None) or [])
    if not sims:
        sims = list(platform.get_children(experiment_id, ItemType.EXPERIMENT,
                                          force=True, raw=False))
    if not sims:
        raise SystemExit(f"No simulations found for experiment {experiment_id}.")

    try:
        platform.refresh_status(exp)
    except Exception as exc:                      # status is best effort
        print(f"  note: could not refresh status ({exc}); relying on output files.")

    records = []
    for sim in sims:
        status = getattr(sim, "status", None)
        status = getattr(status, "name", str(status)) if status is not None else ""
        rec = dict(sim.tags or {})
        rec.update(sim_id=str(sim.id), status=str(status).upper(),
                   sim_dir=str(platform.get_directory(sim)))
        records.append(rec)
    return pd.DataFrame(records)


def simulations_from_directory(exp_dir):
    """No-idmtools fallback: simulation folders with a metadata.json holding tags."""
    records = []
    for d in sorted(Path(exp_dir).expanduser().iterdir()):
        meta = d / "metadata.json"
        if not (d.is_dir() and meta.is_file()):
            continue
        try:
            with open(meta) as f:
                tags = json.load(f).get("tags", {})
        except (OSError, ValueError):
            continue
        if not tags:
            continue
        rec = dict(tags)
        rec.update(sim_id=d.name, status="", sim_dir=str(d))
        records.append(rec)
    if not records:
        raise SystemExit(f"No simulation folders with tagged metadata.json in {exp_dir}.")
    return pd.DataFrame(records)


def normalise_index(df):
    missing = [c for c in TAG_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Simulation tags missing column(s) {missing}. Present: "
                         f"{sorted(df.columns)}. Is this a run_scenarios_UWR.py experiment?")
    df = df.copy()
    for c in ("pickup_sim_index", "Run_Number"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ig2_coverage"] = pd.to_numeric(df["ig2_coverage"], errors="coerce")
    bad = df[["pickup_sim_index", "Run_Number", "ig2_coverage"]].isna().any(axis=1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} simulation(s) have blank/non-numeric tags "
                         f"(first: {df.loc[bad, 'sim_id'].iloc[0]}).")
    df["pickup_sim_index"] = df["pickup_sim_index"].astype(int)
    df["Run_Number"] = df["Run_Number"].astype(int)
    key = df["ig2_kill_key"].astype(str).str.strip()
    df["ig2_kill_key"] = key.where(~key.str.lower().isin(["nan", "none", "null"]), "")
    df["scenario"] = df["scenario"].astype(str)
    df["scenario_kind"] = df["scenario_kind"].astype(str)
    if "status" not in df.columns:
        df["status"] = ""
    return df.sort_values(["pickup_sim_index", "scenario"]).reset_index(drop=True)


def check_index(df, ignore_status):
    if df.duplicated(["scenario", "pickup_sim_index"]).any():
        raise SystemExit("More than one simulation for some (scenario, pickup_sim_index). "
                         "Does the experiment mix several submissions?")
    if "irs" not in set(df["scenario_kind"]):
        raise SystemExit("No IRS_CONTINUE simulations in this experiment: "
                         "nothing to compare against.")
    if "ig2" not in set(df["scenario_kind"]):
        raise SystemExit("No IG2 simulations in this experiment.")

    n_states = df["pickup_sim_index"].nunique()
    print(f"\n{len(df)} simulations: {n_states} pickup state(s) x "
          f"{df['scenario'].nunique()} scenario(s)")
    print(df.groupby("scenario").size().rename("simulations").to_string())

    arms = df.groupby("pickup_sim_index")["scenario"].nunique()
    short = arms[arms < df["scenario"].nunique()]
    if len(short):
        print(f"  WARNING: {len(short)} pickup state(s) lack at least one arm "
              f"(e.g. {list(short.index[:5])}); they will be dropped from all arms.")

    st = df["status"].astype(str).str.upper()
    known = st != ""
    if known.any():
        print("\nStatus counts:")
        print(st[known].value_counts().to_string())
        notdone = known & ~st.isin(SUCCESS_WORDS)
        if notdone.any():
            if ignore_status:
                print(f"  NOTE: {int(notdone.sum())} unfinished simulation(s) kept "
                      f"(--ignore-status).")
            else:
                print(f"  WARNING: dropping {int(notdone.sum())} simulation(s) that did not "
                      f"succeed (their pickup states leave all arms). Use --ignore-status "
                      f"to keep them.")
                df = df[~notdone].reset_index(drop=True)
    return df


# =============================================================================
# 2. Extraction of the summary reports
# =============================================================================

def find_series(obj, must, avoid=()):
    """First list stored under a key whose lower-case name has all `must` tokens."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = str(key).lower()
            if (all(t in k for t in must) and not any(a in k for a in avoid)
                    and isinstance(value, list)):
                return key, value
        for value in obj.values():
            found = find_series(value, must, avoid)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_series(value, must, avoid)
            if found is not None:
                return found
    return None


def age_bin_note(data):
    found = find_series(data, ["age bins"])
    if found is None:
        return "note: no 'Age Bins' list in the report; bin edges not verified."
    try:
        upper = [float(x) for x in found[1]]
    except (TypeError, ValueError):
        return "note: 'Age Bins' not numeric; bin edges not verified."
    if len(upper) != N_AGE_BINS or not np.allclose(upper, REPORT_AGE_EDGES[1:]):
        return (f"WARNING: report age bins {upper} differ from the analyzer's upper edges "
                f"{REPORT_AGE_EDGES[1:]}. Fix REPORT_AGE_EDGES / U5_BINS / PFPR_U5_BIN "
                f"before trusting the results.")
    return f"age bins verified: {upper}"


def extract_one(task):
    """Per year x age bin sums for one simulation. Runs in a worker process."""
    sim_id, sim_dir, years, verbose = task
    rows, missing, notes = [], [], []

    for year in years:
        path = os.path.join(sim_dir, "output", FILE_PATTERN.format(year=year))
        if not os.path.isfile(path):
            missing.append(year)
            continue
        with open(path) as f:
            data = json.load(f)

        pop = find_series(data, ["average population", "age"])
        inc = find_series(data, ["clinical", "incidence"], avoid=("severe",))
        pfpr = find_series(data, ["pfpr", "age"])
        if pop is None or inc is None or pfpr is None:
            raise ValueError(f"{path}: could not find population / clinical incidence / "
                             f"PfPR series. Check the key names.")
        if verbose:
            notes.append(f"matched keys: {pop[0]!r}, {inc[0]!r}, {pfpr[0]!r}")
            notes.append(age_bin_note(data))
            verbose = False

        P = np.asarray(pop[1], dtype=float)
        I = np.asarray(inc[1], dtype=float)
        F = np.asarray(pfpr[1], dtype=float)
        if P.ndim != 2 or P.shape != I.shape or P.shape != F.shape:
            raise ValueError(f"{path}: series shapes differ or are not [period, age bin]: "
                             f"{P.shape}, {I.shape}, {F.shape}")
        if P.shape[1] != N_AGE_BINS:
            raise ValueError(f"{path}: {P.shape[1]} age bins, analyzer expects {N_AGE_BINS}.")

        T = min(PERIODS_USED, P.shape[0])
        P, I, F = P[:T], I[:T], F[:T]
        py = P * PERIOD_DAYS / DAYS_PER_YEAR
        cases = np.nan_to_num(I) * py
        pf_pop = np.where(P > 0, np.nan_to_num(F) * P, 0.0)

        for b in range(N_AGE_BINS):
            rows.append((sim_id, year, b, cases[:, b].sum(), py[:, b].sum(),
                         P[:, b].sum(), pf_pop[:, b].sum(), T))

    cols = ["sim_id", "year", "age_bin", "cases", "py", "pop_periods",
            "pfpr_x_pop", "n_periods"]
    return pd.DataFrame(rows, columns=cols), missing, notes, sim_id


def run_extract(index, out_path, years, workers):
    tasks = [(r.sim_id, r.sim_dir, years, n == 0)
             for n, r in enumerate(index.itertuples())]
    frames, n_done, t0 = [], 0, time.time()
    step = max(1, len(tasks) // 10)
    label = dict(zip(index["sim_id"], zip(index["scenario"], index["pickup_sim_index"])))

    def handle(res):
        df, missing, notes, sim_id = res
        for note in notes:
            print(f"  {note}")
        sc, st = label[sim_id]
        if df.empty:
            print(f"  WARNING: no report files for {sc} / state {st} "
                  f"(did the simulation start? check its stdout/stderr)")
        else:
            if missing:
                print(f"  WARNING: {sc} / state {st}: missing report year(s) {missing} "
                      f"(unfinished simulation?)")
            frames.append(df)

    print(f"\nExtracting {len(tasks)} simulation(s), years {years[0]}-{years[-1]}, "
          f"{workers} worker(s)...")
    if workers <= 1:
        for t in tasks:
            handle(extract_one(t))
            n_done += 1
            if n_done % step == 0:
                print(f"  {n_done}/{len(tasks)}")
    else:
        # First task alone so its key/age-bin notes print before the rest.
        handle(extract_one(tasks[0]))
        n_done = 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(extract_one, t) for t in tasks[1:]]
            for fut in as_completed(futures):
                handle(fut.result())
                n_done += 1
                if n_done % step == 0:
                    print(f"  {n_done}/{len(tasks)}  ({time.time() - t0:.0f}s)")

    if not frames:
        raise SystemExit("Nothing extracted.")

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(index[["sim_id"] + TAG_COLUMNS], on="sim_id", how="left")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows from {out['pickup_sim_index'].nunique()} states, "
          f"{out['scenario'].nunique()} scenarios -> {out_path}")
    return out


# =============================================================================
# 3. Metrics and paired comparison against IRS_CONTINUE
# =============================================================================

def drop_incomplete(df, start_year, last_year):
    """Drop pickup states in which ANY arm lacks the full analysis window."""
    years = list(range(start_year, last_year + 1))
    expected = len(years) * N_AGE_BINS
    ok = df[df["year"].isin(years) & (df["n_periods"] == PERIODS_USED)]
    counts = ok.groupby(["scenario", "pickup_sim_index"]).size()
    pairs = df[["scenario", "pickup_sim_index"]].drop_duplicates()
    counts = counts.reindex(pd.MultiIndex.from_frame(pairs), fill_value=0)
    bad = counts[counts != expected]

    print(f"\nAnalysis years {years[0]}-{years[-1]} ({len(years)} years x {PERIODS_USED} "
          f"periods x {N_AGE_BINS} age bins = {expected} rows per simulation).")

    bad_states = {int(i) for _, i in bad.index}
    arms = df.groupby("pickup_sim_index")["scenario"].nunique()
    bad_states |= {int(i) for i in arms[arms < arms.max()].index}

    if not bad_states:
        return df
    print(f"  WARNING: dropping pickup state(s) {sorted(bad_states)[:20]}"
          f"{' ...' if len(bad_states) > 20 else ''} from ALL arms "
          f"(incomplete or missing arm) so totals stay comparable.")
    for (sc, i), n in list(bad.items())[:20]:
        print(f"    {sc} / state {i}: {n} of {expected} rows")
    kept = df[~df["pickup_sim_index"].isin(bad_states)]
    if kept.empty:
        raise SystemExit("No complete pickup states left.")
    return kept


def summarise_group(d, by):
    """Totals, incidence and PfPR per group `by`."""
    u5 = d[d["age_bin"].isin(U5_BINS)]
    p5 = d[d["age_bin"] == PFPR_U5_BIN]

    g = d.groupby(by)
    out = pd.DataFrame({
        "cases_all": g["cases"].sum(),
        "py_all": g["py"].sum(),
        "pop_all": g["pop_periods"].sum(),
        "pfprpop_all": g["pfpr_x_pop"].sum(),
    })
    gu = u5.groupby(by)
    out["cases_u5"] = gu["cases"].sum()
    out["py_u5"] = gu["py"].sum()
    gp = p5.groupby(by)
    out["pop_p5"] = gp["pop_periods"].sum()
    out["pfprpop_p5"] = gp["pfpr_x_pop"].sum()

    out["incidence_all_per1000py"] = 1000 * out["cases_all"] / out["py_all"]
    out["incidence_u5_per1000py"] = 1000 * out["cases_u5"] / out["py_u5"]
    out["pfpr_u5"] = out["pfprpop_p5"] / out["pop_p5"]
    out["pfpr_all"] = out["pfprpop_all"] / out["pop_all"]
    return out[METRIC_COLS].reset_index()


def per_simulation_metrics(df, start_year, by_year=False):
    d = df[df["year"] >= start_year]
    return summarise_group(d, SIM_KEYS + (["year"] if by_year else []))


def paired_differences(metrics, extra_keys=()):
    """benefit = IRS_CONTINUE - scenario, matched on pickup_sim_index (+ extra_keys)."""
    join = ["pickup_sim_index", *extra_keys]
    base = metrics[metrics["scenario_kind"] == "irs"]
    if base.empty:
        raise SystemExit("No IRS_CONTINUE simulations found.")
    if base.duplicated(join).any():
        raise SystemExit("More than one IRS_CONTINUE simulation per pickup state.")

    others = metrics[metrics["scenario_kind"] != "irs"]
    m = others.merge(base[join + METRIC_COLS], on=join, how="inner",
                     suffixes=("", "_base"))
    lost = len(others) - len(m)
    if lost:
        print(f"  WARNING: {lost} scenario row(s) have no IRS_CONTINUE partner and are dropped.")

    out = m[SIM_KEYS + [k for k in extra_keys if k not in SIM_KEYS]].copy()
    for c in METRIC_COLS:
        out[c] = m[f"{c}_base"] - m[c]
        out[f"base_{c}"] = m[f"{c}_base"]
    for tag, c in (("all", "cases_all"), ("u5", "cases_u5")):
        out[f"pct_cases_averted_{tag}"] = (
            100 * out[c] / out[f"base_{c}"].replace(0, np.nan))
    return out.reset_index(drop=True)


def summarise(paired, extra=()):
    cols = METRIC_COLS + ["pct_cases_averted_all", "pct_cases_averted_u5"]
    keys = ["scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key", *extra]
    rows = []
    for k, g in paired.groupby(keys):
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec["n_pairs"] = len(g)
        for c in cols:
            rec.update(_stats(g[c], c))
        rec["share_pairs_cases_all_ge0"] = float((g["cases_all"] >= 0).mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["ig2_kill_key", "ig2_coverage", *extra])


def check_pairing(yearly_paired, switch_year, tol):
    """Before the switch every arm must equal IRS_CONTINUE (same seed, same inputs)."""
    pre = yearly_paired[yearly_paired["year"] < switch_year]
    if pre.empty:
        print(f"\nPairing check skipped: no report years before {switch_year} "
              f"(extract with --first-year {PROJ_START_YEAR}).")
        return
    rel = pre["cases_all"].abs() / pre["base_cases_all"].replace(0, np.nan)
    worst = rel.groupby(pre["scenario"]).max()
    yrs = f"{int(pre['year'].min())}-{int(pre['year'].max())}"
    print(f"\nPairing check, years {yrs} (should be ~0; max |cases difference| / baseline cases):")
    flagged = False
    for sc, v in worst.items():
        bad = pd.notna(v) and v > tol
        flagged |= bad
        print(f"  {sc:<16} {v:.2e}{'   <-- NOT PAIRED' if bad else ''}")
    if flagged:
        print("  WARNING: arms differ before the switch. Differences after the switch "
              "cannot be attributed to the interventions until this is explained "
              "(different campaign, weather, seed, or state).")


def _stats(v, prefix):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    names = ("mean", "median", "lo95", "hi95")
    if len(v) == 0:
        return {f"{prefix}_{n}": np.nan for n in names}
    return {f"{prefix}_mean": float(v.mean()),
            f"{prefix}_median": float(np.median(v)),
            f"{prefix}_lo95": float(np.percentile(v, 2.5)),
            f"{prefix}_hi95": float(np.percentile(v, 97.5))}


# =============================================================================
# 4. Effective use of IG2 nets (analytic: coverage x retention x age x season)
# =============================================================================

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec"]
_MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_VALUE_COLS = ["age_usage", "season_usage", "usage", "value", "use",
               "usage_prob", "probability", "prob"]


def _month_mid_doy(m):
    t = str(m).strip().lower()
    n = int(float(t)) if t.replace(".", "").isdigit() else _MONTHS.index(t[:3]) + 1
    return float(sum(_MDAYS[:n - 1])) + (_MDAYS[n - 1] + 1) / 2.0


def _read_clean_csv(path):
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.apply(lambda s: s.str.strip()).replace("", np.nan).dropna(how="all")


def _latest_block(d, year):
    """Rows of the latest dated block <= year; blank-year rows if none (as the scenario script)."""
    if "year" not in d.columns:
        return d
    yr = pd.to_numeric(d["year"], errors="coerce")
    dated = yr.dropna()
    earlier = dated[dated <= year]
    if len(earlier):
        return d[yr == earlier.max()]
    blank = d[yr.isna()]
    if len(blank):
        return blank
    if len(dated):
        return d[yr == dated.min()]
    return d


def read_usage_curve(input_dir, filename, kind, year):
    """
    (x, usage) of the IG2 (MASS channel) usage curve in force in `year`.
    Mirrors _read_usage_table / _select_usage_rows in run_scenarios_UWR.py.
    kind = "age" (x in years) or "season" (x = day of year, 1-365).
    """
    path = os.path.join(input_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; pass --input-dir or --no-ig2-curves.")
    df = _read_clean_csv(path)
    if "node_id" in df.columns:
        df = df[pd.to_numeric(df["node_id"], errors="coerce") == NODE_ID]
    if "phase" in df.columns:
        ph = df["phase"].astype(str).str.lower()
        df = df[ph.isin(PHASES_ACCEPTED) | df["phase"].isna()]
    if "channel" in df.columns:
        ch = df["channel"].astype(str).str.upper()
        rows = _latest_block(df[ch == IG2_USE_CHANNEL], year)
        if rows.empty:
            rows = _latest_block(df[df["channel"].isna()], year)
        df = rows
    else:
        df = _latest_block(df, year)
    if df.empty:
        raise ValueError(f"{filename}: no {IG2_USE_CHANNEL} rows for year {year}.")

    vcol = next((c for c in _VALUE_COLS if c in df.columns), None)
    if vcol is None:
        raise ValueError(f"{filename}: needs a usage column among {_VALUE_COLS}")
    y = pd.to_numeric(df[vcol], errors="coerce").to_numpy(float)

    if kind == "age":
        col = next((c for c in ("age", "age_years", "age_yr") if c in df.columns), None)
        if col is None:
            raise ValueError(f"{filename}: needs an age column.")
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    else:
        dc = next((c for c in ("day_of_year", "doy", "day") if c in df.columns), None)
        if dc:
            x = pd.to_numeric(df[dc], errors="coerce").to_numpy(float)
        elif "season_time" in df.columns:
            x = pd.to_numeric(df["season_time"], errors="coerce").to_numpy(float) + 1
            wrap = x > DAYS_PER_YEAR
            if wrap.any() and (x == 1).any():
                x, y = x[~wrap], y[~wrap]
        elif "month" in df.columns:
            x = df["month"].map(_month_mid_doy).to_numpy(float)
        else:
            raise ValueError(f"{filename}: needs a day column or a `month` column.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError(f"{filename}: blank or non-numeric entries in the {kind} curve.")
    o = np.argsort(x)
    return x[o], y[o]


def weibull_survival(t):
    return np.exp(-(np.asarray(t, float) / IG2_DISCARD_L) ** IG2_DISCARD_K)


def mean_retention(cycle_days):
    """Mean fraction of delivered nets still held over `cycle_days` (Weibull)."""
    t = np.linspace(0.0, float(cycle_days), 2001)
    s = weibull_survival(t)
    return float(np.mean((s[:-1] + s[1:]) / 2.0))


def mean_age_multiplier(ages, vals, results, start_year):
    """
    Population-weighted mean of the age-usage curve. Weights are the summed
    average population per report age bin (IRS_CONTINUE arm, analysis window);
    the curve is averaged uniformly inside each bin (flat beyond its ends).
    """
    d = results[(results["scenario_kind"] == "irs") & (results["year"] >= start_year)]
    w = d.groupby("age_bin")["pop_periods"].sum()
    total, acc = float(w.sum()), 0.0
    if total <= 0:
        raise ValueError("No population weights for the age multiplier.")
    for b, wt in w.items():
        lo, hi = REPORT_AGE_EDGES[int(b)], REPORT_AGE_EDGES[int(b) + 1]
        acc += wt * np.interp(np.linspace(lo, hi, 200), ages, vals).mean()
    return acc / total


def season_multiplier(days_of_year, sdays, svals):
    """Seasonal curve repeated yearly; 1 when no curve is used."""
    days_of_year = np.asarray(days_of_year, float)
    if sdays is None:
        return np.ones_like(days_of_year)
    xs = np.concatenate([sdays - DAYS_PER_YEAR, sdays, sdays + DAYS_PER_YEAR])
    return np.interp(days_of_year, xs, np.tile(svals, 3))


def build_rounds(input_dir, campaign_years, first_year, doy, use_curves,
                 results, start_year):
    """One entry per IG2 distribution round, each with its own year's curves."""
    rounds = []
    for y in sorted(campaign_years):
        d0 = (int(y) - first_year) * DAYS_PER_YEAR + int(doy)
        if use_curves:
            ages, avals = read_usage_curve(input_dir, ITN_AGE_FILE, "age", y)
            sdays, svals = read_usage_curve(input_dir, ITN_SEASON_FILE, "season", y)
            A = mean_age_multiplier(ages, avals, results, start_year)
        else:
            A, sdays, svals = 1.0, None, None
        rounds.append(dict(year=int(y), d0=d0, A=A, sdays=sdays, svals=svals))
    return rounds


def ig2_use_profile(c, rounds, n_days):
    """
    Daily held coverage and effective use for distributed coverage c.

    A person's current net comes from the LATEST round that reached them, so the
    share whose net is from round i on a given day is c (1-c)^j, where j is the
    number of later rounds already issued. Use = held x age multiplier x season.
    """
    days = np.arange(1, n_days + 1)
    doy = (days - 1) % DAYS_PER_YEAR + 1
    held = np.zeros(n_days)
    use = np.zeros(n_days)
    d0s = [r["d0"] for r in rounds]
    for i, r in enumerate(rounds):
        if r["d0"] > n_days:
            continue
        m = days >= r["d0"]
        later = np.array([d for d in d0s[i + 1:] if d <= n_days], dtype=float)
        j = (days[m][:, None] >= later[None, :]).sum(1) if len(later) else 0
        h = c * (1.0 - c) ** j * weibull_survival(days[m] - r["d0"])
        held[m] += h
        use[m] += h * r["A"] * season_multiplier(doy[m], r["sdays"], r["svals"])
    return days, held, use


def use_summary(c, rounds, n_days, first_year, start_year):
    days, held, use = ig2_use_profile(c, rounds, n_days)
    year = first_year + (days - 1) // DAYS_PER_YEAR
    win = year >= start_year
    rec = dict(held_window=held[win].mean(), use_window=use[win].mean(),
               use_max_daily=use[win].max())
    for i, r in enumerate(rounds, 1):
        d0 = r["d0"]
        nxt = rounds[i]["d0"] if i < len(rounds) else n_days + 1
        first30 = (days >= d0) & (days < d0 + 30)
        cyc = (days >= d0) & (days < nxt)
        rec[f"use_first30d_round{i}"] = use[first30].mean() if first30.any() else np.nan
        rec[f"use_mean_cycle{i}"] = use[cyc].mean() if cyc.any() else np.nan
        rec[f"held_mean_cycle{i}"] = held[cyc].mean() if cyc.any() else np.nan
    for y in sorted(set(year[win])):
        rec[f"use_{int(y)}"] = use[year == y].mean()
    return rec


def make_use_functions(rounds, n_days, first_year, start_year, metric):
    """use_fn(c) -> the chosen effective-use summary; full_fn(c) -> all summaries."""
    cache = {}

    def full_fn(c):
        k = round(float(c), 6)
        if k not in cache:
            cache[k] = use_summary(k, rounds, n_days, first_year, start_year)
        return cache[k]

    def use_fn(c):
        return float(full_fn(c)[metric])

    return use_fn, full_fn


def print_use_assumptions(rounds, campaign_years, doy):
    print("\nIG2 effective use = c x retention x age multiplier x season multiplier")
    for i, r in enumerate(rounds, 1):
        if r["sdays"] is None:
            print(f"  round {i} ({r['year']}): curves off, multipliers = 1")
            continue
        s_mean = float(np.mean(season_multiplier(np.arange(1, DAYS_PER_YEAR + 1),
                                                 r["sdays"], r["svals"])))
        s_issue = float(season_multiplier([doy], r["sdays"], r["svals"])[0])
        print(f"  round {i} ({r['year']}): mean age multiplier A = {r['A']:.3f}; "
              f"season multiplier: annual mean {s_mean:.3f}, on delivery day {s_issue:.3f}")
    cycle = (int(campaign_years[1] - campaign_years[0]) * DAYS_PER_YEAR
             if len(campaign_years) > 1 else 3 * DAYS_PER_YEAR)
    print(f"  net retention: Weibull k={IG2_DISCARD_K:g}, lambda={IG2_DISCARD_L:g} d; "
          f"mean fraction held over a {cycle}-day cycle = {mean_retention(cycle):.2f}")


def use_grid(use_fn, step=0.01):
    """c -> effective use on a fine grid (made non-decreasing for inversion)."""
    c = np.round(np.arange(step, 1.0 + 1e-9, step), 4)
    u = np.maximum.accumulate([use_fn(float(x)) for x in c])
    return c, np.asarray(u, float)


def required_coverage(target, c_grid, u_grid):
    """Smallest distributed c giving effective use = target (NaN if impossible)."""
    if not np.isfinite(target) or target > u_grid[-1] + 1e-9:
        return np.nan
    return float(np.interp(target, u_grid, c_grid))


def effective_use_table(covs, full_fn, rounds):
    tab = pd.DataFrame([dict(distributed_coverage=float(c), **full_fn(float(c)))
                        for c in covs])
    tab["age_multiplier_round1"] = rounds[0]["A"]
    tab["use_over_coverage_window"] = tab["use_window"] / tab["distributed_coverage"]
    return tab


# =============================================================================
# 5. Minimum effective use (maintain or reduce burden)
# =============================================================================

def find_min(x, val, unit="effective use"):
    """Smallest x from which the statistic stays >= 0 (linear interpolation)."""
    ok = [bool(v >= 0) for v in val]          # NaN -> False
    if not ok[-1]:
        return np.nan, (f"not reached: still < 0 at the highest simulated {unit} "
                        f"({max(x):.3f})")
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    if i == 0:
        return x[0], (f"at or below the lowest simulated {unit} ({x[0]:.3f}); "
                      f"add lower-coverage runs")
    x0, x1, y0, y1 = x[i - 1], x[i], val[i - 1], val[i]
    thr = x1 if y1 == y0 else x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    return thr, f"between {x0:.3f} and {x1:.3f} (linear interpolation)"


CRITERIA = [("weak: median benefit >= -margin", np.nanmedian),
            ("strict: 2.5th percentile >= -margin", lambda x: np.nanpercentile(x, 2.5))]


def threshold_bootstrap(arr, x, stat, n_boot, rng):
    """Resample pickup states (rows) and recompute the threshold."""
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        samp = arr[rng.integers(0, len(arr), len(arr))]
        out[b] = find_min(x, [stat(samp[:, j]) for j in range(len(x))])[0]
    return out


def threshold_table(paired, margin, n_boot, seed, use_fn, full_fn, c_grid, u_grid):
    """
    Minimum IG2 effective use that maintains or reduces burden. Found on the
    effective-use axis (each simulated c is mapped to its use), then also given
    as the equivalent distributed c.
    """
    rng = np.random.default_rng(seed)
    ig2 = paired[paired["scenario_kind"] == "ig2"]
    to_c = lambda u: required_coverage(u, c_grid, u_grid) if pd.notna(u) else np.nan
    rows = []
    for metric in THRESHOLD_METRICS:
        for key, g in ig2.groupby("ig2_kill_key"):
            g = g.assign(adj=g[metric] + margin * g[f"base_{metric}"])
            wide = g.pivot_table(index="pickup_sim_index", columns="ig2_coverage",
                                 values="adj")
            covs = [float(c) for c in wide.columns]
            uses = np.array([use_fn(c) for c in covs])
            order = np.argsort(uses)
            uses = uses[order]
            covs_s = np.array(covs)[order]
            arr = wide.to_numpy()[:, order]

            for label, stat in CRITERIA:
                thr_u, note = find_min(list(uses), [stat(arr[:, j]) for j in range(len(uses))])
                rec = dict(metric=metric, kill_key=key, criterion=label,
                           min_effective_use=thr_u,
                           min_distributed_coverage=to_c(thr_u),
                           note=note, n_pairs=len(arr),
                           max_simulated_coverage=float(covs_s[-1]),
                           max_simulated_effective_use=float(uses[-1]))
                if pd.notna(thr_u):
                    c_thr = rec["min_distributed_coverage"]
                    if pd.notna(c_thr):
                        full = full_fn(c_thr)
                        rec["use_window_at_threshold"] = full["use_window"]
                        rec["use_first30d_round1_at_threshold"] = full["use_first30d_round1"]
                        rec["use_mean_cycle1_at_threshold"] = full.get("use_mean_cycle1", np.nan)
                if n_boot > 0 and len(arr) > 1:
                    boot = threshold_bootstrap(arr, list(uses), stat, n_boot, rng)
                    fin = boot[~np.isnan(boot)]
                    rec["boot_share_not_reached"] = float(np.isnan(boot).mean())
                    if len(fin):
                        lo, med, hi = np.percentile(fin, [2.5, 50, 97.5])
                        rec.update(boot_use_lo95=lo, boot_use_median=med, boot_use_hi95=hi,
                                   boot_c_lo95=to_c(lo), boot_c_median=to_c(med),
                                   boot_c_hi95=to_c(hi))
                rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# 6. Aim 2 tables
# =============================================================================

def add_use_column(df, use_fn):
    df = df.copy()
    df["effective_use"] = np.where(
        df["scenario_kind"] == "ig2",
        [use_fn(float(c)) if k == "ig2" else np.nan
         for c, k in zip(df["ig2_coverage"], df["scenario_kind"])],
        np.nan)
    return df


def scenario_outcomes(metrics, use_fn, by_year=False, use_tab=None):
    """Absolute incidence / PfPR / cases per arm: median and 95% range over states."""
    keys = ["scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key"]
    keys += ["year"] if by_year else []
    rows = []
    for k, g in metrics.groupby(keys):
        rec = dict(zip(keys, k))
        rec["n_pickup_states"] = g["pickup_sim_index"].nunique()
        rec["effective_use"] = (use_fn(float(rec["ig2_coverage"]))
                                if rec["scenario_kind"] == "ig2" else np.nan)
        for col in OUTCOME_LABELS:
            rec.update(_stats(g[col], col))
        rows.append(rec)
    out = pd.DataFrame(rows)

    if by_year and use_tab is not None and len(use_tab):
        ycols = [c for c in use_tab.columns
                 if c.startswith("use_") and len(c) == 8 and c[4:].isdigit()]
        if ycols:
            long = use_tab.melt(id_vars="distributed_coverage", value_vars=ycols,
                                var_name="yr", value_name="effective_use_year")
            long["year"] = long["yr"].str[4:].astype(int)
            long = long.rename(columns={"distributed_coverage": "ig2_coverage"})
            out = out.merge(long[["ig2_coverage", "year", "effective_use_year"]],
                            on=["ig2_coverage", "year"], how="left")
            out.loc[out["scenario_kind"] != "ig2", "effective_use_year"] = np.nan
    sort_keys = ["ig2_kill_key", "ig2_coverage", "scenario"] + (["year"] if by_year else [])
    return out.sort_values(sort_keys).reset_index(drop=True)


def target_table(metrics, use_fn, targets, c_grid, u_grid):
    """
    Outcomes at each effective-use target. Per pickup state, the IG2 outcome is
    interpolated between the simulated arms on the effective-use axis; targets
    outside the simulated range are NOT extrapolated.
    """
    base = metrics[metrics["scenario_kind"] == "irs"].set_index("pickup_sim_index")
    ig2 = metrics[metrics["scenario_kind"] == "ig2"]
    rows = []

    for key, g in ig2.groupby("ig2_kill_key"):
        covs = sorted(float(c) for c in g["ig2_coverage"].unique())
        uses = np.array([use_fn(c) for c in covs])
        order = np.argsort(uses)
        uses_s = uses[order]
        wides = {col: g.pivot_table(index="pickup_sim_index", columns="ig2_coverage",
                                    values=col)[covs].iloc[:, order]
                 for col in OUTCOME_LABELS}
        states = wides["cases_all"].index.intersection(base.index)

        for u in targets:
            c_req = required_coverage(u, c_grid, u_grid)
            inside = uses_s[0] - 1e-9 <= u <= uses_s[-1] + 1e-9
            near = bool(np.isclose(uses_s, u, atol=0.005).any())

            if inside:
                status = "simulated" if near else "interpolated between simulated arms"
            elif np.isnan(c_req):
                status = (f"UNREACHABLE: maximum effective use at c = 1 is "
                          f"{u_grid[-1]:.3f}")
            elif u > uses_s[-1]:
                status = (f"NOT SIMULATED: highest simulated effective use is "
                          f"{uses_s[-1]:.3f}; run c >= {c_req:.2f}")
            else:
                status = (f"NOT SIMULATED: lowest simulated effective use is "
                          f"{uses_s[0]:.3f}")

            rec = dict(kill_key=key, target_effective_use=u,
                       required_distributed_coverage=c_req, status=status,
                       n_pickup_states=len(states))
            for col in OUTCOME_LABELS:
                b = base.loc[states, col].to_numpy(float)
                if inside:
                    at = np.array([np.interp(u, uses_s, wides[col].loc[s].to_numpy(float))
                                   for s in states])
                else:
                    at = np.full(len(states), np.nan)
                rec[f"{col}_irs_median"] = float(np.median(b))
                rec.update(_stats(at, f"{col}_ig2"))
                rec.update(_stats(b - at, f"{col}_averted"))
                if col in CASE_COLS:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pct = np.where(b > 0, 100 * (b - at) / b, np.nan)
                    rec.update(_stats(pct, f"{col}_pct_averted"))
                    if col == "cases_all":
                        ok = ~np.isnan(at)
                        rec["share_states_cases_all_averted_ge0"] = (
                            float(((b - at)[ok] >= 0).mean()) if ok.any() else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# 7. Figures
# =============================================================================

def _mark(ax, targets, thr_use):
    for t in targets:
        ax.axvline(t, color="tab:orange", lw=0.6, ls=":")
    for v in thr_use:
        ax.axvline(v, color="tab:red", lw=1, ls="--")


def plot_outcomes(outcomes, thr, targets, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ig2 = outcomes[outcomes["scenario_kind"] == "ig2"]
    if ig2.empty:
        return
    irs = outcomes[outcomes["scenario_kind"] == "irs"]
    nirs = outcomes[outcomes["scenario_kind"] == "none"]
    thr_use = thr["min_effective_use"].dropna().unique() if len(thr) else []
    panels = [("incidence_all_per1000py", "Clinical incidence, all ages\n(per 1,000 PY)"),
              ("pfpr_u5", "PfPR, 6-59 months"),
              ("cases_all", "Clinical cases, all ages\n(analysis-window total)")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (col, label) in zip(axes, panels):
        for key, g in ig2.groupby("ig2_kill_key"):
            g = g.sort_values("effective_use")
            ax.fill_between(g["effective_use"], g[f"{col}_lo95"], g[f"{col}_hi95"],
                            alpha=0.15, lw=0)
            ax.plot(g["effective_use"], g[f"{col}_median"], marker="o", ms=3,
                    label=f"IG2 ({key or 'n/a'})")
        for df, colour, ls, name in ((irs, "black", "-", "IRS continued"),
                                     (nirs, "0.5", "--", "No IRS, no IG2")):
            if len(df):
                ax.axhline(df[f"{col}_median"].iloc[0], color=colour, ls=ls, lw=1, label=name)
        _mark(ax, targets, thr_use)
        ax.set_xlabel("IG2 effective use (share of population using an IG2 net)")
        ax.set_ylabel(label)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Outcomes vs IG2 effective use (median, 95% range over pickup states; "
                 "orange: targets, red: minimum effective use)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_file(outdir, f"aim2_outcomes_vs_effective_use.{ext}"), dpi=300)
    plt.close(fig)


def plot_benefit(paired, thr, targets, use_fn, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ig2 = paired[paired["scenario_kind"] == "ig2"].copy()
    if ig2.empty:
        return
    ig2["use"] = [use_fn(float(c)) for c in ig2["ig2_coverage"]]
    panels = [("cases_all", "Cases averted vs IRS (all ages)"),
              ("cases_u5", "Cases averted vs IRS (under 5)"),
              ("pfpr_u5", "PfPR reduction vs IRS (6-59 months)")]
    keys = sorted(ig2["ig2_kill_key"].unique())
    fig, axes = plt.subplots(len(keys), 3, figsize=(13, 3.8 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        g = ig2[ig2["ig2_kill_key"] == key]
        for j, (metric, label) in enumerate(panels):
            ax = axes[i][j]
            s = g.groupby("use")[metric].agg(
                med="median", lo=lambda x: np.percentile(x, 2.5),
                hi=lambda x: np.percentile(x, 97.5))
            ax.fill_between(s.index, s["lo"], s["hi"], alpha=0.2, lw=0)
            ax.plot(s.index, s["med"], marker="o", ms=3)
            ax.axhline(0, color="black", lw=0.8)
            t = thr[(thr["metric"] == metric) & (thr["kill_key"] == key)]
            for r in t.itertuples():
                if pd.notna(r.min_effective_use):
                    ax.axvline(r.min_effective_use, color="tab:red", lw=1,
                               ls="--" if r.criterion.startswith("weak") else ":")
            for tg in targets:
                ax.axvline(tg, color="tab:orange", lw=0.6, ls=":")
            ax.set_xlabel("IG2 effective use")
            ax.set_ylabel(label)
            ax.set_title(f"kill decay: {key or 'n/a'}", fontsize=9)
            ax.set_xlim(left=0)
            ax.grid(alpha=0.25)
    fig.suptitle("Benefit vs continued IRS (positive = IG2 better; band: 95% range; red: "
                 "minimum effective use, dashed weak / dotted strict; orange: targets)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_file(outdir, f"aim2_benefit_vs_effective_use.{ext}"), dpi=300)
    plt.close(fig)


# =============================================================================
# 8. Aim 2 driver
# =============================================================================

def run_aim2(results, outdir, args):
    os.makedirs(outdir, exist_ok=True)
    margin = args.margin_pct / 100.0
    start, last = args.analysis_start_year, args.last_year
    targets = list(args.targets_use)

    results = results.copy()
    results["ig2_kill_key"] = results["ig2_kill_key"].fillna("")
    results = drop_incomplete(results, start, last)

    # ---- metrics and pairing ------------------------------------------------
    yearly = per_simulation_metrics(results, int(results["year"].min()), by_year=True)
    yearly_paired = paired_differences(yearly, extra_keys=["year"])
    check_pairing(yearly_paired, args.switch_year, args.pairing_tol)

    metrics = per_simulation_metrics(results, start)
    paired = paired_differences(metrics)
    n_states = paired["pickup_sim_index"].nunique()
    print(f"\nAnalysis window: {start}-{last}; periods 1-{PERIODS_USED} of each year; "
          f"{n_states} pickup states.")

    # ---- effective use ------------------------------------------------------
    use_curves = not args.no_ig2_curves
    camp_years = sorted(args.campaign_years)
    rounds = build_rounds(args.input_dir, camp_years, args.first_year, args.ig2_doy,
                          use_curves, results, start)
    n_days = (last - args.first_year + 1) * DAYS_PER_YEAR
    use_fn, full_fn = make_use_functions(rounds, n_days, args.first_year, start,
                                         args.use_metric)
    print_use_assumptions(rounds, camp_years, args.ig2_doy)
    print(f"  effective-use coverage = {args.use_metric} "
          f"(analysis window {start}-{last})")

    c_grid, u_grid = use_grid(use_fn)
    pd.DataFrame({"distributed_coverage": c_grid, "effective_use": u_grid}).to_csv(
        out_file(outdir, "aim2_use_to_coverage_lookup.csv"), index=False)
    print(f"  maximum effective use reachable at c = 1: {u_grid[-1]:.3f}")

    covs = sorted(results.loc[results["scenario_kind"] == "ig2", "ig2_coverage"].unique())
    use_tab = effective_use_table(covs, full_fn, rounds)
    use_tab.to_csv(out_file(outdir, "aim2_effective_use_by_coverage.csv"), index=False)

    print(f"\n  {'c':>6} {'held':>7} {'use(window)':>12} {'use(30d r1)':>12}")
    for r in use_tab.itertuples():
        print(f"  {r.distributed_coverage:>6.2f} {r.held_window:>7.3f} "
              f"{r.use_window:>12.3f} {r.use_first30d_round1:>12.3f}")

    # ---- Aim 2.1: outcomes of every arm ------------------------------------
    outcomes = scenario_outcomes(metrics, use_fn)
    outcomes.to_csv(out_file(outdir, "aim2_scenario_outcomes.csv"), index=False)
    scenario_outcomes(yearly, use_fn, by_year=True, use_tab=use_tab).to_csv(
        out_file(outdir, "aim2_outcomes_by_year.csv"), index=False)

    print("\nAim 2.1  arms simulated (median across pickup states):")
    print(f"  {'scenario':<16} {'c':>5} {'eff.use':>8} {'inc all':>9} {'inc u5':>9} "
          f"{'PfPR u5':>8}")
    for r in outcomes.itertuples():
        eu = "" if pd.isna(r.effective_use) else f"{r.effective_use:.3f}"
        print(f"  {r.scenario:<16} {r.ig2_coverage:>5.2f} {eu:>8} "
              f"{r.incidence_all_per1000py_median:>9.1f} "
              f"{r.incidence_u5_per1000py_median:>9.1f} {r.pfpr_u5_median:>8.3f}")

    # ---- Aim 2.2: benefit vs continued IRS ----------------------------------
    summary = add_use_column(summarise(paired), use_fn)
    summary.to_csv(out_file(outdir, "aim2_benefit_vs_irs.csv"), index=False)
    add_use_column(summarise(yearly_paired, extra=["year"]), use_fn).to_csv(
        out_file(outdir, "aim2_benefit_vs_irs_by_year.csv"), index=False)

    print("\nAim 2.2  cases averted vs continued IRS (all ages), mean [2.5%, 97.5%]:")
    for r in summary.itertuples():
        eu = "      " if pd.isna(r.effective_use) else f"{r.effective_use:6.3f}"
        print(f"  {r.scenario:<16} use {eu} {r.cases_all_mean:>10.0f} "
              f"[{r.cases_all_lo95:>9.0f}, {r.cases_all_hi95:>9.0f}]  "
              f"{r.pct_cases_averted_all_mean:>6.1f}%  n={r.n_pairs}")

    # ---- targets ------------------------------------------------------------
    tt = target_table(metrics, use_fn, targets, c_grid, u_grid)
    tt.to_csv(out_file(outdir, "aim2_effective_use_targets.csv"), index=False)
    print("\nOutcomes at each effective-use target (averted = IRS minus IG2; "
          "median [2.5%, 97.5%]):")
    for r in tt.itertuples():
        c_txt = ("n/a" if pd.isna(r.required_distributed_coverage)
                 else f"{r.required_distributed_coverage:.2f}")
        print(f"\n  target {r.target_effective_use:.0%} effective use "
              f"(kill={r.kill_key or 'n/a'}): distribute c = {c_txt}")
        print(f"    {r.status}")
        if pd.isna(r.cases_all_averted_median):
            continue
        print(f"    incidence all ages  {r.incidence_all_per1000py_ig2_median:8.1f} "
              f"per 1,000 PY (IRS {r.incidence_all_per1000py_irs_median:.1f})")
        print(f"    PfPR 6-59 months    {r.pfpr_u5_ig2_median:8.3f} "
              f"(IRS {r.pfpr_u5_irs_median:.3f})")
        print(f"    cases averted, all  {r.cases_all_averted_median:10.0f} "
              f"[{r.cases_all_averted_lo95:.0f}, {r.cases_all_averted_hi95:.0f}]  "
              f"({r.cases_all_pct_averted_median:.1f}%)")
        print(f"    cases averted, u5   {r.cases_u5_averted_median:10.0f} "
              f"[{r.cases_u5_averted_lo95:.0f}, {r.cases_u5_averted_hi95:.0f}]")
        print(f"    states with cases averted >= 0: "
              f"{100 * r.share_states_cases_all_averted_ge0:.0f}%")

    need = sorted({np.ceil(c * 20) / 20 for c in tt["required_distributed_coverage"]
                   if pd.notna(c)})
    have = {round(float(c), 4) for c in covs}
    missing = [c for c in need if round(c, 4) not in have]
    if missing:
        print(f"\n  To simulate every target, add to run_scenarios_UWR.py "
              f"EXTRA_IG2_COVERAGES: {missing}")
    if tt["required_distributed_coverage"].isna().any():
        print("  Targets marked UNREACHABLE cannot be met at any distributed coverage "
              "with the current retention / MASS curves. Report them as such, or "
              "redefine effective use (e.g. use among net holders).")

    # ---- minimum effective use ---------------------------------------------
    thr = threshold_table(paired, margin, args.bootstrap, args.seed, use_fn, full_fn,
                          c_grid, u_grid)
    thr.to_csv(out_file(outdir, "aim2_minimum_effective_use.csv"), index=False)

    print(f"\nMinimum IG2 effective use that maintains or reduces burden "
          f"(margin {args.margin_pct:g}% of IRS value, bootstrap {args.bootstrap}):")
    for r in thr.itertuples():
        head = f"  {r.metric:<10} kill={r.kill_key:<4} {r.criterion:<36}"
        if pd.isna(r.min_effective_use):
            print(f"{head} not reached (highest simulated: c={r.max_simulated_coverage:g}, "
                  f"use={r.max_simulated_effective_use:.3f})")
        else:
            c_txt = ("n/a" if pd.isna(r.min_distributed_coverage)
                     else f"{r.min_distributed_coverage:.3f}")
            ci = ""
            if pd.notna(getattr(r, "boot_use_lo95", np.nan)):
                ci = f"  boot [{r.boot_use_lo95:.3f}, {r.boot_use_hi95:.3f}]"
            if getattr(r, "boot_share_not_reached", 0) and r.boot_share_not_reached > 0:
                ci += f"  ({100 * r.boot_share_not_reached:.0f}% of resamples: not reached)"
            print(f"{head} use {r.min_effective_use:.3f}  (distribute c = {c_txt}){ci}")
        print(f"  {'':<10} {'':<9} {r.note}")
    print("  The effective use shown is the mean over the analysis window, because the "
          "thresholds use cases summed over the same window. Use --use-metric to change it.")

    if not args.no_plots:
        plot_outcomes(outcomes, thr, targets, outdir)
        plot_benefit(paired, thr, targets, use_fn, outdir)
        print("\nFigures written (png + pdf): aim2_outcomes_vs_effective_use, "
              "aim2_benefit_vs_effective_use")
    print(f"Aim 2 tables written to {outdir}/")
    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--experiment-id", default=None,
                   help="idmtools experiment ID of the scenario simulations. "
                        "Overrides the EXPERIMENTS dict.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Names from EXPERIMENTS to analyse (default: all).")
    p.add_argument("--outdir", default=None,
                   help="Parent folder for <name>" + OUT_SUFFIX + " (default: current folder).")
    p.add_argument("--exp-dir", default=None,
                   help="Scan this experiment folder instead of calling idmtools.")
    p.add_argument("--reuse-index", action="store_true",
                   help="Reuse the saved sim_index; do not contact idmtools.")
    p.add_argument("--reuse-results", action="store_true",
                   help="Reuse the saved scenario_results; skip extraction.")
    p.add_argument("--ignore-status", action="store_true",
                   help="Keep simulations idmtools does not report as succeeded.")
    p.add_argument("--first-year", type=int, default=PROJ_START_YEAR)
    p.add_argument("--last-year", type=int, default=PROJ_END_YEAR)
    p.add_argument("--analysis-start-year", type=int, default=2027,
                   help="First year compared. Default 2027 = the IRS switch year.")
    p.add_argument("--switch-year", type=int, default=2027,
                   help="IRS_SWITCH_YEAR; years before it form the pairing check.")
    p.add_argument("--campaign-years", type=int, nargs="+", default=[2027, 2030],
                   help="IG2_CAMPAIGN_YEARS (drives the effective-use calculation).")
    p.add_argument("--ig2-doy", type=int, default=IG2_CAMPAIGN_DOY,
                   help="IG2_CAMPAIGN_DOY: day of year of each IG2 distribution.")
    p.add_argument("--input-dir", default=INPUT_DIR,
                   help="Folder with interventions_ITN_age.csv and interventions_ITN_season.csv.")
    p.add_argument("--no-ig2-curves", action="store_true",
                   help="The scenario ran with IG2_USE_CURVES = False: age and season "
                        "multipliers are 1 (retention still applies).")
    p.add_argument("--targets-use", type=float, nargs="+", default=DEFAULT_TARGETS,
                   help="Effective-use targets for Aim 2.1 (default 0.5 0.6 0.8 0.9).")
    p.add_argument("--use-metric", default="use_window", choices=USE_METRICS,
                   help="Summary that defines 'effective-use coverage' (default use_window).")
    p.add_argument("--margin-pct", type=float, default=0.0,
                   help="Allowed excess vs IRS, %% of the IRS value (default 0).")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples of pickup states (0 = off).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pairing-tol", type=float, default=1e-6)
    p.add_argument("--workers", type=int, default=4,
                   help="Extraction processes (keep modest on the head node).")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def resolve_experiments(args):
    if args.experiment_id:
        eid = args.experiment_id.strip()
        name = next((n for n, i in EXPERIMENTS.items() if i == eid), f"exp_{eid[:8]}")
        return {name: eid}
    exps = dict(EXPERIMENTS)
    if args.only:
        unknown = [n for n in args.only if n not in exps]
        if unknown:
            sys.exit(f"--only: {unknown} not in EXPERIMENTS {list(exps)}.")
        exps = {n: exps[n] for n in args.only}
    bad = [n for n, i in exps.items() if not i.strip() or i.startswith("PUT-")]
    if bad:
        sys.exit(f"Set the scenario experiment ID for {bad} in EXPERIMENTS "
                 f"(or pass --experiment-id).")
    return exps


def run_one(name, experiment_id, args):
    outdir = Path(args.outdir or ".").expanduser() / f"{name}{OUT_SUFFIX}"
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = Path(out_file(outdir, "sim_index.csv"))
    results_path = Path(out_file(outdir, "scenario_results.csv"))
    print(f"\n{'=' * 78}\n{name}  ({experiment_id})  step={STEP}\noutput: {outdir}\n{'=' * 78}")

    # 1. Index
    if args.reuse_index and index_path.is_file():
        index = pd.read_csv(index_path)
        index["ig2_kill_key"] = index["ig2_kill_key"].fillna("")
        print(f"Reusing {index_path} ({len(index)} simulations).")
    else:
        raw = (simulations_from_directory(args.exp_dir) if args.exp_dir
               else simulations_from_platform(experiment_id))
        index = check_index(normalise_index(raw), args.ignore_status)
        index.to_csv(index_path, index=False)
        print(f"Wrote {index_path}")

    # 2. Extract
    years = list(range(args.first_year, args.last_year + 1))
    if args.reuse_results and results_path.is_file():
        results = pd.read_csv(results_path)
        print(f"Reusing {results_path} ({len(results)} rows).")
    else:
        results = run_extract(index, str(results_path), years, max(1, args.workers))

    # 3. Aim 2 analysis (pairing, effective use, targets, minimum effective use)
    try:
        results = run_aim2(results, str(outdir), args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"\nERROR: {exc}\nThe effective-use calculation needs the MASS age/season "
                 f"curves (--input-dir), or pass --no-ig2-curves if the scenario ran "
                 f"with IG2_USE_CURVES = False.")

    kept = set(results["pickup_sim_index"].unique())
    with open(out_file(outdir, "run_info.json"), "w") as f:
        json.dump(dict(name=name, step=STEP, experiment_id=experiment_id,
                       simulations=int(len(index)),
                       pickup_states_analyzed=int(len(kept)),
                       args=vars(args)), f, indent=2, default=str)
    print(f"\nDone. Outputs in {outdir}/")

def main():
    args = parse_args()
    for name, experiment_id in resolve_experiments(args).items():
        run_one(name, experiment_id, args)


if __name__ == "__main__":
    main()


def plot_dose_response(paired, thr, outdir):
    import matplotlib.pyplot as plt
    ig2 = paired[paired["scenario_kind"] == "ig2"]
    if ig2.empty:
        return
    panels = [("cases_all", "Cases averted vs IRS (all ages)"),
              ("cases_u5", "Cases averted vs IRS (under 5)"),
              ("pfpr_u5", "PfPR reduction vs IRS (6-59 months)")]
    keys = sorted(ig2["ig2_kill_key"].unique())
    fig, axes = plt.subplots(len(keys), 3, figsize=(13, 3.8 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        g = ig2[ig2["ig2_kill_key"] == key]
        for j, (metric, label) in enumerate(panels):
            ax = axes[i][j]
            s = g.groupby("ig2_coverage")[metric].agg(
                med="median", lo=lambda x: np.percentile(x, 2.5),
                hi=lambda x: np.percentile(x, 97.5))
            ax.fill_between(s.index, s["lo"], s["hi"], alpha=0.2, lw=0)
            ax.plot(s.index, s["med"], marker="o")
            ax.axhline(0, color="black", lw=0.8)
            t = thr[(thr["metric"] == metric) & (thr["kill_key"] == key)]
            for r in t.itertuples():
                if pd.notna(r.min_coverage):
                    ax.axvline(r.min_coverage, color="tab:red", lw=1,
                               ls="--" if r.criterion.startswith("weak") else ":")
            ax.set_xlabel("IG2 coverage distributed (c)")
            ax.set_ylabel(label)
            ax.set_title(f"kill decay: {key or 'n/a'}", fontsize=9)
            ax.grid(alpha=0.25)
    fig.suptitle("Dose-response vs continued IRS (band: 95% range; red: minimum coverage, "
                 "dashed = weak, dotted = strict)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_file(outdir, f"benefit_vs_coverage.{ext}"), dpi=300)
    plt.close(fig)


def make_plots(outdir, yearly, yearly_paired, paired, thr, switch_year, campaign_years):
    import matplotlib
    matplotlib.use("Agg")
    plot_trajectory(yearly, "incidence_all_per1000py",
                    "Clinical incidence per 1,000 person-years (all ages)",
                    outdir, "incidence_by_year", switch_year, campaign_years)
    plot_trajectory(yearly, "pfpr_u5", "PfPR, 6-59 months",
                    outdir, "pfpr_u5_by_year", switch_year, campaign_years)
    plot_benefit_by_year(yearly_paired, outdir, switch_year, campaign_years)
    plot_dose_response(paired, thr, outdir)
    print("Figures written (png + pdf): incidence_by_year, pfpr_u5_by_year, "
          "benefit_by_year, benefit_vs_coverage")


# =============================================================================
# Analysis driver
# =============================================================================

def run_analyze(df, outdir, start_year, last_year, switch_year, margin_pct,
                n_boot, seed, pairing_tol, campaign_years, plots):
    """Returns (complete results, threshold table)."""
    os.makedirs(outdir, exist_ok=True)
    df = df.copy()
    df["ig2_kill_key"] = df["ig2_kill_key"].fillna("")
    margin = margin_pct / 100.0

    df = drop_incomplete(df, start_year, last_year)

    # Year-by-year view, including the common 2026 path.
    yearly = per_simulation_metrics(df, int(df["year"].min()), by_year=True)
    yearly_paired = paired_differences(yearly, extra_keys=["year"])
    check_pairing(yearly_paired, switch_year, pairing_tol)
    by_year = summarise(yearly_paired, extra=["year"])
    by_year.to_csv(out_file(outdir, "benefit_by_year.csv"), index=False)
    yearly.to_csv(out_file(outdir, "metrics_per_simulation_by_year.csv"), index=False)

    # Whole analysis window.
    metrics = per_simulation_metrics(df, start_year)
    metrics.to_csv(out_file(outdir, "metrics_per_simulation.csv"), index=False)

    abs_summary = (metrics.groupby(["scenario", "ig2_coverage", "ig2_kill_key"])
                   [["incidence_all_per1000py", "incidence_u5_per1000py",
                     "pfpr_u5", "pfpr_all"]].agg(["mean", "median"]))
    abs_summary.to_csv(out_file(outdir, "absolute_outcomes_by_scenario.csv"))

    paired = paired_differences(metrics)
    paired.to_csv(out_file(outdir, "paired_differences.csv"), index=False)

    summary = summarise(paired)
    summary.to_csv(out_file(outdir, "scenario_summary.csv"), index=False)

    recovered = irs_effect_recovered(paired)
    if recovered is not None:
        recovered.to_csv(out_file(outdir, "irs_effect_recovered.csv"), index=False)

    thr = threshold_table(paired, margin, n_boot, seed)
    thr.to_csv(out_file(outdir, "minimum_ig2_coverage.csv"), index=False)

    print(f"\nAnalysis window: {start_year}-{last_year}; periods 1-{PERIODS_USED} of each "
          f"year; {paired['pickup_sim_index'].nunique()} pickup states.")
    print("\nCases averted vs continued IRS (all ages), mean [2.5%, 97.5%] over pairs:")
    for r in summary.itertuples():
        print(f"  {r.scenario:<16} {r.cases_all_mean:>10.0f} "
              f"[{r.cases_all_lo95:>9.0f}, {r.cases_all_hi95:>9.0f}]   "
              f"{r.pct_cases_averted_all_mean:>6.1f}%   n={r.n_pairs}")

    if recovered is not None and len(recovered):
        print("\nShare of the IRS effect recovered by IG2 "
              "((NO_IRS - IG2) / (NO_IRS - IRS); 1.0 = as good as IRS):")
        for r in recovered.itertuples():
            print(f"  {r.scenario:<16} all ages {r.irs_effect_recovered_all:>6.2f}   "
                  f"under 5 {r.irs_effect_recovered_u5:>6.2f}")

    print("\nMedian cases averted vs IRS by year (all ages; negative = more cases than IRS):")
    wide = by_year.pivot_table(index="scenario", columns="year", values="cases_all_median")
    print(wide.round(0).to_string())

    print(f"\nMinimum IG2 coverage DISTRIBUTED that maintains or reduces burden "
          f"(margin {margin_pct:g}% of IRS value, bootstrap {n_boot}):")
    for r in thr.itertuples():
        val = "n/a" if pd.isna(r.min_coverage) else f"{r.min_coverage:.3f}"
        ci = ""
        if hasattr(r, "boot_lo95") and pd.notna(r.boot_lo95):
            ci = f"  boot [{r.boot_lo95:.3f}, {r.boot_hi95:.3f}]"
        if (hasattr(r, "boot_share_not_reached") and pd.notna(r.boot_share_not_reached)
                and r.boot_share_not_reached > 0):
            ci += f"  ({100 * r.boot_share_not_reached:.0f}% of resamples: not reached)"
        print(f"  {r.metric:<10} kill={r.kill_key:<4} {r.criterion:<36} {val:>6}{ci}")
        print(f"  {'':<10} {'':<9} {r.note}")

    print("\nNote: the threshold above is in terms of c, the demographic coverage "
          "distributed. Effective use is lower: nets are lost (Weibull retention) and "
          "the age and season usage curves scale use further. The conversion to "
          "effective use follows below.")

    if plots:
        make_plots(outdir, yearly, yearly_paired, paired, thr, switch_year, campaign_years)
    print(f"\nTables written to {outdir}/")
    return df, thr


def run_monthly(periods_path, outdir, kept_states):
    """Calendar-month view (MONTH_STARTS) of the 30-day reporting periods."""
    if not os.path.isfile(periods_path):
        print(f"\nMonthly view skipped: {periods_path} not found (re-run extraction).")
        return
    p = pd.read_csv(periods_path)
    p["ig2_kill_key"] = p["ig2_kill_key"].fillna("")
    p = p[p["pickup_sim_index"].isin(kept_states)]

    # Period t covers days (t-1)*30 .. t*30-1 of the 365-day year; label by its midpoint.
    mid = (p["period"] - 1) * PERIOD_DAYS + PERIOD_DAYS / 2.0
    p["month"] = np.searchsorted(MONTH_STARTS, mid, side="right")

    keys = SIM_KEYS + ["year", "month"]
    g = p.groupby(keys)[["cases_all", "py_all", "cases_u5", "py_u5",
                         "pfprpop_p5", "pop_p5"]].sum().reset_index()
    g["incidence_u5_per1000py"] = 1000 * g["cases_u5"] / g["py_u5"]
    g["pfpr_u5"] = g["pfprpop_p5"] / g["pop_p5"]
    g[keys + ["cases_all", "cases_u5", "incidence_u5_per1000py", "pfpr_u5"]].to_csv(
        out_file(outdir, "monthly_per_simulation.csv"), index=False)

    gk = ["scenario", "scenario_kind", "ig2_coverage", "ig2_kill_key", "year", "month"]
    rows = []
    for k, d in g.groupby(gk):
        rec = dict(zip(gk, k))
        rec["n_states"] = d["pickup_sim_index"].nunique()
        for c in ("cases_all", "incidence_u5_per1000py", "pfpr_u5"):
            v = d[c].dropna().to_numpy()
            rec[f"{c}_median"] = np.median(v) if len(v) else np.nan
            rec[f"{c}_lo95"] = np.percentile(v, 2.5) if len(v) else np.nan
            rec[f"{c}_hi95"] = np.percentile(v, 97.5) if len(v) else np.nan
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out_file(outdir, "monthly_by_scenario.csv"), index=False)
    print(f"\nMonthly (calendar-month) tables written: monthly_by_scenario_{STEP}.csv, "
          f"monthly_per_simulation_{STEP}.csv")


# =============================================================================
# 5. Realised net use from simulation output (optional diagnostic)
# =============================================================================

def read_channel(path, name):
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        channels = json.load(f).get("Channels", {})
    for key, value in channels.items():
        if key.lower() == name.lower():
            return np.asarray(value["Data"], dtype=float)
    return None


def find_counter_file(out_dir):
    """ReportEventCounter.json, or ReportEventCounter_<suffix>.json (e.g. _NetUse)."""
    exact = os.path.join(out_dir, COUNTER_FILE)
    if os.path.isfile(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(out_dir, COUNTER_GLOB)))
    return hits[0] if hits else None


def recorder_new_nets_by_year(path, first_year):
    """
    Nets issued per calendar year from ReportEventRecorder.csv.
    Counts the named issue events (Received_ITN_IG2 / _ANC / _CWC / _CWC_Catchup)
    when present, otherwise the generic Bednet_Got_New_One (older emodpy, where
    broadcast_event is unsupported). The two are never summed, so a net is not
    counted twice. Returns {year: count}, or None if the file is missing/unreadable.
    Time is the simulation day (day 1 = 1 Jan first_year, flat 365-day years).
    """
    if not os.path.isfile(path):
        return None
    named, generic = defaultdict(int), defaultdict(int)
    try:
        reader = pd.read_csv(path, usecols=lambda c: c.strip() in ("Time", "Event_Name"),
                             chunksize=RECORDER_CHUNK_ROWS, skipinitialspace=True)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            names = chunk["Event_Name"].astype(str).str.strip()
            for target, mask in ((named, names.isin(NET_ISSUE_EVENTS)),
                                 (generic, names == NET_NEW_CHANNEL)):
                hit = chunk[mask]
                if hit.empty:
                    continue
                years = first_year + ((hit["Time"].astype(float) - 1) // DAYS_PER_YEAR).astype(int)
                for y, n in years.value_counts().items():
                    target[int(y)] += int(n)
    except (ValueError, KeyError, OSError) as exc:
        print(f"  WARNING: could not read {path}: {exc}")
        return None
    return dict(named) if sum(named.values()) > 0 else dict(generic)


def run_netuse(index, out_path, first_year):
    rows, skipped, from_recorder, no_issue_data = [], [], 0, 0
    for r in index.itertuples():
        out_dir = os.path.join(r.sim_dir, "output")
        counter = find_counter_file(out_dir)
        pop = read_channel(os.path.join(out_dir, INSET_FILE), POP_CHANNEL)
        using = read_channel(counter, NET_USE_CHANNEL)
        if pop is None or using is None:
            skipped.append((r.scenario, r.pickup_sim_index))
            continue

        new = read_channel(counter, NET_NEW_CHANNEL)
        n = min(len(pop), len(using))
        day = np.arange(n)
        d = pd.DataFrame({"year": first_year + day // DAYS_PER_YEAR,
                          "pop": pop[:n], "using": using[:n]})
        d["new"] = new[:n] if new is not None and len(new) >= n else np.nan

        # The counter only holds NET_USE_EVENTS; nets issued come from the
        # event recorder when the counter does not carry them.
        recorder_counts = None
        if d["new"].isna().all():
            recorder_counts = recorder_new_nets_by_year(os.path.join(out_dir, RECORDER_FILE),
                                                        first_year)
            if recorder_counts is None:
                no_issue_data += 1
            else:
                from_recorder += 1

        for year, g in d.groupby("year"):
            rec = {c: getattr(r, c) for c in TAG_COLUMNS}
            if recorder_counts is not None:
                issued = float(recorder_counts.get(int(year), 0))
            else:
                issued = g["new"].sum(min_count=1)
            rec.update(year=int(year),
                       net_use_per_person=g["using"].sum() / g["pop"].sum(),
                       nets_issued_per_1000=1000 * issued / g["pop"].mean())
            rows.append(rec)

    if skipped:
        print(f"  WARNING: {len(skipped)} simulation(s) lack {POP_CHANNEL!r} in {INSET_FILE} "
              f"or {NET_USE_CHANNEL!r} in a {COUNTER_GLOB} file (e.g. {skipped[:3]}); skipped. "
              f"Was ADD_NET_USE_COUNTER on, and does the installed emodpy-malaria "
              f"provide add_report_event_counter?")
    if from_recorder:
        print(f"  note: nets issued for {from_recorder} simulation(s) counted from "
              f"{RECORDER_FILE} ({', '.join(NET_ISSUE_EVENTS)}, or {NET_NEW_CHANNEL} "
              f"if none of those were recorded).")
    if no_issue_data:
        print(f"  WARNING: {no_issue_data} simulation(s) have no net-issue data in "
              f"the counter or {RECORDER_FILE}; nets issued will be blank for them.")
    if not rows:
        print("No net-use data extracted; skipping.")
        return

    out = pd.DataFrame(rows)
    base = (out[out["scenario_kind"] == "irs"]
            [["pickup_sim_index", "year", "net_use_per_person", "nets_issued_per_1000"]]
            .rename(columns={"net_use_per_person": "use_irs_arm",
                             "nets_issued_per_1000": "issued_irs_arm"}))
    out = out.merge(base, on=["pickup_sim_index", "year"], how="left")
    out["use_increment_vs_irs"] = out["net_use_per_person"] - out["use_irs_arm"]
    out["issued_increment_vs_irs"] = out["nets_issued_per_1000"] - out["issued_irs_arm"]
    out.to_csv(out_path, index=False)

    print("\nSimulated Bednet_Using events per person per day, by scenario and year "
          "(diagnostic).")
    print("(Counts each net separately, so it is an UPPER BOUND on the share of people "
          "using a net. Report the analytic effective use above, not this.)")
    print(out.pivot_table(index="scenario", columns="year",
                          values="net_use_per_person").round(3).to_string())
    inc = out[out["scenario_kind"] != "irs"]
    print("\nIncrease over the IRS_CONTINUE arm (approximately the use due to IG2):")
    print(inc.pivot_table(index="scenario", columns="year",
                          values="use_increment_vs_irs").round(3).to_string())
    print("\nExtra nets issued per 1,000 people vs the IRS_CONTINUE arm "
          "(should spike only in IG2 campaign years):")
    print(inc.pivot_table(index="scenario", columns="year",
                          values="issued_increment_vs_irs").round(0).to_string())
    print(f"\nWrote {len(out)} rows -> {out_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--experiment-id", default=None,
                   help="idmtools experiment ID containing the scenario simulations. "
                        "Overrides the EXPERIMENTS dict at the top of the script.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Names from EXPERIMENTS to analyse (default: all).")
    p.add_argument("--outdir", default=None,
                   help="Parent folder for <name>" + OUT_SUFFIX + " (default: current folder).")
    p.add_argument("--exp-dir", default=None,
                   help="Scan this experiment folder instead of calling idmtools.")
    p.add_argument("--reuse-index", action="store_true",
                   help="Reuse the saved sim_index; do not contact idmtools.")
    p.add_argument("--reuse-results", action="store_true",
                   help="Reuse the saved scenario_results; skip extraction.")
    p.add_argument("--ignore-status", action="store_true",
                   help="Keep simulations idmtools does not report as succeeded.")
    p.add_argument("--first-year", type=int, default=PROJ_START_YEAR)
    p.add_argument("--last-year", type=int, default=PROJ_END_YEAR)
    p.add_argument("--analysis-start-year", type=int, default=2027,
                   help="First year compared. Default 2027 = the IRS switch year.")
    p.add_argument("--switch-year", type=int, default=2027,
                   help="IRS_SWITCH_YEAR; years before it form the pairing check.")
    p.add_argument("--campaign-years", type=int, nargs="+", default=[2027, 2030],
                   help="IG2_CAMPAIGN_YEARS (marks the figures and drives the use calculation).")
    p.add_argument("--ig2-doy", type=int, default=IG2_CAMPAIGN_DOY,
                   help="IG2_CAMPAIGN_DOY: day of year of each IG2 distribution.")
    p.add_argument("--input-dir", default=INPUT_DIR,
                   help="Folder with interventions_ITN_age.csv and interventions_ITN_season.csv.")
    p.add_argument("--no-ig2-curves", action="store_true",
                   help="The scenario ran with IG2_USE_CURVES = False: age and season "
                        "multipliers are 1 (retention still applies).")
    p.add_argument("--skip-use", action="store_true",
                   help="Skip the effective-use calculation.")
    p.add_argument("--margin-pct", type=float, default=0.0,
                   help="Allowed excess vs IRS, %% of the IRS value (default 0).")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples of pickup states (0 = off).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pairing-tol", type=float, default=1e-6)
    p.add_argument("--workers", type=int, default=4,
                   help="Extraction processes (keep modest on the head node).")
    p.add_argument("--skip-netuse", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def resolve_experiments(args):
    if args.experiment_id:
        eid = args.experiment_id.strip()
        name = next((n for n, i in EXPERIMENTS.items() if i == eid), f"exp_{eid[:8]}")
        return {name: eid}
    exps = dict(EXPERIMENTS)
    if args.only:
        unknown = [n for n in args.only if n not in exps]
        if unknown:
            sys.exit(f"--only: {unknown} not in EXPERIMENTS {list(exps)}.")
        exps = {n: exps[n] for n in args.only}
    bad = [n for n, i in exps.items() if not i.strip() or i.startswith("PUT-")]
    if bad:
        sys.exit(f"Set the scenario experiment ID for {bad} in EXPERIMENTS "
                 f"(or pass --experiment-id).")
    return exps


def run_one(name, experiment_id, args):
    outdir = Path(args.outdir or ".").expanduser() / f"{name}{OUT_SUFFIX}"
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = Path(out_file(outdir, "sim_index.csv"))
    results_path = Path(out_file(outdir, "scenario_results.csv"))
    print(f"\n{'=' * 78}\n{name}  ({experiment_id})  step={STEP}\noutput: {outdir}\n{'=' * 78}")

    # 1. Index
    if args.reuse_index and index_path.is_file():
        index = pd.read_csv(index_path)
        index["ig2_kill_key"] = index["ig2_kill_key"].fillna("")
        print(f"Reusing {index_path} ({len(index)} simulations).")
    else:
        raw = (simulations_from_directory(args.exp_dir) if args.exp_dir
               else simulations_from_platform(experiment_id))
        index = check_index(normalise_index(raw), args.ignore_status)
        index.to_csv(index_path, index=False)
        print(f"Wrote {index_path}")

    # 2. Extract
    years = list(range(args.first_year, args.last_year + 1))
    if args.reuse_results and results_path.is_file():
        results = pd.read_csv(results_path)
        print(f"Reusing {results_path} ({len(results)} rows).")
    else:
        results = run_extract(index, str(results_path), years, max(1, args.workers))

    # 3. Analyze
    results, thr = run_analyze(
        results, str(outdir), args.analysis_start_year, args.last_year,
        args.switch_year, args.margin_pct, args.bootstrap, args.seed,
        args.pairing_tol, args.campaign_years, plots=not args.no_plots)
    kept = set(results["pickup_sim_index"].unique())

    # 3a. Effective use (analytic) for every coverage and each threshold
    if not args.skip_use:
        try:
            tab = add_effective_use(
                str(outdir), results, thr, args.input_dir, not args.no_ig2_curves,
                args.analysis_start_year, args.last_year, sorted(args.campaign_years),
                args.first_year, args.ig2_doy)
            if tab is not None and not args.no_plots:
                import matplotlib
                matplotlib.use("Agg")
                plot_effective_use(str(outdir), tab, thr)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nWARNING: effective-use calculation skipped: {exc}")

    # 3b. Calendar-month view
    run_monthly(periods_path_for(results_path), outdir, kept)

    # 4. Simulated net use (diagnostic)
    if not args.skip_netuse:
        run_netuse(index[index["pickup_sim_index"].isin(kept)],
                   out_file(outdir, "net_use.csv"), args.first_year)

    with open(out_file(outdir, "run_info.json"), "w") as f:
        json.dump(dict(name=name, step=STEP, experiment_id=experiment_id,
                       simulations=int(len(index)),
                       pickup_states_analyzed=int(len(kept)),
                       args=vars(args)), f, indent=2, default=str)
    print(f"\nDone. Outputs in {outdir}/")


def main():
    args = parse_args()
    for name, experiment_id in resolve_experiments(args).items():
        run_one(name, experiment_id, args)


if __name__ == "__main__":
    main()