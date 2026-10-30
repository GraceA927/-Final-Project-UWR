#!/usr/bin/env python3
"""UWR seasonality assessment using EMOD Pickup Analyzer output.

Compares normalized seasonal shape, not absolute malaria burden.
EMOD reports 30-day intervals; their mapping to calendar months is approximate.
Incomplete intervals and incomplete yearly profiles are excluded.
Replicate runs remain separate until their individual yearly seasonal profiles
have been calculated.

Outputs:
  UWR_seasonal_scores.csv
  UWR_seasonal_by_year.csv
  UWR_seasonal_climatology.csv
  UWR_seasonal_parameter_effects.csv
  UWR_seasonal_top.png
  UWR_seasonal_best_yearly.png
  UWR_seasonal_best_timeseries.png
  UWR_seasonal_joint_ranking.csv (if calibration ranking exists)
"""

import argparse
import calendar
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# SETTINGS
# ============================================================
PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
EXPERIMENT_NAME = "XTLH_Pickup13yrs"
ANALYZER_DIR = os.path.join(PROJECT_DIR, "experiments", f"{EXPERIMENT_NAME}_AnalyzerOutput")
SIM_PATH = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")
OBS_PATH = os.path.join(PROJECT_DIR, "inputs", "monthly_U5_conf.csv")
CALIB_PATH = os.path.join(ANALYZER_DIR, "calibration_fit_ranking.csv")
OUT_DIR = ANALYZER_DIR

OBS_REGION = "Upper West"
OBS_VALUE_COL = "observed_incidence_per_1000"
START_YEAR = 2015
END_YEAR = 2025
U5_MAX_AGE = 5.0
SIM_U5_ONLY = True
PARAMS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
SEED = "Run_Number"
SIM_ID_COL = "sim_id"
BLOCK_DAYS = 30
N_BLOCKS = 12
MONTHS = list(range(1, 13))
MONTH_NAMES = {m: calendar.month_abbr[m] for m in MONTHS}
NAME_TO_NUM = {}
for _m in MONTHS:
    NAME_TO_NUM[calendar.month_name[_m].lower()] = _m
    NAME_TO_NUM[calendar.month_abbr[_m].lower()] = _m
MONTH_STARTS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
ADJUST_OBS_FOR_MONTH_LENGTH = True
PEAK_TOLERANCE_MONTHS = 1
MAX_LAG = 3
W_PEAK = 0.05
W_AMP = 0.30
N_TOP = 6
SIM_COLOR = "blue"
OBS_COLOR = "black"


# ============================================================
# GENERAL HELPERS
# ============================================================
def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}. Available columns: {df.columns.tolist()}")


def save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def combo_label(row):
    return (
        f"xTLH={float(row['x_Temporary_Larval_Habitat']):.3g}, "
        f"const={float(row['scale_constant']):g}, "
        f"rain={float(row['scale_temp_rain']):g}, "
        f"veg={float(row['scale_water_veg']):g}"
    )


def select_combo(df, combo):
    mask = np.ones(len(df), dtype=bool)
    for p in PARAMS:
        vals = pd.to_numeric(df[p], errors="coerce")
        mask &= np.isclose(vals, float(combo[p]), rtol=1e-7, atol=1e-9, equal_nan=False)
    return df.loc[mask].copy()


def normalize(values):
    """Normalize a finite, nonnegative 12-month profile to mean 1."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        return np.full(values.shape, np.nan, dtype=float)
    mean_value = float(values.mean())
    if not np.isfinite(mean_value) or mean_value <= 0:
        return np.full(values.shape, np.nan, dtype=float)
    return values / mean_value


def correlations(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Tuple indexing works with older and newer SciPy versions.
        return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def circ_diff(sim_month, obs_month):
    """Signed circular month difference; positive means simulated peak is later."""
    difference = int(sim_month) - int(obs_month)
    if difference > 6:
        difference -= 12
    elif difference < -6:
        difference += 12
    return difference


def month_axis(ax, rotate=45):
    ax.set_xticks(MONTHS)
    ax.set_xticklabels([MONTH_NAMES[m] for m in MONTHS], rotation=rotate)


# ============================================================
# REPORTING INTERVAL -> APPROXIMATE CALENDAR MONTH
# ============================================================
def block_to_month(block):
    block = np.asarray(block, dtype=float)
    midpoint = BLOCK_DAYS * (block - 0.5)
    return np.searchsorted(MONTH_STARTS, midpoint, side="right")


def validate_block_mapping():
    blocks = np.arange(1, N_BLOCKS + 1)
    mapped = block_to_month(blocks)
    if len(set(mapped)) != N_BLOCKS or not np.array_equal(mapped, MONTHS):
        raise ValueError(f"Invalid reporting-block mapping: {mapped}")
    print("\nApproximate reporting block -> month mapping:")
    for block, month in zip(blocks, mapped):
        print(f"  Block {block:2d} -> {MONTH_NAMES[month]}")


# ============================================================
# LOAD SIMULATED DATA
# ============================================================
def load_simulated(path):
    print(f"\nLoading simulated data: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sim = pd.read_csv(path)
    sim.columns = sim.columns.astype(str).str.strip()

    required = ["report_interval", "is_complete_interval", "Cases", "Pop", "agebin", "Year", SEED, *PARAMS]
    require_columns(sim, required, "simulated")

    # Robustly accept booleans, 1/0, and common string representations.
    complete_text = sim["is_complete_interval"].astype(str).str.strip().str.lower()
    complete = complete_text.isin(["true", "1", "yes", "y"])
    print("Incomplete interval rows removed:", int((~complete).sum()))
    sim = sim.loc[complete].copy()

    numeric_columns = ["report_interval", "Cases", "Pop", "agebin", "Year", *PARAMS]
    for col in numeric_columns:
        sim[col] = pd.to_numeric(sim[col], errors="coerce")
    sim = sim.dropna(subset=numeric_columns + [SEED])
    sim = sim[
        sim["Year"].between(START_YEAR, END_YEAR)
        & sim["report_interval"].between(1, N_BLOCKS)
    ].copy()
    sim["Year"] = sim["Year"].astype(int)
    sim["report_interval"] = sim["report_interval"].astype(int)

    all_bins = sorted(sim["agebin"].dropna().unique().astype(float))
    print("Age bins found:", all_bins)
    if SIM_U5_ONLY:
        used_bins = [age for age in all_bins if age <= U5_MAX_AGE]
        if not used_bins:
            raise ValueError(f"No age bins <= {U5_MAX_AGE}; available bins: {all_bins}")
        print("Age bins used for under-five analysis:", used_bins)
    else:
        used_bins = all_bins
        print("Using all available age bins.")

    sim = sim[sim["agebin"].isin(used_bins) & (sim["Pop"] > 0)].copy()
    if sim.empty:
        raise ValueError("No simulated records remain after filtering.")
    if (sim["Cases"] < 0).any():
        raise ValueError("Negative simulated Cases values found.")
    if not np.isfinite(sim["Cases"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite simulated Cases values found.")
    if not np.isfinite(sim["Pop"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite simulated Pop values found.")

    # Run identity includes sim_id when that column is available and populated.
    use_sim_id = SIM_ID_COL in sim.columns and sim[SIM_ID_COL].notna().any()
    run_keys = PARAMS + ([SIM_ID_COL] if use_sim_id else []) + [SEED]
    identity = run_keys + ["Year", "report_interval", "agebin"]
    duplicate_count = int(sim.duplicated(identity).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate simulation records for a run/year/interval/age bin. "
            "Check the analyzer CSV and simulation identifiers."
        )

    sim["month"] = block_to_month(sim["report_interval"].to_numpy()).astype(int)
    # Assumes Cases is an age-specific incidence rate. If Cases is a raw count,
    # use sum(Cases) instead of population-weighting it here.
    sim["weighted_cases"] = sim["Cases"] * sim["Pop"]
    group_keys = run_keys + ["Year", "month"]
    per_run = sim.groupby(group_keys, as_index=False).agg(
        weighted_cases=("weighted_cases", "sum"),
        population=("Pop", "sum"),
    )
    per_run["U5_incidence"] = per_run["weighted_cases"] / per_run["population"]

    # Keep only run-years containing exactly one record for every month.
    expected_months = set(MONTHS)
    completeness = per_run.groupby(run_keys + ["Year"])["month"].agg(
        lambda x: set(x.astype(int)) == expected_months and x.nunique() == 12
    )
    complete_year_keys = completeness[completeness].reset_index()[run_keys + ["Year"]]
    per_run = per_run.merge(complete_year_keys, on=run_keys + ["Year"], how="inner", validate="many_to_one")
    if per_run.empty:
        raise ValueError("No complete simulated run-years remain after interval filtering.")

    # Normalize each independent run within each year before averaging replicates.
    per_run["seasonal_index"] = per_run.groupby(run_keys + ["Year"])["U5_incidence"].transform(normalize)
    per_run = per_run[np.isfinite(per_run["seasonal_index"])].copy()
    if per_run.empty:
        raise ValueError("All simulated annual profiles were zero or non-finite; cannot normalize seasonality.")

    # Run-level climatology: each complete year contributes one profile per run.
    run_climatology = per_run.groupby(run_keys + ["month"], as_index=False).agg(
        seasonal_index=("seasonal_index", "mean"),
        mean_incidence=("U5_incidence", "mean"),
        complete_years=("Year", "nunique"),
    )
    run_month_counts = run_climatology.groupby(run_keys)["month"].nunique()
    if not run_month_counts.eq(12).all():
        raise ValueError("At least one run climatology does not contain all 12 months.")

    # Yearly profiles averaged across independent runs, preserving Year.
    sim_yearly = per_run.groupby(PARAMS + ["Year", "month"], as_index=False).agg(
        U5_incidence=("U5_incidence", "mean"),
        Simulated_seasonal_index=("seasonal_index", "mean"),
        Replicates=("seasonal_index", "count"),
        SD_seasonal_index=("seasonal_index", "std"),
    )
    # Re-normalize after replicate averaging to ensure each yearly profile has mean 1.
    sim_yearly["Simulated_seasonal_index"] = sim_yearly.groupby(PARAMS + ["Year"])[
        "Simulated_seasonal_index"
    ].transform(normalize)

    # Across-year climatology, useful for summaries and plotting.
    monthly_climatology = sim_yearly.groupby(PARAMS + ["month"], as_index=False).agg(
        U5_incidence=("U5_incidence", "mean"),
        Simulated_seasonal_index=("Simulated_seasonal_index", "mean"),
        Years_available=("Year", "nunique"),
    )
    monthly_climatology["Simulated_seasonal_index"] = monthly_climatology.groupby(PARAMS)[
        "Simulated_seasonal_index"
    ].transform(normalize)

    print("Parameter combinations:", monthly_climatology[PARAMS].drop_duplicates().shape[0])
    run_counts = run_climatology.groupby(PARAMS)[SEED].nunique()
    print("Independent runs per combination (min-max):", int(run_counts.min()), "-", int(run_counts.max()))
    print("Complete run-year profiles:", per_run[[*run_keys, "Year"]].drop_duplicates().shape[0])
    return monthly_climatology, run_climatology, sim_yearly


# ============================================================
# LOAD OBSERVED DATA
# ============================================================
def parse_month(value):
    if pd.isna(value):
        return np.nan
    try:
        number = float(value)
        if number.is_integer() and 1 <= number <= 12:
            return int(number)
    except (TypeError, ValueError):
        pass
    return NAME_TO_NUM.get(str(value).strip().lower(), np.nan)


def load_observed(path):
    print(f"\nLoading observed data: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obs = pd.read_csv(path)
    obs.columns = obs.columns.astype(str).str.strip()
    require_columns(obs, ["Region", "year", "month", OBS_VALUE_COL], "observed")

    if OBS_REGION:
        obs = obs[obs["Region"].astype(str).str.strip().str.lower().eq(OBS_REGION.strip().lower())].copy()
        if obs.empty:
            raise ValueError(f"No observed rows for region {OBS_REGION!r}.")

    month_raw = obs["month"].copy()
    obs["month"] = month_raw.map(parse_month)
    obs["Year"] = pd.to_numeric(obs["year"], errors="coerce")
    obs["Cases"] = pd.to_numeric(obs[OBS_VALUE_COL], errors="coerce")
    unknown = month_raw[obs["month"].isna()].dropna().unique()
    if len(unknown):
        print("WARNING: unrecognized month labels:", list(unknown))

    obs = obs.dropna(subset=["Year", "month", "Cases"])
    obs = obs[obs["Year"].between(START_YEAR, END_YEAR)].copy()
    obs["Year"] = obs["Year"].astype(int)
    obs["month"] = obs["month"].astype(int)
    obs = obs[obs["month"].between(1, 12)].copy()
    if (obs["Cases"] < 0).any():
        raise ValueError("Negative observed values found.")
    if not np.isfinite(obs["Cases"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite observed values found.")

    duplicates = int(obs.duplicated(["Year", "month"]).sum())
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate observed year-month rows after region filtering. "
            "Check whether the input contains multiple districts or reporting rows."
        )
    obs = obs[["Year", "month", "Cases"]].copy()

    # This converts monthly totals/rates to an approximate daily rate. Set the
    # setting to False if the input already represents a daily rate.
    if ADJUST_OBS_FOR_MONTH_LENGTH:
        days = np.array([calendar.monthrange(y, m)[1] for y, m in zip(obs["Year"], obs["month"])], dtype=float)
        obs["Obs_value"] = obs["Cases"].to_numpy(dtype=float) / days
    else:
        obs["Obs_value"] = obs["Cases"].astype(float)

    obs["Obs_seasonal_index"] = obs.groupby("Year")["Obs_value"].transform(normalize)
    print("Observed value column:", OBS_VALUE_COL)
    print("\nObserved year completeness:")
    for year in range(START_YEAR, END_YEAR + 1):
        present = set(obs.loc[obs["Year"] == year, "month"].astype(int))
        missing = [MONTH_NAMES[m] for m in MONTHS if m not in present]
        print(f"  {year}: " + ("complete" if not missing else f"missing {missing}"))
    return obs


# ============================================================
# COMPLETE YEARS
# ============================================================
def common_complete_years(sim_yearly, obs):
    """Find years complete in observations and in every parameter combination."""
    require_columns(sim_yearly, PARAMS + ["Year", "month"], "yearly simulation")
    require_columns(obs, ["Year", "month"], "observed")

    obs_counts = obs.groupby("Year")["month"].nunique()
    obs_years = set(obs_counts[obs_counts == 12].index.astype(int))

    sim_counts = sim_yearly.groupby(PARAMS + ["Year"])["month"].nunique()
    sim_ok = sim_counts.eq(12).groupby(level="Year").all()
    sim_years = set(sim_ok[sim_ok].index.astype(int))

    years = sorted(obs_years & sim_years)
    print("\nComplete observed years:", sorted(obs_years))
    print("Complete simulated years:", sorted(sim_years))
    print("Years available for scoring:", years)
    if not years:
        raise ValueError("No complete year is shared by simulation and observation.")
    return years


# ============================================================
# PROFILE METRICS
# ============================================================
def profile_metrics(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if len(sim) != 12 or len(obs) != 12 or not np.isfinite(sim).all() or not np.isfinite(obs).all():
        raise ValueError("Profile metrics require two complete finite 12-month arrays.")

    r, rho = correlations(sim, obs)
    lags = {lag: correlations(np.roll(sim, -lag), obs)[0] for lag in range(-MAX_LAG, MAX_LAG + 1)}
    finite_lags = {lag: value for lag, value in lags.items() if np.isfinite(value)}
    best_lag = max(finite_lags, key=finite_lags.get) if finite_lags else np.nan
    sim_peak = int(np.argmax(sim)) + 1
    obs_peak = int(np.argmax(obs)) + 1
    peak_diff = circ_diff(sim_peak, obs_peak)
    sim_amp = float(sim.max() - sim.min())
    obs_amp = float(obs.max() - obs.min())
    amp_ratio = sim_amp / obs_amp if obs_amp > 0 else np.nan

    timing_flag = "OK" if abs(peak_diff) <= PEAK_TOLERANCE_MONTHS else ("LATE" if peak_diff > 0 else "EARLY")
    if not np.isfinite(amp_ratio):
        shape_flag = "UNKNOWN"
    elif amp_ratio < 0.8:
        shape_flag = "TOO_FLAT"
    elif amp_ratio > 1.25:
        shape_flag = "TOO_PEAKY"
    else:
        shape_flag = "OK"

    return {
        "Pearson": r,
        "Spearman": rho,
        "Sim_peak_month": sim_peak,
        "Obs_peak_month": obs_peak,
        "Peak_diff_months": peak_diff,
        "Timing_flag": timing_flag,
        "Amplitude_ratio": amp_ratio,
        "Shape_flag": shape_flag,
        "RMSE": float(np.sqrt(np.mean((sim - obs) ** 2))),
        "Best_lag": best_lag,
        "Pearson_at_best_lag": finite_lags.get(best_lag, np.nan),
    }


def composite(metrics):
    amplitude = metrics["Amplitude_ratio"]
    if not (np.isfinite(metrics["Pearson"]) and np.isfinite(amplitude) and amplitude > 0):
        return np.nan
    return metrics["Pearson"] - W_PEAK * abs(metrics["Peak_diff_months"]) - W_AMP * abs(np.log(amplitude))


# ============================================================
# YEARLY PROFILES AND SCORING
# ============================================================
def yearly_profiles(df, value_col, years):
    """Return complete, finite 12-month normalized profiles keyed by year."""
    profiles = {}
    for year in years:
        group = df.loc[df["Year"] == year, ["month", value_col]].copy()
        group["month"] = pd.to_numeric(group["month"], errors="coerce")
        group[value_col] = pd.to_numeric(group[value_col], errors="coerce")
        group = group.dropna(subset=["month", value_col])
        if group["month"].duplicated().any():
            continue
        group = group.set_index("month").reindex(MONTHS)
        values = group[value_col].to_numpy(dtype=float)
        if len(values) == 12 and np.isfinite(values).all():
            normalized = normalize(values)
            if np.isfinite(normalized).all():
                profiles[int(year)] = normalized
    return profiles


def score_all(sim_yearly, obs, years):
    score_rows, year_rows, climatology_rows = [], [], []
    obs_profiles = yearly_profiles(obs, "Obs_value", years)

    for combo, group in sim_yearly.groupby(PARAMS, dropna=False):
        combo_dict = dict(zip(PARAMS, combo))
        sim_profiles = yearly_profiles(group, "Simulated_seasonal_index", years)
        shared = [year for year in years if year in sim_profiles and year in obs_profiles]
        if not shared:
            continue

        sim_climatology = normalize(np.nanmean(np.vstack([sim_profiles[y] for y in shared]), axis=0))
        obs_climatology = normalize(np.nanmean(np.vstack([obs_profiles[y] for y in shared]), axis=0))
        if not np.isfinite(sim_climatology).all() or not np.isfinite(obs_climatology).all():
            continue

        yearly_metrics = []
        for year in shared:
            metrics = profile_metrics(sim_profiles[year], obs_profiles[year])
            yearly_metrics.append(metrics)
            year_rows.append({**combo_dict, "Year": year, **metrics})

        clim_metrics = profile_metrics(sim_climatology, obs_climatology)
        yearly_pearson = np.array([m["Pearson"] for m in yearly_metrics], dtype=float)
        timing_ok = np.mean([abs(m["Peak_diff_months"]) <= PEAK_TOLERANCE_MONTHS for m in yearly_metrics])
        score_rows.append({
            **combo_dict,
            **clim_metrics,
            "Composite": composite(clim_metrics),
            "Mean_yearly_Pearson": float(np.nanmean(yearly_pearson)) if np.isfinite(yearly_pearson).any() else np.nan,
            "Frac_years_timing_ok": float(timing_ok),
            "Years_evaluated": len(shared),
        })
        for month, sim_value, obs_value in zip(MONTHS, sim_climatology, obs_climatology):
            climatology_rows.append({
                **combo_dict,
                "month": month,
                "Simulated_seasonal_index": sim_value,
                "Observed_seasonal_index": obs_value,
                "Years_evaluated": len(shared),
            })

    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise ValueError("No parameter combination could be scored. Check yearly profiles and observed values.")
    scores = scores.sort_values("Composite", ascending=False, na_position="last").reset_index(drop=True)
    scores.insert(0, "Rank", np.arange(1, len(scores) + 1))
    return scores, pd.DataFrame(year_rows), pd.DataFrame(climatology_rows)


# ============================================================
# REPORTS
# ============================================================
def print_report(scores):
    print(f"\nScored {len(scores)} parameter combinations.")
    columns = PARAMS + ["Pearson", "Peak_diff_months", "Amplitude_ratio", "Composite"]
    print("\nTop 10 parameter combinations:")
    print(scores[columns].head(10).round(3).to_string(index=False))
    print("\nBest-lag distribution:")
    print(scores["Best_lag"].value_counts(dropna=True).sort_index().to_string())
    print("\nTiming flags:", scores["Timing_flag"].value_counts(dropna=False).to_dict())
    print("Shape flags:", scores["Shape_flag"].value_counts(dropna=False).to_dict())

    lag_values = scores["Best_lag"].dropna()
    if lag_values.empty:
        print("No finite best lag; inspect the seasonal profiles.")
        return
    mode_lag = lag_values.mode().iloc[0]
    share = scores["Best_lag"].eq(mode_lag).mean()
    gain_values = scores["Pearson_at_best_lag"] - scores["Pearson"]
    gain = gain_values.median(skipna=True)
    print("\nInterpretation:")
    if mode_lag != 0 and share >= 0.7 and pd.notna(gain) and gain > 0.1:
        print(f"{share:.0%} of combinations fit best at lag {mode_lag:+.0f} month(s). Investigate seasonal forcing, intervention timing, and the approximate interval-to-month mapping.")
    elif scores["Shape_flag"].ne("OK").mean() > 0.7:
        print("Seasonal amplitude differs for most combinations. Investigate transmission and seasonal forcing parameters.")
    else:
        print("Seasonal fits vary across combinations. Interpret the ranking alongside prevalence calibration and epidemiological plausibility.")


def parameter_effects(scores):
    frames = []
    for parameter in PARAMS:
        grouped = scores.groupby(parameter, dropna=False).agg(
            Mean_Pearson=("Pearson", "mean"),
            Mean_Amplitude_ratio=("Amplitude_ratio", "mean"),
            Mean_Peak_difference=("Peak_diff_months", "mean"),
            Mean_Composite=("Composite", "mean"),
            N=("Composite", "count"),
        ).reset_index().rename(columns={parameter: "Value"})
        grouped.insert(0, "Parameter", parameter)
        frames.append(grouped)
    effects = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    save_csv(effects, "UWR_seasonal_parameter_effects.csv")
    print("\nMarginal parameter effects:")
    print(effects.round(3).to_string(index=False))
    return effects


def joint_ranking(scores):
    if not os.path.isfile(CALIB_PATH):
        print("\nCalibration ranking not found; skipping joint ranking.")
        return
    ranking = pd.read_csv(CALIB_PATH)
    ranking.columns = ranking.columns.astype(str).str.strip()
    require_columns(ranking, PARAMS, "calibration ranking")
    for p in PARAMS:
        ranking[p] = pd.to_numeric(ranking[p], errors="coerce")
    ranking = ranking.dropna(subset=PARAMS).drop_duplicates(subset=PARAMS, keep="first").reset_index(drop=True)

    # Prefer a real rank column if present; otherwise use file order.
    existing_rank = next((c for c in ["Calib_rank", "Rank", "rank", "calibration_rank"] if c in ranking.columns), None)
    if existing_rank is not None:
        ranking["Calib_rank"] = pd.to_numeric(ranking[existing_rank], errors="coerce")
        if ranking["Calib_rank"].isna().all():
            ranking["Calib_rank"] = np.arange(1, len(ranking) + 1)
    else:
        ranking["Calib_rank"] = np.arange(1, len(ranking) + 1)

    a, b = scores.copy(), ranking.copy()
    keys = []
    for p in PARAMS:
        key = f"_key_{p}"
        a[key] = pd.to_numeric(a[p], errors="coerce").round(6)
        b[key] = pd.to_numeric(b[p], errors="coerce").round(6)
        keys.append(key)
    joint = a.merge(b[keys + ["Calib_rank"]], on=keys, how="inner", validate="one_to_one")
    if joint.empty:
        print("No matching parameter combinations between seasonal and calibration rankings.")
        return
    joint["Rank_sum"] = joint["Calib_rank"] + joint["Rank"]
    joint = joint.sort_values("Rank_sum").drop(columns=keys)
    save_csv(joint, "UWR_seasonal_joint_ranking.csv")
    print("\nJoint calibration and seasonal ranking:")
    print(joint[PARAMS + ["Calib_rank", "Rank", "Pearson", "Peak_diff_months", "Amplitude_ratio", "Rank_sum"]].head(10).round(3).to_string(index=False))


# ============================================================
# PLOTS
# ============================================================
def plot_top(climatology, scores, n=N_TOP):
    top = scores.head(n)
    if top.empty:
        print("No ranked combinations available for top plot.")
        return
    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, top.iterrows()):
        selected = select_combo(climatology, row).sort_values("month")
        ax.plot(selected["month"], selected["Simulated_seasonal_index"], color=SIM_COLOR, lw=2, label="Simulated")
        ax.plot(selected["month"], selected["Observed_seasonal_index"], color=OBS_COLOR, ls="--", lw=2, label="Observed")
        r = row["Pearson"]
        peak = row["Peak_diff_months"]
        ax.set_title(f"Rank {int(row['Rank'])}; r={r:.2f}\nPeak difference: {peak:+.0f} months\n{combo_label(row)}", fontsize=8)
        month_axis(ax)
        ax.set_xlabel("Approximate calendar month")
        ax.set_ylabel("Seasonal index")
        ax.grid(True)
        ax.legend(fontsize=8)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("UWR seasonal climatology: top parameter combinations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "UWR_seasonal_top.png")


def plot_best_yearly(sim_yearly, obs, best, years):
    sim_selected = select_combo(sim_yearly, best)
    sim_profiles = yearly_profiles(sim_selected, "Simulated_seasonal_index", years)
    obs_profiles = yearly_profiles(obs, "Obs_value", years)
    ncols = 3
    nrows = max(1, int(np.ceil(len(years) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, year in zip(axes, years):
        if year not in sim_profiles or year not in obs_profiles:
            ax.set_title(f"{year}: incomplete")
            ax.axis("off")
            continue
        r, _ = correlations(sim_profiles[year], obs_profiles[year])
        ax.plot(MONTHS, sim_profiles[year], color=SIM_COLOR, lw=2, label="Simulated")
        ax.plot(MONTHS, obs_profiles[year], color=OBS_COLOR, ls="--", lw=2, label="Observed")
        ax.set_title(f"{year}: Pearson r={r:.2f}")
        month_axis(ax)
        ax.set_ylabel("Seasonal index")
        ax.grid(True)
        if year == years[0]:
            ax.legend(fontsize=8)
    for ax in axes[len(years):]:
        ax.axis("off")
    fig.suptitle("Best parameter combination: yearly profiles\n" + combo_label(best), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "UWR_seasonal_best_yearly.png")


def plot_best_timeseries(sim_yearly, obs, best):
    selected = select_combo(sim_yearly, best)
    merged = selected[["Year", "month", "Simulated_seasonal_index"]].merge(
        obs[["Year", "month", "Obs_value"]], on=["Year", "month"], how="inner", validate="one_to_one"
    )
    if merged.empty:
        print("No overlapping records for the time-series plot.")
        return
    merged = merged.sort_values(["Year", "month"])
    merged["Sim_normalized"] = merged.groupby("Year")["Simulated_seasonal_index"].transform(normalize)
    merged["Obs_normalized"] = merged.groupby("Year")["Obs_value"].transform(normalize)
    merged["time"] = merged["Year"] + (merged["month"] - 0.5) / 12
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(merged["time"], merged["Sim_normalized"], color=SIM_COLOR, lw=2, label="Simulated")
    ax.scatter(merged["time"], merged["Obs_normalized"], facecolors="none", edgecolors=OBS_COLOR, s=30, label="Observed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Within-year seasonal index")
    ax.set_title("Best combination: seasonal profile over time\n" + combo_label(best))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "UWR_seasonal_best_timeseries.png")


# ============================================================
# MAIN
# ============================================================
def main():
    global OUT_DIR, CALIB_PATH, OBS_VALUE_COL, OBS_REGION
    parser = argparse.ArgumentParser(description="Assess EMOD seasonal profiles against observed Upper West Region under-five malaria data.")
    parser.add_argument("--sim", default=SIM_PATH, help="Path to pickup analyzer CSV")
    parser.add_argument("--obs", default=OBS_PATH, help="Path to observed monthly CSV")
    parser.add_argument("--out", default=OUT_DIR, help="Output directory")
    parser.add_argument("--obs-col", default=OBS_VALUE_COL, help="Observed monthly value column")
    parser.add_argument("--region", default=OBS_REGION, help="Region label in observed CSV")
    args = parser.parse_args()

    OUT_DIR = args.out
    OBS_VALUE_COL = args.obs_col
    OBS_REGION = args.region
    beside_sim = os.path.join(os.path.dirname(os.path.abspath(args.sim)), "calibration_fit_ranking.csv")
    if os.path.isfile(beside_sim):
        CALIB_PATH = beside_sim
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print(f"UWR SEASONALITY | {START_YEAR}-{END_YEAR} | {EXPERIMENT_NAME}")
    print("=" * 65)
    validate_block_mapping()

    monthly_climatology, run_climatology, sim_yearly = load_simulated(args.sim)
    obs = load_observed(args.obs)
    years = common_complete_years(sim_yearly, obs)
    scores, by_year, climatology = score_all(sim_yearly, obs, years)

    save_csv(scores, "UWR_seasonal_scores.csv")
    save_csv(by_year, "UWR_seasonal_by_year.csv")
    save_csv(climatology, "UWR_seasonal_climatology.csv")
    print_report(scores)
    parameter_effects(scores)
    joint_ranking(scores)

    best = scores.iloc[0]
    print("\nBest-ranked parameter combination:")
    print(combo_label(best))
    print("Composite score:", round(float(best["Composite"]), 4) if pd.notna(best["Composite"]) else "undefined")

    plot_top(climatology, scores)
    plot_best_yearly(sim_yearly, obs, best, years)
    plot_best_timeseries(sim_yearly, obs, best)
    print("\nDONE.")
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
