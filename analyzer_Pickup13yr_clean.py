"""
UWR seasonality assessment from the EMOD Pickup Analyzer output.

Reads
-----
<ANALYZER_DIR>/PfPR_ClinicalIncidence_monthly_pickup.csv
    Simulated monthly-block incidence and population by age bin, seed,
    year, and calibration parameter combination.

<PROJECT_DIR>/inputs/monthly_U5_ID_cases.csv
    Observed data with columns:
    Region, year, month, Pop_5, allout_u5, incidence_per_1000

calibration_fit_ranking.csv (optional, beside the simulated CSV)
    Prevalence/calibration ranking, assumed sorted best-first.

Method
------
1. Analyzer `month` values are 30-day reporting blocks (1-12). Each block is
   mapped to the calendar month containing its midpoint (approximation).
2. Simulated age bins with upper bounds <= U5_MAX_AGE are combined with
   population weights within each seed (Run_Number); seeds are then averaged.
3. Observed incidence per 1,000 is recalculated as allout_u5 / Pop_5 * 1000
   and checked against the supplied incidence_per_1000.
4. Observed monthly incidence is divided by days in the month. Each year's
   12-month profile is divided by its own mean, so only seasonal SHAPE is
   compared, not burden.
5. A year is usable only if it has 12 unique months, all finite and
   non-negative values, and a positive annual mean. Only years usable in both
   simulation and observation are scored.
6. Combinations are scored on the climatology and year by year: Pearson,
   Spearman, peak-month error, amplitude ratio, RMSE, best circular lag, and
   a composite. Combinations with no valid composite are excluded explicitly.

Outputs
-------
UWR_seasonal_scores.csv
UWR_seasonal_by_year.csv
UWR_seasonal_climatology.csv
UWR_seasonal_parameter_effects.csv
UWR_seasonal_top.png
UWR_seasonal_best_yearly.png
UWR_seasonal_best_timeseries.png
UWR_seasonal_joint_ranking.csv   (if calibration_fit_ranking.csv exists)

Usage
-----
python uwr_seasonality_fixed.py
python uwr_seasonality_fixed.py --sim path.csv --obs path.csv --out folder
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
EXPERIMENT_NAME = "Pickup_FOR_ECM13yrs_PREG"

ANALYZER_DIR = os.path.join(
    PROJECT_DIR, "experiments", f"{EXPERIMENT_NAME}_output"
)
SIM_PATH = os.path.join(
    ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv"
)
OBS_PATH = os.path.join(
    PROJECT_DIR, "inputs", "monthly_U5_ID_cases.csv"
)
OUT_DIR = "/shared/home/grace/Final-Project-UWR/experiments/output"

OBS_REGION = "Upper West"

START_YEAR = 2015
END_YEAR = 2025

# Assumes analyzer `agebin` values are age-bin upper bounds in years.
U5_MAX_AGE = 5.0

PARAMS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
SEED = "Run_Number"

BLOCK_DAYS = 30
N_BLOCKS = 12

# Non-leap-year month starts (zero-based day of year).
MONTH_STARTS = np.array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
)

MONTHS = list(range(1, 13))
MONTH_NAMES = {m: calendar.month_abbr[m] for m in MONTHS}

_NAME_TO_NUM = {}
for _i in MONTHS:
    _NAME_TO_NUM[calendar.month_name[_i].lower()] = _i
    _NAME_TO_NUM[calendar.month_abbr[_i].lower()] = _i

ADJUST_OBS_FOR_MONTH_LENGTH = True
PEAK_TOLERANCE_MONTHS = 1
MAX_LAG = 3

# Composite = Pearson - W_PEAK*|peak error| - W_AMP*|log(amplitude ratio)|
W_PEAK = 0.05
W_AMP = 0.30

N_TOP = 6
SIM_COLOR = "blue"
OBS_COLOR = "black"


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


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
        f"xTLH={row['x_Temporary_Larval_Habitat']:.3g}, "
        f"const={row['scale_constant']:g}, "
        f"rain={row['scale_temp_rain']:g}, "
        f"veg={row['scale_water_veg']:g}"
    )


def select_combo(df, combo):
    mask = np.ones(len(df), dtype=bool)
    for parameter in PARAMS:
        mask &= np.isclose(
            pd.to_numeric(df[parameter], errors="coerce"),
            float(combo[parameter]),
            rtol=1e-7,
            atol=1e-9,
        )
    return df.loc[mask].copy()


def normalize(values):
    """Divide a 12-month profile by its mean; NaN array if undefined."""
    values = np.asarray(values, dtype=float)

    if values.size != 12 or not np.isfinite(values).all():
        return np.full(values.shape, np.nan, dtype=float)

    mean_value = float(np.mean(values))

    # A seasonal index is undefined when the annual mean is zero.
    if mean_value <= 0:
        return np.full(values.shape, np.nan, dtype=float)

    return values / mean_value


def correlations(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            float(pearsonr(x, y).statistic),
            float(spearmanr(x, y).statistic),
        )


def circ_diff(sim_month, obs_month):
    """Signed month difference; positive means simulated peak is later."""
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
# ANALYZER BLOCK -> APPROXIMATE CALENDAR MONTH
# ============================================================

def block_to_month(block):
    block = np.asarray(block, dtype=float)
    midpoint = BLOCK_DAYS * (block - 0.5)
    return np.searchsorted(MONTH_STARTS, midpoint, side="right")


def validate_block_mapping():
    months = block_to_month(np.arange(1, N_BLOCKS + 1))
    if len(set(months)) != N_BLOCKS or np.any((months < 1) | (months > 12)):
        raise ValueError(
            "The 30-day reporting blocks do not map one-to-one to months. "
            "Check BLOCK_DAYS, N_BLOCKS, and MONTH_STARTS."
        )
    mapping = {
        int(b): MONTH_NAMES[int(m)]
        for b, m in zip(range(1, N_BLOCKS + 1), months)
    }
    print("Approximate block -> calendar month mapping:", mapping)


# ============================================================
# LOAD SIMULATED DATA
# ============================================================

def load_simulated(path):
    print(f"\nLoading simulated data: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    sim = pd.read_csv(path)
    require_columns(
        sim, ["month", "Cases", "Pop", "agebin", "Year", SEED] + PARAMS,
        "simulated",
    )

    for column in ["month", "Cases", "Pop", "agebin", "Year"] + PARAMS:
        sim[column] = pd.to_numeric(sim[column], errors="coerce")

    invalid = [p for p in PARAMS if sim[p].isna().any()]
    if invalid:
        raise ValueError(f"Missing/invalid calibration tags (NaN): {invalid}")

    n_before = len(sim)
    sim = sim.dropna(
        subset=["month", "Cases", "Pop", "agebin", "Year", SEED]
    ).copy()
    if len(sim) < n_before:
        print(
            f"WARNING: dropped {n_before - len(sim):,} simulated rows with "
            "missing values in month/Cases/Pop/agebin/Year/seed."
        )

    sim = sim[sim["Year"].between(START_YEAR, END_YEAR)].copy()
    sim = sim[sim["month"].between(1, N_BLOCKS)].copy()
    sim["Year"] = sim["Year"].astype(int)
    sim["month"] = block_to_month(sim["month"]).astype(int)

    if (sim["Cases"] < 0).any():
        raise ValueError("Negative simulated incidence values found.")

    sim = sim[sim["Pop"] > 0].copy()

    used_age_bins = sorted(
        float(v) for v in sim["agebin"].unique() if float(v) <= U5_MAX_AGE
    )
    if not used_age_bins:
        raise ValueError(
            f"No simulated age bins at or below U5_MAX_AGE={U5_MAX_AGE}. "
            f"Available: {sorted(sim['agebin'].unique())}"
        )
    print(f"Under-five age bins used (upper bounds): {used_age_bins}")
    sim = sim[sim["agebin"].isin(used_age_bins)].copy()

    # Cases is an age-bin incidence rate; population-weight within seed/month.
    sim["weighted_cases"] = sim["Cases"] * sim["Pop"]
    per_seed = sim.groupby(
        PARAMS + [SEED, "Year", "month"], as_index=False
    ).agg(weighted_cases=("weighted_cases", "sum"), Pop_U5=("Pop", "sum"))
    per_seed["U5_incidence"] = per_seed["weighted_cases"] / per_seed["Pop_U5"]

    monthly = per_seed.groupby(
        PARAMS + ["Year", "month"], as_index=False
    ).agg(
        U5_incidence=("U5_incidence", "mean"),
        Seeds=("U5_incidence", "count"),
    )

    print("Parameter combinations:", monthly[PARAMS].drop_duplicates().shape[0])
    print(
        "Seeds per combination (min-max): "
        f"{int(monthly['Seeds'].min())}-{int(monthly['Seeds'].max())}"
    )
    return monthly


# ============================================================
# LOAD OBSERVED DATA
# ============================================================

def parse_month(value):
    if pd.isna(value):
        return np.nan
    try:
        numeric_value = float(value)
        if numeric_value.is_integer() and 1 <= numeric_value <= 12:
            return int(numeric_value)
        return np.nan
    except (TypeError, ValueError):
        pass
    return _NAME_TO_NUM.get(str(value).strip().lower(), np.nan)


def load_observed(path):
    print(f"\nLoading observed data: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    obs = pd.read_csv(path)
    require_columns(
        obs,
        ["Region", "year", "month", "Pop_5", "allout_u5",
         "incidence_per_1000"],
        "observed",
    )

    obs["Region"] = obs["Region"].astype(str).str.strip()
    available_regions = sorted(obs["Region"].dropna().unique())
    obs = obs[obs["Region"].str.casefold().eq(OBS_REGION.casefold())].copy()
    if obs.empty:
        raise ValueError(
            f"No Region matching '{OBS_REGION}'. "
            f"Available Region values: {available_regions}"
        )

    obs["Year"] = pd.to_numeric(obs["year"], errors="coerce")

    # Check unrecognized labels BEFORE they are converted to NaN.
    raw_month = obs["month"].copy()
    parsed_month = raw_month.map(parse_month)
    unknown_mask = parsed_month.isna() & raw_month.notna()
    if unknown_mask.any():
        print(
            "WARNING: unrecognized month labels (rows dropped): "
            f"{raw_month.loc[unknown_mask].unique().tolist()}"
        )
    obs["month"] = parsed_month

    obs["Pop_5"] = pd.to_numeric(obs["Pop_5"], errors="coerce")
    obs["Cases"] = pd.to_numeric(obs["allout_u5"], errors="coerce")
    obs["provided_incidence_per_1000"] = pd.to_numeric(
        obs["incidence_per_1000"], errors="coerce"
    )

    obs = obs.dropna(subset=["Year", "month", "Pop_5", "Cases"]).copy()
    obs = obs[obs["Year"].between(START_YEAR, END_YEAR)].copy()
    if obs.empty:
        raise ValueError(
            f"No valid observed rows remain for {START_YEAR}-{END_YEAR}."
        )

    obs["Year"] = obs["Year"].astype(int)
    obs["month"] = obs["month"].astype(int)

    if (obs["Pop_5"] <= 0).any():
        raise ValueError("Pop_5 must be positive for every observed row.")
    if (obs["Cases"] < 0).any():
        raise ValueError("Negative observed case counts found.")

    # Duplicate region-month rows: sum counts and population.
    obs = obs.groupby(["Year", "month"], as_index=False).agg(
        Cases=("Cases", "sum"),
        Pop_5=("Pop_5", "sum"),
        provided_incidence_per_1000=("provided_incidence_per_1000", "mean"),
    )

    obs["incidence_per_1000"] = obs["Cases"] / obs["Pop_5"] * 1000.0

    valid_provided = obs["provided_incidence_per_1000"].notna()
    if valid_provided.any():
        provided = obs.loc[valid_provided, "provided_incidence_per_1000"]
        recomputed = obs.loc[valid_provided, "incidence_per_1000"]
        print(
            "Maximum absolute difference between supplied and recalculated "
            f"incidence per 1,000: {float((provided - recomputed).abs().max()):.6g}"
        )
        if not np.allclose(provided, recomputed, rtol=0.02, atol=0.1):
            print(
                "WARNING: supplied incidence_per_1000 differs from "
                "allout_u5 / Pop_5 * 1000. Check the source definition, "
                "denominator, and whether allout_u5 means cases. "
                "Continuing with recalculated incidence."
            )

    if ADJUST_OBS_FOR_MONTH_LENGTH:
        days = np.array(
            [calendar.monthrange(y, m)[1]
             for y, m in zip(obs["Year"], obs["month"])],
            dtype=float,
        )
        obs["Obs_value"] = obs["incidence_per_1000"] / days
    else:
        obs["Obs_value"] = obs["incidence_per_1000"].astype(float)

    print(f"\nObserved {OBS_REGION} data summary:")
    print(f"  Rows: {len(obs)}")
    print(f"  Years: {obs['Year'].min()}-{obs['Year'].max()}")
    print(
        "  Incidence per 1,000 (min, median, max): "
        f"{obs['incidence_per_1000'].min():.3f}, "
        f"{obs['incidence_per_1000'].median():.3f}, "
        f"{obs['incidence_per_1000'].max():.3f}"
    )

    print("\nObserved completeness:")
    for year in range(START_YEAR, END_YEAR + 1):
        have = set(obs.loc[obs["Year"] == year, "month"])
        missing = [MONTH_NAMES[m] for m in MONTHS if m not in have]
        print(f"  {year}: " + ("complete" if not missing else f"missing {missing}"))

    return obs


# ============================================================
# COMPLETE-YEAR SELECTION AND PROFILES
# ============================================================

def common_complete_years(sim_monthly, obs):
    """
    Years with 12 months in observation and in at least one combination.
    Combinations lacking a usable profile for a year are handled per
    combination in score_all(), so one damaged run cannot remove a year
    for every other combination.
    """
    obs_counts = obs.groupby("Year")["month"].nunique()
    obs_years = set(obs_counts[obs_counts == 12].index)

    sim_counts = sim_monthly.groupby(PARAMS + ["Year"])["month"].nunique()
    sim_ok = sim_counts.eq(12).groupby("Year").any()
    sim_years = set(sim_ok[sim_ok].index)

    years = sorted(int(y) for y in (obs_years & sim_years))

    print(f"\nComplete observed years:  {sorted(obs_years)}")
    print(f"Simulated years complete in >=1 combination: {sorted(sim_years)}")
    print(f"Years used for scoring:   {years}")

    if not years:
        raise ValueError(
            "No year is complete in both simulation and observation."
        )
    return years


def yearly_profiles(df, value_column, years):
    """
    Return {year: normalized 12-element array}.
    Invalid, incomplete, negative, or zero-mean profiles are excluded.
    """
    profiles = {}

    for year in years:
        group = df.loc[df["Year"] == year]

        if len(group) != 12 or group["month"].nunique() != 12:
            continue

        group = group.sort_values("month")
        if group["month"].tolist() != MONTHS:
            continue

        values = pd.to_numeric(
            group[value_column], errors="coerce"
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all() or np.any(values < 0):
            continue

        normalized = normalize(values)
        if not np.isfinite(normalized).all():
            continue

        profiles[int(year)] = normalized

    return profiles


# ============================================================
# PROFILE METRICS
# ============================================================

def profile_metrics(sim, obs):
    """Compute metrics for normalized 12-month profiles (January-December)."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)

    invalid = {
        "Pearson": np.nan,
        "Spearman": np.nan,
        "Sim_peak_month": np.nan,
        "Obs_peak_month": np.nan,
        "Peak_diff_months": np.nan,
        "Timing_flag": "UNKNOWN",
        "Amplitude_ratio": np.nan,
        "Shape_flag": "UNKNOWN",
        "RMSE": np.nan,
        "Best_lag": np.nan,
        "Pearson_at_best_lag": np.nan,
    }

    if (
        sim.size != 12
        or obs.size != 12
        or not np.isfinite(sim).all()
        or not np.isfinite(obs).all()
    ):
        return invalid

    if np.any(sim < 0) or np.any(obs < 0):
        return invalid

    sim_amplitude = float(np.max(sim) - np.min(sim))
    obs_amplitude = float(np.max(obs) - np.min(obs))

    # A constant profile has no identifiable seasonal peak.
    if sim_amplitude <= 0 or obs_amplitude <= 0:
        result = invalid.copy()
        result["RMSE"] = float(np.sqrt(np.mean((sim - obs) ** 2)))
        return result

    r, rho = correlations(sim, obs)

    lag_correlations = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        lag_correlations[lag] = correlations(np.roll(sim, -lag), obs)[0]

    finite_lags = {
        lag: value for lag, value in lag_correlations.items()
        if np.isfinite(value)
    }
    best_lag = max(finite_lags, key=finite_lags.get) if finite_lags else np.nan

    sim_peak = int(np.argmax(sim)) + 1
    obs_peak = int(np.argmax(obs)) + 1
    peak_difference = circ_diff(sim_peak, obs_peak)
    amplitude_ratio = sim_amplitude / obs_amplitude

    if abs(peak_difference) <= PEAK_TOLERANCE_MONTHS:
        timing_flag = "OK"
    else:
        timing_flag = "LATE" if peak_difference > 0 else "EARLY"

    if amplitude_ratio < 0.8:
        shape_flag = "TOO_FLAT"
    elif amplitude_ratio > 1.25:
        shape_flag = "TOO_PEAKY"
    else:
        shape_flag = "OK"

    return {
        "Pearson": r,
        "Spearman": rho,
        "Sim_peak_month": sim_peak,
        "Obs_peak_month": obs_peak,
        "Peak_diff_months": peak_difference,
        "Timing_flag": timing_flag,
        "Amplitude_ratio": amplitude_ratio,
        "Shape_flag": shape_flag,
        "RMSE": float(np.sqrt(np.mean((sim - obs) ** 2))),
        "Best_lag": best_lag,
        "Pearson_at_best_lag": (
            finite_lags.get(best_lag, np.nan)
            if np.isfinite(best_lag) else np.nan
        ),
    }


def composite(metrics):
    pearson = metrics["Pearson"]
    amplitude_ratio = metrics["Amplitude_ratio"]
    peak_difference = metrics["Peak_diff_months"]

    if not (
        np.isfinite(pearson)
        and np.isfinite(amplitude_ratio)
        and amplitude_ratio > 0
        and np.isfinite(peak_difference)
    ):
        return np.nan

    return (
        pearson
        - W_PEAK * abs(peak_difference)
        - W_AMP * abs(np.log(amplitude_ratio))
    )


# ============================================================
# SCORING
# ============================================================

def score_all(sim_monthly, obs, years):
    obs_profiles = yearly_profiles(obs, "Obs_value", years)

    if not obs_profiles:
        raise ValueError(
            "No usable observed yearly profiles. Check monthly completeness, "
            "observed incidence values, and whether annual mean incidence "
            "is greater than zero."
        )

    score_rows, year_rows, climatology_rows = [], [], []
    skipped = []

    for combination, group in sim_monthly.groupby(PARAMS, dropna=False):
        combination_dict = dict(zip(PARAMS, combination))
        sim_profiles = yearly_profiles(group, "U5_incidence", years)

        shared_years = [
            y for y in years if y in sim_profiles and y in obs_profiles
        ]
        if not shared_years:
            skipped.append(combination_dict)
            continue

        sim_climatology = np.mean(
            np.vstack([sim_profiles[y] for y in shared_years]), axis=0
        )
        obs_climatology = np.mean(
            np.vstack([obs_profiles[y] for y in shared_years]), axis=0
        )

        yearly_metrics = []
        for year in shared_years:
            metrics = profile_metrics(sim_profiles[year], obs_profiles[year])
            yearly_metrics.append(metrics)
            year_rows.append({
                **combination_dict,
                "Year": year,
                **metrics,
                "Composite": composite(metrics),
            })

        climatology_metrics = profile_metrics(sim_climatology, obs_climatology)

        yearly_pearson = np.array(
            [m["Pearson"] for m in yearly_metrics], dtype=float
        )
        valid_peaks = [
            m for m in yearly_metrics if np.isfinite(m["Peak_diff_months"])
        ]
        frac_timing_ok = (
            float(np.mean([
                abs(m["Peak_diff_months"]) <= PEAK_TOLERANCE_MONTHS
                for m in valid_peaks
            ]))
            if valid_peaks else np.nan
        )

        score_rows.append({
            **combination_dict,
            **climatology_metrics,
            "Composite": composite(climatology_metrics),
            "Mean_yearly_Pearson": (
                float(np.nanmean(yearly_pearson))
                if np.isfinite(yearly_pearson).any() else np.nan
            ),
            "Frac_years_timing_ok": frac_timing_ok,
            "Years_evaluated": len(shared_years),
        })

        for month, s_val, o_val in zip(MONTHS, sim_climatology, obs_climatology):
            climatology_rows.append({
                **combination_dict,
                "month": month,
                "Simulated_seasonal_index": s_val,
                "Observed_seasonal_index": o_val,
            })

    if skipped:
        print(
            f"\nSkipped {len(skipped)} parameter combination(s) with no "
            "usable shared years (missing, negative, non-finite, or "
            "zero-mean simulated profiles). Examples:"
        )
        for example in skipped[:5]:
            print("  ", example)

    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise ValueError(
            "No parameter combination could be scored. Check that simulated "
            "U5 incidence is finite and positive in at least one complete "
            "year, and that observed profiles are also usable."
        )

    # Retain only combinations with a valid composite score.
    n_before = len(scores)
    scores = scores.loc[np.isfinite(scores["Composite"])].copy()
    if scores.empty:
        raise ValueError(
            "Parameter combinations were found, but none had a valid "
            "composite score. Inspect the seasonal correlations, amplitudes, "
            "and peak-month metrics (constant simulated profiles give no "
            "peak)."
        )
    if len(scores) < n_before:
        print(
            f"Excluded {n_before - len(scores)} combination(s) with no valid "
            "composite score (e.g., constant climatological profile)."
        )

    n_years = scores["Years_evaluated"]
    if n_years.min() < n_years.max():
        print(
            f"NOTE: combinations were scored on different numbers of years "
            f"({int(n_years.min())}-{int(n_years.max())}). Rankings are less "
            "comparable for combinations with fewer years; see the "
            "Years_evaluated column."
        )

    scores = scores.sort_values("Composite", ascending=False).reset_index(
        drop=True
    )
    scores.insert(0, "Rank", np.arange(1, len(scores) + 1))

    return scores, pd.DataFrame(year_rows), pd.DataFrame(climatology_rows)


# ============================================================
# REPORTING AND PARAMETER EFFECTS
# ============================================================

def print_report(scores):
    print(f"\nScored {len(scores)} parameter combinations.")

    columns = PARAMS + [
        "Pearson", "Spearman", "Peak_diff_months",
        "Amplitude_ratio", "Composite",
    ]
    print("\nTop 10 by composite score:")
    print(scores[columns].head(10).round(3).to_string(index=False))

    print("\nBest-lag distribution (positive = simulated late):")
    print(scores["Best_lag"].value_counts().sort_index().to_string())

    print("\nTiming flags:", scores["Timing_flag"].value_counts().to_dict())
    print("Shape flags: ", scores["Shape_flag"].value_counts().to_dict())

    finite_lags = scores["Best_lag"].dropna()
    if finite_lags.empty:
        print("\nVerdict: No finite lag estimates were available.")
        return

    mode_lag = finite_lags.mode().iloc[0]
    share = (finite_lags == mode_lag).mean()
    gain = (scores["Pearson_at_best_lag"] - scores["Pearson"]).median()

    print("\nVerdict:")
    if mode_lag != 0 and share >= 0.7 and gain > 0.1:
        print(
            f"  {share:.0%} of combinations fit best at lag "
            f"{mode_lag:+.0f} month(s): the offset may be structural. "
            "Check rainfall/temperature forcing, habitat dynamics, SMC "
            "timing, and delays or definitions in observed reporting."
        )
    elif (scores["Shape_flag"] != "OK").mean() > 0.7:
        print(
            "  Seasonal amplitude is outside the target range for most "
            "combinations. Examine seasonal forcing and intervention timing "
            "before changing parameters."
        )
    else:
        print(
            "  Parameter selection may improve the seasonal fit. "
            "Review top-ranked combinations and yearly performance."
        )


def parameter_effects(scores):
    frames = []
    for parameter in PARAMS:
        grouped = scores.groupby(parameter).agg(
            Mean_Pearson=("Pearson", "mean"),
            Mean_Amplitude_ratio=("Amplitude_ratio", "mean"),
            Mean_Peak_difference=("Peak_diff_months", "mean"),
            Mean_Composite=("Composite", "mean"),
            N=("Composite", "size"),
        ).reset_index().rename(columns={parameter: "Value"})
        grouped.insert(0, "Parameter", parameter)
        frames.append(grouped)

    effects = pd.concat(frames, ignore_index=True)
    print("\nParameter effects (marginal means):")
    print(effects.round(3).to_string(index=False))
    return effects


def joint_ranking(scores, calibration_path):
    if not os.path.isfile(calibration_path):
        print("\nNo calibration ranking file; skipping joint ranking.")
        return

    ranking = pd.read_csv(calibration_path)
    require_columns(ranking, PARAMS, "calibration ranking")

    # Assumes calibration_fit_ranking.csv is already sorted best-first.
    ranking = ranking.copy()
    ranking["Calib_rank"] = np.arange(1, len(ranking) + 1)

    seasonal = scores.copy()
    keys = []
    for parameter in PARAMS:
        key = f"_key_{parameter}"
        seasonal[key] = pd.to_numeric(seasonal[parameter], errors="coerce").astype(float).round(6)
        ranking[key] = pd.to_numeric(ranking[parameter], errors="coerce").astype(float).round(6)
        keys.append(key)

    joint = seasonal.merge(ranking[keys + ["Calib_rank"]], on=keys, how="inner")
    if joint.empty:
        print(
            "\nNo parameter combinations matched between the seasonal and "
            "calibration ranking files. Check parameter precision/values."
        )
        return

    joint["Rank_sum"] = joint["Calib_rank"] + joint["Rank"]
    joint = joint.sort_values("Rank_sum").drop(columns=keys)
    save_csv(joint, "UWR_seasonal_joint_ranking.csv")

    columns = PARAMS + [
        "Calib_rank", "Rank", "Pearson",
        "Peak_diff_months", "Amplitude_ratio", "Rank_sum",
    ]
    print("\nTop 10 by prevalence rank + seasonal rank:")
    print(joint[columns].head(10).round(3).to_string(index=False))


# ============================================================
# PLOTS
# ============================================================

def plot_top(climatology, scores, n=N_TOP):
    top = scores.head(n)
    if top.empty:
        print("No top combinations available for plotting.")
        return

    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, top.iterrows()):
        current = select_combo(climatology, row).sort_values("month")
        ax.plot(
            current["month"], current["Simulated_seasonal_index"],
            color=SIM_COLOR, linewidth=2, label="Simulated",
        )
        ax.plot(
            current["month"], current["Observed_seasonal_index"],
            color=OBS_COLOR, linestyle="--", linewidth=2, label="Observed",
        )
        ax.set_title(
            f"#{int(row['Rank'])} r={row['Pearson']:.2f} "
            f"peak {MONTH_NAMES[int(row['Sim_peak_month'])]} vs "
            f"{MONTH_NAMES[int(row['Obs_peak_month'])]} "
            f"amp x{row['Amplitude_ratio']:.2f}\n{combo_label(row)}",
            fontsize=8,
        )
        month_axis(ax)
        ax.set_xlabel("Approximate calendar month")
        ax.set_ylabel("Seasonal index (value / annual mean)")
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[len(top):]:
        ax.axis("off")

    fig.suptitle(
        f"UWR seasonality climatology, top {len(top)} combinations",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "UWR_seasonal_top.png")


def plot_best_yearly(sim_monthly, obs, best, years):
    simulated = select_combo(sim_monthly, best)
    sim_profiles = yearly_profiles(simulated, "U5_incidence", years)
    obs_profiles = yearly_profiles(obs, "Obs_value", years)

    ncols = 3
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, 3.6 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for ax, year in zip(axes, years):
        if year not in sim_profiles or year not in obs_profiles:
            ax.set_title(f"{year} (unusable profile)")
            ax.axis("off")
            continue

        r, _ = correlations(sim_profiles[year], obs_profiles[year])
        ax.plot(MONTHS, sim_profiles[year], color=SIM_COLOR,
                linewidth=2, label="Simulated")
        ax.plot(MONTHS, obs_profiles[year], color=OBS_COLOR,
                linestyle="--", linewidth=2, label="Observed")
        ax.set_title(f"{year} (r={r:.2f})")
        month_axis(ax)
        ax.set_ylabel("Seasonal index")
        ax.grid(True)
        if year == years[0]:
            ax.legend(fontsize=8)

    for ax in axes[len(years):]:
        ax.axis("off")

    fig.suptitle(f"Best combination by year\n{combo_label(best)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "UWR_seasonal_best_yearly.png")


def plot_best_timeseries(sim_monthly, obs, best):
    simulated = select_combo(sim_monthly, best)[["Year", "month", "U5_incidence"]]
    observed = obs[["Year", "month", "Obs_value"]]

    merged = simulated.merge(observed, on=["Year", "month"], how="inner")
    if merged.empty:
        print("No overlapping months for time-series plot.")
        return

    merged = merged.sort_values(["Year", "month"])
    merged["time"] = merged["Year"] + (merged["month"] - 0.5) / 12.0

    sim_mean = merged["U5_incidence"].mean()
    obs_mean = merged["Obs_value"].mean()
    if not (np.isfinite(sim_mean) and np.isfinite(obs_mean)
            and sim_mean > 0 and obs_mean > 0):
        print("Cannot normalize timeline because a mean incidence is zero.")
        return

    merged["Sim_normalized"] = merged["U5_incidence"] / sim_mean
    merged["Obs_normalized"] = merged["Obs_value"] / obs_mean

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(merged["time"], merged["Sim_normalized"], color=SIM_COLOR,
            linewidth=2, label="Simulated (value / mean)")
    ax.scatter(merged["time"], merged["Obs_normalized"], facecolors="none",
               edgecolors=OBS_COLOR, s=35, label="Observed (value / mean)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized U5 incidence")
    ax.set_title(f"Best combination, monthly timeline\n{combo_label(best)}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "UWR_seasonal_best_timeseries.png")


# ============================================================
# MAIN
# ============================================================

def main():
    global OUT_DIR, START_YEAR, END_YEAR

    parser = argparse.ArgumentParser(
        description="Assess observed and simulated UWR malaria seasonality."
    )
    parser.add_argument("--sim", default=SIM_PATH, help="Simulated analyzer CSV")
    parser.add_argument("--obs", default=OBS_PATH, help="Observed monthly CSV")
    parser.add_argument("--out", default=OUT_DIR, help="Output directory")
    parser.add_argument("--start-year", type=int, default=START_YEAR,
                        help="First year included in assessment")
    parser.add_argument("--end-year", type=int, default=END_YEAR,
                        help="Last year included in assessment")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    START_YEAR = args.start_year
    END_YEAR = args.end_year
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    calibration_path = os.path.join(
        os.path.dirname(os.path.abspath(args.sim)),
        "calibration_fit_ranking.csv",
    )

    print("=" * 68)
    print(f"UWR SEASONALITY | {START_YEAR}-{END_YEAR} | {EXPERIMENT_NAME}")
    print("=" * 68)
    print("Observed input:  ", args.obs)
    print("Simulated input: ", args.sim)
    print("Output directory:", OUT_DIR)

    validate_block_mapping()

    simulated_monthly = load_simulated(args.sim)
    observed = load_observed(args.obs)
    years = common_complete_years(simulated_monthly, observed)

    scores, by_year, climatology = score_all(simulated_monthly, observed, years)

    save_csv(scores, "UWR_seasonal_scores.csv")
    save_csv(by_year, "UWR_seasonal_by_year.csv")
    save_csv(climatology, "UWR_seasonal_climatology.csv")

    effects = parameter_effects(scores)
    save_csv(effects, "UWR_seasonal_parameter_effects.csv")

    print_report(scores)
    joint_ranking(scores, calibration_path)

    best = scores.iloc[0]
    plot_top(climatology, scores)
    plot_best_yearly(simulated_monthly, observed, best, years)
    plot_best_timeseries(simulated_monthly, observed, best)

    print("\nDONE. Output directory:", OUT_DIR)
    print(
        "\nInterpretation reminder: this script ranks normalized seasonal "
        "shape. It does not establish that absolute simulated incidence "
        "matches observed incidence per 1,000."
    )


if __name__ == "__main__":
    main()