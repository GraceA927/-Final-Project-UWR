"""
UWR EMOD seasonality diagnostic (2015-2025).

Assesses whether the pickup model reproduces the observed seasonality of
malaria in Upper West Region, Ghana, using the analyzer's
PfPR_ClinicalIncidence_monthly_pickup.csv.

"Cases" in that file is an annualized clinical incidence rate, so seasonality
is compared on normalized U5 incidence profiles (no Cases x Pop / 12). The two
U5 bins (0.25 = 0-3 months, 5.0 = 3 months-5 years) are combined with a
population-weighted mean, not a sum.

Analyses
    1. Year-specific: for every year, each habitat combination's Jan-Dec
       profile (divided by that year's mean) vs the observed profile: Pearson,
       Spearman, peak month, peak timing difference, peak amplitude.
    2. Climatological: normalize each year first, then average by month across
       years, and compare observed vs simulated peak timing and amplitude.
    3. Diagnostic classes: TOO_EARLY / TOO_LATE / CORRECT_TIMING and
       TOO_WEAK / TOO_STRONG / SIMILAR_STRENGTH.

This is a diagnostic, not a new calibration score. The combination used for
plots is simply the one with the highest climatological Pearson correlation.

Outputs (in ANALYZER_DIR)
    UWR_year_specific_seasonality.csv
    UWR_seasonality_combination_summary.csv
    UWR_climatological_seasonality.csv
    UWR_seasonality_diagnostics.csv
    UWR_seasonality_yearly.png
    UWR_seasonality_climatology.png
    UWR_peak_timing.png

Author: Grace A. Asuru
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# SETTINGS

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
ANALYZER_DIR = os.path.join(PROJECT_DIR, "experiments", "UWR_Pickup_ECM_output")
SIMULATED_PATH = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")
OBSERVED_PATH = os.path.join(PROJECT_DIR, "inputs", "monthly_u5_cases.csv")
OUTPUT_DIR = ANALYZER_DIR

START_YEAR = 2015
END_YEAR = 2025
YEARS = list(range(START_YEAR, END_YEAR + 1))

U5_AGEBINS = (0.25, 5.0)

HABITAT_VARS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]
SEED_VAR = "Run_Number"

MONTHS = list(range(1, 13))
MONTH_NAMES = dict(zip(MONTHS, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))
MONTH_NAME_TO_NUMBER = dict(zip(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"],
    MONTHS,
))

# Peak within this many months of the observed peak counts as correct timing.
PEAK_TIMING_TOLERANCE_MONTHS = 1

# Peak amplitude (peak / annual mean) within this relative difference counts
# as similar strength.
PEAK_STRENGTH_TOLERANCE = 0.15


# HELPERS

def circular_month_difference(sim_month, obs_month):
    """
    Signed difference in months (positive = simulated peak is later),
    wrapping December -> January. E.g. sim=8, obs=10 -> -2; sim=1, obs=12 -> +1.
    """
    diff = sim_month - obs_month
    if diff > 6:
        diff -= 12
    if diff < -6:
        diff += 12
    return diff


def normalize_within_year(series):
    """Divide by the mean: removes overall level and keeps the seasonal shape."""
    mean_value = series.mean()
    if pd.isna(mean_value) or mean_value <= 0:
        return pd.Series(np.nan, index=series.index)
    return series / mean_value


def safe_correlation(x, y):
    """(Pearson, Spearman), or (nan, nan) if there are too few or constant values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pearsonr(x, y).statistic, spearmanr(x, y).statistic


def classify_peak_timing(sim_peak, obs_peak):
    difference = circular_month_difference(sim_peak, obs_peak)
    if abs(difference) <= PEAK_TIMING_TOLERANCE_MONTHS:
        return "CORRECT_TIMING"
    return "TOO_EARLY" if difference < 0 else "TOO_LATE"


def classify_peak_strength(sim_amplitude, obs_amplitude):
    if pd.isna(sim_amplitude) or pd.isna(obs_amplitude):
        return "UNKNOWN"
    relative_difference = (sim_amplitude - obs_amplitude) / obs_amplitude
    if abs(relative_difference) <= PEAK_STRENGTH_TOLERANCE:
        return "SIMILAR_STRENGTH"
    return "TOO_WEAK" if relative_difference < 0 else "TOO_STRONG"


def combo_label(row):
    return (
        f"xTLH={row['x_Temporary_Larval_Habitat']:.3g}, "
        f"constant={row['scale_constant']:g}, "
        f"temp_rain={row['scale_temp_rain']:g}, "
        f"water_veg={row['scale_water_veg']:g}"
    )


def select_combo(df, combo):
    """Rows of `df` matching a habitat combination (a row/Series or values list)."""
    values = [combo[v] for v in HABITAT_VARS] if hasattr(combo, "index") else list(combo)
    mask = np.ones(len(df), dtype=bool)
    for col, value in zip(HABITAT_VARS, values):
        mask &= np.isclose(df[col], value)
    return df[mask].copy()


def compare_profiles(merged, sim_col, obs_col):
    """Correlations, peak months/timing and peak amplitude for two 12-month profiles."""
    pearson, spearman = safe_correlation(merged[sim_col], merged[obs_col])

    sim_peak = int(merged.loc[merged[sim_col].idxmax(), "month"])
    obs_peak = int(merged.loc[merged[obs_col].idxmax(), "month"])
    sim_amplitude = merged[sim_col].max()
    obs_amplitude = merged[obs_col].max()

    return {
        "Pearson": pearson,
        "Spearman": spearman,
        "Observed_peak_month": obs_peak,
        "Simulated_peak_month": sim_peak,
        "Peak_difference_months": circular_month_difference(sim_peak, obs_peak),
        "Timing_class": classify_peak_timing(sim_peak, obs_peak),
        "Observed_peak_amplitude": obs_amplitude,
        "Simulated_peak_amplitude": sim_amplitude,
        "Strength_class": classify_peak_strength(sim_amplitude, obs_amplitude),
    }


def best_combination(clim_results):
    """Combination with the highest climatological Pearson correlation."""
    return clim_results.sort_values("Pearson", ascending=False).iloc[0]


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"\nMissing required {label} columns:\n" + "\n".join(f"  - {c}" for c in missing)
        )


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved:\n{path}")
    return path


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


# LOADING

def load_simulated():
    """Monthly U5 incidence per habitat combination, averaged over seeds."""
    print(f"\nLoading simulated data:\n{SIMULATED_PATH}")
    sim = pd.read_csv(SIMULATED_PATH)

    require_columns(
        sim,
        ["month", "Cases", "Pop","agebin", "Year", SEED_VAR] + HABITAT_VARS,
        "simulated",
    )

    sim = sim[sim["Year"].between(START_YEAR, END_YEAR)]
    sim = sim[sim["agebin"].isin(U5_AGEBINS)].copy()

    sim["month"] = pd.to_numeric(sim["month"], errors="coerce")
    sim["Year"] = pd.to_numeric(sim["Year"], errors="coerce")
    sim["Cases"] = pd.to_numeric(sim["Cases"], errors="coerce")
    sim = sim.dropna(subset=["Year", "month", "Cases", "agebin"])
    sim["Year"] = sim["Year"].astype(int)

    #print(f"Simulated rows after filtering: {len(sim):,}")
    #print(f"Simulation years: {sim['Year'].min()} - {sim['Year'].max()}")
    #print(f"Age bins used: {sorted(sim['agebin'].unique())}")
    #print(f"Seeds detected: {sim[SEED_VAR].nunique()}")
    #print(f"Age bins used: {sorted(sim['agebin'].unique())}")
    #print(f"Seeds detected: {sim[SEED_VAR].nunique()}")

    # Population-weighted U5 incidence: sum(Cases x Pop) / sum(Pop).
    sim["incidence_numerator"] = sim["Cases"] * sim["Pop"]
    u5 = sim.groupby(HABITAT_VARS + [SEED_VAR, "Year", "month"], as_index=False).agg(
        incidence_numerator=("incidence_numerator", "sum"),
        Pop_U5=("Pop", "sum"),
    )
    u5["U5_incidence"] = u5["incidence_numerator"] / u5["Pop_U5"]
    u5 = u5[np.isfinite(u5["U5_incidence"])].copy()
    print(f"U5 monthly simulation rows: {len(u5):,}")

    sim_monthly = u5.groupby(HABITAT_VARS + ["Year", "month"], as_index=False).agg(
        U5_incidence=("U5_incidence", "mean"),
        Pop_U5=("Pop_U5", "mean"),
    )
    print(f"Simulation rows after averaging seeds: {len(sim_monthly):,}")
    return sim_monthly


def load_observed():
    """Observed U5 cases: one row per (Year, month)."""
    print(f"\nLoading observed data:\n{OBSERVED_PATH}")
    obs = pd.read_csv(OBSERVED_PATH)

    require_columns(obs, ["Year", "Month", "Age_Group", "Malaria_Cases"], "observed")

    obs = obs[obs["Age_Group"] == "U5"].copy()
    obs["month"] = obs["Month"].map(MONTH_NAME_TO_NUMBER)
    obs["Year"] = pd.to_numeric(obs["Year"], errors="coerce")
    obs["Cases"] = pd.to_numeric(obs["Malaria_Cases"], errors="coerce")

    obs = obs[obs["Year"].between(START_YEAR, END_YEAR)]
    obs = obs.dropna(subset=["Year", "month", "Cases"])
    obs["Year"] = obs["Year"].astype(int)
    obs["month"] = obs["month"].astype(int)

    obs = obs.groupby(["Year", "month"], as_index=False)["Cases"].sum()

    print(f"Observed monthly rows: {len(obs):,}")
    print(f"Observed years: {obs['Year'].min()} - {obs['Year'].max()}")
    return obs


# ANALYSES

def year_specific_seasonality(sim_monthly, obs):
    """Compare each habitat combination to the observed profile, year by year."""
    print("\nCalculating year-specific seasonality...")
    rows = []

    for combo, sim_combo in sim_monthly.groupby(HABITAT_VARS):
        combo_dict = dict(zip(HABITAT_VARS, combo))

        for year in YEARS:
            sim_year = sim_combo[sim_combo["Year"] == year]
            obs_year = obs[obs["Year"] == year]
            if len(sim_year) < 12 or len(obs_year) < 12:
                continue

            merged = sim_year[["month", "U5_incidence"]].merge(
                obs_year[["month", "Cases"]], on="month", how="inner"
            )
            if len(merged) != 12:
                continue

            merged = merged.sort_values("month")
            merged["Sim_norm"] = normalize_within_year(merged["U5_incidence"])
            merged["Obs_norm"] = normalize_within_year(merged["Cases"])
            merged = merged.dropna(subset=["Sim_norm", "Obs_norm"])
            if len(merged) < 12:
                continue

            rows.append({
                **combo_dict,
                "Year": year,
                **compare_profiles(merged, "Sim_norm", "Obs_norm"),
            })

    results = pd.DataFrame(rows)
    save_csv(results, "UWR_year_specific_seasonality.csv")
    return results


def summarize_combinations(yearly_results):
    """Per-combination summary of the year-specific results."""
    def count(column, label):
        return (column, lambda x: (x == label).sum())

    summary = yearly_results.groupby(HABITAT_VARS, as_index=False).agg(
        Mean_Pearson=("Pearson", "mean"),
        Median_Pearson=("Pearson", "median"),
        Mean_Spearman=("Spearman", "mean"),
        Mean_Peak_Difference=("Peak_difference_months", "mean"),
        Mean_Abs_Peak_Difference=("Peak_difference_months", lambda x: np.mean(np.abs(x))),
        Years_Too_Early=count("Timing_class", "TOO_EARLY"),
        Years_Too_Late=count("Timing_class", "TOO_LATE"),
        Years_Correct_Timing=count("Timing_class", "CORRECT_TIMING"),
        Years_Too_Weak=count("Strength_class", "TOO_WEAK"),
        Years_Too_Strong=count("Strength_class", "TOO_STRONG"),
        Years_Similar_Strength=count("Strength_class", "SIMILAR_STRENGTH"),
        Years_Evaluated=("Year", "nunique"),
    ).sort_values("Mean_Pearson", ascending=False)

    save_csv(summary, "UWR_seasonality_combination_summary.csv")
    print("\nTop 10 habitat combinations by mean year-specific Pearson correlation:")
    print(summary.head(10).to_string(index=False))


def climatology(sim_monthly, obs):
    """
    Normalize each year first, then average by month across years, so years
    with higher burden do not dominate. Returns (obs_clim, sim_clim).
    """
    print(f"\nCalculating {START_YEAR}-{END_YEAR} climatological seasonality...")

    obs_clim = obs.copy()
    obs_clim["Obs_norm"] = obs_clim.groupby("Year")["Cases"].transform(normalize_within_year)
    obs_clim = (
        obs_clim.groupby("month", as_index=False)["Obs_norm"].mean()
        .rename(columns={"Obs_norm": "Observed_seasonal_index"})
    )

    sim_clim = sim_monthly.copy()
    sim_clim["Sim_norm"] = (
        sim_clim.groupby(HABITAT_VARS + ["Year"])["U5_incidence"]
        .transform(normalize_within_year)
    )
    sim_clim = (
        sim_clim.groupby(HABITAT_VARS + ["month"], as_index=False)["Sim_norm"].mean()
        .rename(columns={"Sim_norm": "Simulated_seasonal_index"})
    )
    return obs_clim, sim_clim


def compare_climatology(sim_clim, obs_clim):
    rows = []
    for combo, sim_combo in sim_clim.groupby(HABITAT_VARS):
        merged = sim_combo[["month", "Simulated_seasonal_index"]].merge(
            obs_clim[["month", "Observed_seasonal_index"]], on="month", how="inner"
        )
        if len(merged) != 12:
            continue

        merged = merged.sort_values("month")
        rows.append({
            **dict(zip(HABITAT_VARS, combo)),
            **compare_profiles(merged, "Simulated_seasonal_index", "Observed_seasonal_index"),
        })

    results = pd.DataFrame(rows)
    save_csv(results, "UWR_climatological_seasonality.csv")
    return results


def save_diagnostics(clim_results):
    """Concise per-combination interpretation (a diagnostic, not a score)."""
    diagnostics = clim_results.copy()
    diagnostics["Abs_peak_timing_error"] = diagnostics["Peak_difference_months"].abs()
    diagnostics["Timing_direction"] = np.where(
        diagnostics["Peak_difference_months"] < 0, "EARLY",
        np.where(diagnostics["Peak_difference_months"] > 0, "LATE", "ON_TIME"),
    )
    save_csv(diagnostics, "UWR_seasonality_diagnostics.csv")


# PLOTS (all use the best climatological combination, for visualization only)

def plot_yearly(sim_monthly, obs, combo):
    sim_plot = select_combo(sim_monthly, combo)

    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharey=False)
    axes = axes.flatten()

    for i, year in enumerate(YEARS):
        ax = axes[i]
        sim_year = sim_plot[sim_plot["Year"] == year].copy()
        obs_year = obs[obs["Year"] == year].copy()

        if len(sim_year) < 12 or len(obs_year) < 12:
            ax.set_title(f"{year} - insufficient data")
            continue

        sim_year["norm"] = normalize_within_year(sim_year["U5_incidence"])
        obs_year["norm"] = normalize_within_year(obs_year["Cases"])
        sim_year = sim_year.sort_values("month")
        obs_year = obs_year.sort_values("month")

        ax.plot(sim_year["month"], sim_year["norm"], marker="o", label="EMOD")
        ax.plot(obs_year["month"], obs_year["norm"], marker="o", linestyle="--", label="Observed")

        ax.set_title(str(year))
        ax.set_xticks(MONTHS)
        ax.set_xticklabels([MONTH_NAMES[m] for m in MONTHS], rotation=45)
        ax.set_ylabel("Seasonal index")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()

    for ax in axes[len(YEARS):]:  # unused panels
        ax.axis("off")

    fig.suptitle("UWR EMOD vs Observed Seasonality by Year\n" + combo_label(combo), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "UWR_seasonality_yearly.png")


def plot_climatology(sim_clim, obs_clim, combo):
    merged = (
        select_combo(sim_clim, combo)[["month", "Simulated_seasonal_index"]]
        .merge(obs_clim[["month", "Observed_seasonal_index"]], on="month", how="inner")
        .sort_values("month")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(merged["month"], merged["Observed_seasonal_index"],
            marker="o", linewidth=2, label="Observed")
    ax.plot(merged["month"], merged["Simulated_seasonal_index"],
            marker="o", linewidth=2, linestyle="--", label="EMOD")

    ax.set_xticks(MONTHS)
    ax.set_xticklabels([MONTH_NAMES[m] for m in MONTHS])
    ax.set_xlabel("Month")
    ax.set_ylabel("Seasonal index\n(monthly value / annual mean)")
    ax.set_title(f"UWR {START_YEAR}-{END_YEAR} Climatological Seasonality\n" + combo_label(combo))
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "UWR_seasonality_climatology.png")


def annual_peak_months(df, value_col):
    """Peak month per year, for years with all 12 months."""
    records = []
    for year in YEARS:
        year_df = df[df["Year"] == year]
        if len(year_df) == 12:
            peak_month = int(year_df.loc[year_df[value_col].idxmax(), "month"])
            records.append({"Year": year, "Peak_month": peak_month})
    return pd.DataFrame(records)


def plot_peak_timing(sim_monthly, obs, combo):
    observed_peaks = annual_peak_months(obs, "Cases")
    sim_peaks = annual_peak_months(select_combo(sim_monthly, combo), "U5_incidence")

    fig, ax = plt.subplots(figsize=(12, 6))

    if len(observed_peaks) > 0:
        ax.plot(observed_peaks["Year"], observed_peaks["Peak_month"],
                marker="o", linewidth=2, label="Observed")
    if len(sim_peaks) > 0:
        ax.plot(sim_peaks["Year"], sim_peaks["Peak_month"],
                marker="o", linewidth=2, linestyle="--", label="EMOD")

    ax.set_yticks(MONTHS)
    ax.set_yticklabels([MONTH_NAMES[m] for m in MONTHS])
    ax.set_xlabel("Year")
    ax.set_ylabel("Peak month")
    ax.set_title(f"UWR Annual Malaria Peak Timing, {START_YEAR}-{END_YEAR}")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "UWR_peak_timing.png")


# REPORTING

def print_summary(best, yearly_results):
    #print("\n" + "=" * 80)
    #print("SEASONALITY DIAGNOSTIC SUMMARY")
    #print("=" * 80)

    print("\nVisualization combination:")
    print(combo_label(best))

    print(f"\nClimatological Pearson correlation: {best['Pearson']:.3f}")
    print(f"Climatological Spearman correlation: {best['Spearman']:.3f}")

    print(f"\nObserved climatological peak: {MONTH_NAMES[int(best['Observed_peak_month'])]}")
    print(f"Simulated climatological peak: {MONTH_NAMES[int(best['Simulated_peak_month'])]}")
    print(f"Peak timing difference: {best['Peak_difference_months']:+.0f} month(s)")
    print(f"Timing diagnosis: {best['Timing_class']}")

    print(f"\nObserved peak amplitude: {best['Observed_peak_amplitude']:.3f}")
    print(f"Simulated peak amplitude: {best['Simulated_peak_amplitude']:.3f}")
    print(f"Peak strength diagnosis: {best['Strength_class']}")

    yearly_best = select_combo(yearly_results, best)
    if len(yearly_best) > 0:
        print("\nYear-by-year seasonality:")
        display = yearly_best[[
            "Year", "Pearson", "Spearman",
            "Observed_peak_month", "Simulated_peak_month",
            "Peak_difference_months", "Timing_class",
            "Observed_peak_amplitude", "Simulated_peak_amplitude", "Strength_class",
        ]].copy()
        display["Observed_peak_month"] = display["Observed_peak_month"].map(MONTH_NAMES)
        display["Simulated_peak_month"] = display["Simulated_peak_month"].map(MONTH_NAMES)
        print(display.to_string(index=False))


def print_outputs():
    #print("\n" + "=" * 80)
    print("DONE")
    #print("=" * 80)
    print(f"\nOutput directory:\n{OUTPUT_DIR}")
    print("\nFiles produced:")
    for filename in [
        "UWR_year_specific_seasonality.csv",
        "UWR_seasonality_combination_summary.csv",
        "UWR_climatological_seasonality.csv",
        "UWR_seasonality_diagnostics.csv",
        "UWR_seasonality_yearly.png",
        "UWR_seasonality_climatology.png",
        "UWR_peak_timing.png",
    ]:
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            print(f"  ✓ {filename}")


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #print("\n" + "=" * 80)
    print("UWR EMOD SEASONALITY DIAGNOSTIC")
    #print("=" * 80)
    #print("=" * 80)

    sim_monthly = load_simulated()
    obs = load_observed()

    yearly_results = year_specific_seasonality(sim_monthly, obs)
    if len(yearly_results) > 0:
        summarize_combinations(yearly_results)

    obs_clim, sim_clim = climatology(sim_monthly, obs)
    clim_results = compare_climatology(sim_clim, obs_clim)

    if len(clim_results) == 0:

        # print("\n" + "=" * 80)
        # print("SEASONALITY DIAGNOSTIC SUMMARY")
        # print("=" * 80)
        print("\nNo complete habitat combinations were available "
               "for climatological analysis.")
    else:
        save_diagnostics(clim_results)

        best = best_combination(clim_results)
        plot_yearly(sim_monthly, obs, best)
        plot_climatology(sim_clim, obs_clim, best)
        plot_peak_timing(sim_monthly, obs, best)

        print_summary(best, yearly_results)

    print_outputs()


if __name__ == "__main__":
    main()