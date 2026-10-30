import argparse
import os
import calendar
import numpy as np
import pandas as pd
from scipy.stats import t

# ============================================================
# DEFAULT SETTINGS
# ============================================================

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"

ANALYZER_DIR = os.path.join(
    PROJECT_DIR,
    "experiments",
    "XTLH_Pickup13yrs_AnalyzerOutput"
)

DEFAULT_SIM = os.path.join(
    ANALYZER_DIR,
    "PfPR_ClinicalIncidence_monthly_pickup.csv"
)

PARAMS = [
    "x_Temporary_Larval_Habitat",
    "scale_constant",
    "scale_temp_rain",
    "scale_water_veg",
]

SEED_COL = "sim_id"

START_YEAR = 2015
END_YEAR = 2025
U5_MAX_AGE = 5.0

BLOCK_DAYS = 30
MONTH_STARTS = np.array(
    [0, 31, 59, 90, 120, 151,
     181, 212, 243, 273, 304, 334]
)

MONTH_NAMES = list(calendar.month_abbr)[1:]


# ============================================================
# HELPERS
# ============================================================

def block_to_month(block):
    midpoint = BLOCK_DAYS * (block - 0.5)
    return np.searchsorted(
        MONTH_STARTS, midpoint, side="right"
    )


def normalize(x):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)

    if not np.isfinite(mean) or mean <= 0:
        return np.full_like(x, np.nan)

    return x / mean


def required_n_for_precision(sd, half_width, confidence=0.95,
                             max_n=10000):
    """
    Find the minimum sample size n such that:

        t_critical(n-1) * SD / sqrt(n) <= half_width

    SD is estimated from the existing independent simulation runs.
    """

    if not np.isfinite(sd) or sd < 0:
        return np.nan

    if half_width <= 0:
        raise ValueError("half_width must be positive.")

    if sd == 0:
        return 2

    alpha = 1.0 - confidence

    for n in range(2, max_n + 1):
        critical = t.ppf(1 - alpha / 2, df=n - 1)
        margin = critical * sd / np.sqrt(n)

        if margin <= half_width:
            return n

    return np.nan


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = [
        "Year", "report_interval", "Cases", "Pop",
        "agebin", SEED_COL, *PARAMS
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep complete reporting intervals only.
    if "is_complete_interval" in df.columns:
        complete = (
            df["is_complete_interval"]
            .astype(str).str.lower().eq("true")
        )
        df = df.loc[complete].copy()

    numeric = [
        "Year", "report_interval", "Cases", "Pop",
        "agebin", *PARAMS
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=numeric + [SEED_COL]
    )

    df = df[
        df["Year"].between(START_YEAR, END_YEAR)
        & (df["report_interval"].between(1, 12))
        & (df["agebin"] <= U5_MAX_AGE)
        & (df["Pop"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No eligible under-five records found.")

    # Report interval midpoint -> approximate calendar month.
    df["month"] = block_to_month(
        df["report_interval"].to_numpy()
    )

    if not df["month"].between(1, 12).all():
        raise ValueError("Invalid month mapping.")

    # Cases is assumed to be an incidence rate.
    # Population-weight the age-specific rates.
    df["weighted_cases"] = df["Cases"] * df["Pop"]

    run_month = (
        df.groupby(
            PARAMS + [SEED_COL, "Year", "month"],
            as_index=False
        )
        .agg(
            weighted_cases=("weighted_cases", "sum"),
            population=("Pop", "sum"),
        )
    )

    run_month["incidence"] = (
        run_month["weighted_cases"]
        / run_month["population"]
    )

    # Keep only years with all 12 mapped months per run.
    year_counts = run_month.groupby(
        PARAMS + [SEED_COL, "Year"]
    )["month"].nunique()

    complete_keys = year_counts[year_counts == 12].index

    key_frame = complete_keys.to_frame(index=False)

    run_month = run_month.merge(
        key_frame,
        on=PARAMS + [SEED_COL, "Year"],
        how="inner"
    )

    if run_month.empty:
        raise ValueError(
            "No complete simulation years remain. "
            "Check the interval-to-month mapping."
        )

    # Normalize each run-year to focus on seasonal shape.
    run_month["seasonal_index"] = (
        run_month.groupby(
            PARAMS + [SEED_COL, "Year"]
        )["incidence"]
        .transform(normalize)
    )

    # Average each run's normalized profiles across years.
    run_climatology = (
        run_month.groupby(
            PARAMS + [SEED_COL, "month"],
            as_index=False
        )
        .agg(
            seasonal_index=("seasonal_index", "mean"),
            mean_incidence=("incidence", "mean"),
        )
    )

    return run_climatology


# ============================================================
# ESTIMATE REQUIRED RUNS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the number of independent EMOD runs "
            "needed for seasonal-profile precision."
        )
    )

    parser.add_argument("--sim", default=DEFAULT_SIM)

    parser.add_argument("--xtlh", type=float)
    parser.add_argument("--scale-constant", type=float)
    parser.add_argument("--scale-temp-rain", type=float)
    parser.add_argument("--scale-water-veg", type=float)

    parser.add_argument(
        "--half-width", type=float, default=0.10,
        help="Target 95%% CI half-width for seasonal index."
    )

    parser.add_argument(
        "--confidence", type=float, default=0.95
    )

    parser.add_argument(
        "--out", default=os.path.join(
            ANALYZER_DIR, "simulation_replicate_estimates.csv"
        )
    )

    args = parser.parse_args()

    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1.")

    df = load_data(args.sim)

    supplied = [
        args.xtlh,
        args.scale_constant,
        args.scale_temp_rain,
        args.scale_water_veg,
    ]

    # Display available parameter combinations.
    combinations = (
        df[PARAMS + [SEED_COL]]
        .drop_duplicates()
        .groupby(PARAMS, dropna=False)[SEED_COL]
        .nunique()
        .reset_index(name="independent_runs")
        .sort_values("independent_runs", ascending=False)
    )

    print("\nAvailable parameter combinations:")
    print(combinations.to_string(index=False))

    if any(v is not None for v in supplied):
        if not all(v is not None for v in supplied):
            raise ValueError(
                "Supply all four parameters together: "
                "--xtlh, --scale-constant, "
                "--scale-temp-rain, --scale-water-veg."
            )

        target = dict(zip(PARAMS, supplied))
        mask = np.ones(len(df), dtype=bool)

        for p, value in target.items():
            mask &= np.isclose(
                df[p].astype(float),
                value,
                rtol=1e-6,
                atol=1e-8,
            )

        selected = df.loc[mask].copy()

    else:
        # Only auto-select when the data contain one combination.
        if len(combinations) != 1:
            print(
                "\nSTOP: multiple parameter combinations exist. "
                "Specify the four parameter values to avoid "
                "mistaking parameter differences for run variability."
            )
            return

        target = combinations.iloc[0][PARAMS].to_dict()
        selected = df.copy()

    print("\nSelected parameters:")
    for p, value in target.items():
        print(f"  {p}: {value}")

    n_runs = selected[SEED_COL].nunique()
    print(f"\nIndependent simulation runs available: {n_runs}")

    if n_runs < 2:
        print(
            "\nCANNOT ESTIMATE REQUIRED RUNS: this parameter "
            "combination has fewer than two independent runs. "
            "Run additional stochastic replicates at the same "
            "parameter values before estimating between-run variance."
        )
        return

    # Each independent run must contribute all 12 months.
    month_counts = selected.groupby(SEED_COL)["month"].nunique()

    if not (month_counts == 12).all():
        raise ValueError(
            "Not all runs have all 12 months. Check output completeness."
        )

    rows = []

    for month in range(1, 13):
        g = selected.loc[
            selected["month"] == month, "seasonal_index"
        ].dropna()

        n = len(g)
        mean = g.mean()
        sd = g.std(ddof=1)

        if n < 2:
            required_n = np.nan
            margin = np.nan
        else:
            critical = __import__("scipy").stats.t.ppf(
                (1 + args.confidence) / 2, n - 1
            )
            margin = critical * sd / np.sqrt(n)

            required_n = required_n_for_precision(
                sd,
                args.half_width,
                confidence=args.confidence,
            )

        rows.append({
            "month": month,
            "month_name": MONTH_NAMES[month - 1],
            "runs_available": n,
            "mean_seasonal_index": mean,
            "sd_between_runs": sd,
            "current_CI_half_width": margin,
            "target_CI_half_width": args.half_width,
            "required_runs": required_n,
            "additional_runs": (
                max(0, int(required_n) - n)
                if pd.notna(required_n) else np.nan
            ),
        })

    result = pd.DataFrame(rows)

    print("\nMonthly precision assessment:")
    print(result.round(4).to_string(index=False))

    valid = result["required_runs"].dropna()

    if valid.empty:
        print("\nNo valid sample-size estimates.")
        return

    required = int(valid.max())

    print("\n" + "=" * 60)
    print("SIMULATION REPLICATE ESTIMATE")
    print("=" * 60)
    print(f"Runs currently available: {n_runs}")
    print(f"Target confidence level: {args.confidence:.0%}")
    print(f"Target CI half-width: +/- {args.half_width:.3f}")
    print(f"Estimated total runs needed: {required}")
    print(f"Additional runs indicated: {max(0, required - n_runs)}")
    print(
        "\nThis estimate uses observed between-run variability "
        "and a t-based confidence interval. It is a planning "
        "estimate, not a guarantee of precision."
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\nSaved monthly estimates: {args.out}")


if __name__ == "__main__":
    main()
