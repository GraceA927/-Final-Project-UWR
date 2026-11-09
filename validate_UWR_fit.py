"""
UWR model validation: PfPR, clinical incidence, seasonality and fit statistics.

This is separate from the calibration (Prevalence_xLTH_calibration.py is not
changed). It reads the same analyzer output and the calibration's
calibration_scores_param_sets.csv, then:

1. Scores every parameter set (xTLH + habitat mix, mean over seeds) on
   (a) DHS PfPR log-likelihood (from the calibration) and
   (b) monthly U5 clinical incidence vs monthly_U5_conf.csv.
   Simulated incidence is all clinical cases; observed is confirmed cases at
   facilities. They are linked by one reporting ratio per parameter set,
       rho = sum(observed) / sum(simulated)        (2015-2025, monthly)
   i.e. observed-equivalent incidence = rho * simulated incidence.
   rho is the fraction of clinical cases that are treated, tested and
   reported. Apply the same rho to scenario outputs.
2. Selects an ensemble: the ENSEMBLE_N sets with the best joint rank
   (PfPR rank + incidence rank). Ensemble = those sets x all seeds.
3. Plots (ensemble median and 95% interval across runs):
     fig1_pfpr_validation.png        observed vs simulated PfPR
     fig2_incidence_validation.png   observed vs simulated clinical incidence
     fig3_seasonal_incidence.png     seasonal incidence comparison
4. Fit statistics: computed for every ensemble run, reported as
   median [2.5%, 97.5%] across runs -> fit_statistics.csv.
5. Writes validated_parameter_sets.csv (also to inputs/) with rho, for
   PARAMETER_SET_FILE in run_scenariosIRS_UWR.py.

Usage: python validate_UWR_fit.py [EXPERIMENT_NAME]
"""

import os
import sys

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns
except ImportError:
    sns = None


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
INPUT_DIR = os.path.join(PROJECT_DIR, "inputs")
DHS_FILE = "pcr_prevalence.csv"
INC_FILE = "monthly_U5_conf.csv"

try:
    import manifest
    JOB_DIR = manifest.job_directory
except (ImportError, AttributeError):
    JOB_DIR = os.path.join(PROJECT_DIR, "experiments")

EXPERIMENT_NAME = sys.argv[1] if len(sys.argv) > 1 else "Pickup13yrs_season"
ANALYZER_DIR = os.path.join(JOB_DIR, f"{EXPERIMENT_NAME}_AnalyzerOutput")
SIM_CSV = os.path.join(ANALYZER_DIR, "PfPR_ClinicalIncidence_monthly_pickup.csv")
CALIB_SETS_CSV = os.path.join(ANALYZER_DIR, "calibration_scores_param_sets.csv")
OUT_DIR = os.path.join(ANALYZER_DIR, "validation")
PARAM_SET_OUT = "validated_parameter_sets.csv"   # also copied to inputs/

PICKUP_START_YEAR = 2013
PICKUP_END_YEAR = 2025
BLOCK_DAYS = 30
N_BLOCKS = 12
DAYS_PER_YEAR = 365
MONTH_START = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151, 7: 181,
               8: 212, 9: 243, 10: 273, 11: 304, 12: 334}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]
MONTH_ABBR = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

PFPR_AGEBINS = (5.0,)              # 0.5-5 y = DHS 6-59 months
INC_AGEBINS = (0.25, 0.5, 5.0)     # all under-5 bins
OBS_CASES_COL = "adj1_conf_u5"     # reporting-adjusted confirmed U5 cases
OBS_POP_COL = "Pop_5"

ENSEMBLE_N = 5                     # parameter sets carried forward
INTERVAL = (2.5, 97.5)

XCOL = "x_Temporary_Larval_Habitat"
HAB_COLS = ["scale_constant", "scale_temp_rain", "scale_water_veg"]
SET_KEYS = [XCOL] + HAB_COLS
RUN_KEYS = SET_KEYS + ["Run_Number"]

C_SIM = "firebrick"
C_OBS = "black"
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "savefig.facecolor": "white",
}
SAVE_DPI = 600


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_true(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes", "y"])


def block_date(year, block):
    """Mid-point of a 30-day model block as a datetime (flat calendar)."""
    return (pd.to_datetime(year.astype(int).astype(str) + "-01-01")
            + pd.to_timedelta(BLOCK_DAYS * (block.astype(float) - 0.5), unit="D"))


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def interval(x):
    return np.nanpercentile(x, INTERVAL[0]), np.nanpercentile(x, INTERVAL[1])


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def style():
    if sns is not None:
        sns.set_style("whitegrid", {"axes.linewidth": 0.5})


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_sim():
    """Per run x year x block: U5 PfPR (6-59 m) and U5 incidence per 1000."""
    cols = RUN_KEYS + ["Year", "report_interval", "is_complete_interval",
                       "agebin", "PfPR", "Cases", "Pop"]
    df = pd.read_csv(SIM_CSV, usecols=lambda c: c in cols)
    df = df.rename(columns={"Year": "year", "report_interval": "block"})
    df = df[df["block"].between(1, N_BLOCKS)
            & df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)]
    if "is_complete_interval" in df.columns:
        df = df[is_true(df["is_complete_interval"])]
    for c in ["PfPR", "Cases", "Pop", "agebin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Pop"] > 0]
    key = RUN_KEYS + ["year", "block"]

    pf = df[np.isin(df["agebin"].round(6), PFPR_AGEBINS)].copy()
    pf["pos"] = pf["PfPR"] * pf["Pop"]
    pf = pf.groupby(key, as_index=False)[["pos", "Pop"]].sum()
    pf["sim_pfpr"] = pf["pos"] / pf["Pop"]

    # Cases = annualised clinical incidence per person -> cases per 30 days.
    inc = df[np.isin(df["agebin"].round(6), INC_AGEBINS)].copy()
    inc["cases"] = inc["Cases"] * inc["Pop"] * BLOCK_DAYS / DAYS_PER_YEAR
    inc = inc.groupby(key, as_index=False)[["cases", "Pop"]].sum()
    inc["sim_inc"] = inc["cases"] / inc["Pop"] * 1000

    return pf[key + ["sim_pfpr"]].merge(inc[key + ["sim_inc"]], on=key)


def load_dhs():
    df = pd.read_csv(os.path.join(INPUT_DIR, DHS_FILE), sep=None, engine="python")
    for c in ["fieldwork_start", "fieldwork_end"]:
        df[c] = pd.to_datetime(df[c])
    mid = df["fieldwork_start"] + (df["fieldwork_end"] - df["fieldwork_start"]) / 2
    df["year"] = mid.dt.year
    df["model_day"] = mid.dt.month.map(MONTH_START) + mid.dt.day - 1
    df["block"] = df["model_day"] // BLOCK_DAYS + 1
    df = df[df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)
            & df["block"].le(N_BLOCKS)].copy()
    df["obs_pfpr"] = df["DHS_pos"] / df["DHS_n"]
    ci = [clopper_pearson(k, n) for k, n in zip(df["DHS_pos"], df["DHS_n"])]
    df["obs_lo"], df["obs_hi"] = zip(*ci)
    df["date"] = (pd.to_datetime(df["year"].astype(str) + "-01-01")
                  + pd.to_timedelta(df["model_day"], unit="D"))
    return df.reset_index(drop=True)


def load_obs_incidence():
    df = pd.read_csv(os.path.join(INPUT_DIR, INC_FILE))
    df["block"] = df["month"].map({m: i + 1 for i, m in enumerate(MONTH_NAMES)})
    df["obs_inc"] = df[OBS_CASES_COL] / df[OBS_POP_COL] * 1000
    df = df[df["year"].between(PICKUP_START_YEAR, PICKUP_END_YEAR)]
    return df[["year", "block", "obs_inc"]].reset_index(drop=True)


# ---------------------------------------------------------------------
# Scoring and ensemble selection
# ---------------------------------------------------------------------

def score_sets(sim, obs_inc):
    """Per parameter set (mean over seeds): rho and incidence fit."""
    m = (sim.groupby(SET_KEYS + ["year", "block"], as_index=False)["sim_inc"].mean()
         .merge(obs_inc, on=["year", "block"]))
    rows = []
    for keys, g in m.groupby(SET_KEYS):
        rho = g["obs_inc"].sum() / g["sim_inc"].sum()
        fit = rho * g["sim_inc"]
        rows.append((*keys, rho,
                     np.sqrt(np.mean((np.log1p(g["obs_inc"]) - np.log1p(fit)) ** 2)),
                     g["obs_inc"].corr(fit)))
    out = pd.DataFrame(rows, columns=SET_KEYS + ["rho", "inc_log_rmse", "inc_r"])

    if os.path.isfile(CALIB_SETS_CSV):
        cal = pd.read_csv(CALIB_SETS_CSV)
        llcol = "ll" if "ll" in cal.columns else "ll_dhs"
        cal = cal[SET_KEYS + [llcol]].rename(columns={llcol: "ll_pfpr"})
        out[SET_KEYS] = out[SET_KEYS].round(6)
        cal[SET_KEYS] = cal[SET_KEYS].round(6)
        out = out.merge(cal, on=SET_KEYS, how="left")
    else:
        print(f"WARNING: {CALIB_SETS_CSV} not found; ranking on incidence only.")
        out["ll_pfpr"] = np.nan

    out["rank_pfpr"] = out["ll_pfpr"].rank(ascending=False, method="min")
    out["rank_inc"] = out["inc_log_rmse"].rank(ascending=True, method="min")
    out["joint_rank"] = out["rank_pfpr"].fillna(0) + out["rank_inc"]
    return out.sort_values(["joint_rank", "rank_inc"]).reset_index(drop=True)


def select_ensemble(sim, sets):
    chosen = sets.head(ENSEMBLE_N)
    mask = np.zeros(len(sim), dtype=bool)
    rho = pd.Series(np.nan, index=sim.index)
    for _, r in chosen.iterrows():
        mm = np.ones(len(sim), dtype=bool)
        for c in SET_KEYS:
            mm &= np.isclose(sim[c], r[c])
        mask |= mm
        rho[mm] = r["rho"]
    ens = sim[mask].copy()
    ens["rho"] = rho[mask]
    ens["sim_inc_obs"] = ens["sim_inc"] * ens["rho"]     # observed-equivalent
    return ens


def band(ens, value, by):
    g = ens.groupby(by)[value]
    out = g.median().rename("med").to_frame()
    out["lo"] = g.quantile(INTERVAL[0] / 100)
    out["hi"] = g.quantile(INTERVAL[1] / 100)
    return out.reset_index()


# ---------------------------------------------------------------------
# Fit statistics (per ensemble run, then median and interval)
# ---------------------------------------------------------------------

def nse(o, s):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def best_lag(o_prof, s_prof):
    """Months by which the simulated seasonal profile lags (+) observed."""
    o = o_prof - o_prof.mean()
    s = s_prof - s_prof.mean()
    cc = [np.dot(o, np.roll(s, -k)) for k in range(N_BLOCKS)]
    lag = int(np.argmax(cc))
    return lag if lag <= 6 else lag - N_BLOCKS


def fit_statistics(ens, dhs, obs_inc):
    per_run = []
    for keys, run in ens.groupby(RUN_KEYS):
        st = {}
        p = dhs.merge(run, on=["year", "block"])
        d = p["sim_pfpr"] - p["obs_pfpr"]
        st["PfPR bias (sim - obs)"] = d.mean()
        st["PfPR MAE"] = d.abs().mean()
        st["PfPR RMSE"] = np.sqrt((d ** 2).mean())
        st["PfPR surveys inside observed 95% CI (n)"] = (
            (p["sim_pfpr"] >= p["obs_lo"]) & (p["sim_pfpr"] <= p["obs_hi"])).sum()

        m = obs_inc.merge(run, on=["year", "block"])
        o, s = m["obs_inc"].to_numpy(), m["sim_inc_obs"].to_numpy()
        st["Monthly incidence Pearson r"] = np.corrcoef(o, s)[0, 1]
        st["Monthly incidence R2 (r^2)"] = st["Monthly incidence Pearson r"] ** 2
        st["Monthly incidence NSE"] = nse(o, s)
        st["Monthly incidence RMSE (per 1000)"] = np.sqrt(np.mean((s - o) ** 2))
        st["Monthly incidence MAE (per 1000)"] = np.mean(np.abs(s - o))
        st["Monthly incidence MAPE (%)"] = 100 * np.mean(np.abs(s - o) / o)

        a = m.groupby("year")[["obs_inc", "sim_inc_obs"]].sum()
        st["Annual incidence Pearson r"] = a["obs_inc"].corr(a["sim_inc_obs"])
        st["Annual incidence bias (%)"] = 100 * (a["sim_inc_obs"].sum() / a["obs_inc"].sum() - 1)
        st["Annual incidence MAPE (%)"] = 100 * np.mean(
            np.abs(a["sim_inc_obs"] - a["obs_inc"]) / a["obs_inc"])

        m["obs_sh"] = m["obs_inc"] / m.groupby("year")["obs_inc"].transform("sum")
        m["sim_sh"] = m["sim_inc_obs"] / m.groupby("year")["sim_inc_obs"].transform("sum")
        op = m.groupby("block")["obs_sh"].mean().to_numpy()
        sp = m.groupby("block")["sim_sh"].mean().to_numpy()
        st["Seasonal profile correlation"] = np.corrcoef(op, sp)[0, 1]
        st["Seasonal peak month, observed"] = int(np.argmax(op) + 1)
        st["Seasonal peak month, simulated"] = int(np.argmax(sp) + 1)
        st["Seasonal phase lag (months, + = sim late)"] = best_lag(op, sp)
        st["Reporting ratio rho"] = run["rho"].iloc[0]
        per_run.append(st)

    pr = pd.DataFrame(per_run)
    rows = []
    for c in pr.columns:
        lo, hi = interval(pr[c])
        rows.append({"statistic": c, "median": pr[c].median(),
                     "lo_2.5": lo, "hi_97.5": hi, "n_runs": pr[c].notna().sum()})
    summary = pd.DataFrame(rows)

    # Coverage: share of observations inside the ensemble 95% interval.
    pb = band(ens, "sim_pfpr", ["year", "block"]).merge(dhs, on=["year", "block"])
    ib = band(ens, "sim_inc_obs", ["year", "block"]).merge(obs_inc, on=["year", "block"])
    cov = [
        {"statistic": "PfPR coverage: DHS points inside ensemble 95% interval (%)",
         "median": 100 * np.mean((pb["obs_pfpr"] >= pb["lo"]) & (pb["obs_pfpr"] <= pb["hi"]))},
        {"statistic": "PfPR coverage: DHS 95% CI overlaps ensemble 95% interval (%)",
         "median": 100 * np.mean((pb["obs_hi"] >= pb["lo"]) & (pb["obs_lo"] <= pb["hi"]))},
        {"statistic": "Incidence coverage: observed months inside ensemble 95% interval (%)",
         "median": 100 * np.mean((ib["obs_inc"] >= ib["lo"]) & (ib["obs_inc"] <= ib["hi"]))},
    ]
    return pd.concat([summary, pd.DataFrame(cov)], ignore_index=True), pb, ib


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def fig_pfpr(ens, dhs, pb, path):
    b = band(ens, "sim_pfpr", ["year", "block"])
    b["date"] = block_date(b["year"], b["block"])
    b = b.sort_values("date")
    with mpl.rc_context(PUB_RC):
        style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                       gridspec_kw={"width_ratios": [2.2, 1]})
        ax1.fill_between(b["date"], b["lo"], b["hi"], color=C_SIM, alpha=0.18,
                         linewidth=0, label="Simulated 95% interval")
        ax1.plot(b["date"], b["med"], color=C_SIM, linewidth=1.5,
                 label="Simulated median")
        ax1.errorbar(dhs["date"], dhs["obs_pfpr"],
                     yerr=[dhs["obs_pfpr"] - dhs["obs_lo"], dhs["obs_hi"] - dhs["obs_pfpr"]],
                     fmt="o", color=C_OBS, markersize=4, elinewidth=0.8, capsize=2,
                     zorder=5, label="Observed DHS (95% CI)")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("PfPR (children 6–59 months)")
        ax1.set_title("(a) PfPR over time", loc="left")
        ax1.legend(frameon=False, loc="upper right")
        clean_axes(ax1)

        lim = max(pb["hi"].max(), pb["obs_hi"].max()) * 1.1
        ax2.plot([0, lim], [0, lim], color="0.6", linewidth=0.8, linestyle="--",
                 label="1:1 line")
        ax2.errorbar(pb["obs_pfpr"], pb["med"],
                     xerr=[pb["obs_pfpr"] - pb["obs_lo"], pb["obs_hi"] - pb["obs_pfpr"]],
                     yerr=[pb["med"] - pb["lo"], pb["hi"] - pb["med"]],
                     fmt="o", color=C_SIM, ecolor="0.5", markersize=4,
                     elinewidth=0.7, capsize=2)
        for _, r in pb.iterrows():
            ax2.annotate(str(int(r["year"])), (r["obs_pfpr"], r["med"]),
                         textcoords="offset points", xytext=(5, -9), fontsize=7,
                         color="0.3")
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        ax2.set_xlabel("Observed PfPR")
        ax2.set_ylabel("Simulated PfPR")
        ax2.set_title("(b) Observed vs simulated", loc="left")
        ax2.legend(frameon=False, loc="upper left")
        clean_axes(ax2)
        fig.tight_layout()
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {path}")


def fig_incidence(ens, obs_inc, ib, path):
    ib = ib.copy()
    ib["date"] = block_date(ib["year"], ib["block"])
    ib = ib.sort_values("date")
    ann_runs = (ens.merge(obs_inc[["year", "block"]], on=["year", "block"])
                .groupby(RUN_KEYS + ["year"], as_index=False)["sim_inc_obs"].sum())
    ann = band(ann_runs, "sim_inc_obs", ["year"]).merge(
        obs_inc.groupby("year", as_index=False)["obs_inc"].sum(), on="year")
    rho = ens.groupby(SET_KEYS)["rho"].first()

    with mpl.rc_context(PUB_RC):
        style()
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.fill_between(ib["date"], ib["lo"], ib["hi"], color=C_SIM, alpha=0.18,
                         linewidth=0, label="Simulated 95% interval")
        ax1.plot(ib["date"], ib["med"], color=C_SIM, linewidth=1.5,
                 label="Simulated median (× reporting ratio)")
        ax1.plot(ib["date"], ib["obs_inc"], "-o", color=C_OBS, linewidth=1.0,
                 markersize=2.5, label="Observed confirmed U5")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Cases per 1000 U5 per month")
        ax1.set_title(f"(a) Monthly U5 clinical incidence "
                      f"(reporting ratio ρ = {rho.min():.2f}–{rho.max():.2f})",
                      loc="left")
        ax1.legend(frameon=False, loc="upper left", ncol=3)
        clean_axes(ax1)

        x = np.arange(len(ann))
        ax2.bar(x - 0.2, ann["obs_inc"], width=0.38, color="0.25",
                label="Observed")
        ax2.bar(x + 0.2, ann["med"], width=0.38, color=C_SIM,
                yerr=[ann["med"] - ann["lo"], ann["hi"] - ann["med"]],
                error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "0.3"},
                label="Simulated (95% interval)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"'{str(y)[2:]}" for y in ann["year"]])
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Cases per 1000 U5 per year")
        ax2.set_title("(b) Annual incidence", loc="left")
        ax2.set_ylim(0, max(ann["obs_inc"].max(), ann["hi"].max()) * 1.25)
        ax2.legend(frameon=False, loc="upper left", ncol=2)
        clean_axes(ax2)

        lim = max(ib["obs_inc"].max(), ib["hi"].max()) * 1.05
        ax3.plot([0, lim], [0, lim], color="0.6", linewidth=0.8, linestyle="--",
                 label="1:1 line")
        ax3.errorbar(ib["obs_inc"], ib["med"],
                     yerr=[ib["med"] - ib["lo"], ib["hi"] - ib["med"]],
                     fmt="o", color=C_SIM, ecolor="0.75", markersize=3,
                     elinewidth=0.5, alpha=0.9)
        r = ib["obs_inc"].corr(ib["med"])
        ax3.text(0.97, 0.05, f"r = {r:.2f}", transform=ax3.transAxes,
                 ha="right", fontsize=8)
        ax3.set_xlim(0, lim)
        ax3.set_ylim(0, lim)
        ax3.set_xlabel("Observed (per 1000 per month)")
        ax3.set_ylabel("Simulated (per 1000 per month)")
        ax3.set_title("(c) Monthly observed vs simulated", loc="left")
        ax3.legend(frameon=False, loc="upper left")
        clean_axes(ax3)

        fig.tight_layout()
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {path}")


def fig_seasonal(ens, obs_inc, path):
    m = ens.merge(obs_inc[["year", "block"]], on=["year", "block"])
    m["share"] = m["sim_inc_obs"] / m.groupby(RUN_KEYS + ["year"])["sim_inc_obs"].transform("sum")
    run_prof = m.groupby(RUN_KEYS + ["block"], as_index=False)[["share", "sim_inc_obs"]].mean()
    sh = band(run_prof, "share", ["block"])
    ab = band(run_prof, "sim_inc_obs", ["block"])

    o = obs_inc.copy()
    o["share"] = o["obs_inc"] / o.groupby("year")["obs_inc"].transform("sum")
    osh = o.groupby("block")["share"].agg(["mean", "min", "max"])
    oab = o.groupby("block")["obs_inc"].agg(["mean", "min", "max"])
    months = np.arange(1, N_BLOCKS + 1)

    with mpl.rc_context(PUB_RC):
        style()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
        for ax, sim_b, obs_b, ylab, title in [
            (axes[0], ab, oab, "Cases per 1000 U5 per month",
             "(a) Mean monthly incidence"),
            (axes[1], sh, osh, "Share of annual U5 cases",
             "(b) Seasonal shape (share of year)"),
        ]:
            ax.fill_between(months, obs_b["min"], obs_b["max"], color=C_OBS,
                            alpha=0.10, linewidth=0, label="Observed range (years)")
            ax.plot(months, obs_b["mean"], "-o", color=C_OBS, linewidth=1.2,
                    markersize=3, label="Observed mean")
            ax.fill_between(sim_b["block"], sim_b["lo"], sim_b["hi"], color=C_SIM,
                            alpha=0.18, linewidth=0, label="Simulated 95% interval")
            ax.plot(sim_b["block"], sim_b["med"], "-", color=C_SIM, linewidth=1.5,
                    label="Simulated median")
            ax.set_xticks(months)
            ax.set_xticklabels(MONTH_ABBR)
            ax.set_xlabel("Month (30-day model block)")
            ax.set_ylabel(ylab)
            ax.set_title(title, loc="left")
            clean_axes(ax)
        r = np.corrcoef(osh["mean"], sh["med"])[0, 1]
        axes[1].text(0.97, 0.95, f"r = {r:.2f}", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Experiment: {EXPERIMENT_NAME}")

    sim = load_sim()
    dhs = load_dhs()
    obs_inc = load_obs_incidence()
    print(f"Runs: {sim[RUN_KEYS].drop_duplicates().shape[0]} | "
          f"parameter sets: {sim[SET_KEYS].drop_duplicates().shape[0]} | "
          f"DHS surveys: {len(dhs)} | observed months: {len(obs_inc)} "
          f"({obs_inc['year'].min()}-{obs_inc['year'].max()})")

    sets = score_sets(sim, obs_inc)
    print("\nParameter sets ranked (PfPR rank + incidence rank):")
    print(sets.head(10).round(4).to_string(index=False))

    ens = select_ensemble(sim, sets)
    print(f"\nEnsemble: top {ENSEMBLE_N} sets x seeds = "
          f"{ens[RUN_KEYS].drop_duplicates().shape[0]} runs")

    stats_df, pb, ib = fit_statistics(ens, dhs, obs_inc)
    print("\nFit statistics: median [2.5%, 97.5%] across ensemble runs")
    for _, r in stats_df.iterrows():
        rng = ("" if pd.isna(r.get("lo_2.5")) else
               f"  [{r['lo_2.5']:.3g}, {r['hi_97.5']:.3g}]")
        print(f"  {r['statistic']:<62} {r['median']:.3g}{rng}")

    print("\nPfPR by survey (ensemble median [95% interval] vs observed [95% CI]):")
    for _, r in pb.iterrows():
        print(f"  {int(r['year'])} block {int(r['block']):2d}: sim {r['med']:.3f} "
              f"[{r['lo']:.3f}-{r['hi']:.3f}]  obs {r['obs_pfpr']:.3f} "
              f"[{r['obs_lo']:.3f}-{r['obs_hi']:.3f}]")

    stats_df.to_csv(os.path.join(OUT_DIR, "fit_statistics.csv"), index=False)
    sets.to_csv(os.path.join(OUT_DIR, "parameter_set_scores.csv"), index=False)
    pb.to_csv(os.path.join(OUT_DIR, "pfpr_obs_vs_sim.csv"), index=False)
    ib.to_csv(os.path.join(OUT_DIR, "incidence_monthly_obs_vs_sim.csv"), index=False)

    chosen = sets.head(ENSEMBLE_N)[SET_KEYS + ["rho", "ll_pfpr", "inc_log_rmse",
                                               "inc_r", "joint_rank"]]
    chosen.to_csv(os.path.join(OUT_DIR, PARAM_SET_OUT), index=False)
    chosen.to_csv(os.path.join(INPUT_DIR, PARAM_SET_OUT), index=False)
    print(f"\nWrote {os.path.join(INPUT_DIR, PARAM_SET_OUT)} "
          f"(set PARAMETER_SET_FILE = \"{PARAM_SET_OUT}\" in run_scenariosIRS_UWR.py)")

    fig_pfpr(ens, dhs, pb, os.path.join(OUT_DIR, "fig1_pfpr_validation.png"))
    fig_incidence(ens, obs_inc, ib, os.path.join(OUT_DIR, "fig2_incidence_validation.png"))
    fig_seasonal(ens, obs_inc, os.path.join(OUT_DIR, "fig3_seasonal_incidence.png"))


if __name__ == "__main__":
    main()
