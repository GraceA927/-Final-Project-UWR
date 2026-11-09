"""
Publication figures for the Upper West Region (UWR) malaria-control scenarios.

Companion to run_scenarios_UWR.py (simulations) and analyze_scenarios_UWR.py
(stage 1/2 csv outputs). This script only READS the analyzer csvs and draws
figures + tables that answer the four policy questions:

  Q1  Does continuing IRS control malaria better than replacing it with IG2 nets?
  Q2  Which IG2 coverage (60 / 80 / 90 %) gives the best results?
  Q3  Does improving case management (CM) by 5 percentage points reduce cases?
  Q4  Does IG2 + improved CM do better than IG2 alone?

Design (16 arms, paired across the same pickup states, weather and seeds)
  BAU         IRS continues at the current coverage (reference arm)
  NO_IRS      IRS stopped, no replacement
  IRS_<c>     IRS continued at c = 60 / 80 / 90 %
  IG2_<c>_*   IRS replaced by Interceptor G2 nets at c = 60 / 80 / 90 %
  <arm>_CM5   the same arm with effective CM coverage + 5 percentage points

Outcomes (analysis window ANALYSIS_YEARS, default 2027-2032, i.e. after the switch)
  * clinical incidence per 1,000 person-years, all ages
  * PfPR (microscopy), all ages; PfPR in children under 5 when available
Uncertainty: median and 10th-90th percentile across the paired model
realisations (pickup states). Comparisons with BAU are PAIRED: each arm is
compared with BAU from the same pickup state, then summarised.

Outputs (in <analyzer output dir>/publication_figures/):
  Fig1_TimeSeries_incidence_PfPR.{png,pdf}     trajectories 2026-2032
  Fig2_CoverageResponse_IRS_vs_IG2.{png,pdf}   Q1 + Q2
  Fig3_PairedChange_vs_BAU.{png,pdf}           all 15 arms vs BAU (Q1-Q4)
  Fig4_CaseManagement_effect.{png,pdf}         Q3
  Fig5_IG2_plus_CM.{png,pdf}                   Q4
  Table1_Scenario_outcomes.csv / .md           every arm, numbers for the text
  Table2_Policy_questions.csv / .md            direct answers to Q1-Q4
  Figure_captions.md                           draft captions

Usage
  python plot_scenarios_publication.py                         # default dir
  python plot_scenarios_publication.py --in-dir <AnalyzerOutput dir>
  python plot_scenarios_publication.py --titles                # add figure titles
  python plot_scenarios_publication.py --years 2028 2032       # other window
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS (MUST MATCH run_scenarios_UWR.py / analyze_scenarios_UWR.py)
# =============================================================================

PROJECT_DIR = "/shared/home/grace/Final-Project-UWR"
EXPERIMENT_NAME = "Pickup13yrs_Scenario_Final"        # key in analyze_scenarios_UWR.EXPERIMENTS
DEFAULT_IN_DIR = os.path.join(PROJECT_DIR, "experiments",
                              f"{EXPERIMENT_NAME}_AnalyzerOutput")
STEP = "scenarios"

BASELINE = "BAU"
NO_IRS = "NO_IRS"
PROJ_START_YEAR = 2026
PROJ_END_YEAR = 2032
IRS_SWITCH_YEAR = 2027
IG2_CAMPAIGN_YEARS = [2027, 2030]
IG2_CAMPAIGN_DOY = 91
CM_BOOST_PP = 5                    # label only: +5 percentage points
ANALYSIS_YEARS = list(range(IRS_SWITCH_YEAR, PROJ_END_YEAR + 1))
COVERAGES = [60, 80, 90]
LOG_SCALE_RATIO = 8          # Fig 2 switches to a log y-axis above this max/min ratio
LOW_Q, HIGH_Q = 0.10, 0.90         # uncertainty interval across paired realisations

REGION = "Upper West Region, Ghana"

# ---- Outcomes ---------------------------------------------------------------
METRICS = {
    "incidence_per_1000_py": dict(
        label="Clinical incidence\n(cases per 1,000 person-years)",
        short="clinical incidence", scale=1.0, unit="per 1,000 PY", fmt="{:.0f}"),
    "pfpr_all_ages": dict(
        label="PfPR, all ages (%)", short="PfPR (all ages)",
        scale=100.0, unit="%", fmt="{:.1f}"),
    "U5_pfpr": dict(
        label="PfPR, children under 5 (%)", short="PfPR (under 5)",
        scale=100.0, unit="%", fmt="{:.1f}"),
    "U5_clinical_incidence_per_1000_py": dict(
        label="Clinical incidence, under 5\n(cases per 1,000 person-years)",
        short="under-5 clinical incidence", scale=1.0, unit="per 1,000 PY", fmt="{:.0f}"),
}
PRIMARY = "incidence_per_1000_py"
SECONDARY = "pfpr_all_ages"

# ---- Style ------------------------------------------------------------------
# Categorical slots 1-3 of the reference data-viz palette (validated as a set
# for colour-vision deficiency) + a neutral for the reference arm. Coverage is a
# tint of the strategy hue and is ALWAYS also direct-labelled or on an axis.
COLORS = {
    "IG2": "#2a78d6",       # blue
    "IRS": "#eb6834",       # orange
    "NO_IRS": "#1baf7a",    # aqua
    "BAU": "#3b3a37",       # neutral ink (reference)
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MARKERS = {"IG2": "s", "IRS": "o", "NO_IRS": "D", "BAU": "^"}
COVERAGE_TINT = {60: 0.50, 80: 0.75, 90: 1.00}   # mix with white
FORMATS = ["png", "pdf"]
DPI = 600
WIDTH_2COL = 7.2                     # inches (double-column journal width)
WIDTH_1COL = 3.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,              # editable text in Illustrator / journals
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
SHOW_TITLES = False


# =============================================================================
# SCENARIO PARSING AND LABELS
# =============================================================================

CM_RE = re.compile(r"_CM(\d+)$")


def parse_arm(name):
    """'IG2_80_cfp_CM5' -> dict(strategy='IG2_80_cfp', family='IG2', cov=80, cm=5)."""
    name = str(name)
    m = CM_RE.search(name)
    cm = int(m.group(1)) if m else 0
    strategy = CM_RE.sub("", name)
    if strategy == BASELINE or strategy in ("IRS_CONTINUE",):
        family, cov = "BAU", np.nan
    elif strategy == NO_IRS:
        family, cov = "NO_IRS", np.nan
    elif strategy.startswith("IRS_"):
        family, cov = "IRS", float(strategy.split("_")[1])
    elif strategy.startswith("IG2_"):
        family, cov = "IG2", float(strategy.split("_")[1])
    else:
        family, cov = "OTHER", np.nan
    return dict(scenario=name, strategy=strategy, family=family, coverage=cov, cm=cm)


def arm_label(family, cov=np.nan, cm=0, short=False):
    if short:
        base = {"BAU": "BAU", "NO_IRS": "IRS stopped",
                "IRS": f"IRS {cov:.0f}%", "IG2": f"IG2 {cov:.0f}%"}.get(family, family)
        return base + (f" + CM" if cm else "")
    base = {"BAU": "Business as usual (current IRS)",
            "NO_IRS": "IRS stopped, no replacement",
            "IRS": f"Continue IRS, {cov:.0f}% coverage",
            "IG2": f"Replace IRS with IG2 nets, {cov:.0f}% coverage"}.get(family, family)
    return base + (f" + improved CM (+{cm} pp)" if cm else "")


def tint(hex_color, amount):
    """amount=1 -> colour itself; amount=0 -> white."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(1 - amount * (1 - rgb))


def arm_color(family, cov=np.nan):
    c = COLORS.get(family, TEXT_2)
    if family in ("IRS", "IG2") and not np.isnan(cov):
        return tint(c, COVERAGE_TINT.get(int(cov), 1.0))
    return c


# =============================================================================
# DATA
# =============================================================================

def csv_in(in_dir, prefix, required=True):
    path = os.path.join(in_dir, f"{prefix}_{STEP}.csv")
    if not os.path.isfile(path):
        if required:
            raise SystemExit(f"Missing {path}. Run analyze_scenarios_UWR.py first.")
        return None
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].astype(str)
    if "pair_id" not in df.columns:
        df["pair_id"] = df["pickup_sim_index"].astype(int).astype(str)
    df["pair_id"] = df["pair_id"].astype(str)
    arms = pd.DataFrame([parse_arm(s) for s in df["scenario"].unique()])
    df = df.drop(columns=[c for c in ("strategy", "family", "coverage", "cm")
                          if c in df.columns])
    return df.merge(arms, on="scenario", how="left")


def window_metrics(annual, years):
    """Per simulation means over the analysis window (person-time weighted)."""
    a = annual[annual["Year"].isin(years)].copy()
    a["_py"] = a["population"] * a["days_simulated"] / 365.0
    a["_pf"] = a["pfpr_all_ages"] * a["days_simulated"]
    keys = ["sim_id", "pair_id", "scenario", "strategy", "family", "coverage", "cm"]
    out = a.groupby(keys, as_index=False, dropna=False).agg(
        clinical_cases=("clinical_cases", "sum"), person_years=("_py", "sum"),
        _pf=("_pf", "sum"), days=("days_simulated", "sum"))
    out["incidence_per_1000_py"] = out["clinical_cases"] / out["person_years"] * 1000.0
    out["pfpr_all_ages"] = out["_pf"] / out["days"]
    return out.drop(columns=["_pf"])


def u5_from_pfpr(pf, years):
    """Under-five PfPR / incidence over the window (age-bin population weighted)."""
    if pf is None or not {"PfPR", "Pop", "agebin", "interval_days"} <= set(pf.columns):
        return None
    d = pf[(pf["agebin"] <= 5.0) & pf["Year"].isin(years)].copy()
    if d.empty:
        return None
    d["_pp"] = d["PfPR"] * d["Pop"]
    d["_ci"] = d["Cases"] * d["Pop"] if "Cases" in d.columns else np.nan
    iv = d.groupby(["sim_id", "Year", "report_interval", "interval_days"], as_index=False).agg(
        _pp=("_pp", "sum"), _ci=("_ci", "sum"), _pop=("Pop", "sum"))
    iv = iv[iv["_pop"] > 0]
    iv["_wp"] = iv["_pp"] / iv["_pop"] * iv["interval_days"]
    iv["_wi"] = iv["_ci"] / iv["_pop"] * 1000.0 * iv["interval_days"]
    out = iv.groupby("sim_id", as_index=False).agg(_wp=("_wp", "sum"), _wi=("_wi", "sum"),
                                                   _d=("interval_days", "sum"))
    out["U5_pfpr"] = out["_wp"] / out["_d"]
    out["U5_clinical_incidence_per_1000_py"] = out["_wi"] / out["_d"]
    return out[["sim_id", "U5_pfpr", "U5_clinical_incidence_per_1000_py"]]


def load(in_dir, years):
    annual = csv_in(in_dir, "Scenario_AnnualBurden")
    pf = csv_in(in_dir, "PfPR_ClinicalIncidence_monthly", required=False)
    win = window_metrics(annual, years)
    u5 = u5_from_pfpr(pf, years)
    if u5 is not None:
        win = win.merge(u5, on="sim_id", how="left")
    metrics = [m for m in METRICS if m in win.columns and win[m].notna().any()]
    return annual, pf, win, metrics


def check_design(win):
    found = set(win["scenario"])
    if BASELINE not in found:
        raise SystemExit(f"Reference arm {BASELINE} not found; scenarios: {sorted(found)}")
    n_pairs = win["pair_id"].nunique()
    per_arm = win.groupby("scenario")["pair_id"].nunique()
    print(f"Scenarios: {len(found)}; paired realisations: {n_pairs}")
    short = per_arm[per_arm < n_pairs]
    if len(short):
        print(f"WARNING: arms with missing realisations (pairs dropped in paired stats):\n"
              f"{short.to_string()}")
    return n_pairs


def q(x, p):
    x = pd.Series(x).dropna()
    return float(x.quantile(p)) if len(x) else np.nan


def level_summary(win, metric):
    """Median and interval of the absolute outcome, per arm."""
    s = METRICS[metric]["scale"]
    return win.groupby(["scenario", "strategy", "family", "coverage", "cm"],
                       dropna=False)[metric].agg(
        median=lambda x: x.median() * s, lo=lambda x: q(x, LOW_Q) * s,
        hi=lambda x: q(x, HIGH_Q) * s, n="count").reset_index()


def paired(win, metric, ref_of):
    """
    Paired differences arm - reference, where ref_of(row) gives the reference
    scenario name for that arm (None to skip). Returns per-pair rows.
    """
    val = win.set_index(["pair_id", "scenario"])[metric]
    rows = []
    for (pair, scen), v in val.items():
        ref = ref_of(scen)
        if ref is None or (pair, ref) not in val.index:
            continue
        r = val[(pair, ref)]
        rows.append(dict(pair_id=pair, scenario=scen, reference=ref, arm=v, ref=r,
                         abs_diff=v - r, rel_change=(v - r) / r if r else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    arms = pd.DataFrame([parse_arm(s) for s in df["scenario"].unique()])
    return df.merge(arms, on="scenario")


def paired_summary(p, metric):
    s = METRICS[metric]["scale"]
    return p.groupby(["scenario", "reference", "strategy", "family", "coverage", "cm"],
                     dropna=False).agg(
        n_pairs=("pair_id", "nunique"),
        abs_median=("abs_diff", lambda x: x.median() * s),
        abs_lo=("abs_diff", lambda x: q(x, LOW_Q) * s),
        abs_hi=("abs_diff", lambda x: q(x, HIGH_Q) * s),
        rel_median=("rel_change", lambda x: x.median() * 100),
        rel_lo=("rel_change", lambda x: q(x, LOW_Q) * 100),
        rel_hi=("rel_change", lambda x: q(x, HIGH_Q) * 100),
        share_lower=("abs_diff", lambda x: float((x < 0).mean())),
    ).reset_index()


def vs_bau(scen):
    return None if scen == BASELINE else BASELINE


def vs_current_cm(scen):
    a = parse_arm(scen)
    return a["strategy"] if a["cm"] else None


# =============================================================================
# PLOT HELPERS
# =============================================================================

def style_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel_letter(ax, letter, x=-0.02):
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="right", va="bottom", color=TEXT)


def save(fig, out_dir, stem, title=None):
    if title and SHOW_TITLES:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    for ext in FORMATS:
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=DPI)
    plt.close(fig)
    print(f"  wrote {stem}.{{{','.join(FORMATS)}}}")


def mark_switch(ax, text=True):
    ax.axvspan(PROJ_START_YEAR - 0.5, IRS_SWITCH_YEAR - 0.5, color="#f1f0ec", zorder=0)
    ax.axvline(IRS_SWITCH_YEAR - 0.5, color=TEXT_2, lw=0.8, ls="--", zorder=1)
    if text:
        ax.text(IRS_SWITCH_YEAR - 0.45, 0.98, "Strategies\ndiverge (Jan 2027)",
                transform=ax.get_xaxis_transform(), fontsize=6.5, color=TEXT_2,
                va="top", ha="left")


def merge_close_labels(items, gap):
    """Lines that end at (almost) the same value get ONE label, e.g. 'IRS 60/80/90%'."""
    items = sorted(items)
    groups = []
    for y, text in items:
        if groups and abs(y - groups[-1][-1][0]) < gap:
            groups[-1].append((y, text))
        else:
            groups.append([(y, text)])
    out = []
    for g in groups:
        ys = [y for y, _ in g]
        texts = [t for _, t in g]
        m = [re.match(r"^(.*?)\s*(\d+)%$", t) for t in texts]
        if len(g) > 1 and all(m) and len({x.group(1) for x in m}) == 1:
            pcts = sorted(int(x.group(2)) for x in m)
            text = f"{m[0].group(1)} {'/'.join(map(str, pcts))}%"
        else:
            text = ", ".join(texts)
        out.append((float(np.mean(ys)), text))
    return out


def spread_labels(ax, items, min_gap_frac=0.055):
    """items = [(y, text)] at the right end of the lines; merge, then push apart."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = merge_close_labels(items, gap)
    ys = [max(y, lo + 0.6 * gap) for y, _ in items]     # never sit on the x-axis line
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = max(0.0, ys[-1] - hi)          # keep inside the axis
    ys = [y - shift for y in ys]
    x = PROJ_END_YEAR + 0.15
    for y, (_, text) in zip(ys, items):
        ax.text(x, y, text, fontsize=6.5, color=TEXT_2, va="center", ha="left")


def mark_ig2_campaigns(ax):
    for y in IG2_CAMPAIGN_YEARS:
        x = y - 0.5 + (IG2_CAMPAIGN_DOY - 1) / 365.0
        ax.plot([x], [1.0], marker="v", ms=5, color=COLORS["IG2"], clip_on=False,
                transform=ax.get_xaxis_transform(), zorder=5)


# =============================================================================
# FIGURE 1: trajectories
# =============================================================================

def figure1(annual, out_dir, n_pairs):
    a = annual[annual["cm"] == 0]
    metrics = [m for m in (PRIMARY, SECONDARY) if m in a.columns]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(WIDTH_2COL, 2.6 * len(metrics)),
                             sharex=True, squeeze=False)
    families = [("IRS", "Continue IRS"), ("IG2", "Replace IRS with IG2 nets")]
    letters = iter("ABCDEF")
    for i, metric in enumerate(metrics):
        s = METRICS[metric]["scale"]
        ymax = 0
        for j, (fam, title) in enumerate(families):
            ax = axes[i, j]
            arms = [(BASELINE, "BAU", np.nan), (NO_IRS, "NO_IRS", np.nan)]
            arms += [(sc, fam, cov) for sc, cov in
                     a.loc[a["family"] == fam, ["scenario", "coverage"]]
                     .drop_duplicates().sort_values("coverage").itertuples(index=False)]
            ends = []
            for scen, f, cov in arms:
                g = a[a["scenario"] == scen].groupby("Year")[metric]
                if g.ngroups == 0:
                    continue
                med, lo, hi = g.median() * s, g.quantile(LOW_Q) * s, g.quantile(HIGH_Q) * s
                col = arm_color(f, cov)
                ls = "--" if f == "NO_IRS" else "-"
                lw = 2.0 if f == "BAU" else 1.6
                if f in ("BAU", "NO_IRS"):
                    ax.fill_between(med.index, lo, hi, color=col, alpha=0.10, lw=0)
                ax.plot(med.index, med.values, color=col, ls=ls, lw=lw,
                        marker=MARKERS[f], ms=3.5, mfc=col, mec="white", mew=0.5,
                        label=arm_label(f, cov, short=True), zorder=3)
                ymax = max(ymax, float(np.nanmax(hi.values)))
                ends.append((float(med.values[-1]), arm_label(f, cov, short=True)))
            axes[i, j]._end_labels = ends
            mark_switch(ax, text=(i == 0 and j == 0))
            if fam == "IG2":
                mark_ig2_campaigns(ax)
            style_axis(ax)
            if i == 0:
                ax.set_title(title, color=TEXT, fontsize=8.5, loc="left", pad=8)
            if j == 0:
                ax.set_ylabel(METRICS[metric]["label"])
            ax.set_xlim(PROJ_START_YEAR - 0.5, PROJ_END_YEAR + 1.25)
            ax.set_xticks(range(PROJ_START_YEAR, PROJ_END_YEAR + 1))
            panel_letter(ax, next(letters))
        for j in range(2):
            axes[i, j].set_ylim(0, ymax * 1.08)
            spread_labels(axes[i, j], axes[i, j]._end_labels)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    handles = [Line2D([], [], color=COLORS["BAU"], marker="^", lw=2, label="Business as usual (BAU)"),
               Line2D([], [], color=COLORS["NO_IRS"], marker="D", ls="--", label="IRS stopped, no replacement"),
               Line2D([], [], color=COLORS["IRS"], marker="o", label="Continue IRS (60/80/90%; darker = higher)"),
               Line2D([], [], color=COLORS["IG2"], marker="s", label="IG2 nets (60/80/90%; darker = higher)"),
               Line2D([], [], color=COLORS["IG2"], marker="v", lw=0, label="IG2 mass distribution")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "Fig1_TimeSeries_incidence_PfPR",
         f"Projected malaria burden by vector-control strategy, {REGION}")


# =============================================================================
# FIGURE 2: coverage response, IRS vs IG2 (Q1, Q2)
# =============================================================================

def figure2(win, metrics, out_dir):
    w = win[win["cm"] == 0]
    show = [m for m in (PRIMARY, SECONDARY, "U5_pfpr") if m in metrics]
    fig, axes = plt.subplots(1, len(show), figsize=(WIDTH_2COL, 2.8), squeeze=False)
    axes = axes[0]
    for ax, metric, letter in zip(axes, show, "ABC"):
        lev = level_summary(w, metric)
        for ref, f, ls in ((BASELINE, "BAU", "-"), (NO_IRS, "NO_IRS", "--")):
            r = lev[lev["scenario"] == ref]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.axhspan(r["lo"], r["hi"], color=COLORS[f], alpha=0.08, lw=0)
            ax.axhline(r["median"], color=COLORS[f], ls=ls, lw=1.2)
            # NO_IRS sits above BAU: label it above its line and BAU below its
            # line so the two never collide (they are close on a log axis).
            ax.text(1.0, r["median"], " " + arm_label(f, short=True),
                    transform=ax.get_yaxis_transform(), fontsize=6.5, color=TEXT_2,
                    va="bottom" if f == "NO_IRS" else "top", ha="left")
        pos = {float(c): i for i, c in enumerate(COVERAGES)}
        for fam, dx in (("IRS", -0.09), ("IG2", 0.09)):
            g = lev[lev["family"] == fam].sort_values("coverage")
            g = g[g["coverage"].isin(pos)]
            if g.empty:
                continue
            x = g["coverage"].map(pos).to_numpy() + dx
            ax.errorbar(x, g["median"], yerr=[g["median"] - g["lo"], g["hi"] - g["median"]],
                        color=COLORS[fam], marker=MARKERS[fam], ms=5, lw=1.6, capsize=2.5,
                        elinewidth=1.0, mec="white", mew=0.6, zorder=3,
                        label="Continue IRS" if fam == "IRS" else "Replace IRS with IG2")
        ax.set_xticks(range(len(COVERAGES)))
        ax.set_xticklabels([f"{c}%" for c in COVERAGES])
        ax.set_xlim(-0.4, len(COVERAGES) - 0.6)
        ax.set_ylabel(METRICS[metric]["label"])
        vals = pd.concat([lev["lo"], lev["hi"]]).astype(float)
        vals = vals[vals > 0]
        if not vals.empty and vals.max() / vals.min() > LOG_SCALE_RATIO:
            # Vector-control arms sit an order of magnitude below BAU: a log
            # axis keeps both readable instead of crushing IRS/IG2 at zero.
            ax.set_yscale("log")
            ax.set_ylim(vals.min() / 1.6, vals.max() * 1.6)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:g}"))
            ax._log = True
        else:
            ax.set_ylim(bottom=0)
        style_axis(ax)
        panel_letter(ax, letter)
    fig.supxlabel("Coverage (IRS: households sprayed; IG2: population receiving a net)",
                  fontsize=plt.rcParams["axes.labelsize"], y=0.10)
    handles = [Line2D([], [], color=COLORS["IRS"], marker="o", label="Continue IRS"),
               Line2D([], [], color=COLORS["IG2"], marker="s", label="Replace IRS with IG2 nets"),
               Line2D([], [], color=COLORS["BAU"], label="Business as usual (median, 80% UI band)"),
               Line2D([], [], color=COLORS["NO_IRS"], ls="--", label="IRS stopped, no replacement")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 0.94, 1))
    save(fig, out_dir, "Fig2_CoverageResponse_IRS_vs_IG2",
         f"Continuing IRS vs replacing it with IG2 nets, {ANALYSIS_YEARS[0]}-{ANALYSIS_YEARS[-1]}")


# =============================================================================
# FIGURE 3: forest plot vs BAU (all arms)
# =============================================================================

def arm_order(df):
    fam_rank = {"NO_IRS": 0, "IRS": 1, "IG2": 2, "BAU": 3}
    d = df.assign(_f=df["family"].map(fam_rank).fillna(9), _c=df["coverage"].fillna(0))
    return d.sort_values(["_f", "_c", "cm"]).drop(columns=["_f", "_c"])


def figure3(win, metrics, out_dir):
    show = [m for m in (PRIMARY, SECONDARY) if m in metrics]
    sums = {m: paired_summary(paired(win, m, vs_bau), m) for m in show}
    base = arm_order(sums[show[0]])
    # BAU + CM (vs BAU) belongs with the CM effect; keep it, it is informative.
    labels = [arm_label(r.family, r.coverage, r.cm) for r in base.itertuples()]
    y = np.arange(len(base))[::-1]
    fig, axes = plt.subplots(1, len(show), figsize=(WIDTH_2COL, 0.27 * len(base) + 1.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, metric, letter in zip(axes, show, "AB"):
        s = sums[metric].set_index("scenario").loc[base["scenario"]]
        for yi, r in zip(y, s.itertuples()):
            col = COLORS.get(r.family, TEXT_2)
            ax.plot([r.rel_lo, r.rel_hi], [yi, yi], color=col, lw=1.4, solid_capstyle="round")
            ax.plot(r.rel_median, yi, marker=MARKERS.get(r.family, "o"), ms=5.5,
                    color=col, mfc=col if r.cm == 0 else "white", mew=1.3, zorder=3)
        ax.axvline(0, color=TEXT_2, lw=0.8)
        ax.set_xlabel(f"Change in {METRICS[metric]['short']}\nvs business as usual (%)")
        style_axis(ax, grid_axis="x")
        lo, hi = ax.get_xlim()
        ax.text(lo, len(base) - 0.2, "← lower burden than BAU", fontsize=6.5, color=TEXT_2,
                ha="left", va="bottom")
        ax.text(hi, len(base) - 0.2, "higher →", fontsize=6.5, color=TEXT_2,
                ha="right", va="bottom")
        ax.set_ylim(-0.7, len(base) + 0.4)
        panel_letter(ax, letter, x=-0.02 if letter == "B" else -0.02)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    # separators between strategy families
    fam = base["family"].to_numpy()
    for i in range(1, len(fam)):
        if fam[i] != fam[i - 1]:
            for ax in axes:
                ax.axhline(y[i] + 0.5, color=GRID, lw=0.8)
    handles = [Line2D([], [], marker="o", color=TEXT_2, lw=0, ms=5.5, label="Current case management"),
               Line2D([], [], marker="o", color=TEXT_2, mfc="white", lw=0, ms=5.5, mew=1.3,
                      label=f"Improved case management (+{CM_BOOST_PP} pp)"),
               Line2D([], [], color=TEXT_2, lw=1.4,
                      label=f"{int(LOW_Q*100)}th-{int(HIGH_Q*100)}th percentile of paired realisations")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "Fig3_PairedChange_vs_BAU",
         "Paired change in malaria burden relative to business as usual")
    return sums


# =============================================================================
# FIGURE 4: effect of improved case management (Q3)
# =============================================================================

def figure4(win, metrics, out_dir):
    show = [m for m in (PRIMARY, SECONDARY) if m in metrics]
    sums = {m: paired_summary(paired(win, m, vs_current_cm), m) for m in show}
    if sums[show[0]].empty:
        print("  Fig4 skipped: no improved-CM arms found.")
        return None
    base = arm_order(sums[show[0]])
    labels = [arm_label(r.family, r.coverage, short=True) for r in base.itertuples()]
    x = np.arange(len(base))
    fig, axes = plt.subplots(1, len(show), figsize=(WIDTH_2COL, 2.9), squeeze=False)
    axes = axes[0]
    for ax, metric, letter in zip(axes, show, "AB"):
        s = sums[metric].set_index("scenario").loc[base["scenario"]]
        cols = [arm_color(r.family, r.coverage) for r in s.itertuples()]
        # cases averted (positive = benefit) for incidence; % change for PfPR
        if metric == PRIMARY:
            v, lo, hi = -s["abs_median"], -s["abs_hi"], -s["abs_lo"]
            ylabel = "Clinical cases averted per 1,000\nperson-years by +5 pp CM"
        else:
            v, lo, hi = -s["rel_median"], -s["rel_hi"], -s["rel_lo"]
            ylabel = f"Relative reduction in\n{METRICS[metric]['short']} by +5 pp CM (%)"
        ax.bar(x, v, width=0.68, color=cols, edgecolor="white", lw=0.8, zorder=2)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=TEXT, elinewidth=0.9,
                    capsize=2.5, zorder=3)
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        style_axis(ax)
        panel_letter(ax, letter)
    fig.tight_layout()
    save(fig, out_dir, "Fig4_CaseManagement_effect",
         f"Additional impact of raising effective case management by {CM_BOOST_PP} percentage points")
    return sums


# =============================================================================
# FIGURE 5: IG2 + improved CM vs IG2 alone vs IRS (Q4)
# =============================================================================

def figure5(win, metrics, out_dir):
    """
    Q4 as paired differences: does adding +5 pp CM to IG2 beat IG2 alone, and
    does IG2 (+/- CM) beat continuing IRS at the same coverage? Plotting the
    within-realisation difference (not change vs BAU) keeps a few-percent
    effect visible when every vector-control arm is ~90% below BAU.
    """
    show = [m for m in (PRIMARY, SECONDARY) if m in metrics]
    arms = {(a["family"], a["coverage"], a["cm"]): a["scenario"]
            for a in (parse_arm(s) for s in win["scenario"].unique())}
    comps = [  # (arm family, arm cm, ref family, ref cm, label, colour, marker, filled)
        ("IG2", CM_BOOST_PP, "IG2", 0, f"IG2 + CM (+{CM_BOOST_PP} pp) vs IG2 alone",
         COLORS["IG2"], "s", False),
        ("IG2", CM_BOOST_PP, "IRS", 0, f"IG2 + CM (+{CM_BOOST_PP} pp) vs continue IRS",
         COLORS["IRS"], "o", False),
        ("IG2", 0, "IRS", 0, "IG2 alone vs continue IRS", COLORS["IRS"], "o", True),
    ]
    fig, axes = plt.subplots(1, len(show), figsize=(WIDTH_2COL, 2.9), squeeze=False)
    axes = axes[0]
    width = 0.22
    for ax, metric, letter in zip(axes, show, "AB"):
        for k, (af, acm, rf, rcm, lab, col, mk, filled) in enumerate(comps):
            meds, los, his, xs = [], [], [], []
            for i, c in enumerate(COVERAGES):
                a, r = arms.get((af, float(c), acm)), arms.get((rf, float(c), rcm))
                if a is None or r is None:
                    continue
                p = paired(win, metric, lambda s, a=a, r=r: r if s == a else None)
                if p.empty:
                    continue
                rel = p["rel_change"] * 100
                meds.append(rel.median()); los.append(q(rel, LOW_Q)); his.append(q(rel, HIGH_Q))
                xs.append(i + (k - 1) * width)
            if not xs:
                continue
            meds, los, his = map(np.asarray, (meds, los, his))
            ax.errorbar(xs, meds, yerr=[meds - los, his - meds], fmt=mk, ms=5.5,
                        color=col, mfc=col if filled else "white", mec=col, mew=1.3,
                        elinewidth=1.1, capsize=2.5, zorder=3, label=lab)
        ax.axhline(0, color=TEXT_2, lw=0.8, zorder=1)
        ax.text(1.0, 0, " no difference", transform=ax.get_yaxis_transform(),
                fontsize=6.3, color=TEXT_2, va="center")
        ax.set_xticks(np.arange(len(COVERAGES)))
        ax.set_xticklabels([f"{c}%" for c in COVERAGES])
        ax.set_xlim(-0.5, len(COVERAGES) - 0.5)
        ax.set_xlabel("Coverage of IRS or IG2")
        ax.set_ylabel(f"Paired difference in {METRICS[metric]['short']}\n(% of comparator; <0 favours IG2)")
        style_axis(ax)
        panel_letter(ax, letter)
    handles = [Line2D([], [], ls="none", marker=mk, ms=5.5, color=col,
                      mfc=col if filled else "white", mec=col, mew=1.3, label=lab)
               for *_, lab, col, mk, filled in comps]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 0.93, 1))
    save(fig, out_dir, "Fig5_IG2_plus_CM",
         "IG2 nets with improved case management vs IG2 alone and vs continued IRS")


# =============================================================================
# TABLES
# =============================================================================

def fmt_ci(m, lo, hi, f="{:.1f}"):
    if pd.isna(m):
        return "-"
    return f"{f.format(m)} ({f.format(lo)} to {f.format(hi)})"


def write_md(df, path, title, note):
    with open(path, "w") as fh:
        fh.write(f"**{title}**\n\n")
        cols = list(df.columns)
        fh.write("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n")
        for r in df.itertuples(index=False):
            fh.write("| " + " | ".join(str(v) for v in r) + " |\n")
        fh.write(f"\n{note}\n")


def table1(win, metrics, out_dir, years):
    rows = []
    sums = {m: paired_summary(paired(win, m, vs_bau), m).set_index("scenario") for m in metrics}
    levs = {m: level_summary(win, m).set_index("scenario") for m in metrics}
    order = arm_order(levs[metrics[0]].reset_index())
    for r in order.itertuples():
        row = {"Strategy": arm_label(r.family, r.coverage),
               "Case management": f"Improved (+{r.cm} pp)" if r.cm else "Current"}
        for m in metrics:
            f = METRICS[m]["fmt"]
            L = levs[m].loc[r.scenario]
            row[f"{METRICS[m]['short']} ({METRICS[m]['unit']})"] = fmt_ci(L["median"], L["lo"], L["hi"], f)
            if r.scenario in sums[m].index:
                S = sums[m].loc[r.scenario]
                row[f"Change in {METRICS[m]['short']} vs BAU (%)"] = fmt_ci(
                    S["rel_median"], S["rel_lo"], S["rel_hi"])
            else:
                row[f"Change in {METRICS[m]['short']} vs BAU (%)"] = "reference"
        if r.scenario in sums[PRIMARY].index:
            S = sums[PRIMARY].loc[r.scenario]
            row["Cases averted per 1,000 PY vs BAU"] = fmt_ci(-S["abs_median"], -S["abs_hi"],
                                                              -S["abs_lo"], "{:.0f}")
            row["Realisations with fewer cases than BAU (%)"] = f"{S['share_lower']*100:.0f}"
        else:
            row["Cases averted per 1,000 PY vs BAU"] = "reference"
            row["Realisations with fewer cases than BAU (%)"] = "-"
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "Table1_Scenario_outcomes.csv"), index=False)
    write_md(t, os.path.join(out_dir, "Table1_Scenario_outcomes.md"),
             f"Table 1. Projected malaria outcomes by strategy, {REGION}, {years[0]}-{years[-1]}",
             f"Values are medians ({int(LOW_Q*100)}th to {int(HIGH_Q*100)}th percentile) across "
             f"{win['pair_id'].nunique()} paired model realisations. Changes vs business as usual "
             f"(BAU) are paired within realisation. PY = person-years; pp = percentage points; "
             f"CM = case management; IG2 = Interceptor G2 nets; IRS = indoor residual spraying.")
    print("  wrote Table1_Scenario_outcomes.{csv,md}")


def table2(win, out_dir, years):
    """Direct, paired answers to Q1-Q4 on the primary outcome."""
    val = win.pivot_table(index="pair_id", columns="scenario", values=PRIMARY)
    rows = []

    def compare(question, a, b, desc):
        if a not in val or b not in val:
            return
        d = (val[a] - val[b]).dropna()
        rel = (d / val[b].loc[d.index]) * 100
        rows.append({
            "Question": question, "Comparison": desc,
            "Difference in incidence per 1,000 PY (A - B)": fmt_ci(d.median(), q(d, LOW_Q), q(d, HIGH_Q), "{:.0f}"),
            "Relative difference (%)": fmt_ci(rel.median(), q(rel, LOW_Q), q(rel, HIGH_Q)),
            "Realisations where A < B (%)": f"{(d < 0).mean()*100:.0f}",
            "n": len(d)})

    arms = {parse_arm(s)["strategy"]: s for s in val.columns if not parse_arm(s)["cm"]}
    ig2 = {int(parse_arm(s)["coverage"]): s for s in arms.values() if parse_arm(s)["family"] == "IG2"}
    irs = {int(parse_arm(s)["coverage"]): s for s in arms.values() if parse_arm(s)["family"] == "IRS"}
    for c in COVERAGES:
        if c in irs and c in ig2:
            compare("Q1 IRS vs IG2", irs[c], ig2[c], f"A = continue IRS {c}%, B = IG2 {c}%")
    for c in COVERAGES:
        if c in ig2:
            compare("Q1 IRS vs IG2", ig2[c], BASELINE, f"A = IG2 {c}%, B = BAU")
    cs = sorted(ig2)
    for lo_c, hi_c in zip(cs[:-1], cs[1:]):
        compare("Q2 IG2 coverage", ig2[hi_c], ig2[lo_c], f"A = IG2 {hi_c}%, B = IG2 {lo_c}%")
    for s in [BASELINE, NO_IRS] + [irs[c] for c in sorted(irs)] + [ig2[c] for c in cs]:
        boosted = f"{s}_CM{CM_BOOST_PP}"
        a = parse_arm(s)
        compare("Q3 improved CM", boosted, s,
                f"A = {arm_label(a['family'], a['coverage'], short=True)} + CM, B = same, current CM")
    for c in cs:
        compare("Q4 IG2 + CM", f"{ig2[c]}_CM{CM_BOOST_PP}", ig2[c],
                f"A = IG2 {c}% + CM, B = IG2 {c}% alone")
        if c in irs:
            compare("Q4 IG2 + CM", f"{ig2[c]}_CM{CM_BOOST_PP}", irs[c],
                    f"A = IG2 {c}% + CM, B = continue IRS {c}%")
    t = pd.DataFrame(rows)
    if t.empty:
        return
    t.to_csv(os.path.join(out_dir, "Table2_Policy_questions.csv"), index=False)
    write_md(t, os.path.join(out_dir, "Table2_Policy_questions.md"),
             f"Table 2. Paired comparisons answering the policy questions "
             f"(all-age clinical incidence, {years[0]}-{years[-1]})",
             f"Negative differences favour A. Median ({int(LOW_Q*100)}th to {int(HIGH_Q*100)}th "
             f"percentile) of within-realisation differences.")
    print("  wrote Table2_Policy_questions.{csv,md}")


def write_captions(out_dir, n_pairs, years):
    ui = f"{int(LOW_Q*100)}th-{int(HIGH_Q*100)}th percentile"
    w = f"{years[0]}-{years[-1]}"
    text = f"""# Draft figure captions

**Figure 1. Projected malaria burden in the {REGION} under alternative vector-control strategies, {PROJ_START_YEAR}-{PROJ_END_YEAR}.**
Annual all-age clinical incidence (A, B) and all-age *P. falciparum* prevalence by microscopy (C, D)
when indoor residual spraying (IRS) is continued at 60%, 80% or 90% coverage (A, C) or replaced by
Interceptor G2 (IG2) nets distributed to 60%, 80% or 90% of the population (B, D). All strategies
share the same history to the end of {IRS_SWITCH_YEAR - 1} (shaded) and diverge on 1 January {IRS_SWITCH_YEAR}.
Lines are medians across {n_pairs} calibrated model realisations; shaded bands around business as
usual (BAU) and the no-replacement scenario show the {ui}. Triangles mark IG2 mass distributions
(April {', '.join(map(str, IG2_CAMPAIGN_YEARS))}). Case management at current levels.

**Figure 2. Continuing IRS versus replacing it with IG2 nets, by coverage ({w}).**
Mean annual all-age clinical incidence (A), all-age PfPR (B) and PfPR in children under 5 (C) over
{w}. Points are medians and bars the {ui} across {n_pairs} realisations. Horizontal lines show BAU
(solid, with {ui} band) and stopping IRS without replacement (dashed). IRS coverage is the share of
households sprayed; IG2 coverage is the share of the population receiving a net; the two are
comparable programme targets, not equivalent measures. The y-axis is logarithmic when the
vector-control arms are more than eight-fold below business as usual.

**Figure 3. Paired change in malaria burden relative to business as usual ({w}).**
Percentage change in all-age clinical incidence (A) and all-age PfPR (B) for each of the 15
alternative scenarios relative to BAU in the same realisation. Filled symbols: current case
management; open symbols: effective case-management coverage increased by {CM_BOOST_PP} percentage
points from {IRS_SWITCH_YEAR}. Symbols are medians; lines the {ui} of paired differences across
{n_pairs} realisations. Values left of zero indicate lower burden than BAU.

**Figure 4. Additional impact of improving case management by {CM_BOOST_PP} percentage points ({w}).**
Clinical cases averted per 1,000 person-years (A) and relative reduction in all-age PfPR (B) when
effective case-management coverage is raised by {CM_BOOST_PP} percentage points, compared with the
same vector-control strategy at current case management (paired within realisation). Bars are
medians; error bars the {ui} across {n_pairs} realisations.

**Figure 5. Does adding improved case management to IG2 nets add benefit, and how does IG2 compare with continued IRS? ({w}).**
Paired percentage difference in all-age clinical incidence (A) and all-age PfPR (B), by coverage:
IG2 nets with case management +{CM_BOOST_PP} percentage points versus IG2 nets alone (open squares),
IG2 + improved case management versus continued IRS at the same coverage (open circles), and IG2 alone
versus continued IRS (filled circles). Each difference is computed within a realisation and expressed as a
percentage of the comparator; values below zero favour the IG2 arm. Symbols are medians and error bars
the {ui} across {n_pairs} realisations.
"""
    with open(os.path.join(out_dir, "Figure_captions.md"), "w") as fh:
        fh.write(text)
    print("  wrote Figure_captions.md")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Publication figures for the UWR scenarios.")
    p.add_argument("--in-dir", default=DEFAULT_IN_DIR,
                   help="analyze_scenarios_UWR.py output directory (*_AnalyzerOutput).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <in-dir>/publication_figures).")
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("START", "END"),
                   help=f"Analysis window (default {ANALYSIS_YEARS[0]} {ANALYSIS_YEARS[-1]}).")
    p.add_argument("--titles", action="store_true", help="Add figure titles (off for journals).")
    p.add_argument("--formats", nargs="+", default=FORMATS, help="e.g. png pdf svg tiff")
    return p.parse_args()


def main():
    global SHOW_TITLES, FORMATS
    args = parse_args()
    SHOW_TITLES = args.titles
    FORMATS = args.formats
    years = list(range(args.years[0], args.years[1] + 1)) if args.years else ANALYSIS_YEARS
    out_dir = args.out_dir or os.path.join(args.in_dir, "publication_figures")
    os.makedirs(out_dir, exist_ok=True)

    annual, pf, win, metrics = load(args.in_dir, years)
    n_pairs = check_design(win)
    print(f"Analysis window {years[0]}-{years[-1]}; outcomes: {metrics}")

    for name, fn, fargs in [
        ("Figure 1", figure1, (annual, out_dir, n_pairs)),
        ("Figure 2", figure2, (win, metrics, out_dir)),
        ("Figure 3", figure3, (win, metrics, out_dir)),
        ("Figure 4", figure4, (win, metrics, out_dir)),
        ("Figure 5", figure5, (win, metrics, out_dir)),
        ("Table 1", table1, (win, metrics, out_dir, years)),
        ("Table 2", table2, (win, out_dir, years)),
        ("Captions", write_captions, (out_dir, n_pairs, years)),
    ]:
        try:
            fn(*fargs)
        except Exception as exc:          # one failing figure must not stop the rest
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}")
    print(f"\nPublication outputs in {out_dir}")


if __name__ == "__main__":
    main()
