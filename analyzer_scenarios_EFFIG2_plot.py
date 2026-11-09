"""
UWR publication and stakeholder figures, built from the analyzer CSVs.

Reads the files written by analyze_scenarios_UWR.py (stage 2) and writes
figures (PNG + PDF) and a captions text file. It does not run idmtools and
needs only pandas, numpy and matplotlib.

IG2 usage: IG2_USE_CURVES = False in the scenario and analyzer scripts, so IG2
nets carry no age or seasonal usage curves. Effective use at distribution = c;
time-averaged effective use = c x net retention.

Figures
-------
  Figure 1  Annual clinical incidence: IRS vs IG2 at matched coverage,
            each compared with the baseline (no IRS, no IG2)
  Figure 2  Annual PfPR by scenario and age group
  Figure 3  Paired change in clinical incidence relative to IRS benchmark
  Figure 4  IG2 distribution coverage versus clinical incidence and PfPR
  Figure 5  Annual clinical cases averted: IRS vs IG2 at matched coverage,
            relative to the baseline (no IRS, no IG2)
  Figure 6  IG2 distribution coverage and estimated effective-use coverage

Input CSVs (in the analyzer output folder)
------------------------------------------
  Scenario_AnnualBurden_scenarios.csv            Figures 1, 5
  PfPR_ClinicalIncidence_monthly_scenarios.csv   Figure 2
  Scenario_PairedImpact_scenarios.csv            Figure 3
  Scenario_WindowMetrics_scenarios.csv           Figure 4
  IG2_EffectiveUse_scenarios.csv                 Figure 6
"""

import argparse
import os
import re
import textwrap
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DEFAULT_DIR = ("/shared/home/grace/Final-Project-UWR/experiments/"
               "operational_effectiveness_use_AnalyzerOutput")
STEP = "scenarios"

IRS_SWITCH_YEAR = 2027                 # MUST MATCH the scenario script
IG2_USE_CURVES = False                 # MUST MATCH the scenario and analyzer scripts
PRIMARY_METRIC = "incidence_per_1000_py"
REFERENCE_ORDER = ["IRS_90", "IRS_70", "IRS_50"]   # IRS benchmarks (Figure 3)
NO_ARM = "NO_IRS"                      # no IRS and no IG2 (baseline)

# Figures 1 and 5: one panel per coverage level (percent). Must match the arm
# names IRS_<c> and IG2_<c>_<key> in the analyzer CSVs.
COMPARE_COVERAGES = [50, 70, 90]
FIG1_SHARE_Y = True                    # False if the baseline flattens the other curves
FIG5_SHARE_Y = True

# Display names in figures only; the csv files keep the scenario tag names.
ARM_LABELS = {"NO_IRS": "Baseline"}
BASELINE_LABEL = ARM_LABELS[NO_ARM]

# Age groups for Figure 2: group name -> upper edges (years) of the summary-report
# age bins it combines (REPORT_AGE_BINS = 0.25, 0.5, 5, 15, 25, 100).
AGE_GROUPS = {
    "Under 5 years": [0.25, 0.5, 5.0],
    "5\u201314 years": [15.0],
    "15 years and older": [25.0, 100.0],
}

# Fixed colours by role, shared by Figures 1 and 5.
ROLE_COLOR = {"IRS": "#d95f02", "IG2": "#1b9e77"}

LOWER_Q, HIGH_Q = 0.10, 0.90
SHOW_RANGE = False                     # percentile bands / error bars (--show-range)
SHOW_TITLES = True                     # (--no-titles to switch off)
FORMATS = ["png", "pdf"]
DPI = 300


# =============================================================================
# LABELS, STYLES, HELPERS
# =============================================================================

def display(name):
    if name in ARM_LABELS:
        return ARM_LABELS[name]
    m = re.fullmatch(r"IRS_(\d+)", name)
    if m:
        return f"IRS {m.group(1)}%"
    m = re.fullmatch(r"IG2_(\d+)_\w+", name)
    if m:
        return f"IG2 {m.group(1)}%"
    return name


def cov_of(name):
    m = re.search(r"_(\d+)", name)
    return int(m.group(1)) if m else -1


def ordered(names):
    names = list(names)
    irs = sorted([n for n in names if re.fullmatch(r"IRS_\d+", n)], key=cov_of)
    ig2 = sorted([n for n in names if n.startswith("IG2_")], key=lambda n: (cov_of(n), n))
    rest = [n for n in names if n not in irs and n not in ig2 and n != NO_ARM]
    return irs + ([NO_ARM] if NO_ARM in names else []) + rest + ig2


def build_styles(names):
    names = list(names)
    irs = sorted([n for n in names if re.fullmatch(r"IRS_\d+", n)], key=cov_of)
    ig2 = sorted([n for n in names if n.startswith("IG2_")], key=lambda n: (cov_of(n), n))
    styles = {}
    for i, n in enumerate(irs):
        f = 0.5 if len(irs) == 1 else i / (len(irs) - 1)
        styles[n] = dict(color=plt.cm.Oranges(0.50 + 0.45 * f), linestyle="-",
                         marker="o", linewidth=2.0, markersize=5)
    for i, n in enumerate(ig2):
        f = 0.5 if len(ig2) == 1 else i / (len(ig2) - 1)
        styles[n] = dict(color=plt.cm.Blues(0.50 + 0.45 * f), linestyle="--",
                         marker="s", linewidth=2.0, markersize=5)
    for n in names:
        if n == NO_ARM:
            styles[n] = dict(color="0.35", linestyle="-.", marker="^",
                             linewidth=1.8, markersize=5)
        elif n not in styles:
            styles[n] = dict(color="0.6", linestyle="-", marker=None, linewidth=1.5)
    return styles


def ig2_bar_colors(coverages):
    covs = sorted(coverages)
    return {c: plt.cm.Blues(0.45 + 0.5 * (0.5 if len(covs) == 1 else i / (len(covs) - 1)))
            for i, c in enumerate(covs)}


def find_arm(all_names, prefix):
    """First scenario named <prefix> or <prefix>_<suffix> (e.g. IG2_50_cfp)."""
    hits = sorted(n for n in all_names if n == prefix or n.startswith(prefix + "_"))
    return hits[0] if hits else None


def summarise(df, keys, value):
    g = df.groupby(keys)[value]
    return pd.DataFrame({"median": g.median(), "lo": g.quantile(LOWER_Q),
                         "hi": g.quantile(HIGH_Q)}).reset_index()


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def mark_switch(ax, label=True):
    x = IRS_SWITCH_YEAR - 0.5
    ax.axvline(x, color="0.45", linestyle=":", linewidth=1.1)
    if label:
        ax.annotate("Scenarios diverge", xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top",
                    fontsize=8, color="0.35")


def panel_letter(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")


def plain_numbers(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def auto_scale(ax, values):
    """Log axis when the values span more than ~30-fold and are all positive."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) and v.min() > 0 and v.max() / v.min() > 30:
        ax.set_yscale("log")
        plain_numbers(ax)
        return True
    return False


def add_figure_legend(fig, ax, ncol=4, offset=-0.02):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, offset), fontsize=9.5)


def save_figure(fig, out_dir, stem, title=None):
    if SHOW_TITLES and title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)
    for fmt in FORMATS:
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}")


def read_csv(in_dir, prefix, required=True):
    path = os.path.join(in_dir, f"{prefix}_{STEP}.csv")
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(path)
        return None
    df = pd.read_csv(path)
    for c in ("scenario", "scenario_kind", "ig2_kill_key", "reference"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "pair_id" not in df.columns and "pickup_sim_index" in df.columns:
        df["pair_id"] = df["pickup_sim_index"].astype(int).astype(str)
    return df


# =============================================================================
# FIGURE 1: ANNUAL CLINICAL INCIDENCE, IRS vs IG2 AT MATCHED COVERAGE
# =============================================================================

def figure1(annual, out_dir):
    s = summarise(annual, ["scenario", "Year"], PRIMARY_METRIC)
    all_names = list(s["scenario"].unique())
    years = sorted(s["Year"].unique())

    if NO_ARM not in all_names:
        raise ValueError(f"{NO_ARM} is not in the annual csv; rerun the scenarios "
                         f"with INCLUDE_NO_IRS_ARM = True.")

    role_style = {
        NO_ARM: dict(color="0.25", linestyle="-.", linewidth=2.2, marker="^", markersize=5),
        "IRS":  dict(color=ROLE_COLOR["IRS"], linestyle="-", linewidth=2.0),
        "IG2":  dict(color=ROLE_COLOR["IG2"], linestyle="-", linewidth=2.0),
    }

    fig, axes = plt.subplots(1, len(COMPARE_COVERAGES),
                             figsize=(5.0 * len(COMPARE_COVERAGES), 4.8),
                             squeeze=False, sharey=FIG1_SHARE_Y)

    for k, (ax, c) in enumerate(zip(axes[0], COMPARE_COVERAGES)):
        irs_name = find_arm(all_names, f"IRS_{c}")
        ig2_name = find_arm(all_names, f"IG2_{c}")

        arms = [(NO_ARM, BASELINE_LABEL, role_style[NO_ARM])]
        if irs_name:
            arms.append((irs_name, f"IRS ({c}% coverage)", role_style["IRS"]))
        if ig2_name:
            arms.append((ig2_name, f"IG2 ({c}% distributed)", role_style["IG2"]))
        if not (irs_name and ig2_name):
            print(f"  WARNING Figure 1: panel {c}% is missing "
                  f"{'IRS' if not irs_name else 'IG2'} arm.")

        for name, label, style in arms:
            d = s[s["scenario"] == name].sort_values("Year")
            ax.plot(d["Year"], d["median"], label=label, **style)
            if SHOW_RANGE:
                ax.fill_between(d["Year"], d["lo"], d["hi"],
                                color=style["color"], alpha=0.15, linewidth=0)

        mark_switch(ax, label=(k == 0))
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Year")
        if k == 0 or not FIG1_SHARE_Y:
            ax.set_ylabel("Clinical incidence (cases per 1,000 person-years)")
        ax.set_title(f"{c}% coverage", fontsize=11)
        ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "abcdefgh"[k])

    add_figure_legend(fig, axes[0][0], ncol=4, offset=-0.10)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure1_AnnualIncidence_IRSvsIG2",
                "Annual Clinical Malaria Incidence: IRS vs IG2 at Matched Coverage")


# =============================================================================
# FIGURE 2: ANNUAL PfPR BY SCENARIO AND AGE GROUP
# =============================================================================

def grouped_pfpr(pf):
    """Annual PfPR (%) per simulation and age group; weights = population x days."""
    pf = pf.copy()
    pf["_bin"] = pf["agebin"].round(4)
    scale = 100.0 if pf["PfPR"].max() <= 1.0 + 1e-9 else 1.0
    pf["_w"] = pf["Pop"] * pf["interval_days"]
    pf["_wp"] = pf["PfPR"] * pf["_w"]
    present = set(pf["_bin"].unique())
    pieces = []
    for group, edges in AGE_GROUPS.items():
        edges = [round(float(e), 4) for e in edges]
        missing = [e for e in edges if e not in present]
        if missing:
            print(f"  NOTE Figure 2: age bin(s) {missing} for '{group}' not in the data "
                  f"(bins present: {sorted(present)}).")
        sub = pf[pf["_bin"].isin(edges)]
        if sub.empty:
            continue
        a = sub.groupby(["sim_id", "scenario", "pair_id", "Year"], as_index=False).agg(
            _wp=("_wp", "sum"), _w=("_w", "sum"))
        a = a[a["_w"] > 0]
        a["pfpr_pct"] = a["_wp"] / a["_w"] * scale
        a["age_group"] = group
        pieces.append(a)
    if not pieces:
        raise ValueError("no age group could be built from the PfPR csv")
    return pd.concat(pieces, ignore_index=True)


def figure2(pf, out_dir):
    g = grouped_pfpr(pf)
    names = ordered(g["scenario"].unique())
    styles = build_styles(names)
    groups = [k for k in AGE_GROUPS if k in set(g["age_group"])]
    years = sorted(g["Year"].unique())

    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4.6),
                             squeeze=False)
    for k, (ax, grp) in enumerate(zip(axes[0], groups)):
        s = summarise(g[g["age_group"] == grp], ["scenario", "Year"], "pfpr_pct")
        for n in names:
            d = s[s["scenario"] == n].sort_values("Year")
            if d.empty:
                continue
            ax.plot(d["Year"], d["median"], label=display(n), **styles[n])
            if SHOW_RANGE:
                ax.fill_between(d["Year"], d["lo"], d["hi"],
                                color=styles[n]["color"], alpha=0.15, linewidth=0)
        mark_switch(ax, label=(k == 0))
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("PfPR (%)" if k == 0 else "")
        ax.set_title(grp, fontsize=11)
        ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "abcdef"[k])
    add_figure_legend(fig, axes[0][0], ncol=4, offset=-0.10)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure2_AnnualPfPR_byAgeGroup",
                "Annual PfPR by Scenario and Age Group")


# =============================================================================
# FIGURE 3: PAIRED CHANGE IN INCIDENCE RELATIVE TO IRS BENCHMARK
# =============================================================================

def figure3(paired, out_dir, window):
    d = paired[(paired["metric"] == PRIMARY_METRIC) & (paired["scenario_kind"] == "ig2")]
    refs = [r for r in REFERENCE_ORDER if r in set(d["reference"])]
    if not refs:
        raise ValueError("no IG2-vs-IRS paired rows in the paired-impact csv")
    d = d.assign(rel_pct=d["rel_change"] * 100.0)
    s = summarise(d, ["reference", "ig2_coverage"], "rel_pct")
    colors = ig2_bar_colors(s["ig2_coverage"].unique())

    fig, axes = plt.subplots(1, len(refs), figsize=(4.3 * len(refs), 4.8),
                             squeeze=False, sharey=True)
    big = s["median"].abs().max() > 300
    for k, (ax, ref) in enumerate(zip(axes[0], refs)):
        r = s[s["reference"] == ref].sort_values("ig2_coverage")
        x = np.arange(len(r))
        ax.bar(x, r["median"], width=0.6,
               color=[colors[c] for c in r["ig2_coverage"]], edgecolor="0.2",
               linewidth=0.6)
        if SHOW_RANGE:
            ax.errorbar(x, r["median"],
                        yerr=[np.clip(r["median"] - r["lo"], 0, None),
                              np.clip(r["hi"] - r["median"], 0, None)],
                        fmt="none", ecolor="0.2", capsize=3, linewidth=1)
        for xi, v in zip(x, r["median"]):
            ax.annotate(f"{v:+,.0f}%", xy=(xi, v), xytext=(0, 4 if v >= 0 else -4),
                        textcoords="offset points", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c * 100:.0f}" for c in r["ig2_coverage"]])
        ax.set_xlabel("IG2 distribution coverage (%)")
        ax.set_title(f"Relative to {display(ref)}", fontsize=11)
        if big:
            ax.set_yscale("symlog", linthresh=100)
            ax.set_ylim(bottom=-100)
        if k == 0:
            ax.set_ylabel(f"Paired change in clinical incidence, {window} (%)")
        tidy(ax)
        panel_letter(ax, "abc"[k])
    fig.text(0.5, -0.03,
             "Above zero: higher incidence than the IRS benchmark. "
             "Below zero: lower incidence. Bars show the median across paired "
             "pickup states."
             + (" Vertical axis is symmetric-log." if big else ""),
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure3_PairedChange_vs_IRSBenchmark",
                "Paired Change in Clinical Incidence Relative to IRS Benchmark")


# =============================================================================
# FIGURE 4: IG2 DISTRIBUTION COVERAGE VERSUS INCIDENCE AND PfPR
# =============================================================================

def figure4(win, out_dir, window):
    pf_col = "U5_pfpr" if "U5_pfpr" in win.columns else "pfpr_all_ages"
    pf_label = ("PfPR, children under 5 years (%)" if pf_col == "U5_pfpr"
                else "PfPR, all ages (%)")
    scale = 100.0 if win[pf_col].max() <= 1.0 + 1e-9 else 1.0
    win = win.assign(pfpr_pct=win[pf_col] * scale)

    names = ordered(win["scenario"].unique())
    styles = build_styles(names)
    ig2 = [n for n in names if n.startswith("IG2_")]
    if not ig2:
        raise ValueError("no IG2 scenarios in the window-metrics csv")
    cov = win[win["scenario"].isin(ig2)].groupby("scenario")["ig2_coverage"].first()

    panels = [(PRIMARY_METRIC, "Clinical incidence (cases per 1,000 person-years)"),
              ("pfpr_pct", pf_label)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for k, (ax, (col, ylabel)) in enumerate(zip(axes, panels)):
        s = summarise(win, ["scenario"], col).set_index("scenario")
        # reference levels: IRS arms and the baseline
        for n in names:
            if n in ig2:
                continue
            ax.axhline(s.loc[n, "median"], color=styles[n]["color"],
                       linestyle=styles[n]["linestyle"], linewidth=1.8,
                       label=display(n))
        x = np.array([cov[n] * 100 for n in ig2])
        y = np.array([s.loc[n, "median"] for n in ig2])
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=plt.cm.Blues(0.85), marker="s",
                linewidth=2.2, markersize=6, label="IG2 (by distribution coverage)")
        if SHOW_RANGE:
            lo = np.array([s.loc[n, "lo"] for n in ig2])
            hi = np.array([s.loc[n, "hi"] for n in ig2])
            ax.errorbar(x[order], y[order],
                        yerr=[np.clip(y - lo, 0, None)[order],
                              np.clip(hi - y, 0, None)[order]],
                        fmt="none", ecolor=plt.cm.Blues(0.85), capsize=3)
        auto_scale(ax, list(s["median"]))
        ax.set_xticks(sorted(x))
        ax.set_xlim(min(x) - 8, max(x) + 8)
        ax.set_xlabel("IG2 distribution coverage (%)")
        ax.set_ylabel(f"{ylabel}, {window}")
        if ax.get_yscale() != "log":
            ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "ab"[k])
    add_figure_legend(fig, axes[0], ncol=5, offset=-0.09)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure4_IG2Coverage_vs_Incidence_PfPR",
                "IG2 Distribution Coverage versus Clinical Incidence and PfPR")


# =============================================================================
# FIGURE 5: ANNUAL CASES AVERTED, IRS vs IG2 AT MATCHED COVERAGE
# =============================================================================

def figure5(annual, out_dir, population):
    """
    Cases averted = baseline incidence minus the arm's incidence, within each
    pickup state and year, then the median across pickup states.
    One panel per coverage in COMPARE_COVERAGES, IRS and IG2 side by side.
    """
    piv = annual.pivot_table(index=["pair_id", "Year"], columns="scenario",
                             values=PRIMARY_METRIC)
    all_names = list(piv.columns)
    if NO_ARM not in all_names:
        raise ValueError(f"{NO_ARM} is not in the annual csv; rerun the scenarios "
                         f"with INCLUDE_NO_IRS_ARM = True.")

    factor = (population / 1000.0) if population else 1.0
    if population:
        ylabel = f"Clinical cases averted per {population:,} population"
    else:
        ylabel = "Clinical cases averted per 1,000 person-years"

    years = [int(y) for y in sorted(piv.index.get_level_values("Year").unique())
             if y >= IRS_SWITCH_YEAR]
    if not years:
        raise ValueError(f"no years on or after {IRS_SWITCH_YEAR} in the annual csv")
    year_mask = piv.index.get_level_values("Year") >= IRS_SWITCH_YEAR

    fig, axes = plt.subplots(1, len(COMPARE_COVERAGES),
                             figsize=(5.4 * len(COMPARE_COVERAGES), 4.8),
                             squeeze=False, sharey=FIG5_SHARE_Y)
    rows = []

    for k, (ax, c) in enumerate(zip(axes[0], COMPARE_COVERAGES)):
        arms = []
        irs_name = find_arm(all_names, f"IRS_{c}")
        ig2_name = find_arm(all_names, f"IG2_{c}")
        if irs_name:
            arms.append(("IRS", irs_name, "IRS "))
        if ig2_name:
            arms.append(("IG2", ig2_name, "IG2 "))
        if not (irs_name and ig2_name):
            print(f"  WARNING Figure 5: panel {c}% is missing "
                  f"{'IRS' if not irs_name else 'IG2'} arm.")

        width = 0.8 / max(len(arms), 1)
        cum_text = []
        for j, (role, name, label) in enumerate(arms):
            averted = ((piv[NO_ARM] - piv[name]) * factor)[year_mask]   # baseline - arm
            grouped = averted.groupby(level="Year")
            annual_med = grouped.median()
            cum = float(averted.groupby(level="pair_id").sum().median())

            xs = np.arange(len(years)) + (j - (len(arms) - 1) / 2) * width
            heights = np.array([annual_med.loc[y] for y in years])
            ax.bar(xs, heights, width=width * 0.95, color=ROLE_COLOR[role],
                   edgecolor="0.2", linewidth=0.5, label=label)
            if SHOW_RANGE:
                q_lo = np.array([grouped.quantile(LOWER_Q).loc[y] for y in years])
                q_hi = np.array([grouped.quantile(HIGH_Q).loc[y] for y in years])
                ax.errorbar(xs, heights,
                            yerr=[np.clip(heights - q_lo, 0, None),
                                  np.clip(q_hi - heights, 0, None)],
                            fmt="none", ecolor="0.2", capsize=2, linewidth=0.8)

            for y, h in zip(years, heights):
                rows.append(dict(coverage_pct=c, arm=role, scenario=name,
                                 baseline=NO_ARM, period=str(y),
                                 median_cases_averted=h))
            rows.append(dict(coverage_pct=c, arm=role, scenario=name,
                             baseline=NO_ARM,
                             period=f"cumulative {years[0]}-{years[-1]}",
                             median_cases_averted=cum))
            cum_text.append((role, cum))

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Year")
        if k == 0 or not FIG5_SHARE_Y:
            ax.set_ylabel(ylabel)
        ax.set_title(f"{c}% coverage", fontsize=11)
        if cum_text:
            unit = "cases" if population else "cases/1,000 py"
            txt = f"Cumulative {years[0]}\u2013{years[-1]}:\n" + "\n".join(
                f"{r}: {v:,.0f} {unit}" for r, v in cum_text)
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=8, color="0.25")
        tidy(ax)
        panel_letter(ax, "abcdefgh"[k])

    add_figure_legend(fig, axes[0][0], ncol=2, offset=-0.08)
    fig.text(0.5, -0.13,
             f"Cases averted are measured against the {BASELINE_LABEL.lower()} "
             "(no IRS, no IG2). Above zero: cases averted. Below zero: additional "
             "cases. Bars show the median across paired pickup states.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure5_CasesAverted_IRSvsIG2",
                "Annual Clinical Cases Averted: IRS vs IG2 at Matched Coverage")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Figure5_cases_summary.csv"),
                              index=False)


# =============================================================================
# FIGURE 6: IG2 DISTRIBUTION COVERAGE AND ESTIMATED EFFECTIVE USE
# =============================================================================

def figure6(eff, out_dir):
    eff = eff.sort_values("ig2_coverage")
    x = eff["ig2_coverage"].to_numpy() * 100
    y_dist = eff["effective_use_at_distribution"].to_numpy() * 100
    y_avg = eff["effective_use_time_avg"].to_numpy() * 100

    retention = (float(eff["net_retention"].iloc[0])
                 if "net_retention" in eff.columns else np.nan)

    # Consistency check between this script's setting and the analyzer output.
    if not IG2_USE_CURVES and not np.allclose(y_dist, x, atol=0.5):
        print("  WARNING Figure 6: IG2_USE_CURVES is False here, but the effective-use "
              "csv has use at distribution different from c. Re-run the analyzer "
              "(stage 2) after setting IG2_USE_CURVES = False there.")
    if IG2_USE_CURVES and np.allclose(y_dist, x, atol=0.5):
        print("  WARNING Figure 6: IG2_USE_CURVES is True here, but the effective-use "
              "csv shows no usage-curve effect. Check the setting in the analyzer.")

    fig, ax = plt.subplots(figsize=(6.6, 6.0))

    if IG2_USE_CURVES:
        ax.plot([0, 100], [0, 100], color="0.55", linestyle=":", linewidth=1.3,
                label="Equal to distribution coverage")
        ax.plot(x, y_dist, color=plt.cm.Oranges(0.8), marker="o", linestyle="--",
                linewidth=2.0, label="At distribution (age and seasonal use applied)")
        for xi, v in zip(x, y_dist):
            ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9, color=plt.cm.Oranges(0.8))
        avg_label = "Time-averaged over the net cycle (net retention added)"
    else:
        # No usage curves: use at distribution equals distribution coverage,
        # so it is the diagonal itself.
        ax.plot([0, 100], [0, 100], color=plt.cm.Oranges(0.8), linestyle="--",
                linewidth=2.0,
                label="At distribution (equal to distribution coverage)")
        avg_label = "Time-averaged over the net cycle (net retention applied)"

    ax.fill_between(x, y_avg, x, color=plt.cm.Blues(0.3), alpha=0.35, linewidth=0)
    ax.plot(x, y_avg, color=plt.cm.Blues(0.9), marker="s", linewidth=2.2, label=avg_label)
    for xi, v in zip(x, y_avg):
        ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=9, color=plt.cm.Blues(0.9))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("IG2 distribution coverage (%)")
    ax.set_ylabel("Estimated average effective-use coverage (%)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    tidy(ax)

    if IG2_USE_CURVES:
        factors = ""
        if {"net_retention", "age_multiplier", "season_multiplier"} <= set(eff.columns):
            f = eff.iloc[0]
            factors = (f"Time-averaged effective use = distribution coverage \u00d7 "
                       f"net retention ({f['net_retention']:.2f}) \u00d7 age multiplier "
                       f"({f['age_multiplier']:.2f}) \u00d7 season multiplier "
                       f"({f['season_multiplier']:.2f}). ")
        note = (factors + "Estimated effective-use coverage is calculated analytically "
                "from IG2 distribution coverage, assumed net retention, age-specific "
                "usage and seasonal usage. It is not a directly observed measure of "
                "individual net use.")
    else:
        ret_txt = f" ({retention:.2f})" if np.isfinite(retention) else ""
        note = ("Age and seasonal usage curves are not applied to IG2 nets, so use at "
                "distribution equals distribution coverage. Time-averaged effective use "
                f"= distribution coverage \u00d7 net retention{ret_txt}. This is "
                "calculated analytically from the assumed net retention "
                "directly observed measure of individual net use.")
    note = "\n".join(textwrap.wrap(note, 90))
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, style="italic")

    fig.tight_layout()
    save_figure(fig, out_dir, "Figure6_IG2Coverage_vs_EstimatedEffectiveUse",
                "IG2 Distribution Coverage and Estimated Effective-Use Coverage")


# =============================================================================
# CAPTIONS
# =============================================================================

def write_captions(out_dir, n_pairs, window, population):
    pair_txt = f"{n_pairs} paired pickup states" if n_pairs else "paired pickup states"
    pop_txt = (f"per {population:,} population" if population
               else "per 1,000 person-years")
    covs_txt = ", ".join(str(c) for c in COMPARE_COVERAGES[:-1])
    covs_txt = f"{covs_txt} and {COMPARE_COVERAGES[-1]}" if covs_txt else str(COMPARE_COVERAGES[0])

    if IG2_USE_CURVES:
        usage_txt = ("IG2 nets carry age-specific and seasonal usage curves, so the share "
                     "of people using a net is lower than the distribution coverage.")
        fig6_body = ("Estimated share of the population effectively using an IG2 net for "
                     "each distribution coverage. Effective use is estimated from the "
                     "model's net retention (Weibull discard), age-specific usage curve "
                     "and seasonal usage curve. It is not an observed measure of net use.")
    else:
        usage_txt = ("No age-specific or seasonal usage curves are applied to IG2 nets, so "
                     "use at distribution equals the distribution coverage; net loss over "
                     "time (Weibull discard) still reduces the share of people holding a net.")
        fig6_body = ("Estimated share of the population effectively using an IG2 net for "
                     "each distribution coverage. No age-specific or seasonal usage curves "
                     "are applied, so use at distribution equals distribution coverage "
                     "(dashed line). The time-averaged effective use over the net cycle "
                     "is the distribution coverage multiplied by the mean net retention "
                     "implied by the Weibull discard. It is a model-based estimate, not "
                     "an observed measure of net use.")

    items = [
        ("Figure 1. Annual clinical malaria incidence, IRS versus IG2 at matched coverage",
         f"Clinical incidence (cases per 1,000 person-years) for each year, median across "
         f"{pair_txt}. Each panel ({covs_txt}% coverage) compares IRS continuing at the "
         f"stated coverage from 1 January {IRS_SWITCH_YEAR} with IG2 distribution at the "
         f"same coverage, and both are compared with the baseline (no IRS, no IG2). Before "
         f"{IRS_SWITCH_YEAR} all arms follow the same path. IRS coverage is the share of "
         f"structures sprayed and IG2 coverage is the share of the population receiving a "
         f"net, so they are comparable scenario levels, not equivalent measures. "
         f"{usage_txt}"),
        ("Figure 2. Annual PfPR by scenario and age group",
         f"Annual Plasmodium falciparum parasite prevalence (PfPR, %) by age group, "
         f"median across {pair_txt}. PfPR is averaged over 30-day reporting intervals "
         f"within each year, weighted by population and interval length."),
        ("Figure 3. Paired change in clinical incidence relative to the IRS benchmark",
         f"Median paired relative change in clinical incidence ({window}) of IG2 "
         f"distribution at each coverage compared with IRS at the stated benchmark "
         f"coverage, computed within each pickup state. Positive values indicate higher "
         f"incidence than the benchmark. Relative changes are large when the benchmark "
         f"incidence is close to zero."),
        ("Figure 4. IG2 distribution coverage versus clinical incidence and PfPR",
         f"Median clinical incidence (a) and PfPR (b) over {window} for each IG2 "
         f"distribution coverage, with the IRS arms and the baseline (no IRS, no IG2) "
         f"shown as horizontal lines. IG2 coverage is the share of the population "
         f"receiving a net in each campaign; IRS coverage is the share of structures "
         f"sprayed, so the two are comparable scenario levels, not equivalent measures."),
        ("Figure 5. Annual clinical cases averted, IRS versus IG2 at matched coverage",
         f"Annual clinical cases averted ({pop_txt}) by IRS and by IG2 distribution at the "
         f"same coverage ({covs_txt}%), relative to the baseline (no IRS, no IG2), from "
         f"1 January {IRS_SWITCH_YEAR}. Cases averted are the baseline incidence minus the "
         f"arm's incidence, computed within each pickup state, shown as the median across "
         f"{pair_txt}. Positive values are cases averted; negative values are additional "
         f"cases. Cumulative totals are in Figure5_cases_summary.csv."),
        ("Figure 6. IG2 distribution coverage and estimated effective-use coverage",
         fig6_body),
    ]
    path = os.path.join(out_dir, "figure_captions.txt")
    with open(path, "w", encoding="utf-8") as f:
        for head, body in items:
            f.write(head + "\n" + "\n".join(textwrap.wrap(body, 100)) + "\n\n")
    print("  wrote figure_captions.txt")


# =============================================================================
# MAIN
# =============================================================================

def run(name, fn, *args):
    try:
        fn(*args)
    except Exception as exc:        # keep going so one missing csv does not stop the rest
        print(f"  FAILED {name}: {exc}")
        traceback.print_exc()


def parse_args():
    p = argparse.ArgumentParser(description="UWR publication figures from analyzer csvs.")
    p.add_argument("--input-dir", default=DEFAULT_DIR, help="analyzer output folder")
    p.add_argument("--output-dir", default=None,
                   help="default: <input-dir>/publication_figures")
    p.add_argument("--population", type=int, default=None,
                   help="Figure 5 in cases per this population (default: per 1,000 person-years)")
    p.add_argument("--window-start", type=int, default=IRS_SWITCH_YEAR,
                   help="first year of the window quoted in axis labels")
    p.add_argument("--show-range", action="store_true",
                   help=f"add {int(LOWER_Q * 100)}th-{int(HIGH_Q * 100)}th percentiles")
    p.add_argument("--no-titles", action="store_true", help="omit figure titles")
    return p.parse_args()


def main():
    global SHOW_RANGE, SHOW_TITLES
    args = parse_args()
    SHOW_RANGE = args.show_range
    SHOW_TITLES = not args.no_titles
    out_dir = args.output_dir or os.path.join(args.input_dir, "publication_figures")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 11,
        "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.9, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    print(f"Reading {args.input_dir}")
    annual = read_csv(args.input_dir, "Scenario_AnnualBurden")
    window = f"{args.window_start}\u2013{int(annual['Year'].max())}"
    n_pairs = annual["pair_id"].nunique() if "pair_id" in annual.columns else 0

    print("Writing figures:")
    run("Figure 1", figure1, annual, out_dir)

    pf = read_csv(args.input_dir, "PfPR_ClinicalIncidence_monthly", required=False)
    if pf is None:
        print("  skipped Figure 2: PfPR csv not found")
    else:
        run("Figure 2", figure2, pf, out_dir)

    paired = read_csv(args.input_dir, "Scenario_PairedImpact", required=False)
    if paired is None:
        print("  skipped Figure 3: paired-impact csv not found")
    else:
        run("Figure 3", figure3, paired, out_dir, window)

    win = read_csv(args.input_dir, "Scenario_WindowMetrics", required=False)
    if win is None:
        print("  skipped Figure 4: window-metrics csv not found")
    else:
        run("Figure 4", figure4, win, out_dir, window)

    run("Figure 5", figure5, annual, out_dir, args.population)

    eff = read_csv(args.input_dir, "IG2_EffectiveUse", required=False)
    if eff is None:
        print("  skipped Figure 6: effective-use csv not found")
    else:
        run("Figure 6", figure6, eff, out_dir)

    write_captions(out_dir, n_pairs, window, args.population)
    print(f"\nFigures written to {out_dir}")


if __name__ == "__main__":
    main()