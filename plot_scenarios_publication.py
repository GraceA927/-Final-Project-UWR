"""
Upper West Region malaria intervention scenarios: publication analysis

Reads the CSV outputs produced by analyze_scenarios_UWR.py and reports three
primary outcomes:
  1. Clinical malaria incidence (cases per 1,000 person-years)
  2. Plasmodium falciparum parasite prevalence (PfPR, %), overall and by age
  3. Clinical cases averted relative to a specified reference scenario

Two complementary scenario comparisons are supported:

  operational:
    IRS continuation at 95% and 100% structure coverage; replacement of IRS
    with IG2 nets distributed to 50%, 60%, 70%, 80%, 90%, 95%, and 100% of
    the population.

  matched:
    IRS and IG2 scenarios at the same nominal coverage levels (60%, 80%,
    and 90%). This is termed the “matched nominal-coverage comparison”.
    Matching the percentages does not make the coverage measures equivalent:
    IRS coverage is the proportion of structures sprayed, whereas IG2
    coverage is the proportion of the population receiving nets.

The two comparisons are analysed and saved separately. They share BAU as the
business-as-usual reference where that arm is present. The --reference IRS
option compares IG2 with IRS only at coverage levels available for both arms.

The plotting script does not calculate net retention, age-specific use,
seasonal use, or effective net use. Those assumptions must be implemented in
the simulation configuration; the present script analyses their resulting
modelled outcomes.

Example:
  python plot_scenarios_UWR_primary_outcomes.py --comparison both
  python plot_scenarios_UWR_primary_outcomes.py --comparison operational --reference BAU
  python plot_scenarios_UWR_primary_outcomes.py --comparison matched --reference IRS
  python plot_scenarios_UWR_primary_outcomes.py --comparison both --cm 5 --formats png pdf
"""

import argparse
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DEFAULT_IN_DIR = ("/shared/home/grace/Final-Project-UWR/experiments/"
                  "Pickup13yrs_Scenario_Final_AnalyzerOutput")
STEP = "scenarios"
BAU = "BAU"
IRS_SWITCH_YEAR = 2027
LOW_Q, HIGH_Q = 0.10, 0.90

COMPARISONS = {
    "operational": {
        "title": "Operational coverage scenarios",
        "irs_coverages": [95, 100],
        "ig2_coverages": [50, 60, 70, 80, 90, 95, 100],
    },
    "matched": {
        "title": "Matched nominal-coverage comparison",
        "irs_coverages": [60, 80, 90],
        "ig2_coverages": [60, 80, 90],
    },
}

AGE_GROUPS = [
    ("All ages", "pfpr_all", None),
    ("Under 5 years", "pfpr_u5", [0.25, 0.5, 5.0]),
    ("5–14 years", "pfpr_5_14", [15.0]),
    ("15 years and older", "pfpr_15p", [25.0, 100.0]),
]

FAMILY_COLORS = {"BAU": "#3b3a37", "IRS": "#eb6834", "IG2": "#2a78d6",
                 "NO_IRS": "#1baf7a", "OTHER": "#777777"}
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 300
LOG_RATIO = 30

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0, "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


# =============================================================================
# SCENARIO IDENTIFICATION
# =============================================================================

CM_RE = re.compile(r"_CM(\d+)$")
ARM_RE = re.compile(r"^(IRS|IG2)_(\d+)")


def parse_arm(name):
    name = str(name)
    m = CM_RE.search(name)
    cm = int(m.group(1)) if m else 0
    strategy = CM_RE.sub("", name)
    if strategy == BAU:
        family, coverage = "BAU", 0
    elif strategy == "NO_IRS":
        family, coverage = "NO_IRS", 0
    else:
        a = ARM_RE.match(strategy)
        family, coverage = (a.group(1), int(a.group(2))) if a else ("OTHER", 0)
    return {"scenario": name, "strategy": strategy, "family": family,
            "coverage": coverage, "cm": cm}


def arm_label(family, coverage=0, full=True):
    if family == "BAU":
        return "Business as usual (current IRS programme)"
    if family == "IRS":
        return f"Continue IRS ({coverage}% of structures sprayed)"
    if family == "IG2":
        return f"Replace IRS with IG2 ({coverage}% of population receiving nets)"
    if family == "NO_IRS":
        return "IRS withdrawn without replacement"
    return family


def read_csv(in_dir, prefix, cm, required=True):
    path = os.path.join(in_dir, f"{prefix}_{STEP}.csv")
    if not os.path.isfile(path):
        if required:
            raise SystemExit(f"Missing {path}. Run analyze_scenarios_UWR.py first.")
        return None
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].astype(str)
    if "pair_id" not in df.columns:
        if "pickup_sim_index" in df.columns:
            df["pair_id"] = df["pickup_sim_index"].astype(str)
        elif "sim_id" in df.columns:
            df["pair_id"] = df["sim_id"].astype(str)
        else:
            raise SystemExit(f"{path}: needs pair_id, pickup_sim_index, or sim_id.")
    df["pair_id"] = df["pair_id"].astype(str)
    arm_data = pd.DataFrame([parse_arm(s) for s in df["scenario"].unique()])
    df = df.drop(columns=[c for c in ("strategy", "family", "coverage", "cm")
                          if c in df.columns])
    df = df.merge(arm_data, on="scenario", how="left")
    return df[df["cm"] == cm].reset_index(drop=True)


def scenario_sets(annual, comparison):
    """Return actual scenario names selected for a comparison."""
    spec = COMPARISONS[comparison]
    present = {str(s): parse_arm(s) for s in annual["scenario"].unique()}
    by_key = {(a["family"], a["coverage"]): name for name, a in present.items()}
    selected = []
    if ("BAU", 0) in by_key:
        selected.append(by_key[("BAU", 0)])
    selected += [by_key[("IRS", c)] for c in spec["irs_coverages"]
                 if ("IRS", c) in by_key]
    selected += [by_key[("IG2", c)] for c in spec["ig2_coverages"]
                 if ("IG2", c) in by_key]
    return selected


def validate_scenarios(annual, selected, comparison):
    actual = set(annual["scenario"].unique())
    selected = [s for s in selected if s in actual]
    if not selected:
        raise SystemExit(f"No scenarios found for {comparison}. Available: {sorted(actual)}")
    missing = []
    spec = COMPARISONS[comparison]
    for c in spec["irs_coverages"]:
        if not any(parse_arm(s)["family"] == "IRS" and
                   parse_arm(s)["coverage"] == c for s in selected):
            missing.append(f"IRS_{c}")
    for c in spec["ig2_coverages"]:
        if not any(parse_arm(s)["family"] == "IG2" and
                   parse_arm(s)["coverage"] == c for s in selected):
            missing.append(f"IG2_{c}")
    if missing:
        print(f"  NOTE ({comparison}): scenario arms not present and therefore omitted: "
              + ", ".join(missing))
    return selected


def make_reference_map(selected, mode):
    """Map each arm to its reference. IRS reference only pairs equal coverages."""
    by_key = {(parse_arm(s)["family"], parse_arm(s)["coverage"]): s for s in selected}
    def ref_of(s):
        a = parse_arm(s)
        if mode == "BAU":
            return by_key.get(("BAU", 0)) if a["family"] != "BAU" else None
        if mode == "IRS":
            return by_key.get(("IRS", a["coverage"])) if a["family"] == "IG2" else None
        return None
    return ref_of


# =============================================================================
# OUTCOME SUMMARIES
# =============================================================================

def window_incidence(annual, years):
    a = annual[annual["Year"].isin(years)].copy()
    a["py"] = a["population"] * a["days_simulated"] / 365.0
    a["cases"] = (a["clinical_cases"] if "clinical_cases" in a.columns else
                  a["incidence_per_1000_py"] * a["py"] / 1000.0)
    out = a.groupby(["scenario", "pair_id"], as_index=False).agg(
        cases=("cases", "sum"), py=("py", "sum"), n_years=("Year", "nunique"))
    out["incidence"] = out["cases"] / out["py"] * 1000.0
    return out


def grouped_pfpr(pf):
    pf = pf.copy()
    pf["_bin"] = pf["agebin"].round(4)
    scale = 100.0 if pf["PfPR"].max() <= 1.0 + 1e-9 else 1.0
    pf["_w"] = pf["Pop"] * pf["interval_days"]
    pf["_wp"] = pf["PfPR"] * pf["_w"]
    present = set(pf["_bin"].unique())
    pieces = []
    for name, key, edges in AGE_GROUPS:
        sub = pf if edges is None else pf[pf["_bin"].isin([round(float(x), 4)
                                                            for x in edges])]
        missing = ([] if edges is None else
                   [x for x in edges if round(float(x), 4) not in present])
        if missing:
            print(f"  NOTE: PfPR age bins missing for {name}: {missing}")
        if sub.empty:
            continue
        g = sub.groupby(["scenario", "pair_id", "Year"], as_index=False).agg(
            wp=("_wp", "sum"), w=("_w", "sum"))
        g = g[g["w"] > 0].copy()
        g["pfpr"] = g["wp"] / g["w"] * scale
        g["key"], g["name"], g["scale"] = key, name, scale
        pieces.append(g)
    if not pieces:
        raise SystemExit("No PfPR age group could be constructed from the PfPR CSV.")
    return pd.concat(pieces, ignore_index=True)


def window_pfpr(pfg, years):
    g = pfg[pfg["Year"].isin(years)].groupby(
        ["scenario", "pair_id", "key"], as_index=False).agg(
            wp=("wp", "sum"), w=("w", "sum"), scale=("scale", "first"))
    g["value"] = g["wp"] / g["w"] * g["scale"]
    return g.pivot(index=["scenario", "pair_id"], columns="key",
                   values="value").reset_index()


def quantile(x, p):
    x = pd.Series(x).dropna()
    return float(x.quantile(p)) if len(x) else np.nan


def time_summary(df, value):
    g = df.groupby(["scenario", "Year"])[value]
    return pd.DataFrame({"median": g.median(), "low": g.quantile(LOW_Q),
                         "high": g.quantile(HIGH_Q)}).reset_index()


def window_summary(win, column):
    g = win.groupby("scenario")[column]
    return pd.DataFrame({"median": g.median(), "low": g.quantile(LOW_Q),
                         "high": g.quantile(HIGH_Q)})


def cases_averted_annual(annual, selected, ref_of, years, factor):
    d = annual[(annual["Year"].isin(years)) &
               (annual["scenario"].isin(selected))]
    piv = d.pivot_table(index=["pair_id", "Year"], columns="scenario",
                        values="incidence_per_1000_py")
    rows = []
    for s in piv.columns:
        ref = ref_of(s)
        if ref is None or ref not in piv.columns:
            continue
        delta = (piv[ref] - piv[s]).dropna() * factor
        for (pair_id, year), value in delta.items():
            rows.append({"pair_id": pair_id, "Year": year, "scenario": s,
                         "reference": ref, "cases_averted": value})
    return pd.DataFrame(rows, columns=["pair_id", "Year", "scenario",
                                       "reference", "cases_averted"])


def paired_window_reduction(win, selected, ref_of):
    d = win[win["scenario"].isin(selected)]
    piv = d.pivot_table(index="pair_id", columns="scenario", values="incidence")
    rows = []
    for s in piv.columns:
        ref = ref_of(s)
        if ref is None or ref not in piv.columns:
            continue
        delta = (piv[ref] - piv[s]).dropna()
        for pair_id, value in delta.items():
            baseline = piv.loc[pair_id, ref]
            rows.append({"pair_id": pair_id, "scenario": s,
                         "cases_rate_difference": value,
                         "relative_reduction_pct": value / baseline * 100.0
                         if baseline else np.nan})
    return pd.DataFrame(rows)


def cumulative_cases(av, n_years):
    if av.empty:
        return pd.DataFrame(columns=["scenario", "pair_id", "cumulative"])
    g = av.groupby(["scenario", "pair_id"])["cases_averted"].agg(
        total="sum", n="count").reset_index()
    return g[g["n"] == n_years].rename(columns={"total": "cumulative"})[
        ["scenario", "pair_id", "cumulative"]]


# =============================================================================
# PUBLICATION PLOT HELPERS
# =============================================================================

def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def scenario_color(s):
    a = parse_arm(s)
    return FAMILY_COLORS.get(a["family"], FAMILY_COLORS["OTHER"])


def scenario_style(s):
    a = parse_arm(s)
    marker = {"BAU": "^", "IRS": "o", "IG2": "s", "NO_IRS": "v"}.get(
        a["family"], "D")
    return marker


def scenario_legend(selected):
    """One legend entry per intervention family; coverage is shown in panel titles."""
    families = []
    for s in selected:
        family = parse_arm(s)["family"]
        if family not in families and family in ("BAU", "IRS", "IG2"):
            families.append(family)
    labels = {
        "BAU": "Business as usual",
        "IRS": "Continue IRS",
        "IG2": "Replace IRS with IG2 nets",
    }
    markers = {"BAU": "^", "IRS": "o", "IG2": "s"}
    handles = []
    for family in families:
        handles.append(Line2D([], [], color=FAMILY_COLORS[family],
                              marker=markers[family], lw=1.8, ms=5,
                              label=labels[family]))
    return handles


def mark_switch(ax, label=False):
    ax.axvline(IRS_SWITCH_YEAR - 0.5, color=TEXT_2, lw=0.8, ls="--", zorder=1)
    if label:
        ax.annotate("Intervention scenarios begin", xy=(IRS_SWITCH_YEAR - 0.5, 1),
                    xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=7.5,
                    color=TEXT_2)


def add_time_lines(ax, summary, selected, show_range):
    vals = []
    for s in selected:
        d = summary[summary["scenario"] == s].sort_values("Year")
        if d.empty:
            continue
        color, marker = scenario_color(s), scenario_style(s)
        ax.plot(d["Year"], d["median"], color=color, marker=marker,
                lw=1.8, ms=4, label=arm_label(parse_arm(s)["family"],
                                               parse_arm(s)["coverage"]))
        if show_range:
            ax.fill_between(d["Year"], d["low"], d["high"], color=color,
                            alpha=0.10, linewidth=0)
        vals.extend(d["median"].dropna().tolist())
    return vals


def save_figure(fig, out, stem, formats):
    for ext in formats:
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=DPI)
    plt.close(fig)
    print(f"  wrote {stem}")


def figure1(annual, selected, ctx):
    """Clinical incidence: one panel per nominal coverage, as in the supplied style."""
    summary = time_summary(annual[annual["scenario"].isin(selected)],
                           "incidence_per_1000_py")
    coverages = ctx["coverages"]
    ncols = min(3, len(coverages))
    nrows = int(np.ceil(len(coverages) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.7 * nrows),
                             sharey=True, squeeze=False)
    for i, coverage in enumerate(coverages):
        ax = axes.ravel()[i]
        panel = [s for s in selected
                 if parse_arm(s)["family"] == "BAU"]
        panel += [s for s in selected
                  if parse_arm(s)["family"] in ("IRS", "IG2")
                  and parse_arm(s)["coverage"] == coverage]
        vals = add_time_lines(ax, summary, panel, ctx["show_range"])
        mark_switch(ax, i == 0)
        ax.set_xticks(sorted(summary["Year"].unique()))
        ax.set_xlabel("Year")
        ax.set_title(f"{coverage}% nominal coverage")
        if vals and min(vals) > 0 and max(vals) / min(vals) > LOG_RATIO:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        style_axis(ax)
        ax.text(-0.02, 1.04, "abcdefghijklmnopqrstuvwxyz"[i],
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                ha="right", va="bottom", color=TEXT)
    for ax in axes.ravel()[len(coverages):]:
        ax.axis("off")
    axes.ravel()[0].set_ylabel(
        "Clinical incidence\n(cases per 1,000 person-years)")
    handles = scenario_legend([s for s in selected
                               if parse_arm(s)["family"] in ("BAU", "IRS", "IG2")])
    fig.legend(handles=handles, loc="lower center",
               ncol=min(4, max(1, len(handles))), frameon=False,
               bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout()
    save_figure(fig, ctx["out"], "Fig1_ClinicalIncidence_TimeSeries", ctx["formats"])

def figure2(pfg, selected, ctx):
    """PfPR time series: rows are age groups and columns are coverage levels."""
    groups = [(name, key) for name, key, _ in AGE_GROUPS
              if key in set(pfg["key"])]
    coverages = ctx["coverages"]
    years = sorted(pfg["Year"].unique())
    fig, axes = plt.subplots(len(groups), len(coverages),
                             figsize=(3.8 * len(coverages), 2.7 * len(groups)),
                             sharex=True, squeeze=False)
    panel_index = 0
    for i, (name, key) in enumerate(groups):
        summary = time_summary(
            pfg[(pfg["key"] == key) & (pfg["scenario"].isin(selected))], "pfpr")
        for j, coverage in enumerate(coverages):
            ax = axes[i, j]
            panel = [s for s in selected if parse_arm(s)["family"] == "BAU"]
            panel += [s for s in selected
                      if parse_arm(s)["family"] in ("IRS", "IG2")
                      and parse_arm(s)["coverage"] == coverage]
            add_time_lines(ax, summary, panel, ctx["show_range"])
            mark_switch(ax, panel_index == 0)
            ax.set_ylim(bottom=0)
            style_axis(ax)
            if i == 0:
                ax.set_title(f"{coverage}% nominal coverage")
            if j == 0:
                ax.set_ylabel(f"{name}\nPfPR (%)")
            if i == len(groups) - 1:
                ax.set_xticks(years)
                ax.set_xlabel("Year")
            ax.text(-0.02, 1.04, "abcdefghijklmnopqrstuvwxyz"[panel_index],
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    ha="right", va="bottom", color=TEXT)
            panel_index += 1
    handles = scenario_legend([s for s in selected
                               if parse_arm(s)["family"] in ("BAU", "IRS", "IG2")])
    fig.legend(handles=handles, loc="lower center",
               ncol=min(4, max(1, len(handles))), frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout()
    save_figure(fig, ctx["out"], "Fig2_PfPR_byAgeGroup_TimeSeries", ctx["formats"])

def figure3(win, selected, ctx):
    columns = [("incidence", "Clinical incidence\n(cases per 1,000 person-years)")]
    columns += [(key, f"PfPR, {name.lower()} (%)") for name, key, _ in AGE_GROUPS
                if key in win.columns and win[key].notna().any()]
    ncols = 2
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0, 3.8 * nrows), squeeze=False)
    for k, (col, ylabel) in enumerate(columns):
        ax = axes.ravel()[k]
        stats = window_summary(win[win["scenario"].isin(selected)], col)
        for s in selected:
            if s not in stats.index:
                continue
            a = parse_arm(s)
            x = a["coverage"] if a["family"] in ("IRS", "IG2") else np.nan
            if not np.isfinite(x):
                continue
            # Small horizontal offsets distinguish the two intervention families.
            x_plot = x + (-1.2 if a["family"] == "IRS" else 1.2)
            r = stats.loc[s]
            ax.errorbar(x_plot, r["median"],
                        yerr=[[max(r["median"] - r["low"], 0)],
                              [max(r["high"] - r["median"], 0)]],
                        color=scenario_color(s), marker=scenario_style(s), ms=6,
                        lw=1.4, capsize=3, linestyle="none", zorder=3)
        ax.set_xlabel("Nominal coverage level (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({parse_arm(s)["coverage"] for s in selected
                              if parse_arm(s)["family"] in ("IRS", "IG2")}))
        ax.set_xlim(45, 105)
        if col == "incidence":
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(bottom=0)
        style_axis(ax)
        ax.text(0.02, 0.96, "IRS: structures sprayed\nIG2: population receiving nets",
                transform=ax.transAxes, va="top", fontsize=7.5, color=TEXT_2)
        family_handles = []
        for family, label in (("IRS", "IRS continuation"), ("IG2", "IG2 replacement")):
            if any(parse_arm(s)["family"] == family for s in selected):
                family_handles.append(Line2D([], [], color=FAMILY_COLORS[family],
                                             marker={"IRS": "o", "IG2": "s"}[family],
                                             lw=1.5, label=label))
        ax.legend(handles=family_handles, frameon=False, fontsize=7.5, loc="best")
    for ax in axes.ravel()[len(columns):]:
        ax.axis("off")
    fig.suptitle(f"Modelled outcomes across {ctx['comparison_title'].lower()}\n"
                 f"Analysis period: {ctx['win_txt']}", y=1.02, fontsize=11)
    fig.text(0.5, -0.005,
             "Points show medians across paired pickup states; error bars show the "
             "10th–90th percentiles. IRS and IG2 percentages describe different "
             "coverage denominators and should not be interpreted as equivalent exposure.",
             ha="center", va="top", fontsize=8, style="italic")
    fig.tight_layout()
    save_figure(fig, ctx["out"], "Fig3_CoverageResponse_Incidence_PfPR", ctx["formats"])


def figure4(av, selected, ctx):
    """Annual cases averted: one panel per coverage level."""
    if av.empty:
        print("  Fig4 skipped: no selected arm has the requested reference comparison.")
        return
    summary = av.groupby(["scenario", "Year"])["cases_averted"].agg(
        median="median", low=lambda x: quantile(x, LOW_Q),
        high=lambda x: quantile(x, HIGH_Q)).reset_index()
    coverages = ctx["coverages"]
    ncols = min(3, len(coverages))
    nrows = int(np.ceil(len(coverages) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 3.7 * nrows),
                             sharey=True, squeeze=False)
    years = sorted(summary["Year"].unique())
    for i, coverage in enumerate(coverages):
        ax = axes.ravel()[i]
        members = [s for s in selected
                   if parse_arm(s)["family"] in ("IRS", "IG2")
                   and parse_arm(s)["coverage"] == coverage
                   and s in set(summary["scenario"])]
        width = 0.8 / max(len(members), 1)
        for j, scen in enumerate(members):
            fam = parse_arm(scen)["family"]
            d = summary[summary["scenario"] == scen].set_index("Year").reindex(years)
            x = np.arange(len(years)) + (j - (len(members) - 1) / 2) * width
            ax.bar(x, d["median"], width=width * 0.92,
                   color=FAMILY_COLORS[fam], edgecolor="0.2",
                   linewidth=0.5, zorder=2)
            ax.errorbar(x, d["median"],
                        yerr=[np.clip(d["median"] - d["low"], 0, None),
                              np.clip(d["high"] - d["median"], 0, None)],
                        fmt="none", ecolor="0.15", capsize=3,
                        elinewidth=1.0, zorder=3)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel("Year")
        ax.set_title(f"{coverage}% nominal coverage")
        style_axis(ax)
        ax.text(-0.02, 1.04, "abcdefghijklmnopqrstuvwxyz"[i],
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                ha="right", va="bottom", color=TEXT)
    for ax in axes.ravel()[len(coverages):]:
        ax.axis("off")
    axes.ravel()[0].set_ylabel(
        f"Cases averted vs {ctx['reference_label']}\n({ctx['unit_year']})")
    fams = []
    for fam, label in (("IRS", "Continue IRS"), ("IG2", "Replace IRS with IG2 nets")):
        if any(parse_arm(s)["family"] == fam for s in selected):
            fams.append(Line2D([], [], color=FAMILY_COLORS[fam], marker="o",
                               lw=1.5, label=label))
    fig.legend(handles=fams, loc="lower center", ncol=max(1, len(fams)),
               frameon=False, bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout()
    save_figure(fig, ctx["out"], "Fig4_CasesAverted_Annual", ctx["formats"])

def figure5(av, rel, selected, ctx):
    if av.empty:
        print("  Fig5 skipped: no selected arm has the requested reference comparison.")
        return
    n_years = len(ctx["years"])
    cum = av.groupby(["scenario", "pair_id"])["cases_averted"].agg(
        total="sum", n="count").reset_index()
    cum = cum[cum["n"] == n_years].rename(columns={"total": "cases_averted"})
    cstats = cum.groupby("scenario")["cases_averted"].agg(
        median="median", low=lambda x: quantile(x, LOW_Q),
        high=lambda x: quantile(x, HIGH_Q))
    rstats = rel.groupby("scenario")["relative_reduction_pct"].agg(
        median="median", low=lambda x: quantile(x, LOW_Q),
        high=lambda x: quantile(x, HIGH_Q)) if not rel.empty else pd.DataFrame()
    scenarios = [s for s in selected if s in cstats.index]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    panels = [(axes[0], cstats, "Cumulative clinical cases averted", ctx["unit_cum"]),
              (axes[1], rstats, "Reduction in clinical incidence", "%")]
    for ax, stats, title, unit in panels:
        if stats.empty:
            ax.axis("off")
            continue
        for i, s in enumerate(scenarios):
            if s not in stats.index:
                continue
            r = stats.loc[s]
            ax.bar(i, r["median"], color=scenario_color(s), edgecolor="0.2",
                   linewidth=0.4)
            ax.errorbar(i, r["median"],
                        yerr=[[max(r["median"] - r["low"], 0)],
                              [max(r["high"] - r["median"], 0)]],
                        fmt="none", ecolor="0.15", capsize=2)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([f"{parse_arm(s)['family']}\n{parse_arm(s)['coverage']}%"
                            if parse_arm(s)["family"] != "BAU" else "BAU"
                            for s in scenarios])
        ax.set_ylabel(unit)
        ax.set_title(title)
        style_axis(ax)
    fig.suptitle(f"Cumulative impact over {ctx['win_txt']} vs {ctx['reference_label']}",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, ctx["out"], "Fig5_CasesAverted_Cumulative", ctx["formats"])


# =============================================================================
# TABLES AND CAPTIONS
# =============================================================================

def format_ci(m, lo, hi, fmt="{:.1f}"):
    if pd.isna(m):
        return "—"
    return f"{fmt.format(m)} ({fmt.format(lo)} to {fmt.format(hi)})"


def write_table(win, av, rel, selected, ctx):
    metrics = [("incidence", "Clinical incidence (per 1,000 person-years)", "{:.1f}")]
    metrics += [(key, f"PfPR, {name.lower()} (%)", "{:.1f}")
                for name, key, _ in AGE_GROUPS if key in win.columns]
    summaries = {key: window_summary(win[win["scenario"].isin(selected)], key)
                 for key, _, _ in metrics}
    cum = (av.groupby(["scenario", "pair_id"])["cases_averted"].sum().reset_index()
           if not av.empty else pd.DataFrame())
    cstats = cum.groupby("scenario")["cases_averted"].agg(
        median="median", low=lambda x: quantile(x, LOW_Q),
        high=lambda x: quantile(x, HIGH_Q)) if not cum.empty else pd.DataFrame()
    rstats = rel.groupby("scenario")["relative_reduction_pct"].agg(
        median="median", low=lambda x: quantile(x, LOW_Q),
        high=lambda x: quantile(x, HIGH_Q)) if not rel.empty else pd.DataFrame()
    rows = []
    for s in selected:
        a = parse_arm(s)
        row = {"Scenario": arm_label(a["family"], a["coverage"])}
        for key, label, fmt in metrics:
            stat = summaries[key]
            row[label] = (format_ci(*stat.loc[s, ["median", "low", "high"]], fmt)
                          if s in stat.index else "—")
        if s in cstats.index:
            row[f"Cumulative cases averted ({ctx['unit_cum']})"] = format_ci(
                *cstats.loc[s, ["median", "low", "high"]], "{:,.0f}")
        else:
            row[f"Cumulative cases averted ({ctx['unit_cum']})"] = "Not estimated"
        if s in rstats.index:
            row["Incidence reduction (%)"] = format_ci(
                *rstats.loc[s, ["median", "low", "high"]], "{:.1f}")
        else:
            row["Incidence reduction (%)"] = "Not estimated"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(ctx["out"], "Table1_Scenario_outcomes.csv"), index=False)
    table.to_markdown(os.path.join(ctx["out"], "Table1_Scenario_outcomes.md"), index=False)
    print("  wrote Table1_Scenario_outcomes.{csv,md}")


def write_captions(ctx):
    text = f"""# Draft figure captions

**Figure 1. Clinical malaria incidence under alternative intervention scenarios.** Annual clinical malaria incidence (cases per 1,000 person-years), summarised by the median across paired pickup states. The intervention scenarios begin on 1 January {IRS_SWITCH_YEAR}. The {ctx['comparison_title'].lower()} are shown. Shaded bands, where requested, represent the 10th–90th percentile range across pickup states.

**Figure 2. *Plasmodium falciparum* parasite prevalence by age group.** Annual PfPR (%) for all ages, children under 5 years, people aged 5–14 years, and people aged 15 years and older. Estimates are weighted by population and reporting-interval length and summarised across paired pickup states.

**Figure 3. Modelled clinical incidence and PfPR across coverage levels.** Window-average outcomes over {ctx['win_txt']}. Points represent medians and error bars the 10th–90th percentiles across pickup states. IRS coverage denotes the proportion of structures sprayed; IG2 coverage denotes the proportion of the population receiving nets. These coverage measures have different denominators and are not directly equivalent.

**Figure 4. Annual clinical cases averted.** Annual cases averted relative to {ctx['reference_label']}, calculated as reference incidence minus scenario incidence within each paired pickup state. Positive values indicate fewer clinical cases; negative values indicate additional cases.

**Figure 5. Cumulative clinical cases averted and relative incidence reduction.** Cumulative cases averted and percentage reduction in clinical incidence over {ctx['win_txt']} relative to {ctx['reference_label']}. Estimates are paired by pickup state.

**Interpretation of net-use assumptions.** Age-specific use, seasonal use, net retention, and effective net use are determined in the upstream simulation configuration. This analysis script does not independently estimate these quantities; the plotted outcomes reflect the assumptions encoded in the simulation runs.
"""
    with open(os.path.join(ctx["out"], "Figure_captions.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("  wrote Figure_captions.md")


# =============================================================================
# ANALYSIS DRIVER
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Publication analysis of UWR IRS and IG2 malaria intervention scenarios.")
    p.add_argument("--in-dir", default=DEFAULT_IN_DIR, help="analyzer output directory")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: comparison-specific subdirectories)")
    p.add_argument("--comparison", choices=["operational", "matched", "both"],
                   default="both", help="scenario comparison(s) to analyse")
    p.add_argument("--reference", choices=["BAU", "IRS"], default="BAU",
                   help="reference for cases averted; IRS only pairs IG2 and IRS at equal nominal coverage")
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("START", "END"),
                   help=f"analysis window (default {IRS_SWITCH_YEAR} to final year)")
    p.add_argument("--cm", type=int, choices=[0, 5], default=0,
                   help="case-management variant: 0=current; 5=+5 percentage points")
    p.add_argument("--population", type=int, default=None,
                   help="scale cases averted to this population (default: per 1,000)")
    p.add_argument("--show-range", action="store_true",
                   help="show 10th–90th percentile bands on time-series plots")
    p.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"],
                   help="figure formats")
    return p.parse_args()


def run_comparison(annual_all, pfg_all, args, comparison, years, all_years):
    selected = scenario_sets(annual_all, comparison)
    selected = validate_scenarios(annual_all, selected, comparison)
    annual = annual_all[annual_all["scenario"].isin(selected)].copy()
    pfg = (pfg_all[pfg_all["scenario"].isin(selected)].copy()
           if pfg_all is not None else None)
    ref_of = make_reference_map(selected, args.reference)
    if args.reference == "BAU" and BAU not in selected:
        raise SystemExit("BAU reference requested, but BAU is absent from the selected scenarios.")
    if args.reference == "IRS" and not any(ref_of(s) for s in selected):
        raise SystemExit("IRS reference requested, but no IG2 arm has an IRS arm at the same nominal coverage.")

    if args.out_dir:
        out = args.out_dir if args.comparison != "both" else os.path.join(args.out_dir, comparison)
    else:
        out = os.path.join(args.in_dir, f"publication_{comparison}_vs_{args.reference.lower()}")
    os.makedirs(out, exist_ok=True)

    factor = args.population / 1000.0 if args.population else 1.0
    per = f"per {args.population:,} population" if args.population else "per 1,000 population"
    ctx = {
        "out": out, "formats": args.formats, "show_range": args.show_range,
        "comparison": comparison, "comparison_title": COMPARISONS[comparison]["title"],
        "coverages": sorted({parse_arm(s)["coverage"] for s in selected
                             if parse_arm(s)["family"] in ("IRS", "IG2")}),
        "reference": args.reference,
        "reference_label": ("business as usual" if args.reference == "BAU"
                            else "IRS at the same nominal coverage level"),
        "win_txt": f"{years[0]}–{years[-1]}", "years": years,
        "unit_year": f"{per} per year", "unit_cum": per,
    }
    win = window_incidence(annual, years)
    if pfg is not None:
        win = win.merge(window_pfpr(pfg, years), on=["scenario", "pair_id"], how="left")
    av = cases_averted_annual(annual, selected, ref_of, years, factor)
    rel = paired_window_reduction(win, selected, ref_of)
    if not av.empty:
        av.to_csv(os.path.join(out, "CasesAverted_annual.csv"), index=False)

    n_pairs = annual["pair_id"].nunique()
    print(f"\n{COMPARISONS[comparison]['title']} | reference={args.reference} | "
          f"window={ctx['win_txt']} | paired pickup states={n_pairs}")
    print("Selected arms:")
    for s in selected:
        a = parse_arm(s)
        print(f"  - {arm_label(a['family'], a['coverage'])} [{s}]")
    if args.reference == "IRS":
        pairs = [(s, ref_of(s)) for s in selected if ref_of(s)]
        print("Matched IRS-reference pairs:")
        for s, r in pairs:
            print(f"  {s} vs {r}")
        if not pairs:
            print("  None")

    figure1(annual, selected, ctx)
    if pfg is not None:
        figure2(pfg, selected, ctx)
    else:
        print("  PfPR CSV not found; PfPR figure and table columns skipped.")
    figure3(win, selected, ctx)
    figure4(av, selected, ctx)
    figure5(av, rel, selected, ctx)
    write_table(win, av, rel, selected, ctx)
    write_captions(ctx)
    print(f"Outputs: {out}")


def main():
    args = parse_args()
    annual = read_csv(args.in_dir, "Scenario_AnnualBurden", args.cm)
    if annual.empty:
        raise SystemExit(f"No rows found for --cm {args.cm}. Check scenario names and CM suffixes.")
    pf = read_csv(args.in_dir, "PfPR_ClinicalIncidence_monthly", args.cm, required=False)
    pfg = grouped_pfpr(pf) if pf is not None and not pf.empty else None
    all_years = sorted(int(y) for y in annual["Year"].unique())
    start, end = args.years if args.years else (IRS_SWITCH_YEAR, all_years[-1])
    years = [y for y in all_years if start <= y <= end]
    if not years:
        raise SystemExit(f"No annual data in {start}–{end}; available years: {all_years}")
    if len(years) != end - start + 1:
        print(f"NOTE: requested window {start}–{end}; available years used: {years}")
    if annual["pair_id"].nunique() < 2:
        print("NOTE: fewer than two paired pickup states; uncertainty intervals may be uninformative.")
    comparisons = list(COMPARISONS) if args.comparison == "both" else [args.comparison]
    for comparison in comparisons:
        run_comparison(annual, pfg, args, comparison, years, all_years)


if __name__ == "__main__":
    main()
