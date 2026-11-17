"""
UWR publication and stakeholder figures, built from the analyzer CSVs.

Reads the files written by analyze_scenarios_UWR.py (stage 2) and writes
figures (PNG) and a captions text file. It does not run idmtools and needs
only pandas, numpy and matplotlib.

Design it plots (see run_scenarios_UWR.py)
------------------------------------------
  BAU                      continued IRS, the reference for every comparison
  NO_IRS                   optional lower bound (hidden unless SHOW_NO_IRS)
  IG2_<c>_<kill>_<usage>   IG2 at distribution coverage c under a net-use
                           assumption: full, no_age, no_season, no_retention
                           or none (age use, seasonal use, net retention
                           switched on/off)
  Older runs named IG2_<c>_<kill> (no usage suffix) are still read; their usage
  is taken from --legacy-usage-curves (default: detected from the
  effective-use csv).

Main figures (1-5, 8) show ONE usage assumption (PRIMARY_USAGE / --usage).
Figures 4, 6, 7 and 8 compare all usage assumptions found in the data. Every
title and the output folder name say which assumption is shown.

If the data also holds IRS_<c> arms (an older design), they are plotted too and
COMPARATOR = "IRS" compares IG2 with IRS at matched coverage.

Figures
-------
  Figure 1  Annual clinical incidence: BAU vs IG2 at each coverage
  Figure 2  Annual PfPR by scenario and age group
  Figure 3  Change in clinical incidence relative to BAU (or IRS benchmark)
  Figure 4  IG2 coverage vs incidence and PfPR, one line per usage assumption
  Figure 5  Annual clinical cases averted relative to BAU
  Figure 6  IG2 distribution coverage vs estimated effective-use coverage,
            one line per usage assumption
  Figure 7  Net-use assumptions: incidence, PfPR and cumulative cases averted
  Figure 8  Coverage threshold: change vs BAU by coverage and usage assumption,
            with the minimum coverage that maintains or reduces incidence

Input CSVs (in the analyzer output folder)
------------------------------------------
  Scenario_AnnualBurden_scenarios.csv            Figures 1, 5, 7
  PfPR_ClinicalIncidence_monthly_scenarios.csv   Figure 2
  Scenario_WindowMetrics_scenarios.csv           Figures 3, 4, 7, 8
  IG2_EffectiveUse_scenarios.csv                 Figures 6, 8 (effective use)

For Figures 6 and 8 the effective-use csv should have one row per IG2 arm
(columns ig2_coverage, effective_use_at_distribution, effective_use_time_avg and
usage_assumption or scenario). A csv with one row per coverage is read as a
single usage assumption.
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
               "UWR_BAU_vs_IG2_netuse_AnalyzerOutput")
STEP = "scenarios"

IRS_SWITCH_YEAR = 2027                 # MUST MATCH the scenario script
PRIMARY_METRIC = "incidence_per_1000_py"
NO_ARM = "NO_IRS"                      # no IRS and no IG2 (lower bound)
BAU_ARM = "BAU"                        # continued IRS (run script BAU_ARM_NAME)
COMPARATOR = "BAU"                     # "BAU", or "IRS" if IRS_<c> arms exist
SHOW_NO_IRS = False                    # draw the no-IRS/no-IG2 lower bound
SHOW_BAU = True

# Usage assumptions (keys of USAGE_ASSUMPTIONS in run_scenarios_UWR.py).
PRIMARY_USAGE = "full"                 # shown in Figures 1, 2, 3, 5 (--usage)
USAGE_ORDER = ["full", "no_age", "no_season", "no_age_season", "no_retention", "none"]
USAGE_LABELS = {
    "full": "WITH net-use assumptions (age + seasonal use + retention)",
    "no_age": "seasonal use + retention (no age use)",
    "no_season": "age use + retention (no seasonal use)",
    "no_age_season": "WITHOUT net-use assumptions (use = coverage + net retention)",
    "no_retention": "age + seasonal use (no net loss)",
    "none": "No net-use assumptions and no net loss (use = coverage)",
}
USAGE_SHORT = {"full": "Full", "no_age": "No age use", "no_season": "No seasonal use",
               "no_age_season": "Without (use = coverage + retention)", "no_retention": "No net loss",
               "none": "No adjustment, no net loss"}
USAGE_COLOR = {"full": "#1b9e77", "no_age": "#7570b3", "no_season": "#e7298a",
               "no_age_season": "#a6761d", "no_retention": "#66a61e", "none": "#e6ab02"}
LEGACY_USAGE = "full"                  # usage of arms named without a usage suffix
EFFECTIVE_USE_TARGETS = [40, 60, 80]   # % ; must match analyze_scenarios_UWR.py

# Coverage panels for Figures 1 and 5 (percent). None = every IG2 coverage in
# the data.
COMPARE_COVERAGES = None
FIG1_SHARE_Y = True
FIG5_SHARE_Y = True

OTHER_INT = "with other interventions"
ARM_LABELS = {"NO_IRS": "No IRS, no IG2",
              "BAU": "Business as usual (continued IRS)"}
BASELINE_LABEL = ARM_LABELS[BAU_ARM]
LABEL_LINES = True                     # write each line's name at its right-hand end

# Age groups for Figure 2: group name -> upper edges (years) of the summary-report
# age bins it combines (REPORT_AGE_BINS = 0.25, 0.5, 5, 15, 25, 100).
AGE_GROUPS = {
    "Under 5 years": [0.25, 0.5, 5.0],
    "5\u201314 years": [15.0],
    "15 years and older": [25.0, 100.0],
}

ROLE_COLOR = {"IRS": "#d95f02", "IG2": "#1b9e77"}

LOWER_Q, HIGH_Q = 0.10, 0.90
SHOW_RANGE = False                     # percentile bands / error bars (--show-range)
SHOW_TITLES = True                     # (--no-titles to switch off)
FORMATS = ["png"]                      # add "pdf" for vector output
DPI = 300

FIG3_MIN_PAIRS = 2   # fewer shared pickup states than this -> unpaired comparison
MIN_PAIRS = 2


# =============================================================================
# ARM NAMES, LABELS, STYLES
# =============================================================================

IG2_NEW_RE = re.compile(r"^IG2_(\d+)_([A-Za-z0-9]+)_(full|no_age_season|no_age|no_season|no_retention|none)$")
IG2_OLD_RE = re.compile(r"^IG2_(\d+)(?:_([A-Za-z0-9]+))?$")


def parse_ig2(name):
    """(coverage %, kill key, usage) of an IG2 arm name, or None."""
    m = IG2_NEW_RE.match(str(name))
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    m = IG2_OLD_RE.match(str(name))
    if m:
        return int(m.group(1)), m.group(2) or "", LEGACY_USAGE
    return None


def is_ig2(name):
    return parse_ig2(name) is not None


def usage_of(name):
    p = parse_ig2(name)
    return p[2] if p else None


def cov_of(name):
    p = parse_ig2(name)
    if p:
        return p[0]
    m = re.fullmatch(r"IRS_(\d+)", str(name))
    return int(m.group(1)) if m else -1


def _usage_rank(u):
    return USAGE_ORDER.index(u) if u in USAGE_ORDER else len(USAGE_ORDER)


def ig2_arms(names, usage=None):
    out = [n for n in names if is_ig2(n) and (usage is None or usage_of(n) == usage)]
    return sorted(out, key=lambda n: (cov_of(n), _usage_rank(usage_of(n))))


def irs_arms(names):
    return sorted([n for n in names if re.fullmatch(r"IRS_\d+", str(n))], key=cov_of)


def find_ig2(names, c, usage):
    hits = [n for n in ig2_arms(names, usage) if cov_of(n) == c]
    return hits[0] if hits else None


def find_irs(names, c):
    n = f"IRS_{c}"
    return n if n in set(names) else None


def usages_in(names):
    found = {usage_of(n) for n in names if is_ig2(n)}
    return ([u for u in USAGE_ORDER if u in found]
            + sorted(u for u in found if u not in USAGE_ORDER))


def usage_label(u):
    return USAGE_LABELS.get(u, str(u))


USAGE_TITLE = {   # figure subtitles: say plainly WITH or WITHOUT the net-use assumptions
    "full": "WITH net-use assumptions (age-specific use, seasonal use, net retention)",
    "no_age_season": ("WITHOUT net-use assumptions (every net used all year; "
                      "use = coverage + net retention)"),
    "none": "No net-use assumptions and no net loss (every net kept and used; use = coverage)",
}


USAGE_FILE_TAG = {   # folder / file-name suffix per net-use assumption
    "full": "with_netuse_assumptions",
    "no_age_season": "without_netuse_assumptions",
    "none": "no_netuse_no_netloss",
}


USAGE_CAPTION = {   # caption sentence describing the net-use assumption shown
    "full": ("IG2 nets are shown WITH net-use assumptions: age-specific usage, seasonal "
             "usage and net retention (Weibull discard), so the share of people using a "
             "net is lower than the distribution coverage."),
    "no_age_season": ("IG2 nets are shown WITHOUT net-use assumptions: everyone who still "
                      "has a net uses it all year (no age-specific or seasonal usage), but "
                      "nets are lost over time (Weibull discard), so effective use equals "
                      "the distribution coverage at distribution and then declines with net "
                      "retention."),
    "none": ("IG2 nets are shown with no net-use assumptions and no net loss: every net is "
             "kept and used all year, so effective use equals the distribution coverage."),
}


def usage_file_tag(u):
    return USAGE_FILE_TAG.get(u, f"usage_{u}")


def usage_tag():
    """Text saying which net-use assumption the main figures show."""
    return USAGE_TITLE.get(PRIMARY_USAGE, "IG2 net use: " + usage_label(PRIMARY_USAGE))


def display(name):
    if name in ARM_LABELS:
        return ARM_LABELS[name]
    m = re.fullmatch(r"IRS_(\d+)", name)
    if m:
        return f"IRS {m.group(1)}%"
    p = parse_ig2(name)
    if p:
        return f"IG2 {p[0]}%"
    return name


def end_label(name):
    """Short name written at a line's end."""
    if name == BAU_ARM:
        return "Continued IRS (BAU)"
    if name == NO_ARM:
        return "No IRS, no IG2"
    return display(name)


def hidden(name):
    if name == NO_ARM:
        return not SHOW_NO_IRS
    if name == BAU_ARM:
        return COMPARATOR != "BAU" and not SHOW_BAU
    return False


def ordered(names):
    names = [n for n in names if not hidden(n)]
    irs = irs_arms(names)
    ig2 = ig2_arms(names)
    refs = [n for n in (BAU_ARM, NO_ARM) if n in names]
    rest = [n for n in names if n not in irs and n not in ig2 and n not in refs]
    return refs + irs + rest + ig2


def build_styles(names):
    names = list(names)
    irs = irs_arms(names)
    ig2 = ig2_arms(names)
    covs = sorted({cov_of(n) for n in ig2})
    styles = {}
    for i, n in enumerate(irs):
        f = 0.5 if len(irs) == 1 else i / (len(irs) - 1)
        styles[n] = dict(color=plt.cm.Oranges(0.50 + 0.45 * f), linestyle="-",
                         marker="o", linewidth=2.0, markersize=5)
    for n in ig2:
        f = 0.5 if len(covs) == 1 else covs.index(cov_of(n)) / (len(covs) - 1)
        styles[n] = dict(color=plt.cm.Blues(0.45 + 0.5 * f), linestyle="--",
                         marker="s", linewidth=2.0, markersize=5)
    for n in names:
        if n == BAU_ARM:
            styles[n] = dict(color="0.15", linestyle="-", marker="^",
                             linewidth=2.2, markersize=5)
        elif n == NO_ARM:
            styles[n] = dict(color="0.55", linestyle=":", marker="v",
                             linewidth=1.6, markersize=5)
        elif n not in styles:
            styles[n] = dict(color="0.6", linestyle="-", marker=None, linewidth=1.5)
    return styles


def ig2_bar_colors(coverages):
    covs = sorted(coverages)
    return {c: plt.cm.Blues(0.45 + 0.5 * (0.5 if len(covs) == 1 else i / (len(covs) - 1)))
            for i, c in enumerate(covs)}


def usage_style(u):
    return dict(color=USAGE_COLOR.get(u, "0.4"), linewidth=2.0, marker="s", markersize=5)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

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


def add_figure_legend(fig, ncol=4, offset=-0.02):
    """Legend from every panel of the figure, each label once."""
    seen = {}
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    handles, labels = list(seen.values()), list(seen.keys())
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, offset), fontsize=9.5)


def label_line_ends(ax, items, x_end, width=16, fontsize=7.5):
    """
    Write each line's name just right of its last point. items = (y, text, color).
    Labels are pushed apart vertically (axes fraction) using their wrapped
    height, so they do not overlap. Needs a linear y axis with final limits.
    """
    if not items:
        return
    trans = ax.transData + ax.transAxes.inverted()
    pts = []
    for y, text, color in items:
        lines = textwrap.wrap(text, width) or [text]
        pts.append((trans.transform((x_end, y))[1], lines, color))
    pts.sort(key=lambda p: p[0])

    def h(lines):                      # label height in axes fraction
        return 0.034 * len(lines) + 0.012

    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):                      # push up
        ys[i] = max(ys[i], ys[i - 1] + 0.5 * (h(pts[i - 1][1]) + h(pts[i][1])))
    over = ys[-1] + 0.5 * h(pts[-1][1]) - 1.0
    if over > 0:                                     # shift down if off the top
        ys = [y - over for y in ys]
    for i in range(len(ys) - 2, -1, -1):             # keep the gaps after shifting
        ys[i] = min(ys[i], ys[i + 1] - 0.5 * (h(pts[i][1]) + h(pts[i + 1][1])))

    x_ax = trans.transform((x_end, 0))[0]
    for (y0, lines, color), y in zip(pts, ys):
        ax.annotate("\n".join(lines),
                    xy=(x_ax, y0), xycoords="axes fraction",
                    xytext=(x_ax + 0.03, y), textcoords="axes fraction",
                    va="center", ha="left", fontsize=fontsize, color=color,
                    fontweight="bold", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6,
                                    shrinkA=0, shrinkB=2))


def save_figure(fig, out_dir, stem, title=None, show_usage=True):
    if SHOW_TITLES and title:
        sub = f"{usage_tag()}; all arms {OTHER_INT}" if show_usage \
            else f"all arms {OTHER_INT}"
        fig.suptitle(f"{title}\n({sub})", fontsize=13, fontweight="bold", y=1.08)
    if show_usage:                     # file name says which net-use assumption it shows
        stem = f"{stem}_{usage_file_tag(PRIMARY_USAGE)}"
    for fmt in FORMATS:
        fig.savefig(os.path.join(out_dir, f"{stem}.{fmt}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}")


def read_csv(in_dir, prefix, required=True):
    path = os.path.join(in_dir, f"{prefix}_{STEP}.csv")
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(path)
        return None
    df = pd.read_csv(path)
    for c in ("scenario", "scenario_kind", "ig2_kill_key", "reference", "usage_assumption"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "pair_id" not in df.columns and "pickup_sim_index" in df.columns:
        df["pair_id"] = df["pickup_sim_index"].astype(int).astype(str)
    return df


def relative_change(win, ref, arm, col):
    """
    Relative change (%) in col of arm vs ref over the window, one value per
    comparison. Paired within pickup state when at least FIG3_MIN_PAIRS pickup
    states have both arms; otherwise every ref run is compared with every arm
    run (unpaired). Returns (values, paired_flag, n).
    """
    r = win.loc[win["scenario"] == ref, ["pair_id", col]].dropna()
    a = win.loc[win["scenario"] == arm, ["pair_id", col]].dropna()
    if r.empty or a.empty:
        return np.array([]), False, 0
    m = r.merge(a, on="pair_id", suffixes=("_ref", "_arm"))
    paired = m["pair_id"].nunique() >= FIG3_MIN_PAIRS
    if not paired:
        m = r.merge(a, how="cross", suffixes=("_ref", "_arm"))
    m = m[m[f"{col}_ref"] > 0]
    v = (m[f"{col}_arm"] / m[f"{col}_ref"] - 1.0) * 100.0
    return v.to_numpy(), paired, len(v)


def cases_averted(piv, ref, name, year_mask, factor):
    """
    Cases averted (ref - arm) per run, indexed by (pair_id, Year). Paired by
    pickup state when at least MIN_PAIRS pickup states have both arms; otherwise
    every run of ref is compared with every run of arm (unpaired).
    Returns (series, paired_flag).
    """
    sub = piv.loc[year_mask, [ref, name]]
    both = sub.dropna()
    if both.index.get_level_values("pair_id").nunique() >= MIN_PAIRS:
        return (both[ref] - both[name]) * factor, True
    r = sub[ref].dropna().rename("ref").reset_index()
    a = sub[name].dropna().rename("arm").reset_index()
    if r.empty or a.empty:
        return pd.Series(dtype=float), False
    m = r.merge(a, on="Year", suffixes=("_ref", "_arm"))
    m["pair_id"] = m["pair_id_ref"].astype(str) + "x" + m["pair_id_arm"].astype(str)
    s = (m["ref"] - m["arm"]) * factor
    s.index = pd.MultiIndex.from_frame(m[["pair_id", "Year"]])
    return s, False


# =============================================================================
# FIGURE 1: ANNUAL CLINICAL INCIDENCE
# =============================================================================

def figure1(annual, out_dir):
    s = summarise(annual, ["scenario", "Year"], PRIMARY_METRIC)
    all_names = list(s["scenario"].unique())
    years = sorted(s["Year"].unique())
    covs = COMPARE_COVERAGES

    if BAU_ARM not in all_names:
        raise ValueError(f"{BAU_ARM} is not in the annual csv.")

    ref_style = dict(color="0.15", linestyle="-", linewidth=2.2, marker="^", markersize=5)
    irs_style = dict(color=ROLE_COLOR["IRS"], linestyle="-", linewidth=2.0)
    ig2_style = dict(color=ROLE_COLOR["IG2"], linestyle="--", linewidth=2.0,
                     marker="s", markersize=5)

    fig, axes = plt.subplots(1, len(covs),
                             figsize=((5.6 if LABEL_LINES else 5.0) * len(covs), 4.8),
                             squeeze=False, sharey=FIG1_SHARE_Y)
    ends = []

    for k, (ax, c) in enumerate(zip(axes[0], covs)):
        irs_name = find_irs(all_names, c)
        ig2_name = find_ig2(all_names, c, PRIMARY_USAGE)

        arms = [(BAU_ARM, ref_style)]
        if SHOW_NO_IRS and NO_ARM in all_names:
            arms.append((NO_ARM, dict(color="0.55", linestyle=":", linewidth=1.8,
                                      marker="v", markersize=5)))
        if irs_name:
            arms.append((irs_name, irs_style))
        elif COMPARATOR == "IRS":
            print(f"  WARNING Figure 1: panel {c}% has no IRS_{c} arm.")
        if ig2_name:
            arms.append((ig2_name, ig2_style))
        else:
            print(f"  WARNING Figure 1: panel {c}% has no IG2 arm for usage "
                  f"'{PRIMARY_USAGE}'.")

        panel_ends = []
        for name, style in arms:
            d = s[s["scenario"] == name].sort_values("Year")
            if d.empty:
                continue
            ax.plot(d["Year"], d["median"], label=f"{display(name)}", **style)
            if SHOW_RANGE:
                ax.fill_between(d["Year"], d["lo"], d["hi"],
                                color=style["color"], alpha=0.15, linewidth=0)
            panel_ends.append((float(d["median"].iloc[-1]), end_label(name), style["color"]))
        ends.append(panel_ends)

        mark_switch(ax, label=(k == 0))
        if LABEL_LINES:
            ax.set_xlim(years[0] - 0.3, years[-1] + 0.3)
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Year")
        if k == 0 or not FIG1_SHARE_Y:
            ax.set_ylabel("Clinical incidence (cases per 1,000 person-years)")
        ax.set_title(f"IG2 at {c}% distribution coverage", fontsize=11)
        ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "abcdefgh"[k])

    add_figure_legend(fig, ncol=4, offset=-0.10)
    fig.tight_layout()
    if LABEL_LINES:              # after tight_layout so the y-limits are final
        fig.subplots_adjust(wspace=0.45)
        for ax, panel_ends in zip(axes[0], ends):
            label_line_ends(ax, panel_ends, years[-1])
    save_figure(fig, out_dir, "Figure1_AnnualIncidence",
                "Annual Clinical Malaria Incidence: IG2 vs Continued IRS")


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
    names = [n for n in ordered(g["scenario"].unique())
             if not is_ig2(n) or usage_of(n) == PRIMARY_USAGE]
    styles = build_styles(names)
    groups = [k for k in AGE_GROUPS if k in set(g["age_group"])]
    years = sorted(g["Year"].unique())

    fig, axes = plt.subplots(1, len(groups),
                             figsize=((5.8 if LABEL_LINES else 4.6) * len(groups), 4.6),
                             squeeze=False)
    ends = []
    for k, (ax, grp) in enumerate(zip(axes[0], groups)):
        s = summarise(g[g["age_group"] == grp], ["scenario", "Year"], "pfpr_pct")
        panel_ends = []
        for n in names:
            d = s[s["scenario"] == n].sort_values("Year")
            if d.empty:
                continue
            ax.plot(d["Year"], d["median"], label=display(n), **styles[n])
            panel_ends.append((float(d["median"].iloc[-1]), end_label(n),
                               styles[n]["color"]))
            if SHOW_RANGE:
                ax.fill_between(d["Year"], d["lo"], d["hi"],
                                color=styles[n]["color"], alpha=0.15, linewidth=0)
        ends.append(panel_ends)
        mark_switch(ax, label=(k == 0))
        if LABEL_LINES:
            ax.set_xlim(years[0] - 0.3, years[-1] + 0.3)
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("PfPR (%)" if k == 0 else "")
        ax.set_title(grp, fontsize=11)
        ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "abcdef"[k])
    add_figure_legend(fig, ncol=5, offset=-0.10)
    fig.tight_layout()
    if LABEL_LINES:
        fig.subplots_adjust(wspace=0.5)
        for ax, panel_ends in zip(axes[0], ends):
            label_line_ends(ax, panel_ends, years[-1])
    save_figure(fig, out_dir, "Figure2_AnnualPfPR_byAgeGroup",
                "Annual PfPR by Scenario and Age Group")


# =============================================================================
# FIGURE 3: CHANGE IN INCIDENCE RELATIVE TO BAU (OR IRS BENCHMARK)
# =============================================================================

def figure3(win, out_dir, window):
    names = set(win["scenario"])
    irs = irs_arms(names)
    refs = sorted(irs, key=cov_of, reverse=True) if (COMPARATOR == "IRS" and irs) \
        else [BAU_ARM]
    if BAU_ARM not in names and refs == [BAU_ARM]:
        raise ValueError("no BAU arm in the window-metrics csv")
    ig2 = ig2_arms(names, PRIMARY_USAGE)
    if not ig2:
        raise ValueError(f"no IG2 arms with usage '{PRIMARY_USAGE}' in the window-metrics csv")
    colors = ig2_bar_colors([cov_of(n) for n in ig2])
    arms = ([(BAU_ARM, "BAU", "0.55")] if refs != [BAU_ARM] and BAU_ARM in names else []) + \
           [(n, f"IG2 {cov_of(n)}%", colors[cov_of(n)]) for n in ig2]

    rows = []
    for ref in refs:
        for arm, lab, col in arms:
            if arm == ref:
                continue
            v, paired, n = relative_change(win, ref, arm, PRIMARY_METRIC)
            if n == 0:
                continue
            rows.append(dict(reference=ref, arm=arm, label=lab, color=col, paired=paired,
                             n=n, median=np.median(v), lo=np.quantile(v, LOWER_Q),
                             hi=np.quantile(v, HIGH_Q)))
    s = pd.DataFrame(rows)
    if s.empty:
        raise ValueError("no comparable runs for Figure 3")
    s.drop(columns="color").to_csv(os.path.join(out_dir, "Figure3_change_summary.csv"),
                                   index=False)
    any_unpaired = not s["paired"].all()

    fig, axes = plt.subplots(1, len(refs), figsize=(max(5.0, 4.6 * len(refs)), 4.8),
                             squeeze=False, sharey=True)
    big = s["median"].abs().max() > 300
    for k, (ax, ref) in enumerate(zip(axes[0], refs)):
        r = s[s["reference"] == ref].reset_index(drop=True)
        x = np.arange(len(r))
        bars = ax.bar(x, r["median"], width=0.6, color=r["color"], edgecolor="0.2",
                      linewidth=0.6)
        for b, p in zip(bars, r["paired"]):
            if not p:
                b.set_hatch("//")
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
        ax.set_xticklabels(r["label"])
        ax.set_xlabel("Scenario")
        ax.set_title(f"Relative to {display(ref)}", fontsize=11)
        if big:
            ax.set_yscale("symlog", linthresh=100)
            ax.set_ylim(bottom=-100, top=s["median"].max() * 3)
        if k == 0:
            ax.set_ylabel(f"Change in clinical incidence, {window} (%)")
        tidy(ax)
        panel_letter(ax, "abc"[k])
    fig.text(0.5, -0.03,
             "Above zero: higher incidence than the reference. Below zero: lower "
             "incidence. Bars show the median across paired pickup states"
             + ("; hatched bars share too few pickup states with the reference and "
                "compare all runs unpaired" if any_unpaired else "") + "."
             + (" Vertical axis is symmetric-log." if big else ""),
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure3_Change_vs_Reference",
                "Change in Clinical Incidence Relative to "
                + ("Continued IRS" if refs == [BAU_ARM] else "IRS Benchmark"))


# =============================================================================
# FIGURE 4: IG2 COVERAGE vs INCIDENCE AND PfPR, ONE LINE PER USAGE ASSUMPTION
# =============================================================================

def figure4(win, out_dir, window):
    pf_col = "U5_pfpr" if "U5_pfpr" in win.columns else "pfpr_all_ages"
    pf_label = ("PfPR, children under 5 years (%)" if pf_col == "U5_pfpr"
                else "PfPR, all ages (%)")
    scale = 100.0 if win[pf_col].max() <= 1.0 + 1e-9 else 1.0
    win = win.assign(pfpr_pct=win[pf_col] * scale)

    names = list(win["scenario"].unique())
    usages = usages_in(names)
    if not usages:
        raise ValueError("no IG2 scenarios in the window-metrics csv")
    refs = [n for n in [BAU_ARM] + irs_arms(names) + ([NO_ARM] if SHOW_NO_IRS else [])
            if n in names]
    ref_styles = build_styles(refs)

    panels = [(PRIMARY_METRIC, "Clinical incidence (cases per 1,000 person-years)"),
              ("pfpr_pct", pf_label)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for k, (ax, (col, ylabel)) in enumerate(zip(axes, panels)):
        s = summarise(win, ["scenario"], col).set_index("scenario")
        for n in refs:
            ax.axhline(s.loc[n, "median"], color=ref_styles[n]["color"],
                       linestyle="-." if n == BAU_ARM else ref_styles[n]["linestyle"],
                       linewidth=1.8, label=display(n))
        for u in usages:
            arms = ig2_arms(names, u)
            x = np.array([cov_of(n) for n in arms], float)
            y = np.array([s.loc[n, "median"] for n in arms])
            ax.plot(x, y, label=f"IG2: {usage_label(u)}", linestyle="--",
                    **usage_style(u))
            if SHOW_RANGE:
                lo = np.array([s.loc[n, "lo"] for n in arms])
                hi = np.array([s.loc[n, "hi"] for n in arms])
                ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                            fmt="none", ecolor=USAGE_COLOR.get(u, "0.4"), capsize=3)
        covs = sorted({cov_of(n) for n in ig2_arms(names)})
        auto_scale(ax, list(s["median"]))
        ax.set_xticks(covs)
        ax.set_xlim(min(covs) - 8, max(covs) + 8)
        ax.set_xlabel("IG2 distribution coverage (%)")
        ax.set_ylabel(f"{ylabel}, {window}")
        if ax.get_yscale() != "log":
            ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "ab"[k])
    add_figure_legend(fig, ncol=2, offset=-0.20)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure4_IG2Coverage_vs_Incidence_PfPR",
                "IG2 Distribution Coverage versus Clinical Incidence and PfPR",
                show_usage=False)


# =============================================================================
# FIGURE 5: ANNUAL CASES AVERTED
# =============================================================================

def figure5(annual, out_dir, population):
    """
    COMPARATOR "BAU": cases averted by IG2 (and by IRS_<c> arms, if present)
    relative to business as usual. COMPARATOR "IRS": IG2 relative to IRS at the
    same coverage. Computed within each pickup state and year, then the median
    across pickup states. One panel per coverage.
    """
    piv = annual.pivot_table(index=["pair_id", "Year"], columns="scenario",
                             values=PRIMARY_METRIC)
    all_names = list(piv.columns)
    covs = COMPARE_COVERAGES
    vs_irs = COMPARATOR == "IRS"
    if not vs_irs and BAU_ARM not in all_names:
        raise ValueError(f"{BAU_ARM} is not in the annual csv.")

    factor = (population / 1000.0) if population else 1.0
    per = f"per {population:,} population" if population else "per 1,000 person-years"
    ylabel = (f"Cases averted by IG2 vs IRS, {per}" if vs_irs
              else f"Clinical cases averted vs continued IRS, {per}")

    years = [int(y) for y in sorted(piv.index.get_level_values("Year").unique())
             if y >= IRS_SWITCH_YEAR]
    if not years:
        raise ValueError(f"no years on or after {IRS_SWITCH_YEAR} in the annual csv")
    year_mask = piv.index.get_level_values("Year") >= IRS_SWITCH_YEAR

    fig, axes = plt.subplots(1, len(covs), figsize=(5.2 * len(covs), 4.8),
                             squeeze=False, sharey=FIG5_SHARE_Y)
    rows = []

    for k, (ax, c) in enumerate(zip(axes[0], covs)):
        arms = []
        irs_name = find_irs(all_names, c)
        ig2_name = find_ig2(all_names, c, PRIMARY_USAGE)
        if vs_irs:
            ref = irs_name
            if irs_name and ig2_name:
                arms.append(("IG2", ig2_name, f"IG2 {c}% (vs IRS {c}%)"))
            if not irs_name:
                print(f"  WARNING Figure 5: panel {c}% has no IRS_{c} arm.")
        else:
            ref = BAU_ARM
            if irs_name:
                arms.append(("IRS", irs_name, display(irs_name)))
            if ig2_name:
                arms.append(("IG2", ig2_name, "IG2 (coverage in panel title)"))

        width = 0.8 / max(len(arms), 1)
        cum_text = []
        for j, (role, name, label) in enumerate(arms):
            averted, paired = cases_averted(piv, ref, name, year_mask, factor)
            if averted.empty:
                print(f"  WARNING Figure 5: no runs of {name} or {ref} after "
                      f"{IRS_SWITCH_YEAR}; panel {c}% skips {role}.")
                continue
            if not paired:
                label = f"{label} (unpaired)"
                print(f"  NOTE Figure 5: {name} and {ref} share fewer than "
                      f"{MIN_PAIRS} pickup states; using all run combinations (unpaired).")
            grouped = averted.groupby(level="Year")
            annual_med = grouped.median()
            cum_all = averted.groupby(level="pair_id").sum()
            cum = float(cum_all.median())

            xs = np.arange(len(years)) + (j - (len(arms) - 1) / 2) * width
            heights = np.array([annual_med.get(y, np.nan) for y in years])
            ax.bar(xs, heights, width=width * 0.95, color=ROLE_COLOR[role],
                   edgecolor="0.2", linewidth=0.5, label=label)
            q_lo_s, q_hi_s = grouped.quantile(LOWER_Q), grouped.quantile(HIGH_Q)
            q_lo = np.array([q_lo_s.get(y, np.nan) for y in years])
            q_hi = np.array([q_hi_s.get(y, np.nan) for y in years])
            n_runs = grouped.size()
            ax.errorbar(xs, heights,
                        yerr=[np.clip(np.nan_to_num(heights - q_lo), 0, None),
                              np.clip(np.nan_to_num(q_hi - heights), 0, None)],
                        fmt="none", ecolor="0.15", capsize=4, capthick=1.2,
                        elinewidth=1.2, zorder=3)

            for y, h, lo, hi in zip(years, heights, q_lo, q_hi):
                rows.append(dict(coverage_pct=c, arm=role, scenario=name,
                                 baseline=ref, period=str(y), paired=paired,
                                 n=int(n_runs.get(y, 0)), median_cases_averted=h,
                                 q_low_cases_averted=lo, q_high_cases_averted=hi))
            rows.append(dict(coverage_pct=c, arm=role, scenario=name,
                             baseline=ref, paired=paired, n=len(cum_all),
                             period=f"cumulative {years[0]}-{years[-1]}",
                             median_cases_averted=cum,
                             q_low_cases_averted=float(cum_all.quantile(LOWER_Q)),
                             q_high_cases_averted=float(cum_all.quantile(HIGH_Q))))
            cum_lo = float(cum_all.quantile(LOWER_Q))
            cum_hi = float(cum_all.quantile(HIGH_Q))
            cum_text.append((role, cum, cum_lo, cum_hi,
                             f"n={len(cum_all)}{'' if paired else ', unpaired'}"))

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
            txt = (f"Cumulative {years[0]}\u2013{years[-1]} "
                   f"(median [{int(LOWER_Q * 100)}th\u2013{int(HIGH_Q * 100)}th]):\n"
                   + "\n".join(
                       f"{r}: {v:,.0f} [{lo:,.0f} to {hi:,.0f}] {unit} ({n})"
                       if np.isfinite(v) else f"{r}: no runs"
                       for r, v, lo, hi, n in cum_text))
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=8, color="0.25")
        tidy(ax)
        panel_letter(ax, "abcdefgh"[k])

    for ax in (axes[0][:1] if FIG5_SHARE_Y else axes[0]):   # headroom for the text
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))

    add_figure_legend(fig, ncol=2, offset=-0.12)
    fig.text(0.5, -0.20,
             ("Bars: IRS incidence minus IG2 incidence at the same coverage, within each "
              "pickup state. Above zero: IG2 averts more cases than IRS."
              if vs_irs else
              f"Cases averted are measured against {BASELINE_LABEL[0].lower() + BASELINE_LABEL[1:]}. "
              "Above zero: cases averted. Below zero: additional cases.")
             + " Bars show the median across paired pickup states; error bars show "
             f"the {int(LOWER_Q * 100)}th\u2013{int(HIGH_Q * 100)}th percentiles.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure5_CasesAverted",
                "Annual Clinical Cases Averted by IG2 Relative to "
                + ("IRS at Matched Coverage" if vs_irs else "Continued IRS"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Figure5_cases_summary.csv"),
                              index=False)


# =============================================================================
# FIGURE 6: IG2 COVERAGE AND ESTIMATED EFFECTIVE USE, PER USAGE ASSUMPTION
# =============================================================================

def effective_use_table(eff):
    """Add a usage_assumption column to the effective-use csv if it lacks one."""
    eff = eff.copy()
    if "usage_assumption" in eff.columns and (eff["usage_assumption"] != "").any():
        eff["usage_assumption"] = eff["usage_assumption"].replace("", LEGACY_USAGE)
    elif "scenario" in eff.columns:
        eff["usage_assumption"] = eff["scenario"].map(lambda n: usage_of(n) or LEGACY_USAGE)
    else:
        eff["usage_assumption"] = LEGACY_USAGE
    return eff


def figure6(eff, out_dir):
    eff = effective_use_table(eff)
    usages = [u for u in USAGE_ORDER if u in set(eff["usage_assumption"])] + \
        sorted(u for u in set(eff["usage_assumption"]) if u not in USAGE_ORDER)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.plot([0, 100], [0, 100], color="0.55", linestyle=":", linewidth=1.3)
    for t in EFFECTIVE_USE_TARGETS:
        ax.axhline(t, color="0.7", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(1, t + 1, f"target {t}%", fontsize=7.5, color="0.45")
    ends = [(100.0, "Equal to distribution coverage", "0.45")]

    for u in usages:
        d = eff[eff["usage_assumption"] == u].sort_values("ig2_coverage")
        x = d["ig2_coverage"].to_numpy() * 100
        y_avg = d["effective_use_time_avg"].to_numpy() * 100
        ax.plot(x, y_avg, label=usage_label(u), **usage_style(u))
        if u == PRIMARY_USAGE:
            y_dist = d["effective_use_at_distribution"].to_numpy() * 100
            ax.plot(x, y_dist, color=USAGE_COLOR.get(u, "0.4"), linestyle="--",
                    linewidth=1.2, marker="o", markersize=3.5, alpha=0.8)
            for xi, v in zip(x, y_avg):
                ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(0, -13),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            color=USAGE_COLOR.get(u, "0.4"))
        ends.append((float(y_avg[-1]), USAGE_SHORT.get(u, str(u)),
                     USAGE_COLOR.get(u, "0.4")))
        # distribution coverage needed for each effective-use target (line is linear in c)
        factor = float(np.median(y_avg / x)) if len(x) else np.nan
        for t in EFFECTIVE_USE_TARGETS:
            need = t / factor if factor > 0 else np.inf
            if need <= 100:
                ax.plot([need], [t], marker="*", markersize=12,
                        color=USAGE_COLOR.get(u, "0.4"), markeredgecolor="black", zorder=5)
                ax.annotate(f"{need:.0f}%", xy=(need, t), xytext=(6, 4),
                            textcoords="offset points", fontsize=8,
                            color=USAGE_COLOR.get(u, "0.4"))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("IG2 distribution coverage (%)")
    ax.set_ylabel("Estimated average effective-use coverage (%)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False,
              title="Time-averaged over the net cycle", title_fontsize=8.5)
    tidy(ax)
    fig.tight_layout()
    if LABEL_LINES:
        label_line_ends(ax, ends, 90 if len(ends) else 100, width=14)

    note = ("Time-averaged effective use = distribution coverage \u00d7 net retention "
            "\u00d7 age multiplier \u00d7 season multiplier, with each switched-off "
            "factor set to 1. The dashed line is use at distribution under the "
            f"'{USAGE_SHORT.get(PRIMARY_USAGE, PRIMARY_USAGE)}' assumption. Values are "
            "calculated analytically from the assumed net retention and usage curves; "
            "they are not an observed measure of net use. Stars mark the distribution "
            "coverage needed to reach each effective-use target (dashed lines); a "
            "missing star means the target is not reachable at 100% coverage.")
    fig.text(0.5, -0.02, "\n".join(textwrap.wrap(note, 95)), ha="center", va="top",
             fontsize=8.5, style="italic")
    save_figure(fig, out_dir, "Figure6_IG2Coverage_vs_EstimatedEffectiveUse",
                "IG2 Distribution Coverage and Estimated Effective-Use Coverage",
                show_usage=False)


# =============================================================================
# FIGURE 7: NET-USE ASSUMPTIONS
# =============================================================================

def figure7(win, annual, out_dir, window, population):
    names = list(win["scenario"].unique())
    usages = usages_in(names)
    covs = sorted({cov_of(n) for n in ig2_arms(names)})
    if len(usages) < 2:
        raise ValueError("Figure 7 needs at least two usage assumptions in the data")

    pf_col = "U5_pfpr" if "U5_pfpr" in win.columns else "pfpr_all_ages"
    pf_label = ("PfPR, under 5 years (%)" if pf_col == "U5_pfpr" else "PfPR, all ages (%)")
    scale = 100.0 if win[pf_col].max() <= 1.0 + 1e-9 else 1.0
    win = win.assign(pfpr_pct=win[pf_col] * scale)

    width = 0.8 / len(usages)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    def offset(j):
        return (j - (len(usages) - 1) / 2) * width

    panels = [(PRIMARY_METRIC, f"Clinical incidence, {window}\n(cases per 1,000 person-years)"),
              ("pfpr_pct", f"{pf_label}, {window}")]
    for k, (ax, (col, ylabel)) in enumerate(zip(axes[:2], panels)):
        s = summarise(win, ["scenario"], col).set_index("scenario")
        for j, u in enumerate(usages):
            xs, hs, los, his = [], [], [], []
            for i, c in enumerate(covs):
                n = find_ig2(names, c, u)
                if n is not None:
                    xs.append(i + offset(j))
                    hs.append(s.loc[n, "median"])
                    los.append(s.loc[n, "lo"])
                    his.append(s.loc[n, "hi"])
            ax.bar(xs, hs, width=width * 0.95, color=USAGE_COLOR.get(u, "0.5"),
                   edgecolor="0.2", linewidth=0.5, label=usage_label(u))
            hs, los, his = np.array(hs), np.array(los), np.array(his)
            ax.errorbar(xs, hs, yerr=[np.clip(hs - los, 0, None), np.clip(his - hs, 0, None)],
                        fmt="none", ecolor="0.15", capsize=3, elinewidth=1.1, zorder=3)
        if BAU_ARM in s.index:
            ax.axhspan(s.loc[BAU_ARM, "lo"], s.loc[BAU_ARM, "hi"], color="0.15",
                       alpha=0.10, linewidth=0, zorder=0)
            ax.axhline(s.loc[BAU_ARM, "median"], color="0.15", linestyle="-.",
                       linewidth=1.8, label=display(BAU_ARM))
        ax.set_xticks(range(len(covs)))
        ax.set_xticklabels([f"{c}%" for c in covs])
        ax.set_xlabel("IG2 distribution coverage")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        tidy(ax)
        panel_letter(ax, "ab"[k])

    # (c) cumulative cases averted vs BAU after the switch
    piv = annual.pivot_table(index=["pair_id", "Year"], columns="scenario",
                             values=PRIMARY_METRIC)
    factor = (population / 1000.0) if population else 1.0
    per = f"per {population:,} population" if population else "per 1,000 person-years"
    year_mask = piv.index.get_level_values("Year") >= IRS_SWITCH_YEAR
    ys = sorted(int(y) for y in piv.index.get_level_values("Year").unique()
                if y >= IRS_SWITCH_YEAR)
    rows = []
    ax = axes[2]
    for j, u in enumerate(usages):
        for i, c in enumerate(covs):
            n = find_ig2(names, c, u)
            if n is None or n not in piv.columns or BAU_ARM not in piv.columns:
                continue
            averted, paired = cases_averted(piv, BAU_ARM, n, year_mask, factor)
            if averted.empty:
                continue
            cum = averted.groupby(level="pair_id").sum()
            med, lo, hi = float(cum.median()), float(cum.quantile(LOWER_Q)), \
                float(cum.quantile(HIGH_Q))
            x = i + offset(j)
            ax.bar(x, med, width=width * 0.95, color=USAGE_COLOR.get(u, "0.5"),
                   edgecolor="0.2", linewidth=0.5,
                   hatch=None if paired else "//")
            ax.errorbar([x], [med], yerr=[[max(med - lo, 0)], [max(hi - med, 0)]],
                        fmt="none", ecolor="0.15", capsize=3, elinewidth=1.1, zorder=3)
            rows.append(dict(scenario=n, ig2_coverage_pct=c, usage_assumption=u,
                             paired=paired, n=len(cum), median_cum_cases_averted=med,
                             q_low=lo, q_high=hi))
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(range(len(covs)))
    ax.set_xticklabels([f"{c}%" for c in covs])
    ax.set_xlabel("IG2 distribution coverage")
    ax.set_ylabel(f"Cumulative cases averted vs continued IRS\n{per}, "
                  f"{ys[0]}\u2013{ys[-1]}" if ys else "Cumulative cases averted")
    tidy(ax)
    panel_letter(ax, "c")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Figure7_netuse_summary.csv"),
                              index=False)

    add_figure_legend(fig, ncol=2, offset=-0.16)
    fig.text(0.5, -0.22,
             "Bars show the median across paired pickup states; error bars show the "
             f"{int(LOWER_Q * 100)}th\u2013{int(HIGH_Q * 100)}th percentiles. Dash-dot line "
             "and grey band: continued IRS (median and percentile range).",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure7_NetUseAssumptions",
                "Effect of Net-Use Assumptions on IG2 Impact", show_usage=False)


# =============================================================================
# FIGURE 8: COVERAGE THRESHOLD
# =============================================================================

def threshold_coverage(x, y):
    """
    Smallest distribution coverage (%) at which the change vs BAU is <= 0.
    x ascending, y = median change (%). Returns (threshold or nan, status).
    """
    if len(x) == 0:
        return np.nan, "no data"
    if y[0] <= 0:
        return float(x[0]), f"at or below the lowest tested coverage ({x[0]:g}%)"
    for i in range(len(x) - 1):
        if y[i] > 0 and y[i + 1] <= 0:
            thr = x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            return float(thr), f"interpolated between {x[i]:g}% and {x[i + 1]:g}%"
    return np.nan, f"not reached within the tested range (up to {x[-1]:g}%)"


def figure8(win, eff, out_dir, window):
    names = list(win["scenario"].unique())
    usages = usages_in(names)
    if BAU_ARM not in names or not usages:
        raise ValueError("Figure 8 needs the BAU arm and IG2 arms")
    eff = effective_use_table(eff) if eff is not None else None

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    rows, all_y = [], []
    for u in usages:
        pts = []
        for n in ig2_arms(names, u):
            v, paired, cnt = relative_change(win, BAU_ARM, n, PRIMARY_METRIC)
            if cnt:
                pts.append((cov_of(n), float(np.median(v)), paired))
        pts.sort()
        if not pts:
            continue
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts])
        all_y.extend(y)
        ax.plot(x, y, label=usage_label(u), linestyle="--", **usage_style(u))

        thr, status = threshold_coverage(list(x), list(y))
        eff_thr = np.nan
        if eff is not None and np.isfinite(thr):
            d = eff[eff["usage_assumption"] == u].sort_values("ig2_coverage")
            if len(d):
                eff_thr = float(np.interp(thr, d["ig2_coverage"].to_numpy() * 100,
                                          d["effective_use_time_avg"].to_numpy() * 100))
        if np.isfinite(thr):
            ax.plot([thr], [0], marker="*", markersize=14, color=USAGE_COLOR.get(u, "0.4"),
                    markeredgecolor="black", zorder=5)
        rows.append(dict(usage_assumption=u, threshold_coverage_pct=thr, status=status,
                         effective_use_time_avg_at_threshold_pct=eff_thr,
                         tested_coverages_pct=" ".join(f"{c:g}" for c in x)))

    ax.axhline(0, color="black", linewidth=0.9)
    covs = sorted({cov_of(n) for n in ig2_arms(names)})
    ax.set_xticks(covs)
    ax.set_xlim(min(covs) - 5, max(covs) + 5)
    if all_y and max(abs(v) for v in all_y) > 300:
        ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("IG2 distribution coverage (%)")
    ax.set_ylabel(f"Change in clinical incidence vs continued IRS, {window} (%)")
    ax.legend(loc="best", fontsize=8.5, frameon=False, title="Net-use assumption",
              title_fontsize=8.5)
    tidy(ax)
    fig.text(0.5, -0.03,
             "\n".join(textwrap.wrap(
                 "Stars mark the minimum distribution coverage at which median incidence "
                 "equals continued IRS, by linear interpolation between tested coverages "
                 "(a coarse estimate when few coverages are tested). Below zero: IG2 gives "
                 "lower incidence than continued IRS.", 95)),
             ha="center", va="top", fontsize=8.5, style="italic")
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure8_CoverageThreshold",
                "Minimum IG2 Coverage Needed to Match Continued IRS", show_usage=False)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Threshold_summary.csv"), index=False)


# =============================================================================
# CAPTIONS
# =============================================================================

def write_captions(out_dir, n_pairs, window, population):
    pair_txt = f"{n_pairs} paired pickup states" if n_pairs else "paired pickup states"
    pop_txt = f"per {population:,} population" if population else "per 1,000 person-years"
    covs = COMPARE_COVERAGES or []
    covs_txt = ", ".join(str(c) for c in covs[:-1])
    covs_txt = f"{covs_txt} and {covs[-1]}" if covs_txt else (str(covs[0]) if covs else "")
    usage_txt = USAGE_CAPTION.get(
        PRIMARY_USAGE,
        f"IG2 nets are shown under the '{usage_tag()}' net-use assumption "
        "(age-specific usage, seasonal usage and net retention switched on or off "
        "as stated), so the share of people using a net can be lower than the "
        "distribution coverage.")
    cov_note = ("IRS coverage is the share of structures sprayed and IG2 coverage is the "
                "share of the population receiving a net, so they are comparable scenario "
                "levels, not equivalent measures.")

    items = [
        ("Figure 1. Annual clinical malaria incidence, IG2 versus continued IRS",
         f"Clinical incidence (cases per 1,000 person-years) for each year, median across "
         f"{pair_txt}. Each panel ({covs_txt}% coverage) compares IG2 distribution at that "
         f"coverage from 1 January {IRS_SWITCH_YEAR} with business as usual (continued IRS). "
         f"All arms keep the other interventions (case management, ANC/CWC nets, SMC, "
         f"IPTp, vaccine). Before {IRS_SWITCH_YEAR} all arms follow the same path. "
         f"{cov_note} {usage_txt}"),
        ("Figure 2. Annual PfPR by scenario and age group",
         f"Annual Plasmodium falciparum parasite prevalence (PfPR, %) by age group, median "
         f"across {pair_txt}. PfPR is averaged over 30-day reporting intervals within each "
         f"year, weighted by population and interval length. {usage_txt}"),
        ("Figure 3. Change in clinical incidence relative to the reference",
         f"Median relative change in clinical incidence ({window}) of IG2 at each coverage "
         f"compared with continued IRS; all arms keep the other interventions. Changes are "
         f"computed within each pickup state where at least {FIG3_MIN_PAIRS} pickup states "
         f"have both arms; otherwise (hatched bars) every reference run is compared with "
         f"every arm run. Positive values: higher incidence than the reference. Values are "
         f"in Figure3_change_summary.csv. {usage_txt}"),
        ("Figure 4. IG2 distribution coverage versus clinical incidence and PfPR",
         f"Median clinical incidence (a) and PfPR (b) over {window} for each IG2 "
         f"distribution coverage, one line per net-use assumption, with continued IRS "
         f"as a horizontal line. {cov_note}"),
        ("Figure 5. Annual clinical cases averted relative to continued IRS",
         f"Annual clinical cases averted ({pop_txt}) by IG2 at each coverage ({covs_txt}%) "
         f"from 1 January {IRS_SWITCH_YEAR}: continued-IRS incidence minus IG2 incidence, "
         f"computed within each pickup state, shown as the median across {pair_txt}; error "
         f"bars show the {int(LOWER_Q * 100)}th\u2013{int(HIGH_Q * 100)}th percentiles. "
         f"Positive values are cases averted; negative values are additional cases. "
         f"Cumulative totals are in Figure5_cases_summary.csv. {usage_txt}"),
        ("Figure 6. IG2 distribution coverage and estimated effective-use coverage",
         "Estimated share of the population effectively using an IG2 net for each "
         "distribution coverage and net-use assumption. Time-averaged effective use is "
         "distribution coverage multiplied by mean net retention (Weibull discard), the "
         "age-usage multiplier and the seasonal-usage multiplier, with each switched-off "
         "factor equal to 1. It is a model-based estimate, not an observed measure of "
         "net use."),
        ("Figure 7. Effect of net-use assumptions on IG2 impact",
         f"Median clinical incidence (a) and PfPR (b) over {window}, and cumulative clinical "
         f"cases averted relative to continued IRS from {IRS_SWITCH_YEAR} ({pop_txt}) (c), "
         f"for each IG2 coverage under each net-use assumption (with and without "
         f"age-specific usage, seasonal usage and net retention). Error bars show the "
         f"{int(LOWER_Q * 100)}th\u2013{int(HIGH_Q * 100)}th percentiles across pickup "
         f"states; hatched bars compare unpaired runs. Values are in "
         f"Figure7_netuse_summary.csv."),
        ("Figure 8. Minimum IG2 distribution coverage to match continued IRS",
         f"Median change in clinical incidence ({window}) relative to continued IRS by IG2 "
         f"distribution coverage and net-use assumption. Stars mark the minimum "
         f"distribution coverage at which incidence equals continued IRS, estimated by "
         f"linear interpolation between tested coverages; the corresponding time-averaged "
         f"effective-use coverage is in Threshold_summary.csv."),
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
                   help="default: <input-dir>/EFFIG2_figures_<usage>_vs<comparator> "
                        "(one folder per usage; Figures 6-8 in ..._comparison_...)")
    p.add_argument("--population", type=int, default=None,
                   help="cases averted per this population (default: per 1,000 person-years)")
    p.add_argument("--window-start", type=int, default=IRS_SWITCH_YEAR,
                   help="first year of the window quoted in axis labels")
    p.add_argument("--show-range", action="store_true",
                   help=f"add {int(LOWER_Q * 100)}th-{int(HIGH_Q * 100)}th percentiles")
    p.add_argument("--no-titles", action="store_true", help="omit figure titles")
    p.add_argument("--usage", default="all", choices=["all"] + USAGE_ORDER,
                   help="net-use assumption shown in Figures 1-5; 'all' (default) "
                        "makes them once for every assumption in the data")
    p.add_argument("--legacy-usage-curves", choices=["auto", "on", "off"], default="auto",
                   help="for older IG2_<c> arms with no usage suffix: were usage curves "
                        "applied? auto = read the effective-use csv")
    p.add_argument("--comparator", choices=["IRS", "BAU"], default=COMPARATOR,
                   help="BAU (continued IRS), or IRS at matched coverage if IRS_<c> "
                        "arms are in the data")
    return p.parse_args()


def detect_legacy_usage(in_dir):
    """'full' if the effective-use csv shows use at distribution below coverage."""
    eff = read_csv(in_dir, "IG2_EffectiveUse", required=False)
    if eff is None or not {"ig2_coverage", "effective_use_at_distribution"} <= set(eff.columns):
        return None
    curves = not np.allclose(eff["effective_use_at_distribution"], eff["ig2_coverage"],
                             atol=0.005)
    return "full" if curves else "none"


def main():
    global SHOW_RANGE, SHOW_TITLES, COMPARATOR, PRIMARY_USAGE, LEGACY_USAGE, COMPARE_COVERAGES
    args = parse_args()
    SHOW_RANGE = args.show_range
    SHOW_TITLES = not args.no_titles
    COMPARATOR = args.comparator

    if args.legacy_usage_curves == "on":
        LEGACY_USAGE = "full"
    elif args.legacy_usage_curves == "off":
        LEGACY_USAGE = "none"
    else:
        LEGACY_USAGE = detect_legacy_usage(args.input_dir) or LEGACY_USAGE

    print(f"Reading {args.input_dir}")
    annual = read_csv(args.input_dir, "Scenario_AnnualBurden")
    names = list(annual["scenario"].unique())

    present = usages_in(names)
    if not present:
        raise SystemExit("No IG2 arms found in the annual csv.")
    if args.usage == "all":
        usages = present
    elif args.usage in present:
        usages = [args.usage]
    else:
        print(f"  NOTE: usage '{args.usage}' is not in the data; using '{present[0]}'.")
        usages = [present[0]]
    if COMPARATOR == "IRS" and not irs_arms(names):
        print("  NOTE: no IRS_<c> arms in the data; comparing with BAU instead.")
        COMPARATOR = "BAU"
    print(f"Net-use assumptions in data: {present}; making Figures 1-5 for: {usages}; "
          f"comparator: {COMPARATOR}")

    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 11,
        "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.9, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    window = f"{args.window_start}\u2013{int(annual['Year'].max())}"
    n_pairs = annual["pair_id"].nunique() if "pair_id" in annual.columns else 0
    pf = read_csv(args.input_dir, "PfPR_ClinicalIncidence_monthly", required=False)
    win = read_csv(args.input_dir, "Scenario_WindowMetrics", required=False)
    eff = read_csv(args.input_dir, "IG2_EffectiveUse", required=False)

    def folder(tag):
        if args.output_dir:
            return os.path.join(args.output_dir, tag) if len(usages) > 1 or tag.startswith(
                "comparison") else args.output_dir
        return os.path.join(args.input_dir, f"EFFIG2_figures_{tag}_vs{COMPARATOR}")

    # ---- Figures 1-5: one set per net-use assumption -------------------------
    for usage in usages:
        PRIMARY_USAGE = usage
        COMPARE_COVERAGES = sorted({cov_of(n) for n in ig2_arms(names, usage)})
        out_dir = folder(usage_file_tag(usage))
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nWriting Figures 1, 2, 3, 5 — {usage_tag()}:")
        run("Figure 1", figure1, annual, out_dir)
        if pf is None:
            print("  skipped Figure 2: PfPR csv not found")
        else:
            run("Figure 2", figure2, pf, out_dir)
        if win is None:
            print("  skipped Figure 3: window-metrics csv not found")
        else:
            run("Figure 3", figure3, win, out_dir, window)
        run("Figure 5", figure5, annual, out_dir, args.population)
        write_captions(out_dir, n_pairs, window, args.population)
        print(f"  -> {out_dir}")

    # ---- Figures 6-8: compare the net-use assumptions (written once) --------
    PRIMARY_USAGE = "full" if "full" in present else present[0]
    out_dir = folder("comparison_netuse_assumptions")
    os.makedirs(out_dir, exist_ok=True)
    print("\nWriting Figures 4, 6, 7, 8 (compare net-use assumptions):")
    if eff is None:
        print("  skipped Figure 6: effective-use csv not found")
    else:
        run("Figure 6", figure6, eff, out_dir)
    if win is None:
        print("  skipped Figures 4, 7 and 8: window-metrics csv not found")
    else:
        run("Figure 4", figure4, win, out_dir, window)
        run("Figure 7", figure7, win, annual, out_dir, window, args.population)
        run("Figure 8", figure8, win, eff, out_dir, window)
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()