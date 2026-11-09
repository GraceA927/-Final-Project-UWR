"""
UWR figures: IG2 DISTRIBUTION coverage versus EFFECTIVE USE, and IRS vs IG2.

Reproduces the figure set of analyzer_scenarios_EFFIG2_plot.py for the final
16-arm experiment (Pickup13yrs_Scenario_Final: IRS / IG2 at 60, 80, 90%,
BAU, NO_IRS, each with current and +5 pp case management), in the same visual
style as plot_scenarios_publication.py.

What is different from the reference script
--------------------------------------------
* Coverage levels are 60/80/90 (were 50/70/90); IRS benchmarks IRS_90/80/60.
* Only current-CM arms are used (the _CM5 arms answer a different question and
  are in plot_scenarios_publication.py). Use --cm 5 to draw the CM5 arms.
* "Baseline" is NO_IRS (no IRS, no IG2), as in the reference; BAU is also drawn.
* Effective use is taken from TWO sources and compared with distribution:
    analytic   c x mean net retention (Weibull discard, k=3, L=1095 d) over the
               same window the simulation covers. IG2_USE_CURVES is False in
               run_scenarios_UWR.py, so no age/season multipliers are applied.
               (IG2_EffectiveUse_scenarios.csv was written by the analyzer with
               IG2_USE_CURVES = True and is therefore NOT used here.)
    simulated  model output: daily Bednet_Using (ReportEventCounter) divided
               by Statistical Population, IG2 arm minus NO_IRS in the same
               pickup state (removes routine ANC/CWC nets).
* Paired IG2-vs-IRS changes are computed here from the window metrics
  (the paired-impact csv only has BAU / NO_IRS as references).

Figures (PNG 600 dpi + PDF) in <in-dir>/effective_use_figures/
  Figure1_AnnualIncidence_IRSvsIG2
  Figure2_AnnualPfPR_byAgeGroup
  Figure3_PairedChange_vs_IRSBenchmark
  Figure4_Outcome_vs_Distribution_and_EffectiveUse
  Figure5_CasesAverted_IRSvsIG2         (+ Figure5_cases_summary.csv)
  Figure6_Distribution_vs_EffectiveUse  (+ IG2_DistributionVsEffectiveUse.csv)
  Figure7_IG2_NetUse_overTime
  figure_captions.md

Usage
  python plot_scenarios_effective_use.py
  python plot_scenarios_effective_use.py --in-dir <AnalyzerOutput> --titles
"""

import argparse
import os
import textwrap
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

import plot_scenarios_publication as P   # shared palette, rcParams, helpers

# =============================================================================
# SETTINGS (must match run_scenarios_UWR.py)
# =============================================================================

DEFAULT_IN_DIR = P.DEFAULT_IN_DIR
OUT_SUBDIR = "effective_use_figures"
STEP = P.STEP

COVERAGES = [60, 80, 90]
IRS_BENCHMARKS = ["IRS_90", "IRS_80", "IRS_60"]
NO_ARM, BAU = P.NO_IRS, P.BASELINE
BASELINE_LABEL = "Baseline (no IRS, no IG2)"

IRS_SWITCH_YEAR = P.IRS_SWITCH_YEAR
PROJ_START_YEAR, PROJ_END_YEAR = P.PROJ_START_YEAR, P.PROJ_END_YEAR
IG2_CAMPAIGN_YEARS = [2027, 2030]
IG2_CAMPAIGN_DOY = 91                 # 1 April
IG2_DISCARD_K, IG2_DISCARD_L = 3.0, 1095.0
IG2_USE_CURVES = False                # as simulated (run_scenarios_UWR.py)

PRIMARY = "incidence_per_1000_py"
LOW_Q, HIGH_Q = P.LOW_Q, P.HIGH_Q
DAYS_PER_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

AGE_GROUPS = {                         # summary-report bin upper edges (years)
    "Under 5 years": [0.25, 0.5, 5.0],
    "5–14 years": [15.0],
    "15 years and older": [25.0, 100.0],
}

COLORS, TEXT, TEXT_2 = P.COLORS, P.TEXT, P.TEXT_2
SIM_COLOR = COLORS["IG2"]             # simulated effective use
ANA_COLOR = "#8a8984"                 # analytic estimate (neutral)


# =============================================================================
# DATA
# =============================================================================

def read(in_dir, prefix, usecols=None, required=True):
    path = os.path.join(in_dir, f"{prefix}_{STEP}.csv")
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(path)
        print(f"  NOTE: {os.path.basename(path)} not found")
        return None
    df = pd.read_csv(path, usecols=usecols)
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str)
    if "pair_id" not in df.columns and "pickup_sim_index" in df.columns:
        df["pair_id"] = df["pickup_sim_index"].astype(int)
    return df


def arms_for_cm(names, cm):
    """{('IRS', 60): 'IRS_60', ('IG2', 60): 'IG2_60_cfp', ('NO_IRS', None): ...}"""
    out = {}
    for n in names:
        a = P.parse_arm(n)
        if a["cm"] != cm:
            continue
        cov = None if np.isnan(a["coverage"]) else int(a["coverage"])
        out[(a["family"], cov)] = n
    return out


def keep_cm(df, cm):
    return df[df["scenario"].map(lambda s: P.parse_arm(s)["cm"] == cm)]


def summarise(df, keys, value):
    g = df.groupby(keys)[value]
    return pd.DataFrame({"median": g.median(), "lo": g.quantile(LOW_Q),
                         "hi": g.quantile(HIGH_Q)}).reset_index()


def label(arm_key):
    fam, cov = arm_key
    if fam == NO_ARM:
        return BASELINE_LABEL
    if fam == "BAU":
        return "Business as usual (current IRS)"
    if fam == "IRS":
        return f"IRS ({cov}% of households sprayed)"
    return f"IG2 ({cov}% of population given a net)"


def style(arm_key):
    fam, cov = arm_key
    col = P.arm_color(fam, np.nan if cov is None else cov)
    ls = {"NO_IRS": "--", "BAU": "-", "IRS": "-", "IG2": "-"}[fam]
    return dict(color=col, ls=ls, marker=P.MARKERS[fam], ms=3.5, mec="white", mew=0.5,
                lw=2.0 if fam in ("IRS", "IG2") else 1.6)


# ---- effective use ----------------------------------------------------------

def day_calendar():
    """Flat 365-day calendar of the projection: model day -> (Year, Month, doy)."""
    n = (PROJ_END_YEAR - PROJ_START_YEAR + 1) * DAYS_PER_YEAR
    months = np.repeat(np.arange(1, 13), MONTH_DAYS)
    d = np.arange(n)
    return pd.DataFrame({"day": d, "Year": PROJ_START_YEAR + d // DAYS_PER_YEAR,
                         "Month": months[d % DAYS_PER_YEAR], "doy": d % DAYS_PER_YEAR + 1})


def analytic_retention_daily():
    """Share of distributed nets still held each day, re-set at every campaign."""
    cal = day_calendar()
    starts = [(y - PROJ_START_YEAR) * DAYS_PER_YEAR + IG2_CAMPAIGN_DOY - 1
              for y in IG2_CAMPAIGN_YEARS]
    ret = np.zeros(len(cal))
    for s in starts:                              # later campaigns overwrite
        t = cal["day"].to_numpy() - s
        on = t >= 0
        ret[on] = np.exp(-((t[on] + 0.5) / IG2_DISCARD_L) ** IG2_DISCARD_K)
    cal["retention"] = ret
    cal["after_first"] = cal["day"] >= starts[0]
    return cal


def simulated_net_use(in_dir, cm):
    """Monthly IG2-attributable net use per person, per pair (IG2 arm - NO_IRS)."""
    nu = read(in_dir, "NetUse_monthly", required=False,
              usecols=["sim_id", "model_month", "Year", "Month", "Bednet_Using",
                       "scenario", "pickup_sim_index"])
    ic = read(in_dir, "All_Age_InsetChart", required=False,
              usecols=["sim_id", "model_month", "Statistical Population"])
    if nu is None or ic is None:
        return None
    m = nu.merge(ic, on=["sim_id", "model_month"], how="inner")
    m["use"] = m["Bednet_Using"] / m["Statistical Population"]
    piv = m.pivot_table(index=["pair_id", "Year", "Month"], columns="scenario", values="use")
    arms = arms_for_cm(piv.columns, cm)
    ref = arms.get((NO_ARM, None))
    if ref is None:
        return None
    rows = []
    for c in COVERAGES:
        name = arms.get(("IG2", c))
        if name is None:
            continue
        d = (piv[name] - piv[ref]).rename("ig2_use").reset_index()
        d["coverage"] = c
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["t"] = out["Year"] + (out["Month"] - 0.5) / 12
    return out


def effective_use_table(sim):
    """One row per coverage: distribution vs analytic vs simulated effective use."""
    cal = analytic_retention_daily()
    win = cal[cal["after_first"]]
    ret_window = float(win["retention"].mean())
    t = np.arange((IG2_CAMPAIGN_YEARS[1] - IG2_CAMPAIGN_YEARS[0]) * DAYS_PER_YEAR) + 0.5
    ret_cycle = float(np.exp(-(t / IG2_DISCARD_L) ** IG2_DISCARD_K).mean())
    first = (IG2_CAMPAIGN_YEARS[0], int(cal.loc[cal["doy"] == IG2_CAMPAIGN_DOY, "Month"].iloc[0]))
    rows = []
    for c in COVERAGES:
        r = dict(coverage_pct=c, distribution=c / 100,
                 analytic_at_distribution=c / 100,
                 analytic_cycle_avg=c / 100 * ret_cycle,
                 analytic_window_avg=c / 100 * ret_window,
                 net_retention_cycle=ret_cycle, net_retention_window=ret_window)
        if sim is not None:
            s = sim[sim["coverage"] == c]
            mon = (s["Year"] * 100 + s["Month"]) >= first[0] * 100 + first[1]
            per_pair = s[mon].groupby("pair_id")["ig2_use"].mean()
            at = s[(s["Year"] == first[0]) & (s["Month"] == first[1])]["ig2_use"]
            r.update(sim_window_median=per_pair.median(),
                     sim_window_lo=per_pair.quantile(LOW_Q),
                     sim_window_hi=per_pair.quantile(HIGH_Q),
                     sim_at_distribution_median=at.median(),
                     sim_at_distribution_lo=at.quantile(LOW_Q),
                     sim_at_distribution_hi=at.quantile(HIGH_Q),
                     n_pairs=per_pair.size)
        rows.append(r)
    return pd.DataFrame(rows), cal


# =============================================================================
# FIGURE 1: annual incidence, IRS vs IG2 at matched coverage
# =============================================================================

def figure1(annual, arms, out_dir):
    s = summarise(annual, ["scenario", "Year"], PRIMARY)
    fig, axes = plt.subplots(1, len(COVERAGES), figsize=(P.WIDTH_2COL, 2.7),
                             sharey=True, squeeze=False)
    for k, (ax, c) in enumerate(zip(axes[0], COVERAGES)):
        for key in [(NO_ARM, None), ("BAU", None), ("IRS", c), ("IG2", c)]:
            name = arms.get(key)
            if name is None:
                print(f"  WARNING Figure 1: no arm {key}")
                continue
            d = s[s["scenario"] == name].sort_values("Year")
            st = style(key)
            ax.fill_between(d["Year"], d["lo"], d["hi"], color=st["color"], alpha=0.12, lw=0)
            ax.plot(d["Year"], d["median"], label=label(key) if key[0] in (NO_ARM, "BAU")
                    else key[0], zorder=3, **st)
        P.mark_switch(ax, text=False)
        ax.set_xticks(range(PROJ_START_YEAR, PROJ_END_YEAR + 1))
        ax.set_xticklabels([f"'{y % 100:02d}" for y in range(PROJ_START_YEAR, PROJ_END_YEAR + 1)])
        ax.set_xlim(PROJ_START_YEAR - 0.5, PROJ_END_YEAR + 0.3)
        ax.set_xlabel("Year")
        ax.set_title(f"{c}% coverage", loc="left", fontsize=8.5, pad=6)
        if k == 0:
            ax.set_ylabel("Clinical incidence\n(cases per 1,000 person-years)")
        ax.set_ylim(bottom=0)
        P.style_axis(ax)
        P.panel_letter(ax, "ABC"[k])
    h = [Line2D([], [], **{**style((NO_ARM, None)), "mec": None}, label=BASELINE_LABEL),
         Line2D([], [], **{**style(("BAU", None)), "mec": None}, label="Business as usual"),
         Line2D([], [], **{**style(("IRS", 90)), "mec": None}, label="IRS at panel coverage"),
         Line2D([], [], **{**style(("IG2", 90)), "mec": None}, label="IG2 at panel coverage")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    P.save(fig, out_dir, "Figure1_AnnualIncidence_IRSvsIG2",
           "Annual clinical malaria incidence: IRS vs IG2 at matched coverage")


# =============================================================================
# FIGURE 2: annual PfPR by age group
# =============================================================================

def grouped_pfpr(pf):
    pf = pf.copy()
    pf["_bin"] = pf["agebin"].round(4)
    scale = 100.0 if pf["PfPR"].max() <= 1.0 + 1e-9 else 1.0
    pf["_w"] = pf["Pop"] * pf["interval_days"]
    pf["_wp"] = pf["PfPR"] * pf["_w"]
    present = set(pf["_bin"].unique())
    out = []
    for grp, edges in AGE_GROUPS.items():
        edges = [round(e, 4) for e in edges]
        miss = [e for e in edges if e not in present]
        if miss:
            print(f"  NOTE Figure 2: bins {miss} for '{grp}' absent (have {sorted(present)})")
        sub = pf[pf["_bin"].isin(edges)]
        if sub.empty:
            continue
        a = sub.groupby(["sim_id", "scenario", "pair_id", "Year"], as_index=False)[["_wp", "_w"]].sum()
        a = a[a["_w"] > 0]
        a["pfpr_pct"] = a["_wp"] / a["_w"] * scale
        a["age_group"] = grp
        out.append(a)
    return pd.concat(out, ignore_index=True)


def figure2(pf, arms, out_dir):
    g = grouped_pfpr(pf)
    groups = [k for k in AGE_GROUPS if k in set(g["age_group"])]
    order = [(NO_ARM, None), ("BAU", None)] + [("IRS", c) for c in COVERAGES] + \
            [("IG2", c) for c in COVERAGES]
    fig, axes = plt.subplots(1, len(groups), figsize=(P.WIDTH_2COL, 2.7), squeeze=False)
    for k, (ax, grp) in enumerate(zip(axes[0], groups)):
        s = summarise(g[g["age_group"] == grp], ["scenario", "Year"], "pfpr_pct")
        top = s[s["scenario"].isin(arms.values())]["median"].max()
        for key in order:
            name = arms.get(key)
            if name is None:
                continue
            d = s[s["scenario"] == name].sort_values("Year")
            ax.plot(d["Year"], d["median"], zorder=3, **{**style(key), "lw": 1.4})
        P.mark_switch(ax, text=False)
        ax.set_xticks(range(PROJ_START_YEAR, PROJ_END_YEAR + 1))
        ax.set_xticklabels([f"'{y % 100:02d}" for y in range(PROJ_START_YEAR, PROJ_END_YEAR + 1)])
        ax.set_xlim(PROJ_START_YEAR - 0.5, PROJ_END_YEAR + 0.3)
        ax.set_xlabel("Year")
        ax.set_title(grp, loc="left", fontsize=8.5, pad=6)
        if k == 0:
            ax.set_ylabel("PfPR (%)")
        ax.set_ylim(0, top * 1.08)
        P.style_axis(ax)
        P.panel_letter(ax, "ABC"[k])
    h = [Line2D([], [], **{**style(key), "mec": None},
                label=P.arm_label(key[0], np.nan if key[1] is None else key[1], short=True)
                if key[0] != NO_ARM else "Baseline (no IRS, no IG2)")
         for key in order if key in arms]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    P.save(fig, out_dir, "Figure2_AnnualPfPR_byAgeGroup", "Annual PfPR by scenario and age group")


# =============================================================================
# FIGURE 3: paired change of IG2 relative to IRS benchmarks
# =============================================================================

def figure3(win, arms, out_dir, window):
    val = win.set_index(["pair_id", "scenario"])[PRIMARY].unstack()
    refs = [r for r in IRS_BENCHMARKS if r in val.columns]
    fig, axes = plt.subplots(1, len(refs), figsize=(P.WIDTH_2COL, 2.6),
                             sharey=True, squeeze=False)
    rows = []
    for k, (ax, ref) in enumerate(zip(axes[0], refs)):
        for i, c in enumerate(COVERAGES):
            name = arms.get(("IG2", c))
            if name is None:
                continue
            rel = ((val[name] - val[ref]) / val[ref] * 100).dropna()
            m, lo, hi = rel.median(), rel.quantile(LOW_Q), rel.quantile(HIGH_Q)
            rows.append(dict(reference=ref, ig2_coverage_pct=c, median=m, lo=lo, hi=hi,
                             share_ig2_lower=float((rel < 0).mean()), n=rel.size))
            ax.bar(i, m, width=0.62, color=P.arm_color("IG2", c), zorder=2)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=TEXT,
                        elinewidth=0.8, capsize=2.5, zorder=3)
            ax.annotate(f"{m:+.0f}%", xy=(i + 0.31, m), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=6.8, color=TEXT)
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_xticks(range(len(COVERAGES)))
        ax.set_xticklabels([f"{c}%" for c in COVERAGES])
        ax.set_xlabel("IG2 distribution coverage")
        ax.set_title(f"Relative to IRS {ref.split('_')[1]}%", loc="left", fontsize=8.5, pad=6)
        if k == 0:
            ax.set_ylabel(f"Paired change in clinical\nincidence, {window} (%)")
        P.style_axis(ax)
        P.panel_letter(ax, "ABC"[k])
    lo_y = min(ax.get_ylim()[0] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(lo_y * 1.12, max(5, ax.get_ylim()[1]))
    fig.text(0.5, -0.02, "Below zero: IG2 has lower incidence than the IRS benchmark in the same "
             f"realisation. Bars: median; error bars: {int(LOW_Q*100)}th–{int(HIGH_Q*100)}th "
             "percentile across paired pickup states.", ha="center", fontsize=6.8,
             color=TEXT_2, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    P.save(fig, out_dir, "Figure3_PairedChange_vs_IRSBenchmark",
           "Paired change in clinical incidence: IG2 relative to IRS benchmarks")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Figure3_paired_vs_IRS.csv"), index=False)


# =============================================================================
# FIGURE 4: outcome vs IG2 coverage - distributed vs effectively used
# =============================================================================

def figure4(win, arms, eff, out_dir, window):
    pf_col = "U5_pfpr" if "U5_pfpr" in win.columns else "pfpr_all_ages"
    scale = 100.0 if win[pf_col].max() <= 1.0 + 1e-9 else 1.0
    win = win.assign(pfpr_pct=win[pf_col] * scale)
    panels = [(PRIMARY, "Clinical incidence\n(cases per 1,000 person-years)"),
              ("pfpr_pct", "PfPR, children under 5 (%)" if pf_col == "U5_pfpr"
               else "PfPR, all ages (%)")]
    use_col = "sim_window_median" if "sim_window_median" in eff else "analytic_window_avg"
    use_lab = ("simulated effective use" if use_col.startswith("sim")
               else "estimated effective use")
    fig, axes = plt.subplots(1, 2, figsize=(P.WIDTH_2COL, 3.0))
    for k, (ax, (col, ylab)) in enumerate(zip(axes, panels)):
        s = summarise(win, ["scenario"], col).set_index("scenario")
        ig_names = [arms[("IG2", c)] for c in COVERAGES if ("IG2", c) in arms]
        irs_keys = [("IRS", c) for c in COVERAGES if ("IRS", c) in arms]
        top = max(s.loc[ig_names, "hi"].max(),
                  max(s.loc[arms[k], "median"] for k in irs_keys)) * 1.25
        ax.set_ylim(0, top)
        items = []
        for key in irs_keys:
            v = s.loc[arms[key], "median"]
            ax.axhline(v, color=style(key)["color"], lw=1.0, zorder=1)
            items.append((v, f"IRS {key[1]}%"))
        for v, txt in P.merge_close_labels(items, top * 0.05):
            ax.text(1.0, v, " " + txt, transform=ax.get_yaxis_transform(),
                    fontsize=6.3, color=TEXT_2, va="center")
        dec = 0 if col == PRIMARY else 1
        off = [f"{'Baseline' if k[0] == NO_ARM else 'BAU'} {s.loc[arms[k], 'median']:,.{dec}f}"
               for k in [(NO_ARM, None), ("BAU", None)] if k in arms]
        ax.text(0.02, 0.97, "Off scale (median): " + "; ".join(off),
                transform=ax.transAxes, fontsize=6.3, color=TEXT_2, va="top")
        ig = [(c, arms[("IG2", c)]) for c in COVERAGES if ("IG2", c) in arms]
        y = np.array([s.loc[n, "median"] for _, n in ig])
        lo = np.array([s.loc[n, "lo"] for _, n in ig])
        hi = np.array([s.loc[n, "hi"] for _, n in ig])
        x_dist = np.array([c for c, _ in ig], float)
        x_use = np.array([eff.set_index("coverage_pct").loc[c, use_col] * 100 for c, _ in ig])
        for xd, xu, yy in zip(x_dist, x_use, y):
            ax.annotate("", xy=(xu, yy), xytext=(xd, yy),
                        arrowprops=dict(arrowstyle="->", color=COLORS["IG2"], lw=0.8,
                                        shrinkA=4, shrinkB=4))
        ax.errorbar(x_dist, y, yerr=[y - lo, hi - y], fmt="s", color=COLORS["IG2"],
                    mfc=COLORS["IG2"], mec="white", ms=5.5, lw=1.2, capsize=2.5, zorder=4,
                    label="IG2 vs distribution coverage")
        ax.errorbar(x_use, y, yerr=[y - lo, hi - y], fmt="s", color=COLORS["IG2"],
                    mfc="white", mec=COLORS["IG2"], mew=1.2, ms=5.5, lw=1.2, capsize=2.5,
                    zorder=4, label=f"IG2 vs {use_lab}")
        ax.set_xlim(30, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlabel("Population coverage of IG2 nets")
        ax.set_ylabel(f"{ylab}, {window}")
        P.style_axis(ax, grid_axis="both")
        P.panel_letter(ax, "AB"[k])
    h = [Line2D([], [], color=COLORS["IG2"], marker="s", mec="white", ls="none",
                label="IG2 plotted at distribution coverage"),
         Line2D([], [], color=COLORS["IG2"], marker="s", mfc="white", ls="none",
                label=f"Same IG2 arm plotted at {use_lab} (time-averaged)"),
         Line2D([], [], color=P.arm_color("IRS", 90), label="IRS arms (median)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.11, 0.95, 1))
    P.save(fig, out_dir, "Figure4_Outcome_vs_Distribution_and_EffectiveUse",
           "Burden versus IG2 coverage: distributed vs effectively used")


# =============================================================================
# FIGURE 5: annual cases averted vs baseline
# =============================================================================

def figure5(annual, arms, out_dir, population):
    piv = annual.pivot_table(index=["pair_id", "Year"], columns="scenario", values=PRIMARY)
    ref = arms[(NO_ARM, None)]
    factor = population / 1000.0 if population else 1.0
    ylab = (f"Clinical cases averted\nper {population:,} population" if population
            else "Clinical cases averted\nper 1,000 person-years")
    years = [y for y in sorted(piv.index.get_level_values("Year").unique()) if y >= IRS_SWITCH_YEAR]
    mask = piv.index.get_level_values("Year") >= IRS_SWITCH_YEAR
    fig, axes = plt.subplots(1, len(COVERAGES), figsize=(P.WIDTH_2COL, 2.7),
                             sharey=True, squeeze=False)
    rows = []
    for k, (ax, c) in enumerate(zip(axes[0], COVERAGES)):
        fams = [f for f in ("IRS", "IG2") if (f, c) in arms]
        w = 0.8 / max(len(fams), 1)
        txt = []
        for j, fam in enumerate(fams):
            name = arms[(fam, c)]
            av = ((piv[ref] - piv[name]) * factor)[mask]
            g = av.groupby(level="Year")
            med, lo, hi = g.median(), g.quantile(LOW_Q), g.quantile(HIGH_Q)
            cum = av.groupby(level="pair_id").sum()
            xs = np.arange(len(years)) + (j - (len(fams) - 1) / 2) * w
            hgt = med.loc[years].to_numpy()
            ax.bar(xs, hgt, width=w * 0.9, color=P.arm_color(fam, c), zorder=2,
                   hatch="////" if fam == "IG2" else None, edgecolor="white", lw=0.4)
            ax.errorbar(xs, hgt, yerr=[hgt - lo.loc[years].to_numpy(),
                                       hi.loc[years].to_numpy() - hgt],
                        fmt="none", ecolor=TEXT, elinewidth=0.6, capsize=1.5, zorder=3)
            for y, v in zip(years, hgt):
                rows.append(dict(coverage_pct=c, arm=fam, scenario=name, baseline=ref,
                                 period=str(y), median_cases_averted=v))
            rows.append(dict(coverage_pct=c, arm=fam, scenario=name, baseline=ref,
                             period=f"cumulative {years[0]}-{years[-1]}",
                             median_cases_averted=cum.median(),
                             lo=cum.quantile(LOW_Q), hi=cum.quantile(HIGH_Q)))
            txt.append(f"{fam}: {cum.median():,.0f}")
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([f"'{y % 100:02d}" for y in years])
        ax.set_xlabel("Year")
        ax.set_title(f"{c}% coverage", loc="left", fontsize=8.5, pad=14)
        ax.text(0.0, 1.02, f"Cumulative {years[0]}–{years[-1] % 100:02d}: " + ", ".join(txt),
                transform=ax.transAxes, fontsize=6.3, color=TEXT_2, va="bottom")
        if k == 0:
            ax.set_ylabel(ylab)
        P.style_axis(ax)
        P.panel_letter(ax, "ABC"[k])
    h = [matplotlib.patches.Patch(fc=P.arm_color("IRS", 90), label="IRS"),
         matplotlib.patches.Patch(fc=P.arm_color("IG2", 90), hatch="////", ec="white",
                                  label="IG2 nets")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    P.save(fig, out_dir, "Figure5_CasesAverted_IRSvsIG2",
           "Annual clinical cases averted vs baseline (no IRS, no IG2)")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "Figure5_cases_summary.csv"), index=False)


# =============================================================================
# FIGURE 6: distribution coverage vs effective use
# =============================================================================

def figure6(eff, out_dir):
    x = eff["coverage_pct"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(P.WIDTH_1COL + 0.6, P.WIDTH_1COL + 0.4))
    ax.plot([0, 100], [0, 100], color=TEXT_2, ls=":", lw=0.9, zorder=1)
    ax.text(97, 97, "use = distribution", rotation=45, rotation_mode="anchor",
            ha="right", va="bottom", fontsize=6.3, color=TEXT_2,
            transform_rotates_text=True)
    ax.plot(x, eff["analytic_window_avg"] * 100, color=ANA_COLOR, ls="--", marker="D",
            ms=4, lw=1.2, label="Estimated: coverage × net retention", zorder=3)
    if "sim_window_median" in eff:
        m = eff["sim_at_distribution_median"] * 100
        ax.errorbar(x, m, yerr=[m - eff["sim_at_distribution_lo"] * 100,
                                eff["sim_at_distribution_hi"] * 100 - m],
                    fmt="o", color=SIM_COLOR, mfc="white", mew=1.2, ms=5, capsize=2,
                    lw=1, label="Simulated: month of distribution", zorder=4)
        m = eff["sim_window_median"] * 100
        ax.errorbar(x, m, yerr=[m - eff["sim_window_lo"] * 100, eff["sim_window_hi"] * 100 - m],
                    fmt="-s", color=SIM_COLOR, mec="white", ms=5.5, capsize=2, lw=1.6,
                    label=f"Simulated: average Apr {IG2_CAMPAIGN_YEARS[0]}–Dec {PROJ_END_YEAR}",
                    zorder=5)
        for xi, v in zip(x, m):
            ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(6, -8), textcoords="offset points",
                        fontsize=6.8, color=SIM_COLOR)
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("IG2 distribution coverage\n(share of population given a net)")
    ax.set_ylabel("Effective use of IG2 nets\n(share of population sleeping under one)")
    P.style_axis(ax, grid_axis="both")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=6.6)
    fig.tight_layout()
    P.save(fig, out_dir, "Figure6_Distribution_vs_EffectiveUse",
           "IG2 distribution coverage and effective use")


# =============================================================================
# FIGURE 7: monthly simulated IG2 net use over time
# =============================================================================

def figure7(sim, cal, out_dir):
    s = sim.groupby(["coverage", "t"])["ig2_use"].agg(
        median="median", lo=lambda v: v.quantile(LOW_Q), hi=lambda v: v.quantile(HIGH_Q)
    ).reset_index()
    mon = cal.groupby(["Year", "Month"], as_index=False)["retention"].mean()
    mon["t"] = mon["Year"] + (mon["Month"] - 0.5) / 12
    fig, ax = plt.subplots(figsize=(P.WIDTH_2COL, 2.9))
    for c in COVERAGES:
        col = P.arm_color("IG2", c)
        d = s[s["coverage"] == c]
        ax.fill_between(d["t"], d["lo"] * 100, d["hi"] * 100, color=col, alpha=0.18, lw=0)
        ax.plot(d["t"], d["median"] * 100, color=col, lw=1.8, zorder=3)
        ax.plot(mon["t"], mon["retention"] * c, color=col, ls="--", lw=0.9, zorder=2)
        ax.axhline(c, color=col, ls=":", lw=0.8, zorder=1)
        ax.text(PROJ_END_YEAR + 1.02, c, f"{c}% distributed", fontsize=6.3, color=TEXT_2,
                va="center")
        last = d.iloc[-1]
        ax.text(PROJ_END_YEAR + 1.02, last["median"] * 100, f"IG2 {c}%: {last['median']*100:.0f}%",
                fontsize=6.3, color=TEXT_2, va="center")
    for y in IG2_CAMPAIGN_YEARS:
        x = y + (IG2_CAMPAIGN_DOY - 1) / 365
        ax.plot([x], [1.0], marker="v", ms=5, color=COLORS["IG2"], clip_on=False,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(PROJ_START_YEAR, PROJ_END_YEAR + 1)
    ax.set_xticks(range(PROJ_START_YEAR, PROJ_END_YEAR + 2))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Year")
    ax.set_ylabel("Population using an IG2 net")
    P.style_axis(ax)
    h = [Line2D([], [], color=COLORS["IG2"], lw=1.8, label="Simulated use (median, 80% band)"),
         Line2D([], [], color=COLORS["IG2"], ls="--", lw=0.9,
                label="Estimated: coverage × net retention"),
         Line2D([], [], color=COLORS["IG2"], ls=":", lw=0.8, label="Distribution coverage"),
         Line2D([], [], color=COLORS["IG2"], marker="v", lw=0, label="IG2 mass distribution")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 0.88, 1))
    P.save(fig, out_dir, "Figure7_IG2_NetUse_overTime",
           "Simulated IG2 net use over time versus distribution coverage")


# =============================================================================
# CAPTIONS
# =============================================================================

def write_captions(out_dir, eff, n_pairs, window, cm):
    ui = f"{int(LOW_Q*100)}th–{int(HIGH_Q*100)}th percentile"
    cm_txt = "current case management" if cm == 0 else f"case management +{cm} pp"
    rw = eff["net_retention_window"].iloc[0]
    rc = eff["net_retention_cycle"].iloc[0]
    sim_txt = ""
    if "sim_window_median" in eff:
        sim_txt = "; ".join(f"{int(r.coverage_pct)}% distributed → "
                            f"{r.sim_window_median*100:.0f}% simulated average use "
                            f"(estimate {r.analytic_window_avg*100:.0f}%)"
                            for r in eff.itertuples())
    caps = [
        ("Figure 1. Annual clinical malaria incidence, IRS versus IG2 at matched coverage.",
         f"Median (lines) and {ui} (bands) across {n_pairs} paired pickup states. Each panel "
         f"compares IRS continued at the stated share of households sprayed with IG2 nets "
         f"distributed to the same share of the population, from 1 January {IRS_SWITCH_YEAR}, "
         f"against the baseline with neither (dashed) and business as usual. {cm_txt.capitalize()}."),
        ("Figure 2. Annual PfPR by scenario and age group.",
         f"Median across {n_pairs} pickup states; PfPR averaged over reporting intervals within "
         f"each year, weighted by population and interval length."),
        ("Figure 3. Paired change in clinical incidence of IG2 relative to IRS benchmarks.",
         f"Within-realisation relative change ({window}) of each IG2 arm against IRS at 90%, 80% "
         f"and 60% coverage. Bars: median; error bars: {ui}. Below zero favours IG2."),
        ("Figure 4. Burden against IG2 coverage, expressed as distribution and as effective use.",
         f"Median clinical incidence (A) and PfPR (B) over {window} for each IG2 arm, plotted "
         f"twice: at the share of the population given a net (filled) and at the time-averaged "
         f"share actually using one in the model (open; arrows). Horizontal lines: IRS arms, "
         f"baseline and business as usual are off scale and given as numbers. Error bars: {ui}."),
        ("Figure 5. Annual clinical cases averted, IRS versus IG2 at matched coverage.",
         f"Baseline (no IRS, no IG2) incidence minus arm incidence, within each pickup state; "
         f"bars are medians, error bars the {ui}. Cumulative totals in Figure5_cases_summary.csv."),
        ("Figure 6. IG2 distribution coverage and effective use.",
         f"Effective use is the share of the population sleeping under an IG2 net. Simulated: "
         f"daily model count of people using a net (Bednet_Using) divided by population, IG2 arm "
         f"minus the baseline arm in the same pickup state (removes routine ANC/CWC nets), in the "
         f"month of the first distribution (open circles) and averaged from April "
         f"{IG2_CAMPAIGN_YEARS[0]} to December {PROJ_END_YEAR} (filled squares; {ui}). Estimated: "
         f"distribution coverage × mean net retention over the same period ({rw:.2f}; "
         f"{rc:.2f} over one 3-year cycle) from the Weibull discard (k = {IG2_DISCARD_K:g}, "
         f"λ = {IG2_DISCARD_L:g} days). No age or seasonal usage curves were applied to IG2 "
         f"nets in the simulations. {sim_txt}."),
        ("Figure 7. Simulated IG2 net use over time.",
         f"Monthly share of the population using an IG2 net (solid, median and {ui}), the "
         f"retention-based estimate (dashed) and the distribution coverage (dotted). Triangles: "
         f"mass distributions (1 April {', '.join(map(str, IG2_CAMPAIGN_YEARS))}). The simulated "
         f"count is per net, so a person holding two nets is counted twice."),
    ]
    with open(os.path.join(out_dir, "figure_captions.md"), "w", encoding="utf-8") as fh:
        fh.write("# Figure captions: IG2 distribution vs effective use\n\n")
        for head, body in caps:
            fh.write(f"**{head}**\n" + textwrap.fill(body, 100) + "\n\n")
    print("  wrote figure_captions.md")


# =============================================================================
# MAIN
# =============================================================================

def run(name, fn, *args):
    try:
        fn(*args)
    except Exception as exc:                 # one failure must not stop the rest
        print(f"  FAILED {name}: {exc}")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in-dir", default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", default=None, help=f"default: <in-dir>/{OUT_SUBDIR}")
    ap.add_argument("--cm", type=int, default=0, help="0 = current CM arms, 5 = +5 pp arms")
    ap.add_argument("--population", type=int, default=None,
                    help="Figure 5 per this population (default per 1,000 person-years)")
    ap.add_argument("--titles", action="store_true")
    args = ap.parse_args()
    P.SHOW_TITLES = args.titles
    out_dir = args.out_dir or os.path.join(args.in_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {args.in_dir}")
    annual = keep_cm(read(args.in_dir, "Scenario_AnnualBurden"), args.cm)
    win = keep_cm(read(args.in_dir, "Scenario_WindowMetrics"), args.cm)
    arms = arms_for_cm(annual["scenario"].unique(), args.cm)
    n_pairs = annual["pair_id"].nunique()
    window = f"{IRS_SWITCH_YEAR}–{int(annual['Year'].max())}"
    print(f"Arms (CM +{args.cm} pp): {sorted(arms.values())}; pairs: {n_pairs}")

    sim = simulated_net_use(args.in_dir, args.cm)
    eff, cal = effective_use_table(sim)
    eff.to_csv(os.path.join(out_dir, "IG2_DistributionVsEffectiveUse.csv"), index=False)
    cols = [c for c in ("coverage_pct", "analytic_window_avg", "sim_at_distribution_median",
                        "sim_window_median", "sim_window_lo", "sim_window_hi") if c in eff]
    print("Distribution vs effective use:\n" + eff[cols].round(3).to_string(index=False))

    old = read(args.in_dir, "IG2_EffectiveUse", required=False)
    if old is not None and "age_multiplier" in old and (old["age_multiplier"] < 0.999).any():
        print("  NOTE: IG2_EffectiveUse_scenarios.csv was computed WITH age/season usage "
              "curves (analyzer IG2_USE_CURVES=True) but the simulations ran without them; "
              "it is ignored here. Set IG2_USE_CURVES=False in analyze_scenarios_UWR.py.")

    print("Writing figures:")
    run("Figure 1", figure1, annual, arms, out_dir)
    pf = read(args.in_dir, "PfPR_ClinicalIncidence_monthly", required=False)
    if pf is not None:
        run("Figure 2", figure2, keep_cm(pf, args.cm), arms, out_dir)
    run("Figure 3", figure3, win, arms, out_dir, window)
    run("Figure 4", figure4, win, arms, eff, out_dir, window)
    run("Figure 5", figure5, annual, arms, out_dir, args.population)
    run("Figure 6", figure6, eff, out_dir)
    if sim is not None:
        run("Figure 7", figure7, sim, cal, out_dir)
    else:
        print("  skipped Figure 7: NetUse / InsetChart csv not found")
    write_captions(out_dir, eff, n_pairs, window, args.cm)
    print(f"\nFigures written to {out_dir}")


if __name__ == "__main__":
    main()
