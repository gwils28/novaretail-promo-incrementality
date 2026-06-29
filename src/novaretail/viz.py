"""Figures in French on the plot style of `causal_ts.viz` (three hues at most, direct labels, SVG export)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causal_ts import viz as base

AQUA = base.AQUA
BLUE = base.BLUE
CRITICAL = base.CRITICAL
GOOD = base.GOOD
GRID = base.GRID
MUTED = base.MUTED
ORANGE = base.ORANGE
TEXT_2 = base.TEXT_2
setup = base.setup
exporter = base.exporter
mark_intervention = base.mark_intervention


def plot_impact(result, title: str | None = None, campaign_end=None, figsize=(10, 8), units: str = "log1p(ventes)"):
    """The three CausalImpact panels with French labels and the campaign window shaded."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    y = result.y
    ax = axes[0]
    ax.plot(y.index, y.values, color=BLUE, label="observé")
    if result.fit is not None:
        ax.plot(result.fit.index, result.fit["mean"], color=ORANGE, linewidth=1.2, alpha=0.8, label="ajustement un pas")
    ax.plot(result.cf.index, result.cf["mean"], color=ORANGE, linestyle="--", label="contrefactuel")
    ax.fill_between(result.cf.index, result.cf["lower"], result.cf["upper"], color=ORANGE, alpha=0.18, linewidth=0)
    ax.set_title(title or "Observé et contrefactuel")
    ax.set_ylabel(units)
    ax.legend(loc="upper left", ncol=3)
    pw, cu = result.pointwise(), result.cumulative()
    for a, df, t in ((axes[1], pw, "Effet ponctuel (observé moins contrefactuel)"), (axes[2], cu, "Effet cumulé")):
        a.axhline(0, color=TEXT_2, linewidth=1)
        a.plot(df.index, df["effect"], color=BLUE)
        a.fill_between(df.index, df["lower"], df["upper"], color=BLUE, alpha=0.18, linewidth=0)
        a.set_title(t)
    for a in axes:
        mark_intervention(a, result.pre_end, "début" if a is axes[0] else None)
        if campaign_end is not None:
            a.axvspan(result.pre_end, pd.Timestamp(campaign_end), color=GRID, alpha=0.5, linewidth=0)
    fig.tight_layout()
    return fig, axes


def plot_units_effects(units: dict, title: str = "Effet en unités : pendant, après, net"):
    """Three bars with intervals: lift during the campaign, dip after, net."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = ["during", "after", "net"]
    labels = ["pendant la campagne", "14 jours après", "net"]
    vals = [units[f"{n}_units"] for n in names]
    lo = [units[f"{n}_units"] - units[f"{n}_lower"] for n in names]
    hi = [units[f"{n}_upper"] - units[f"{n}_units"] for n in names]
    colors = [BLUE, ORANGE, AQUA]
    ax.bar(labels, vals, color=colors, yerr=[lo, hi], capsize=4, error_kw={"ecolor": TEXT_2, "elinewidth": 1})
    ax.axhline(0, color=TEXT_2, linewidth=1)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("unités vendues")
    fig.tight_layout()
    return fig, ax


def plot_placebo_sweep(sweep: pd.DataFrame, real: dict | None = None,
                       title: str = "Placebo dans le temps : fausses dates"):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(sweep))
    col = np.where(sweep["excludes_zero"], CRITICAL, BLUE)
    ax.errorbar(x, sweep["rel_effect"] * 100, yerr=[(sweep["rel_effect"] - sweep["rel_lower"]) * 100,
                (sweep["rel_upper"] - sweep["rel_effect"]) * 100], fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=3)
    ax.scatter(x, sweep["rel_effect"] * 100, c=col, zorder=3)
    ax.axhline(0, color=TEXT_2, linewidth=1)
    ax.set_xticks(x, [d.strftime("%d/%m/%y") for d in sweep.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("effet relatif (%)")
    if real is not None:
        ax.axhspan(real["rel_lower"] * 100, real["rel_upper"] * 100, color=ORANGE, alpha=0.15, linewidth=0)
        ax.axhline(real["rel_effect"] * 100, color=ORANGE, linestyle="--", linewidth=1.2)
        ax.text(x[-1], real["rel_effect"] * 100, " effet réel", color=ORANGE, va="bottom", ha="right", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_space_ranking(table: pd.DataFrame, title: str = "Placebo dans l'espace : chaque donneur traité à son tour"):
    df = table.sort_values("rel_effect")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.5))
    y = np.arange(len(df))
    flags = zip(df["is_treated"], df["excludes_zero"], strict=True)
    colors = [ORANGE if t else (CRITICAL if e else MUTED) for t, e in flags]
    ax.hlines(y, df["rel_lower"] * 100, df["rel_upper"] * 100, color=colors, linewidth=2)
    ax.scatter(df["rel_effect"] * 100, y, c=colors, zorder=3)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, [str(i) for i in df.index], fontsize=8)
    ax.set_xlabel("effet relatif propre (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_coverage(table: pd.DataFrame, title: str = "Couverture à 95 % par estimateur (vérité connue)"):
    """Bars of coverage with the nominal line; `table` indexed by estimator with columns coverage, width_rel."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    df = table.sort_values("coverage")
    colors = [GOOD if abs(c - 0.95) <= 0.03 else (ORANGE if c >= 0.85 else CRITICAL) for c in df["coverage"]]
    ax.barh(df.index, df["coverage"], color=colors)
    ax.axvline(0.95, color=TEXT_2, linestyle="--", linewidth=1)
    for i, c in enumerate(df["coverage"]):
        ax.text(c + 0.01, i, f"{c:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("part des intervalles qui contiennent la vérité")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_zero_map(share: pd.DataFrame, title: str = "Part de jours à ventes nulles par magasin et famille"):
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(share) + 1.5))
    im = ax.imshow(share.values, aspect="auto", cmap=base.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_yticks(range(len(share)), [str(i) for i in share.index], fontsize=7)
    ax.set_xticks(range(share.shape[1]), [str(c) for c in share.columns], rotation=90, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_campaign_detection(count: pd.Series, active: pd.Series, runs_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(count.index, count.values, color=MUTED, linewidth=1, label="onpromotion")
    ax.scatter(count.index[active.values], count.values[active.values], color=BLUE, s=8, label="jour actif", zorder=3)
    for r in runs_df.itertuples():
        ax.axvspan(r.start, r.end, color=AQUA if r.eligible else ORANGE, alpha=0.35, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("articles en promotion")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig, ax
