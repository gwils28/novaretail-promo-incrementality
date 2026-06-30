"""Validation of the design: placebos per campaign, planted campaigns on the real panel, pool sensitivity.

- placebo-in-time: the same model on fake dates inside the pre period (windows with a missing day are skipped,
  not filled). The share of fake dates whose interval excludes zero is the empirical false-positive rate of the
  design on this series, to be read against the rate measured on never-treated series, not against 5%.
- placebo-in-space and leave-one-out isolation: each donor in turn as the treated unit, with the other donors and
  the calendar as regressors; the treated series is never a donor for anybody. Donors whose own effect excludes
  zero are removed one at a time (smallest tail probability first), as in `causal_ts.diagnostics`.
- planted campaigns: a known multiplicative lift (with its pull-forward dip and, optionally, cannibalisation of
  one donor) written into a quiet window of a real series; every estimator is scored on the cumulative log
  effect by coverage, bias and power. Effect zero gives the false-positive rate on never-treated windows.
- pool sensitivity: the estimate as a function of the pool size and with each donor left out.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from causal_ts.baselines import diff_in_diff, ols, synthetic_control
from causal_ts.bsts import causal_impact
from causal_ts.dgp import effect_profile

from novaretail.campaigns import DetectionRule, campaign_days
from novaretail.donors import RegistryRule, eligible_donors, pool_units
from novaretail.estimate import assemble, augmented_sc
from novaretail.features import transform


def _quiet(**kw):
    warnings.simplefilter("ignore")
    return causal_impact(**kw)


def placebo_time(y: pd.Series, X: pd.DataFrame, campaign: pd.Series, n_dates: int = 8, min_pre: int = 60,
                 **kw) -> pd.DataFrame:
    """Fake intervention dates in the pre period, each with the campaign's window length; truncated at the start."""
    pre_end = campaign["start"] - pd.Timedelta(1, unit="D")
    y_pre, X_pre = y.loc[:pre_end], X.loc[:pre_end]
    post_len = int(campaign["length"] + campaign["n_post"])
    last = len(y_pre) - post_len - 1
    if last <= min_pre:
        return pd.DataFrame()
    rows = []
    for pos in np.unique(np.linspace(min_pre, last, n_dates).astype(int)):
        fake, end = y_pre.index[pos], y_pre.index[pos + post_len]
        if y_pre.loc[fake:end].iloc[1:].isna().any():
            continue
        with warnings.catch_warnings():
            r = _quiet(y=y_pre.loc[:end], X=X_pre.loc[:end], pre_end=fake, post_end=end, **kw)
        s = r.summary()
        rows.append({"fake_pre_end": fake, "n_pre": pos + 1, "n_post": post_len, "rel_effect": s["rel_effect"],
                     "rel_lower": s["rel_lower"], "rel_upper": s["rel_upper"], "p_tail": s["p_tail"],
                     "excludes_zero": s["excludes_zero"]})
    return pd.DataFrame(rows).set_index("fake_pre_end") if rows else pd.DataFrame()


def _frame(wide_sales, calendars, campaign, pool, kind="log1p"):
    target = (int(campaign["store_nbr"]), str(campaign["family"]))
    y, X = assemble(wide_sales, calendars, campaign, pool, kind=kind)
    donors = X[[c for c in X.columns if c.startswith("d_")]]
    cal = X[[c for c in X.columns if not c.startswith("d_")]]
    frame = pd.concat([y.rename(f"{target[0]}:{target[1]}"), donors.set_axis([f"{s}:{f}" for s, f in pool], axis=1)],
                      axis=1)
    return frame, cal


def placebo_space(wide_sales, calendars, campaign: pd.Series, pool: list, include_treated: bool = True,
                  **kw) -> pd.DataFrame:
    """Each unit in turn as treated, the other donors plus the calendar as regressors."""
    frame, cal = _frame(wide_sales, calendars, campaign, pool)
    treated = frame.columns[0]
    pre_end = campaign["start"] - pd.Timedelta(1, unit="D")
    units = ([treated] if include_treated else []) + list(frame.columns[1:])
    rows = []
    for u in units:
        donors = [c for c in frame.columns[1:] if c != u]
        X = pd.concat([frame[donors], cal], axis=1)
        with warnings.catch_warnings():
            r = _quiet(y=frame[u], X=X, pre_end=pre_end, post_end=campaign["post_end"], **kw)
        s = r.summary()
        pw = r.pointwise()["effect"]
        pre_err = (r.y.loc[:pre_end] - r.fit["mean"]).dropna()
        rows.append({"unit": u, "is_treated": u == treated, "rel_effect": s["rel_effect"], "rel_lower": s["rel_lower"],
                     "rel_upper": s["rel_upper"], "p_tail": s["p_tail"], "excludes_zero": s["excludes_zero"],
                     "rmspe_ratio": float(np.sqrt(np.mean(pw**2)) / np.sqrt(np.mean(pre_err**2)))})
    df = pd.DataFrame(rows).set_index("unit")
    df["rank_p_rel"] = [float(np.mean(df["rel_effect"].abs() >= abs(v))) for v in df["rel_effect"]]
    df["rank_p_rmspe"] = [float(np.mean(df["rmspe_ratio"] >= v)) for v in df["rmspe_ratio"]]
    return df


def isolate(wide_sales, calendars, campaign: pd.Series, pool: list, max_rounds: int = 5,
            **kw) -> tuple[list, list, pd.DataFrame]:
    """Leave-one-out isolation of contaminated donors: (removed units, kept units, first-round table)."""
    kept = list(pool)
    removed = []
    first = None
    for _ in range(max_rounds):
        if len(kept) < 2:
            break
        tab = placebo_space(wide_sales, calendars, campaign, kept, include_treated=False, **kw)
        first = tab if first is None else first
        flagged = tab[tab["excludes_zero"]]
        if flagged.empty:
            break
        worst = flagged.sort_values(["p_tail", "rel_effect"], key=lambda s: s.abs() if s.name == "rel_effect" else s,
                                    ascending=[True, False]).index[0]
        s, f = worst.split(":", 1)
        unit = (int(s), f)
        removed.append(unit)
        kept.remove(unit)
    return removed, kept, first


def pool_sensitivity(wide_sales, calendars, campaign: pd.Series, table: pd.DataFrame, sizes=(3, 6, 9, 12),
                     leave_one_out: bool = True, **kw) -> pd.DataFrame:
    """Relative effect against the pool size (donors ranked by the registry) and with each donor left out."""
    admitted = table[table["reason"].isin(["kept", "beyond_cap"])]
    ranked = [(int(r.store_nbr), str(r.family)) for r in admitted.itertuples()]
    pre_end = campaign["start"] - pd.Timedelta(1, unit="D")
    rows = []

    def run(pool, label):
        y, X = assemble(wide_sales, calendars, campaign, pool)
        with warnings.catch_warnings():
            s = _quiet(y=y, X=X, pre_end=pre_end, post_end=campaign["post_end"], **kw).summary()
        rows.append({"spec": label, "n_donors": len(pool), "rel_effect": s["rel_effect"], "rel_lower": s["rel_lower"],
                     "rel_upper": s["rel_upper"], "excludes_zero": s["excludes_zero"]})

    run([], "aucun donneur (niveau local + calendrier)")
    for k in sizes:
        if k <= len(ranked):
            run(ranked[:k], f"{k} donneurs")
    if leave_one_out:
        base = pool_units(table)
        for u in base:
            run([d for d in base if d != u], f"sans {u[0]}:{u[1]}")
    return pd.DataFrame(rows).set_index("spec")


# --- planted campaigns on the real panel ---------------------------------------------------------------------
@dataclass(frozen=True)
class PlantSpec:
    n_pre: int = 120
    length: int = 7
    horizon: int = 14
    rel_lift: float = 0.10
    shape: str = "step"  # step, ramp, pulse (profiles of causal_ts.dgp)
    pull_forward: float = 0.0  # share of the units gained during the campaign given back over the horizon
    cannibal_share: float = 0.0  # share of the lift taken from the first donor of the pool
    min_volume: float = 30.0


def quiet_windows(wide_raw: pd.DataFrame, wide_promo: pd.DataFrame, usable: pd.DataFrame, spec: PlantSpec,
                  n: int, seed: int = 0, detection: DetectionRule = DetectionRule()) -> list[dict]:
    """Random (unit, start) pairs where nothing happened: no campaign day from 30 days before the pre period
    to the end of the horizon, every day usable, volume above the floor."""
    rng = np.random.default_rng(seed)
    units = list(wide_raw.columns)
    span = spec.n_pre + spec.length + spec.horizon
    out = []
    tries = 0
    camp_cache = {}
    while len(out) < n and tries < n * 50:
        tries += 1
        u = units[rng.integers(len(units))]
        t0 = int(rng.integers(spec.n_pre + 30, len(wide_raw) - spec.length - spec.horizon))
        if u not in camp_cache:
            camp_cache[u] = campaign_days(wide_promo[u], usable[u], detection)
        lo, hi = t0 - spec.n_pre - 30, t0 + spec.length + spec.horizon
        if camp_cache[u].iloc[lo:hi].any() or (~usable[u].iloc[lo:hi]).any():
            continue
        if wide_raw[u].iloc[t0 - spec.n_pre : t0].mean() < spec.min_volume:
            continue
        idx = wide_raw.index
        out.append({"store_nbr": u[0], "family": u[1], "start": idx[t0], "end": idx[t0 + spec.length - 1],
                    "length": spec.length, "pre_start": idx[t0 - spec.n_pre],
                    "post_end": idx[t0 + span - spec.n_pre - 1],
                    "n_post": spec.horizon, "mean_count": np.nan})
    return out


def plant(wide_raw: pd.DataFrame, window: dict, spec: PlantSpec, pool: list) -> tuple[pd.DataFrame, dict]:
    """Write the lift into a copy of the raw panel; return the panel and the truth on the log1p scale."""
    w = wide_raw.copy()
    unit = (window["store_nbr"], window["family"])
    prof = spec.rel_lift * effect_profile(spec.length, spec.shape)
    during = pd.date_range(window["start"], periods=spec.length, freq="D")
    after = pd.date_range(window["end"] + pd.Timedelta(1, unit="D"), periods=spec.horizon, freq="D")
    base_d = w.loc[during, unit].values
    base_a = w.loc[after, unit].values
    w.loc[during, unit] = np.round(base_d * (1 + prof))
    dip = spec.pull_forward * spec.rel_lift * spec.length / spec.horizon  # daily relative dip
    w.loc[after, unit] = np.round(base_a * (1 - dip))
    if spec.cannibal_share and pool:
        d0 = pool[0]
        w.loc[during, d0] = np.round(w.loc[during, d0].values * (1 - spec.cannibal_share * prof))
    truth_log = float(np.sum(transform(w.loc[during, unit].values) - transform(base_d))
                      + np.sum(transform(w.loc[after, unit].values) - transform(base_a)))
    return w, {"true_cum_log": truth_log, "true_rel_lift": spec.rel_lift, "true_dip": dip}


ESTIMATORS = {
    "bsts_llevel": lambda y, X, pre_end: _cf_from_bsts(y, X, pre_end, "llevel"),
    "bsts_static": lambda y, X, pre_end: _cf_from_bsts(y, X, pre_end, "static"),
    "ols": lambda y, X, pre_end: ols(y.interpolate(), X, pre_end),
    "synthetic_control": lambda y, X, pre_end: synthetic_control(y.interpolate(), _donor_cols(X), pre_end),
    "augmented_sc": lambda y, X, pre_end: augmented_sc(y, X, pre_end),
    "diff_in_diff": lambda y, X, pre_end: diff_in_diff(y.interpolate(), _donor_cols(X), pre_end),
}


def _donor_cols(X):
    return X[[c for c in X.columns if c.startswith("d_")]]


def _cf_from_bsts(y, X, pre_end, level):
    from causal_ts.baselines import CfEstimate

    with warnings.catch_warnings():
        r = _quiet(y=y, X=X, pre_end=pre_end, level=level, nsim=1000)
    s = r.summary()
    return CfEstimate(f"bsts_{level}", r.cf, s["effect_cum"], s["effect_cum_lower"], s["effect_cum_upper"])


def planted_study(wide_raw, wide_promo, usable, stores, calendars, specs: list[PlantSpec], n_per_spec: int,
                  estimators: list[str] | None = None, seed: int = 0, registry: RegistryRule = RegistryRule(),
                  detection: DetectionRule = DetectionRule()) -> pd.DataFrame:
    """One row per (planted campaign, estimator) with the estimate, interval, truth and the scores."""
    names = estimators or list(ESTIMATORS)
    rows = []
    for k, spec in enumerate(specs):
        for j, win in enumerate(quiet_windows(wide_raw, wide_promo, usable, spec, n_per_spec, seed + 1000 * k,
                                              detection)):
            camp = pd.Series(win)
            tab = eligible_donors(wide_raw, wide_promo, usable, stores, camp, registry, detection)
            pool = pool_units(tab) if len(tab) else []
            if len(pool) < 3:
                continue
            w, truth = plant(wide_raw, win, spec, pool)
            y, X = assemble(w.where(usable), calendars, camp, pool)
            pre_end = camp["start"] - pd.Timedelta(1, unit="D")
            for name in names:
                try:
                    est = ESTIMATORS[name](y, X, pre_end)
                except Exception as exc:  # noqa: BLE001 - a failed fit is a row, not a crash of the study
                    rows.append({"spec": k, "plant": j, "estimator": name, "error": repr(exc)})
                    continue
                rows.append({"spec": k, "plant": j, "estimator": name, "store_nbr": win["store_nbr"],
                             "family": win["family"], "start": win["start"], "n_donors": len(pool),
                             "estimate": est.cum_effect, "lower": est.cum_lower, "upper": est.cum_upper,
                             **truth, "covers": est.covers(truth["true_cum_log"]), "excludes_zero": est.excludes_zero,
                             "width": est.width, "error": None, "rel_lift": spec.rel_lift, "shape": spec.shape,
                             "pull_forward": spec.pull_forward, "cannibal_share": spec.cannibal_share})
    return pd.DataFrame(rows)


def score(study: pd.DataFrame) -> pd.DataFrame:
    """Coverage, bias, power (or false-positive rate at effect zero), width per estimator and spec."""
    ok = study[study["error"].isna()].copy()
    ok["bias"] = ok["estimate"] - ok["true_cum_log"]
    g = ok.groupby(["spec", "rel_lift", "shape", "pull_forward", "cannibal_share", "estimator"], observed=True)
    out = g.agg(n=("estimate", "size"), coverage=("covers", "mean"), bias=("bias", "mean"),
                rmse=("bias", lambda b: float(np.sqrt(np.mean(b**2)))), power=("excludes_zero", "mean"),
                width=("width", "median")).reset_index()
    out["failed"] = study[study["error"].notna()].groupby("estimator").size().reindex(out["estimator"]).fillna(0).values
    return out
