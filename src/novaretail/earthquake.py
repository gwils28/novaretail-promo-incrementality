"""The 16 April 2016 earthquake as a test bench for the counterfactual method.

A dated exogenous shock with a known geography: Manabi and Esmeraldas were hit, the rest of the country was shaken
and then bought for the relief effort. Treated units are the stores of the two provinces (daily store totals on
the log scale); donors are the stores of provinces far from the epicentre, declared here and not chosen by fit.
Pichincha (Quito) and Guayas (Guayaquil) are excluded from the donor pool on purpose: they are the relief hubs,
so the interference assumption is least credible there. The placebo-in-space over donor stores is what tells
whether "unaffected" holds, and the by-family table shows which families carried the relief purchases.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from causal_ts.bsts import causal_impact

from novaretail.backtest import placebo_time
from novaretail.calendar import QUAKE, QUAKE_STATES, calendar_features
from novaretail.features import regressors

DONOR_STATES = ("Azuay", "Loja", "El Oro", "Tungurahua", "Chimborazo", "Cotopaxi", "Imbabura", "Bolivar",
                "Pastaza", "Los Rios", "Santa Elena", "Santo Domingo de los Tsachilas")


def store_totals(panel: pd.DataFrame) -> pd.DataFrame:
    """date x store_nbr daily totals over families; closed or not-yet-open store days are NaN."""
    p = panel[~panel["before_opening"]]
    tot = p.groupby(["date", "store_nbr"], observed=True)["sales"].sum(min_count=1).unstack("store_nbr")
    closed = p.groupby(["date", "store_nbr"], observed=True)["closed"].all().unstack("store_nbr")
    return tot.where(~closed.astype("boolean").fillna(True).astype(bool))


def quake_units(stores: pd.DataFrame, totals: pd.DataFrame, pre_days: int = 180) -> tuple[list[int], list[int]]:
    """Treated stores (affected provinces) and donor stores (declared far provinces) open over the window."""
    start = QUAKE - pd.Timedelta(pre_days, unit="D")
    open_all = totals.loc[start:QUAKE].notna().mean() > 0.95
    st = stores.set_index("store_nbr")
    treated = [int(s) for s in totals.columns if st.loc[s, "state"] in QUAKE_STATES and open_all[s]]
    donors = [int(s) for s in totals.columns if st.loc[s, "state"] in DONOR_STATES and open_all[s]]
    return treated, donors


def quake_frame(totals: pd.DataFrame, treated: list[int], donors: list[int], pre_days: int = 180,
                post_days: int = 56) -> pd.DataFrame:
    """Log of the treated total and of each donor store over [QUAKE - pre_days, QUAKE + post_days)."""
    start, end = QUAKE - pd.Timedelta(pre_days, unit="D"), QUAKE + pd.Timedelta(post_days - 1, unit="D")
    sl = totals.loc[start:end]
    frame = pd.DataFrame({"treated": np.log1p(sl[treated].sum(axis=1, min_count=len(treated)))})
    for d in donors:
        frame[f"d_{d}"] = np.log1p(sl[d]).interpolate(limit_direction="both")
    frame.index = pd.DatetimeIndex(frame.index, freq="D")
    return frame


def run_quake(panel: pd.DataFrame, stores: pd.DataFrame, holidays: pd.DataFrame, oil: pd.DataFrame | None = None,
              pre_days: int = 180, post_days: int = 56, nsim: int = 2000, param_draws: int = 0, seed: int = 0) -> dict:
    """Aggregate effect, placebo-in-time, placebo-in-space over donor stores, by-family effects."""
    totals = store_totals(panel)
    treated, donors = quake_units(stores, totals, pre_days)
    if not treated or len(donors) < 2:
        raise ValueError(f"not enough units: treated={treated}, donors={donors}")
    frame = quake_frame(totals, treated, donors, pre_days, post_days)
    cal = regressors(calendar_features(frame.index, holidays, None, oil))
    pre_end = QUAKE - pd.Timedelta(1, unit="D")
    donor_cols = [c for c in frame.columns if c != "treated"]
    X = pd.concat([frame[donor_cols], cal], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = causal_impact(frame["treated"], X, pre_end, nsim=nsim, param_draws=param_draws, seed=seed)
        camp = pd.Series({"start": QUAKE, "length": 1, "n_post": post_days - 1})
        sweep = placebo_time(frame["treated"], X, camp, n_dates=6, nsim=max(500, nsim // 4), seed=seed)
        space = []
        for d in donor_cols:
            others = [c for c in donor_cols if c != d]
            r = causal_impact(frame[d], pd.concat([frame[others], cal], axis=1), pre_end, nsim=max(500, nsim // 4))
            s = r.summary()
            space.append({"unit": d, "rel_effect": s["rel_effect"], "rel_lower": s["rel_lower"],
                          "rel_upper": s["rel_upper"], "p_tail": s["p_tail"], "excludes_zero": s["excludes_zero"],
                          "is_treated": False})
        s0 = res.summary()
        space.append({"unit": "treated", "rel_effect": s0["rel_effect"], "rel_lower": s0["rel_lower"],
                      "rel_upper": s0["rel_upper"], "p_tail": s0["p_tail"], "excludes_zero": s0["excludes_zero"],
                      "is_treated": True})
    weekly = weekly_effects(res)
    s0.update(units_summary(res))
    return {"treated": treated, "donors": donors, "result": res, "summary": s0, "sweep": sweep,
            "space": pd.DataFrame(space).set_index("unit"), "weekly": weekly, "frame": frame,
            "by_family": quake_by_family(panel, stores, treated, donors, holidays, pre_days, post_days, nsim // 4)}


def units_summary(res, alpha: float = 0.05) -> dict:
    """Relative and absolute effect in sales units over the whole post window, from the simulated paths."""
    y = np.expm1(res.y_post.values)
    paths = np.expm1(res.paths)
    cf = paths.sum(axis=1)
    eff = y.sum() - cf
    lo, hi = np.quantile(eff, [alpha / 2, 1 - alpha / 2])
    rel = eff / cf
    rlo, rhi = np.quantile(rel, [alpha / 2, 1 - alpha / 2])
    return {"units_effect": float(eff.mean()), "units_lower": float(lo), "units_upper": float(hi),
            "units_rel": float(rel.mean()), "units_rel_lower": float(rlo), "units_rel_upper": float(rhi),
            "units_excludes_zero": bool(lo > 0 or hi < 0)}


def weekly_effects(res) -> pd.DataFrame:
    """Relative effect per week after the shock, from the simulated paths."""
    y = np.expm1(res.y_post.values)
    paths = np.expm1(res.paths)
    weeks = ((res.cf.index - res.cf.index[0]).days // 7).values
    rows = []
    for w in np.unique(weeks):
        m = weeks == w
        rel = (y[m].sum() - paths[:, m].sum(axis=1)) / paths[:, m].sum(axis=1)
        lo, hi = np.quantile(rel, [0.025, 0.975])
        rows.append({"week": int(w + 1), "rel_effect": float(rel.mean()), "rel_lower": float(lo),
                     "rel_upper": float(hi)})
    return pd.DataFrame(rows).set_index("week")


def quake_by_family(panel, stores, treated, donors, holidays, pre_days=180, post_days=28, nsim=500) -> pd.DataFrame:
    """Per family: log of the treated-store total against the donor-store totals of the same family."""
    start, end = QUAKE - pd.Timedelta(pre_days, unit="D"), QUAKE + pd.Timedelta(post_days - 1, unit="D")
    p = panel[(panel["date"] >= start) & (panel["date"] <= end) & ~panel["structural_zero"]]
    pre_end = QUAKE - pd.Timedelta(1, unit="D")
    rows = []
    for fam, g in p.groupby("family", observed=True):
        w = g.pivot_table(index="date", columns="store_nbr", values="sales", observed=True)
        w = w.reindex(pd.date_range(start, end, freq="D"))
        tr = [s for s in treated if s in w.columns]
        dn = [s for s in donors if s in w.columns]
        if not tr or len(dn) < 2:
            continue
        y = np.log1p(w[tr].sum(axis=1, min_count=1)).interpolate(limit_direction="both")
        X = np.log1p(w[dn]).interpolate(limit_direction="both")
        X.columns = [f"d_{c}" for c in X.columns]
        if (y.loc[:pre_end] > 0).mean() < 0.9 or X.std().min() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = causal_impact(y, X, pre_end, nsim=nsim).summary()
        rows.append({"family": fam, "rel_effect": s["rel_effect"], "rel_lower": s["rel_lower"],
                     "rel_upper": s["rel_upper"], "excludes_zero": s["excludes_zero"]})
    return pd.DataFrame(rows).set_index("family").sort_values("rel_effect", ascending=False)
