"""Double machine learning for the response of sales to promotion intensity.

Favorita has no price, so there is no price elasticity to estimate. What the data can identify, under the
usual DML assumption (the intensity is as good as random given the controls), is a partially linear model
    log1p(sales) = theta * intensity + g(controls) + e,
with controls = calendar (holidays, payday, weekday, month), store and family identities, oil, and lags of the
outcome. Robinson's partialling
out with cross-fitting gives theta with a valid standard error even when g is a gradient-boosted tree.
Cross-fitting folds are blocks of time (months), not random rows, because the rows of one series are dependent.
The lags of the outcome are a choice to be stated: they absorb the slow level of the series that a tree on
identities would otherwise fit badly, and they can absorb part of a persistent treatment's effect, so the
spec without lags is reported alongside. The intensity is the count scaled by the series' upper quantile
(features.promo_intensity): a proxy for the promoted share, unit-free, and reported as such. The reference
library (DoubleML) is a cross-check on the same residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from novaretail.features import CALENDAR_COLUMNS

LAGS = (7, 14, 28)
MONTHS = [f"month_{m}" for m in range(2, 13)]  # seasonality the trees cannot infer from identities and lags


def build_table(panel: pd.DataFrame, calendars: dict[int, pd.DataFrame], intensity: pd.DataFrame,
                families: list[str] | None = None, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """One row per usable store-family-day with the outcome, the intensity, the controls and a month group."""
    p = panel[~panel["structural_zero"] & ~panel["promo_unknown"]].copy()
    if families:
        p = p[p["family"].isin(families)]
    p["y"] = np.log1p(p["sales"])
    inten = intensity.stack(["store_nbr", "family"], future_stack=True).rename("intensity").reset_index()
    p = p.merge(inten, on=["date", "store_nbr", "family"], how="left")
    cal = pd.concat({s: c[CALENDAR_COLUMNS + MONTHS + ["oil_grad7"]] for s, c in calendars.items()},
                    names=["store_nbr", "date"])
    p = p.merge(cal.reset_index(), on=["store_nbr", "date"], how="left")
    p = p.sort_values(["store_nbr", "family", "date"])
    g = p.groupby(["store_nbr", "family"], observed=True)["y"]
    for lag in lags:
        p[f"y_lag{lag}"] = g.shift(lag)
    p["t"] = (p["date"] - p["date"].min()).dt.days  # a flexible time trend for the trees: the panel grows over
    # the years and so does the promotion count, a confound that identities, months and lags do not remove
    p["store_code"] = p["store_nbr"].astype("category").cat.codes
    p["family_code"] = p["family"].astype("category").cat.codes
    p["month"] = p["date"].dt.to_period("M").astype(str)
    return p.dropna(subset=[f"y_lag{lag}" for lag in lags] + ["intensity"]).reset_index(drop=True)


def controls(table: pd.DataFrame, with_lags: bool = True) -> list[str]:
    cols = CALENDAR_COLUMNS + MONTHS + ["oil_grad7", "t", "store_code", "family_code"]
    if with_lags:
        cols += [c for c in table.columns if c.startswith("y_lag")]
    return [c for c in cols if c in table.columns]


def _learner(seed: int):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
                                         random_state=seed)


def residualise(table: pd.DataFrame, y_col: str, d_cols: list[str], x_cols: list[str], n_folds: int = 5,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted residuals of y and of each d on the controls, folds = blocks of months."""
    X = table[x_cols].values
    y = table[y_col].values
    D = table[d_cols].values
    groups = table["month"].values
    ry = np.zeros_like(y)
    rD = np.zeros_like(D)
    for train, test in GroupKFold(n_splits=n_folds).split(X, y, groups):
        m = _learner(seed).fit(X[train], y[train])
        ry[test] = y[test] - m.predict(X[test])
        for j in range(D.shape[1]):
            md = _learner(seed + 1 + j).fit(X[train], D[train, j])
            rD[test, j] = D[test, j] - md.predict(X[test])
    return ry, rD


def plr(table: pd.DataFrame, d_cols: list[str], x_cols: list[str], n_folds: int = 5, seed: int = 0) -> dict:
    """Partially linear DML: OLS of the y residual on the d residuals, HC-robust standard errors."""
    ry, rD = residualise(table, "y", d_cols, x_cols, n_folds, seed)
    XtX = rD.T @ rD
    theta = np.linalg.solve(XtX, rD.T @ ry)
    e = ry - rD @ theta
    meat = (rD * e[:, None]).T @ (rD * e[:, None])
    bread = np.linalg.inv(XtX)
    cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    return {"theta": dict(zip(d_cols, theta, strict=True)), "se": dict(zip(d_cols, se, strict=True)), "n": len(ry),
            "r2_partial": float(1 - e.var() / ry.var())}


def by_family(table: pd.DataFrame, x_cols: list[str], n_folds: int = 5, seed: int = 0,
              min_rows: int = 2000) -> pd.DataFrame:
    """theta per family (stores pooled): the response of log sales to a unit of intensity."""
    rows = []
    for fam, g in table.groupby("family", observed=True):
        if len(g) < min_rows or g["intensity"].std() == 0:
            continue
        r = plr(g.reset_index(drop=True), ["intensity"], [c for c in x_cols if c != "family_code"], n_folds, seed)
        rows.append({"family": fam, "theta": r["theta"]["intensity"], "se": r["se"]["intensity"], "n": r["n"],
                     "share_promoted_days": float((g["intensity"] > 0).mean())})
    df = pd.DataFrame(rows).set_index("family")
    df["lower"], df["upper"] = df["theta"] - 1.96 * df["se"], df["theta"] + 1.96 * df["se"]
    return df.sort_values("theta", ascending=False)


def dose_response(table: pd.DataFrame, x_cols: list[str], bins=(0.0, 0.2, 0.5, 0.8, 1.01, 10.0), n_folds: int = 5,
                  seed: int = 0) -> pd.DataFrame:
    """theta per intensity bin, against the no-promotion days: a step response curve."""
    t = table.copy()
    labels = [f"({bins[i]:.1f}, {bins[i + 1]:.1f}]" for i in range(len(bins) - 1)]
    cats = pd.cut(t["intensity"], bins=[-1e-9, *bins[1:]], labels=labels)
    cats = cats.cat.rename_categories({cats.cat.categories[0]: "0"})
    dummies = pd.get_dummies(cats, prefix="dose", dtype=float).iloc[:, 1:]  # bin "0" is the reference
    t = pd.concat([t, dummies], axis=1)
    r = plr(t, list(dummies.columns), x_cols, n_folds, seed)
    out = pd.DataFrame({"theta": pd.Series(r["theta"]), "se": pd.Series(r["se"])})
    out["lower"], out["upper"] = out["theta"] - 1.96 * out["se"], out["theta"] + 1.96 * out["se"]
    out["n_rows"] = dummies.sum().values
    out.index = [i.replace("dose_", "") for i in out.index]
    return out


def crosscheck_doubleml(table: pd.DataFrame, x_cols: list[str], n_folds: int = 5, seed: int = 0,
                        n_max: int = 200_000) -> dict:
    """Same model with the DoubleML library (PLR, partialling out), on a random subsample if the table is large."""
    import doubleml as dml

    t = table.sample(min(n_max, len(table)), random_state=seed) if len(table) > n_max else table
    data = dml.DoubleMLData(t[["y", "intensity", *x_cols]].reset_index(drop=True), y_col="y", d_cols="intensity",
                            x_cols=x_cols)
    model = dml.DoubleMLPLR(data, _learner(seed), _learner(seed + 1), n_folds=n_folds, score="partialling out")
    model.fit()
    return {"theta": float(model.coef[0]), "se": float(model.se[0]), "n": len(t)}


def holdout_check(table: pd.DataFrame, x_cols: list[str], split_date: str, n_folds: int = 5, seed: int = 0) -> dict:
    """theta on the training years and on the held-out last year: stability of the response over time."""
    split = pd.Timestamp(split_date)
    a, b = table[table["date"] < split], table[table["date"] >= split]
    ra = plr(a.reset_index(drop=True), ["intensity"], x_cols, n_folds, seed)
    rb = plr(b.reset_index(drop=True), ["intensity"], x_cols, n_folds, seed)
    return {"train": {"theta": ra["theta"]["intensity"], "se": ra["se"]["intensity"], "n": ra["n"]},
            "holdout": {"theta": rb["theta"]["intensity"], "se": rb["se"]["intensity"], "n": rb["n"]}}
