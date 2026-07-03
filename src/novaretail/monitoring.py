"""Monitoring of a counterfactual design in production: the four signals and the business guard-rail.

Nothing observable says whether a counterfactual is right. Block 3 of the tech watch names four substitutes,
each tied to a failure mode the Monte Carlo can reproduce, and one money check that catches the failure mode the
data cannot show (a relationship that breaks at the campaign date):
1. pre-period fit of each campaign (one-step R2, coverage of the one-step interval, Ljung-Box);
2. rolling placebo-in-time on never-treated windows: the empirical false-positive rate of the design;
3. placebo-in-space with isolation on each campaign: donors that moved with the campaign;
4. stability of the donor-target relationship between campaigns of the same target (coefficient drift, Chow);
5. reconciliation of estimated incremental margin with realised margin, whose drift is the alarm.
Favorita carries no margin, so the reconciliation is demonstrated on a simulated ledger with a stated unit margin;
the code path is the one a real ledger would feed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from novaretail.estimate import PROTOCOL

ALERTS = {"fpr_max": 0.10, "fpr_window": 50, "isolation_rounds_max": 1, "coef_sign_change": True,
          "chow_p_min": 0.01, "reconciliation_drift_months": 3}


def fit_signals(estimates: pd.DataFrame) -> pd.DataFrame:
    """Signal 1: per campaign, the three fit checks and a block flag."""
    lo, hi = PROTOCOL["coverage_range"]
    cols = ["campaign_id", "store_nbr", "family", "start", "r2_one_step", "coverage_one_step", "ljung_box_p"]
    df = estimates[cols].copy()
    df["blocked"] = ((df["r2_one_step"] < PROTOCOL["r2_min"]) | ~df["coverage_one_step"].between(lo, hi)
                     | (df["ljung_box_p"] < PROTOCOL["ljung_box_p_min"]))
    return df


def rolling_fpr(placebos: pd.DataFrame, window: int = ALERTS["fpr_window"]) -> pd.DataFrame:
    """Signal 2: share of never-treated placebo intervals excluding zero over the last `window` runs, in time."""
    p = placebos.sort_values("start").reset_index(drop=True)
    p["fpr_rolling"] = p["excludes_zero"].astype(float).rolling(window, min_periods=max(10, window // 5)).mean()
    p["alert"] = p["fpr_rolling"] > ALERTS["fpr_max"]
    return p[["start", "store_nbr", "family", "excludes_zero", "fpr_rolling", "alert"]]


def interference_signals(isolations: pd.DataFrame) -> pd.DataFrame:
    """Signal 3: per campaign, donors removed by the isolation and the number of rounds; review beyond one."""
    df = isolations.copy()
    df["review"] = df["rounds"] > ALERTS["isolation_rounds_max"]
    return df


def relationship_drift(coefs: pd.DataFrame) -> pd.DataFrame:
    """Signal 4: regression coefficients of the same donor on the same target across successive campaigns.

    `coefs` has one row per (target, campaign start, donor) with the fitted `beta`. A sign change between two
    successive campaigns blocks the donor for that target.
    """
    df = coefs.sort_values(["target", "donor", "start"]).copy()
    df["beta_prev"] = df.groupby(["target", "donor"])["beta"].shift(1)
    df["drift"] = df["beta"] - df["beta_prev"]
    df["sign_change"] = (np.sign(df["beta"]) != np.sign(df["beta_prev"])) & df["beta_prev"].notna()
    return df


def simulated_ledger(estimates: pd.DataFrame, unit_margin: float = 1.0, seed: int = 0, drift_from: str | None = None,
                     drift_per_month: float = 0.0) -> pd.DataFrame:
    """A realised-margin ledger consistent with the estimates, plus noise and an optional silent drift.

    Realised margin over the post window = (counterfactual units + true incremental units) * unit margin. The
    truth is taken as the estimate plus noise, and from `drift_from` on, the counterfactuals are assumed to be
    off by `drift_per_month` more every month: what a relationship break looks like from the ledger.
    """
    rng = np.random.default_rng(seed)
    df = estimates[["campaign_id", "start", "net_units", "net_cf_units", "net_lower", "net_upper"]].copy()
    df["month"] = pd.to_datetime(df["start"]).dt.to_period("M")
    noise = rng.normal(0, (df["net_upper"] - df["net_lower"]) / 4)
    bias = 0.0
    if drift_from is not None:
        months = (df["month"] - pd.Period(drift_from, "M")).apply(lambda d: max(0, d.n))
        bias = -drift_per_month * months * df["net_cf_units"]
    df["realised_units"] = df["net_cf_units"] + df["net_units"] + noise + bias
    df["realised_margin"] = df["realised_units"] * unit_margin
    df["implied_margin"] = (df["net_cf_units"] + df["net_units"]) * unit_margin
    df["implied_margin_lower"] = (df["net_cf_units"] + df["net_lower"]) * unit_margin
    return df


def reconciliation(ledger: pd.DataFrame) -> pd.DataFrame:
    """Monthly gap between realised and implied margin, its share, and the drift alarm (same sign 3 months)."""
    m = ledger.groupby("month").agg(realised=("realised_margin", "sum"), implied=("implied_margin", "sum"),
                                    n=("campaign_id", "size"))
    m["gap"] = m["realised"] - m["implied"]
    m["gap_share"] = m["gap"] / m["implied"]
    k = ALERTS["reconciliation_drift_months"]
    sgn = np.sign(m["gap_share"])
    m["drift_alarm"] = (sgn.rolling(k).sum().abs() == k) & (m["gap_share"].abs().rolling(k).min() > 0.02)
    return m


def decision_value(estimates: pd.DataFrame, reconciled_families: set[str] | None = None) -> pd.Series:
    """The uplift that enters a decision: the lower bound unless the family's reconciliation gap is closed."""
    rec = reconciled_families or set()
    return pd.Series(np.where(estimates["family"].isin(rec), estimates["net_units"], estimates["net_lower"]),
                     index=estimates.index, name="decision_units")
