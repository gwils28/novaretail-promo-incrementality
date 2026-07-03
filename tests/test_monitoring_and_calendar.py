import numpy as np
import pandas as pd

from novaretail.calendar_opt import plan_summary, reallocate, schedule_without_conflicts, value_per_day
from novaretail.monitoring import fit_signals, reconciliation, relationship_drift, rolling_fpr, simulated_ledger


def _estimates(n=40, seed=0):
    rng = np.random.default_rng(seed)
    fams = ["GROCERY I", "BEVERAGES", "CLEANING", "HOME CARE", "DAIRY"]
    rows = []
    for i in range(n):
        cf = rng.uniform(500, 5000)
        net = rng.normal(0.08, 0.05) * cf
        half = rng.uniform(0.3, 0.8) * abs(net) + 20
        rows.append({"campaign_id": i, "store_nbr": 1 + i % 3, "family": fams[i % 5],
                     "length": int(rng.integers(3, 10)),
                     "start": pd.Timestamp("2016-01-01") + pd.Timedelta(int(rng.integers(0, 500)), unit="D"),
                     "net_units": net, "net_cf_units": cf, "net_lower": net - half, "net_upper": net + half,
                     "measurable": i % 7 != 0, "r2_one_step": rng.uniform(0.3, 0.9),
                     "coverage_one_step": rng.uniform(0.88, 0.99), "ljung_box_p": rng.uniform(0, 1)})
    return pd.DataFrame(rows)


def test_fit_signals_block_rule():
    est = _estimates()
    fit = fit_signals(est)
    manual = ((est["r2_one_step"] < 0.5) | ~est["coverage_one_step"].between(0.90, 0.98)
              | (est["ljung_box_p"] < 0.01))
    assert (fit["blocked"] == manual).all()


def test_rolling_fpr_and_drift():
    rng = np.random.default_rng(1)
    pl = pd.DataFrame({"start": pd.date_range("2016-01-01", periods=120), "store_nbr": 1, "family": "X",
                       "excludes_zero": rng.random(120) < 0.05})
    pl.loc[80:, "excludes_zero"] = True
    f = rolling_fpr(pl, window=20)
    assert not f["alert"].iloc[:60].any() and f["alert"].iloc[-1]
    coefs = pd.DataFrame({"target": "1:X", "donor": ["a", "a", "a", "b", "b"], "beta": [0.5, 0.6, -0.2, 1.0, 1.1],
                          "start": pd.to_datetime(["2016-01-01", "2016-03-01", "2016-06-01", "2016-01-01",
                                                   "2016-03-01"])})
    d = relationship_drift(coefs)
    assert d["sign_change"].sum() == 1


def test_reconciliation_alarm_fires_only_with_a_drift():
    est = _estimates(n=120, seed=3)
    meas = est[est["measurable"]]
    clean = reconciliation(simulated_ledger(meas, seed=0))
    drift = reconciliation(simulated_ledger(meas, seed=0, drift_from="2016-09", drift_per_month=0.03))
    assert clean["drift_alarm"].sum() <= 1
    assert drift["drift_alarm"].sum() >= 1
    assert drift.loc[drift["drift_alarm"]].index[0] >= pd.Period("2016-10", "M")


def test_calendar_reallocation_keeps_the_budget_and_avoids_conflicts():
    est = _estimates(n=60, seed=5)
    values = value_per_day(est)
    plan = reallocate(values)
    per_store = plan.groupby("store_nbr")[["hist_days", "new_days", "unspent"]].sum()
    assert ((per_store["new_days"] + per_store["unspent"]) == per_store["hist_days"]).all()
    assert (plan.loc[~plan["measured"], "delta_days"] == 0).all()
    sched = schedule_without_conflicts(plan)
    weeks = sched.groupby(["store_nbr", "week"])["family"].apply(set)
    assert not any({"CLEANING", "HOME CARE"} <= s for s in weeks)  # a declared substitution edge
    summ = plan_summary(plan)
    assert 0 <= summ["share_moved"] <= 1
