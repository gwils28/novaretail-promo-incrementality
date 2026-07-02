import numpy as np
import pandas as pd
import pytest

from novaretail.campaigns import DetectionRule, active_days, detect_all, detect_unit, runs
from novaretail.data import build_panel, to_wide
from novaretail.dgp import SynthSpec, make_synthetic
from novaretail.donors import RegistryRule, eligible_donors, linked_families, pool_units
from novaretail.features import promo_intensity, regressors, unit_series


def _count(background: float, bumps: list[tuple[int, int, int]], n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    c = rng.poisson(background, n).astype(float)
    for s, L, k in bumps:
        c[s : s + L] += k
    return pd.Series(c, index=pd.date_range("2015-01-01", periods=n, freq="D"))


def test_runs_merge_short_gaps():
    a = pd.Series([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]).astype(bool)
    assert runs(a, merge_gap=1) == [(1, 4), (7, 9)]
    assert runs(a, merge_gap=0) == [(1, 2), (4, 4), (7, 9)]


def test_detect_unit_finds_a_campaign_with_a_quiet_pre_period():
    c = _count(1.0, [(200, 7, 12), (260, 1, 12), (330, 30, 12)])
    usable = pd.Series(True, index=c.index)
    df = detect_unit(c, usable, DetectionRule(n_pre=120))
    elig = df[df["eligible"]]
    assert len(elig) == 1
    r = elig.iloc[0]
    assert r["start"] == c.index[200] and r["length"] == 7 and r["n_post"] == 14
    assert set(df["reason"]) >= {"eligible", "too_short", "too_long"}


def test_detect_unit_refuses_a_busy_pre_period_and_truncates_the_post_window():
    bumps = [(s, 5, 12) for s in range(100, 200, 12)] + [(230, 5, 12), (240, 5, 12)]
    c = _count(1.0, bumps)
    usable = pd.Series(True, index=c.index)
    df = detect_unit(c, usable, DetectionRule(n_pre=120))
    first = df[df["start"] == c.index[100]].iloc[0]
    assert not first["eligible"] and first["reason"] == "no_pre_period"
    r230 = df[df["start"] == c.index[230]].iloc[0]
    assert r230["post_truncated"] and r230["post_end"] == c.index[239]
    assert not r230["eligible"] and r230["reason"] == "pre_not_quiet"


def test_active_days_tracks_a_rising_background():
    c = _count(1.0, [(150, 5, 10)])
    c.iloc[250:] += 20  # a regime change: nothing active once the background has moved
    a = active_days(c)
    assert a.iloc[150:155].all()
    assert not a.iloc[320:].any()


@pytest.fixture(scope="module")
def synth_panel():
    sp = make_synthetic(SynthSpec(end="2016-06-30", n_campaigns=80, n_stores=6, seed=3))
    panel, _ = build_panel(sp.train)
    wide_raw = to_wide(panel, structural_as_nan=False)
    wide_promo = to_wide(panel, "onpromotion")
    flag = (panel["structural_zero"] | panel["promo_unknown"]).astype(float)
    usable = ~to_wide(panel.assign(flag=flag), "flag").astype(bool)
    return sp, panel, wide_raw, wide_promo, usable


def test_detect_all_recovers_planted_campaigns(synth_panel):
    sp, panel, wide_raw, wide_promo, usable = synth_panel
    det = detect_all(wide_promo, usable)
    elig = det[det["eligible"]]
    assert len(elig) > 0
    truth = sp.campaigns.set_index(["store_nbr", "family", "start"])
    hits = sum((r.store_nbr, r.family, r.start) in truth.index for r in elig.itertuples())
    misses = elig[[(r.store_nbr, r.family, r.start) not in truth.index for r in elig.itertuples()]]
    assert hits / len(elig) > 0.85, misses[["store_nbr", "family", "start", "length", "mean_count", "background"]]


def test_donor_registry_rules(synth_panel):
    sp, panel, wide_raw, wide_promo, usable = synth_panel
    det = detect_all(wide_promo, usable)
    camp = det[det["eligible"] & (det["family"] == "POULTRY")].iloc[0]
    tab = eligible_donors(wide_raw, wide_promo, usable, sp.stores, camp, RegistryRule(max_pool=6))
    assert "MEATS" in linked_families("POULTRY")
    meats = tab[(tab["store_nbr"] == camp["store_nbr"]) & (tab["family"] == "MEATS")]
    assert (meats["reason"] == "substitution_edge").all() and not meats["in_pool"].any()
    assert tab["in_pool"].sum() <= 6
    assert all(u != (camp["store_nbr"], camp["family"]) for u in pool_units(tab))
    same_city = tab[(tab["kind"] == "same_family_other_store") & (tab["reason"] == "same_city")]
    quito = sp.stores[sp.stores["city"] == "Quito"]["store_nbr"].tolist()
    if camp["store_nbr"] in quito:
        assert len(same_city) == len(quito) - 1


def test_features_scale_and_regressors(synth_panel):
    sp, panel, wide_raw, wide_promo, usable = synth_panel
    y = unit_series(to_wide(panel), 1, "GROCERY I")
    assert y.loc["2016-01-01"] != y.loc["2016-01-01"]  # NaN on a closed day
    inten = promo_intensity(wide_promo)
    assert inten.max().max() <= 2.0 and inten.min().min() >= 0
    from novaretail.calendar import calendar_features
    cal = calendar_features(y.index, sp.holidays, sp.stores.iloc[0], sp.oil)
    X = regressors(cal)
    assert "dow_6" in X.columns and "oil_grad7" in X.columns and (X.std() > 0).all()
