import numpy as np
import pandas as pd
import pytest

from novaretail.calendar import calendar_features
from novaretail.campaigns import detect_all
from novaretail.data import build_panel, to_wide
from novaretail.dgp import SynthSpec, make_synthetic
from novaretail.donors import RegistryRule, eligible_donors, pool_units
from novaretail.estimate import assemble, augmented_sc, credibility, estimate_campaign


@pytest.fixture(scope="module")
def world():
    sp = make_synthetic(SynthSpec(end="2016-06-30", n_campaigns=80, n_stores=6, seed=5))
    panel, _ = build_panel(sp.train)
    wide = to_wide(panel)
    wide_raw = to_wide(panel, structural_as_nan=False)
    wide_promo = to_wide(panel, "onpromotion")
    flag = (panel["structural_zero"] | panel["promo_unknown"]).astype(float)
    usable = ~to_wide(panel.assign(flag=flag), "flag").astype(bool)
    cals = {int(s["store_nbr"]): calendar_features(wide.index, sp.holidays, s, sp.oil) for _, s in sp.stores.iterrows()}
    det = detect_all(wide_promo, usable)
    return sp, wide, wide_raw, wide_promo, usable, cals, det


def _pick(sp, det, wide_raw):
    """An eligible detected campaign that is a planted one with a sizeable lift on a high-volume series."""
    truth = sp.campaigns.set_index(["store_nbr", "family", "start"])
    elig = det[det["eligible"]]
    for r in elig.itertuples():
        key = (r.store_nbr, r.family, r.start)
        volume = wide_raw.loc[r.pre_start:r.start, (r.store_nbr, r.family)].mean()
        if key in truth.index and truth.loc[key, "true_log_lift"] > 0.12 and r.n_post == 14 and volume > 30:
            return det.loc[r.Index], truth.loc[key]
    pytest.skip("no large planted campaign detected in this synthetic world")


def test_bsts_recovers_a_planted_lift(world):
    sp, wide, wide_raw, wide_promo, usable, cals, det = world
    camp, truth = _pick(sp, det, wide_raw)
    tab = eligible_donors(wide_raw, wide_promo, usable, sp.stores, camp, RegistryRule(max_pool=8))
    pool = pool_units(tab)
    assert len(pool) >= 3, tab["reason"].value_counts()
    y, X = assemble(wide, cals, camp, pool)
    est = estimate_campaign(y, X, camp, pool, nsim=600)
    s = est.summary()
    # the lift during the campaign is positive and its interval contains the planted relative lift
    assert s["during_units"] > 0
    lift = truth["true_rel_lift"]
    assert s["during_rel_lower"] <= lift <= s["during_rel_upper"], (lift, s["during_rel"])
    assert s["after_units"] < s["during_units"]  # the dip after is smaller than the lift (pull-forward share < 1)
    assert set(est.quality) >= {"r2_one_step", "coverage_one_step", "ljung_box_p"}
    v = credibility(est.quality)
    assert isinstance(v["measurable"], bool)


def test_augmented_sc_weights_and_shape(world):
    sp, wide, wide_raw, wide_promo, usable, cals, det = world
    camp, truth = _pick(sp, det, wide_raw)
    tab = eligible_donors(wide_raw, wide_promo, usable, sp.stores, camp, RegistryRule(max_pool=6))
    y, X = assemble(wide, cals, camp, pool_units(tab))
    pre_end = camp["start"] - pd.Timedelta(1, unit="D")
    est = augmented_sc(y, X, pre_end)
    assert est.name == "augmented_sc"
    assert abs(est.weights.sum() - 1) < 1e-6 and (est.weights >= -1e-9).all()
    assert len(est.cf) == camp["length"] + camp["n_post"]
    assert np.isfinite(est.cum_effect) and est.cum_lower < est.cum_upper


def test_wild_paths_are_dropped_and_counted(world):

    from novaretail.estimate import effects_in_units

    sp, wide, wide_raw, wide_promo, usable, cals, det = world
    camp, truth = _pick(sp, det, wide_raw)
    tab = eligible_donors(wide_raw, wide_promo, usable, sp.stores, camp, RegistryRule(max_pool=6))
    y, X = assemble(wide, cals, camp, pool_units(tab))
    est = estimate_campaign(y, X, camp, pool_units(tab), nsim=400)
    r = est.result
    r.paths = r.paths.copy()
    r.paths[:20] += 12.0  # 5 percent of the paths leave for the stratosphere
    u = effects_in_units(r, camp["end"])
    assert u["wild_paths_share"] == pytest.approx(0.05)
    assert abs(u["during_cf_units"] - est.units["during_cf_units"]) / est.units["during_cf_units"] < 0.2
    v = credibility(est.quality, units=u)
    assert v["check_paths_sane"]
