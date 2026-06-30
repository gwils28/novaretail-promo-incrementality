import numpy as np
import pandas as pd
import pytest

from novaretail.backtest import (
    PlantSpec,
    isolate,
    placebo_space,
    placebo_time,
    plant,
    planted_study,
    pool_sensitivity,
    quiet_windows,
    score,
)
from novaretail.calendar import calendar_features
from novaretail.campaigns import detect_all
from novaretail.data import build_panel, to_wide
from novaretail.dgp import SynthSpec, make_synthetic
from novaretail.donors import RegistryRule, eligible_donors, pool_units
from novaretail.estimate import assemble


@pytest.fixture(scope="module")
def world():
    sp = make_synthetic(SynthSpec(end="2016-06-30", n_campaigns=60, n_stores=6, seed=7))
    panel, _ = build_panel(sp.train)
    wide = to_wide(panel)
    wide_raw = to_wide(panel, structural_as_nan=False)
    wide_promo = to_wide(panel, "onpromotion")
    flag = (panel["structural_zero"] | panel["promo_unknown"]).astype(float)
    usable = ~to_wide(panel.assign(flag=flag), "flag").astype(bool)
    cals = {int(s["store_nbr"]): calendar_features(wide.index, sp.holidays, s, sp.oil) for _, s in sp.stores.iterrows()}
    det = detect_all(wide_promo, usable)
    elig = det[det["eligible"] & (det["n_post"] == 14)]
    vol = [wide_raw.loc[r.pre_start:r.start, (r.store_nbr, r.family)].mean() for r in elig.itertuples()]
    camp = elig.iloc[int(np.argmax(vol))]
    tab = eligible_donors(wide_raw, wide_promo, usable, sp.stores, camp, RegistryRule(max_pool=6))
    return sp, wide, wide_raw, wide_promo, usable, cals, camp, tab


def test_placebo_time_and_space(world):
    sp, wide, wide_raw, wide_promo, usable, cals, camp, tab = world
    pool = pool_units(tab)
    y, X = assemble(wide, cals, camp, pool)
    sweep = placebo_time(y, X, camp, n_dates=4, nsim=300)
    assert 1 <= len(sweep) <= 4 and set(sweep.columns) >= {"rel_effect", "excludes_zero"}
    space = placebo_space(wide, cals, camp, pool, nsim=300)
    assert space["is_treated"].sum() == 1 and len(space) == len(pool) + 1
    assert 0 < space["rank_p_rel"].iloc[0] <= 1
    removed, kept, first = isolate(wide, cals, camp, pool, nsim=300, max_rounds=1)
    assert set(removed) | set(kept) == set(pool) and len(first) == len(pool)


def test_pool_sensitivity_rows(world):
    sp, wide, wide_raw, wide_promo, usable, cals, camp, tab = world
    sens = pool_sensitivity(wide, cals, camp, tab, sizes=(3,), leave_one_out=False, nsim=200)
    assert "aucun donneur (niveau local + calendrier)" in sens.index and "3 donneurs" in sens.index


def test_plant_writes_the_truth(world):
    sp, wide, wide_raw, wide_promo, usable, cals, camp, tab = world
    spec = PlantSpec(length=5, rel_lift=0.2, pull_forward=0.5, cannibal_share=0.3)
    wins = quiet_windows(wide_raw, wide_promo, usable, spec, n=3, seed=1)
    assert len(wins) == 3
    pool = pool_units(tab)
    w, truth = plant(wide_raw, wins[0], spec, pool)
    unit = (wins[0]["store_nbr"], wins[0]["family"])
    during = pd.date_range(wins[0]["start"], periods=5)
    assert (w.loc[during, unit] >= wide_raw.loc[during, unit]).all()
    assert truth["true_cum_log"] > 0 and truth["true_dip"] == pytest.approx(0.5 * 0.2 * 5 / 14)
    assert (w.loc[during, pool[0]] <= wide_raw.loc[during, pool[0]]).all()


@pytest.mark.slow
def test_planted_study_scores(world):
    sp, wide, wide_raw, wide_promo, usable, cals, camp, tab = world
    specs = [PlantSpec(rel_lift=0.0), PlantSpec(rel_lift=0.3)]
    study = planted_study(wide_raw, wide_promo, usable, sp.stores, cals, specs, n_per_spec=4,
                          estimators=["bsts_llevel", "ols", "augmented_sc"], seed=2)
    sc = score(study)
    assert set(sc["estimator"]) == {"bsts_llevel", "ols", "augmented_sc"}
    assert sc["coverage"].between(0, 1).all()
