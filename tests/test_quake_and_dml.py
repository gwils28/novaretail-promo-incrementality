import numpy as np
import pytest

from novaretail.calendar import calendar_features
from novaretail.data import build_panel, to_wide
from novaretail.dgp import SynthSpec, make_synthetic
from novaretail.dml import build_table, by_family, controls, dose_response, plr
from novaretail.earthquake import quake_frame, quake_units, run_quake, store_totals
from novaretail.features import promo_intensity


@pytest.fixture(scope="module")
def world():
    sp = make_synthetic(SynthSpec(start="2014-01-01", end="2016-08-31", n_campaigns=250, n_stores=10, seed=11))
    panel, _ = build_panel(sp.train)
    return sp, panel


def test_quake_units_and_frame(world):
    sp, panel = world
    totals = store_totals(panel)
    treated, donors = quake_units(sp.stores, totals)
    assert set(treated) == {6, 7}  # Manta and Esmeraldas
    assert 10 not in donors and 1 not in donors  # late store and Quito are out
    frame = quake_frame(totals, treated, donors)
    assert frame.index.freqstr == "D" and frame["treated"].notna().mean() > 0.98  # 25 Dec and 1 Jan are missing


@pytest.mark.slow
def test_quake_effect_is_positive_on_the_synthetic_shock(world):
    sp, panel = world
    out = run_quake(panel, sp.stores, sp.holidays, sp.oil, nsim=600, post_days=28)
    s = out["summary"]
    assert s["rel_effect"] > 0 and s["excludes_zero"]
    assert out["weekly"].loc[1, "rel_effect"] > out["weekly"].loc[4, "rel_effect"]
    assert len(out["space"]) == len(out["donors"]) + 1


def test_dml_recovers_the_family_ordering_of_the_response(world):
    sp, panel = world
    wide = to_wide(panel)
    cals = {int(s["store_nbr"]): calendar_features(wide.index, sp.holidays, s, sp.oil) for _, s in sp.stores.iterrows()}
    inten = promo_intensity(to_wide(panel, "onpromotion"))
    table = build_table(panel, cals, inten)
    x = controls(table)
    assert "y_lag7" in x and "store_code" in x
    r = plr(table.sample(40000, random_state=0).reset_index(drop=True), ["intensity"], x, n_folds=3)
    assert r["theta"]["intensity"] > 0 and r["se"]["intensity"] > 0
    fam = by_family(table, x, n_folds=3, min_rows=1500)
    truth = (sp.campaigns.assign(resp=sp.campaigns["true_log_lift"] / sp.campaigns["promoted_share"])
             .groupby("family")["resp"].first())
    joined = fam.join(truth, how="inner")
    rho = joined[["theta", "resp"]].corr(method="spearman").iloc[0, 1]
    assert rho > 0.4, joined
    dr = dose_response(table.sample(40000, random_state=1).reset_index(drop=True), x, n_folds=3)
    assert len(dr) >= 3 and np.isfinite(dr["theta"]).all()
