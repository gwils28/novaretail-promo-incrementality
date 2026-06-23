import numpy as np
import pandas as pd
import pytest

from novaretail.calendar import calendar_features, day_off_table
from novaretail.data import build_panel, to_wide
from novaretail.dgp import SynthSpec, make_synthetic


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(SynthSpec(end="2016-12-31", n_campaigns=60, n_stores=6, seed=1))


def test_train_is_kaggle_shaped_and_drops_christmas(synth):
    t = synth.train
    assert list(t.columns) == ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
    assert not ((t["date"].dt.month == 12) & (t["date"].dt.day == 25)).any()
    assert (t["onpromotion"][t["date"] < synth.spec.promo_known_from] == 0).all()


def test_truth_matches_planted_campaigns(synth):
    c = synth.campaigns.iloc[0]
    t = synth.train.merge(synth.truth, on=["date", "store_nbr", "family"])
    sf = (t["store_nbr"] == c["store_nbr"]) & (t["family"] == c["family"])
    win = t[sf & t["date"].between(c["start"], c["end"])]
    assert (win["campaign_id"] == c["campaign_id"]).all()
    assert (win["onpromotion"] >= c["count"]).all()
    # planted lifts are positive on average over the campaigns of the panel
    allc = t[t["campaign_id"] >= 0]
    assert allc["sales"].sum() > allc["y0"].sum()


def test_panel_density_and_structural_zeros(synth):
    panel, rep = build_panel(synth.train)
    assert rep["rows_expected"] == len(panel)
    assert rep["missing_calendar_days"] == ["12-25"]
    assert rep["duplicated_keys"] == 0
    late = synth.spec.late_store
    if late in panel["store_nbr"].unique():
        p = panel[panel["store_nbr"] == late]
        assert p.loc[p["date"] < synth.spec.late_open, "before_opening"].all()
        assert not p.loc[p["date"] >= synth.spec.late_open, "before_opening"].any()
    jan1 = panel[(panel["date"].dt.month == 1) & (panel["date"].dt.day == 1) & ~panel["before_opening"]]
    assert jan1["closed"].all()
    assert panel.loc[panel["date"] < synth.spec.promo_known_from, "promo_unknown"].all()
    wide = to_wide(panel)
    assert wide.loc["2016-01-01"].isna().all()  # structural zeros are NaN in the wide matrix
    assert rep["share_zero_structural"] > 0.5  # most zeros of this panel are structural


def test_calendar_transfer_semantics(synth):
    off = day_off_table(synth.holidays)
    assert pd.Timestamp("2016-05-24") not in set(off["date"])  # transferred: not a day off
    assert pd.Timestamp("2016-05-27") in set(off["date"])  # the Transfer row is the day off
    dates = pd.date_range("2016-04-01", "2016-06-30")
    store = synth.stores.iloc[0]
    cal = calendar_features(dates, synth.holidays, store, synth.oil)
    assert cal.loc["2016-05-27", "holiday_national"] == 1 and cal.loc["2016-05-24", "holiday_national"] == 0
    assert cal.loc["2016-04-16", "quake"] == 1 and cal.loc["2016-04-15", "quake"] == 0
    assert cal.loc["2016-04-16", "event"] == 0  # earthquake rows are not generic events
    assert cal.loc["2016-04-15", "payday"] == 1 and cal.loc["2016-04-30", "payday"] == 1
    assert np.isfinite(cal[["oil", "oil_lag7", "oil_grad7"]].values).all()
    manta = synth.stores[synth.stores["state"] == "Manabi"].iloc[0]
    assert calendar_features(dates, synth.holidays, manta).loc["2016-04-20", "quake_regional"] == 1
