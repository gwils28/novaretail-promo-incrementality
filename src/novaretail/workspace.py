"""Processed data loaded once into the matrices every analysis uses (wide sales, promotions, usability, calendars)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from novaretail.calendar import calendar_features
from novaretail.data import PROCESSED, to_wide


@dataclass
class World:
    panel: pd.DataFrame
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    wide: pd.DataFrame  # log-ready sales with structural zeros as NaN
    wide_raw: pd.DataFrame  # sales with zeros kept
    wide_promo: pd.DataFrame
    usable: pd.DataFrame  # not structural zero and promotion status known
    calendars: dict[int, pd.DataFrame]

    @property
    def units(self) -> list[tuple[int, str]]:
        return list(self.wide.columns)


def load_world(processed=PROCESSED, stores_subset: list[int] | None = None) -> World:
    panel = pd.read_parquet(processed / "panel.parquet")
    if stores_subset:
        panel = panel[panel["store_nbr"].isin(stores_subset)]
    stores = pd.read_parquet(processed / "stores.parquet")
    oil = pd.read_parquet(processed / "oil.parquet")
    holidays = pd.read_parquet(processed / "holidays.parquet")
    wide = to_wide(panel)
    wide_raw = to_wide(panel, structural_as_nan=False)
    wide_promo = to_wide(panel, "onpromotion")
    flag = (panel["structural_zero"] | panel["promo_unknown"]).astype(float)
    usable = ~to_wide(panel.assign(flag=flag), "flag").astype(bool)
    idx = pd.DatetimeIndex(wide.index, freq="D")
    cals = {int(s["store_nbr"]): calendar_features(idx, holidays, s, oil) for _, s in stores.iterrows()
            if int(s["store_nbr"]) in set(wide.columns.get_level_values(0))}
    return World(panel, stores, oil, holidays, wide, wide_raw, wide_promo, usable, cals)
