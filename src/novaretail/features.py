"""Outcome transforms and the regressor matrix used by every estimator.

Sales are modelled on the `log1p` scale by default (`asinh` as an alternative for families with many zeros):
promotion effects are multiplicative, the noise is closer to Gaussian, and a relative effect reads directly as
a difference of logs. Effects in units are recovered from the simulated counterfactual paths, never from the
exponential of a mean. Structural zeros (see `data.py`) are NaN: `statsmodels` state-space models treat a
missing observation as uninformative, which is exactly what a closed store is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRANSFORMS = {
    "log1p": (np.log1p, np.expm1),
    "asinh": (np.arcsinh, np.sinh),
    "none": (lambda x: x, lambda x: x),
}
# regressors passed as exogenous variables next to the donors; weekday dummies carry the deterministic weekly
# pattern (a stochastic seasonal would be slower and not more accurate at this horizon)
CALENDAR_COLUMNS = ["holiday_national", "holiday_regional", "holiday_local", "day_before_holiday", "work_day",
                    "payday", "after_payday"] + [f"dow_{d}" for d in range(1, 7)]
OIL_COLUMNS = ["oil_grad7"]


def transform(x, kind: str = "log1p"):
    return TRANSFORMS[kind][0](x)


def inverse(x, kind: str = "log1p"):
    return TRANSFORMS[kind][1](x)


def unit_series(wide: pd.DataFrame, store_nbr: int, family: str, kind: str = "log1p") -> pd.Series:
    """One store-family series on the transformed scale, structural zeros kept as NaN."""
    s = wide[(store_nbr, family)].astype(float)
    return transform(s, kind).rename(f"{store_nbr}:{family}")


def regressors(calendar: pd.DataFrame, use_oil: bool = True) -> pd.DataFrame:
    """Calendar (and oil) columns that enter the counterfactual model as exogenous variables."""
    cols = CALENDAR_COLUMNS + ([c for c in OIL_COLUMNS if c in calendar.columns] if use_oil else [])
    X = calendar[cols].astype(float)
    return X.loc[:, X.std() > 0]  # a constant column (no regional holiday for this store) is dropped


def promo_intensity(wide_promo: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """`onpromotion` scaled by the store-family's upper quantile: a unit-free intensity, roughly the promoted share.

    Favorita gives no item count per family, so the promoted share is not observable; the scaled count is the
    proxy used by the dose-response analysis and is reported as such.
    """
    scale = wide_promo.quantile(q).replace(0, np.nan)
    return (wide_promo / scale).fillna(0).clip(upper=1.5)
