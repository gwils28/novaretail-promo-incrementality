"""Calendar and covariate features: holidays (with transfers), payday, earthquake, oil.

Holiday semantics of the Favorita file: a row with `transferred=True` is a holiday that was *not* a day off that
day (it was moved); the day off is the row of type `Transfer`. `Bridge` days are extra days off, `Work Day` rows
are Saturdays worked to pay a bridge, `Additional` rows are extra days around a holiday (24 December), `Event`
rows are not days off (Mother's day, football, the earthquake). Locale: `National`, `Regional` (state) or `Local`
(city), so the flags are built per store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUAKE = pd.Timestamp("2016-04-16")
QUAKE_STATES = ("Manabi", "Esmeraldas")  # provinces closest to the epicentre (Pedernales)
QUAKE_WEEKS = 6  # length of the event window in the holidays file (16 April to 16 May 2016 is 31 event rows)


def day_off_table(holidays: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, locale, locale_name) that is actually a day off, after transfers."""
    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"])
    off = h[h["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"]) & ~h["transferred"]]
    return off[["date", "locale", "locale_name", "description"]].drop_duplicates()


def calendar_features(dates: pd.DatetimeIndex, holidays: pd.DataFrame, store: pd.Series | None = None,
                      oil: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily regressors for one store (or national only when `store` is None).

    Columns: holiday_national, holiday_regional, holiday_local, day_before_holiday, work_day, event,
    payday, after_payday, days_to_payday, quake, quake_week{1..6}, quake_regional, month dummies, weekday dummies,
    oil (log, forward filled), oil_lag7, oil_lag28, oil_grad7.
    """
    df = pd.DataFrame(index=dates)
    off = day_off_table(holidays)
    nat = off[off["locale"] == "National"]["date"]
    df["holiday_national"] = dates.isin(nat).astype(int)
    reg = loc = pd.Series([], dtype="datetime64[ns]")
    if store is not None:
        reg = off[(off["locale"] == "Regional") & (off["locale_name"] == store["state"])]["date"]
        loc = off[(off["locale"] == "Local") & (off["locale_name"] == store["city"])]["date"]
    df["holiday_regional"] = dates.isin(reg).astype(int)
    df["holiday_local"] = dates.isin(loc).astype(int)
    any_off = pd.DatetimeIndex(pd.concat([nat, reg, loc]))
    df["day_before_holiday"] = dates.shift(1, freq="D").isin(any_off).astype(int)
    h = holidays.assign(date=pd.to_datetime(holidays["date"]))
    df["work_day"] = dates.isin(h[h["type"] == "Work Day"]["date"]).astype(int)
    events = h[(h["type"] == "Event") & ~h["description"].str.contains("Terremoto", case=False)]["date"]
    df["event"] = dates.isin(events).astype(int)
    df["payday"] = ((dates.day == 15) | (dates.day == dates.days_in_month)).astype(int)
    df["after_payday"] = ((dates.day.isin([16, 17])) | (dates.day <= 2)).astype(int)
    nxt = np.where(dates.day <= 15, 15 - dates.day, dates.days_in_month - dates.day)
    df["days_to_payday"] = nxt
    k = (dates - QUAKE).days
    df["quake"] = ((k >= 0) & (k < 7 * QUAKE_WEEKS)).astype(int)
    for w in range(QUAKE_WEEKS):
        df[f"quake_week{w + 1}"] = ((k >= 7 * w) & (k < 7 * (w + 1))).astype(int)
    df["quake_regional"] = df["quake"] * int(store is not None and store["state"] in QUAKE_STATES)
    for m in range(2, 13):
        df[f"month_{m}"] = (dates.month == m).astype(int)
    for d in range(1, 7):
        df[f"dow_{d}"] = (dates.dayofweek == d).astype(int)
    if oil is not None:
        o = oil.assign(date=pd.to_datetime(oil["date"])).set_index("date")["dcoilwtico"]
        o = o.reindex(dates.union(o.index)).ffill().bfill().reindex(dates)
        lo = np.log(o)
        df["oil"] = lo.values
        df["oil_lag7"] = lo.shift(7).bfill().values
        df["oil_lag28"] = lo.shift(28).bfill().values
        df["oil_grad7"] = (lo - lo.shift(7)).fillna(0).values
    return df


def holiday_report(holidays: pd.DataFrame) -> dict:
    h = holidays.copy()
    return {"rows": len(h), "by_type": h["type"].value_counts().to_dict(), "transferred": int(h["transferred"].sum()),
            "transfer_rows": int((h["type"] == "Transfer").sum()),
            "earthquake_event_rows": int(h["description"].str.contains("Terremoto", case=False).sum()),
            "by_locale": h["locale"].value_counts().to_dict()}
