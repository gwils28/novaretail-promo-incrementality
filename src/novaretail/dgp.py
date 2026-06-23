"""Synthetic Favorita-like raw files with a known no-promotion counterfactual.

The generator writes the same five tables as the Kaggle competition (train, stores, oil, holidays_events,
transactions) with the same traps: a dense store x family x day panel, no rows on 25 December, a store that opens
late, a family never sold in one store, store-wide closures on 1 January, a transferred holiday, the earthquake
event rows, `onpromotion` as a count of promoted items. Sales follow a log-additive model (store factor as a random
walk, family weekly and yearly seasonality, payday bump, holiday effects, oil elasticity) with Poisson sampling so
that low-volume families are intermittent.

Promotions are planted campaigns: `onpromotion` jumps to a count for a few days, sales rise by a family-specific
response to the promoted share, a pull-forward dip follows the campaign, and a declared substitute family in the
same store loses part of the lift (cannibalisation). Some families also carry a permanent low promotion count with
no effect at all, as in the real data where `onpromotion` is rarely zero after 2015. The counterfactual `y0`
(sales with every planted campaign removed, dips and cannibalisation included) is kept for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ["GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY", "BREAD/BAKERY", "POULTRY", "MEATS",
            "PERSONAL CARE", "DELI", "HOME CARE", "LIQUOR,WINE,BEER", "FROZEN FOODS", "PREPARED FOODS"]
# (family, substitute) pairs used by the generator for cannibalisation; the registry in donors.py declares its own
# exclusion list independently, so that the study can also test what happens when the two disagree.
SUBSTITUTES = {"POULTRY": "MEATS", "PREPARED FOODS": "DELI", "CLEANING": "HOME CARE", "BREAD/BAKERY": "PREPARED FOODS"}
STORES = [  # store_nbr, city, state, type, cluster (states follow the real Favorita geography)
    (1, "Quito", "Pichincha", "D", 13), (2, "Quito", "Pichincha", "D", 13), (3, "Guayaquil", "Guayas", "A", 3),
    (4, "Guayaquil", "Guayas", "B", 6), (5, "Cuenca", "Azuay", "D", 13), (6, "Manta", "Manabi", "D", 1),
    (7, "Esmeraldas", "Esmeraldas", "E", 10), (8, "Loja", "Loja", "C", 15), (9, "Ambato", "Tungurahua", "D", 13),
    (10, "Quito", "Pichincha", "A", 11),
]
WEEK = np.array([-0.05, -0.10, -0.08, -0.02, 0.08, 0.20, -0.03])  # Monday .. Sunday, log scale
QUAKE = pd.Timestamp("2016-04-16")
QUAKE_STATES = ("Manabi", "Esmeraldas")


@dataclass(frozen=True)
class SynthSpec:
    start: str = "2013-01-01"
    end: str = "2017-08-15"
    n_stores: int = 10
    families: tuple[str, ...] = tuple(FAMILIES)
    late_store: int = 10  # opens on `late_open`
    late_open: str = "2015-10-01"
    never_sold: tuple[tuple[int, str], ...] = ((8, "LIQUOR,WINE,BEER"),)
    promo_known_from: str = "2014-04-01"  # onpromotion is zero before this date, as in the real file
    n_campaigns: int = 400
    campaign_len: tuple[int, int] = (3, 14)
    response: tuple[float, float] = (0.05, 0.40)  # family-specific log response to a fully promoted family
    pull_forward: float = 0.08  # relative dip over the 10 days after a campaign, scaled by the campaign lift
    cannibalisation: float = 0.30  # share of the lift taken from the substitute family
    always_on_share: float = 0.3  # share of store-families with a permanent low promotion count (no effect)
    quake_lift: float = 0.35  # log lift in the affected states over the first two weeks
    quake_national: float = 0.10  # log lift everywhere over the first week (relief purchases)
    noise_sd: float = 0.08
    seed: int = 0

    def with_(self, **kw) -> SynthSpec:
        return replace(self, **kw)


@dataclass
class SynthPanel:
    train: pd.DataFrame  # the Kaggle train.csv columns: id, date, store_nbr, family, sales, onpromotion
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    transactions: pd.DataFrame
    truth: pd.DataFrame  # same keys as train: y0 (no-promotion counterfactual, on the sales scale), campaign_id
    campaigns: pd.DataFrame  # planted campaigns: campaign_id, store_nbr, family, start, end, count, true_rel_lift
    spec: SynthSpec = field(repr=False, default=SynthSpec())

    def write(self, folder: str | Path) -> Path:
        """Write the five Kaggle-shaped csv files plus truth.parquet and campaigns.parquet."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("train", "stores", "oil", "holidays", "transactions"):
            fname = "holidays_events" if name == "holidays" else name
            getattr(self, name).to_csv(folder / f"{fname}.csv", index=False)
        self.truth.to_parquet(folder / "truth.parquet", index=False)
        self.campaigns.to_parquet(folder / "campaigns.parquet", index=False)
        return folder


def _holidays(years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        nat = [(f"{y}-01-01", "Primer dia del ano"), (f"{y}-05-01", "Dia del Trabajo"),
               (f"{y}-05-24", "Batalla de Pichincha"), (f"{y}-08-10", "Primer Grito de Independencia"),
               (f"{y}-10-09", "Independencia de Guayaquil"), (f"{y}-11-02", "Dia de Difuntos"),
               (f"{y}-11-03", "Independencia de Cuenca"), (f"{y}-12-25", "Navidad")]
        for d, desc in nat:
            rows.append((d, "Holiday", "National", "Ecuador", desc, False))
        rows.append((f"{y}-12-24", "Additional", "National", "Ecuador", "Navidad-1", False))
        rows.append((f"{y}-07-25", "Holiday", "Local", "Guayaquil", "Fundacion de Guayaquil", False))
        rows.append((f"{y}-11-18", "Holiday", "Regional", "Loja", "Provincializacion de Loja", False))
    df = pd.DataFrame(rows, columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    # one transferred holiday, as in the real file: 24 May 2016 moved to Friday 27 May 2016
    df.loc[(df["date"] == "2016-05-24"), "transferred"] = True
    extra = pd.DataFrame([("2016-05-27", "Transfer", "National", "Ecuador", "Traslado Batalla de Pichincha", False),
                          ("2016-01-02", "Work Day", "National", "Ecuador", "Recupero puente", False)],
                         columns=df.columns)
    quake_days = pd.date_range(QUAKE, periods=31, freq="D")
    quake = pd.DataFrame([(str(d.date()), "Event", "National", "Ecuador", f"Terremoto Manabi+{k}", False)
                          for k, d in enumerate(quake_days)], columns=df.columns)
    df = pd.concat([df, extra, quake], ignore_index=True).sort_values("date").reset_index(drop=True)
    return df


def make_synthetic(spec: SynthSpec = SynthSpec()) -> SynthPanel:
    rng = np.random.default_rng(spec.seed)
    dates = pd.date_range(spec.start, spec.end, freq="D")
    n_days = len(dates)
    stores = pd.DataFrame(STORES[: spec.n_stores], columns=["store_nbr", "city", "state", "type", "cluster"])
    fams = list(spec.families)
    S, F = len(stores), len(fams)

    # --- covariates --------------------------------------------------------------------------------------
    oil = 95 + np.cumsum(rng.normal(0, 1.2, n_days))
    oil = np.clip(oil - 0.03 * np.arange(n_days), 25, None)  # a slow decline, as in 2014-2016
    oil_df = pd.DataFrame({"date": dates, "dcoilwtico": np.round(oil, 2)})
    oil_df.loc[oil_df["date"].dt.dayofweek >= 5, "dcoilwtico"] = np.nan  # no quotes on weekends
    holidays = _holidays(range(dates[0].year, dates[-1].year + 1))
    hol = holidays[(holidays["locale"] == "National") & (holidays["type"].isin(["Holiday", "Transfer", "Additional"]))
                   & (~holidays["transferred"])]
    hol_dates = pd.to_datetime(hol["date"])
    is_holiday = np.isin(dates, hol_dates)
    day_before = np.isin(dates.shift(1, freq="D"), hol_dates)
    payday = (dates.day == 15) | (dates.day == dates.days_in_month)
    after_pay = np.isin(dates, dates[payday].shift(1, freq="D")) | np.isin(dates, dates[payday].shift(2, freq="D"))
    doy = dates.dayofyear.values / 365.25
    quake_k = (dates - QUAKE).days.values

    # --- latent structure ----------------------------------------------------------------------------------
    store_factor = np.cumsum(rng.normal(0, 0.006, size=(n_days, S)), axis=0)  # slow local level per store
    store_scale = rng.uniform(-0.6, 0.6, S)
    fam_scale = rng.uniform(1.5, 7.0, F)  # log mean sales per family
    fam_year = rng.uniform(0.0, 0.25, F)
    fam_phase = rng.uniform(0, 2 * np.pi, F)
    fam_week = rng.uniform(0.6, 1.4, F)
    oil_beta = rng.uniform(-0.15, 0.0, F)  # Ecuador's economy follows oil; sales mildly pro-cyclical
    oil_z = (np.log(oil) - np.log(oil).mean()) / np.log(oil).std()
    response = dict(zip(fams, rng.uniform(*spec.response, F), strict=True))

    log_y0 = np.empty((n_days, S, F))
    for s in range(S):
        for f in range(F):
            log_y0[:, s, f] = (fam_scale[f] + store_scale[s] + store_factor[:, s]
                               + fam_week[f] * WEEK[dates.dayofweek.values]
                               + fam_year[f] * np.sin(2 * np.pi * doy + fam_phase[f])
                               + 0.12 * payday + 0.06 * after_pay + 0.15 * day_before - 0.25 * is_holiday
                               + oil_beta[f] * oil_z)
    # earthquake: a regional lift for two weeks, a national one for a week, then a regional dip
    affected = stores["state"].isin(QUAKE_STATES).values
    q_reg = np.where((quake_k >= 0) & (quake_k < 14), spec.quake_lift,
                     np.where((quake_k >= 14) & (quake_k < 42), -0.10, 0.0))
    q_nat = np.where((quake_k >= 0) & (quake_k < 7), spec.quake_national, 0.0)
    log_y0 += q_nat[:, None, None] + (q_reg[:, None] * affected[None, :])[:, :, None]

    # --- promotions ----------------------------------------------------------------------------------------
    onpromo = np.zeros((n_days, S, F), dtype=int)
    known_from = np.searchsorted(dates, pd.Timestamp(spec.promo_known_from))
    always_on = rng.random((S, F)) < spec.always_on_share
    for s, f in zip(*np.where(always_on), strict=True):
        onpromo[known_from:, s, f] = rng.poisson(1.5, n_days - known_from)
    open_from = np.searchsorted(dates, pd.Timestamp(spec.late_open))
    items = rng.integers(20, 120, size=F)  # items per family, so that the promoted share is count / items
    lift = np.zeros_like(log_y0)
    camp_id = np.full((n_days, S, F), -1, dtype=int)
    rows = []
    cid = 0
    tries = 0
    while cid < spec.n_campaigns and tries < spec.n_campaigns * 20:
        tries += 1
        s, f = rng.integers(S), rng.integers(F)
        L = rng.integers(spec.campaign_len[0], spec.campaign_len[1] + 1)
        t0 = rng.integers(known_from + 60, n_days - L - 20)
        if (camp_id[t0 - 150 : t0 + L + 20, s, f] >= 0).any():  # keep campaigns apart so the truth stays readable
            continue
        if stores["store_nbr"].iloc[s] == spec.late_store and t0 < open_from:
            continue
        if (stores["store_nbr"].iloc[s], fams[f]) in spec.never_sold:
            continue
        count = rng.integers(max(3, items[f] // 6), items[f] // 2 + 1)
        share = count / items[f]
        eff = response[fams[f]] * share
        onpromo[t0 : t0 + L, s, f] += count
        lift[t0 : t0 + L, s, f] += eff
        dip = -spec.pull_forward * (eff / 0.10)  # a 10% lift leaves an 8% dip over the next ten days
        lift[t0 + L : t0 + L + 10, s, f] += dip
        sub = SUBSTITUTES.get(fams[f])
        if sub in fams:
            lift[t0 : t0 + L, s, fams.index(sub)] -= spec.cannibalisation * eff
        camp_id[t0 : t0 + L, s, f] = cid
        rows.append({"campaign_id": cid, "store_nbr": int(stores["store_nbr"].iloc[s]), "family": fams[f],
                     "start": dates[t0], "end": dates[t0 + L - 1], "length": int(L), "count": int(count),
                     "promoted_share": float(share), "true_log_lift": float(eff),
                     "true_rel_lift": float(np.expm1(eff)), "substitute": sub if sub in fams else None})
        cid += 1
    campaigns = pd.DataFrame(rows)

    noise = rng.normal(0, spec.noise_sd, size=log_y0.shape)
    mu0 = np.exp(log_y0 + noise)
    mu1 = np.exp(log_y0 + lift + noise)
    y0 = rng.poisson(mu0).astype(float)  # Poisson sampling: intermittent zeros at low volume
    y1 = np.where(lift != 0, rng.poisson(mu1), y0).astype(float)
    # structural zeros: late opening, never sold, store-wide closures on 1 January
    late = stores.index[stores["store_nbr"] == spec.late_store]
    if len(late):
        y0[:open_from, late[0], :] = 0
        y1[:open_from, late[0], :] = 0
        onpromo[:open_from, late[0], :] = 0
    for store_nbr, fam in spec.never_sold:
        if fam in fams and (stores["store_nbr"] == store_nbr).any():
            s = stores.index[stores["store_nbr"] == store_nbr][0]
            y0[:, s, fams.index(fam)] = 0
            y1[:, s, fams.index(fam)] = 0
    closed = (dates.month == 1) & (dates.day == 1)
    y0[closed] = 0
    y1[closed] = 0

    # --- long tables, Kaggle shape; 25 December rows are dropped as in the real file ------------------------
    idx = pd.MultiIndex.from_product([dates, stores["store_nbr"], fams], names=["date", "store_nbr", "family"])
    train = pd.DataFrame({"sales": y1.ravel(), "onpromotion": onpromo.ravel()}, index=idx).reset_index()
    truth = pd.DataFrame({"y0": y0.ravel(), "campaign_id": camp_id.ravel()}, index=idx).reset_index()
    keep = ~((train["date"].dt.month == 12) & (train["date"].dt.day == 25))
    train, truth = train[keep].reset_index(drop=True), truth[keep].reset_index(drop=True)
    train.insert(0, "id", np.arange(len(train)))
    transactions = (train.groupby(["date", "store_nbr"])["sales"].sum().pipe(lambda s: (s / 15).round())
                    .rename("transactions").reset_index())
    transactions = transactions[transactions["transactions"] > 0]
    return SynthPanel(train=train, stores=stores, oil=oil_df, holidays=holidays, transactions=transactions,
                      truth=truth, campaigns=campaigns, spec=spec)
