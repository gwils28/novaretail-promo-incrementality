"""The donor registry: which series may serve as controls for a campaign, decided by design before any fit.

Rules (block 3 of the tech watch, applied to Favorita):
1. a donor is another family in the same store with no declared substitution or complement edge to the target
   family, or the same family in another store of the same cluster or state (same-family donors carry the
   family's seasonality but can be reached by customers of the treated store only across cities, which the
   registry accepts for different cities and refuses for the same city);
2. a donor is not itself in a campaign (active promotion by the detection rule) from 30 days before the start
   to the end of the post window; its background promotion level is allowed, as it is in the target;
3. a donor has no structural zero in the campaign and post window, at most the tolerated share in the pre
   period (a store-wide closure such as 1 January hits the target the same day; the gap is interpolated in the
   regressor and the day is missing in the target), and is not a low-volume series (median below `min_median`);
4. the pool is capped by the pre-period correlation of weekly differences with the target, and the cap is a
   reported choice: the sensitivity analysis re-estimates with pools of several sizes.
Fit quality never removes a donor that the rules admit, and never admits one that they refuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from novaretail.campaigns import DetectionRule, active_days

# Declared before the results: pairs of Favorita families that plausibly substitute or complement each other.
# A same-store donor from a family linked to the target is refused whatever its fit.
SUBSTITUTION_EDGES = [
    ("GROCERY I", "GROCERY II"), ("HOME AND KITCHEN I", "HOME AND KITCHEN II"), ("CLEANING", "HOME CARE"),
    ("BEAUTY", "PERSONAL CARE"), ("LADIESWEAR", "LINGERIE"), ("BOOKS", "MAGAZINES"),
    ("HOME APPLIANCES", "PLAYERS AND ELECTRONICS"), ("HARDWARE", "LAWN AND GARDEN"),
    ("MEATS", "POULTRY"), ("MEATS", "SEAFOOD"), ("POULTRY", "SEAFOOD"), ("DELI", "PREPARED FOODS"),
    ("BREAD/BAKERY", "PREPARED FOODS"), ("DELI", "MEATS"), ("BEVERAGES", "LIQUOR,WINE,BEER"),
    ("DAIRY", "EGGS"), ("FROZEN FOODS", "PREPARED FOODS"), ("PRODUCE", "FROZEN FOODS"),
    ("GROCERY I", "BREAD/BAKERY"), ("GROCERY I", "DAIRY"), ("GROCERY I", "BEVERAGES"),
    ("BABY CARE", "PERSONAL CARE"), ("PET SUPPLIES", "GROCERY II"), ("CELEBRATION", "SCHOOL AND OFFICE SUPPLIES"),
]


def linked_families(family: str, edges=SUBSTITUTION_EDGES) -> set[str]:
    return {b for a, b in edges if a == family} | {a for a, b in edges if b == family}


@dataclass(frozen=True)
class RegistryRule:
    max_pool: int = 12
    min_median: float = 5.0  # units per day, on the raw scale
    same_family_other_store: bool = True
    allow_same_city: bool = False
    pre_window_before_start: int = 30


def eligible_donors(wide_sales_raw: pd.DataFrame, wide_promo: pd.DataFrame, usable: pd.DataFrame,
                    stores: pd.DataFrame, campaign: pd.Series, rule: RegistryRule = RegistryRule(),
                    detection: DetectionRule = DetectionRule(), edges=SUBSTITUTION_EDGES) -> pd.DataFrame:
    """One row per candidate with its kind, the reason of refusal (or 'kept') and the pre-period correlation."""
    target = (int(campaign["store_nbr"]), str(campaign["family"]))
    st = stores.set_index("store_nbr")
    s_info = st.loc[target[0]]
    pre_start, start, post_end = campaign["pre_start"], campaign["start"], campaign["post_end"]
    win_start = start - pd.Timedelta(rule.pre_window_before_start, unit="D")
    linked = linked_families(target[1], edges)
    y_pre = np.log1p(wide_sales_raw.loc[pre_start:start, target].iloc[:-1]).diff(7).dropna()

    rows = []
    for unit in wide_sales_raw.columns:
        if unit == target:
            continue
        s, f = int(unit[0]), str(unit[1])
        same_store, same_family = s == target[0], f == target[1]
        if same_store:
            kind = "same_store_other_family"
        elif same_family and rule.same_family_other_store:
            other = st.loc[s]
            if other["cluster"] != s_info["cluster"] and other["state"] != s_info["state"]:
                continue
            if other["city"] == s_info["city"] and not rule.allow_same_city:
                kind, reason = "same_family_other_store", "same_city"
                rows.append({"store_nbr": s, "family": f, "kind": kind, "reason": reason, "corr_pre": np.nan})
                continue
            kind = "same_family_other_store"
        else:
            continue
        reason = "kept"
        if same_store and f in linked:
            reason = "substitution_edge"
        elif (~usable.loc[start:post_end, unit]).any() or \
                (~usable.loc[pre_start:start, unit]).mean() > detection.max_structural_share:
            reason = "structural_zeros"
        elif wide_sales_raw.loc[pre_start:start, unit].median() < rule.min_median:
            reason = "low_volume"
        elif active_days(wide_promo[unit], detection).loc[win_start:post_end].any():
            reason = "promoted_in_window"
        x_pre = np.log1p(wide_sales_raw.loc[pre_start:start, unit].iloc[:-1]).diff(7).dropna()
        aligned = len(x_pre) == len(y_pre) and x_pre.std() > 0
        corr = float(np.corrcoef(y_pre.values, x_pre.values)[0, 1]) if aligned else np.nan
        rows.append({"store_nbr": s, "family": f, "kind": kind, "reason": reason, "corr_pre": corr})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    kept = df[df["reason"] == "kept"].sort_values("corr_pre", ascending=False)
    df["in_pool"] = False
    df.loc[kept.index[: rule.max_pool], "in_pool"] = True
    df.loc[kept.index[rule.max_pool :], "reason"] = "beyond_cap"
    return df.sort_values(["in_pool", "corr_pre"], ascending=[False, False]).reset_index(drop=True)


def pool_units(table: pd.DataFrame) -> list[tuple[int, str]]:
    return [(int(r.store_nbr), str(r.family)) for r in table[table["in_pool"]].itertuples()]


def registry_report(table: pd.DataFrame) -> dict:
    return {"candidates": int(len(table)), "pool": int(table["in_pool"].sum()),
            "by_reason": table["reason"].value_counts().to_dict(),
            "by_kind_in_pool": table.loc[table["in_pool"], "kind"].value_counts().to_dict()}
