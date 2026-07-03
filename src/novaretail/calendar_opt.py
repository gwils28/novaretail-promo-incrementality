"""Promotion calendar reallocation from measured uplifts, at constant budget.

The budget is counted in campaign-days per store (the data has no cost; a campaign-day is the unit the brief
manipulates). Each (store, family) gets an expected net uplift per campaign-day from the measured campaigns:
the lower bound of the net interval by default (the decision value of the monitoring block), the point estimate
once the family is reconciled. The greedy allocation gives campaign-days to the highest value per day first, with
two constraints: no family above its historical share times a cap (the response is measured at the historical
intensity, not beyond), and no two linked families (substitution edges) promoted the same week in the same store.
A reallocation is only as good as the uplifts it reads; the output carries the share of the budget that moves and
the families whose value is unmeasured (no credible campaign), which stay at their historical allocation.
"""

from __future__ import annotations

import pandas as pd

from novaretail.donors import linked_families


def value_per_day(estimates: pd.DataFrame, use_lower: bool = True, reconciled: set[str] | None = None) -> pd.DataFrame:
    """Per (store, family): mean net uplift per campaign-day over measurable campaigns, and the historical days."""
    e = estimates[estimates["measurable"]].copy()
    rec = reconciled or set()
    e["value"] = e["net_lower"].where(use_lower & ~e["family"].isin(rec), e["net_units"])
    e["days"] = e["length"]
    g = e.groupby(["store_nbr", "family"], observed=True).agg(value=("value", "sum"), days=("days", "sum"),
                                                                n_campaigns=("campaign_id", "size"))
    g["value_per_day"] = g["value"] / g["days"]
    hist = estimates.groupby(["store_nbr", "family"], observed=True)["length"].sum().rename("hist_days")
    return g.join(hist, how="outer").fillna({"n_campaigns": 0, "hist_days": 0})


def reallocate(values: pd.DataFrame, cap_ratio: float = 2.0, step: int = 1) -> pd.DataFrame:
    """Greedy reallocation of each store's historical campaign-days to the best measured value per day."""
    rows = []
    for store, g in values.groupby(level="store_nbr"):
        budget = int(g["hist_days"].sum())
        measured = g[g["n_campaigns"] > 0].copy()
        unmeasured = g[g["n_campaigns"] == 0]
        alloc = {fam: 0 for fam in g.index.get_level_values("family")}
        for (_, fam), r in unmeasured.iterrows():
            alloc[fam] = int(r["hist_days"])  # stays as is: nothing measured, nothing moved
        remaining = budget - sum(alloc.values())
        caps = {fam: int(max(r["hist_days"], 1) * cap_ratio) for (_, fam), r in measured.iterrows()}
        order = measured.sort_values("value_per_day", ascending=False)
        while remaining > 0:
            placed = False
            for (_, fam), r in order.iterrows():
                if r["value_per_day"] <= 0 or alloc[fam] >= caps[fam]:
                    continue
                alloc[fam] += step
                remaining -= step
                placed = True
                if remaining <= 0:
                    break
            if not placed:
                break  # no positive value left: the rest of the budget is not spent
        for (_, fam), r in g.iterrows():
            rows.append({"store_nbr": store, "family": fam, "hist_days": int(r["hist_days"]), "new_days": alloc[fam],
                         "value_per_day": r["value_per_day"], "measured": r["n_campaigns"] > 0,
                         "unspent": 0})
        if remaining > 0:
            rows[-1]["unspent"] = remaining
    out = pd.DataFrame(rows)
    out["delta_days"] = out["new_days"] - out["hist_days"]
    return out


def schedule_without_conflicts(plan: pd.DataFrame, weeks: int = 52, seed: int = 0) -> pd.DataFrame:
    """Place each family's campaign-days on weeks so that no two linked families share a week in a store."""
    rows = []
    for store, g in plan.groupby("store_nbr"):
        occupied: dict[int, set[str]] = {w: set() for w in range(weeks)}
        for r in g.sort_values("new_days", ascending=False).itertuples():
            days = int(r.new_days)
            linked = linked_families(r.family)
            w = 0
            while days > 0 and w < weeks:
                if not (occupied[w] & linked):
                    take = min(7, days)
                    occupied[w].add(r.family)
                    rows.append({"store_nbr": store, "family": r.family, "week": w, "days": take})
                    days -= take
                w += 1
    return pd.DataFrame(rows)


def plan_summary(plan: pd.DataFrame) -> dict:
    moved = plan.loc[plan["delta_days"] > 0, "delta_days"].sum()
    total = plan["hist_days"].sum()
    return {"stores": int(plan["store_nbr"].nunique()), "budget_days": int(total), "days_moved": int(moved),
            "share_moved": float(moved / total) if total else 0.0, "unspent_days": int(plan["unspent"].sum()),
            "families_unmeasured": int((~plan["measured"]).groupby(plan["family"]).all().sum()),
            "expected_gain_units": float((plan["delta_days"] * plan["value_per_day"].fillna(0)).sum())}
