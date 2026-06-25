"""Campaign detection: turning the `onpromotion` count into measurable treatment windows.

The data has no campaign table. For one store-family, `onpromotion` has a background level (often non-zero from
2015 on) and episodes where it jumps. A campaign is a run of days where the count exceeds the running background
by a margin, long enough to matter, preceded by a pre period quiet enough to learn the counterfactual from, and
followed by a horizon in which the pull-forward dip can show. Everything below is a rule fixed before the
results (plan.md section 4); the number of campaigns that survive the rules is itself a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionRule:
    background_window: int = 60  # rolling median window for the background count
    ratio: float = 1.5  # active if count >= max(min_count, ratio * background + margin + margin_sd * sqrt(background))
    margin: float = 2.0
    margin_sd: float = 2.0  # Poisson-scale margin: a jump must clear the day-to-day noise of the count
    min_count: int = 3
    min_len: int = 3  # days; one- or two-day blips of a noisy background are not campaigns
    max_len: int = 21  # longer runs are a regime change, not a campaign
    merge_gap: int = 1  # runs separated by at most this many days are one campaign
    n_pre: int = 120  # days of pre period required
    quiet_share: float = 0.10  # max share of active days in the pre period
    clean_tail: int = 30  # no active day in the last `clean_tail` days before the start
    horizon: int = 14  # post window after the campaign end (pull-forward)
    max_structural_share: float = 0.05  # structural zeros tolerated in the pre period


def active_days(count: pd.Series, rule: DetectionRule = DetectionRule()) -> pd.Series:
    """Boolean series: the count is above its running background."""
    bg = count.shift(1).rolling(rule.background_window, min_periods=14).median()
    bg = bg.fillna(0)
    thr = np.maximum(rule.min_count, rule.ratio * bg + rule.margin + rule.margin_sd * np.sqrt(bg))
    return (count >= thr) & count.notna()


def runs(active: pd.Series, merge_gap: int) -> list[tuple[int, int]]:
    """Start and end positions (inclusive) of runs of True, merging runs separated by short gaps."""
    a = active.values.astype(int)
    if a.sum() == 0:
        return []
    d = np.diff(np.concatenate([[0], a, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    out = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:], strict=True):
        if s - out[-1][1] - 1 <= merge_gap:
            out[-1] = (out[-1][0], int(e))
        else:
            out.append((int(s), int(e)))
    return out


def campaign_days(count: pd.Series, usable: pd.Series, rule: DetectionRule = DetectionRule()) -> pd.Series:
    """Boolean series: active days that belong to a run of at least `min_len` days (blips excluded)."""
    act = active_days(count, rule) & usable
    out = pd.Series(False, index=count.index)
    for s, e in runs(act, rule.merge_gap):
        if act.iloc[s : e + 1].sum() >= rule.min_len:
            out.iloc[s : e + 1] = True
    return out


def detect_unit(count: pd.Series, usable: pd.Series, rule: DetectionRule = DetectionRule()) -> pd.DataFrame:
    """Campaigns of one store-family. `usable` is False on structural zeros and on promo-unknown dates.

    Returns one row per run with the eligibility verdict and its reason; eligible rows are the campaigns. The
    quiet-pre-period and clean-tail rules count campaign days (runs of `min_len` or more), not isolated blips.
    """
    act = active_days(count, rule) & usable
    camp = campaign_days(count, usable, rule)
    idx = count.index
    n = len(idx)
    rows = []
    rr = runs(act, rule.merge_gap)
    for k, (s, e) in enumerate(rr):
        L = e - s + 1
        n_active = int(act.iloc[s : e + 1].sum())
        pre = slice(max(0, s - rule.n_pre), s)
        post_end = min(n - 1, e + rule.horizon)
        reason = None
        truncated = False
        if n_active < rule.min_len:
            reason = "too_short"
        elif L > rule.max_len:
            reason = "too_long"
        elif s < rule.n_pre:
            reason = "no_pre_period"
        elif (~usable.iloc[pre]).mean() > rule.max_structural_share or (~usable.iloc[s : post_end + 1]).any():
            reason = "structural_zeros"
        elif camp.iloc[pre].mean() > rule.quiet_share:
            reason = "pre_not_quiet"
        elif camp.iloc[max(0, s - rule.clean_tail) : s].any():
            reason = "promo_in_clean_tail"
        if k + 1 < len(rr) and rr[k + 1][0] <= post_end:
            truncated = True
            post_end = rr[k + 1][0] - 1
        bg = float(count.shift(1).rolling(rule.background_window, min_periods=14).median().iloc[s] or 0)
        rows.append({"start": idx[s], "end": idx[e], "length": L, "n_active": n_active, "pre_start": idx[pre.start],
                     "post_end": idx[post_end], "n_post": post_end - e, "post_truncated": truncated,
                     "mean_count": float(count.iloc[s : e + 1].mean()), "background": bg,
                     "pre_active_share": float(camp.iloc[pre].mean()), "eligible": reason is None,
                     "reason": reason or "eligible"})
    return pd.DataFrame(rows)


def detect_all(wide_promo: pd.DataFrame, usable: pd.DataFrame, rule: DetectionRule = DetectionRule(),
               units: list[tuple[int, str]] | None = None) -> pd.DataFrame:
    """Run the detection on every store-family (or the given units); campaign_id over the eligible rows."""
    out = []
    for unit in units or list(wide_promo.columns):
        df = detect_unit(wide_promo[unit], usable[unit], rule)
        if len(df):
            df.insert(0, "family", unit[1])
            df.insert(0, "store_nbr", unit[0])
            out.append(df)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True).sort_values(["start", "store_nbr", "family"]).reset_index(drop=True)
    res["campaign_id"] = np.where(res["eligible"], np.cumsum(res["eligible"]) - 1, -1)
    return res


def detection_report(runs_df: pd.DataFrame) -> dict:
    if runs_df.empty:
        return {"runs": 0, "eligible": 0}
    return {"runs": int(len(runs_df)), "eligible": int(runs_df["eligible"].sum()),
            "by_reason": runs_df["reason"].value_counts().to_dict(),
            "length_quantiles": runs_df.loc[runs_df["eligible"], "length"].quantile([0.1, 0.5, 0.9]).to_dict(),
            "post_truncated": int(runs_df.loc[runs_df["eligible"], "post_truncated"].sum())}
