"""Per-campaign estimation: assemble the series, fit the counterfactual, read the effect in units.

The counterfactual is `causal_ts.bsts.causal_impact` (local level plus static regression on the donors and the
calendar regressors, maximum likelihood on the pre period, joint simulated paths). Everything happens on the
`log1p` scale; effects in units come from the simulated paths back-transformed day by day, split into the
campaign window (the lift) and the horizon after it (the pull-forward dip), so that the net effect is the sum
of the two. The credibility verdict applies the thresholds of plan.md section 4 to the pre-period fit, the
placebo-in-time sweep and the donor isolation; a campaign that fails any of them is "non mesurable", which is a
result, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from causal_ts.baselines import CfEstimate, synthetic_control
from causal_ts.bsts import ImpactResult, causal_impact
from sklearn.linear_model import RidgeCV
from statsmodels.stats.diagnostic import acorr_ljungbox

from novaretail.features import inverse, regressors, transform

PROTOCOL = {"r2_min": 0.5, "coverage_range": (0.90, 0.98), "ljung_box_p_min": 0.01, "placebo_dates": 8,
            "placebo_max_firing": 2, "min_pre_days": 120, "wild_paths_max": 0.05}


@dataclass
class CampaignEstimate:
    campaign: dict
    pool: list[tuple[int, str]]
    result: ImpactResult
    units: dict = field(default_factory=dict)  # during / after / net effects in sales units with intervals
    quality: dict = field(default_factory=dict)
    kind: str = "log1p"

    def summary(self) -> dict:
        s = self.result.summary()
        return {**{k: v for k, v in self.campaign.items() if k in ("campaign_id", "store_nbr", "family", "start",
                                                                   "end", "length", "n_post", "mean_count")},
                "n_donors": len(self.pool), "rel_log": s["rel_effect"], "rel_log_lower": s["rel_lower"],
                "rel_log_upper": s["rel_upper"], "p_tail": s["p_tail"], **self.units, **self.quality}


def assemble(wide_sales: pd.DataFrame, calendars: dict[int, pd.DataFrame], campaign: pd.Series,
             pool: list[tuple[int, str]], kind: str = "log1p", use_oil: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """Target series and regressor matrix (donors on the same scale plus calendar) over [pre_start, post_end]."""
    target = (int(campaign["store_nbr"]), str(campaign["family"]))
    sl = slice(campaign["pre_start"], campaign["post_end"])
    y = transform(wide_sales.loc[sl, target].astype(float), kind).rename("target")
    donors = transform(wide_sales.loc[sl, pool].astype(float), kind)
    donors.columns = [f"d_{s}_{f}" for s, f in pool]
    # a donor's structural zero (store closure) is a gap in a regressor: interpolated, since the target is
    # missing on the same day in the same store and the state-space model cannot take NaN in exog
    donors = donors.interpolate(limit_direction="both")
    cal = regressors(calendars[target[0]].loc[sl], use_oil=use_oil)
    X = pd.concat([donors, cal], axis=1)
    y.index = pd.DatetimeIndex(y.index, freq="D")
    X.index = y.index
    return y, X


def pre_fit_quality(result: ImpactResult, burn: int = 14) -> dict:
    """Same quantities as `ImpactResult.pre_fit_quality`, robust to missing observations in the pre period."""
    y_pre = result.y.loc[: result.pre_end]
    f = result.fit.iloc[burn:]
    yv = y_pre.values[burn:]
    ok = np.isfinite(yv)
    err = (yv - f["mean"].values)[ok]
    inside = ((yv >= f["lower"].values) & (yv <= f["upper"].values))[ok]
    sr = np.asarray(result.std_resid).ravel()[burn:]
    sr = sr[np.isfinite(sr)]
    lb = acorr_ljungbox(sr, lags=[min(10, max(1, len(sr) // 5))], return_df=True)
    return {"rmse_pre": float(np.sqrt(np.mean(err**2))), "r2_one_step": float(1 - err.var() / yv[ok].var()),
            "coverage_one_step": float(inside.mean()), "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
            "n_pre_obs": int(ok.sum())}


WILD_LOG_MARGIN = 3.0  # a path that climbs 3 log units (x20) above the series' own maximum is a divergent draw


def effects_in_units(result: ImpactResult, campaign_end, kind: str = "log1p") -> dict:
    """Sum of observed minus counterfactual sales over the campaign days, over the horizon after, and in total.

    Read path by path: each simulated counterfactual path is back-transformed day by day, summed over the window,
    and the effect is the observed sum minus that. The point estimate is the median over paths and the interval
    its quantiles. Divergent paths (a parameter draw with an absurd level variance sends a few paths to values
    the series never approached; on the sales scale `expm1` turns them into millions of units) are dropped and
    counted: `wild_paths_share` is a credibility check, not a silent trim.
    """
    y_post = inverse(result.y_post.values, kind)
    ceiling = float(np.nanmax(result.y.values)) + WILD_LOG_MARGIN
    wild = (result.paths > ceiling).any(axis=1)
    paths = inverse(result.paths[~wild], kind)  # (nsim - wild, n_post)
    during = result.cf.index <= pd.Timestamp(campaign_end)
    a = result.alpha
    out = {"wild_paths_share": float(wild.mean())}
    if len(paths) < 50:
        return {**out, **{f"{n}_{k}": np.nan for n in ("during", "after", "net")
                          for k in ("units", "lower", "upper", "rel", "rel_lower", "rel_upper", "cf_units")},
                "pull_forward_share": np.nan}
    for name, mask in (("during", during), ("after", ~during), ("net", np.ones_like(during))):
        if mask.sum() == 0:
            out.update({f"{name}_units": 0.0, f"{name}_lower": 0.0, f"{name}_upper": 0.0, f"{name}_rel": np.nan})
            continue
        obs = float(y_post[mask].sum())
        cf = paths[:, mask].sum(axis=1)
        eff = obs - cf
        lo, hi = np.quantile(eff, [a / 2, 1 - a / 2])
        rel = eff / cf
        rlo, rhi = np.quantile(rel, [a / 2, 1 - a / 2])
        out.update({f"{name}_units": float(np.median(eff)), f"{name}_lower": float(lo), f"{name}_upper": float(hi),
                    f"{name}_rel": float(np.median(rel)), f"{name}_rel_lower": float(rlo),
                    f"{name}_rel_upper": float(rhi), f"{name}_cf_units": float(np.median(cf)),
                    f"{name}_days": int(mask.sum())})
    out["pull_forward_share"] = float(-out["after_units"] / out["during_units"]) if out["during_units"] > 0 else np.nan
    return out


def estimate_campaign(y: pd.Series, X: pd.DataFrame, campaign: pd.Series, pool: list[tuple[int, str]], *,
                      level: str = "llevel", param_draws: int = 0, nsim: int = 2000, seed: int = 0,
                      kind: str = "log1p") -> CampaignEstimate:
    pre_end = campaign["start"] - pd.Timedelta(1, unit="D")
    r = causal_impact(y, X, pre_end, post_end=campaign["post_end"], level=level, param_draws=param_draws,
                      nsim=nsim, seed=seed)
    est = CampaignEstimate(campaign=dict(campaign), pool=list(pool), result=r, kind=kind)
    est.units = effects_in_units(r, campaign["end"], kind)
    est.quality = pre_fit_quality(r)
    return est


def credibility(quality: dict, sweep: pd.DataFrame | None = None, removed_donors: list | None = None,
                protocol: dict = PROTOCOL, units: dict | None = None) -> dict:
    """Protocol verdict: every check named with its value and whether it passes; `measurable` is the conjunction."""
    lo, hi = protocol["coverage_range"]
    checks = {
        "fit_r2": (quality["r2_one_step"], quality["r2_one_step"] >= protocol["r2_min"]),
        "fit_coverage": (quality["coverage_one_step"], lo <= quality["coverage_one_step"] <= hi),
        "fit_ljung_box": (quality["ljung_box_p"], quality["ljung_box_p"] >= protocol["ljung_box_p_min"]),
    }
    if units is not None and "wild_paths_share" in units:
        checks["paths_sane"] = (units["wild_paths_share"], units["wild_paths_share"] <= protocol["wild_paths_max"])
    if sweep is not None:
        firing = int(sweep["excludes_zero"].sum())
        checks["placebo_in_time"] = (firing, firing <= protocol["placebo_max_firing"])
    if removed_donors is not None:
        checks["donor_isolation"] = (len(removed_donors), True)  # removal is a correction, not a failure
    verdict = {f"check_{k}": v for k, (_, v) in checks.items()}
    verdict.update({f"value_{k}": val for k, (val, _) in checks.items()})
    verdict["measurable"] = all(v for _, v in checks.values())
    return verdict


def augmented_sc(y: pd.Series, X: pd.DataFrame, pre_end, alpha: float = 0.05) -> CfEstimate:
    """Ridge-augmented synthetic control (Ben-Michael, Feller and Rothstein, 2021), donor-only version.

    Convex weights from the classical synthetic control, plus a correction for the pre-period imbalance: for each
    post date, a ridge regression across donors of their post value on their pre-period path gives an outcome
    model m; the counterfactual is X_post w + (y_pre - X_pre w)' eta. Only the donor columns are used (calendar
    regressors are the same for every unit and carry no cross-sectional information). Interval: iid from the
    pre-period residuals, scored by coverage like the other baselines.
    """
    donors = [c for c in X.columns if c.startswith("d_")]
    Xd = X[donors]
    y_pre, y_post = y.loc[:pre_end], y.loc[pre_end:].iloc[1:]
    ok = y_pre.notna()
    y_pre_ok = y_pre[ok]
    sc = synthetic_control(y_pre_ok.combine_first(y).loc[:y_post.index[-1]].interpolate(), Xd, pre_end, alpha)
    w = sc.weights.values
    A = Xd.loc[y_pre_ok.index].values  # (T0, J)
    gap = y_pre_ok.values - A @ w  # pre-period imbalance (T0,)
    post = Xd.loc[y_post.index].values  # (T1, J)
    cf = post @ w
    corr = np.zeros(len(y_post))
    for t in range(len(y_post)):
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(A.T, post[t])  # observations = donors
        corr[t] = gap @ ridge.coef_
    cf = cf + corr
    # in-sample augmented residuals on the pre period (leave-one-out ridge across donors is the same model)
    fitted_pre = np.zeros(len(y_pre_ok))
    for t in range(len(y_pre_ok)):
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(A.T, A[t])
        fitted_pre[t] = A[t] @ w + gap @ ridge.coef_
    resid_sd = float(np.std(y_pre_ok.values - fitted_pre, ddof=1))
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2)
    n = len(y_post)
    cf_df = pd.DataFrame({"mean": cf, "lower": cf - z * resid_sd, "upper": cf + z * resid_sd}, index=y_post.index)
    cum = float(np.nansum(y_post.values - cf))
    half = z * resid_sd * np.sqrt(n)
    return CfEstimate("augmented_sc", cf_df, cum, cum - half, cum + half, pd.Series(w, index=donors))
