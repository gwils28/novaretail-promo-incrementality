"""Command line: one command per step, results written to data/results as parquet, json and pickles.

    nr ingest [--source favorita|synthetic]     raw files -> panel.parquet, manifest.json
    nr campaigns                                  detection on every store-family -> campaigns.parquet
    nr estimate [--max N] [--param-draws K]       per campaign: registry, counterfactual, placebos, isolation
    nr backtest [--n N]                           planted campaigns on the real panel, scores per estimator
    nr quake                                      the earthquake test bench
    nr dml                                        response to promotion intensity
    nr monitor                                    signals and the simulated reconciliation
    nr calendar                                   reallocation of campaign-days
Every number quoted in a README comes from one of these json files.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd

from novaretail.data import RAW, RESULTS, ingest


def _dump(name: str, obj) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, pd.Timestamp | pd.Period):
        return str(o)
    if isinstance(o, np.integer | np.floating | np.bool_):
        return o.item()
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="index")
    return str(o)


def cmd_ingest(args) -> None:
    if args.source == "synthetic":
        from novaretail.dgp import SynthSpec, make_synthetic

        make_synthetic(SynthSpec(end=args.synthetic_end, n_stores=args.synthetic_stores)).write(RAW / "synthetic")
    m = ingest(args.source)
    print(json.dumps(m["panel"], indent=2, default=_json_default))


def cmd_campaigns(args) -> None:
    from novaretail.campaigns import detect_all, detection_report
    from novaretail.workspace import load_world

    w = load_world()
    det = detect_all(w.wide_promo, w.usable)
    det.to_parquet(RESULTS / "campaigns.parquet", index=False)
    rep = detection_report(det)
    rep["by_family_eligible"] = det[det["eligible"]].groupby("family", observed=True).size().to_dict()
    _dump("campaigns", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "by_family_eligible"}, indent=2, default=_json_default))


def cmd_estimate(args) -> None:
    from novaretail.backtest import isolate, placebo_time
    from novaretail.donors import RegistryRule, eligible_donors, pool_units
    from novaretail.estimate import PROTOCOL, assemble, credibility, estimate_campaign
    from novaretail.workspace import load_world

    w = load_world()
    det = pd.read_parquet(RESULTS / "campaigns.parquet")
    elig = det[det["eligible"]].copy()
    if args.max and len(elig) > args.max:  # stratified by family so that every family keeps some campaigns
        elig = elig.groupby("family", observed=True, group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(args.max * len(g) / len(elig)))), random_state=args.seed))
    rows, donors_tabs, sweeps, spaces, pickles = [], [], [], [], {}
    t0 = time.time()
    for i, (_, camp) in enumerate(elig.iterrows()):
        tab = eligible_donors(w.wide_raw, w.wide_promo, w.usable, w.stores, camp, RegistryRule(max_pool=args.pool))
        pool = pool_units(tab) if len(tab) else []
        tab = tab.assign(campaign_id=camp["campaign_id"])
        donors_tabs.append(tab)
        if len(pool) < 3:
            rows.append({**camp.to_dict(), "n_donors": len(pool), "measurable": False, "status": "pool_too_small"})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            removed, kept, first = isolate(w.wide, w.calendars, camp, pool, nsim=500, seed=args.seed)
            if first is not None:
                spaces.append(first.assign(campaign_id=camp["campaign_id"]))
            if len(kept) < 3:
                rows.append({**camp.to_dict(), "n_donors": len(kept), "measurable": False, "status": "pool_too_small",
                             "removed_donors": len(removed)})
                continue
            y, X = assemble(w.wide, w.calendars, camp, kept)
            est = estimate_campaign(y, X, camp, kept, param_draws=args.param_draws, nsim=args.nsim, seed=args.seed)
            sweep = placebo_time(y, X, camp, n_dates=PROTOCOL["placebo_dates"], nsim=500, seed=args.seed)
        if len(sweep):
            sweeps.append(sweep.assign(campaign_id=camp["campaign_id"]).reset_index())
        verdict = credibility(est.quality, sweep if len(sweep) else None, removed)
        s = est.summary()
        betas = est.result.params.filter(like="beta.d_")
        rows.append({**camp.to_dict(), **s, **verdict, "removed_donors": len(removed), "isolation_rounds": len(removed),
                     "status": "measurable" if verdict["measurable"] else "not_credible",
                     "betas": {k.replace("beta.d_", ""): float(v) for k, v in betas.items()}})
        if len(pickles) < args.keep and verdict["measurable"]:
            pickles[int(camp["campaign_id"])] = est
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(elig)} campaigns, {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / "estimates.parquet", index=False)
    pd.concat(donors_tabs).to_parquet(RESULTS / "donors.parquet", index=False)
    if sweeps:
        pd.concat(sweeps).to_parquet(RESULTS / "placebo_time.parquet", index=False)
    if spaces:
        pd.concat(spaces).to_parquet(RESULTS / "placebo_space.parquet")
    with open(RESULTS / "estimates.pkl", "wb") as fh:
        pickle.dump(pickles, fh)
    meas = res[res.get("measurable", pd.Series(False, index=res.index)).fillna(False).astype(bool)]
    rep = {"campaigns_run": len(res), "by_status": res["status"].value_counts().to_dict(),
           "measurable": len(meas), "seconds": round(time.time() - t0),
           "during_rel_median": float(meas["during_rel"].median()) if len(meas) else None,
           "during_rel_quantiles": (meas["during_rel"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
                                   if len(meas) else None),
           "pull_forward_share_median": float(meas["pull_forward_share"].median()) if len(meas) else None,
           "share_net_lower_above_zero": float((meas["net_lower"] > 0).mean()) if len(meas) else None,
           "share_net_upper_below_zero": float((meas["net_upper"] < 0).mean()) if len(meas) else None,
           "checks_failed": {c: int((~res[c].astype("boolean").fillna(True)).sum())
                             for c in res.columns if c.startswith("check_")},
           "by_family": (meas.groupby("family", observed=True)["during_rel"].agg(["size", "median"])
                         .to_dict(orient="index") if len(meas) else {})}
    _dump("estimates", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "by_family"}, indent=2, default=_json_default))


def cmd_backtest(args) -> None:
    from novaretail.backtest import PlantSpec, planted_study, score
    from novaretail.workspace import load_world

    w = load_world()
    specs = [PlantSpec(rel_lift=0.0), PlantSpec(rel_lift=0.05), PlantSpec(rel_lift=0.15), PlantSpec(rel_lift=0.30),
             PlantSpec(rel_lift=0.15, shape="ramp"), PlantSpec(rel_lift=0.15, shape="pulse"),
             PlantSpec(rel_lift=0.15, pull_forward=0.5), PlantSpec(rel_lift=0.15, cannibal_share=0.5)]
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = planted_study(w.wide_raw, w.wide_promo, w.usable, w.stores, w.calendars, specs, args.n, seed=args.seed)
    study.to_parquet(RESULTS / "planted.parquet", index=False)
    sc = score(study)
    sc.to_parquet(RESULTS / "planted_scores.parquet", index=False)
    rep = {"seconds": round(time.time() - t0), "rows": len(study), "scores": sc.to_dict(orient="records")}
    _dump("backtest", rep)
    print(sc.to_string())


def cmd_quake(args) -> None:
    from novaretail.earthquake import run_quake
    from novaretail.workspace import load_world

    w = load_world()
    out = run_quake(w.panel, w.stores, w.holidays, w.oil, nsim=args.nsim, param_draws=args.param_draws, seed=args.seed)
    with open(RESULTS / "quake.pkl", "wb") as fh:
        pickle.dump(out, fh)
    rep = {"treated": out["treated"], "donors": out["donors"], "summary": out["summary"],
           "placebo_time_firing": int(out["sweep"]["excludes_zero"].sum()) if len(out["sweep"]) else None,
           "placebo_time_dates": int(len(out["sweep"])),
           "donors_flagged": out["space"][~out["space"]["is_treated"] & out["space"]["excludes_zero"]].index.tolist(),
           "weekly": out["weekly"].round(3).to_dict(orient="index"),
           "by_family": out["by_family"].round(3).to_dict(orient="index")}
    _dump("quake", rep)
    print(json.dumps({k: rep[k] for k in ("treated", "donors", "summary", "placebo_time_firing", "donors_flagged")},
                     indent=2, default=_json_default))


def cmd_dml(args) -> None:
    from novaretail.dml import build_table, by_family, controls, crosscheck_doubleml, dose_response, holdout_check, plr
    from novaretail.features import promo_intensity
    from novaretail.workspace import load_world

    w = load_world()
    t0 = time.time()
    table = build_table(w.panel, w.calendars, promo_intensity(w.wide_promo))
    if args.sample and len(table) > args.sample:
        table = table.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    x_lags, x_nolags = controls(table), controls(table, with_lags=False)
    rep = {"rows": len(table), "pooled_with_lags": plr(table, ["intensity"], x_lags, seed=args.seed),
           "pooled_without_lags": plr(table, ["intensity"], x_nolags, seed=args.seed)}
    fam = by_family(table, x_lags, seed=args.seed)
    fam.to_parquet(RESULTS / "dml_by_family.parquet")
    dr = dose_response(table, x_lags, seed=args.seed)
    dr.to_parquet(RESULTS / "dml_dose.parquet")
    rep["by_family"] = fam.round(4).to_dict(orient="index")
    rep["dose_response"] = dr.round(4).to_dict(orient="index")
    split = str(table["date"].max() - pd.DateOffset(years=1))[:10]
    rep["holdout"] = holdout_check(table, x_lags, split, seed=args.seed)
    rep["holdout_split"] = split
    try:
        rep["doubleml"] = crosscheck_doubleml(table, x_lags, seed=args.seed)
    except ImportError:
        rep["doubleml"] = "doubleml not installed (uv sync --group dml)"
    rep["seconds"] = round(time.time() - t0)
    _dump("dml", rep)
    print(json.dumps({k: rep[k] for k in ("rows", "pooled_with_lags", "pooled_without_lags", "holdout", "doubleml")},
                     indent=2, default=_json_default))


def cmd_monitor(args) -> None:
    from novaretail.monitoring import (
        fit_signals,
        interference_signals,
        reconciliation,
        relationship_drift,
        rolling_fpr,
        simulated_ledger,
    )

    est = pd.read_parquet(RESULTS / "estimates.parquet")
    est = est[est["status"] != "pool_too_small"]
    fit = fit_signals(est)
    planted = pd.read_parquet(RESULTS / "planted.parquet")
    nulls = planted[(planted["rel_lift"] == 0) & (planted["estimator"] == "bsts_llevel") & planted["error"].isna()]
    fpr = rolling_fpr(nulls, window=min(50, max(10, len(nulls) // 2)))
    iso = interference_signals(est[["campaign_id", "store_nbr", "family", "start", "removed_donors"]]
                               .rename(columns={"removed_donors": "rounds"}))
    coefs = pd.DataFrame([{"target": f"{r.store_nbr}:{r.family}", "start": r.start, "donor": d, "beta": b}
                          for r in est.itertuples() if isinstance(r.betas, dict) for d, b in r.betas.items()])
    drift = relationship_drift(coefs) if len(coefs) else pd.DataFrame()
    meas = est[est["measurable"].fillna(False).astype(bool)]
    ledger_clean = simulated_ledger(meas, seed=args.seed)
    mid = str(pd.to_datetime(meas["start"]).quantile(0.5))[:7]
    ledger_drift = simulated_ledger(meas, seed=args.seed, drift_from=mid, drift_per_month=0.02)
    rec_clean, rec_drift = reconciliation(ledger_clean), reconciliation(ledger_drift)
    rec_clean.to_parquet(RESULTS / "reconciliation_clean.parquet")
    rec_drift.to_parquet(RESULTS / "reconciliation_drift.parquet")
    fpr.to_parquet(RESULTS / "rolling_fpr.parquet", index=False)
    if len(drift):
        drift.to_parquet(RESULTS / "relationship_drift.parquet", index=False)
    rep = {"campaigns": len(est), "blocked_by_fit": int(fit["blocked"].sum()),
           "fpr_never_treated": float(nulls["excludes_zero"].mean()) if len(nulls) else None,
           "fpr_alert_share": float(fpr["alert"].mean()) if len(fpr) else None,
           "isolation_reviews": int(iso["review"].sum()),
           "donor_sign_changes": int(drift["sign_change"].sum()) if len(drift) else 0,
           "reconciliation_clean_alarms": int(rec_clean["drift_alarm"].sum()),
           "reconciliation_drift_alarms": int(rec_drift["drift_alarm"].sum()),
           "reconciliation_drift_first_alarm": (str(rec_drift.index[rec_drift["drift_alarm"]][0])
                                               if rec_drift["drift_alarm"].any() else None),
           "drift_injected_from": mid}
    _dump("monitor", rep)
    print(json.dumps(rep, indent=2, default=_json_default))


def cmd_calendar(args) -> None:
    from novaretail.calendar_opt import plan_summary, reallocate, schedule_without_conflicts, value_per_day

    est = pd.read_parquet(RESULTS / "estimates.parquet")
    est = est[est["status"] != "pool_too_small"]
    est["measurable"] = est["measurable"].fillna(False).astype(bool)
    values = value_per_day(est)
    plan = reallocate(values)
    plan.to_parquet(RESULTS / "calendar_plan.parquet", index=False)
    schedule_without_conflicts(plan).to_parquet(RESULTS / "calendar_schedule.parquet", index=False)
    rep = plan_summary(plan)
    _dump("calendar", rep)
    print(json.dumps(rep, indent=2, default=_json_default))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="nr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("--source", default="favorita", choices=["favorita", "synthetic"])
    s.add_argument("--synthetic-end", default="2017-08-15")
    s.add_argument("--synthetic-stores", type=int, default=10)
    sub.add_parser("campaigns")
    s = sub.add_parser("estimate")
    s.add_argument("--max", type=int, default=300)
    s.add_argument("--pool", type=int, default=12)
    s.add_argument("--param-draws", type=int, default=0)
    s.add_argument("--nsim", type=int, default=2000)
    s.add_argument("--keep", type=int, default=12, help="estimates pickled for the notebooks")
    s = sub.add_parser("backtest")
    s.add_argument("--n", type=int, default=30, help="planted campaigns per spec")
    s = sub.add_parser("quake")
    s.add_argument("--nsim", type=int, default=2000)
    s.add_argument("--param-draws", type=int, default=0)
    s = sub.add_parser("dml")
    s.add_argument("--sample", type=int, default=400_000)
    sub.add_parser("monitor")
    sub.add_parser("calendar")
    args = p.parse_args(argv)
    {"ingest": cmd_ingest, "campaigns": cmd_campaigns, "estimate": cmd_estimate, "backtest": cmd_backtest,
     "quake": cmd_quake, "dml": cmd_dml, "monitor": cmd_monitor, "calendar": cmd_calendar}[args.cmd](args)


if __name__ == "__main__":
    main()
