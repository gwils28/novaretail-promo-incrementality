"""Dagster definitions: the chain ingestion -> features -> campaign estimation -> calendar, local mode.

Each asset wraps one CLI step and materialises the same files under data/results, so that a notebook, the CLI
and the pipeline read one source of truth. MLflow (file backend under mlruns/) logs the parameters and the
summary metrics of every estimation run, and registers the estimates table as an artifact: the model registry
of this project is a registry of *estimates with their validity checks*, not of fitted models, because every
campaign refits its own counterfactual.

    uv run dagster dev -f pipeline/definitions.py        # UI on http://localhost:3000
    uv run dagster asset materialize -f pipeline/definitions.py --select '*'
"""


import json
from argparse import Namespace
from pathlib import Path

import dagster as dg

from novaretail.data import PROCESSED, RESULTS, ROOT

MLRUNS = ROOT / "mlruns"


def _mlflow_log(run_name: str, params: dict, metrics_file: Path, artifacts: list[Path]) -> str | None:
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(f"file://{MLRUNS}")
    mlflow.set_experiment("novaretail-promo-incrementality")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        if metrics_file.exists():
            rep = json.loads(metrics_file.read_text())
            for k, v in rep.items():
                if isinstance(v, int | float) and not isinstance(v, bool):
                    mlflow.log_metric(k, float(v))
            mlflow.log_artifact(str(metrics_file))
        for a in artifacts:
            if a.exists():
                mlflow.log_artifact(str(a))
        return run.info.run_id


class EstimateConfig(dg.Config):
    source: str = "favorita"
    max_campaigns: int = 300
    pool: int = 12
    param_draws: int = 0
    nsim: int = 2000
    seed: int = 0


@dg.asset(group_name="ingestion", description="Raw Favorita files -> dense panel with structural-zero flags")
def panel(context: dg.AssetExecutionContext, config: EstimateConfig) -> dg.MaterializeResult:
    from novaretail.cli import cmd_ingest

    cmd_ingest(Namespace(source=config.source, synthetic_end="2017-08-15", synthetic_stores=10))
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    return dg.MaterializeResult(metadata={"rows": manifest["panel"]["rows_expected"],
                                          "files_sha256": dg.MetadataValue.json(
                                              {k: v["sha256"][:12] for k, v in manifest["files"].items()})})


@dg.asset(deps=[panel], group_name="features", description="Campaign windows detected from onpromotion")
def campaigns(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    from novaretail.cli import cmd_campaigns

    cmd_campaigns(Namespace())
    rep = json.loads((RESULTS / "campaigns.json").read_text())
    return dg.MaterializeResult(metadata={"runs": rep["runs"], "eligible": rep["eligible"],
                                          "by_reason": dg.MetadataValue.json(rep["by_reason"])})


@dg.asset(deps=[campaigns], group_name="causal",
          description="Per-campaign counterfactual, placebos, isolation, credibility verdict; logged to MLflow")
def estimates(context: dg.AssetExecutionContext, config: EstimateConfig) -> dg.MaterializeResult:
    from novaretail.cli import cmd_estimate

    cmd_estimate(Namespace(max=config.max_campaigns, pool=config.pool, param_draws=config.param_draws,
                           nsim=config.nsim, seed=config.seed, keep=12))
    rep = json.loads((RESULTS / "estimates.json").read_text())
    run_id = _mlflow_log("estimates", {"max_campaigns": config.max_campaigns, "pool": config.pool,
                                       "param_draws": config.param_draws, "nsim": config.nsim, "seed": config.seed},
                         RESULTS / "estimates.json", [RESULTS / "estimates.parquet", RESULTS / "donors.parquet"])
    return dg.MaterializeResult(metadata={"measurable": rep["measurable"], "campaigns_run": rep["campaigns_run"],
                                          "mlflow_run_id": run_id or "mlflow not installed"})


@dg.asset(deps=[campaigns], group_name="validation", description="Planted campaigns: coverage per estimator")
def backtest(context: dg.AssetExecutionContext, config: EstimateConfig) -> dg.MaterializeResult:
    from novaretail.cli import cmd_backtest

    cmd_backtest(Namespace(n=30, seed=config.seed))
    rep = json.loads((RESULTS / "backtest.json").read_text())
    _mlflow_log("backtest", {"n_per_spec": 30, "seed": config.seed}, RESULTS / "backtest.json",
                [RESULTS / "planted_scores.parquet"])
    return dg.MaterializeResult(metadata={"rows": rep["rows"], "seconds": rep["seconds"]})


@dg.asset(deps=[estimates, backtest], group_name="monitoring", description="Signals and reconciliation")
def monitoring(context: dg.AssetExecutionContext, config: EstimateConfig) -> dg.MaterializeResult:
    from novaretail.cli import cmd_monitor

    cmd_monitor(Namespace(seed=config.seed))
    rep = json.loads((RESULTS / "monitor.json").read_text())
    return dg.MaterializeResult(metadata={k: v for k, v in rep.items() if isinstance(v, int | float)})


@dg.asset(deps=[estimates], group_name="decision", description="Campaign-day reallocation at constant budget")
def promo_calendar(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    from novaretail.cli import cmd_calendar

    cmd_calendar(Namespace())
    rep = json.loads((RESULTS / "calendar.json").read_text())
    return dg.MaterializeResult(metadata={k: v for k, v in rep.items() if isinstance(v, int | float)})


@dg.asset_check(asset=estimates, description="The design is blocked when the never-treated false-positive rate drifts")
def placebo_guard(context) -> dg.AssetCheckResult:
    f = RESULTS / "monitor.json"
    if not f.exists():
        return dg.AssetCheckResult(passed=True, metadata={"note": "monitor not run yet"})
    rep = json.loads(f.read_text())
    fpr = rep.get("fpr_never_treated")
    return dg.AssetCheckResult(passed=fpr is None or fpr <= 0.10, metadata={"fpr_never_treated": fpr or 0.0})


weekly = dg.ScheduleDefinition(name="weekly_refresh", cron_schedule="0 6 * * 1",
                               job=dg.define_asset_job("refresh", selection="*"))

defs = dg.Definitions(assets=[panel, campaigns, estimates, backtest, monitoring, promo_calendar],
                      asset_checks=[placebo_guard], schedules=[weekly])
