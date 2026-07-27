import importlib.util
from pathlib import Path

import pytest

dagster = pytest.importorskip("dagster")


def test_definitions_load_and_order():
    spec = importlib.util.spec_from_file_location("defs", Path(__file__).parents[1] / "pipeline" / "definitions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    graph = mod.defs.resolve_asset_graph()
    keys = {k.to_user_string() for k in graph.get_all_asset_keys()}
    assert {"panel", "campaigns", "estimates", "backtest", "monitoring", "promo_calendar"} <= keys
    assert dagster.AssetKey("campaigns") in mod.estimates.dependency_keys


def test_mlflow_log_on_sqlite(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    spec = importlib.util.spec_from_file_location("defs", Path(__file__).parents[1] / "pipeline" / "definitions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(mod, "MLRUNS", tmp_path / "mlruns")
    rep = tmp_path / "rep.json"
    rep.write_text('{"fpr_never_treated": 0.17, "flag": true, "note": "x"}')
    run_id = mod._mlflow_log("t", {"seed": 0}, rep, [])
    import mlflow

    run = mlflow.get_run(run_id)
    assert run.data.metrics == {"fpr_never_treated": 0.17}
    assert run.data.params == {"seed": "0"}
