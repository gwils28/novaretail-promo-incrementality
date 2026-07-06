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
