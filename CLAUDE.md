# CLAUDE.md

Applied project (22 June to 27 July 2026): causal incrementality of promotions on the Favorita panel, following the
tech watch `gwils28/time_series_causal_statistics` (package `causal_ts`, git dependency pinned in `pyproject.toml`).
Brief and framing decisions: `plan.md` (French). READMEs, notebooks and figure labels in **French**; code,
docstrings and commit messages in **English**; no emojis, no em dashes.
Git identity: the user only (noreply email in the local config); never add Co-Authored-By trailers.

## Commands

```bash
uv run pytest -m "not slow and not data"   # unit tests; `data` needs data/raw Favorita files, `slow` runs studies
uv run ruff check src tests tools pipeline  # line length 120
uv run nr --help                            # CLI: ingest, panel, campaigns, estimate, backtest, quake, dml, monitor
```

## Constraints that are easy to break

- `data/raw/` holds the Kaggle zip dropped by the user; never modify or regenerate it. `data/processed/` and
  `data/results/` are rebuilt by the CLI and gitignored except `*.json` summaries.
- **Numbers in READMEs come from CLI output** written to `data/results/*.json`. After touching code that feeds
  them, re-run and update by hand.
- Notebooks are built from percent-format sources in `notebooks/src/` with `tools/nbbuild.py`, committed executed, import the package;
  no logic in notebooks. Plot style: `novaretail.viz.setup()` (wraps `causal_ts.viz`), max 3 hues, French labels.
- Do not reimplement what `causal_ts` provides (counterfactual, placebos, isolation, Chow, baselines); extend it.
- Treatment is `onpromotion` intensity, not a flag; campaigns are detected windows (`campaigns.py`) and the
  protocol thresholds in `plan.md` section 4 were fixed before any result: do not tune them to the results.
- Favorita has no price column: never call the DML output a price elasticity.
- Notebook sources (percent format) live in `notebooks/src/` and are committed next to the executed notebooks;
  `make notebooks` re-executes, `uv run python tools/nbbuild.py SRC OUT` rebuilds one from its source.
