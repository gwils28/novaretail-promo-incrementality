"""Ingestion of the Favorita files and construction of the store x family x day panel.

The Kaggle zip (or the synthetic files of `dgp.py`) is read from `data/raw/<source>/`. The panel is checked for
density, re-indexed on the full calendar (the files have no row on 25 December) and the structural zeros are
flagged so that the modelling code can tell "nothing sold" from "nothing could be sold":
- `before_opening`: the store had no sale at all before this date (opened later);
- `never_sold`: the store-family has no sale over the whole sample;
- `closed`: a store-wide zero on an otherwise open store (1 January, 25 December, local closures);
- `promo_unknown`: before the first date with any `onpromotion > 0` in the file (the original data had NaN
  there; the competition file fills it with zeros, which is not the same information).
Every flag is a column on the panel; nothing is imputed. A manifest (sha256, rows, date range) is written next to
the processed files as the data-versioning record of the run.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
KAGGLE_ZIP = "store-sales-time-series-forecasting.zip"
FILES = {"train": "train.csv", "stores": "stores.csv", "oil": "oil.csv", "holidays": "holidays_events.csv",
         "transactions": "transactions.csv"}


def extract_kaggle_zip(zip_path: Path = RAW / KAGGLE_ZIP, dest: Path = RAW / "favorita") -> Path:
    """Unzip the competition archive once into data/raw/favorita/."""
    if dest.exists() and all((dest / f).exists() for f in FILES.values()):
        return dest
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found: download it from the Kaggle competition page and drop it there")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def load_raw(folder: Path) -> dict[str, pd.DataFrame]:
    """Read the five tables with their dtypes fixed (dates parsed, family as category)."""
    folder = Path(folder)
    out = {k: pd.read_csv(folder / v) for k, v in FILES.items()}
    for k in ("train", "oil", "holidays", "transactions"):
        out[k]["date"] = pd.to_datetime(out[k]["date"])
    out["train"]["family"] = out["train"]["family"].astype("category")
    out["holidays"]["transferred"] = out["holidays"]["transferred"].astype(str).str.lower().eq("true")
    return out


def file_manifest(folder: Path) -> dict:
    """sha256, size and row count of each raw file: the data version of a run."""
    folder = Path(folder)
    rec = {}
    for k, v in FILES.items():
        p = folder / v
        h = hashlib.sha256()
        n_lines = 0
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                n_lines += chunk.count(b"\n")
        rec[k] = {"file": v, "sha256": h.hexdigest(), "bytes": p.stat().st_size, "rows": n_lines - 1}
    return rec


def build_panel(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dense panel on the full daily calendar, with the structural-zero flags. Returns (panel, report).

    The panel is long: date, store_nbr, family, sales, onpromotion, and the boolean flags. Rows absent from the
    file (25 December) are added with sales NaN and `closed=True`.
    """
    dates = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    stores = np.sort(train["store_nbr"].unique())
    fams = list(train["family"].cat.categories) if hasattr(train["family"], "cat") else sorted(train["family"].unique())
    full = pd.MultiIndex.from_product([dates, stores, fams], names=["date", "store_nbr", "family"])
    t = train.set_index(["date", "store_nbr", "family"])[["sales", "onpromotion"]]
    dup = t.index.duplicated().sum()
    panel = t.reindex(full)
    missing_rows = int(panel["sales"].isna().sum())
    gaps = panel.index[panel["sales"].isna()].get_level_values(0).unique()
    missing_dates = sorted({d.strftime("%m-%d") for d in gaps})
    panel = panel.reset_index()
    panel["onpromotion"] = panel["onpromotion"].fillna(0).astype(int)
    panel["family"] = panel["family"].astype("category")

    # structural zeros ---------------------------------------------------------------------------------------
    store_day = panel.groupby(["store_nbr", "date"], observed=True)["sales"].sum(min_count=1)
    opening = store_day[store_day > 0].reset_index().groupby("store_nbr")["date"].min().rename("opening")
    panel = panel.merge(opening, on="store_nbr", how="left")
    panel["before_opening"] = panel["date"] < panel["opening"]
    sf_total = panel.groupby(["store_nbr", "family"], observed=True)["sales"].sum(min_count=1)
    never = sf_total[sf_total.fillna(0) == 0].index
    key = pd.MultiIndex.from_frame(panel[["store_nbr", "family"]])
    panel["never_sold"] = key.isin(never)
    sd = panel.merge(store_day.rename("store_day_sales").reset_index(), on=["store_nbr", "date"], how="left")
    panel["closed"] = (sd["store_day_sales"].fillna(0) == 0) & ~panel["before_opening"]
    first_promo = panel.loc[panel["onpromotion"] > 0, "date"].min()
    panel["promo_unknown"] = panel["date"] < first_promo if pd.notna(first_promo) else True
    panel["structural_zero"] = panel["before_opening"] | panel["never_sold"] | panel["closed"]
    panel = panel.drop(columns="opening")

    sales = panel["sales"].fillna(0)
    report = {
        "n_dates": len(dates), "n_stores": len(stores), "n_families": len(fams), "rows_file": len(train),
        "rows_expected": len(full), "duplicated_keys": int(dup), "missing_rows_added": missing_rows,
        "missing_calendar_days": missing_dates,
        "first_promotion_date": str(first_promo.date()) if pd.notna(first_promo) else None,
        "share_zero_sales_file": float((train["sales"] == 0).mean()),
        "share_zero_structural": float(((sales == 0) & panel["structural_zero"]).sum() / max(1, (sales == 0).sum())),
        "share_zero_intermittent": float(((sales == 0) & ~panel["structural_zero"]).sum() / len(panel)),
        "stores_opened_late": {int(k): str(v.date()) for k, v in opening.items() if v > dates[0]},
        "never_sold_pairs": int(len(never)),
        "closed_store_days": int((panel["closed"] & ~panel["never_sold"]).sum() / max(1, len(fams))),
    }
    return panel, report


def to_wide(panel: pd.DataFrame, value: str = "sales", structural_as_nan: bool = True) -> pd.DataFrame:
    """date x (store_nbr, family) matrix; structural zeros become NaN so that fits ignore them."""
    v = panel[value].astype(float)
    if structural_as_nan and value == "sales":
        v = v.where(~panel["structural_zero"])
    return v.to_frame(value).assign(date=panel["date"], store_nbr=panel["store_nbr"], family=panel["family"]) \
        .pivot_table(index="date", columns=["store_nbr", "family"], values=value, observed=True, dropna=False)


def ingest(source: str = "favorita", out: Path = PROCESSED) -> dict:
    """Raw files -> panel.parquet, stores/oil/holidays parquet, manifest.json and panel_report.json."""
    folder = RAW / source
    if source == "favorita":
        folder = extract_kaggle_zip()
    raw = load_raw(folder)
    panel, report = build_panel(raw["train"])
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "panel.parquet", index=False)
    for k in ("stores", "oil", "holidays", "transactions"):
        raw[k].to_parquet(out / f"{k}.parquet", index=False)
    manifest = {"source": source, "files": file_manifest(folder), "panel": report}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
