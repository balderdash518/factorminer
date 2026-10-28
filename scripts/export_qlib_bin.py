#!/usr/bin/env python3
"""Export a Qlib binary daily dataset to the canonical FactorMiner CSV.

Qlib stores each field as ``features/<code>/<field>.day.bin``: little-endian
float32 values whose first element is the start offset into
``calendars/day.txt``. Rows are kept only inside the instrument file's
membership intervals and the requested date range; suspended days (no volume)
are dropped. Qlib prices are adjusted and volume is divided by the same
factor, so ``amount = close * volume`` approximates traded value; ``vwap`` is
not available, so leave ``$vwap`` and ``$amt`` out of ``data.features``.

Example::

    uv run python scripts/export_qlib_bin.py --qlib-dir ~/.qlib/qlib_data/cn_data \\
        --market all --start 2016-01-01 --end 2020-09-25 \\
        --universe ASHARE_ALL --output output/data/ashare-all-2016-2020
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")
INDEX_PREFIXES = ("SH000", "SZ399")  # index series stored next to stocks


def read_membership(path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    members: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        code, start, end = line.split("\t")[:3]
        if code.upper().startswith(INDEX_PREFIXES):
            continue
        members.setdefault(code.upper(), []).append((pd.Timestamp(start), pd.Timestamp(end)))
    return members


def read_field(path: Path, calendar: pd.DatetimeIndex) -> pd.Series:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < 2:
        return pd.Series(dtype="float64")
    start = int(raw[0])
    values = raw[1:].astype("float64")
    return pd.Series(values, index=calendar[start : start + len(values)])


def export(qlib_dir: Path, market: str, start: str, end: str, universe: str) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(
        pd.to_datetime((qlib_dir / "calendars" / "day.txt").read_text().split())
    )
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for code, intervals in sorted(read_membership(qlib_dir / "instruments" / f"{market}.txt").items()):
        folder = qlib_dir / "features" / code.lower()
        if not folder.is_dir():
            continue
        panel = pd.DataFrame({f: read_field(folder / f"{f}.day.bin", calendar) for f in FIELDS})
        inside = np.zeros(len(panel), dtype=bool)
        for a, b in intervals:
            inside |= (panel.index >= max(a, lo)) & (panel.index <= min(b, hi))
        panel = panel[inside & (panel["volume"] > 0) & panel["close"].notna()]
        if panel.empty:
            continue
        panel = panel.rename_axis("datetime").reset_index()
        panel.insert(1, "asset_id", code)
        frames.append(panel)
    data = pd.concat(frames, ignore_index=True)
    data["amount"] = data["close"] * data["volume"]
    data["universe"] = universe
    return data.sort_values(["datetime", "asset_id"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qlib-dir", type=Path, required=True)
    parser.add_argument("--market", default="all", help="instruments/<market>.txt")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--universe", required=True, help="label written to the universe column")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    qlib_dir = args.qlib_dir.expanduser()
    data = export(qlib_dir, args.market, args.start, args.end, args.universe)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "market_data.csv"
    data.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    per_day = data.groupby("datetime")["asset_id"].count()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(qlib_dir),
        "source_format": "Qlib little-endian float32 binary; first float is calendar offset",
        "market": args.market,
        "universe": args.universe,
        "rows": int(len(data)),
        "asset_count": int(data["asset_id"].nunique()),
        "assets_per_day": {"min": int(per_day.min()), "median": int(per_day.median()),
                           "max": int(per_day.max())},
        "start": str(data["datetime"].min().date()),
        "end": str(data["datetime"].max().date()),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "price_basis": "Qlib adjusted OHLC; volume divided by the same adjustment factor",
        "amount_basis": "close * volume proxy, not exchange-reported turnover; no vwap",
        "dropped": "index series (SH000*, SZ399*), suspended days (volume <= 0), rows outside membership",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "asset_count", "assets_per_day", "start", "end")}))


if __name__ == "__main__":
    main()
