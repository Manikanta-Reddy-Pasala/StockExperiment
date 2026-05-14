"""Model A — Mean Reversion (RSI 14).

Strategy:
  * Universe: N100 (loaded from logs/momrot/universes/n100_current.json) or fallback N50.
  * Daily scan: for each non-held symbol, compute 14-day RSI on close.
    Enter at next-day open (we approximate with same-day close) when RSI < 30.
  * Exit when RSI > 70 OR max-hold = 30 calendar days.
  * Max 5 concurrent positions. Equal-weight slot sizing (cash / slots-left).

Outputs cap-sim-compatible per-symbol .md files compatible with realistic_capital_sim.py.

Usage:
  python tools/backtests/model_mean_reversion.py \
    --from 2023-05-13 --to 2024-05-12 \
    --out /app/exports/backtests/model_a_2023_2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.backtests.run_ema_200_400_backtest import NIFTY50_SYMBOLS  # noqa: E402

log = logging.getLogger("model_a")


def load_n100() -> List[str]:
    path = ROOT / "logs" / "momrot" / "universes" / "n100_current.json"
    if path.exists():
        data = json.loads(path.read_text())
        return [s["symbol"] for s in data["stocks"]]
    return [s for s, _ in NIFTY50_SYMBOLS]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def run(symbols: List[str], start: str, end: str, max_pos: int,
        out_dir: Path) -> None:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=90)

    log.info(f"Loading bars for {len(symbols)} symbols")
    daily: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            log.info(f"  {i}/{len(symbols)}")
        df = read_cached(sym, "D",
                         int(warmup_dt.timestamp()),
                         int(end_dt.timestamp()))
        if df.empty:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["rsi"] = compute_rsi(df["close"])
        daily[sym] = df

    # All trading dates seen (intersection of bars, sorted union)
    all_dates = set()
    for df in daily.values():
        all_dates.update(df["timestamp"].tolist())
    trade_ts = sorted(t for t in all_dates
                      if start_dt.timestamp() <= t <= end_dt.timestamp())

    per_symbol: Dict[str, List[Dict]] = {s: [] for s in symbols}
    held: Dict[str, Dict] = {}  # symbol -> {entry_price, entry_ts}

    for ts in trade_ts:
        dt = datetime.fromtimestamp(ts)
        # Exits first
        to_close = []
        for sym, pos in held.items():
            df = daily[sym]
            row = df[df["timestamp"] == ts]
            if row.empty:
                continue
            close = float(row.iloc[0]["close"])
            rsi = float(row.iloc[0]["rsi"])
            age_days = (dt - pos["entry_dt"]).days
            reason = None
            if rsi > 70:
                reason = "rsi70"
            elif age_days >= 30:
                reason = "time_stop"
            if reason:
                kind = "TARGET" if close >= pos["entry_price"] else "STOP"
                stage = "Target hit" if kind == "TARGET" else "Stop hit"
                per_symbol[sym].append({
                    "Stage": stage,
                    "ts": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": close,
                })
                to_close.append(sym)
        for sym in to_close:
            del held[sym]

        # Entries
        if len(held) < max_pos:
            slots = max_pos - len(held)
            # Find candidates with RSI < 30 today, not held
            cands = []
            for sym, df in daily.items():
                if sym in held:
                    continue
                row = df[df["timestamp"] == ts]
                if row.empty:
                    continue
                rsi = float(row.iloc[0]["rsi"])
                close = float(row.iloc[0]["close"])
                if rsi < 30 and close > 0:
                    cands.append((sym, rsi, close))
            # Most-oversold first
            cands.sort(key=lambda x: x[1])
            for sym, _rsi, close in cands[:slots]:
                held[sym] = {"entry_price": close, "entry_dt": dt}
                per_symbol[sym].append({
                    "Stage": "First Entry",
                    "ts": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": close,
                })

    # Close any remaining at end_dt
    end_ts = int(end_dt.timestamp())
    for sym, pos in held.items():
        df = daily[sym]
        # latest bar <= end_ts
        valid = df[df["timestamp"] <= end_ts]
        if valid.empty:
            continue
        close = float(valid.iloc[-1]["close"])
        kind = "TARGET" if close >= pos["entry_price"] else "STOP"
        stage = "Target hit" if kind == "TARGET" else "Stop hit"
        per_symbol[sym].append({
            "Stage": stage,
            "ts": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "price": close,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sym, events in per_symbol.items():
        if not events:
            continue
        short = sym.split(":")[-1].replace("-EQ", "").lower()
        lines = [
            f"# {sym}", "",
            "## Backtest Summary",
            f"- Strategy: Model A Mean Reversion (RSI14<30 → RSI>70/30d)",
            f"- Entries: {sum(1 for e in events if 'Entry' in e['Stage'])}",
            "",
            "## Strategy Cycles", "",
            "| Stage | Timestamp | Price | sl | target | note |",
            "|---|---|---|---|---|---|",
        ]
        for e in events:
            lines.append(f"| {e['Stage']} | {e['ts']} | {e['price']:.2f} | - | - | - |")
        (out_dir / f"{short}.md").write_text("\n".join(lines))
        written += 1
    log.info(f"Wrote {written} per-symbol files to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    syms = load_n100()
    log.info(f"Universe N100 size: {len(syms)}")
    run(syms, args.date_from, args.date_to, args.max, Path(args.out))


if __name__ == "__main__":
    main()
