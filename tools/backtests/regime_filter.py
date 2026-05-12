"""Regime filter — gates trade entries by market state.

No news. No LLM. Uses only Nifty 50 daily bars (price + ATR) since Fyers
doesn't expose India VIX historical via this API.

Regime states:
  bull      = close > 50DMA AND 50DMA > 200DMA
  neutral   = close > 200DMA but not bull
  bear      = close < 200DMA

Volatility states (ATR% of close):
  calm      = ATR% < 1.0%
  normal    = 1.0-2.0%
  volatile  = > 2.0%

Filter rule (configurable):
  allow trade if regime in {bull, neutral} AND vol in {calm, normal}
  block trade if regime == bear OR vol == volatile

Usage as library:
  from regime_filter import RegimeGate
  gate = RegimeGate.from_fyers(user_id=1, start='2025-05-12', end='2026-05-12')
  if gate.allow_entry(ts):
      ...

Usage as cli (cap-sim wrapper):
  python tools/backtests/regime_filter.py \
    --case /path/to/case_dir --capital 200000 \
    --start 2025-05-12 --end 2026-05-12
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("regime_filter")


@dataclass
class RegimeGate:
    df: pd.DataFrame   # index=date, cols=[close, ema50, ema200, atr_pct]
    bear_block: bool = True
    volatile_block: bool = True
    atr_volatile_threshold: float = 2.0

    @classmethod
    def from_fyers(cls, user_id: int, start: str, end: str,
                    symbol: str = "NSE:NIFTY50-INDEX") -> "RegimeGate":
        """Fetch Nifty 50 daily bars + compute regime indicators."""
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        # Fetch in chunks of 365 days for warmup + window (Fyers limit)
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        start_extended_dt = datetime.strptime(start, "%Y-%m-%d") - pd.Timedelta(days=380)
        all_rows = []
        cur = start_extended_dt
        while cur < end_dt:
            chunk_end = min(cur + pd.Timedelta(days=360), end_dt)
            r = svc.history(user_id=user_id, symbol=symbol, exchange="NSE",
                            interval="D",
                            start_date=cur.strftime("%Y-%m-%d"),
                            end_date=chunk_end.strftime("%Y-%m-%d"))
            if r and r.get("status") == "success":
                all_rows.extend(r["data"]["candles"])
            cur = chunk_end + pd.Timedelta(days=1)
        if not all_rows:
            raise RuntimeError(f"Fyers history fail for {symbol}")
        res = {"status": "success", "data": {"candles": all_rows}}
        if not res or res.get("status") != "success":
            raise RuntimeError(f"Fyers history fail: {res}")
        rows = res["data"]["candles"]
        df = pd.DataFrame(rows)
        df["timestamp"] = df["timestamp"].astype(int)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        df = df.sort_values("date").reset_index(drop=True)
        # EMA 50/200 (Wilder-like; pandas ewm is OK)
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        # ATR%
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        df["atr_pct"] = atr14 / df["close"] * 100
        df = df.set_index("date")
        return cls(df=df[["close", "ema50", "ema200", "atr_pct"]])

    def get_regime(self, dt) -> str:
        """Return 'bull'/'neutral'/'bear' for given date."""
        if isinstance(dt, str):
            dt = datetime.strptime(dt[:10], "%Y-%m-%d").date()
        elif hasattr(dt, "date"):
            dt = dt.date()
        # Find nearest date <= dt
        try:
            row = self.df.loc[:dt].iloc[-1]
        except (KeyError, IndexError):
            return "unknown"
        if pd.isna(row["ema50"]) or pd.isna(row["ema200"]):
            return "unknown"
        if row["close"] > row["ema50"] and row["ema50"] > row["ema200"]:
            return "bull"
        if row["close"] > row["ema200"]:
            return "neutral"
        return "bear"

    def get_volatility(self, dt) -> str:
        if isinstance(dt, str):
            dt = datetime.strptime(dt[:10], "%Y-%m-%d").date()
        elif hasattr(dt, "date"):
            dt = dt.date()
        try:
            row = self.df.loc[:dt].iloc[-1]
        except (KeyError, IndexError):
            return "unknown"
        if pd.isna(row["atr_pct"]):
            return "unknown"
        if row["atr_pct"] < 1.0: return "calm"
        if row["atr_pct"] < self.atr_volatile_threshold: return "normal"
        return "volatile"

    def allow_entry(self, dt) -> bool:
        r = self.get_regime(dt)
        v = self.get_volatility(dt)
        if self.bear_block and r == "bear": return False
        if self.volatile_block and v == "volatile": return False
        if r == "unknown" or v == "unknown": return True   # don't block on missing data
        return True


def filter_events_by_regime(case_dir: Path, gate: RegimeGate, out_dir: Path) -> int:
    """Copy per-stock .md files to out_dir, removing ENTRY rows whose
    timestamps fail the regime gate. Cleans the cycle table only — full
    P&L tables left alone since cap-sim only reads cycle table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    blocked = 0
    kept = 0
    for f in sorted(case_dir.iterdir()):
        if not f.name.endswith(".md") or f.name.startswith("_"):
            # copy summary/control files as-is
            (out_dir / f.name).write_text(f.read_text())
            continue
        lines_out = []
        in_cycles = False
        in_table = False
        for line in f.read_text().splitlines():
            s = line.strip()
            if s.startswith("## Strategy Cycles"):
                in_cycles = True
                in_table = False
            elif s.startswith("## "):
                in_cycles = False
                in_table = False
            if in_cycles and s.startswith("| Stage"):
                in_table = True
                lines_out.append(line)
                continue
            if in_cycles and in_table and s.startswith("|") and "|---|" not in s:
                cells = [c.strip() for c in s.split("|")[1:-1]]
                if len(cells) >= 2 and ("First Entry" in cells[0] or "Second Entry" in cells[0]):
                    ts = cells[1]
                    if not gate.allow_entry(ts):
                        blocked += 1
                        continue
                    kept += 1
            lines_out.append(line)
        (out_dir / f.name).write_text("\n".join(lines_out))
    log.info(f"Gate: kept {kept} entries, blocked {blocked}")
    return blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2025-05-12")
    ap.add_argument("--end", default="2026-05-12")
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--atr-threshold", type=float, default=2.0)
    ap.add_argument("--allow-bear", action="store_true",
                    help="Don't block bear-regime entries")
    ap.add_argument("--allow-volatile", action="store_true",
                    help="Don't block volatile-regime entries")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    log.info(f"Building Nifty 50 regime gate {args.start} → {args.end}")
    gate = RegimeGate.from_fyers(args.user_id, args.start, args.end)
    gate.bear_block = not args.allow_bear
    gate.volatile_block = not args.allow_volatile
    gate.atr_volatile_threshold = args.atr_threshold

    # Quick stats
    log.info("Regime distribution over window:")
    win_start = datetime.strptime(args.start, "%Y-%m-%d").date()
    win_end = datetime.strptime(args.end, "%Y-%m-%d").date()
    win = gate.df.loc[win_start:win_end]
    r_counts: Dict[str, int] = {"bull": 0, "neutral": 0, "bear": 0}
    v_counts: Dict[str, int] = {"calm": 0, "normal": 0, "volatile": 0}
    for d in win.index:
        r_counts[gate.get_regime(d)] = r_counts.get(gate.get_regime(d), 0) + 1
        v_counts[gate.get_volatility(d)] = v_counts.get(gate.get_volatility(d), 0) + 1
    log.info(f"  regime: {r_counts}")
    log.info(f"  volatility: {v_counts}")

    filter_events_by_regime(Path(args.case), gate, Path(args.out))


if __name__ == "__main__":
    main()
