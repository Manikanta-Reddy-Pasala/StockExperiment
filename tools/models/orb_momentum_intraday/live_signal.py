"""Live intraday signal generator for orb_momentum_intraday.

Uses the SHARED core strategy.py (rank_momentum / orb_trade / params) — the same
code the backtest runs, so live and backtest cannot drift.

Designed to be called REPEATEDLY through the morning (e.g. an intraday cron every
5 min, 09:30–10:00). On each call it:
  1. Ranks today's top-SELECT_TOP momentum leaders (daily DB close + today's N500).
  2. Pulls today's 5-min bars so far for those leaders (Fyers).
  3. For any leader that has broken above its opening-range high BEFORE the cutoff
     and isn't already held, emits a BUY with stop (ORL) + target (ORH+2×width).
  4. After EOD_FLAT, emits SELL (force-flat) for anything still open.

This writes a signals JSON; a separate executor consumes it (places the long +
bracket order). NOTE: the intraday cron + executor are not wired yet — this is the
signal layer. Run ad-hoc:
  python -m tools.models.orb_momentum_intraday.live_signal --signals-out /tmp/orb.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine                       # noqa: E402
from tools.shared.index_membership import universe_union               # noqa: E402
from tools.models.orb_momentum_intraday import strategy as S           # noqa: E402
from tools.models.orb_momentum_intraday.data import fetch_5min, _fyers # noqa: E402

MODEL_NAME = "orb_momentum_intraday"


def _today_leaders():
    """Top-SELECT_TOP momentum leaders for today (daily DB close + live N500)."""
    eng = _get_engine()
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union(S.INDEX))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND data_source='fyers' "
            "AND date >= :a ORDER BY symbol,date"
        ), c, params={"s": syms, "a": (date.today() - timedelta(days=90)).isoformat()})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    elig = set(universe_union(S.INDEX))   # live: today's official N500 list
    return S.rank_momentum(cl, len(cl) - 1, elig)


def emit_signals(now: dt.datetime = None) -> dict:
    """Compute the current intraday signal set (buys + EOD flats)."""
    now = now or dt.datetime.now()
    cutoff_passed = (now.hour * 60 + now.minute) >= S.ENTRY_CUTOFF_MIN
    eod = (now.hour * 60 + now.minute) >= S.EOD_FLAT_MIN
    out = {"model": MODEL_NAME, "ts": now.isoformat(), "buys": [], "flats": []}
    if eod:
        out["flats"] = ["*"]   # force-flat everything still open
        return out
    if cutoff_passed:
        return out             # no new entries after the morning cutoff
    leaders = _today_leaders()
    if not leaders:
        return out
    fy = _fyers()
    today = now.date()
    for sym in leaders:
        df = fetch_5min(sym, today, today, fy=fy)
        if df is None or len(df) < S.OR_BARS + 1:
            continue
        rng = S.opening_range(df)
        if rng is None:
            continue
        orh, orl = rng
        width = orh - orl
        last = float(df["c"].iloc[-1])
        if last >= orh:   # broken out — signal a long with bracket
            out["buys"].append({
                "symbol": sym, "ref_high": round(orh, 2),
                "entry_hint": round(orh * (1 + S.SLIPPAGE), 2),
                "stop": round(orl, 2),
                "target": round(orh + S.TARGET_MULT * width, 2),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--now", default=None, help="ISO datetime override (testing)")
    a = ap.parse_args()
    now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now()
    sig = emit_signals(now)
    Path(a.signals_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.signals_out).write_text(json.dumps(sig, indent=2))
    print(f"{MODEL_NAME}: {len(sig['buys'])} buys, flats={sig['flats']} -> {a.signals_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
