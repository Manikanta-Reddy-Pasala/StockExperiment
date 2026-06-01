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


def _today_leaders(today: date = None):
    """Top-SELECT_TOP momentum leaders for today (daily DB close + live N500).

    Returns [] if the daily panel is STALE (last close older than
    DAILY_STALE_MAX_DAYS — failed nightly pull / dead feed) so ORB never ranks
    leaders off days-old data.
    """
    today = today or date.today()
    eng = _get_engine()
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union(S.INDEX))]
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND data_source='fyers' "
            "AND date >= :a ORDER BY symbol,date"
        ), c, params={"s": syms, "a": (today - timedelta(days=90)).isoformat()})
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    # Daily-panel freshness gate (parity with the other models' stale gates).
    last_day = cl.index[-1].date()
    if (today - last_day).days > S.DAILY_STALE_MAX_DAYS:
        print(f"orb: daily panel STALE — last close {last_day} > "
              f"{S.DAILY_STALE_MAX_DAYS}d before {today}; no leaders.")
        return []
    elig = set(universe_union(S.INDEX))   # live: today's official N500 list
    return S.rank_momentum(cl, len(cl) - 1, elig)


def _held_symbols() -> set:
    """Symbols ORB currently holds (multi-holding ledger), Fyers form."""
    try:
        from src.services.trading.multi_holding_service import get_holdings
        return {h["symbol"] for h in get_holdings(MODEL_NAME) if h.get("symbol")}
    except Exception:
        return set()


def _invested() -> float:
    """orb invested_amount (capital cap) from model_settings."""
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        row = next((s for s in get_all_settings() if s["model_name"] == MODEL_NAME), None)
        return float(row.get("invested_amount") or 0) if row else 0.0
    except Exception:
        return 0.0


def emit_signals(now: dt.datetime = None) -> dict:
    """Current intraday signal set in the MULTI-executor schema:
    {model, ts, sells:[{symbol,reason}], buys:[{symbol,qty,...}]}.

    - At/after EOD_FLAT (15:10): emit SELLS for EVERY currently-held name
      (force square-off; reason EOD_FLAT). No buys.
    - Before the entry cutoff: emit a BUY for each top leader that has broken
      above its opening-range high and is NOT already held, sized to one slot
      (invested/SELECT_TOP). After the cutoff: no new buys.
    The executor (fyers_executor_multi) places these; stop/target are carried as
    hints for a future bracket-order extension (the executor ignores them today).
    """
    now = now or dt.datetime.now()
    mins = now.hour * 60 + now.minute
    out = {"model": MODEL_NAME, "ts": now.isoformat(), "sells": [], "buys": []}

    # ---- EOD square-off: sell everything still open ----
    if mins >= S.EOD_FLAT_MIN:
        for sym in sorted(_held_symbols()):
            out["sells"].append({"symbol": sym, "reason": "EOD_FLAT"})
        return out

    if mins >= S.ENTRY_CUTOFF_MIN:
        return out             # no new entries after the morning cutoff

    today = now.date()
    leaders = _today_leaders(today)
    if not leaders:
        return out
    held = _held_symbols()
    invested = _invested()
    fy = _fyers()
    for sym in leaders:
        if sym in held:
            continue           # already in this name — don't re-buy
        df = fetch_5min(sym, today, today, fy=fy)
        if df is None or len(df) < S.OR_BARS + 1:
            continue
        # Intraday FRESHNESS gate — refuse to act on stale/wrong-day bars:
        #  (1) the latest bar must be TODAY (not a cached/old session), and
        #  (2) it must be recent (live feed alive), else a halted/dead feed
        #      could read an old close as a "breakout". A missed entry is safe.
        if df["day"].iloc[-1] != today:
            continue
        try:
            last_bar_age_min = (now - df["dt"].iloc[-1].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60.0
        except Exception:
            last_bar_age_min = 0.0
        if last_bar_age_min > S.STALE_BAR_MAX_MIN:
            print(f"orb: {sym} last 5-min bar {last_bar_age_min:.0f}m old "
                  f"(> {S.STALE_BAR_MAX_MIN}m) — stale feed, skip.")
            continue
        rng = S.opening_range(df)
        if rng is None:
            continue
        orh, orl = rng
        width = orh - orl
        last = float(df["c"].iloc[-1])
        if last >= orh:   # broken out — signal a long, one slot of capital
            entry_hint = round(orh * (1 + S.SLIPPAGE), 2)
            qty = S.slot_qty(invested, S.SELECT_TOP, entry_hint)
            if qty < 1:
                continue
            out["buys"].append({
                "symbol": sym, "qty": qty, "ref_high": round(orh, 2),
                "entry_hint": entry_hint, "stop": round(orl, 2),
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
    print(f"{MODEL_NAME}: {len(sig['buys'])} buys, {len(sig['sells'])} sells "
          f"-> {a.signals_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
