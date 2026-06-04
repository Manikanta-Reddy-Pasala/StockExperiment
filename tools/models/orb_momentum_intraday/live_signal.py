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
from tools.shared.index_membership import universe_union, eligible_at  # noqa: E402
from tools.models.orb_momentum_intraday import strategy as S           # noqa: E402
from tools.models.orb_momentum_intraday.data import fetch_5min, _fyers # noqa: E402

MODEL_NAME = "orb_momentum_intraday"


def _orb_alert_missing(detail: str):
    """Telegram alert (deduped) that ORB has missing/stale data this run."""
    try:
        from tools.live.telegram_notify import alert_data_missing
        alert_data_missing(MODEL_NAME, detail)
    except Exception:
        pass


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
        _orb_alert_missing("No N500 daily price data returned from the DB.")
        return []
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    # Daily-panel freshness gate (parity with the other models' stale gates).
    last_day = cl.index[-1].date()
    if (today - last_day).days > S.DAILY_STALE_MAX_DAYS:
        print(f"orb: daily panel STALE — last close {last_day} > "
              f"{S.DAILY_STALE_MAX_DAYS}d before {today}; no leaders.")
        _orb_alert_missing(f"Daily ranking data STALE — last close {last_day} (>{S.DAILY_STALE_MAX_DAYS}d old).")
        return []
    # POINT-IN-TIME eligibility (NOT universe_union). universe_union is the
    # survivorship-biased UNION of every name that was EVER in the index — it
    # let non-current members rank top-3 and trade live (SPARC bought 2026-06-02
    # though it's not a current N500 member), which the backtest would NEVER do
    # and which mismatched the displayed watchlist. backtest.py and the admin
    # watchlist both rank over eligible_at(INDEX, date) — the live path must too.
    elig = set(eligible_at(S.INDEX, last_day))
    return S.rank_momentum(cl, len(cl) - 1, elig)


def _held_symbols() -> set:
    """Symbols ORB currently holds (multi-holding ledger), Fyers form."""
    try:
        from src.services.trading.multi_holding_service import get_holdings
        return {h["symbol"] for h in get_holdings(MODEL_NAME) if h.get("symbol")}
    except Exception:
        return set()


def _entered_today(today: date) -> bool:
    """True if ORB already opened a position (a BUY fill) today. ORB takes ONE
    all-in trade per day with no re-entry after the exit — matching the backtest
    — so once today's BUY is logged, later scans must not re-enter."""
    try:
        from src.models.database import get_database_manager
        from src.models.model_ledger_models import ModelTrade
        db = get_database_manager()
        with db.get_session() as s:
            row = (s.query(ModelTrade)
                   .filter(ModelTrade.model_name == MODEL_NAME, ModelTrade.side == "BUY")
                   .order_by(ModelTrade.trade_at.desc()).first())
            return bool(row and row.trade_at and row.trade_at.date() == today)
    except Exception:
        return False


def _invested() -> float:
    """orb invested_amount (capital cap) from model_settings."""
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        row = next((s for s in get_all_settings() if s["model_name"] == MODEL_NAME), None)
        return float(row.get("invested_amount") or 0) if row else 0.0
    except Exception:
        return 0.0


def _fresh_today_bars(sym: str, today: date, now: dt.datetime, fy):
    """Today's 5-min bars for `sym`, or None if missing/stale/wrong-day.

    Shared freshness gate for BOTH the entry breakout test and the intraday
    stop/target exit test so they act on the same data quality bar.
    """
    df = fetch_5min(sym, today, today, fy=fy)
    if df is None or len(df) < S.OR_BARS + 1:
        return None
    if df["day"].iloc[-1] != today:          # not today's session (cached/old)
        return None
    try:
        age = (now - df["dt"].iloc[-1].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60.0
    except Exception:
        age = 0.0
    if age > S.STALE_BAR_MAX_MIN:             # halted/dead feed — don't act
        print(f"orb: {sym} last 5-min bar {age:.0f}m old (> {S.STALE_BAR_MAX_MIN}m) — stale, skip.")
        return None
    return df


def emit_signals(now: dt.datetime = None) -> dict:
    """Current intraday signal set in the MULTI-executor schema:
    {model, ts, sells:[{symbol,reason}], buys:[{symbol,qty,...}]}.

    Mirrors the backtest (orb_trade) end-to-end so live == backtest:
    - At/after EOD_FLAT (15:15): emit SELLS for EVERY currently-held name
      (force square-off; reason EOD_FLAT). No buys.
    - BEFORE EOD, every scan: for each held name, emit a SELL if it has hit its
      STOP (ORL) or TARGET (ORH+2×width) — strategy.live_exit_reason, the SAME
      stop/target rule and priority as orb_trade. (orb_trade exits 41% of trades
      via stop/target; without this live would ride everything to EOD and NOT
      reproduce the backtest. The cron scans every 5 min all session so these
      fire intraday.)
    - Before the entry cutoff (10:00): emit a BUY for each top leader that has
      broken above its opening-range high and is NOT already held, sized to one
      slot (invested/SELECT_TOP). After the cutoff: no new buys.
    """
    now = now or dt.datetime.now()
    mins = now.hour * 60 + now.minute
    out = {"model": MODEL_NAME, "ts": now.isoformat(), "sells": [], "buys": []}

    # ---- EOD square-off: sell everything still open ----
    if mins >= S.EOD_FLAT_MIN:
        for sym in sorted(_held_symbols()):
            out["sells"].append({"symbol": sym, "reason": "EOD_FLAT"})
        return out

    today = now.date()
    fy = _fyers()
    held = _held_symbols()

    # ---- intraday STOP/TARGET exits (backtest parity; runs every scan pre-EOD) ----
    sold = set()
    for sym in sorted(held):
        df = _fresh_today_bars(sym, today, now, fy)
        if df is None:
            continue           # can't verify -> hold; the 15:15 EOD flatten is the backstop
        reason = S.live_exit_reason(df, mins)
        if reason in ("STOP", "TARGET"):
            out["sells"].append({"symbol": sym, "reason": reason})
            sold.add(sym)

    # ---- entry: morning only, SINGLE ALL-IN position ----
    if mins >= S.ENTRY_CUTOFF_MIN:
        return out             # no new entries after the morning cutoff
    if held or sold or _entered_today(today):
        # already in the day's position, just sold it, or already traded today —
        # ORB takes ONE all-in trade per day (no re-entry), matching the backtest.
        return out
    leaders = _today_leaders(today)
    if not leaders:
        return out
    invested = _invested()
    leaders_bars, _missing_intraday = [], []
    for sym in leaders:
        df = _fresh_today_bars(sym, today, now, fy)   # min_post_bars handled in pick_leader
        leaders_bars.append(df)
        if df is None and mins >= (9 * 60 + 35):
            _missing_intraday.append(sym.replace("NSE:", "").replace("-EQ", ""))
    # pick_leader = the SAME choice the backtest makes: earliest breakout (rank
    # tiebreak) among the watched leaders. Commit the FULL capital to that one.
    ci = S.pick_leader(leaders_bars)
    if ci is not None:
        chosen, df = leaders[ci], leaders_bars[ci]
        rng = S.opening_range(df, min_post_bars=1)
        if rng is not None:
            orh, orl = rng
            width = orh - orl
            entry_hint = round(orh * (1 + S.SLIPPAGE), 2)
            qty = S.full_qty(invested, entry_hint)     # ALL-IN, full capital
            if qty >= 1:
                out["buys"].append({
                    "symbol": chosen, "qty": qty, "ref_high": round(orh, 2),
                    "entry_hint": entry_hint, "stop": round(orl, 2),
                    "target": round(orh + S.TARGET_MULT * width, 2),
                })
    if _missing_intraday:
        _orb_alert_missing("No intraday 5-min bars for: " + ", ".join(_missing_intraday)
                           + " — Fyers intraday feed may be down.")
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
    # Telegram/PWA verdict ping (ORB previously NEVER pinged a signal). Only
    # ping when there's an actual action — ORB scans ~6x a morning and is FLAT
    # most of them; a daily "no change" ping would be pure noise for an
    # intraday scanner. Deduped per (model, verdict, day); MOMROT_TG_NOTIFY=1.
    if sig["buys"] or sig["sells"]:
        try:
            from src.services.notification_service import notify_model_decision
            _dec = (
                [{"signal": "EXIT", "symbol": s.get("symbol", ""), "side": "SELL"}
                 for s in sig["sells"]]
                + [{"signal": "ORB_BREAKOUT", "symbol": b.get("symbol", ""),
                    "side": "BUY", "price": b.get("entry_hint")}
                   for b in sig["buys"]]
            )
            notify_model_decision(MODEL_NAME, _dec, trigger="ORB_SCAN")
        except Exception as _ne:
            print(f"notify decision failed: {_ne}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
