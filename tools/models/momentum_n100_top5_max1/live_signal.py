"""Model 3 — Momentum Rotation live signal generator.

Ranks the N100 universe by 30d return, picks top-N, emits ENTRY1 /
TARGET_HIT / STOP_HIT signals consumed by tools/live/fyers_executor.py.

Strategy:
  - Universe: real NIFTY 100 (NSE constituents from ind_nifty100list.csv)
  - top_n = 5
  - max_concurrent = 1
  - rebalance: 1st of month (or first trading day on/after)

Logic per run:
  1. Load current ledger -> currently held symbol (if any)
  2. Rank universe by 30d return; pick top-N
  3. If held NOT in top-N -> emit STOP_HIT (rotation exit)
  4. Emit ENTRY1 for rank-1 stock if not already held

Usage:
  python tools/models/momentum_n100_top5_max1/live_signal.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --top-n 5 --rebalance-only \
    --signals-out /app/logs/momrot/signals/$(date +%F)_momrot_n100.json

Flags:
  --rebalance-only       only emit signals on 1st-of-month (or after weekend)
  --force                emit regardless of date
  --ledger PATH          live ledger to read current holdings (optional)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok  # noqa: E402
from tools.shared.nse_calendar import is_first_trading_day_of_month  # noqa: E402

log = logging.getLogger("momrot_signal")


def is_rebalance_day(today: datetime, last_rotation: datetime = None) -> bool:
    """True if today is the monthly rebalance trigger.

    Rule: rebalance once per month, on the FIRST NSE TRADING DAY of the month
    (holiday-aware via nse_calendar). If we already rotated this calendar
    month, skip (the dedup that stops the trigger re-firing).
    """
    if last_rotation and last_rotation.year == today.year and last_rotation.month == today.month:
        return False
    # First NSE trading day of the month (skips a 1st that is a weekend/holiday).
    return is_first_trading_day_of_month(today)


# Mid-month check: trigger an extra rank check on the first weekday on/after
# day 15 of each month. Only emits a ROTATE signal if the new rank-1 leads
# the currently-held stock's 30d return by >= MID_MONTH_LEAD_PCT. Backtested
# on 2023-26 N100 universe: +19.7pp CAGR over plain monthly (+81.4% vs
# +61.7% baseline, Calmar 1.31 → 1.75) with honest costs included.
MID_MONTH_LEAD_PCT = 5.0   # rotate mid-month only if new rank-1 leads by 5pp


def is_mid_month_check_day(today: datetime) -> bool:
    """True if today is the mid-month check trigger.

    Rule: first weekday on/after day 15 of month, but NOT also a rebalance
    day (avoids double-firing in odd calendars).
    """
    if today.day < 15 or today.day > 21:
        return False
    if today.weekday() >= 5:
        return False
    # Earliest weekday on/after 15 — anchor by walking back from today
    anchor = datetime(today.year, today.month, 15)
    while anchor.weekday() >= 5:
        anchor += timedelta(days=1)
    return today.date() == anchor.date()


def load_universe(path: str) -> List[Dict]:
    """Load the ranking universe from a JSON universe file.

    Args:
        path: Path to the universe JSON (the n100 constituents snapshot).

    Returns:
        list[dict]: The "stocks" array, each entry having at least a "symbol"
        and usually a "name".
    """
    with open(path) as f:
        return json.load(f)["stocks"]


def is_model_enabled() -> bool:
    """model_settings.enabled for this model. Fail-CLOSED on error/missing row
    so a disabled model (or a config/DB problem) can never place real orders.
    """
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == "momentum_n100_top5_max1":
                return bool(s.get("enabled"))
        return False
    except Exception as e:
        log.warning(f"enabled-flag read failed: {e} — defaulting to OFF")
        return False


def _last_rotation_date():
    """Entry date of the current open position (None if flat), used to dedup
    the monthly rebalance to at most once per calendar month (else the day-1-7
    gate re-fires every weekday in the first week).
    """
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger("momentum_n100_top5_max1")
        if l and l.get("open_entry_date"):
            return datetime.fromisoformat(l["open_entry_date"])
    except Exception as e:
        log.debug(f"last_rotation read failed: {e}")
    return None


def get_close_at(symbol: str, target_ts: int) -> float:
    """Fetch the most recent daily close at/just before a unix timestamp.

    Reads a 90-day window from the cached daily OHLCV and returns the last
    available close in that window.

    Args:
        symbol: Fyers symbol, e.g. "NSE:RELIANCE-EQ".
        target_ts: Unix epoch seconds for the "as of" date.

    Returns:
        float: Last close in the window, or 0.0 if no data is cached.
    """
    # Pull a 90-day trailing window so a missing exact day still yields the
    # last traded close (last row), not a gap.
    df = read_cached(symbol, "D", target_ts - 90 * 86400, target_ts)
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["close"])


def get_close_trading_days_ago(symbol: str, today_ts: int, n: int) -> float:
    """Close exactly `n` TRADING days before the latest cached bar.

    Matches backtest.py's `cl.iloc[di - LOOKBACK]` (trading-day index). The
    old path subtracted `n` CALENDAR days (~30% shorter window, drifting with
    weekends/holidays) — that made the live signal diverge from the backtest.

    Returns 0.0 if fewer than n+1 bars are cached (cannot form the ratio).
    """
    # Wide window so >= n+1 trading bars are present (n trading days span ~7/5
    # calendar; +90d cushions holidays/IPO gaps).
    df = read_cached(symbol, "D", today_ts - (n * 3 + 90) * 86400, today_ts)
    if len(df) < n + 1:
        return 0.0
    return float(df.iloc[-(n + 1)]["close"])


def rank_universe(stocks: List[Dict], today_ts: int,
                  lookback_days: int = 30) -> List[tuple]:
    """Rank the universe by trailing 30-day return (the live ranking step).

    Mirrors backtest.py's rank_at: for each stock, return = close_now /
    close_30d_ago - 1, then sort best-first. Stocks lacking either price are
    dropped.

    Args:
        stocks: Universe entries (each with "symbol", optional "name").
        today_ts: Unix epoch seconds for "now".
        lookback_days: Momentum lookback window (default 30, matches backtest).

    Returns:
        list[tuple]: (symbol, name, 30d_return_pct, current_price) sorted
        descending by return.
    """
    rows = []
    for s in stocks:
        sym = s["symbol"]
        c_now = get_close_at(sym, today_ts)
        # 30 TRADING days back (backtest parity), not 30 calendar days.
        c_past = get_close_trading_days_ago(sym, today_ts, lookback_days)
        if c_now > 0 and c_past > 0:  # need both endpoints to compute a return
            ret = (c_now / c_past - 1) * 100  # 30d return in percentage points
            rows.append((sym, s.get("name", sym), ret, c_now))
    rows.sort(key=lambda r: -r[2])  # best 30d return first
    return rows


def load_held(ledger_path: Path) -> List[Dict]:
    """Read currently-held positions from a file-based paper ledger.

    Used only when an explicit --ledger path is passed; otherwise the model
    reads the DB ledger via held_from_db(). Fails soft to flat (empty) on any
    read/parse error.

    Args:
        ledger_path: Path to the paper-ledger JSON, or None.

    Returns:
        list[dict]: The ledger's "open" positions, or [] if missing/unreadable.
    """
    if not ledger_path or not ledger_path.exists():
        return []
    try:
        with open(ledger_path) as f:
            return json.load(f).get("open", [])
    except Exception as e:
        log.warning(f"ledger read fail: {e}")
        return []  # treat unreadable ledger as flat rather than crashing


def held_from_db() -> List[Dict]:
    """Current open position from model_ledger (DB) as emit_signals' held list.

    This makes the model STATEFUL: it reads what it actually holds so it can
    emit a rotation SELL when the held stock drops out of the top-N. Without
    this the executor (max_concurrent=1) silently skips the new BUY and the
    position never rotates. Mirrors momentum_pseudo_n100_adv.get_current_position.
    """
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger("momentum_n100_top5_max1")
        if l and l.get("open_symbol"):
            # Shape matches load_held() output so emit_signals treats file and
            # DB ledgers identically: one held position with its entry price.
            return [{
                "symbol": l["open_symbol"],
                "entry_price": float(l.get("open_entry_px") or 0),
            }]
    except Exception as e:
        log.warning(f"DB ledger read failed: {e}; treating as flat")
    return []  # no open position (or read failed) -> model runs as flat


def emit_signals(top_picks: List[tuple], held: List[Dict],
                  top_n: int, retain_top_n: int = 1) -> List[Dict]:
    """Turn a ranking + current holding into SELL/BUY signal dicts.

    Delegates the keep/rotate decision to the shared decide_rotation core so
    live and backtest stay in lockstep, then materialises the resulting sell
    and/or buy into the signal dict shape that fyers_executor consumes.

    Args:
        top_picks: Output of rank_universe — (symbol, name, ret_pct, price)
            best-first.
        held: Current holding list (0 or 1 entry) from the ledger/DB.
        top_n: Display ranking size (informational; not the exit band).
        retain_top_n: Exit retention band passed to decide_rotation; 1 == top-1
            rotation (matches the canonical backtest).

    Returns:
        list[dict]: 0–2 signal dicts. A SELL (rotation exit) is emitted when the
        held stock drops out of the retention band; a BUY (ENTRY1) for the new
        rank-1 when it differs from what's held.
    """
    # Decision via the SHARED rotation core — same rule backtest.py uses, so
    # live and backtest cannot drift. retain_top_n=1 == top-1 rotation.
    ranked = [p[0] for p in top_picks]          # just the symbols, rank order
    by_sym = {p[0]: p for p in top_picks}       # symbol -> full pick tuple
    held_sym = held[0]["symbol"] if held else None
    held_entry = float(held[0].get("entry_price", 0)) if held else 0.0
    dec = decide_rotation(held_sym, ranked, retain_top_n)

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals = []

    if dec.sell:
        price = get_close_at(dec.sell, int(datetime.now().timestamp()))
        # Label the exit by P&L vs entry: at/above entry = TARGET_HIT, below =
        # STOP_HIT. Both are rotation exits (no real SL/target in this model).
        kind = "TARGET_HIT" if price >= held_entry else "STOP_HIT"
        # SELL signal dict shape consumed by fyers_executor (side=SELL).
        signals.append({
            "model": "momentum_rotation",
            "universe": "n100_real",
            "symbol": dec.sell,
            "company": dec.sell,
            "ts": today_str,
            "side": "SELL",
            "signal": kind,
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"rotation exit (dropped out of top-{retain_top_n})",
        })

    if dec.buy:
        sym, name, ret, price = by_sym[dec.buy]
        # BUY signal dict shape consumed by fyers_executor (side=BUY, ENTRY1).
        signals.append({
            "model": "momentum_rotation",
            "universe": "n100_real",
            "symbol": sym,
            "company": name,
            "ts": today_str,
            "side": "BUY",
            "signal": "ENTRY1",
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"30d momentum rank-1 ({ret:+.2f}%)",
        })

    return signals


def main():
    """CLI entry point — gate on the date, rank, emit signals, persist + notify.

    Flow:
      1. Apply the date gate (mid-month check OR rebalance-only, unless --force).
         If gated out, write an empty signals file (+ skip notification) and
         return early.
      2. Load the universe and the current holding (file ledger if --ledger,
         else the DB ledger — this is what makes the model stateful).
      3. Rank by 30d return and emit SELL/BUY signals via emit_signals.
      4. Apply the mid-month lead-threshold filter when in mid-month mode.
      5. Write the signals JSON, persist the top-N ranking for the UI, and fire
         the audit + notification hooks (scheduled runs only).

    Returns:
        int: 0 on success (also the early-return exit code), suitable for
        SystemExit.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-file", required=True)
    ap.add_argument("--top-n", type=int, default=5,
                    help="Ranking display size only.")
    ap.add_argument("--retain-top-n", type=int, default=1,
                    help="Exit retention band: hold while in top-N by 30d ret, "
                         "rotate when out. 1=top-1 rotation (matches backtest). "
                         "Default 1.")
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--ledger", default=None,
                    help="Paper ledger JSON to read current holdings")
    ap.add_argument("--rebalance-only", action="store_true",
                    help="Skip if today is not rebalance trigger day")
    ap.add_argument("--mid-month-check", action="store_true",
                    help="Day-15 check: emit ROTATE only if rank-1 leads "
                         "current held by >= MID_MONTH_LEAD_PCT (default 5pp)")
    ap.add_argument("--force", action="store_true",
                    help="Bypass rebalance-day check (initial deploy / manual)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    log.info(f"momentum_rotation_signal run: today={today.date()} "
             f"weekday={today.strftime('%A')} day_of_month={today.day}")

    # Enabled gate (fail-closed) — a model toggled OFF must not emit/trade.
    if not args.force and not is_model_enabled():
        log.warning("momentum_n100_top5_max1: model_settings.enabled is False "
                    "— writing empty signals file and exiting.")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.signals_out, "w") as f:
            json.dump([], f)
        return 0

    # Mid-month gate (mutually exclusive with rebalance-only; --force still
    # overrides). Only emit on day-15 weekday AND lead >= threshold.
    if args.mid_month_check and not args.force:
        if not is_mid_month_check_day(today):
            log.info("Not mid-month check day (need day-15-weekday). Skipping.")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.signals_out, "w") as f:
                json.dump([], f)
            return 0

    if args.rebalance_only and not args.force and not args.mid_month_check:
        if not is_rebalance_day(today, _last_rotation_date()):
            log.info(f"Not rebalance day (need first NSE trading day of month "
                     f"+ not already rotated this month). Skipping.")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.signals_out, "w") as f:
                json.dump([], f)
            try:
                from src.services.notification_service import notify_skip
                notify_skip("momentum_n100_top5_max1", "not rebalance day")
            except Exception as _ne:
                log.debug(f"notify_skip failed: {_ne}")
            return 0

    stocks = load_universe(args.universe_file)
    log.info(f"Universe: {len(stocks)} symbols from {args.universe_file}")

    # Stateful: prefer file ledger if explicitly passed, else read DB ledger
    # (held_from_db) so a rotation SELL can fire when the held stock falls out.
    held = load_held(Path(args.ledger)) if args.ledger else held_from_db()
    log.info(f"Currently held: {[h['symbol'] for h in held]}")

    today_ts = int(today.timestamp())
    # 30d-return ranking of the whole universe, best-first.
    ranks = rank_universe(stocks, today_ts)
    log.info(f"Ranked {len(ranks)} stocks. Top-{args.top_n}:")
    for i, (sym, name, ret, price) in enumerate(ranks[:args.top_n], 1):
        log.info(f"  {i}. {sym:<14} {ret:+7.2f}%  @ ₹{price:.2f}")

    signals = emit_signals(ranks, held, args.top_n, retain_top_n=args.retain_top_n)

    # Mid-month lead-threshold filter:
    # If today is the mid-month check day, suppress rotation unless the
    # new rank-1 leads the currently-held stock's 30d return by at least
    # MID_MONTH_LEAD_PCT. Keeps trade count low while still catching
    # genuine new winners that broke out mid-cycle.
    if args.mid_month_check and held and signals:
        held_sym = held[0]["symbol"]
        ranked_ret = [(r[0], r[2]) for r in ranks]
        if midmonth_lead_ok(held_sym, ranked_ret, MID_MONTH_LEAD_PCT):
            log.info(f"mid-month: ROTATE — rank-1 leads held {held_sym} "
                     f"by >= {MID_MONTH_LEAD_PCT}pp (or held dropped from ranking).")
        else:
            log.info(f"mid-month: no rotation for held {held_sym} "
                     f"(already rank-1, or lead < {MID_MONTH_LEAD_PCT}pp). Suppressing.")
            signals = []

    log.info(f"Emitting {len(signals)} signals")

    # Canonical signals file the cron executor (fyers_executor) reads & acts on.
    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.signals_out, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {args.signals_out}")

    # Persist top-N ranking for the Today's Picks UI. Always written (even
    # on non-rebalance days the user wants to *see* the current ranking).
    ranking_dir = Path("/app/logs/momrot/ranking")
    ranking_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = ranking_dir / f"{today.strftime('%Y-%m-%d')}.json"
    top_payload = {
        "model": "momentum_n100_top5_max1",
        "date": today.strftime("%Y-%m-%d"),
        "universe_size": len(stocks),
        "top_n": [
            {
                "rank": i + 1,
                "symbol": sym,
                "name": name,
                "ret_30d_pct": round(ret, 2),
                "price": round(price, 2),
            }
            for i, (sym, name, ret, price) in enumerate(ranks[:5])
        ],
    }
    ranking_path.write_text(json.dumps(top_payload, indent=2, default=str))
    log.info(f"Wrote ranking -> {ranking_path}")

    # Audit hook — record the ranking snapshot and each signal (or a HOLD).
    try:
        from src.services.audit_service import write_rankings, write_signal
        write_rankings("momentum_n100_top5_max1", today.date(),
                       top_payload.get("universe_size") or 0,
                       0, top_payload.get("top_n") or [])
        # Audit signals ONLY for scheduled (cron) runs, not manual --force.
        if not args.force:
            if signals:
                for _sig in signals:
                    write_signal("momentum_n100_top5_max1", today.date(),
                                 _sig.get("signal", ""), _sig.get("symbol", ""),
                                 _sig.get("side", ""), price=_sig.get("price"),
                                 reason=(_sig.get("note") or "")[:120])
            else:
                # No signal this run -> log an explicit HOLD row for the audit.
                write_signal("momentum_n100_top5_max1", today.date(), "HOLD", "", "NONE",
                             reason="no signal emitted")
    except Exception as _e:
        log.debug(f"audit hook failed: {_e}")

    # Notification funnel — ping the verdict even on no-change (eval day only).
    # n100's live_signal runs stateless (held=[]) so it always emits ENTRY1;
    # the ledger is the truth. If already holding, the ENTRY won't execute
    # (max_concurrent=1) → that's a no-change tick.
    if not args.force:
        try:
            from src.services.notification_service import (
                notify_model_decision, current_held,
            )
            _held = current_held("momentum_n100_top5_max1")
            # If already holding, max_concurrent=1 means the BUY won't execute,
            # so the effective signal set is empty (a no-change tick).
            _eff = [] if _held else signals
            # 30d return of the held stock for the notification context.
            _ret = next((r[2] for r in ranks if r[0] == _held), None) if _held else None
            notify_model_decision(
                "momentum_n100_top5_max1", _eff, held_symbol=_held,
                held_ret=_ret,
                trigger="MID_MONTH" if args.mid_month_check else "CRON",
            )
        except Exception as _ne:
            log.debug(f"notify decision failed: {_ne}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
