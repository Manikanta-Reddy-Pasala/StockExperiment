"""momentum_pseudo_n100_adv — monthly live signal generator.

Ranks the pseudo-N100 universe (yearly-PIT top-100 by 20d ADV from N500)
by 30-day return, picks top-5, emits ENTRY1 / TARGET_HIT / STOP_HIT.

Strategy:
  - Universe: yearly-PIT pseudo-N100 (read from yearly_universes.json) —
    rebuilt at year-start using current data at that time (PIT-safe)
  - top_n = 5 (display); single position, rank-1 rotation (top-RETAIN=1)
  - rebalance: 1st trading day of month (monthly). Mid-month day-15 check exists
    as an opt-in --mid-month-check path but defaults OFF (2026-05-31: the
    mid-month config lost to RET1/monthly on the fixed-anchor re-check).
  - Filter: skip stocks with price > MAX_PRICE (₹3000) — share-count floor
    heuristic so 1 share ≤ 10% of ₹30K live capital

Usage:
  python tools/models/momentum_pseudo_n100_adv/live_signal.py \
    --universes-file tools/models/momentum_pseudo_n100_adv/yearly_universes.json \
    --top-n 5 --rebalance-only \
    --signals-out /app/logs/momrot_pseudo/signals/$(date +%F)_pseudo_n100.json

Model flow / where this file sits:
  data_pull.py    -> daily N500 OHLCV + yearly_universes.json rebuild
  build_universe.py -> produces the PIT top-100-by-ADV snapshot
  live_signal.py  -> (THIS FILE) the production decision step. Reads the PIT
                     universe, ranks it (same filters as backtest.py), reads
                     the model's open position from the DB
                     (model_ledger_service), and emits SELL / ENTRY1 signals
                     to a JSON file that fyers_executor.py later acts on.
  cron.py         -> schedules emit (09:25, rebalance-gated) + execute (09:30)
  backtest.py     -> offline validation of this exact selection logic

The actual hold/rotate decision is delegated to the SHARED rotation core
(tools/shared/rotation_strategy.decide_rotation) — identical to backtest.py —
so the live path and the backtest can never diverge.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok  # noqa: E402
from tools.shared.nse_calendar import is_first_trading_day_of_month  # noqa: E402
from tools.models.momentum_pseudo_n100_adv import strategy as S  # noqa: E402
from tools.models.momentum_pseudo_n100_adv.strategy import (  # noqa: E402  shared w/ backtest
    LOOKBACK, MAX_PRICE, SMA_LONG, RETAIN, MIDMONTH_LEAD)

log = logging.getLogger("momrot_pseudo_signal")

MODEL_NAME = "momentum_pseudo_n100_adv"
SMALLCAP_CSV = "/app/src/data/symbols/nifty_smallcap250.csv"
MID_MONTH_LEAD_PCT = MIDMONTH_LEAD  # shared with backtest (strategy.MIDMONTH_LEAD)


def is_mid_month_check_day(today: datetime) -> bool:
    """Live mirror of the backtest 'mid' calendar rule (SHARED
    strategy.is_mid_month_check_day) — first NSE trading day on/after the 15th,
    holiday-aware. Keeps live + backtest mid-month timing identical."""
    return S.is_mid_month_check_day(today)


def _load_smallcap_set() -> set:
    """Backtest excludes Nifty Smallcap 250 names (+2pp CAGR, DD unchanged).
    Live must mirror. Returns empty set on file missing — fail-soft.

    Returns:
        set[str]: plain NSE symbols (EQ series only) to subtract from the
        pseudo-N100 universe. Empty if the CSV is absent (no filter applied).
    """
    import csv as _csv
    out: set = set()
    try:
        with open(SMALLCAP_CSV) as f:
            for r in _csv.DictReader(f):
                if r.get("Series", "").strip() == "EQ":
                    out.add(r["Symbol"].strip())
    except FileNotFoundError:
        log.warning(f"smallcap CSV missing: {SMALLCAP_CSV} — no smallcap filter applied")
    return out


_SMALLCAP_SET = _load_smallcap_set()


# ---- Helpers ----

def is_rebalance_day(today: datetime, last_rotation: datetime = None) -> bool:
    """True if today is the monthly rebalance trigger.

    The backtest rebalances on the FIRST TRADING DAY of the month; live now
    matches exactly via nse_calendar (holiday-aware), and de-dupes so a month
    rotates at most once.

    Args:
        today: The date being evaluated.
        last_rotation: Date of the previous rotation, if known. If it falls in
            the same calendar month as `today`, this returns False (already
            rebalanced this month).

    Returns:
        bool: True only on the first NSE trading day of an un-rotated month.
    """
    if (last_rotation and last_rotation.year == today.year
            and last_rotation.month == today.month):
        return False  # already rebalanced this calendar month
    return is_first_trading_day_of_month(today)


def load_yearly_universes(path: str) -> Dict[str, List[str]]:
    """Read the yearly PIT universe map from disk.

    Args:
        path: Path to yearly_universes.json.

    Returns:
        Dict[str, List[str]]: {year_start_iso: [symbol, ...]} mapping.
    """
    with open(path) as f:
        return json.load(f)


def pick_universe_for(today: datetime,
                      yearly: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """Select the PIT universe in force today (latest year-start key <= today).

    Args:
        today: Current date.
        yearly: The {year_start_iso: [symbols]} map from load_yearly_universes.

    Returns:
        Tuple[str, List[str]]: (chosen_year_key, symbols). Falls back to the
        earliest key if no key is <= today (e.g. backtesting before the first
        rebuild date).
    """
    today_d = today.date()
    chosen_key: Optional[str] = None
    for key in sorted(yearly.keys()):
        try:
            d = datetime.strptime(key, "%Y-%m-%d").date()
        except ValueError:
            continue  # ignore non-date keys defensively
        if d <= today_d:
            chosen_key = key  # keep advancing to the most recent past year-start
    if chosen_key is None:
        chosen_key = sorted(yearly.keys())[0]  # before first rebuild: use earliest
    return chosen_key, yearly[chosen_key]


def get_close_at(symbol: str, target_ts: int) -> float:
    """Return the last cached close at/before target_ts for a symbol.

    Args:
        symbol: Fyers symbol, e.g. "NSE:RELIANCE-EQ".
        target_ts: Unix seconds; the close on/before this instant is returned.

    Returns:
        float: Most recent close within a 90-day window ending at target_ts,
        or 0.0 if no cached data exists.
    """
    # 90-day window back from target_ts is enough to find the most recent bar.
    df = read_cached(symbol, "D", target_ts - 90 * 86400, target_ts)
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["close"])


def get_close_trading_days_ago(symbol: str, today_ts: int, n: int) -> float:
    """Close exactly `n` TRADING days before the latest cached bar (backtest
    parity with cl.iloc[di-LOOKBACK]). The old calendar-day subtraction gave a
    ~30% shorter, weekend-drifting window. 0.0 if fewer than n+1 bars cached.
    """
    df = read_cached(symbol, "D", today_ts - (n * 3 + 90) * 86400, today_ts)
    if len(df) < n + 1:
        return 0.0
    return float(df.iloc[-(n + 1)]["close"])


def _close_above_sma200(symbol: str, today_ts: int) -> bool:
    """200d SMA uptrend gate (backtest parity).

    Requires ≥200 daily closes in cache. Returns False on insufficient data
    (fail-CLOSED — matches backtest which skips names without 200d history).

    Lookback: 200 trading days × ~7/5 calendar:trading ratio + holidays
    buffer = 420 calendar days. Tighter window misses SMA200 for some names
    (Indian markets ~250 trading days/yr).

    Args:
        symbol: Fyers symbol, e.g. "NSE:RELIANCE-EQ".
        today_ts: Unix seconds for "now".

    Returns:
        bool: True if the latest close exceeds the 200d SMA; False if it does
        not, or if fewer than 200 cached closes exist (fail-CLOSED).
    """
    lookback = int(SMA_LONG * 1.6 + 100) * 86400  # ~420 calendar days -> ≥200 trading bars
    df = read_cached(symbol, "D", today_ts - lookback, today_ts)
    if df.empty or len(df) < SMA_LONG:
        return False  # insufficient history — skip (backtest does the same)
    closes = df["close"].astype(float)
    sma = closes.iloc[-SMA_LONG:].mean()  # mean of the last 200 daily closes
    return float(closes.iloc[-1]) > float(sma)


def rank_universe(symbols: List[str], today_ts: int,
                  lookback_days: int = LOOKBACK) -> List[tuple]:
    """Return [(symbol, name, 30d_return_pct, current_price)] sorted desc.

    Filters applied (backtest parity):
      - Drop Nifty Smallcap 250 names
      - Drop current_price > MAX_PRICE
      - Drop names where close ≤ 200d SMA (uptrend gate)

    Args:
        symbols: Plain NSE symbols from the PIT universe (e.g. "RELIANCE").
        today_ts: Unix seconds for "now" (price + ranking anchor).
        lookback_days: Momentum window for the return calc (default 30).

    Returns:
        List[tuple]: (fyers_symbol, plain_symbol, ret_30d_pct, current_price)
        ordered highest-return first. Rank-1 is the entry candidate.
    """
    rows = []
    for plain_sym in symbols:
        if plain_sym in _SMALLCAP_SET:
            continue  # smallcap exclusion (backtest parity)
        fyers_sym = f"NSE:{plain_sym}-EQ"
        c_now = get_close_at(fyers_sym, today_ts)
        # 30 TRADING days back (backtest parity), not 30 calendar days.
        c_past = get_close_trading_days_ago(fyers_sym, today_ts, lookback_days)
        if c_now <= 0 or c_past <= 0:
            continue  # missing price on either anchor — cannot rank
        if c_now > MAX_PRICE:
            continue  # MAX_PRICE (₹3000) gate — share-count floor / giant-loser guard
        if not _close_above_sma200(fyers_sym, today_ts):
            continue  # 200d-SMA uptrend gate
        ret = (c_now / c_past - 1) * 100  # 30-day trailing return %
        rows.append((fyers_sym, plain_sym, ret, c_now))
    rows.sort(key=lambda r: -r[2])  # rank by 30d return, descending
    return rows


def get_current_position() -> Optional[Dict]:
    """Read model's open position from model_ledger."""
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger(MODEL_NAME)
        if l and l.get("open_symbol"):
            return l
    except Exception as e:
        log.warning(f"ledger read failed: {e}")
    return None


def _last_rotation_date():
    """Entry date of the open position (None if flat) — dedups the monthly
    rebalance to once per calendar month (else day 1-7 re-fires each weekday)."""
    try:
        pos = get_current_position()
        if pos and pos.get("open_entry_date"):
            return datetime.fromisoformat(pos["open_entry_date"])
    except Exception as e:
        log.debug(f"last_rotation read failed: {e}")
    return None


def is_model_enabled() -> bool:
    """Query model_settings.enabled. Fail-CLOSED on read errors to avoid
    trading without confirming the operator has the model active."""
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == MODEL_NAME:
                return bool(s.get("enabled"))
        return False
    except Exception as e:
        log.warning(f"enabled-flag read failed: {e} — defaulting to OFF")
        return False


def emit_signals(top_picks: List[tuple], pos: Optional[Dict],
                 top_n: int, retain_top_n: int = RETAIN) -> List[Dict]:
    # Decision comes from the SHARED rotation core (tools/shared/rotation_strategy)
    # — the exact same rule backtest.py uses, so live and backtest cannot drift.
    # retain_top_n default = RETAIN (1 = top-1 rotation; wins on the fixed anchor).
    ranked = [p[0] for p in top_picks]
    by_sym = {p[0]: p for p in top_picks}
    held_sym = pos.get("open_symbol") if pos else None
    dec = decide_rotation(held_sym, ranked, retain_top_n)

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals: List[Dict] = []

    if dec.sell:
        price = get_close_at(dec.sell, int(datetime.now().timestamp()))
        entry_px = float(pos.get("open_entry_px") or 0) if pos else 0.0
        kind = "TARGET_HIT" if price >= entry_px else "STOP_HIT"
        signals.append({
            "model": MODEL_NAME,
            "universe": "pseudo_n100",
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
        signals.append({
            "model": MODEL_NAME,
            "universe": "pseudo_n100",
            "symbol": sym,
            "company": name,
            "ts": today_str,
            "side": "BUY",
            "signal": "ENTRY1",
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"30d momentum rank-1 ({ret:+.2f}%) — pseudo-N100",
        })

    return signals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes-file", required=True,
                    help="Path to yearly_universes.json (PIT universe map)")
    ap.add_argument("--top-n", type=int, default=5,
                    help="Ranking display size only.")
    ap.add_argument("--retain-top-n", type=int, default=RETAIN,
                    help="Exit retention band: hold while in top-N by 30d ret, "
                         f"rotate when out. Default {RETAIN} (2026-05-30 sweep). "
                         "1=legacy top-1 rotation.")
    ap.add_argument("--mid-month-check", action="store_true",
                    help="Day-15 check: emit ROTATE only if rank-1 leads the "
                         f"held stock's 30d return by >= {MIDMONTH_LEAD}pp "
                         "(holiday-aware day-15 gate). Mirrors the backtest 'mid' "
                         "calendar; cron runs this as a second monthly job.")
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--ranking-out", default=None,
                    help="Where to write the Today's Picks ranking JSON. "
                         "Default = the canonical ranking dir. The admin "
                         "display path passes a /tmp path so a page view never "
                         "clobbers the morning's audited ranking snapshot.")
    ap.add_argument("--rebalance-only", action="store_true",
                    help="Skip if today is not rebalance trigger day")
    ap.add_argument("--force", action="store_true",
                    help="Bypass rebalance-day + enabled checks")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    log.info(f"{MODEL_NAME} signal run: today={today.date()} "
             f"weekday={today.strftime('%A')} day_of_month={today.day}")

    # Even when disabled / non-rebalance day, we still want the Today's Picks
    # UI to show the ranking. Compute it up front and write to the per-model
    # ranking dir, then proceed with the enabled-flag + rebalance gates.
    yearly = load_yearly_universes(args.universes_file)
    universe_key, symbols = pick_universe_for(today, yearly)
    today_ts = int(today.timestamp())
    ranks = rank_universe(symbols, today_ts)
    log.info(f"PIT universe: {universe_key} → {len(symbols)} symbols")

    # --ranking-out overrides the canonical path so display-only runs (admin
    # Today's Picks) route to /tmp and don't clobber the audited snapshot.
    if args.ranking_out:
        ranking_path = Path(args.ranking_out)
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ranking_dir = Path("/app/logs/momrot_pseudo/ranking")
        ranking_dir.mkdir(parents=True, exist_ok=True)
        ranking_path = ranking_dir / f"{today.strftime('%Y-%m-%d')}.json"
    ranking_payload = {
        "model": MODEL_NAME,
        "date": today.strftime("%Y-%m-%d"),
        "universe_size": len(symbols),
        "top_n": [
            {
                "rank": i + 1,
                "symbol": plain,
                "name": plain,
                "ret_30d_pct": round(ret, 2),
                "price": round(price, 2),
            }
            for i, (_fyers, plain, ret, price) in enumerate(ranks[:5])
        ],
    }
    ranking_path.write_text(json.dumps(ranking_payload, indent=2, default=str))
    log.info(f"Wrote ranking -> {ranking_path}")

    # Enabled-flag gate (skippable via --force)
    if not args.force and not is_model_enabled():
        log.info(f"{MODEL_NAME}: model_settings.enabled is False — "
                 "writing empty signals file and exiting.")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        try:
            from src.services.notification_service import notify_skip
            notify_skip(MODEL_NAME, "model disabled")
        except Exception as _ne:
            log.debug(f"notify_skip failed: {_ne}")
        return 0

    # Mid-month gate (mutually exclusive with rebalance-only; --force overrides).
    # Only proceed on the day-15 weekday; the lead-threshold filter below then
    # suppresses the rotation unless rank-1 leads the held stock by >= the gate.
    if args.mid_month_check and not args.force:
        if not is_mid_month_check_day(today):
            log.info("Not mid-month check day (need first trading day on/after "
                     "the 15th). Skipping.")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.signals_out).write_text(json.dumps([]))
            return 0

    # Monthly rebalance gate
    if args.rebalance_only and not args.force and not args.mid_month_check:
        if not is_rebalance_day(today, _last_rotation_date()):
            log.info("Not rebalance day (need day<=7 weekday + not already "
                     "rotated this month). Skipping.")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.signals_out).write_text(json.dumps([]))
            try:
                from src.services.notification_service import notify_skip
                notify_skip(MODEL_NAME, "not rebalance day")
            except Exception as _ne:
                log.debug(f"notify_skip failed: {_ne}")
            return 0

    pos = get_current_position()
    log.info(f"Currently held: {pos.get('open_symbol') if pos else 'none'}")
    log.info(f"Ranked {len(ranks)} stocks (after MAX_PRICE={MAX_PRICE} filter). "
             f"Top-{args.top_n}:")
    for i, (sym, name, ret, price) in enumerate(ranks[:args.top_n], 1):
        log.info(f"  {i}. {sym:<20} {ret:+7.2f}%  @ ₹{price:.2f}")

    signals = emit_signals(ranks, pos, args.top_n, retain_top_n=args.retain_top_n)

    # Mid-month lead-threshold filter (mirror of n100): on the day-15 check,
    # suppress rotation unless the new rank-1 leads the held stock's 30d return
    # by >= MID_MONTH_LEAD_PCT (or the held stock dropped out of the ranking).
    # Keeps mid-cycle trade count low while still catching genuine breakouts.
    if args.mid_month_check and pos and pos.get("open_symbol") and signals:
        held_sym = pos["open_symbol"]
        ranked_ret = [(r[0], r[2]) for r in ranks]
        if midmonth_lead_ok(held_sym, ranked_ret, MID_MONTH_LEAD_PCT):
            log.info(f"mid-month: ROTATE — rank-1 leads held {held_sym} "
                     f"by >= {MID_MONTH_LEAD_PCT}pp (or held dropped from ranking).")
        else:
            log.info(f"mid-month: no rotation for held {held_sym} "
                     f"(already rank-1, or lead < {MID_MONTH_LEAD_PCT}pp). Suppressing.")
            signals = []

    log.info(f"Emitting {len(signals)} signals")

    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.signals_out, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {args.signals_out}")

    # Audit hook
    try:
        from src.services.audit_service import write_rankings, write_signal
        write_rankings(MODEL_NAME, today.date(),
                       ranking_payload.get("universe_size") or 0,
                       0, ranking_payload.get("top_n") or [])
        # Audit signals ONLY for scheduled (cron) runs, not manual --force.
        if not args.force:
            if signals:
                for _sig in signals:
                    write_signal(MODEL_NAME, today.date(),
                                 _sig.get("signal", ""), _sig.get("symbol", ""),
                                 _sig.get("side", ""), price=_sig.get("price"),
                                 reason=(_sig.get("note") or "")[:120])
            else:
                write_signal(MODEL_NAME, today.date(), "HOLD", "", "NONE",
                             reason="no signal emitted")
    except Exception as _e:
        log.debug(f"audit hook failed: {_e}")

    # Notification funnel — ping the verdict even on no-change (eval day only).
    if not args.force:
        try:
            from src.services.notification_service import notify_model_decision
            _held = pos.get("open_symbol") if pos else None
            _ret = next((r[2] for r in ranks if r[0] == _held), None) if _held else None
            notify_model_decision(MODEL_NAME, signals, held_symbol=_held,
                                  held_ret=_ret, trigger="CRON")
        except Exception as _ne:
            log.debug(f"notify decision failed: {_ne}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
