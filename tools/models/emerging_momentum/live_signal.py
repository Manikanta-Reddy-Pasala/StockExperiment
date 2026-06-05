"""Live signal generator for emerging_momentum — SINGLE-POSITION rotation.

Mirrors tools/models/momentum_n100_top5_max1/live_signal.py 1:1, but ranks the
POINT-IN-TIME emerging pool (top-100 ADV from N500-minus-N100) instead of the
real Nifty 100. Uses the SHARED core `strategy.py` (build_pools / pool_for_date /
rank_pool / midret_pool / params) so backtest and live cannot drift, and the
shared rotation rule tools.shared.rotation_strategy.decide_rotation /
midmonth_lead_ok so the decision matches the backtest engine.

Strategy:
  - Universe: PIT emerging mid/small (top-100 20d-ADV of N500 minus N100),
    rebuilt per year-start (strategy.build_pools).
  - Rank: 15-trading-day return, ret > 0, price in (0, 3000]; NO sma gate.
  - max_concurrent = 1, retain_top_n = 3 (hold while in top-3).
  - rebalance: 1st trading day of month + mid-month (day-15-weekday) lead-gate.

Logic per run (single-position, STATEFUL via the DB model_ledger):
  1. Load current open position from model_ledger (open_symbol).
  2. Rank the PIT pool by 15d return.
  3. SELL (rotation exit) when the held name drops out of the top-3 band.
  4. BUY (ENTRY1) the new rank-1 when it differs from what's held.
  Mid-month runs additionally require the new rank-1 to lead the held name's
  15d return by >= MIDMONTH_LEAD percentage points (suppressed otherwise).

Emits a SINGLE-position signals file consumed by tools/live/fyers_executor.py
(--model-name emerging_momentum). NOT the multi-holding executor/ledger.

Usage:
  python tools/models/emerging_momentum/live_signal.py \
    --signals-out /app/logs/emerging_momentum/signals/<date>.json \
    --rebalance-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402
from tools.shared.index_membership import universe_union  # noqa: E402
from tools.shared.rotation_strategy import decide_rotation, midmonth_lead_ok  # noqa: E402
from tools.shared.nse_calendar import is_first_trading_day_of_month  # noqa: E402
from tools.models.emerging_momentum import strategy as S  # noqa: E402

log = logging.getLogger("emerging_momentum_signal")
MODEL_NAME = "emerging_momentum"
STATE_DIR = Path("/app/logs/emerging_momentum")


def panel_is_stale(last_date, today, max_gap_days: int = 7) -> bool:
    """True if the panel's most recent equity day is too old to trade on.

    Parity with the per-symbol bar_is_stale gate n100/pseudo use — emerging
    ranks off a pivoted panel, so the freshness check is at the panel level:
    if the last equity day is more than ``max_gap_days`` calendar days before
    today (failed nightly OHLCV pull / dead feed), abort rather than emit real
    ENTRY/STOP/TARGET signals priced off a stale close. Accepts date,
    datetime, or pandas Timestamp on either side.
    """
    try:
        ld = pd.Timestamp(last_date).date()
        td = pd.Timestamp(today).date()
        return (td - ld).days > max_gap_days
    except Exception:
        return False


def is_rebalance_day(today: datetime, last_rotation: datetime = None) -> bool:
    """True if today is the monthly rebalance trigger (1st NSE trading day of
    month, holiday-aware). Skips if we already rotated this calendar month."""
    if last_rotation and last_rotation.year == today.year and last_rotation.month == today.month:
        return False
    return is_first_trading_day_of_month(today)


def is_mid_month_check_day(today: datetime) -> bool:
    """First NSE trading day on/after the 15th (holiday-aware). Delegates to the
    SHARED strategy.is_mid_month_check_day — the EXACT rule the backtest calendar
    uses (build_calendar 'mid'), so live + backtest can never disagree on when
    the mid-month check fires."""
    return S.is_mid_month_check_day(today)


def is_model_enabled() -> bool:
    """model_settings.enabled for this model. Fail-CLOSED on error/missing row
    so a disabled model (or a config/DB problem) can never place real orders."""
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == MODEL_NAME:
                return bool(s.get("enabled"))
        return False
    except Exception as e:
        log.warning(f"enabled-flag read failed: {e} — defaulting to OFF")
        return False


def _last_rotation_date():
    """Entry date of the current open position (None if flat), used to dedup the
    monthly rebalance to at most once per calendar month."""
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger(MODEL_NAME)
        if l and l.get("open_entry_date"):
            return datetime.fromisoformat(l["open_entry_date"])
    except Exception as e:
        log.debug(f"last_rotation read failed: {e}")
    return None


def held_from_db() -> List[Dict]:
    """Current open position from the single-position model_ledger (DB).

    Makes the model STATEFUL: it reads what it actually holds so it can emit a
    rotation SELL when the held name drops out of the top-3. Mirrors
    momentum_n100_top5_max1.held_from_db (open_symbol / open_entry_px)."""
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger(MODEL_NAME)
        if l and l.get("open_symbol"):
            return [{
                "symbol": l["open_symbol"],
                "entry_price": float(l.get("open_entry_px") or 0),
                "qty": int(l.get("open_qty") or 0),
                "profit_taken": bool(l.get("profit_taken", False)),
            }]
    except Exception as e:
        log.warning(f"DB ledger read failed: {e}; treating as flat")
    return []


def load_panel(symbols, days_back=420):
    """Load the close + ADV panel for the N500 universe (+ index for the
    equity-trading-day mask). 420 calendar days covers the 20d-ADV + 15d-return
    warmup with holiday slack."""
    eng = _get_engine()
    end = datetime.now().date()
    start = end - timedelta(days=days_back)
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b AND data_source='fyers' "
            "ORDER BY symbol,date"
        ), c, params={"s": symbols, "a": start, "b": end})
    df["date"] = pd.to_datetime(df["date"])
    df["adv"] = df["close"].astype(float) * df["volume"].astype(float)
    return df


def emit_signals(ranked: List[str], midret: List[tuple], held: List[Dict],
                 cl, di, retain_top_n: int = 3) -> List[Dict]:
    """Turn a ranking + current holding into SELL/BUY signal dicts.

    Delegates the keep/rotate decision to the shared decide_rotation core so
    live and backtest stay in lockstep, then materialises the resulting sell
    and/or buy into the signal dict shape fyers_executor consumes.

    Args:
        ranked: Symbols best-first from strategy.rank_pool (filtered).
        midret: (symbol, 15d_ret_pct) pool-order list from strategy.midret_pool;
            used only to look up a name's return for the signal note.
        held: Current holding list (0 or 1 entry) from the DB ledger.
        cl: close panel (date x symbol) for price lookup.
        di: latest row index in `cl` (today).
        retain_top_n: exit retention band (3 = canonical Config-1).

    Returns:
        list[dict]: 0–2 signal dicts (SELL rotation exit and/or BUY ENTRY1).
    """
    ret_by_sym = dict(midret)
    held_sym = held[0]["symbol"] if held else None
    held_entry = float(held[0].get("entry_price", 0)) if held else 0.0
    dec = decide_rotation(held_sym, ranked, retain_top_n)

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals = []

    def _px(sym):
        try:
            v = cl[sym].iloc[di]
            return float(v) if pd.notna(v) else 0.0
        except Exception:
            return 0.0

    if dec.sell:
        price = _px(dec.sell)
        kind = "TARGET_HIT" if price >= held_entry else "STOP_HIT"
        signals.append({
            "model": "emerging_momentum",
            "universe": "emerging_n500_minus_n100",
            "symbol": dec.sell,
            "company": dec.sell.split(":")[-1].replace("-EQ", ""),
            "ts": today_str,
            "side": "SELL",
            "signal": kind,
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"rotation exit (dropped out of top-{retain_top_n})",
        })

    if dec.buy:
        price = _px(dec.buy)
        ret = ret_by_sym.get(dec.buy, 0.0)
        signals.append({
            "model": "emerging_momentum",
            "universe": "emerging_n500_minus_n100",
            "symbol": dec.buy,
            "company": dec.buy.split(":")[-1].replace("-EQ", ""),
            "ts": today_str,
            "side": "BUY",
            "signal": "ENTRY1",
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"vol-adj momentum rank-1 ({ret:+.2f}%)",
        })

    return signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--ranking-out", default=None,
                    help="Where to write the Today's Picks ranking JSON. "
                         "Default = the canonical ranking dir.")
    ap.add_argument("--retain-top-n", type=int, default=S.RETAIN,
                    help="Exit retention band: hold while in top-N by 15d ret, "
                         "rotate when out. Default 3 (Config-1 canonical).")
    ap.add_argument("--rebalance-only", action="store_true",
                    help="Skip if today is not the monthly rebalance trigger.")
    ap.add_argument("--mid-month-check", action="store_true",
                    help="Day-15 check: rotate only if rank-1 leads held by "
                         ">= MIDMONTH_LEAD (default 5pp).")
    ap.add_argument("--force", action="store_true",
                    help="Bypass date gate (initial deploy / manual).")
    ap.add_argument("--stop-check", action="store_true",
                    help="DAILY ATR-from-entry hard-stop check: emit a STOP_HIT "
                         "exit if the held name's low pierced entry - "
                         "ATR_STOP_MULT*ATR. Runs every trading day.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    log.info(f"emerging_momentum_signal run: today={today.date()} "
             f"weekday={today.strftime('%A')} day_of_month={today.day}")

    sig_path = Path(args.signals_out)
    sig_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_empty():
        with open(sig_path, "w") as f:
            json.dump([], f)

    # Enabled gate (fail-closed).
    if not args.force and not is_model_enabled():
        log.warning(f"{MODEL_NAME}: model_settings.enabled is False — "
                    f"writing empty signals file and exiting.")
        _write_empty()
        return 0

    # ---- DAILY ATR-from-entry hard-stop check (separate daily mode) ----
    if args.stop_check:
        held = held_from_db()
        if not held or not held[0].get("symbol"):
            log.info("stop-check: flat — nothing to check."); _write_empty(); return 0
        h = held[0]
        sym = h["symbol"]; entry_px = float(h.get("entry_price") or 0)
        if entry_px <= 0:
            log.warning("stop-check: held has no entry_px; skip."); _write_empty(); return 0
        df = load_panel([sym])
        if df.empty:
            log.warning(f"stop-check: no data for {sym}; skip."); _write_empty(); return 0
        df = df.sort_values("date")
        if panel_is_stale(df["date"].iloc[-1], today):
            log.error("stop-check: data stale; skip (fail-safe)."); _write_empty(); return 1
        atr_val = S.atr_latest(df["high"].astype(float), df["low"].astype(float),
                               df["close"].astype(float))
        day_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        # --- PARTIAL PROFIT-TAKE: book HALF once at entry*(1+PROFIT_TAKE_PCT) ---
        # Checked on the close (an UP move), BEFORE the down-side ATR stop. Fires
        # once per position (model_ledger.profit_taken). Mirrors the backtest walk
        # so live + backtest cannot drift.
        _PT = float(getattr(S, "PROFIT_TAKE_PCT", 0.0) or 0.0)
        held_qty = int(h.get("qty") or 0)
        if (_PT > 0 and not h.get("profit_taken") and held_qty >= 2
                and entry_px > 0 and last_close >= entry_px * (1 + _PT)):
            sell_qty = held_qty // 2
            signals = [{
                "model": MODEL_NAME, "symbol": sym,
                "company": sym.split(":")[-1].replace("-EQ", ""),
                "ts": today.strftime("%Y-%m-%d %H:%M:%S"),
                "side": "SELL", "signal": "PROFIT_TAKE", "reason": "PROFIT_TAKE",
                "qty": sell_qty, "price": float(last_close), "sl": 0.0, "target": 0.0,
                "note": f"partial profit-take: close {last_close:.2f} >= "
                        f"entry {entry_px:.2f} x (1+{_PT:.0%}); booking half "
                        f"({sell_qty}/{held_qty}), runner rides the ATR stop",
            }]
            with open(sig_path, "w") as f:
                json.dump(signals, f)
            log.info(f"stop-check: PROFIT_TAKE emitted for {sym} — sell {sell_qty}/{held_qty} @ {last_close:.2f}")
            return 0
        hit, lvl = S.atr_stop_hit(entry_px, atr_val, day_low)
        log.info(f"stop-check {sym}: entry={entry_px:.2f} ATR={atr_val} "
                 f"stop_level={lvl} day_low={day_low:.2f} -> {'HIT' if hit else 'ok'}")
        if hit:
            signals = [{
                "model": MODEL_NAME, "symbol": sym,
                "company": sym.split(":")[-1].replace("-EQ", ""),
                "ts": today.strftime("%Y-%m-%d %H:%M:%S"),
                "side": "SELL", "signal": "STOP_HIT",
                "price": float(lvl), "sl": 0.0, "target": 0.0,
                "note": f"ATR-from-entry stop: low {day_low:.2f} <= "
                        f"entry {entry_px:.2f} - {S.ATR_STOP_MULT}x ATR ({lvl:.2f})",
            }]
            with open(sig_path, "w") as f:
                json.dump(signals, f)
            log.info(f"stop-check: STOP_HIT emitted for {sym} @ {lvl:.2f}")
        else:
            _write_empty()
        return 0

    # Mid-month gate (mutually exclusive with rebalance-only; --force overrides).
    if args.mid_month_check and not args.force:
        if not is_mid_month_check_day(today):
            log.info("Not mid-month check day (need day-15-weekday). Skipping.")
            _write_empty()
            return 0

    if args.rebalance_only and not args.force and not args.mid_month_check:
        if not is_rebalance_day(today, _last_rotation_date()):
            log.info("Not rebalance day (need first NSE trading day of month "
                     "+ not already rotated this month). Skipping.")
            _write_empty()
            try:
                from src.services.notification_service import notify_skip
                notify_skip(MODEL_NAME, "not rebalance day")
            except Exception as _ne:
                log.debug(f"notify_skip failed: {_ne}")
            return 0

    # ---- Load data + build the PIT pool / indicators ----
    syms = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))] + [S.INDEX]
    df = load_panel(syms)
    if df.empty:
        log.error("No data."); _write_empty(); return 1
    cl = df.pivot(index="date", columns="symbol", values="close")
    adv_rs = df.pivot(index="date", columns="symbol", values="adv")
    # Restrict to EQUITY trading days (index-only rows poison rolling windows).
    equity_dates = adv_rs.drop(columns=[S.INDEX], errors="ignore").dropna(how="all").index
    cl = cl.loc[equity_dates].ffill()
    adv_rs = adv_rs.loc[equity_dates]
    adv20 = S.indicators(cl, adv_rs)
    dates = cl.index
    di = len(dates) - 1  # today = last equity day

    # Freshness gate (parity with n100/pseudo bar_is_stale): if the last equity
    # day in the panel is stale (failed nightly pull / dead feed), abort rather
    # than emit real ENTRY/STOP/TARGET signals off an old close.
    if len(dates) and panel_is_stale(dates[di], today):
        log.error(f"Panel stale — last equity day {pd.Timestamp(dates[di]).date()} "
                  f"vs today {today.date()}; aborting (no signals).")
        try:
            from tools.live.telegram_notify import alert_data_missing
            alert_data_missing(MODEL_NAME, f"Daily price data STALE — last close {pd.Timestamp(dates[di]).date()} (>7d old).")
        except Exception as _e:
            log.debug(f"tg alert failed: {_e}")
        _write_empty()
        return 1

    anchors, pools = S.build_pools(adv20, dates, today.date())
    pool = S.pool_for_date(anchors, pools, dates[di])
    log.info(f"PIT pool: {len(pool)} emerging symbols")

    ranked = S.rank_pool(cl, pool, di)
    midret = S.midret_pool(cl, pool, di)
    log.info(f"Ranked {len(ranked)} leaders. Top-5: "
             f"{[s.split(':')[-1] for s in ranked[:5]]}")

    held = held_from_db()
    log.info(f"Currently held: {[h['symbol'] for h in held]}")

    signals = emit_signals(ranked, midret, held, cl, di,
                           retain_top_n=args.retain_top_n)

    # Mid-month lead-threshold filter: suppress rotation unless the new rank-1
    # leads the held name's 15d return by >= MIDMONTH_LEAD. Uses the SHARED
    # midmonth_lead_ok with the pool-order return list (backtest parity).
    if args.mid_month_check and held and signals:
        held_sym = held[0]["symbol"]
        if midmonth_lead_ok(held_sym, midret, S.MIDMONTH_LEAD):
            log.info(f"mid-month: ROTATE — lead gate passed for held {held_sym}.")
        else:
            log.info(f"mid-month: no rotation for held {held_sym} "
                     f"(lead < {S.MIDMONTH_LEAD}pp). Suppressing.")
            signals = []

    log.info(f"Emitting {len(signals)} signals")

    # Canonical signals file the cron executor (fyers_executor) reads & acts on.
    with open(sig_path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {sig_path}")

    # Persist top-N ranking for the Today's Picks UI (always written).
    if args.ranking_out:
        ranking_path = Path(args.ranking_out)
    else:
        ranking_path = STATE_DIR / "ranking" / f"{today.strftime('%Y-%m-%d')}.json"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ret_by_sym = dict(midret)
    top_payload = {
        "model": MODEL_NAME,
        "date": today.strftime("%Y-%m-%d"),
        "universe_size": len(pool),
        "top_n": [
            {
                "rank": i + 1,
                "symbol": s.split(":")[-1].replace("-EQ", ""),
                "name": s.split(":")[-1].replace("-EQ", ""),
                "ret_15d_pct": round(ret_by_sym.get(s, 0.0), 2),
                # UI "Return" column reads ret_30d_pct → emit the actual 30d return.
                "ret_30d_pct": (round((float(cl[s].iloc[di]) / float(cl[s].iloc[di - 30]) - 1) * 100, 2)
                                if di >= 30 and pd.notna(cl[s].iloc[di - 30]) and float(cl[s].iloc[di - 30]) > 0 else 0.0),
                "price": round(float(cl[s].iloc[di]), 2),
            }
            for i, s in enumerate(ranked[:5])
        ],
    }
    ranking_path.write_text(json.dumps(top_payload, indent=2, default=str))
    log.info(f"Wrote ranking -> {ranking_path}")

    # Audit hook — record the ranking snapshot and each signal (or a HOLD).
    try:
        from src.services.audit_service import write_rankings, write_signal
        write_rankings(MODEL_NAME, today.date(),
                       top_payload.get("universe_size") or 0,
                       0, top_payload.get("top_n") or [])
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

    # Notification funnel — single-position decision ping (scheduled runs only).
    if not args.force:
        try:
            from src.services.notification_service import notify_model_decision
            _held = held[0]["symbol"] if held else None
            _ret = ret_by_sym.get(_held) if _held else None
            notify_model_decision(
                MODEL_NAME, signals, held_symbol=_held, held_ret=_ret,
                trigger="MID_MONTH" if args.mid_month_check else "CRON",
            )
        except Exception as _ne:
            log.debug(f"notify decision failed: {_ne}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
