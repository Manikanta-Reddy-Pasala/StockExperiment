"""n20_daily_large_only — daily live signal generator.

Daily rotation: ranks top-20 N500 stocks by 20d ADV ∩ Nifty 100 with
close > 200d SMA (uptrend filter), then picks rank-1 by 30d return.

Strategy:
  - Universe: top-20 by 20d ADV from N500 ∩ NSE Nifty 100
  - Uptrend filter: close > 200d SMA
  - Rank by 30d return desc
  - top_n = 1 (max_concurrent=1)
  - Rebalance: DAILY (not monthly — much higher turnover)

Logic per run:
  1. Build today's PIT universe (top-20 ADV ∩ N100, uptrend-filtered)
  2. Rank universe by 30d return
  3. If held NOT in top-1 → emit STOP_HIT / TARGET_HIT (rotation exit)
  4. Emit ENTRY1 for rank-1 if not already held

Role in the model flow (data_pull -> live_signal -> cron -> backtest)
---------------------------------------------------------------------
This is the LIVE trading leg, run every weekday morning by cron.py
(emit_signal -> execute_orders). It reads the open position from the DB
ledger (get_current_position), rebuilds today's point-in-time ranking from
the N500 OHLCV that data_pull.py keeps fresh, and writes a signals JSON that
fyers_executor.py then turns into real orders.

The actual SELL/BUY decision is delegated to the SHARED rotation core
(tools/shared/rotation_strategy.decide_rotation) — the exact same rule
backtest.py drives through the shared engine — so the published backtest
numbers describe live behaviour and the two paths cannot drift.

Side effects beyond the signals file: a per-model ranking JSON for the
Today's Picks UI, audit rows (rankings + signals) to the DB, and a
Telegram/PWA notification of the verdict (scheduled runs only, not --force).

Usage:
  python tools/models/n20_daily_large_only/live_signal.py \
    --signals-out /app/logs/n20_daily/signals/$(date +%F)_n20.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402
from tools.shared.universes import nifty100_symbols, nifty500_symbols  # noqa: E402
from tools.shared.rotation_strategy import decide_rotation  # noqa: E402
from tools.models.n20_daily_large_only.strategy import (  # noqa: E402  shared w/ backtest
    UNIV_SIZE, ADV_WIN, SMA_LONG, RETAIN, LOOKBACK as LOOKBACK_RET)

log = logging.getLogger("n20_daily_signal")

MODEL_NAME = "n20_daily_large_only"


def is_weekday(today: datetime) -> bool:
    """Return True for Mon-Fri (weekday() 0-4); False on Sat/Sun.

    Args:
        today: the datetime to test.

    Returns:
        bool: True if `today` falls on a trading weekday.
    """
    return today.weekday() < 5


def is_model_enabled() -> bool:
    """Check model_settings.enabled. Fail-closed on error (skip trading).

    Reads the model_settings rows and looks up this model's `enabled` flag.

    Returns:
        bool: True only if this model's row exists and is enabled. Any error
        (or a missing row) returns False so a config/DB problem never lets the
        model trade unintentionally.
    """
    try:
        from src.services.trading.model_ledger_service import get_all_settings
        for s in get_all_settings():
            if s["model_name"] == MODEL_NAME:
                return bool(s.get("enabled"))
        return False
    except Exception as e:
        log.warning(f"enabled-flag read failed: {e} — defaulting to DISABLED")
        return False


def get_current_position() -> Optional[Dict]:
    """Read this model's open position from the DB ledger.

    The live path is stateful: what we currently hold determines whether a
    SELL is needed, so the position is sourced from the persisted ledger
    rather than recomputed.

    Returns:
        Optional[Dict]: the ledger row when an open position exists (non-empty
        `open_symbol`), else None (flat, or on read error).
    """
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger(MODEL_NAME)
        # Only treat as held if the ledger actually names an open symbol.
        if l and l.get("open_symbol"):
            return l
    except Exception as e:
        log.warning(f"ledger read failed: {e}")
    return None


def load_panel(symbols: List[str], days_back: int = 400) -> pd.DataFrame:
    """Load OHLCV panel for symbol list ending today.

    Args:
        symbols: fyers-form symbols (e.g. "NSE:RELIANCE-EQ") to fetch.
        days_back: calendar lookback window; default 400 to fully warm up the
            200d SMA and the 20d ADV used downstream.

    Returns:
        pd.DataFrame: long-format rows (symbol, date, close, volume) with
        `date` parsed to datetime; empty if no data is found.
    """
    eng = _get_engine()
    end = datetime.now().date()
    start = end - timedelta(days=days_back)
    with eng.connect() as c:
        df = pd.read_sql(
            text(
                "SELECT symbol,date,close,volume FROM historical_data "
                "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b "
                "AND data_source='fyers' ORDER BY symbol,date"
            ),
            c, params={"s": symbols, "a": start, "b": end},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_pit_universe_and_rank(df: pd.DataFrame, n100: set
                                ) -> List[tuple]:
    """Return ranked top-20 ∩ N100 with uptrend filter, by 30d return.

    Mirrors backtest.py's `rank_at` for the single latest day: rebuild the
    ADV/SMA matrices from the loaded panel, take the day's top-20 by 20d ADV,
    apply the uptrend and Nifty-100 filters, then order by 30d return.

    Args:
        df: long-format OHLCV panel (symbol, date, close, volume) from
            load_panel.
        n100: set of plain Nifty-100 symbols (large-cap filter).

    Returns:
        List[tuple]: rows ordered best-first as
        [(fyers_symbol, plain_symbol, 30d_return_pct, current_price)].
        Empty if there is no data or no candidate survives the filters.
    """
    if df.empty:
        return []
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # ADV proxy in rupees = close * volume (rupee turnover, not share count).
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)

    # FIX 6 — staleness filter. pivot().ffill() (below) carries the last close
    # of a delisted/suspended name forward as a flat line, so it could rank or
    # be held at a phantom price. From the RAW df (pre-ffill), keep only names
    # whose true last bar is within STALE_SESSIONS trading sessions of the
    # panel's latest date; drop the rest from the candidate universe.
    STALE_SESSIONS = 5
    all_dates = sorted(df["date"].unique())
    if len(all_dates) > STALE_SESSIONS:
        cutoff_date = all_dates[-(STALE_SESSIONS + 1)]  # STALE_SESSIONS back
    else:
        cutoff_date = all_dates[0]
    last_seen = df.groupby("symbol")["date"].max()
    fresh_syms = set(last_seen[last_seen >= cutoff_date].index)

    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    if cl.empty:
        return []
    adv20 = adv.rolling(ADV_WIN).mean()     # 20-day average daily turnover
    sma200 = cl.rolling(SMA_LONG).mean()    # 200-day SMA (long uptrend ref)

    today_row = cl.iloc[-1]
    # Top-20 by 20d ADV, rebuilt fresh from the latest day's turnover snapshot.
    # FIX 6 — only rank names that actually traded recently (exclude stale/
    # delisted symbols the ffill would otherwise carry forward at a phantom px).
    pit_adv = adv20.iloc[-1].dropna().sort_values(ascending=False)
    pit_adv = pit_adv[pit_adv.index.isin(fresh_syms)]
    pit_univ = pit_adv.head(UNIV_SIZE).index.tolist()

    # Uptrend filter: keep only names trading above their 200d SMA today.
    sma_today = sma200.iloc[-1]
    pit_univ = [s for s in pit_univ if pd.notna(sma_today.get(s))
                and pd.notna(today_row.get(s))
                and float(today_row[s]) > float(sma_today[s])]
    # Large-cap filter: intersect with the NSE Nifty 100 constituents.
    pit_univ = [s for s in pit_univ
                if s.replace("NSE:", "").replace("-EQ", "") in n100]

    # Need at least 30 trading days + today to compute the 30d return.
    if len(cl) < LOOKBACK_RET + 1:
        return []
    ref_row = cl.iloc[-LOOKBACK_RET - 1]     # close 30 trading days ago
    rows = []
    for sym in pit_univ:
        c_now = float(today_row.get(sym, 0) or 0)
        c_past = float(ref_row.get(sym, 0) or 0)
        # Skip symbols missing either endpoint price (would corrupt the ratio).
        if c_now <= 0 or c_past <= 0:
            continue
        ret = (c_now / c_past - 1) * 100     # trailing 30-day return %
        plain = sym.replace("NSE:", "").replace("-EQ", "")
        rows.append((sym, plain, ret, c_now))
    rows.sort(key=lambda r: -r[2])           # rank by 30d return, highest first
    return rows


def emit_signals(top_picks: List[tuple], pos: Optional[Dict],
                 top_n: int) -> List[Dict]:
    """Turn today's ranking + held position into SELL/BUY signal dicts.

    Delegates the actual decision to the shared rotation core and serialises
    its verdict into the signal records fyers_executor.py consumes.

    Args:
        top_picks: ranked rows from build_pit_universe_and_rank
            [(fyers_symbol, plain, 30d_ret_pct, price)], best-first.
        pos: current open position (ledger row) or None if flat.
        top_n: retention band passed to decide_rotation (1 live).

    Returns:
        List[Dict]: zero, one, or two signal records — an exit (SELL) for the
        held symbol if it dropped out of the top-N, and/or an ENTRY1 (BUY) for
        rank-1 if not already held.
    """
    # Decision via the SHARED rotation core — same rule backtest.py uses, so
    # live and backtest cannot drift. top_n is the retention band (=1 live).
    ranked = [p[0] for p in top_picks]
    by_sym = {p[0]: p for p in top_picks}     # fyers_symbol -> full pick row
    held_sym = pos.get("open_symbol") if pos else None
    dec = decide_rotation(held_sym, ranked, retain_top_n=top_n)

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals: List[Dict] = []

    if dec.sell:
        # Exit price = today's price if the held name is still in the ranked
        # set; 0.0 otherwise (it dropped out of the universe entirely).
        exit_price = float(by_sym[dec.sell][3]) if dec.sell in by_sym else 0.0
        entry_px = float(pos.get("open_entry_px") or 0) if pos else 0.0
        # Cosmetic label only: a profitable rotation exit reads as TARGET_HIT,
        # a loss as STOP_HIT (there are no real SL/target prices in rotation).
        kind = "TARGET_HIT" if exit_price >= entry_px else "STOP_HIT"
        signals.append({
            "model": MODEL_NAME,
            "universe": "n20_adv_n100",
            "symbol": dec.sell,
            "company": dec.sell,
            "ts": today_str,
            "side": "SELL",
            "signal": kind,
            "price": float(exit_price),
            "sl": 0.0, "target": 0.0,
            "note": f"daily rotation exit (dropped out of top-{top_n})",
        })

    if dec.buy:
        sym, name, ret, price = by_sym[dec.buy]
        signals.append({
            "model": MODEL_NAME,
            "universe": "n20_adv_n100",
            "symbol": sym,
            "company": name,
            "ts": today_str,
            "side": "BUY",
            "signal": "ENTRY1",
            "price": float(price),
            "sl": 0.0, "target": 0.0,
            "note": f"30d momentum rank-1 ({ret:+.2f}%) — N40 ADV∩N100",
        })

    return signals


def main() -> int:
    """CLI entrypoint: build today's ranking, emit signals, log side effects.

    Flow: parse args -> gate on enabled flag + weekday (unless --force) ->
    load N500 OHLCV -> rank -> read held position -> emit signals to the
    --signals-out file -> write the Today's Picks ranking JSON -> audit to DB
    -> send the decision notification (scheduled runs only).

    On any gate miss or hard failure an empty/partial signals file is still
    written so downstream execute_orders has a well-formed file to read.

    Returns:
        int: process exit code — 0 on success or a clean skip, 1 on a hard
        data/config failure (missing Nifty-100 CSV or no OHLCV).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--ranking-out", default=None,
                    help="Where to write the Today's Picks ranking JSON. "
                         "Default = the canonical ranking dir. The admin "
                         "display path passes a /tmp path so a page view never "
                         "clobbers the morning's audited ranking snapshot.")
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--force", action="store_true",
                    help="Bypass weekday + enabled checks")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    log.info(f"{MODEL_NAME} signal run: today={today.date()} "
             f"weekday={today.strftime('%A')}")

    if not args.force and not is_model_enabled():
        log.warning(f"{MODEL_NAME}: model_settings.enabled is False — "
                    "writing empty signals file and exiting.")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        try:
            from src.services.notification_service import notify_skip
            notify_skip(MODEL_NAME, "model disabled")
        except Exception as _ne:
            log.debug(f"notify_skip failed: {_ne}")
        return 0

    if not args.force and not is_weekday(today):
        log.info("Weekend — skipping daily rotation.")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        return 0

    # Build Nifty 100 set (plain symbol form)
    n100 = {s for s, _ in nifty100_symbols()}
    if not n100:
        log.error("Nifty 100 CSV missing — run tools/analysis/download_niftyindices.py")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        return 1
    log.info(f"Nifty 100 set: {len(n100)} symbols")

    # Load OHLCV for full N500 (PIT ranking pool)
    n500_fyers = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]
    log.info(f"Loading N500 OHLCV for {len(n500_fyers)} symbols...")
    df = load_panel(n500_fyers, days_back=400)
    if df.empty:
        log.error("No historical data — abort.")
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        return 1

    ranks = build_pit_universe_and_rank(df, n100)
    log.info(f"PIT-ranked {len(ranks)} stocks (top-{UNIV_SIZE} ADV ∩ N100 + "
             f"uptrend). Top-{args.top_n}:")
    for i, (sym, name, ret, price) in enumerate(ranks[:max(args.top_n, 5)], 1):
        log.info(f"  {i}. {sym:<20} {ret:+7.2f}%  @ ₹{price:.2f}")

    pos = get_current_position()
    log.info(f"Currently held: {pos.get('open_symbol') if pos else 'none'}")

    signals = emit_signals(ranks, pos, args.top_n)
    log.info(f"Emitting {len(signals)} signals")

    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.signals_out, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    log.info(f"Wrote {args.signals_out}")

    # Per-model ranking JSON for Today's Picks UI. Top-5 always written so
    # the picks page works even on weekends / when the model is disabled.
    today_str = today.strftime("%Y-%m-%d")
    ranking_payload = {
        "model": MODEL_NAME,
        "date": today_str,
        "universe_size": len(ranks),
        "top_n": [
            {
                "rank": i + 1,
                "symbol": plain,
                "name": plain,
                "ret_30d_pct": round(ret, 2),
                "price": round(price, 2),
            }
            for i, (_fyers_sym, plain, ret, price) in enumerate(ranks[:5])
        ],
    }
    # --ranking-out overrides the canonical path so display-only runs (admin
    # Today's Picks) route to /tmp and don't clobber the audited snapshot.
    if args.ranking_out:
        ranking_path = Path(args.ranking_out)
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ranking_dir = Path("/app/logs/n20_daily/ranking")
        ranking_dir.mkdir(parents=True, exist_ok=True)
        ranking_path = ranking_dir / f"{today_str}.json"
    ranking_path.write_text(
        json.dumps(ranking_payload, indent=2, default=str)
    )
    log.info(f"Wrote ranking -> {ranking_path}")

    # Audit: persist rankings + signals to DB
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

    # Notification funnel — ping the verdict even on no-change (trading day only).
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
