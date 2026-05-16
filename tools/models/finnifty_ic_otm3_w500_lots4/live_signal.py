"""finnifty_ic_otm3_w500_lots4 — live signal for monthly Iron Condor.

Strategy rules:
  - Underlying: FINNIFTY
  - Trade window: Monday of each new monthly expiry cycle
  - One trade per monthly expiry (no overlap)
  - Sell OTM 3% CE + OTM 3% PE
  - Buy wings ±500 points OTM further
  - Lots: 4
  - Stop: combined pair value >= 3x entry net credit → EXIT all 4 legs
  - Else hold to expiry day, settle intrinsic
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine

log = logging.getLogger("finnifty_ic_otm3_signal")

MODEL_NAME = "finnifty_ic_otm3_w500_lots4"
UNDERLYING = "FINNIFTY"
SPOT_SYM = "NSE:FINNIFTY-INDEX"
OTM_PCT = 3.0
WING_WIDTH = 500
STOP_MULT = 3.0
LOTS = 4
STRIKE_STEP = 50
LOT_SIZE = 65  # post Sep 2024


def get_spot_close(d: date) -> Optional[float]:
    eng = _get_engine()
    with eng.connect() as c:
        row = c.execute(text(
            "SELECT close FROM historical_data "
            "WHERE symbol=:s AND date <= :d ORDER BY date DESC LIMIT 1"
        ), {"s": SPOT_SYM, "d": d}).fetchone()
    return float(row.close) if row else None


def next_monthly_expiry(after: date) -> Optional[date]:
    eng = _get_engine()
    with eng.connect() as c:
        row = c.execute(text(
            "SELECT MIN(expiry) FROM option_universe "
            "WHERE underlying=:u AND expiry > :d AND expiry_kind='monthly'"
        ), {"u": UNDERLYING, "d": after}).fetchone()
    return row[0] if row and row[0] else None


def find_strike(exp: date, target: int, opt_type: str) -> Optional[Dict]:
    eng = _get_engine()
    with eng.connect() as c:
        row = c.execute(text(
            "SELECT symbol, strike FROM option_universe "
            "WHERE underlying=:u AND expiry=:e AND opt_type=:o "
            "ORDER BY ABS(strike - :t) LIMIT 1"
        ), {"u": UNDERLYING, "e": exp, "o": opt_type, "t": target}).fetchone()
    if not row:
        return None
    return {"symbol": row.symbol, "strike": int(row.strike)}


def get_option_last(symbol: str, on: date) -> Optional[float]:
    eng = _get_engine()
    with eng.connect() as c:
        row = c.execute(text(
            "SELECT close FROM historical_options "
            "WHERE symbol=:s AND candle_time::date <= :d AND interval='D' "
            "ORDER BY candle_time DESC LIMIT 1"
        ), {"s": symbol, "d": on}).fetchone()
    return float(row.close) if row else None


def round_strike(p: float) -> int:
    return int(round(p / STRIKE_STEP) * STRIKE_STEP)


def get_current_position() -> Optional[Dict]:
    try:
        from src.services.trading.model_ledger_service import get_ledger
        l = get_ledger(MODEL_NAME)
        return l if l and l.get("open_symbol") else None
    except Exception:
        return None


def get_open_ic_legs() -> Optional[Dict]:
    """Read 4-leg IC state from a small JSON sidecar (model_ledger only
    tracks 1 symbol). Sidecar holds the 4 leg symbols + entry credit."""
    p = Path(f"/app/logs/{MODEL_NAME}/state.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_open_ic_legs(state: Dict) -> None:
    p = Path(f"/app/logs/{MODEL_NAME}/state.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, default=str))


def clear_open_ic_legs() -> None:
    p = Path(f"/app/logs/{MODEL_NAME}/state.json")
    if p.exists():
        p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    today = date.today()
    signals: List[Dict] = []

    legs = get_open_ic_legs()

    if legs:
        # Already holding 4-leg IC. Check exit conditions.
        log.info(f"Open IC: {legs.get('label')} entry credit={legs.get('net_credit')}")
        exp = datetime.fromisoformat(legs["expiry"]).date()
        ce_px = get_option_last(legs["ce_sym"], today) or 0
        pe_px = get_option_last(legs["pe_sym"], today) or 0
        wce_px = get_option_last(legs["wce_sym"], today) or 0
        wpe_px = get_option_last(legs["wpe_sym"], today) or 0
        pair_value = (ce_px + pe_px) - (wce_px + wpe_px)
        net_credit = float(legs["net_credit"])

        if pair_value >= net_credit * STOP_MULT:
            reason = "STOP"
            log.info(f"STOP triggered: pair_value={pair_value:.2f} >= "
                     f"{STOP_MULT}x credit={net_credit:.2f}")
        elif today >= exp:
            reason = "EXPIRY"
            log.info(f"EXPIRY reached: today={today} >= exp={exp}")
        else:
            log.info(f"Holding IC. pair_value={pair_value:.2f} vs "
                     f"stop={net_credit*STOP_MULT:.2f}, days_to_exp="
                     f"{(exp-today).days}")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.signals_out).write_text(json.dumps([]))
            return 0

        # Emit 4 EXIT signals (buy-back the 2 sold + sell the 2 wings)
        for label, sym in [("ce", legs["ce_sym"]), ("pe", legs["pe_sym"])]:
            signals.append({
                "signal": "STOP_HIT" if reason == "STOP" else "EXPIRY",
                "symbol": sym, "side": "BUY",  # buy-back the sold leg
                "qty": LOTS * LOT_SIZE, "reason": reason,
                "model": MODEL_NAME, "leg": label,
            })
        for label, sym in [("wce", legs["wce_sym"]), ("wpe", legs["wpe_sym"])]:
            signals.append({
                "signal": "STOP_HIT" if reason == "STOP" else "EXPIRY",
                "symbol": sym, "side": "SELL",  # sell the long wing
                "qty": LOTS * LOT_SIZE, "reason": reason,
                "model": MODEL_NAME, "leg": label,
            })
        clear_open_ic_legs()

    else:
        # Flat. Check if today is a Monday + new monthly cycle.
        if today.weekday() != 0:  # Monday only
            log.info(f"Not Monday ({today.strftime('%A')}), skipping entry scan")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.signals_out).write_text(json.dumps([]))
            return 0

        exp = next_monthly_expiry(today)
        if not exp:
            log.warning("No upcoming monthly expiry found in option_universe")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.signals_out).write_text(json.dumps([]))
            return 0

        # Only enter if expiry is within current month window (Mon of new cycle)
        days_to_exp = (exp - today).days
        if days_to_exp < 10 or days_to_exp > 35:
            log.info(f"Expiry {exp} ({days_to_exp}d) outside typical entry "
                     f"window 10-35d, skipping")
            Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.signals_out).write_text(json.dumps([]))
            return 0

        spot = get_spot_close(today)
        if not spot:
            log.error("No spot close for FINNIFTY")
            return 1
        ce_k = round_strike(spot * (1 + OTM_PCT / 100))
        pe_k = round_strike(spot * (1 - OTM_PCT / 100))
        wce_k = ce_k + WING_WIDTH
        wpe_k = pe_k - WING_WIDTH

        ce = find_strike(exp, ce_k, "CE")
        pe = find_strike(exp, pe_k, "PE")
        wce = find_strike(exp, wce_k, "CE")
        wpe = find_strike(exp, wpe_k, "PE")
        if not all([ce, pe, wce, wpe]):
            log.warning(f"Missing strike(s) ce={ce} pe={pe} wce={wce} wpe={wpe}")
            return 1

        ce_px = get_option_last(ce["symbol"], today) or 0
        pe_px = get_option_last(pe["symbol"], today) or 0
        wce_px = get_option_last(wce["symbol"], today) or 0
        wpe_px = get_option_last(wpe["symbol"], today) or 0
        net_credit = (ce_px + pe_px) - (wce_px + wpe_px)
        if net_credit <= 0:
            log.warning(f"Net credit non-positive ({net_credit:.2f}), skip entry")
            return 0

        log.info(f"ENTRY: spot={spot:.2f} exp={exp} "
                 f"CE={ce_k}({ce['symbol']}) PE={pe_k}({pe['symbol']}) "
                 f"net_credit={net_credit:.2f} max_loss="
                 f"{(WING_WIDTH - net_credit) * LOT_SIZE * LOTS:.0f}")

        # Emit 4 ENTRY signals
        signals.append({
            "signal": "ENTRY1", "symbol": ce["symbol"], "side": "SELL",
            "qty": LOTS * LOT_SIZE, "price": ce_px,
            "model": MODEL_NAME, "leg": "ce_short",
        })
        signals.append({
            "signal": "ENTRY1", "symbol": pe["symbol"], "side": "SELL",
            "qty": LOTS * LOT_SIZE, "price": pe_px,
            "model": MODEL_NAME, "leg": "pe_short",
        })
        signals.append({
            "signal": "ENTRY1", "symbol": wce["symbol"], "side": "BUY",
            "qty": LOTS * LOT_SIZE, "price": wce_px,
            "model": MODEL_NAME, "leg": "wce_long",
        })
        signals.append({
            "signal": "ENTRY1", "symbol": wpe["symbol"], "side": "BUY",
            "qty": LOTS * LOT_SIZE, "price": wpe_px,
            "model": MODEL_NAME, "leg": "wpe_long",
        })
        save_open_ic_legs({
            "label": f"{UNDERLYING} IC OTM3 w500",
            "expiry": exp.isoformat(),
            "ce_sym": ce["symbol"], "pe_sym": pe["symbol"],
            "wce_sym": wce["symbol"], "wpe_sym": wpe["symbol"],
            "ce_k": ce_k, "pe_k": pe_k, "wce_k": wce_k, "wpe_k": wpe_k,
            "ce_px": ce_px, "pe_px": pe_px,
            "wce_px": wce_px, "wpe_px": wpe_px,
            "net_credit": net_credit,
            "lots": LOTS, "lot_size": LOT_SIZE,
            "entry_date": today.isoformat(),
        })

    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.signals_out).write_text(json.dumps(signals, indent=2, default=str))
    log.info(f"Wrote {len(signals)} signal(s) to {args.signals_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
