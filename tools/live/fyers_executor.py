"""Live Fyers executor — places real orders. Off by default.

Reads signals JSON. Applies risk_manager. Places Fyers LIMIT orders with
tolerance and a MARKET fallback. Writes order log + reconciles status.

Pass-ordering (Phase 2 Task 5):
  Pass 1  — All EXIT signals (STOP_HIT/TARGET_HIT/EXIT) first.
            Each sell is polled to FILLED before any buy proceeds.
            Any unfilled sell ABORTS the run (no phantom buys on phantom cash).
  Pass 2  — Reload `current_amount` from DB (post-sells), rebuild RiskManager,
            then process ENTRY1/ENTRY2 signals with refreshed sizing.

Order-placement (Phase 2 Task 6):
  place_limit_with_fallback() — LIMIT @ last_price ± tol_pct%, re-quote at
  half-window with retry_pct, MARKET as last resort. All knobs are on the
  RiskManager.cfg (LIMIT_TOL_PCT / LIMIT_RETRY_PCT / LIMIT_FALLBACK_S env).

Hard safeguards:
  - Requires LIVE_TRADING=true env var (defaults off)
  - Requires user-id config + valid Fyers access_token in DB
  - Per-trade cap enforced via RiskManager
  - Daily-loss kill-switch via RiskManager
  - Backward-compat: if --model-name missing, falls back to env-based capital

Usage:
  LIVE_TRADING=true python tools/live/fyers_executor.py \
    --signals signals/2026-05-17_n20.json --user-id 1 \
    --model-name n20_daily_large_only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.live.risk_manager import RiskManager, Position  # noqa: E402

log = logging.getLogger("fyers_executor")


# --------- symbol helpers ---------

def to_fyers_symbol(plain: str) -> str:
    s = plain.upper()
    if s.startswith("NSE:"):
        return s
    return f"NSE:{s.replace('.NS', '')}-EQ"


# --------- low-level Fyers wrappers ---------

def _placeorder(svc, user_id: int, symbol: str, qty: int, side: str,
                pricetype: str = "MARKET", price: float = 0.0,
                product: str = "INTRADAY", tag: str = "") -> Dict:
    """Thin wrapper around FyersService.placeorder using the standardized API.

    Returns the response dict (with `status`, `data.orderid` on success).
    Never raises — exceptions are converted to {"status":"error", ...}.
    """
    fyers_sym = to_fyers_symbol(symbol)
    try:
        return svc.placeorder(
            user_id=user_id,
            symbol=fyers_sym,
            quantity=str(int(qty)),
            action=side.upper(),
            product=product,
            pricetype=pricetype.upper(),
            price=str(price) if pricetype.upper() != "MARKET" else "0",
            trigger_price="0",
            disclosed_quantity="0",
            validity="DAY",
            tag=tag,
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _extract_order_id(res: Dict) -> str:
    """Fyers responses vary between raw API ({s:ok,id:...}) and standardized
    wrapper ({status:success,data:{orderid:...}})."""
    if not isinstance(res, dict):
        return ""
    return (
        res.get("id")
        or res.get("orderId")
        or (res.get("data") or {}).get("orderid")
        or (res.get("data") or {}).get("id")
        or ""
    )


def _is_ok(res: Dict) -> bool:
    status = (res or {}).get("status") or (res or {}).get("s") or ""
    return str(status).lower() in ("ok", "success")


def _get_order_status(svc, user_id: int, order_id: str) -> Optional[Dict]:
    """Fetch order status by polling the order book and matching `id`.

    Returns the matched order dict (with `status` int) or None.

    Fyers status codes (per docs):
      1=cancelled, 2=traded/filled, 3=unused, 4=transit, 5=rejected,
      6=pending, 7=expired
    """
    try:
        ob = svc.orderbook(user_id=user_id)
        rows = (ob or {}).get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("orderBook") or rows.get("orders") or []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or row.get("orderId") or "")
            if rid and rid == str(order_id):
                return row
    except Exception as e:
        log.debug(f"orderbook fetch failed: {e}")
    return None


def _wait_for_fill(svc, user_id: int, order_id: str,
                   timeout_s: int = 60, poll_s: float = 2.0) -> Tuple[bool, str]:
    """Poll Fyers order book until status is terminal.

    Returns (filled: bool, terminal_status: str).
    `terminal_status` is one of: 'filled', 'cancelled', 'rejected', 'expired',
    'timeout', 'not_found'.
    """
    if not order_id:
        return False, "not_found"
    deadline = time.time() + timeout_s
    last_status = "unknown"
    while time.time() < deadline:
        row = _get_order_status(svc, user_id, order_id)
        if row:
            st = row.get("status")
            # Map status int → string
            if st == 2:
                return True, "filled"
            if st == 1:
                return False, "cancelled"
            if st == 5:
                return False, "rejected"
            if st == 7:
                return False, "expired"
            last_status = f"code={st}"
        time.sleep(poll_s)
    log.warning(f"_wait_for_fill timeout after {timeout_s}s on order {order_id} (last={last_status})")
    return False, "timeout"


def _cancel_order(svc, user_id: int, order_id: str) -> bool:
    if not order_id:
        return False
    try:
        res = svc.cancelorder(user_id=user_id, orderid=order_id)
        return _is_ok(res)
    except Exception as e:
        log.warning(f"cancel order {order_id} failed: {e}")
        return False


def _modify_order_limit(svc, user_id: int, order_id: str, new_price: float) -> bool:
    """Try modify; if not supported, caller should cancel+replace."""
    try:
        res = svc.modifyorder(user_id=user_id, orderid=order_id, price=str(new_price))
        return _is_ok(res)
    except Exception as e:
        log.debug(f"modifyorder failed ({e}); caller will cancel+replace")
        return False


# --------- LIMIT-with-tolerance + MARKET-fallback ---------

def place_limit_with_fallback(svc, user_id: int, symbol: str, qty: int,
                              side: str, last_price: float, rm_cfg,
                              tag: str = "") -> Dict:
    """Place a LIMIT order with tolerance; widen once; MARKET as last resort.

    Returns dict: {filled, status, order_id, fill_price (best-effort), reason}.

    Timeline (total = `rm_cfg.limit_fallback_s` seconds, default 20s):
      t=0      : place LIMIT @ tol_pct off last_price
      t=~50%   : if unfilled, cancel + replace with retry_pct (wider)
      t=full   : if still unfilled, cancel + place MARKET, log WARN
    """
    tol = float(rm_cfg.limit_tol_pct) / 100.0
    retry = float(rm_cfg.limit_retry_pct) / 100.0
    total_s = int(rm_cfg.limit_fallback_s)
    first_window_s = max(5, total_s // 2)
    second_window_s = max(5, total_s - first_window_s)

    if side.upper() == "BUY":
        first_px = round(last_price * (1.0 + tol), 2)
        retry_px = round(last_price * (1.0 + retry), 2)
    else:
        first_px = round(last_price * (1.0 - tol), 2)
        retry_px = round(last_price * (1.0 - retry), 2)

    log.info(f"  LIMIT {side} {symbol} qty={qty} px={first_px} "
             f"(last={last_price}, tol={rm_cfg.limit_tol_pct}%)")
    res = _placeorder(svc, user_id, symbol, qty, side,
                      pricetype="LIMIT", price=first_px, tag=tag)
    if not _is_ok(res):
        return {"filled": False, "status": "place_failed", "order_id": "",
                "fill_price": None, "reason": str(res)}
    order_id = _extract_order_id(res)
    filled, term = _wait_for_fill(svc, user_id, order_id,
                                   timeout_s=first_window_s, poll_s=2.0)
    if filled:
        return {"filled": True, "status": "limit_filled", "order_id": order_id,
                "fill_price": first_px, "reason": "tol"}
    if term in ("rejected", "cancelled", "expired"):
        # Order is already gone — try a fresh widened LIMIT
        log.warning(f"  LIMIT {symbol} terminal={term}, retry with widened tol")
    else:
        # Still pending — try modify first, cancel+replace as fallback
        if not _modify_order_limit(svc, user_id, order_id, retry_px):
            _cancel_order(svc, user_id, order_id)
            res2 = _placeorder(svc, user_id, symbol, qty, side,
                               pricetype="LIMIT", price=retry_px, tag=tag)
            if _is_ok(res2):
                order_id = _extract_order_id(res2)
            else:
                log.error(f"  LIMIT retry place failed for {symbol}: {res2}")
        else:
            log.info(f"  LIMIT {symbol} modified to {retry_px}")

    log.info(f"  LIMIT {side} {symbol} re-quote px={retry_px} "
             f"(retry_tol={rm_cfg.limit_retry_pct}%)")
    filled, term = _wait_for_fill(svc, user_id, order_id,
                                   timeout_s=second_window_s, poll_s=2.0)
    if filled:
        return {"filled": True, "status": "limit_retry_filled", "order_id": order_id,
                "fill_price": retry_px, "reason": "retry_tol"}

    # MARKET fallback
    _cancel_order(svc, user_id, order_id)
    log.warning(f"  MARKET fallback for {side} {symbol} qty={qty} "
                f"(last={last_price}, prior order={order_id})")
    res3 = _placeorder(svc, user_id, symbol, qty, side, pricetype="MARKET", tag=tag)
    if not _is_ok(res3):
        return {"filled": False, "status": "market_failed", "order_id": "",
                "fill_price": None, "reason": str(res3)}
    market_id = _extract_order_id(res3)
    filled, term = _wait_for_fill(svc, user_id, market_id,
                                   timeout_s=30, poll_s=2.0)
    return {
        "filled": filled,
        "status": "market_filled" if filled else f"market_{term}",
        "order_id": market_id,
        # Fyers fills near LTP for MARKET; best-effort placeholder = last_price
        "fill_price": last_price if filled else None,
        "reason": "market_fallback",
    }


# --------- Ledger hooks ---------

def _record_model_buy(model_name, symbol, qty, price, order_id):
    if not model_name:
        return
    try:
        from src.services.trading.model_ledger_service import record_buy
        record_buy(model_name, symbol, qty, price, fyers_order_id=order_id)
        log.info(f"  ledger: recorded BUY for {model_name}")
    except Exception as e:
        log.warning(f"  ledger record_buy failed for {model_name}: {e}")


def _record_model_sell(model_name, exit_price, reason, order_id):
    if not model_name:
        return
    try:
        from src.services.trading.model_ledger_service import record_sell
        record_sell(model_name, exit_price, reason, fyers_order_id=order_id)
        log.info(f"  ledger: recorded SELL for {model_name}")
    except Exception as e:
        log.warning(f"  ledger record_sell failed for {model_name}: {e}")


# --------- main ---------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--model-name", default=None,
                    help="Per-model ledger routing. If set, capital comes from "
                         "model_settings.current_amount (Phase 2 Task 4) and "
                         "every BUY/SELL is recorded in model_ledger DB.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print orders, don't place. Always on if LIVE_TRADING != true.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    live = os.environ.get("LIVE_TRADING", "false").lower() == "true"
    if not live:
        log.warning("LIVE_TRADING != 'true' — forcing dry-run mode")
        args.dry_run = True

    # ---- Build RiskManager (per-model if provided, else env) ----
    rm = RiskManager.for_model_or_env(args.model_name)
    log.info(f"Risk: model={args.model_name or '(env)'} "
             f"capital=₹{rm.cfg.capital_inr:,} "
             f"max_concurrent={rm.cfg.max_concurrent} live={live} "
             f"dry_run={args.dry_run} "
             f"limit_tol={rm.cfg.limit_tol_pct}% "
             f"retry_tol={rm.cfg.limit_retry_pct}% "
             f"fallback_s={rm.cfg.limit_fallback_s}")

    with open(args.signals) as f:
        signals = json.load(f)

    if not args.dry_run:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        cfg = svc.get_broker_config(args.user_id)
        if not cfg or not cfg.get("access_token"):
            log.error(f"No Fyers token for user_id={args.user_id} — aborting")
            return 2
    else:
        svc = None

    # Canonical ledger paths
    LEDGER_DIR = Path("/app/logs/momrot/ledger")
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_FILE = LEDGER_DIR / "momrot_ledger.json"
    HISTORY_FILE = LEDGER_DIR / "trade_history.jsonl"

    # Load current ledger (open positions)
    if LEDGER_FILE.exists():
        try:
            with open(LEDGER_FILE) as f:
                ledger = json.load(f)
        except Exception:
            ledger = {"open": []}
    else:
        ledger = {"open": []}
    open_positions: List[Position] = []
    for p in ledger.get("open", []):
        open_positions.append(Position(
            symbol=p["symbol"], qty=int(p["qty"]),
            entry_price=float(p["entry_price"]),
            side=p.get("side", "BUY"),
            sl=float(p.get("sl", 0) or 0),
            target=float(p.get("target", 0) or 0),
        ))

    def _save_ledger():
        if args.dry_run:
            return
        with open(LEDGER_FILE, "w") as f:
            json.dump({
                "updated_at": datetime.now().isoformat(),
                "open": [
                    {"symbol": p.symbol, "qty": p.qty,
                     "entry_price": p.entry_price, "side": p.side,
                     "sl": p.sl, "target": p.target,
                     "entry_ts": getattr(p, "entry_ts", None)}
                    for p in open_positions
                ],
            }, f, indent=2, default=str)

    def _append_history(rec: dict):
        if args.dry_run:
            return
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # Partition signals into exits and entries (preserve order within each)
    exit_signals = [s for s in signals
                    if s.get("signal") in ("STOP_HIT", "TARGET_HIT", "EXIT")]
    entry_signals = [s for s in signals
                     if s.get("signal") in ("ENTRY1", "ENTRY2")]
    other_signals = [s for s in signals
                     if s.get("signal") not in (
                         "STOP_HIT", "TARGET_HIT", "EXIT", "ENTRY1", "ENTRY2"
                     )]

    log.info(f"Signal mix: exits={len(exit_signals)} entries={len(entry_signals)} "
             f"other={len(other_signals)}")

    placed = 0
    closed = 0
    skipped = 0

    # ---- PASS 1: EXITS FIRST (block on each fill) ----
    for sig in exit_signals:
        sig_type = sig.get("signal")
        sym = sig["symbol"]
        price = float(sig.get("price") or 0)
        held = next((p for p in open_positions if p.symbol == sym), None)
        if held is None:
            log.info(f"SKIP exit {sym}: not held")
            skipped += 1
            continue
        exit_side = "SELL" if held.side == "BUY" else "BUY"
        order_id = "DRY"
        status = "dry-run"
        fill_price = price

        if args.dry_run:
            log.info(f"DRY-RUN PASS-1 EXIT {sym} qty={held.qty} "
                     f"side={exit_side} @ ~{price} (would LIMIT then MARKET)")
        else:
            res = place_limit_with_fallback(
                svc, args.user_id, sym, held.qty, exit_side,
                last_price=price, rm_cfg=rm.cfg,
                tag=f"exit:{sig_type}",
            )
            status = res.get("status", "unknown")
            order_id = res.get("order_id", "")
            if not res.get("filled"):
                log.error(f"EXIT NOT FILLED {sym}: status={status} reason={res.get('reason')}"
                          f" — ABORTING ENTRIES to avoid phantom-cash buys")
                # Best-effort save of what we did so far, then bail.
                return 3
            fill_price = res.get("fill_price") or price

        pnl = (fill_price - held.entry_price) * held.qty * (
            1 if held.side == "BUY" else -1
        )
        model_for_signal = args.model_name or sig.get("model", "momentum_n100_top5_max1")
        _append_history({
            "ts": datetime.now().isoformat(),
            "event": "EXIT", "reason": sig_type, "pass": 1,
            "symbol": sym, "qty": held.qty,
            "entry_price": held.entry_price, "exit_price": fill_price,
            "pnl": round(pnl, 2),
            "order_id": order_id, "status": status,
            "model": model_for_signal,
        })
        if not args.dry_run:
            _record_model_sell(model_for_signal, fill_price,
                               sig.get("reason", sig_type), order_id)
        open_positions = [p for p in open_positions if p.symbol != sym]
        _save_ledger()
        log.info(f"{'DRY-RUN' if args.dry_run else 'CLOSED'} PASS-1 {sym} qty={held.qty} "
                 f"entry={held.entry_price} exit={fill_price} pnl=₹{pnl:.0f} "
                 f"reason={sig_type}")
        closed += 1

    # ---- After all sells confirmed: rebuild RiskManager from refreshed DB ----
    if closed > 0 and args.model_name and not args.dry_run:
        rm = RiskManager.for_model_or_env(args.model_name)
        log.info(f"Risk RELOADED post-sells: capital=₹{rm.cfg.capital_inr:,} "
                 f"(after {closed} exit fills)")

    # ---- PASS 2: ENTRIES with refreshed sizing ----
    for sig in entry_signals:
        sym = sig["symbol"]
        price = float(sig.get("price") or 0)
        side = sig["side"]
        ok, reason = rm.can_enter(sym, price, side, open_positions)
        if not ok:
            log.info(f"SKIP {sym}: {reason}")
            skipped += 1
            continue
        qty = rm.size_position(price, open_positions)
        if qty < 1:
            log.info(f"SKIP {sym}: qty<1 (capital=₹{rm.cfg.capital_inr:,})")
            skipped += 1
            continue

        order_id = "DRY"
        status = "dry-run"
        fill_price = price

        if args.dry_run:
            log.info(f"DRY-RUN PASS-2 ENTRY {sym} qty={qty} "
                     f"side={side} @ ~{price} (would LIMIT then MARKET)")
        else:
            res = place_limit_with_fallback(
                svc, args.user_id, sym, qty, side,
                last_price=price, rm_cfg=rm.cfg,
                tag=f"entry:{sig.get('signal')}",
            )
            status = res.get("status", "unknown")
            order_id = res.get("order_id", "")
            if not res.get("filled"):
                log.error(f"ENTRY NOT FILLED {sym}: status={status} reason={res.get('reason')}"
                          f" — skipping this symbol, continuing")
                skipped += 1
                continue
            fill_price = res.get("fill_price") or price

        entry_ts = datetime.now().isoformat()
        model_for_signal = args.model_name or sig.get("model", "momentum_n100_top5_max1")
        _append_history({
            "ts": entry_ts, "event": "ENTRY", "signal": sig.get("signal"),
            "pass": 2,
            "symbol": sym, "side": side, "qty": qty, "price": fill_price,
            "sl": sig.get("sl"), "target": sig.get("target"),
            "order_id": order_id, "status": status,
            "model": model_for_signal,
        })
        if not args.dry_run and side == "BUY":
            _record_model_buy(model_for_signal, sym, qty, fill_price, order_id)
        log.info(f"{'DRY-RUN' if args.dry_run else 'PLACED'} PASS-2 {side} {sym} "
                 f"qty={qty} @ {fill_price} status={status}")
        placed += 1
        new_pos = Position(
            symbol=sym, qty=qty, entry_price=fill_price, side=side,
            sl=float(sig.get("sl", 0) or 0),
            target=float(sig.get("target", 0) or 0),
        )
        try:
            object.__setattr__(new_pos, "entry_ts", entry_ts)
        except Exception:
            pass
        open_positions.append(new_pos)
        _save_ledger()

    log.info(f"Done: placed={placed} closed={closed} skipped={skipped} "
             f"(live={live}, dry_run={args.dry_run}, "
             f"model={args.model_name or '(env)'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
