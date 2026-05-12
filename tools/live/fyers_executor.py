"""Live Fyers executor — places real orders. Off by default.

Reads signals JSON. Applies risk_manager. Places Fyers market orders.
Writes order log + reconciles status.

Hard safeguards:
  - Requires LIVE_TRADING=true env var (defaults off)
  - Requires user-id config + valid Fyers access_token in DB
  - Refuses to place > max_per_trade_inr
  - Refuses if daily-loss kill-switch triggered

Usage:
  LIVE_TRADING=true python tools/live/fyers_executor.py \
    --signals signals/2026-05-12_ema_200_400_nifty50.json --user-id 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.live.risk_manager import RiskManager, Position  # noqa: E402

log = logging.getLogger("fyers_executor")


def to_fyers_symbol(plain: str) -> str:
    s = plain.upper()
    if s.startswith("NSE:"):
        return s
    return f"NSE:{s.replace('.NS', '')}-EQ"


def place_fyers_order(svc, user_id: int, symbol: str, qty: int,
                      side: str, order_type: str = "MARKET") -> Dict:
    """Place a market order via Fyers. side=BUY|SELL."""
    fyers_sym = to_fyers_symbol(symbol)
    payload = {
        "symbol": fyers_sym,
        "qty": qty,
        "type": 2,            # 2 = Market
        "side": 1 if side == "BUY" else -1,
        "productType": "INTRADAY",   # change to CNC for delivery
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
    }
    try:
        return svc.place_order(user_id=user_id, order=payload)
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print orders, don't place. Always on if LIVE_TRADING != true.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    live = os.environ.get("LIVE_TRADING", "false").lower() == "true"
    if not live:
        log.warning("LIVE_TRADING != 'true' — forcing dry-run mode")
        args.dry_run = True

    rm = RiskManager.from_env()
    log.info(f"Risk: capital=₹{rm.cfg.capital_inr:,} "
             f"max_concurrent={rm.cfg.max_concurrent} live={live} "
             f"dry_run={args.dry_run}")

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

    out_log = ROOT / "order_log" / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    # Use empty open_positions for simplicity; production should track real positions
    open_positions: List[Position] = []
    placed = 0
    skipped = 0

    for sig in signals:
        if sig["signal"] not in ("ENTRY1", "ENTRY2"):
            continue
        sym = sig["symbol"]
        side = sig["side"]
        price = float(sig["price"])
        ok, reason = rm.can_enter(sym, price, side, open_positions)
        if not ok:
            log.info(f"SKIP {sym}: {reason}")
            skipped += 1
            continue
        qty = rm.size_position(price, open_positions)
        if qty < 1:
            log.info(f"SKIP {sym}: qty<1")
            skipped += 1
            continue

        order_id = "DRY"
        status = "dry-run"
        if not args.dry_run:
            res = place_fyers_order(svc, args.user_id, sym, qty, side)
            status = res.get("s") or res.get("status", "unknown")
            order_id = res.get("id") or res.get("orderId", "")
            if status not in ("ok", "success"):
                log.error(f"ORDER FAIL {sym}: {res}")
                continue

        rec = {
            "ts": datetime.now().isoformat(),
            "symbol": sym, "side": side, "qty": qty, "price": price,
            "sl": sig.get("sl"), "target": sig.get("target"),
            "order_id": order_id, "status": status,
        }
        with open(out_log, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log.info(f"{'DRY-RUN' if args.dry_run else 'PLACED'} {side} {sym} "
                 f"qty={qty} @ {price} status={status}")
        placed += 1
        open_positions.append(Position(
            symbol=sym, qty=qty, entry_price=price, side=side,
            sl=float(sig.get("sl", 0)), target=float(sig.get("target", 0)),
        ))

    log.info(f"Done: placed={placed} skipped={skipped} (live={live}, dry_run={args.dry_run})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
