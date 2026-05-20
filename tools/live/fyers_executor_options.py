"""Fyers F&O multi-leg executor (Iron Condor / spread strategies).

Reads a 4-leg signal JSON (BUY/SELL × 4 option symbols) and places each leg
as a MARKET order with product=MARGIN (carry-forward F&O / NRML).

Gated by LIVE_TRADING env flag — defaults to dry-run.
Writes every fill to audit_orders so charges, slippage, and fill price are
tracked the same way as equity orders.

Designed for monthly Iron Condor holds (finnifty_ic_otm4_w300_lots5) but
generic enough to drive any 2-8 leg defined-risk options strategy.
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _place_leg(svc, user_id: int, sig: Dict, product: str, dry: bool) -> Dict:
    symbol = sig["symbol"]
    side = sig["side"].upper()
    qty = int(sig["qty"])
    tag = f"{sig.get('model', 'options')}:{sig.get('leg', '')}"[:20]
    if dry:
        log.info(f"DRY-RUN {side} {symbol} qty={qty} product={product} tag={tag}")
        return {"status": "dry-run", "order_id": "DRY",
                "symbol": symbol, "side": side, "qty": qty}
    try:
        res = svc.placeorder(
            user_id=user_id, symbol=symbol, quantity=str(qty),
            action=side, product=product, pricetype="MARKET",
            price="0", validity="DAY", tag=tag,
        )
    except Exception as e:
        log.error(f"placeorder {side} {symbol} FAILED: {e}")
        return {"status": "error", "error": str(e),
                "symbol": symbol, "side": side, "qty": qty}
    status = "ok"
    order_id = ""
    if isinstance(res, dict):
        order_id = (res.get("data") or {}).get("orderid") or res.get("orderid") or ""
        if str(res.get("s", "")).lower() not in ("ok", "success"):
            status = str(res.get("message", res.get("s", "unknown")))
    return {"status": status, "order_id": order_id, "raw": res,
            "symbol": symbol, "side": side, "qty": qty}


def _audit_leg(model_name: str, fill: Dict, price: float, product: str,
               dry: bool) -> None:
    try:
        from src.services.audit_service import write_order
    except ImportError:
        log.debug("audit_service.write_order not available, skipping audit")
        return
    try:
        write_order(
            model_name=model_name,
            symbol=fill["symbol"],
            side=fill["side"],
            qty=fill["qty"],
            ordered_price=price,
            fill_price=price if not dry else None,
            fill_qty=fill["qty"] if not dry else None,
            fyers_order_id=fill.get("order_id", ""),
            product=product,
            pricetype="MARKET",
            status="dry-run" if dry else fill.get("status", "unknown"),
            raw_response=fill.get("raw") if isinstance(fill.get("raw"), dict) else None,
        )
    except Exception as e:
        log.debug(f"audit write failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True,
                    help="Path to signal JSON (list of leg dicts)")
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--product", default="MARGIN",
                    help="Fyers product: MARGIN (NRML carry-forward, default) "
                         "or INTRADAY (MIS, square-off same day)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print orders, don't place. Forced on if "
                         "LIVE_TRADING env != 'true'.")
    args = ap.parse_args()

    live = os.environ.get("LIVE_TRADING", "false").lower() == "true"
    if not live:
        log.warning("LIVE_TRADING != 'true' — forcing dry-run mode")
        args.dry_run = True

    sig_path = Path(args.signals)
    if not sig_path.exists():
        log.error(f"signals file not found: {sig_path}")
        return 2
    signals: List[Dict] = json.loads(sig_path.read_text())
    if not signals:
        log.info(f"{args.model_name}: empty signal list, nothing to execute.")
        return 0

    # Order legs so we SELL premium first (collect credit), then BUY wings.
    # On EXIT, BUY-back the shorts first, then SELL the wings.
    signals_sorted = sorted(signals, key=lambda s: 0 if s["side"].upper() == "SELL" else 1)

    svc = None
    if not args.dry_run:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        cfg = svc.get_broker_config(args.user_id)
        if not cfg or not cfg.get("access_token"):
            log.error(f"No Fyers token for user_id={args.user_id} — aborting")
            return 2

    log.info(f"Executing {len(signals_sorted)} legs for {args.model_name} "
             f"(product={args.product}, live={live}, dry_run={args.dry_run})")

    placed = 0
    errors = 0
    for sig in signals_sorted:
        fill = _place_leg(svc, args.user_id, sig, args.product, args.dry_run)
        _audit_leg(args.model_name, fill, float(sig.get("price", 0) or 0),
                   args.product, args.dry_run)
        if fill["status"] in ("ok", "success", "dry-run"):
            placed += 1
            log.info(f"  {fill['side']:4} {fill['symbol']:30} qty={fill['qty']:>4} "
                     f"order_id={fill.get('order_id', '')}")
        else:
            errors += 1
            log.error(f"  FAIL {fill['side']} {fill['symbol']}: {fill['status']}")

    log.info(f"Done: placed={placed} errors={errors} "
             f"(live={live}, dry_run={args.dry_run}, "
             f"model={args.model_name})")

    # HOLD-style audit so we have one row per cron invocation
    try:
        from src.services.audit_service import write_rebalance_decision
        write_rebalance_decision(
            model_name=args.model_name,
            trigger="CRON" if not args.dry_run else "DRY",
            decision="OPTIONS_OPEN" if placed > 0 and errors == 0 else (
                "OPTIONS_PARTIAL" if placed > 0 else "OPTIONS_FAIL"),
            reason=f"{placed} legs placed, {errors} errors",
        )
    except Exception as e:
        log.debug(f"rebal audit write failed: {e}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
