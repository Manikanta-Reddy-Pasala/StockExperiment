"""Reconcile a model's open-position entry_px against actual Fyers fill price.

Use when ledger was written with LIMIT order price (pre-bug-fix) but Fyers
actually filled at a different traded price. Fetches Fyers holdings +
positions, finds the matching open symbol, updates model_ledger.open_entry_px
to the real avg_price and rebalances ledger.cash so the books reconcile.

Safe: read-only by default. Pass --apply to commit.

Usage:
    docker exec trading_system_app python -m tools.live.reconcile_entry_px \
        --model n20_daily_large_only
    docker exec trading_system_app python -m tools.live.reconcile_entry_px \
        --model n20_daily_large_only --apply
"""
from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("reconcile_entry_px")


def _fyers_avg_price(symbol_bare: str, user_id: int = 1):
    """Return (avg_price, qty) from Fyers holdings or positions matching `symbol_bare`.
    Returns (None, None) on any failure or no-match.
    """
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        holdings = (svc.holdings(user_id) or {}).get("data") or []
        for h in holdings:
            sym = (h.get("symbol") or "").upper().replace("NSE:", "").replace("-EQ", "")
            if sym == symbol_bare.upper():
                avg = float(h.get("average_price") or 0)
                qty = int(float(h.get("quantity") or 0))
                if avg > 0 and qty > 0:
                    return avg, qty
        # Fall back to today's CNC positions
        raw = svc._get_api_instance(user_id)._make_request("GET", "positions") or {}
        positions = (raw or {}).get("data") or []
        for p in positions:
            sym = (p.get("symbol") or "").upper().replace("NSE:", "").replace("-EQ", "")
            if sym != symbol_bare.upper():
                continue
            if (p.get("productType") or "").upper() != "CNC":
                continue
            avg = float(p.get("buyAvg") or p.get("netAvg") or 0)
            qty = int(float(p.get("netQty") or 0))
            if avg > 0 and qty > 0:
                return avg, qty
    except Exception as e:
        log.error(f"Fyers fetch failed: {e}")
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_name to reconcile")
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--apply", action="store_true",
                    help="Commit the DB change. Without this, dry-run only.")
    ap.add_argument("--entry-px", type=float, default=None,
                    help="Override Fyers avg_price with this value (use when "
                         "position already squared on Fyers but ledger stuck).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from src.models.database import get_database_manager
    from src.models.model_ledger_models import ModelLedger
    from src.services.audit_service import write_config_change

    db = get_database_manager()

    # Step 1: read snapshot into local vars (no ORM instance held past this).
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=args.model).first()
        if not l:
            log.error(f"No ledger for model={args.model}")
            return 2
        if not l.open_symbol or not l.open_qty:
            log.info(f"{args.model}: no open position — nothing to reconcile")
            return 0
        open_symbol = l.open_symbol
        bare = open_symbol.upper().replace("NSE:", "").replace("-EQ", "")
        old_px = float(l.open_entry_px or 0)
        old_cash = float(l.cash or 0)
        old_qty = int(l.open_qty)

    # Step 2: hit Fyers OUTSIDE the session — FyersService uses its own DB
    # session which can expire attributes on the parent if reused.
    if args.entry_px is not None:
        real_px = float(args.entry_px)
        real_qty = old_qty
        log.info(f"Using user-supplied entry_px={real_px}")
    else:
        real_px, real_qty = _fyers_avg_price(bare, args.user_id)
        if real_px is None:
            log.error(f"Could not find {bare} in Fyers holdings/positions")
            return 3
        log.info(f"Fyers: {bare} avg={real_px} qty={real_qty}")

    if abs(real_px - old_px) < 0.01:
        log.info(f"{args.model}: entry_px already matches Fyers ({old_px:.2f}) — no change")
        return 0

    # Cash adjustment: ledger debited qty*old_px from cash at buy time. Truth
    # is qty*real_px. Diff = qty*(old_px - real_px). When real_px < old_px
    # (over-debited), credit cash by the diff so books match Fyers.
    cash_delta = float(old_qty) * (old_px - real_px)
    new_cash = old_cash + cash_delta

    log.info(f"=== Reconcile plan for {args.model} ===")
    log.info(f"  open_symbol      : {open_symbol}")
    log.info(f"  open_qty         : {old_qty}")
    log.info(f"  open_entry_px    : {old_px:.4f} -> {real_px:.4f}")
    log.info(f"  cost basis       : {old_qty * old_px:,.2f} -> {old_qty * real_px:,.2f}")
    log.info(f"  cash             : {old_cash:,.2f} -> {new_cash:,.2f}")
    log.info(f"  cash delta       : {cash_delta:+,.2f}")

    if not args.apply:
        log.warning("DRY-RUN — no changes committed. Re-run with --apply to persist.")
        return 0

    # Step 3: fresh session for the write.
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=args.model).first()
        if not l or not l.open_symbol:
            log.error(f"{args.model}: ledger state changed between read and write — abort")
            return 4
        l.open_entry_px = Decimal(str(real_px))
        l.cash = Decimal(str(new_cash))
        s.flush()
    try:
        write_config_change(
            model_name=args.model,
            field="open_entry_px",
            old_value=old_px, new_value=real_px,
            reason=f"Reconcile to Fyers avg_price; cash adjusted by {cash_delta:+.2f}",
            changed_by="reconcile_entry_px",
        )
    except Exception:
        pass
    log.info(f"{args.model}: COMMITTED entry_px={real_px:.4f} cash={new_cash:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
