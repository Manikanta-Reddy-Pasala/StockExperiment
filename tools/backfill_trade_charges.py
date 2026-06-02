"""Backfill approx broker charges onto EXISTING model_trades rows.

New trades are auto-stamped by the ModelTrade before_insert listener; this
one-shot fills the charges_inr column for trades that predate that listener
(or the column). Idempotent: by default only touches rows where charges_inr IS
NULL. Pass --all to recompute every row (e.g. after a rate change in
tools/live/broker_charges).

Charge per leg = compute_charges(side, qty, price, product) where product is the
model's policy product (ORB intraday, the rest delivery). DEPOSIT/WITHDRAW = 0.

Run (inside the app container):
  python tools/backfill_trade_charges.py [--all] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.database import get_database_manager           # noqa: E402
from src.models.model_ledger_models import ModelTrade          # noqa: E402
from src.services.trading.model_ledger_service import product_for_model  # noqa: E402
from tools.live.broker_charges import compute_charges          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Recompute every row (default: only NULL charges_inr).")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    db = get_database_manager()
    updated = 0
    total = 0
    by_model: dict[str, float] = {}
    with db.get_session() as s:
        q = s.query(ModelTrade)
        if not a.all:
            q = q.filter(ModelTrade.charges_inr.is_(None))
        rows = q.all()
        for t in rows:
            total += 1
            side = (t.side or "").upper()
            if side in ("BUY", "SELL"):
                try:
                    prod = product_for_model(t.model_name)
                    charge = compute_charges(side, int(t.qty or 0),
                                             float(t.price or 0), prod).get("total", 0.0)
                except Exception:
                    charge = 0.0
            else:
                charge = 0.0
            by_model[t.model_name] = by_model.get(t.model_name, 0.0) + float(charge)
            if not a.dry_run:
                t.charges_inr = charge
            updated += 1
        if not a.dry_run:
            s.commit()

    print(f"{'[DRY-RUN] ' if a.dry_run else ''}backfilled {updated}/{total} trade rows")
    for m, c in sorted(by_model.items()):
        print(f"  {m:<32} ₹{round(c, 2)}")
    print(f"  {'TOTAL (all models)':<32} ₹{round(sum(by_model.values()), 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
