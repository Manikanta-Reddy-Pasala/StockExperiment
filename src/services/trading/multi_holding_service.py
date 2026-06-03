"""Multi-holding ledger service — for K>1 models (e.g. momentum_retest_n500).

Parallel to the single-holding methods in model_ledger_service (record_buy /
record_sell). Those are UNTOUCHED. This module adds buy/sell/read for models that
hold N positions at once, storing them in `model_holdings` (one row per symbol)
while REUSING the shared accounting from model_ledger_service:
  - cash + realized_pnl + win/loss stats live in ModelLedger (shared)
  - every fill is logged to ModelTrade (shared audit log)
  - charges use _compute_real_charges (shared SEBI-rate calc)

So only POSITION STORAGE differs (model_holdings vs ledger.open_symbol). A
single-holding model never calls these; a multi-holding model never uses
record_buy/record_sell.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from src.models.database import get_database_manager
from src.models.model_ledger_models import (
    ModelSettings, ModelLedger, ModelTrade, ModelHolding,
)
from src.services.trading.model_ledger_service import (
    _normalize_symbol, _compute_real_charges, _ledger_dict, _order_seen,
    partial_sell_outcome,
)

log = logging.getLogger(__name__)


def get_holdings(model_name: str) -> List[Dict]:
    """Return this model's open positions (list of {symbol, qty, entry_px, entry_date})."""
    db = get_database_manager()
    with db.get_session() as s:
        rows = s.query(ModelHolding).filter_by(model_name=model_name).all()
        return [{
            "symbol": h.symbol, "qty": int(h.qty),
            "entry_px": float(h.entry_px),
            "entry_date": h.entry_date.isoformat() if h.entry_date else None,
        } for h in rows]


def held_symbols(model_name: str) -> set:
    """Set of normalized symbols currently held by the model."""
    return {h["symbol"] for h in get_holdings(model_name)}


def record_buy_multi(model_name: str, symbol: str, qty: int, price: float,
                     fyers_order_id: str = None, product: str = "CNC",
                     trade_at=None) -> Dict:
    """Record a BUY for a multi-holding model.

    cash -= qty*price + charges; current_amount -= charges; insert (or weighted-
    average accumulate) a model_holdings row; log a ModelTrade. Mirrors
    record_buy's accounting but stores the position in model_holdings.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l or not settings:
            raise ValueError(f"Unknown model: {model_name}")
        if _order_seen(s, model_name, fyers_order_id):
            log.warning(f"{model_name}: multi-BUY order {fyers_order_id} already "
                        f"recorded — skipping duplicate (idempotent).")
            return _ledger_dict(l)
        norm = _normalize_symbol(symbol)
        qty_d, price_d = Decimal(str(qty)), Decimal(str(price))
        charges = _compute_real_charges("BUY", qty, price, product)
        cost = qty_d * price_d + charges
        if l.cash < cost:
            log.warning(f"{model_name}: cash shortfall on multi-BUY {qty}x{norm}"
                        f" (cost {float(cost):.0f}, cash {float(l.cash):.0f}) — ledger absorbs")
        l.cash = l.cash - cost
        h = s.query(ModelHolding).filter_by(model_name=model_name, symbol=norm).first()
        if h:  # weighted-average accumulate (rare: retry/duplicate fill)
            tot = Decimal(str(h.qty)) + qty_d
            h.entry_px = ((Decimal(str(h.qty)) * h.entry_px) + (qty_d * price_d)) / tot
            h.qty = int(tot)
        else:
            s.add(ModelHolding(model_name=model_name, symbol=norm, qty=qty,
                               entry_px=price_d, entry_date=date.today()))
        settings.current_amount = (settings.current_amount or Decimal(0)) - charges
        # trade_at: pass the broker's actual fill time when re-recording a
        # historical fill (reconcile/backfill) so the trade is attributed to the
        # day it ACTUALLY happened, not the re-record time. A late re-record that
        # defaults to now() mis-dates the trade and pollutes today's realized P&L
        # (e.g. a 06-02 round-trip re-recorded at 06-03 01:12 showed as today's
        # P&L on a flat intraday model). None → column default (now) for live
        # fills, which are recorded seconds after the fill — correct.
        _bt = ModelTrade(model_name=model_name, side="BUY", symbol=norm, qty=qty,
                         price=price_d, value=cost, reason="ENTRY",
                         fyers_order_id=fyers_order_id)
        if trade_at is not None:
            _bt.trade_at = trade_at
        s.add(_bt)
        log.info(f"{model_name}: BUY {qty}x{norm}@{price} (multi-holding)")
        return _ledger_dict(l)


def record_sell_multi(model_name: str, symbol: str, exit_price: float,
                      reason: str = "ROTATE", fyers_order_id: str = None,
                      product: str = "CNC", qty: int = None, trade_at=None) -> Dict:
    """Record a SELL of one held symbol for a multi-holding model.

    cash += proceeds - charges; realized_pnl/wins/losses updated; ModelTrade
    logged with pnl. ``qty`` = ACTUAL filled quantity; None / >= held / <=0
    means a full close (holding row deleted). A genuine partial fill sells only
    that many shares and RETAINS the residual holding (no phantom-flat row).
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if _order_seen(s, model_name, fyers_order_id):
            log.warning(f"{model_name}: multi-SELL order {fyers_order_id} already "
                        f"recorded — skipping duplicate (idempotent).")
            return _ledger_dict(l)
        norm = _normalize_symbol(symbol)
        h = s.query(ModelHolding).filter_by(model_name=model_name, symbol=norm).first()
        if not h:
            raise ValueError(f"{model_name}: not holding {norm}, cannot sell")
        sell_qty, remaining_qty, is_full = partial_sell_outcome(int(h.qty), qty)
        entry = Decimal(str(h.entry_px))
        price_d = Decimal(str(exit_price)); qty_d = Decimal(str(sell_qty))
        charges = _compute_real_charges("SELL", sell_qty, exit_price, product)
        proceeds = qty_d * price_d - charges
        pnl = proceeds - (qty_d * entry)
        l.cash = l.cash + proceeds
        l.realized_pnl = (l.realized_pnl or Decimal(0)) + pnl
        l.total_trades = (l.total_trades or 0) + 1
        if pnl > 0:
            l.wins = (l.wins or 0) + 1
        elif pnl < 0:
            l.losses = (l.losses or 0) + 1   # exact break-even counts as neither
        settings.current_amount = (settings.current_amount or Decimal(0)) - charges
        # trade_at: real broker fill time on re-record (see record_buy_multi) so
        # a back-dated round-trip is not mis-attributed to today's realized P&L.
        _st = ModelTrade(model_name=model_name, side="SELL", symbol=norm, qty=sell_qty,
                         price=price_d, value=proceeds, pnl=pnl, reason=reason,
                         fyers_order_id=fyers_order_id)
        if trade_at is not None:
            _st.trade_at = trade_at
        s.add(_st)
        if is_full:
            s.delete(h)
        else:
            h.qty = remaining_qty   # partial — keep residual holding
            log.warning(f"{model_name}: PARTIAL SELL {sell_qty}/{sell_qty + remaining_qty} "
                        f"{norm} — retaining {remaining_qty} shares.")
        log.info(f"{model_name}: SELL {sell_qty}x{norm}@{exit_price} pnl={float(pnl):.0f} ({reason})")
        return _ledger_dict(l)


def mtm_nav(model_name: str, price_lookup) -> float:
    """Mark-to-market NAV = cash + sum(qty * current price) over holdings.

    Args:
        price_lookup: callable(symbol) -> float current price (or None).
    """
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        cash = float(l.cash) if l else 0.0
        nav = cash
        for h in s.query(ModelHolding).filter_by(model_name=model_name).all():
            px = price_lookup(h.symbol)
            if px:
                nav += int(h.qty) * float(px)
        return nav
