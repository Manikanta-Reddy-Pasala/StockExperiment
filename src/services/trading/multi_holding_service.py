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
    _normalize_symbol, _compute_real_charges, _ledger_dict,
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
                     fyers_order_id: str = None, product: str = "CNC") -> Dict:
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
        s.add(ModelTrade(model_name=model_name, side="BUY", symbol=norm, qty=qty,
                         price=price_d, value=cost, reason="ENTRY",
                         fyers_order_id=fyers_order_id))
        log.info(f"{model_name}: BUY {qty}x{norm}@{price} (multi-holding)")
        return _ledger_dict(l)


def record_sell_multi(model_name: str, symbol: str, exit_price: float,
                      reason: str = "ROTATE", fyers_order_id: str = None,
                      product: str = "CNC") -> Dict:
    """Record a SELL of one held symbol for a multi-holding model.

    cash += proceeds - charges; realized_pnl/wins/losses updated; model_holdings
    row removed; ModelTrade logged with pnl. Mirrors record_sell's accounting.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        norm = _normalize_symbol(symbol)
        h = s.query(ModelHolding).filter_by(model_name=model_name, symbol=norm).first()
        if not h:
            raise ValueError(f"{model_name}: not holding {norm}, cannot sell")
        qty = int(h.qty); entry = Decimal(str(h.entry_px))
        price_d = Decimal(str(exit_price)); qty_d = Decimal(str(qty))
        charges = _compute_real_charges("SELL", qty, exit_price, product)
        proceeds = qty_d * price_d - charges
        pnl = proceeds - (qty_d * entry)
        l.cash = l.cash + proceeds
        l.realized_pnl = (l.realized_pnl or Decimal(0)) + pnl
        l.total_trades = (l.total_trades or 0) + 1
        if pnl > 0:
            l.wins = (l.wins or 0) + 1
        else:
            l.losses = (l.losses or 0) + 1
        settings.current_amount = (settings.current_amount or Decimal(0)) - charges
        s.add(ModelTrade(model_name=model_name, side="SELL", symbol=norm, qty=qty,
                         price=price_d, value=proceeds, pnl=pnl, reason=reason,
                         fyers_order_id=fyers_order_id))
        s.delete(h)
        log.info(f"{model_name}: SELL {qty}x{norm}@{exit_price} pnl={float(pnl):.0f} ({reason})")
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
