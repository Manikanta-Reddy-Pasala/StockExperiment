"""Per-model capital ledger service.

Business logic owning:
  - settings (allocated capital, enabled flag) for each trading model
  - cash + open-position state for each model
  - trade audit log
  - portfolio aggregate stats

Used by live executors (route every buy/sell through ledger), admin/settings
UI (display + edit), and dashboard (per-model + portfolio totals).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import text

from src.models.database import get_database_manager
from src.models.model_ledger_models import (
    ModelLedger, ModelSettings, ModelTrade,
)

log = logging.getLogger(__name__)

# Models the system knows about. Adding a 3rd later = append here +
# the settings UI creates the row when user fills the form.
KNOWN_MODELS = [
    {
        "name": "momentum_n100_top5_max1",
        "default_capital": 0,
        "description": "Equity monthly rotation top-1 from N100 by 60d return",
    },
    {
        "name": "midcap_narrow_60d_breakout",
        "default_capital": 0,
        "description": "Equity 60d-high swing on midcap_narrow (event-driven)",
    },
]


def ensure_models_seeded() -> None:
    """Create settings+ledger rows for any KNOWN_MODELS missing from DB.

    Safe to call repeatedly. Allocated capital defaults to the value above
    but user can edit via settings UI without losing trade history.
    """
    db = get_database_manager()
    with db.get_session() as s:
        for m in KNOWN_MODELS:
            existing = s.query(ModelSettings).filter_by(model_name=m["name"]).first()
            if existing:
                continue
            s.add(ModelSettings(
                model_name=m["name"],
                enabled=True,
                allocated_capital=Decimal(m["default_capital"]),
                description=m["description"],
            ))
            # Flush so settings row exists before ledger FK
            s.flush()
            s.add(ModelLedger(
                model_name=m["name"],
                cash=Decimal(m["default_capital"]),
                realized_pnl=Decimal(0),
                total_trades=0,
                wins=0,
                losses=0,
            ))
            s.flush()
            log.info(f"Seeded ledger for {m['name']} cap={m['default_capital']}")


# ---- Settings ----

def get_all_settings() -> List[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        rows = s.query(ModelSettings).order_by(ModelSettings.model_name).all()
        return [_settings_dict(r) for r in rows]


def deposit(model_name: str, amount: float) -> Dict:
    """Add fresh capital to a model. Adds to both allocated_capital AND cash.

    Use case: monthly top-up from user's bank. Money flows in as new cash
    and is added to the cost basis. Other models untouched.
    """
    if amount <= 0:
        raise ValueError("deposit amount must be > 0")
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        delta = Decimal(str(amount))
        settings.allocated_capital = (settings.allocated_capital or Decimal(0)) + delta
        ledger.cash = (ledger.cash or Decimal(0)) + delta
        s.add(ModelTrade(
            model_name=model_name,
            side="DEPOSIT",
            symbol="-",
            qty=0,
            price=Decimal(0),
            value=delta,
            reason="DEPOSIT",
        ))
        log.info(f"{model_name}: deposit ₹{amount:,.0f} (allocated now "
                 f"₹{float(settings.allocated_capital):,.0f}, cash "
                 f"₹{float(ledger.cash):,.0f})")
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
        }


def withdraw(model_name: str, amount: float) -> Dict:
    """Pull cash out of a model. Decreases allocated_capital AND cash.

    Safety: refuses if cash < amount.
    """
    if amount <= 0:
        raise ValueError("withdraw amount must be > 0")
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        delta = Decimal(str(amount))
        if ledger.cash < delta:
            raise ValueError(
                f"Insufficient cash in {model_name}: have ₹{float(ledger.cash):,.0f}, "
                f"want to withdraw ₹{amount:,.0f}"
            )
        ledger.cash = ledger.cash - delta
        settings.allocated_capital = max(
            Decimal(0), (settings.allocated_capital or Decimal(0)) - delta
        )
        s.add(ModelTrade(
            model_name=model_name,
            side="WITHDRAW",
            symbol="-",
            qty=0,
            price=Decimal(0),
            value=delta,
            reason="WITHDRAW",
        ))
        log.info(f"{model_name}: withdraw ₹{amount:,.0f} (allocated now "
                 f"₹{float(settings.allocated_capital):,.0f}, cash "
                 f"₹{float(ledger.cash):,.0f})")
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
        }


def reset_model(model_name: str) -> Dict:
    """Hard reset: zero allocated_capital, zero cash, zero realized_pnl, clear
    open position, reset counters. Trade audit log is NOT deleted (history kept).
    Use to start fresh before depositing real cost basis.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not settings or not ledger:
            raise ValueError(f"Unknown model: {model_name}")
        settings.allocated_capital = Decimal(0)
        ledger.cash = Decimal(0)
        ledger.realized_pnl = Decimal(0)
        ledger.total_trades = 0
        ledger.wins = 0
        ledger.losses = 0
        ledger.open_symbol = None
        ledger.open_qty = None
        ledger.open_entry_px = None
        ledger.open_entry_date = None
        log.info(f"{model_name}: model reset to zero")
        return {
            "settings": _settings_dict(settings),
            "ledger": _ledger_dict(ledger),
        }


def set_enabled(model_name: str, enabled: bool) -> Dict:
    db = get_database_manager()
    with db.get_session() as s:
        settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
        if not settings:
            raise ValueError(f"Unknown model: {model_name}")
        settings.enabled = enabled
        return _settings_dict(settings)


# ---- Ledger snapshot ----

def get_ledger(model_name: str) -> Optional[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l:
            return None
        return _ledger_dict(l)


def get_all_ledgers() -> List[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        rows = s.query(ModelLedger).order_by(ModelLedger.model_name).all()
        return [_ledger_dict(r) for r in rows]


# ---- Bootstrap an existing position (e.g. momentum_n100 already live) ----

def seed_position(model_name: str, symbol: str, qty: int,
                  entry_px: float, entry_date_str: str) -> Dict:
    """Manually seed a model's open position (no Fyers order placed)."""
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l:
            raise ValueError(f"Unknown model: {model_name}")
        if l.open_symbol:
            raise ValueError(
                f"{model_name} already has open position {l.open_symbol}, "
                f"reset first"
            )
        cost = Decimal(str(qty)) * Decimal(str(entry_px))
        if l.cash < cost:
            raise ValueError(
                f"Not enough cash in {model_name} ledger "
                f"(₹{float(l.cash):,.0f}) to seed position cost ₹{float(cost):,.0f}"
            )
        l.open_symbol = symbol
        l.open_qty = qty
        l.open_entry_px = Decimal(str(entry_px))
        l.open_entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        l.cash = l.cash - cost
        s.add(ModelTrade(
            model_name=model_name,
            side="BUY",
            symbol=symbol,
            qty=qty,
            price=Decimal(str(entry_px)),
            value=cost,
            reason="SEED",
        ))
        return _ledger_dict(l)


def reset_position(model_name: str) -> Dict:
    """Mark position as flat, returning estimated NAV back to cash.

    DOES NOT place any Fyers order — manual reconciliation tool only.
    """
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l or not l.open_symbol:
            return _ledger_dict(l) if l else None
        cost = Decimal(str(l.open_qty)) * l.open_entry_px
        l.cash = l.cash + cost
        l.open_symbol = None
        l.open_qty = None
        l.open_entry_px = None
        l.open_entry_date = None
        return _ledger_dict(l)


# ---- Live buy/sell hooks (called by executor) ----

def record_buy(model_name: str, symbol: str, qty: int, price: float,
               brokerage: float = 20, fyers_order_id: str = None) -> Dict:
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l:
            raise ValueError(f"Unknown model: {model_name}")
        if l.open_symbol:
            raise ValueError(f"{model_name}: already holding {l.open_symbol}")
        cost = Decimal(str(qty)) * Decimal(str(price)) + Decimal(str(brokerage))
        if l.cash < cost:
            raise ValueError(
                f"{model_name}: cash {float(l.cash):,.0f} < cost {float(cost):,.0f}"
            )
        l.cash = l.cash - cost
        l.open_symbol = symbol
        l.open_qty = qty
        l.open_entry_px = Decimal(str(price))
        l.open_entry_date = date.today()
        s.add(ModelTrade(
            model_name=model_name,
            side="BUY",
            symbol=symbol,
            qty=qty,
            price=Decimal(str(price)),
            value=cost,
            reason="ENTRY",
            fyers_order_id=fyers_order_id,
        ))
        return _ledger_dict(l)


def record_sell(model_name: str, exit_price: float, reason: str,
                brokerage: float = 20, stt_pct: float = 0.001,
                fyers_order_id: str = None) -> Dict:
    db = get_database_manager()
    with db.get_session() as s:
        l = s.query(ModelLedger).filter_by(model_name=model_name).first()
        if not l or not l.open_symbol:
            raise ValueError(f"{model_name}: no open position")
        qty = l.open_qty
        entry_px = l.open_entry_px
        proc = Decimal(str(qty)) * Decimal(str(exit_price))
        fees = proc * Decimal(str(stt_pct)) + Decimal(str(brokerage))
        net = proc - fees
        pnl = net - (Decimal(str(qty)) * entry_px)
        l.cash = l.cash + net
        l.realized_pnl = (l.realized_pnl or Decimal(0)) + pnl
        l.total_trades = (l.total_trades or 0) + 1
        if pnl > 0:
            l.wins = (l.wins or 0) + 1
        else:
            l.losses = (l.losses or 0) + 1
        symbol = l.open_symbol
        l.open_symbol = None
        l.open_qty = None
        l.open_entry_px = None
        l.open_entry_date = None
        s.add(ModelTrade(
            model_name=model_name,
            side="SELL",
            symbol=symbol,
            qty=qty,
            price=Decimal(str(exit_price)),
            value=net,
            pnl=pnl,
            reason=reason,
            fyers_order_id=fyers_order_id,
        ))
        return _ledger_dict(l)


# ---- Trade history ----

def get_trades(model_name: str, limit: int = 50) -> List[Dict]:
    db = get_database_manager()
    with db.get_session() as s:
        rows = (s.query(ModelTrade)
                  .filter_by(model_name=model_name)
                  .order_by(ModelTrade.trade_at.desc())
                  .limit(limit).all())
        return [_trade_dict(r) for r in rows]


# ---- Aggregate ----

def get_portfolio_stats(price_lookup=None) -> Dict:
    """Return per-model stats + portfolio total.

    price_lookup: optional callable(symbol) -> last_price for MTM of open
    positions. If None, uses last entry price as proxy.
    """
    db = get_database_manager()
    with db.get_session() as s:
        settings_rows = {x.model_name: x for x in s.query(ModelSettings).all()}
        ledger_rows = s.query(ModelLedger).all()

        models = []
        total_allocated = Decimal(0)
        total_nav = Decimal(0)
        total_realized = Decimal(0)
        total_trades = 0

        for l in ledger_rows:
            cfg = settings_rows.get(l.model_name)
            cash = l.cash or Decimal(0)
            pos_value = Decimal(0)
            mtm_price = None
            if l.open_symbol and l.open_qty:
                if price_lookup:
                    try:
                        mtm_price = price_lookup(l.open_symbol)
                    except Exception:
                        mtm_price = None
                if mtm_price is None and l.open_entry_px:
                    mtm_price = float(l.open_entry_px)
                if mtm_price is not None:
                    pos_value = Decimal(str(mtm_price)) * Decimal(str(l.open_qty))

            nav = cash + pos_value
            allocated = cfg.allocated_capital if cfg else Decimal(0)
            pnl_total = nav - allocated
            return_pct = (
                float(pnl_total / allocated * 100) if allocated > 0 else 0
            )

            models.append({
                "model_name": l.model_name,
                "enabled": bool(cfg and cfg.enabled),
                "allocated_capital": float(allocated),
                "cash": float(cash),
                "position_value": float(pos_value),
                "nav": float(nav),
                "pnl_total": float(pnl_total),
                "return_pct": round(return_pct, 2),
                "realized_pnl": float(l.realized_pnl or 0),
                "open_symbol": l.open_symbol,
                "open_qty": l.open_qty,
                "open_entry_px": float(l.open_entry_px) if l.open_entry_px else None,
                "open_entry_date": l.open_entry_date.isoformat() if l.open_entry_date else None,
                "open_mtm_price": mtm_price,
                "total_trades": l.total_trades or 0,
                "wins": l.wins or 0,
                "losses": l.losses or 0,
                "win_rate_pct": round(
                    100.0 * (l.wins or 0) / max(1, l.total_trades or 0), 1
                ),
            })

            total_allocated += allocated
            total_nav += nav
            total_realized += l.realized_pnl or Decimal(0)
            total_trades += l.total_trades or 0

        total_pnl = total_nav - total_allocated
        total_return_pct = (
            float(total_pnl / total_allocated * 100) if total_allocated > 0 else 0
        )

        return {
            "models": models,
            "total": {
                "allocated_capital": float(total_allocated),
                "nav": float(total_nav),
                "pnl_total": float(total_pnl),
                "return_pct": round(total_return_pct, 2),
                "realized_pnl": float(total_realized),
                "total_trades": total_trades,
            },
            "as_of": datetime.utcnow().isoformat(),
        }


# ---- internal helpers ----

def _settings_dict(s: ModelSettings) -> Dict:
    return {
        "model_name": s.model_name,
        "enabled": s.enabled,
        "allocated_capital": float(s.allocated_capital or 0),
        "description": s.description,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _ledger_dict(l: ModelLedger) -> Dict:
    return {
        "model_name": l.model_name,
        "cash": float(l.cash or 0),
        "open_symbol": l.open_symbol,
        "open_qty": l.open_qty,
        "open_entry_px": float(l.open_entry_px) if l.open_entry_px else None,
        "open_entry_date": l.open_entry_date.isoformat() if l.open_entry_date else None,
        "realized_pnl": float(l.realized_pnl or 0),
        "total_trades": l.total_trades or 0,
        "wins": l.wins or 0,
        "losses": l.losses or 0,
        "win_rate_pct": round(
            100.0 * (l.wins or 0) / max(1, l.total_trades or 0), 1
        ),
        "updated_at": l.updated_at.isoformat() if l.updated_at else None,
    }


def _trade_dict(t: ModelTrade) -> Dict:
    return {
        "id": t.id,
        "model_name": t.model_name,
        "side": t.side,
        "symbol": t.symbol,
        "qty": t.qty,
        "price": float(t.price),
        "value": float(t.value),
        "pnl": float(t.pnl) if t.pnl is not None else None,
        "reason": t.reason,
        "fyers_order_id": t.fyers_order_id,
        "trade_at": t.trade_at.isoformat() if t.trade_at else None,
    }
