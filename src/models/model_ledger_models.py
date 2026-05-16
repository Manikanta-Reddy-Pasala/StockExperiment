"""Per-model capital + ledger tracking.

Each enabled trading model gets:
  - ModelSettings row: user-entered allocated capital, enable flag
  - ModelLedger row: cash balance, current open position, realized PnL
  - ModelTrade rows: per-fill audit log

Designed for N models. Add a model = insert a row, no schema changes.
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Numeric, Boolean, DateTime, Date, ForeignKey,
)
from sqlalchemy.orm import relationship

from .stock_models import Base


class ModelSettings(Base):
    """User-controlled per-model settings."""
    __tablename__ = "model_settings"

    model_name = Column(String(64), primary_key=True)
    enabled = Column(Boolean, default=True, nullable=False)
    allocated_capital = Column(Numeric(14, 2), nullable=False)  # absolute ₹
    description = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelLedger(Base):
    """Per-model cash + open position + cumulative stats."""
    __tablename__ = "model_ledger"

    model_name = Column(String(64), ForeignKey("model_settings.model_name"),
                       primary_key=True)
    cash = Column(Numeric(14, 2), nullable=False, default=0)
    open_symbol = Column(String(64))           # null when flat
    open_qty = Column(Integer)
    open_entry_px = Column(Numeric(14, 4))
    open_entry_date = Column(Date)
    realized_pnl = Column(Numeric(14, 2), default=0)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelTrade(Base):
    """Audit log of every BUY/SELL routed through a model ledger."""
    __tablename__ = "model_trades"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(64), ForeignKey("model_settings.model_name"),
                       nullable=False, index=True)
    side = Column(String(4), nullable=False)   # BUY | SELL
    symbol = Column(String(64), nullable=False)
    qty = Column(Integer, nullable=False)
    price = Column(Numeric(14, 4), nullable=False)
    value = Column(Numeric(14, 2), nullable=False)
    pnl = Column(Numeric(14, 2))               # only on SELL
    reason = Column(String(32))                # ENTRY | TARGET | TRAIL | SMA | MAX_HOLD
    fyers_order_id = Column(String(64))
    trade_at = Column(DateTime, default=datetime.utcnow, index=True)
