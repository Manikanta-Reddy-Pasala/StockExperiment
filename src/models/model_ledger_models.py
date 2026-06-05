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
    """User-controlled per-model settings.

    Capital model (post 2026-05-17 split):
      invested_amount — cumulative principal in (deposits − withdrawals).
                        Used as denominator in return-% calc.
      current_amount  — latest NAV snapshot (cash + open MTM).
                        Updated by record_buy / record_sell / MTM refresh.
    """
    __tablename__ = "model_settings"

    model_name = Column(String(64), primary_key=True)
    enabled = Column(Boolean, default=True, nullable=False)
    # signals_only: when True the model still emits signals + ranking (observe),
    # but executors place NO real orders and do not mutate the ledger. Default
    # False = enabled models trade live; flip per-model in Settings to observe.
    signals_only = Column(Boolean, default=False, nullable=False)
    invested_amount = Column(Numeric(14, 2), nullable=False)  # principal in
    current_amount = Column(Numeric(14, 2), nullable=False, default=0)  # NAV
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
    # True once a partial profit-take has fired for the CURRENT open position, so
    # the daily check books HALF only once (reset to False on every fresh buy).
    # Only used by models with strategy.PROFIT_TAKE_PCT > 0 (emerging).
    profit_taken = Column(Boolean, default=False, nullable=False, server_default="false")
    realized_pnl = Column(Numeric(14, 2), default=0)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelHolding(Base):
    """One open position of a MULTI-holding model (e.g. momentum_retest_n500 K=3).

    Single-holding models keep their one position in ModelLedger.open_symbol;
    this table holds the N open positions of multi-holding models — one row per
    (model_name, symbol). Cash / realized_pnl / cumulative stats still live in
    ModelLedger (shared), so only the position storage differs. Single-holding
    flow never touches this table.
    """
    __tablename__ = "model_holdings"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(64), ForeignKey("model_settings.model_name"),
                        nullable=False, index=True)
    symbol = Column(String(64), nullable=False)        # normalized NSE:SYM-EQ
    qty = Column(Integer, nullable=False)
    entry_px = Column(Numeric(14, 4), nullable=False)
    entry_date = Column(Date, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelTrade(Base):
    """Audit log of every BUY/SELL routed through a model ledger."""
    __tablename__ = "model_trades"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(64), ForeignKey("model_settings.model_name"),
                       nullable=False, index=True)
    side = Column(String(16), nullable=False)  # BUY | SELL | DEPOSIT | WITHDRAW
    symbol = Column(String(64), nullable=False)
    qty = Column(Integer, nullable=False)
    price = Column(Numeric(14, 4), nullable=False)
    value = Column(Numeric(14, 2), nullable=False)
    pnl = Column(Numeric(14, 2))               # only on SELL
    reason = Column(String(32))                # ENTRY | TARGET | TRAIL | SMA | MAX_HOLD
    fyers_order_id = Column(String(64))
    # Approx broker charges (Fyers rates) for THIS leg — auto-stamped by the
    # before_insert listener below so all ~9 insert sites get it for free.
    charges_inr = Column(Numeric(14, 4))
    # Naive timestamp in container's local time (TZ=Asia/Kolkata → IST).
    # UI fmtIST helper treats naive ISO as already-IST and doesn't re-shift.
    trade_at = Column(DateTime, default=datetime.now, index=True)


# ---------------------------------------------------------------------------
# SQLAlchemy event listeners — buffer 'set' events into the parent session,
# flush into audit_config_changes only AFTER the session commits.
#
# Earlier version called write_config_change() directly inside the 'set'
# event; that opened a *new* session while the parent session was mid-flush,
# which detached the ModelSettings row and broke toggle-enabled with
# 'Instance not bound to a Session'. We now stash deltas on session.info
# and write them in an after_commit hook, where the parent session is safely
# closed.
# ---------------------------------------------------------------------------
try:
    from sqlalchemy import event as _sa_event
    from sqlalchemy.orm import Session as _SASession

    @_sa_event.listens_for(ModelTrade, "before_insert")
    def _stamp_trade_charges(_mapper, _connection, target):
        """Auto-compute approx broker charges (Fyers rates) for every BUY/SELL
        leg so model_trades carries charges without each insert site doing it.
        Best-effort + pure; NEVER blocks the trade insert."""
        try:
            if target.charges_inr is not None:
                return
            side = (target.side or "").upper()
            if side not in ("BUY", "SELL"):
                target.charges_inr = 0
                return
            # Lazy imports avoid a circular import with the service layer.
            from tools.live.broker_charges import compute_charges
            from src.services.trading.model_ledger_service import product_for_model
            prod = product_for_model(target.model_name)
            c = compute_charges(side, int(target.qty or 0),
                                float(target.price or 0), prod)
            target.charges_inr = c.get("total", 0)
        except Exception:
            pass  # a charge estimate must never fail a real trade insert

    _SETTINGS_FIELDS = ("enabled", "invested_amount", "current_amount", "description")
    _LEDGER_FIELDS = (
        "cash", "open_symbol", "open_qty", "open_entry_px",
        "open_entry_date", "realized_pnl",
    )

    def _buffer_change(target, field, old_v, new_v, reason):
        try:
            sess = _SASession.object_session(target)
            if sess is None:
                return
            sess.info.setdefault("_audit_buffer", []).append({
                "model_name": getattr(target, "model_name", None),
                "field": field, "old": old_v, "new": new_v, "reason": reason,
            })
        except Exception:
            pass

    def _settings_attr_changed(target, value, oldvalue, initiator):
        if oldvalue == value or oldvalue is None:
            return
        if initiator.key not in _SETTINGS_FIELDS:
            return
        _buffer_change(target, initiator.key, oldvalue, value, "SETTINGS_UPDATE")

    def _ledger_attr_changed(target, value, oldvalue, initiator):
        if oldvalue == value or oldvalue is None:
            return
        if initiator.key not in _LEDGER_FIELDS:
            return
        _buffer_change(target, initiator.key, oldvalue, value, "LEDGER_UPDATE")

    @_sa_event.listens_for(_SASession, "after_commit")
    def _flush_audit_buffer(session):
        buf = session.info.pop("_audit_buffer", None)
        if not buf:
            return
        # Open a fresh session for the audit writes — never reuse the
        # parent (it just committed, attribute reads on its objects are
        # detached). audit_service.write_config_change opens its own.
        for ch in buf:
            try:
                from src.services.audit_service import write_config_change
                write_config_change(ch["model_name"], ch["field"],
                                    ch["old"], ch["new"], ch["reason"])
            except Exception:
                pass

    for _f in _SETTINGS_FIELDS:
        _sa_event.listen(getattr(ModelSettings, _f), "set",
                         _settings_attr_changed, retval=False)
    for _f in _LEDGER_FIELDS:
        _sa_event.listen(getattr(ModelLedger, _f), "set",
                         _ledger_attr_changed, retval=False)
except Exception:
    pass
