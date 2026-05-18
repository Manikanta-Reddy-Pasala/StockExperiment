"""Thin helpers that wrap audit table writes.

All functions are 'never raise' — audit must never break trading. Callers
log a debug message on failure and continue.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, Optional

log = logging.getLogger("audit")


def _safe_dec(v) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def _safe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def write_order(model_name: Optional[str], symbol: str, side: str, qty: int,
                ordered_price: float, fill_price: Optional[float],
                fill_qty: Optional[int], product: str, pricetype: str,
                status: str, fyers_order_id: str = "",
                signal_id: Optional[int] = None,
                error_text: Optional[str] = None,
                raw_request: Optional[Dict[str, Any]] = None,
                raw_response: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Insert an audit_orders row. Returns row id or None on failure."""
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditOrder
        slip = None
        if fill_price is not None and fill_qty:
            try:
                slip = float(fill_qty) * (float(fill_price) - float(ordered_price))
            except Exception:
                slip = None
        db = get_database_manager()
        with db.get_session() as s:
            row = AuditOrder(
                model_name=model_name,
                signal_id=signal_id,
                fyers_order_id=fyers_order_id or "",
                symbol=symbol,
                side=side,
                qty=qty,
                ordered_price=_safe_dec(ordered_price),
                fill_price=_safe_dec(fill_price),
                fill_qty=_safe_int(fill_qty),
                product=product,
                pricetype=pricetype,
                status=status,
                slippage_inr=_safe_dec(slip),
                error_text=error_text,
                raw_request=raw_request,
                raw_response=raw_response,
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as e:
        log.warning(f"audit write_order failed: {e}")
        return None


def write_rebalance_decision(
    model_name: str, trigger: str, decision: str, reason: str,
    held_symbol: Optional[str] = None,
    held_qty: Optional[int] = None,
    held_entry_px: Optional[float] = None,
    held_mtm_px: Optional[float] = None,
    rank1_symbol: Optional[str] = None,
    rank1_price: Optional[float] = None,
    qty_sized: Optional[int] = None,
    qty_clamped: Optional[int] = None,
    clamp_reason: Optional[str] = None,
) -> Optional[int]:
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditRebalanceDecision
        db = get_database_manager()
        with db.get_session() as s:
            row = AuditRebalanceDecision(
                model_name=model_name,
                trigger=trigger,
                held_symbol=held_symbol,
                held_qty=_safe_int(held_qty),
                held_entry_px=_safe_dec(held_entry_px),
                held_mtm_px=_safe_dec(held_mtm_px),
                rank1_symbol=rank1_symbol,
                rank1_price=_safe_dec(rank1_price),
                decision=decision,
                reason=reason,
                qty_sized=_safe_int(qty_sized),
                qty_clamped=_safe_int(qty_clamped),
                clamp_reason=clamp_reason,
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as e:
        log.warning(f"audit write_rebalance_decision failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def write_rankings(model_name: str, trading_date: date,
                   universe_size: int, qualifying_count: int,
                   ranking: list) -> None:
    """Bulk insert a model's top-N ranking for one session."""
    if not ranking:
        return
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditModelRanking
        db = get_database_manager()
        with db.get_session() as s:
            for r in ranking:
                s.add(AuditModelRanking(
                    model_name=model_name,
                    trading_date=trading_date,
                    universe_size=universe_size,
                    qualifying_count=qualifying_count,
                    rank=_safe_int(r.get("rank")),
                    symbol=r.get("symbol", ""),
                    name=r.get("name", ""),
                    score=_safe_dec(r.get("ret_30d_pct") or r.get("score")),
                    price=_safe_dec(r.get("price")),
                    extra={k: v for k, v in r.items()
                           if k not in ("rank", "symbol", "name", "score", "price")},
                ))
    except Exception as e:
        log.warning(f"audit write_rankings failed: {e}")


def write_signal(model_name: str, trading_date: date, signal_type: str,
                 symbol: str, side: str, price: Optional[float] = None,
                 qty_planned: Optional[int] = None,
                 reason: str = "", extra: Optional[Dict] = None) -> Optional[int]:
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditModelSignal
        db = get_database_manager()
        with db.get_session() as s:
            row = AuditModelSignal(
                model_name=model_name,
                trading_date=trading_date,
                signal_type=signal_type,
                symbol=symbol,
                side=side,
                price=_safe_dec(price),
                qty_planned=_safe_int(qty_planned),
                reason=reason[:128] if reason else None,
                extra=extra or None,
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as e:
        log.warning(f"audit write_signal failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def write_config_change(model_name: str, field: str, old_value, new_value,
                        reason: str, changed_by: str = "system") -> None:
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditConfigChange

        def _jsonable(v):
            if v is None:
                return None
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, (datetime, date)):
                return v.isoformat()
            return v

        db = get_database_manager()
        with db.get_session() as s:
            s.add(AuditConfigChange(
                changed_by=changed_by,
                model_name=model_name,
                field=field,
                old_value={"v": _jsonable(old_value)},
                new_value={"v": _jsonable(new_value)},
                reason=reason,
            ))
    except Exception as e:
        log.warning(f"audit write_config_change failed: {e}")


def write_data_quality(snapshot: list) -> None:
    """Insert one row per model from /admin/system/models-status response."""
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditDataQuality
        db = get_database_manager()
        with db.get_session() as s:
            for m in snapshot:
                items = m.get("items") or []
                # Pull a few common numeric metrics if present
                cov_item = next((i for i in items if "coverage" in (i.get("extra") or "")), None)
                cov = None
                if cov_item and isinstance(cov_item.get("extra"), str):
                    try:
                        cov = float(cov_item["extra"].split("%")[0].split()[-1])
                    except Exception:
                        cov = None
                stale_item = next((i for i in items if "old" in (i.get("extra") or "")), None)
                stale_days = None
                if stale_item and isinstance(stale_item.get("extra"), str):
                    try:
                        stale_days = int(stale_item["extra"].split("d")[0].strip())
                    except Exception:
                        stale_days = None
                uni_item = next((i for i in items if "universe" in i.get("label", "").lower()), None)
                uni_size = _safe_int(uni_item.get("value")) if uni_item else None
                uni_age = None
                if uni_item and isinstance(uni_item.get("extra"), str) and "age" in uni_item["extra"]:
                    try:
                        uni_age = int(uni_item["extra"].split()[2].rstrip("d"))
                    except Exception:
                        uni_age = None
                s.add(AuditDataQuality(
                    model_name=m.get("name", ""),
                    universe_size=uni_size,
                    universe_age_days=uni_age,
                    coverage_pct=_safe_dec(cov),
                    stale_days=stale_days,
                    data_sufficient=bool(m.get("data_sufficient")),
                    wired=bool(m.get("wired")),
                    raw_items=items,
                ))
    except Exception as e:
        log.warning(f"audit write_data_quality failed: {e}")


def write_system_event(event_type: str, component: str,
                       metadata: Optional[Dict] = None) -> None:
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditSystemEvent
        db = get_database_manager()
        with db.get_session() as s:
            s.add(AuditSystemEvent(
                event_type=event_type,
                component=component,
                metadata_json=metadata or None,
            ))
    except Exception as e:
        log.warning(f"audit write_system_event failed: {e}")
