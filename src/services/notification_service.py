"""Single notification funnel — DB feed + Telegram fan-out.

Every notable event flows through one of the public functions here:

  notify(...)              low-level primitive: persist a row + optional Telegram
  notify_model_decision()  a model's rebalance/eval-day verdict (incl. no-change)
  notify_skip()            "model ran but skipped" (e.g. not a rebalance day)
  notify_order()           executor order placed / failed (funnel for _tg_safe)

Design:
  * Always persists a `notifications` row — the PWA in-app feed reads the
    last 7 days. Telegram is just one channel, sent only for the meaningful
    subset (telegram=True).
  * Weekend-silent + per-day dedupe on model decisions, so a model that
    evaluates twice a day with no change pings once.
  * NEVER raises. Notifications must never block trading — every function
    swallows its own errors and returns a dict.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

log = logging.getLogger("notify")


# Verdict notifications (plan ping / no-change) fire ONLY when the scheduled
# cron emit sets this env flag — diagnostic / preview / UI display runs of
# live_signal stay silent so they cannot leak a Telegram or PWA-feed row.
# Execution + failure pings (notify_order via the executor) are NOT gated.
def _verdict_notify_enabled() -> bool:
    return os.environ.get("MOMROT_TG_NOTIFY") == "1"


# ---- event types -----------------------------------------------------------
SIGNAL = "SIGNAL"              # a tradable action was emitted (ENTRY/ROTATE/EXIT)
NO_CHANGE = "NO_CHANGE"        # rebalance/eval ran, nothing to do (still holding)
SKIP = "SKIP"                  # model ran but gated out (not a rebalance day, disabled)
ORDER_PLACED = "ORDER_PLACED"  # Fyers order placed / ledger recorded
ORDER_FAILED = "ORDER_FAILED"  # order/ledger failure
SYSTEM = "SYSTEM"              # boot, data failure, misc

_SCHEMA_READY = False


def _ensure_schema() -> None:
    """Idempotent, cached create_all so producers in subprocesses (live_signal,
    executor) don't depend on the web app having booted first."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    try:
        from src.models.database import get_database_manager
        get_database_manager().create_tables()
        _SCHEMA_READY = True
    except Exception as e:  # pragma: no cover - defensive
        log.debug(f"notify ensure_schema: {e}")


def is_trading_day(d: datetime = None) -> bool:
    """NSE trading day (Mon-Fri minus NSE holidays).

    Delegates to the single calendar source of truth (tools.shared.nse_calendar)
    so holiday logic lives in exactly one place. Fails soft to the weekday-only
    rule if that import is unavailable (keeps notifications non-blocking)."""
    d = d or datetime.now()
    try:
        from tools.shared.nse_calendar import is_trading_day as _itd
        return _itd(d)
    except Exception as e:  # pragma: no cover - defensive
        log.debug(f"nse_calendar unavailable, weekday-only fallback: {e}")
        return d.weekday() < 5


def current_held(model_name: str):
    """DB-truth held symbol for a model (model_ledger.open_symbol), or None.

    Used so a model's decision notification reflects the real position even
    when its live_signal runs stateless (e.g. n100 runs without --ledger and
    always emits ENTRY1 — only the ledger knows it's already holding)."""
    try:
        from src.services.trading.model_ledger_service import get_ledger
        ledger = get_ledger(model_name)
        if ledger and ledger.get("open_symbol"):
            return ledger["open_symbol"]
    except Exception as e:  # pragma: no cover - defensive
        log.debug(f"current_held({model_name}): {e}")
    return None


# ---- pure helpers (unit-tested without DB) ---------------------------------

def _decision_signature(signals, held_symbol) -> str:
    """Stable signature of a model's verdict. Same verdict twice in a day =
    same signature = deduped; a change within the day re-notifies."""
    if signals:
        parts = sorted(
            f"{s.get('signal', '')}:{s.get('symbol', '')}:{s.get('side', '')}"
            for s in signals
        )
        return "|".join(parts)
    return f"HOLD:{held_symbol or 'FLAT'}"


def _decision_message(model_name, signals, held_symbol, held_ret, note):
    """Return (title, body) for a model decision."""
    if signals:
        lines = []
        for s in signals[:5]:
            sig = s.get("signal", "?")
            sym = s.get("symbol", "?")
            side = s.get("side", "?")
            price = s.get("price")
            px_txt = ""
            try:
                if price is not None:
                    px_txt = f" @ ₹{float(price):,.2f}"
            except (TypeError, ValueError):
                px_txt = ""
            lines.append(f"`{sig}` {side} `{sym}`{px_txt}")
        if len(signals) > 5:
            lines.append(f"…+{len(signals) - 5} more")
        title = f"📋 {model_name}: plan"
        body = "Planned for execution (09:30):\n" + "\n".join(lines)
    else:
        title = f"{model_name}: no change"
        if held_symbol:
            ret_txt = ""
            try:
                if held_ret is not None:
                    ret_txt = f" ({float(held_ret):+.2f}%)"
            except (TypeError, ValueError):
                ret_txt = ""
            body = f"No change — holding `{held_symbol}`{ret_txt}"
        else:
            body = "No change — flat, no entry today"
    if note:
        body = f"{body}\n_{note}_"
    return title, body


# ---- low-level primitive ----------------------------------------------------

def _tg_send(text: str):
    from tools.live.telegram_notify import send
    return send(text, parse_mode="Markdown")


def _dedupe_exists(session, model_name, dedupe_key, trading_date) -> bool:
    from src.models.audit_models import Notification
    if not dedupe_key:
        return False
    q = session.query(Notification.id).filter(
        Notification.trading_date == trading_date,
        Notification.dedupe_key == dedupe_key,
    )
    if model_name is None:
        q = q.filter(Notification.model_name.is_(None))
    else:
        q = q.filter(Notification.model_name == model_name)
    return session.query(q.exists()).scalar()


def notify(event_type, *, title, model=None, body="", level="info",
           telegram=False, meta=None, dedupe_key=None, today=None) -> dict:
    """Record a notification (+ optional Telegram). Never raises.

    Returns {"ok": True, "id": <row id>, ...} or {"ok": True, "skipped": True}
    when deduped, or {"ok": False, "error": ...} on failure.
    """
    try:
        today = today or datetime.now()
        trading_date = today.date()
        _ensure_schema()
        from src.models.database import get_database_manager
        from src.models.audit_models import Notification
        db = get_database_manager()

        # Dedupe check first (own short txn).
        with db.get_session() as s:
            if _dedupe_exists(s, model, dedupe_key, trading_date):
                return {"ok": True, "skipped": True, "reason": "dup"}

        channels = ["db"]
        tg_ok = None
        if telegram:
            msg = f"*{title}*"
            if body:
                msg += f"\n{body}"
            try:
                res = _tg_send(msg)
                tg_ok = bool(res.get("ok"))
            except Exception as e:
                tg_ok = False
                log.debug(f"notify telegram failed: {e}")
            channels.append("telegram")

        with db.get_session() as s:
            row = Notification(
                created_at=today, trading_date=trading_date, model_name=model,
                event_type=event_type, level=level, title=str(title)[:200],
                body=body or "", channels=channels, telegram_ok=tg_ok,
                dedupe_key=(dedupe_key or "")[:160], meta=meta or {},
            )
            s.add(row)
            s.flush()
            return {"ok": True, "id": row.id, "telegram_ok": tg_ok}
    except Exception as e:
        log.warning(f"notify failed ({event_type} {model}): {e}")
        return {"ok": False, "error": str(e)}


# ---- high-level producers ---------------------------------------------------

def notify_model_decision(model_name, signals, *, held_symbol=None,
                          held_ret=None, trigger="CRON", note=None,
                          today=None) -> dict:
    """A model's rebalance/eval-day verdict. Pings even when nothing changed.

    Weekend-silent; deduped per (model, verdict, day) so a model that fires
    twice a day with the same verdict notifies once.
    """
    if not _verdict_notify_enabled():
        return {"ok": True, "skipped": True, "reason": "notify_gated"}
    today = today or datetime.now()
    if not is_trading_day(today):
        return {"ok": True, "skipped": True, "reason": "weekend"}
    sig = _decision_signature(signals, held_symbol)
    event = SIGNAL if signals else NO_CHANGE
    level = "success" if signals else "info"
    title, body = _decision_message(model_name, signals, held_symbol,
                                    held_ret, note)
    return notify(
        event, title=title, model=model_name, body=body, level=level,
        telegram=True, dedupe_key=f"{trigger}:{sig}", today=today,
        meta={"trigger": trigger, "held": held_symbol,
              "signals": signals or []},
    )


def notify_skip(model_name, reason, *, today=None, telegram=False) -> dict:
    """Record a 'model ran but skipped' event (e.g. not a rebalance day).

    Trading-day only (weekend-silent). DB-only by default — these show up in
    the PWA feed so the user can confirm the model evaluated, but don't spam
    Telegram every non-rebalance weekday.
    """
    if not _verdict_notify_enabled():
        return {"ok": True, "skipped": True, "reason": "notify_gated"}
    today = today or datetime.now()
    if not is_trading_day(today):
        return {"ok": True, "skipped": True, "reason": "weekend"}
    return notify(
        SKIP, title=f"{model_name}: skipped", model=model_name, body=reason,
        level="muted", telegram=telegram, dedupe_key=f"SKIP:{reason}",
        today=today, meta={"reason": reason},
    )


def notify_order(text, *, today=None, is_fail=None) -> dict:
    """Funnel for executor Telegram messages → DB feed + Telegram.

    Infers level/event from the leading emoji so existing _tg_safe call sites
    stay unchanged. Always sent to Telegram (executor messages are
    user-critical) and recorded in the feed.

    Args:
        is_fail: Optional explicit failure flag. When the caller KNOWS the
            outcome (executor passes True on a clear failure ping, False on the
            EXECUTED success ping), pass it to bypass the brittle substring
            heuristic — e.g. a future "Margin available" success would otherwise
            be misclassified as ORDER_FAILED. When None (default), fall back to
            the leading-emoji + keyword heuristic for backward compatibility.
    """
    text = text or ""
    stripped = text.strip()
    first = (stripped.splitlines() or [""])[0][:200]
    rest = "\n".join(stripped.splitlines()[1:])
    if is_fail is None:
        is_fail = (
            stripped.startswith("⚠️") or stripped.startswith("❌")
            or "FAIL" in text or "Margin" in text or "Shortfall" in text
        )
    event = ORDER_FAILED if is_fail else ORDER_PLACED
    level = "error" if is_fail else "success"
    # Strip leading emoji + markdown stars from the title for the feed.
    clean = first.lstrip("✅💰⚠️❌ ").replace("*", "").strip()
    return notify(event, title=clean or first, body=rest, level=level,
                  telegram=True, today=today, meta={"raw": text})
