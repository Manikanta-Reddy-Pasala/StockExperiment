"""Daily heartbeat summary — ONE Telegram listing every model's state.

Why this exists
---------------
On a non-rebalance day the rebalance-gated models (n100 / pseudo / n40 /
emerging) call ``notify_skip(telegram=False)`` → they post a "skipped" row to
the PWA feed but send NO Telegram, by design (the 2026-06-03 anti-spam fix that
killed per-model "no change" spam). The user still wants a once-a-day
confirmation that EVERY model evaluated and what it currently holds.

This sends a SINGLE consolidated ping (not one per model) after the morning
trade window settles — a heartbeat without the per-model spam. Scheduled at
09:40 IST by scheduler.py (executes run 09:30-09:35). Weekend / NSE-holiday
silent. Deduped once per trading day.

Run manually:  python tools/daily_summary.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("daily_summary")

# Canonical display order (gated single-pos models first, then daily/intraday).
MODEL_ORDER = [
    "momentum_n100_top5_max1",
    "momentum_pseudo_n100_adv",
    "n20_daily_large_only",
    "midcap_narrow_60d_breakout",
    "emerging_momentum",
    "momentum_retest_n500",
    "orb_momentum_intraday",
]

# Short labels so the message stays compact on a phone.
SHORT = {
    "momentum_n100_top5_max1": "n100",
    "momentum_pseudo_n100_adv": "pseudo",
    "n20_daily_large_only": "n40",
    "midcap_narrow_60d_breakout": "midcap",
    "emerging_momentum": "emerging",
    "momentum_retest_n500": "retest",
    "orb_momentum_intraday": "orb",
}


def _fmt_sym(sym: str) -> str:
    return (sym or "").replace("NSE:", "").replace("-EQ", "")


def compose_summary(today_str: str, models: list) -> tuple:
    """Pure: build (title, body) from a list of per-model state dicts.

    Each model dict: {
        "model": <canonical name>,
        "holdings": [(symbol, qty), ...],   # current open positions ([] = flat)
        "action":  <str>,                    # what it did today (human phrase)
        "today_pnl": <float|None>,           # today's realized+MTM P&L if known
    }
    """
    title = f"📊 Daily model summary — {today_str}"
    lines = []
    for m in models:
        label = SHORT.get(m["model"], m["model"])
        holds = m.get("holdings") or []
        if holds:
            pos = ", ".join(f"{_fmt_sym(s)}×{q}" for s, q in holds)
        else:
            pos = "flat"
        line = f"• *{label}*: {pos} — {m.get('action', 'no run logged')}"
        tp = m.get("today_pnl")
        if tp is not None:
            sign = "+" if tp >= 0 else "-"
            line += f"  (today {sign}₹{abs(tp):,.0f})"
        lines.append(line)
    body = "\n".join(lines)
    return title, body


def _gather(today: datetime) -> list:
    """Read each model's current holdings + today's action from the DB."""
    from src.models.database import get_database_manager
    from src.models.model_ledger_models import ModelLedger, ModelHolding
    from src.models.audit_models import AuditOrder, Notification
    from sqlalchemy import func

    db = get_database_manager()
    out = []
    with db.get_session() as s:
        ledgers = {l.model_name: l for l in s.query(ModelLedger).all()}
        holds_by_model: dict = {}
        for h in s.query(ModelHolding).all():
            if int(h.qty or 0) > 0:
                holds_by_model.setdefault(h.model_name, []).append((h.symbol, int(h.qty)))

        # Real orders placed today (a true BUY/SELL action).
        orders_by_model: dict = {}
        for r in (s.query(AuditOrder.model_name, AuditOrder.side, AuditOrder.symbol,
                          AuditOrder.fill_qty, AuditOrder.status)
                    .filter(func.date(AuditOrder.placed_at) == today.date())
                    .filter(AuditOrder.status == "filled").all()):
            orders_by_model.setdefault(r[0], []).append(
                f"{r[1]} {_fmt_sym(r[2])}×{int(r[3] or 0)}")

        # Latest decision notification per model today (SKIP / NO_CHANGE / SIGNAL),
        # excluding the telegram channel so display-run noise is ignored.
        decision_by_model: dict = {}
        for n in (s.query(Notification)
                    .filter(func.date(Notification.created_at) == today.date())
                    .filter(Notification.event_type.in_(("SKIP", "NO_CHANGE", "SIGNAL")))
                    .order_by(Notification.created_at.asc()).all()):
            decision_by_model[n.model_name] = n.event_type  # last wins

        names = list(MODEL_ORDER) + [m for m in ledgers if m not in MODEL_ORDER]
        for name in names:
            if name not in ledgers:
                continue
            l = ledgers[name]
            holds = list(holds_by_model.get(name, []))
            if l.open_symbol and int(l.open_qty or 0) > 0:
                holds.insert(0, (l.open_symbol, int(l.open_qty)))

            if orders_by_model.get(name):
                action = "✅ traded: " + ", ".join(orders_by_model[name])
            else:
                ev = decision_by_model.get(name)
                action = {
                    "SKIP": "evaluated — no rebalance today",
                    "NO_CHANGE": "evaluated — hold, no change",
                    "SIGNAL": "signalled (no fill recorded)",
                }.get(ev, "no run logged")
            out.append({"model": name, "holdings": holds, "action": action,
                        "today_pnl": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the message, do not send Telegram / write feed")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    today = datetime.now()
    try:
        from tools.shared.nse_calendar import is_trading_day
        if not is_trading_day(today.date()):
            log.info("daily_summary: not a trading day — silent.")
            return
    except Exception as e:
        log.warning(f"daily_summary: nse_calendar unavailable ({e}); proceeding.")

    models = _gather(today)
    title, body = compose_summary(today.strftime("%Y-%m-%d"), models)

    if a.dry_run:
        print(title)
        print(body)
        return

    from src.services.notification_service import notify, _verdict_notify_enabled
    if not _verdict_notify_enabled():
        log.info("daily_summary: MOMROT_TG_NOTIFY!=1 — feed-only path gated, skipping.")
        return
    res = notify("SUMMARY", title=title, body=body, level="info", telegram=True,
                 dedupe_key=f"DAILY_SUMMARY:{today.date()}", today=today)
    log.info(f"daily_summary sent: {res}")


if __name__ == "__main__":
    main()
