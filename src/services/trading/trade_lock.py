"""Cross-process trading lock — serialises live order placement.

The system places real Fyers orders from TWO process families against ONE
shared broker account:
  * the cron executor (scheduler container), and
  * the manual /run-now, /buy-now, rebalance, exit endpoints (web/gunicorn,
    which itself runs MULTIPLE workers).

Nothing previously serialised them, so a manual click could race the cron, two
gunicorn workers could each place the same buy, and the executor's per-symbol
"Fyers already holds -> skip" guard does not cover the manual endpoints. This
module provides a single global advisory lock (Postgres `pg_advisory_lock`,
shared by every container that talks to the same DB — no extra infra) that all
order-placement paths acquire before touching the broker.

Semantics (see `lock_proceed_decision`):
  * lock acquired                       -> proceed.
  * lock held by another process        -> do NOT proceed (caller skips / 409).
  * lock infra unreachable (DB error)   -> proceed (fail-open) + loud warning,
    because halting the whole account on a lock hiccup is worse than the rare
    race the executor's own duplicate-buy guard still backstops.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from sqlalchemy import text

log = logging.getLogger("trade_lock")

# Single global advisory-lock key for "live order placement". Arbitrary but
# fixed 64-bit int; every process must use the same value.
TRADING_LOCK_KEY = 730_051_001


def lock_proceed_decision(acquired: bool, infra_error: bool) -> bool:
    """Whether the caller should proceed to place orders.

    proceed iff we hold the lock, OR the lock infra failed (fail-open). A lock
    simply held by another process => abort (the case we must not race).
    """
    return bool(acquired or infra_error)


@contextmanager
def trading_lock(wait_s: float = 30.0, key: int = TRADING_LOCK_KEY):
    """Context manager yielding True if the caller may place orders.

    Holds a dedicated DB connection for the lock's lifetime (pg advisory locks
    are connection-scoped) and releases on exit. Polls `pg_try_advisory_lock`
    up to `wait_s` seconds. On any DB/infra error, yields True (fail-open).
    """
    from src.models.database import get_database_manager

    conn = None
    acquired = False
    infra_error = False
    try:
        eng = get_database_manager().engine
        conn = eng.connect()
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            got = conn.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()
            if got:
                acquired = True
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    except Exception as e:  # infra (DB unreachable, etc.) — fail-open
        infra_error = True
        log.warning("trading_lock infra error (%s) — proceeding fail-open", e)

    proceed = lock_proceed_decision(acquired, infra_error)
    if not proceed:
        log.warning("trading_lock: held by another process — caller should skip")
    try:
        yield proceed
    finally:
        if conn is not None:
            try:
                if acquired:
                    conn.execute(
                        text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            except Exception as e:
                log.warning("trading_lock unlock failed (%s)", e)
            finally:
                conn.close()
