"""NSE trading-calendar helpers (single source of truth for "is the market open").

Separation of concerns: ALL holiday / first-trading-day logic lives here.
Callers (notification_service, the momentum live_signal rebalance gate, the
executor holiday guard) import from this module rather than re-deriving the
weekday/holiday rule themselves.

IMPORTANT: NSE_HOLIDAYS must be refreshed YEARLY. NSE publishes the next
calendar year's trading-holiday list in December. When a new year starts,
add that year's dates here (and prune very old years to keep the set small).
Dates are the NSE *equity* segment trading holidays; the weekend rule in
``is_trading_day`` covers Saturdays/Sundays separately, so only weekday
holidays need to be listed (weekend holidays are harmless duplicates).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Union

# NSE equity-segment trading holidays. Refresh yearly (see module docstring).
# 2025 + 2026 weekday holidays. Saturdays/Sundays are handled by the weekday
# rule, so a holiday that lands on a weekend need not be listed.
NSE_HOLIDAYS = frozenset({
    # ---- 2025 ----
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-ul-Fitr (Ramadan Eid)
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali Laxmi Pujan (Muhurat session aside)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas

    # ---- 2026 (best-known NSE dates; refresh when NSE publishes final list) ----
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 4),    # Holi
    date(2026, 3, 19),   # Id-ul-Fitr (approx.)
    date(2026, 3, 26),   # Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day (Saturday — harmless, weekend rule covers)
    date(2026, 9, 14),   # Ganesh Chaturthi (approx.)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra (approx.)
    date(2026, 11, 9),   # Diwali Laxmi Pujan (approx.)
    date(2026, 11, 10),  # Diwali Balipratipada (approx.)
    date(2026, 11, 24),  # Guru Nanak Jayanti (approx.)
    date(2026, 12, 25),  # Christmas
})


def _as_date(d: Union[date, datetime, None]) -> date:
    """Coerce a datetime/date/None to a plain ``date`` (None -> today)."""
    if d is None:
        return datetime.now().date()
    if isinstance(d, datetime):
        return d.date()
    return d


def is_trading_day(d: Union[date, datetime, None] = None) -> bool:
    """True if ``d`` is an NSE trading day: a weekday (Mon-Fri) not in NSE_HOLIDAYS.

    Args:
        d: date/datetime to test (defaults to today).

    Returns:
        bool: True only on Mon-Fri that are not listed NSE holidays.
    """
    dd = _as_date(d)
    return dd.weekday() < 5 and dd not in NSE_HOLIDAYS


def first_trading_day_of_month(d: Union[date, datetime, None] = None) -> date:
    """Return the first NSE trading day on/after the 1st of ``d``'s month.

    Walks forward from the 1st until a trading day is found (skips a 1st that
    falls on a weekend or holiday).
    """
    dd = _as_date(d)
    cur = date(dd.year, dd.month, 1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def is_first_trading_day_of_month(d: Union[date, datetime, None] = None) -> bool:
    """True if ``d`` is the first NSE trading day of its month."""
    dd = _as_date(d)
    return dd == first_trading_day_of_month(dd)
