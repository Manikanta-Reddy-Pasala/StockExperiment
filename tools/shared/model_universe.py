"""Central point-in-time universe gate for live order execution.

Belt-and-suspenders for EVERY model: regardless of any per-model ranking bug,
the order executor refuses to place a BUY for a symbol that was not a
point-in-time member of the model's declared index on the trade date.

Origin: 2026-06-02 — ORB bought SPARC, a name in the survivorship-biased
universe_union (703 names that were EVER in the index) but NOT a current
Nifty-500 member. Each model's signal layer was fixed to rank over
eligible_at(index, date); THIS module is the second, central line of defence so
a FUTURE regression in ANY model still cannot place a real order for an
off-universe name. The two layers are independent on purpose.

MODEL_INDEX maps model_name -> the index name understood by index_membership.
File-based models (their universe IS a curated JSON snapshot built offline, not
the live index) map to None and are intentionally NOT gated here — the file is
their universe contract, and the signal generator can only pick names from it.

Fail-OPEN by design: if a model is unknown, not index-gated, or the membership
table cannot be read / is empty, the gate returns None ("no opinion, allow").
A data-loading hiccup must never block a legitimate live trade — the gate only
ever BLOCKS when it can positively prove the symbol is off-universe.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from tools.shared.index_membership import eligible_at

# model_name -> index for eligible_at, or None to skip (file-based universe).
# Keys are the model_name strings written into the signals file / model_settings.
MODEL_INDEX: dict[str, Optional[str]] = {
    "momentum_retest_n500":       "n500",
    "n20_daily_large_only":       "n100",   # n40: selection is ∩ Nifty-100
    "emerging_momentum":          "n500",   # universe = n500 minus n100; n500 is a safe superset gate
    # file-based (curated JSON universe) — NOT index-gated:
    "momentum_n100_top5_max1":    None,
    "momentum_pseudo_n100_adv":   None,
}

_SENTINEL = "__unknown__"


def _plain(sym: str) -> str:
    """Strip Fyers wrapper -> bare upper ticker (SPARC from NSE:SPARC-EQ)."""
    return (sym or "").replace("NSE:", "").replace("-EQ", "").strip().upper()


def is_in_universe(model_name: str, symbol: str,
                   on_date: Optional[date] = None) -> Optional[bool]:
    """Is `symbol` a point-in-time member of `model_name`'s index on `on_date`?

    Returns:
        True  — symbol IS in the model's PIT index universe (allow).
        False — symbol is NOT in it (the executor should BLOCK the buy).
        None  — no opinion: model is file-based / unknown, or the membership
                table is missing/empty/unreadable. Caller treats None as allow
                (fail-open).
    """
    if on_date is None:
        on_date = date.today()
    idx = MODEL_INDEX.get(model_name, _SENTINEL)
    if idx is None or idx == _SENTINEL:
        return None
    try:
        elig = eligible_at(idx, on_date)
    except Exception:
        return None
    if not elig:
        return None
    return _plain(symbol) in elig
