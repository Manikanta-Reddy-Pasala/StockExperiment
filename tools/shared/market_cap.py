"""Market-cap classifier — maps an NSE symbol to large / mid / small cap.

Source = the three NSE index constituent CSVs under src/data/symbols/
(downloaded from niftyindices.com by tools/analysis/download_niftyindices.py):

  - Nifty 100        -> "large"   (NSE definition: top-100 by full market cap)
  - Nifty Midcap 150 -> "mid"     (ranks ~101-250)
  - Nifty Smallcap 250 -> "small" (ranks ~251-500)

These are CURRENT snapshots, so the label is the stock's cap TODAY, not at the
historical trade date. A name can therefore read "small" in a backtest ledger
even though it was a large-cap (in Nifty 100) when the trade happened and was
later demoted (e.g. ZEEL). Use classify_pit() when point-in-time large-cap
membership matters (it defers to index_membership.eligible_at for n100).

Precedence when a symbol appears in more than one list (overlaps happen around
reconstitution): large > mid > small.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

_SYM_DIR = Path(__file__).resolve().parents[2] / "src" / "data" / "symbols"
_FILES = [("large", "nifty100.csv"),
          ("mid", "nifty_midcap150.csv"),
          ("small", "nifty_smallcap250.csv")]


def _plain(sym: str) -> str:
    """Strip the Fyers wrapper: 'NSE:RELIANCE-EQ' -> 'RELIANCE'."""
    return sym.replace("NSE:", "").replace("-EQ", "").strip().upper()


@lru_cache(maxsize=1)
def _cap_map() -> dict[str, str]:
    """{plain_symbol: 'large'|'mid'|'small'} from the 3 CSVs (large wins overlaps)."""
    out: dict[str, str] = {}
    # Load small -> mid -> large LAST so large overwrites any overlap (precedence).
    for cap, fname in reversed(_FILES):
        p = _SYM_DIR / fname
        if not p.exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("Series", "").strip() == "EQ":
                    out[r["Symbol"].strip().upper()] = cap
    return out


def classify(symbol: str) -> str:
    """Cap class of `symbol` by CURRENT index membership.

    Args:
        symbol: plain ('RELIANCE') or Fyers ('NSE:RELIANCE-EQ') form.

    Returns:
        'large' | 'mid' | 'small' | 'unknown' (not in any of the 3 indices).
    """
    return _cap_map().get(_plain(symbol), "unknown")


def classify_pit(symbol: str, on_date) -> str:
    """Point-in-time cap of `symbol` on `on_date`.

    'large' IFF the symbol was in NSE Nifty 100 on `on_date` (the only index we
    have true PIT membership for, via index_membership.eligible_at). Otherwise it
    was NOT large that day, so we fall back to the current Midcap-150 /
    Smallcap-250 CSVs — and if the current CSV says 'large' (i.e. the name was
    PROMOTED into Nifty 100 since the trade), we DOWNGRADE it to 'mid' rather than
    emit a false PIT-large. (We lack PIT mid-vs-small history, so a non-N100 name
    is best-effort mid/small from today's snapshot.)
    """
    try:
        from tools.shared.index_membership import eligible_at
        if _plain(symbol) in {s.upper() for s in eligible_at("n100", on_date)}:
            return "large"
    except Exception:
        pass
    cur = classify(symbol)
    return "mid" if cur == "large" else cur   # not in N100 then -> can't be PIT-large
