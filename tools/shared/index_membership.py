"""Point-in-time NSE index membership lookup.

Reads src/data/symbols/n{100,500}_membership.csv (built by
tools/analysis/build_membership_table.py) and exposes:

    eligible_at(index_name, on_date) -> set[str]

WHY: avoids survivorship bias in backtests. The old code applied today's
nifty100.csv to 2016 data, which silently excludes stocks that LEFT the
index (e.g. YESBANK) and includes stocks that hadn't yet ENTERED (e.g.
ADANIENT). Both inflate look-ahead CAGR and under-estimate drawdown.

Membership rule (LAST-KNOWN-STATE):
    Members at date d = members of the most recent snapshot whose date
    <= d. Before the first snapshot, falls back to first snapshot's
    members (best approximation given Wayback coverage).

The CSV stores half-open intervals (start_date inclusive, end_date
exclusive). end_date = 2099-12-31 sentinel means "still in index as of
the most recent snapshot".

NEVER mixes with the current ind_nifty{100,500}list.csv at backtest time
— callers must filter the universe through eligible_at(d), not load the
current CSV and filter through this module.
"""
from __future__ import annotations

import csv
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS_DIR = ROOT / "src" / "data" / "symbols"


@lru_cache(maxsize=4)
def _load_intervals(index_name: str) -> list[tuple[str, date, date]]:
    """Parse the membership CSV into a list of (symbol, start, end) tuples.

    Cached so the file is read once per process. Sorted by symbol then
    start_date so dedupe / scans are stable.
    """
    path = SYMBOLS_DIR / f"{index_name}_membership.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Index membership file not found: {path}. "
            f"Run tools/analysis/build_membership_table.py first."
        )
    out: list[tuple[str, date, date]] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append((
                r["symbol"].strip(),
                date.fromisoformat(r["start_date"]),
                date.fromisoformat(r["end_date"]),
            ))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def eligible_at(index_name: str, on_date: date) -> set[str]:
    """Return the set of symbols in `index_name` (e.g. "n100") on `on_date`.

    Returns plain symbols (no NSE: prefix, no -EQ suffix) so the caller
    can convert to the convention they need.

    Half-open semantics: a symbol is eligible if
        start_date <= on_date < end_date.
    """
    intervals = _load_intervals(index_name)
    return {sym for sym, sd, ed in intervals if sd <= on_date < ed}


def eligible_fyers(index_name: str, on_date: date) -> set[str]:
    """Same as eligible_at but returns Fyers-style symbols (NSE:SYM-EQ)."""
    return {f"NSE:{s}-EQ" for s in eligible_at(index_name, on_date)}


def universe_union(index_name: str) -> set[str]:
    """Union of all symbols that were EVER in the index across all snapshots.

    Useful for the SQL pre-load: pull historical prices for this superset
    once, then filter to eligible_at(d) inside the rank step.
    """
    intervals = _load_intervals(index_name)
    return {sym for sym, _, _ in intervals}


if __name__ == "__main__":
    # quick sanity: counts at a few representative dates
    for idx in ("n100", "n500"):
        print(f"\n=== {idx} ===")
        u = universe_union(idx)
        print(f"  union of all snapshots: {len(u)} symbols")
        for d in (date(2017, 1, 1), date(2018, 6, 1), date(2020, 1, 1),
                  date(2023, 1, 1), date(2026, 5, 1)):
            elig = eligible_at(idx, d)
            print(f"  {d.isoformat()}: {len(elig)} eligible")
