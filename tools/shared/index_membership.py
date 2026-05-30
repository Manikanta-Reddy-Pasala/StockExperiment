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

# Ticker-rename map: membership CSV (built from period-correct Wayback NSE
# snapshots) records a stock under the symbol it traded as AT THAT TIME, but
# historical_data was backfilled by Fyers under the CURRENT symbol over the
# whole price history. Without this map the renamed stock resolves to a symbol
# the price DB has never heard of and is SILENTLY dropped from the ranking
# universe — in 2022-2023 that lost ~10 real Nifty 100 members (incl. Tata
# Motors, Zomato, Samvardhana Motherson), several of them strong momentum
# names, corrupting the backtest universe. Verified 2026-05-29 against
# localhost trading_system DB (each RHS has full 2022-05 -> 2026-05 coverage
# except LTIM, see note). Keys = old/period symbol, values = current Fyers
# symbol. Applied at load time so eligible_at / universe_union (n100/n200/n500
# + every backtest) all resolve to the symbol the price DB actually stores.
_TICKER_ALIAS: dict[str, str] = {
    "TATAMOTORS": "TMPV",        # Tata Motors demerger -> Tata Motors PV (2025)
    "ZOMATO": "ETERNAL",         # Zomato renamed Eternal (2025)
    "MOTHERSUMI": "MOTHERSON",   # Motherson Sumi -> Samvardhana Motherson
    "SRTRANSFIN": "SHRIRAMFIN",  # Shriram Transport -> Shriram Finance merger
    "CADILAHC": "ZYDUSLIFE",     # Cadila Healthcare -> Zydus Lifesciences
    "L&TFH": "LTF",              # L&T Finance Holdings -> L&T Finance
    "MCDOWELL-N": "UNITDSPR",    # United Spirits ticker change
    "ADANITRANS": "ADANIENSOL",  # Adani Transmission -> Adani Energy Solutions
    "INFRATEL": "INDUSTOWER",    # Bharti Infratel -> Indus Towers merger
    "IBULHSGFIN": "SAMMAANCAP",  # Indiabulls Housing -> Sammaan Capital
    # LTIM price history only exists from 2024-12 in the DB; pre-2024 LTIM
    # (post Mindtree merger, Nov-2022) remains an accepted partial gap.
    "MINDTREE": "LTIM",          # Mindtree -> LTIMindtree (LTI merger, 2022)
    # --- n200 renames (added 2026-05-29 to close the n200 2022-2023 gap) ---
    "GMRINFRA": "GMRAIRPORT",    # GMR Infrastructure -> GMR Airports
    "TATAGLOBAL": "TATACONSUM",  # Tata Global Beverages -> Tata Consumer
    "TV18BRDCST": "NETWORK18",   # TV18 Broadcast -> Network18 (merger)
    "JUBILANT": "JUBLPHARMA",    # Jubilant Life Sci -> Jubilant Pharmova (demerge)
    "STRTECH": "STLTECH",        # Sterlite Tech ticker change (DB partial: 2024-12+)
    "LTI": "LTIM",               # L&T Infotech -> LTIMindtree (DB partial: 2024-12+)
}


@lru_cache(maxsize=4)
def _load_intervals(index_name: str) -> list[tuple[str, date, date]]:
    """Parse the membership CSV into a list of (symbol, start, end) tuples.

    Cached so the file is read once per process. Sorted by symbol then
    start_date so dedupe / scans are stable. Symbols are mapped through
    _TICKER_ALIAS so renamed stocks resolve to the symbol the price DB stores
    (see _TICKER_ALIAS docstring). A renamed symbol may then share intervals
    with its native later listing (e.g. ADANITRANS->ADANIENSOL + native
    ADANIENSOL); eligible_at / universe_union return sets so this de-dups.
    """
    path = SYMBOLS_DIR / f"{index_name}_membership.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Index membership file not found: {path}. "
            f"Run tools/analysis/download_niftyindices.py / parse_nse_index_pdfs.py first."
        )
    out: list[tuple[str, date, date]] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sym = r["symbol"].strip()
            sym = _TICKER_ALIAS.get(sym, sym)
            out.append((
                sym,
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
