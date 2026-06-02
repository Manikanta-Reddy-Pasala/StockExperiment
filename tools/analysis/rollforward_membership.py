"""Roll the point-in-time index-membership table forward to TODAY's NSE lists.

WHY: eligible_at() (tools/shared/index_membership) reads
src/data/symbols/n{100,500}_membership.csv. Its leading edge — rows with the
end_date 2099-12-31 sentinel — IS the "current members" set the live universe
guard and the live signal models trust. That table is otherwise built offline
from NSE factsheet PDFs (build_membership_from_nse_factsheets.py, manual), so
when NSE reconstitutes an index (Mar + Sep) the table's current slice goes
stale until someone rebuilds it by hand — exactly the lag that would let the
universe guard block a freshly-added member or pass a freshly-removed one.

This script closes that gap WITHOUT the factsheet PDFs: the current
nifty100.csv / nifty500.csv are already refreshed weekly from niftyindices.com
(data_scheduler.refresh_universe_csvs). After that download, reconcile the
membership table's OPEN intervals against the freshly-downloaded current list:

    left   = open member not in current  -> close its interval (end_date = as_of)
    joined = current member not yet open  -> append (symbol, as_of, sentinel)

The full historical intervals are preserved untouched, so backtests keep their
true PIT history; only the leading edge is rolled forward. Idempotent: a run
with no membership change writes nothing.

Comparison is done in ticker-alias-normalised space (the same _TICKER_ALIAS the
reader applies) so a renamed-but-still-in-index name (e.g. ZOMATO->ETERNAL) is
NOT seen as one name leaving + a different name joining.

SAFETY: refuses to mutate when the current CSV looks broken (member count far
below the index size — likely a partial/failed download), and backs up each
membership file to *.rollforward.bak before writing.

Run:  python3 tools/analysis/rollforward_membership.py [--as-of YYYY-MM-DD] [--dry-run]
Exit: 0 = ok (no change or applied), 1 = aborted on a sanity guard / error.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.index_membership import _TICKER_ALIAS  # noqa: E402

SYMBOLS_DIR = ROOT / "src" / "data" / "symbols"
SENTINEL = "2099-12-31"

# (index_name, current-list CSV, minimum sane member count). The minimum guards
# against a partial/failed niftyindices download silently truncating history.
INDEXES = [
    ("n100", "nifty100.csv", 90),
    ("n500", "nifty500.csv", 480),
]


def _norm(sym: str) -> str:
    """Alias-normalise a bare ticker (mirror index_membership._load_intervals)."""
    sym = sym.strip().upper()
    return _TICKER_ALIAS.get(sym, sym)


def _load_current(csv_path: Path) -> set[str]:
    """Current index members from a niftyindices CSV (Symbol/Series columns).

    Equity series only; skips NSE corporate-action DUMMY placeholder scrips
    (e.g. DUMMYVEDL1-4) which are not tradable. Returns alias-normalised tickers.
    """
    out: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = (row.get("Symbol") or "").strip()
            series = (row.get("Series") or "EQ").strip().upper()
            if not sym or series != "EQ" or sym.upper().startswith("DUMMY"):
                continue
            out.add(_norm(sym))
    return out


def _read_membership(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_membership(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["symbol"], r["start_date"]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol", "start_date", "end_date"])
        w.writeheader()
        w.writerows(rows)


def rollforward_index(index_name: str, current_csv: str, min_count: int,
                      as_of: str, dry_run: bool) -> tuple[int, int, bool]:
    """Reconcile one index. Returns (n_left, n_joined, aborted)."""
    cur_path = SYMBOLS_DIR / current_csv
    mem_path = SYMBOLS_DIR / f"{index_name}_membership.csv"
    if not cur_path.exists() or not mem_path.exists():
        print(f"  [{index_name}] SKIP — missing {cur_path.name} or {mem_path.name}")
        return (0, 0, True)

    current = _load_current(cur_path)
    if len(current) < min_count:
        print(f"  [{index_name}] ABORT — current list has {len(current)} members "
              f"(< {min_count}); refusing to mutate history on a suspect download.")
        return (0, 0, True)

    rows = _read_membership(mem_path)
    # Open intervals (current members per the table) keyed by normalised symbol.
    open_rows = {_norm(r["symbol"]): r for r in rows if r["end_date"] == SENTINEL}
    open_norms = set(open_rows)

    left = open_norms - current          # in table-open but not in live list
    joined = current - open_norms        # in live list but not yet open

    if not left and not joined:
        print(f"  [{index_name}] up to date — {len(open_norms)} members, no change.")
        return (0, 0, False)

    if as_of <= "0":  # never; guarded below
        pass
    # Guard: as_of must post-date every open interval's start (can't close an
    # interval before it began). If violated, the as_of is wrong — abort.
    for norm in left:
        if as_of <= open_rows[norm]["start_date"]:
            print(f"  [{index_name}] ABORT — as_of {as_of} <= start "
                  f"{open_rows[norm]['start_date']} of leaving member "
                  f"{open_rows[norm]['symbol']}; bad as_of date.")
            return (0, 0, True)

    print(f"  [{index_name}] change detected as of {as_of}: "
          f"{len(left)} left {sorted(left)}, {len(joined)} joined {sorted(joined)}")
    if dry_run:
        return (len(left), len(joined), False)

    # Close intervals for names that left.
    for norm in left:
        open_rows[norm]["end_date"] = as_of
    # Open intervals for names that joined (store the current/live symbol).
    for norm in joined:
        rows.append({"symbol": norm, "start_date": as_of, "end_date": SENTINEL})

    shutil.copy2(mem_path, mem_path.with_suffix(".csv.rollforward.bak"))
    _write_membership(mem_path, rows)
    print(f"  [{index_name}] WROTE {mem_path.name} "
          f"(backup {mem_path.name}.rollforward.bak)")
    return (len(left), len(joined), False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat(),
                    help="Effective date for membership changes (default: today).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report changes without writing.")
    a = ap.parse_args()

    print("=" * 72)
    print(f"Roll-forward index membership to current NSE lists (as_of {a.as_of}"
          f"{', DRY-RUN' if a.dry_run else ''})")
    print("=" * 72)

    aborted_any = False
    changed_any = False
    for index_name, current_csv, min_count in INDEXES:
        nl, nj, aborted = rollforward_index(index_name, current_csv, min_count,
                                            a.as_of, a.dry_run)
        aborted_any = aborted_any or aborted
        changed_any = changed_any or bool(nl or nj)

    if aborted_any:
        print("RESULT: one or more indexes ABORTED (see above).")
        return 1
    print(f"RESULT: ok ({'changes applied' if changed_any else 'no change'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
