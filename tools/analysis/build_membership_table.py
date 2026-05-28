"""Build point-in-time index membership tables from Wayback snapshots.

Input: /tmp/n_snapshots/n{100,500}/n{idx}_YYYY-MM-DD.csv
       Each = niftyindices.com IndexConstituent CSV at that date.

Output: src/data/symbols/n{100,500}_membership.csv
        Columns: symbol,start_date,end_date  (inclusive start, exclusive end)

Rule (LAST-KNOWN-STATE):
  Between snapshot S_i (date d_i) and the next snapshot S_{i+1} (date d_{i+1}),
  the index membership is assumed equal to S_i's constituent list. Symbols
  that appear in S_i but not S_{i+1} are treated as REMOVED at d_{i+1}.
  Symbols that appear in S_{i+1} but not S_i are treated as ADDED at d_{i+1}.

  For any backtest date d:
    eligible(d) = members of the most recent snapshot whose date <= d.
    Before the first snapshot we have no data; we use the first snapshot's
    members and flag those rows with start_date = first snapshot's date.

Limitations DOCUMENTED for caller:
  * n100 has a 4.5-year gap (2019-02 -> 2023-08). Drift in this window not
    captured. Use these results as "approximate point-in-time", not perfect.
  * Snapshot timestamps are Wayback capture dates, not exact NSE rebalance
    effective dates. Membership change attributed to the snapshot date.
"""
import csv
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
SNAP_DIR = Path("/tmp/n_snapshots")
OUT_DIR = ROOT / "src" / "data" / "symbols"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_snapshot(path: Path) -> set[str]:
    """Return set of EQ symbol strings from a niftyindices CSV.

    Resilient to garbage / 0-byte files / HTML error pages that some Wayback
    captures returned. Returns empty set on any parse failure.
    """
    out: set[str] = set()
    try:
        if path.stat().st_size == 0:
            return out
        raw = path.read_bytes()
        if b"\x00" in raw[:200]:           # NUL byte = binary garbage
            return out
        if raw[:15].lower().startswith(b"<!doctype html") or b"<html" in raw[:100].lower():
            return out
        text = raw.decode("utf-8", errors="ignore").splitlines()
        reader = csv.DictReader(text)
        for r in reader:
            sym = (r.get("Symbol") or "").strip()
            series = (r.get("Series") or "").strip()
            if series == "EQ" and sym:
                out.add(sym)
    except Exception as e:
        print(f"  parse err on {path.name}: {e}")
    return out


def build_membership(idx: str):
    print(f"\n=== n{idx} ===")
    snap_dir = SNAP_DIR / f"n{idx}"
    files = sorted(snap_dir.glob(f"n{idx}_*.csv"))
    # Drop empty / failed snapshots
    snaps: list[tuple[date, set[str]]] = []
    for f in files:
        members = parse_snapshot(f)
        if not members:
            print(f"  SKIP empty: {f.name}")
            continue
        date_str = f.stem.split("_")[1]  # n100_2019-02-01 -> 2019-02-01
        d = date.fromisoformat(date_str)
        snaps.append((d, members))
        print(f"  {date_str}: {len(members)} symbols")
    if not snaps:
        return

    # For each (symbol, snapshot_date), record presence.
    # Then derive intervals: for each symbol, intervals are runs of
    # consecutive snapshots in which it appears.
    all_syms = set().union(*(s[1] for s in snaps))
    print(f"  union symbols across all snapshots: {len(all_syms)}")

    intervals: list[tuple[str, date, date]] = []
    SENTINEL_FAR = date(2099, 12, 31)
    SENTINEL_NEAR = date(2010, 1, 1)   # for symbols present in FIRST snapshot,
    # we have no data before that date; assume they were also members earlier
    # (best approximation given Wayback coverage starts mid-history).
    first_snap_date = snaps[0][0]
    for sym in sorted(all_syms):
        in_run = False
        run_start: date | None = None
        for i, (snap_date, members) in enumerate(snaps):
            present = sym in members
            if present and not in_run:
                # If this is the FIRST snapshot and the symbol is present,
                # backdate the start so pre-snapshot backtest dates resolve.
                run_start = SENTINEL_NEAR if snap_date == first_snap_date else snap_date
                in_run = True
            elif not present and in_run:
                # Symbol left index AT this snapshot_date.
                intervals.append((sym, run_start, snap_date))
                in_run = False
                run_start = None
        if in_run:
            # Still present at last snapshot -> extend to far future
            intervals.append((sym, run_start, SENTINEL_FAR))

    out_path = OUT_DIR / f"n{idx}_membership.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "start_date", "end_date"])
        for s, sd, ed in intervals:
            w.writerow([s, sd.isoformat(), ed.isoformat()])
    print(f"  wrote {len(intervals)} intervals -> {out_path}")

    # Show the inclusion/exclusion delta between adjacent snapshots
    for i in range(1, len(snaps)):
        prev_d, prev_m = snaps[i-1]
        d_, m_ = snaps[i]
        added = m_ - prev_m
        removed = prev_m - m_
        print(f"  {prev_d.isoformat()} -> {d_.isoformat()}: "
              f"+{len(added)} -{len(removed)}")


if __name__ == "__main__":
    build_membership("100")
    build_membership("500")
