"""Fetch historical NSE Nifty 100 / 500 constituent CSVs from web.archive.org.

Why: niftyindices.com only serves the CURRENT constituent CSV; historical
versions are not addressable on their site. Wayback's CDX API exposes every
capture of those URLs that the Internet Archive happens to have taken.

Run:
    python tools/analysis/fetch_wayback_index_snapshots.py [--out /tmp/n_snapshots]

Output:
    {out}/n100/n100_YYYY-MM-DD.csv
    {out}/n500/n500_YYYY-MM-DD.csv

Coverage is sparse and uneven (Wayback captures the CSVs only when it crawls
the page). Use tools/analysis/build_membership_table.py downstream to derive
intervals; it documents the resulting gaps.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

URL_PATTERNS = {
    "100": [
        "niftyindices.com/IndexConstituent/ind_nifty100list.csv",
        "archives.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
    "500": [
        "niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ],
}
CDX_BASE = "https://web.archive.org/cdx/search/cdx"


def cdx_query(url_pat: str, frm: int = 2016, to: int = 2026) -> list:
    """Hit the Wayback CDX API. Collapse by content digest so each unique
    capture appears once. Returns raw rows minus the header.
    """
    cmd = ["curl", "-s", "--max-time", "120",
           f"{CDX_BASE}?url={url_pat}&from={frm}&to={to}"
           "&output=json&filter=statuscode:200&collapse=digest"]
    try:
        out = subprocess.check_output(cmd, timeout=130).decode()
        rows = json.loads(out)
        return rows[1:] if len(rows) > 1 else []
    except Exception as e:
        print(f"  CDX err for {url_pat}: {e}")
        return []


def fetch_all(index_short: str, out_dir: Path) -> None:
    """Download every unique Wayback snapshot of the index constituent CSV.

    De-duplicates by (capture-day, digest). Multiple captures on the same
    day with identical content are kept once.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list = []
    for pat in URL_PATTERNS[index_short]:
        rs = cdx_query(pat)
        print(f"  {pat}: {len(rs)} captures")
        rows.extend(rs)
    # Dedup by (yyyymmdd, content-digest).
    seen: set = set()
    uniq: list = []
    for r in rows:
        ts, dig = r[1], r[5]
        key = (ts[:8], dig)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    uniq.sort(key=lambda r: r[1])
    print(f"  unique snapshots: {len(uniq)}")

    for r in uniq:
        ts, original_url = r[1], r[2]
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        target = out_dir / f"n{index_short}_{date_str}.csv"
        if target.exists() and target.stat().st_size > 1000:
            print(f"    {date_str}: cached ({target.stat().st_size} bytes)")
            continue
        # `id_` flag asks Wayback for the raw archived bytes (no toolbar).
        arch_url = f"https://web.archive.org/web/{ts}id_/{original_url}"
        subprocess.run(["curl", "-s", "--max-time", "30",
                        "-A", "Mozilla/5.0",
                        "-o", str(target), arch_url], timeout=35)
        sz = target.stat().st_size if target.exists() else 0
        print(f"    {date_str}: {sz} bytes")
        time.sleep(0.3)  # be polite to Wayback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/n_snapshots",
                    help="Output directory (default /tmp/n_snapshots)")
    a = ap.parse_args()
    out_root = Path(a.out)
    for idx in ("100", "500"):
        print(f"=== Nifty {idx} ===")
        fetch_all(idx, out_root / f"n{idx}")


if __name__ == "__main__":
    main()
