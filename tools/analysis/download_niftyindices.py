"""Download current NSE index constituents from niftyindices.com — the single
REPLACEMENT for the old NSE get-quotes scraper AND the per-index NSE scrapers
(refresh_nifty100/500/midcap150/smallcap250.py) and the Wayback membership
fetch+parse.

Why this replaces all of them:
  * niftyindices is NOT WAF-blocked (plain HTTPS, any IP incl the VM) — no
    headless Chromium, no residential-IP/laptop dependency.
  * Its constituent CSVs are the authoritative index membership in the EXACT
    `Company Name,Industry,Symbol,Series,ISIN Code` layout the models already
    parse (tools/shared/universes.py), so they drop straight into
    src/data/symbols/*.csv.

Writes:
  * src/data/symbols/{nifty100,nifty500,nifty_midcap150,nifty_smallcap250}.csv
    (the model universe files — replaces refresh_nifty*.py)
  * exports/index_constituents/current/<raw>.csv (archive copy)
  * marks nifty_index_membership (n50/100/200/500) when --load-db

Run: python3 tools/analysis/download_niftyindices.py [--load-db] [--date YYYY-MM-DD]
"""
from __future__ import annotations
import sys, csv, io, argparse, ssl
from pathlib import Path
from datetime import date
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ROOT / "src" / "data" / "symbols"
ARCHIVE = ROOT / "exports" / "index_constituents" / "current"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
BASE = "https://niftyindices.com/IndexConstituent/"

# niftyindices file -> (membership index_name | None, model universe filename | None)
FILES = {
    "ind_nifty50list.csv":          ("n50",  None),
    "ind_nifty100list.csv":         ("n100", "nifty100.csv"),
    "ind_nifty200list.csv":         ("n200", None),
    "ind_nifty500list.csv":         ("n500", "nifty500.csv"),
    "ind_niftymidcap150list.csv":   (None,   "nifty_midcap150.csv"),
    "ind_niftysmallcap250list.csv": (None,   "nifty_smallcap250.csv"),
}


def fetch(fname: str):
    url = BASE + fname
    try:
        import requests
        return requests.get(url, headers={"User-Agent": UA}, timeout=30).text
    except ImportError:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl._create_unverified_context()
        with urlopen(Request(url, headers={"User-Agent": UA}), timeout=30, context=ctx) as r:
            return r.read().decode("utf-8-sig", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-db", action="store_true")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    rev = (date.fromisoformat(a.date) if a.date else date.today()).isoformat()
    SYMBOLS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    membership, counts = [], {}
    for fname, (idx, target) in FILES.items():
        try:
            raw = fetch(fname)
        except Exception as e:
            print(f"  ERR {fname}: {type(e).__name__}: {e}"); continue
        if "Symbol" not in raw.splitlines()[0]:
            print(f"  ERR {fname}: unexpected response (no Symbol header)"); continue
        (ARCHIVE / fname).write_text(raw)
        if target:
            (SYMBOLS / target).write_text(raw)        # drop-in model universe file
        rows = list(csv.DictReader(io.StringIO(raw)))
        counts[idx or target] = len(rows)
        if idx:
            for r in rows:
                if r.get("Symbol") and r.get("Series", "EQ").strip() in ("EQ", ""):
                    membership.append({"index_name": idx, "symbol": r["Symbol"].strip(),
                                       "review_date": rev})
    print(f"downloaded @ {rev}: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  model universes -> {SYMBOLS} | archive -> {ARCHIVE}")

    if a.load_db and membership:
        sys.path.insert(0, str(ROOT))
        from sqlalchemy import text
        from tools.shared.ohlcv_cache import _get_engine
        from tools.analysis.mcap_db import init_tables
        eng = _get_engine(); init_tables(eng)
        with eng.begin() as c:
            c.execute(text("""INSERT INTO nifty_index_membership (index_name,symbol,review_date)
                              VALUES (:index_name,:symbol,:review_date)
                              ON CONFLICT (index_name,symbol,review_date) DO NOTHING"""),
                      membership)
        print(f"  DB: marked {len(membership)} membership rows @ {rev}")


if __name__ == "__main__":
    main()
