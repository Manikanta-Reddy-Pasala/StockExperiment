"""Download current NSE index constituents from niftyindices.com — the
REPLACEMENT for the old NSE get-quotes scraper (nse_mcap_scraper.py).

Why this replaces the scraper:
  * niftyindices is NOT WAF-blocked (plain HTTPS works from ANY IP incl the VM)
    — no headless Chromium, no residential-IP / laptop dependency.
  * The constituent CSVs are the authoritative index membership, refreshed by
    NSE — exactly what we need for clean PIT universes.
  * Per-stock free-float mcap (what the scraper fetched) is NOT in these CSVs,
    but the climber overlay uses a price-derived proxy from the DB anyway, and
    real historical FF-mcap is loaded from the index factsheets
    (parse_nse_index_pdfs.py). So no mcap scrape is needed at all.

Pulls ind_nifty{50,100,200,500}list.csv -> exports/index_constituents/current/
+ marks nifty_index_membership (review_date = run date).

Run: python3 tools/analysis/download_niftyindices.py [--load-db] [--date YYYY-MM-DD]
"""
from __future__ import annotations
import sys, csv, io, argparse, ssl
from pathlib import Path
from datetime import date
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exports" / "index_constituents" / "current"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
INDICES = {"n50": "ind_nifty50list.csv", "n100": "ind_nifty100list.csv",
           "n200": "ind_nifty200list.csv", "n500": "ind_nifty500list.csv"}
BASE = "https://niftyindices.com/IndexConstituent/"


def fetch_csv(fname: str):
    url = BASE + fname
    # Prefer requests (bundles certifi); fall back to urllib w/ certifi, then to
    # an unverified context (niftyindices is a public report host).
    try:
        import requests
        raw = requests.get(url, headers={"User-Agent": UA}, timeout=30).text
    except ImportError:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl._create_unverified_context()
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=30, context=ctx) as r:
            raw = r.read().decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    return rows, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-db", action="store_true")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    rev = date.fromisoformat(a.date) if a.date else date.today()
    OUT.mkdir(parents=True, exist_ok=True)

    membership = []
    counts = {}
    for idx, fname in INDICES.items():
        try:
            rows, raw = fetch_csv(fname)
        except Exception as e:
            print(f"  ERR {idx} ({fname}): {type(e).__name__}: {e}"); continue
        (OUT / fname).write_text(raw)
        syms = [r["Symbol"].strip() for r in rows
                if r.get("Symbol") and r.get("Series", "EQ").strip() in ("EQ", "")]
        counts[idx] = len(syms)
        for s in syms:
            membership.append({"index_name": idx, "symbol": s, "review_date": rev.isoformat()})
    print(f"downloaded {len(counts)} indices @ {rev}: "
          + " ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  saved -> {OUT}")

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
