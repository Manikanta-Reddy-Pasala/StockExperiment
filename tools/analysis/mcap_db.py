"""Persist market-cap snapshots + Nifty index membership to Postgres for a
permanent historical track (so future runs accumulate a real time-series
instead of overwriting one CSV).

Two tables:
  market_cap_history     — one row per (symbol, snapshot_date): total + free-float
                           market cap (₹ Cr), LTP, derived free-float shares.
                           Loaded from exports/nse_mcap.csv after each scrape.
  nifty_index_membership — one row per (index_name, symbol, review_date): the
                           full constituent list snapshotted every NSE review
                           (effective end of March / end of September).

CLI:
  python3 tools/analysis/mcap_db.py init                 # create tables
  python3 tools/analysis/mcap_db.py load-mcap [--date YYYY-MM-DD] [--csv path]
  python3 tools/analysis/mcap_db.py snapshot-membership [--date YYYY-MM-DD]
  python3 tools/analysis/mcap_db.py status               # row counts + latest dates
"""
from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.index_membership import eligible_at

MCAP_CSV = ROOT / "exports" / "nse_mcap.csv"
INDICES = ["n100", "n500"]   # CSV-backed PIT membership we can snapshot

DDL = [
    """
    CREATE TABLE IF NOT EXISTS market_cap_history (
        symbol         TEXT             NOT NULL,
        snapshot_date  DATE             NOT NULL,
        total_mcap_cr  DOUBLE PRECISION,
        ff_mcap_cr     DOUBLE PRECISION,
        ltp            DOUBLE PRECISION,
        ff_shares      DOUBLE PRECISION,
        source         TEXT             DEFAULT 'nse_scrape',
        captured_at    TIMESTAMP        DEFAULT now(),
        PRIMARY KEY (symbol, snapshot_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_mcap_hist_date ON market_cap_history (snapshot_date)",
    """
    CREATE TABLE IF NOT EXISTS nifty_index_membership (
        index_name   TEXT       NOT NULL,
        symbol       TEXT       NOT NULL,
        review_date  DATE       NOT NULL,
        captured_at  TIMESTAMP  DEFAULT now(),
        PRIMARY KEY (index_name, symbol, review_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_idx_memb_date ON nifty_index_membership (index_name, review_date)",
]


def init_tables(eng):
    with eng.begin() as c:
        for stmt in DDL:
            c.execute(text(stmt))
    print("tables ready: market_cap_history, nifty_index_membership")


def load_mcap(eng, snapshot_date: date, csv_path: Path):
    """Upsert one mcap snapshot from the scrape CSV. Idempotent per date."""
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); return 0
    rows = []
    for r in csv.DictReader(open(csv_path)):
        try:
            ff = float(r["ff_mcap_cr"]) if r.get("ff_mcap_cr") else None
        except ValueError:
            ff = None
        try:
            tot = float(r["total_mcap_cr"]) if r.get("total_mcap_cr") else None
        except ValueError:
            tot = None
        try:
            ltp = float(r["ltp"]) if r.get("ltp") else None
        except ValueError:
            ltp = None
        ff_shares = (ff * 1e7 / ltp) if (ff and ltp and ltp > 0) else None
        rows.append({"symbol": r["symbol"], "snapshot_date": snapshot_date,
                     "total_mcap_cr": tot, "ff_mcap_cr": ff, "ltp": ltp,
                     "ff_shares": ff_shares, "source": "nse_scrape"})
    sql = text("""
        INSERT INTO market_cap_history
            (symbol, snapshot_date, total_mcap_cr, ff_mcap_cr, ltp, ff_shares, source)
        VALUES (:symbol, :snapshot_date, :total_mcap_cr, :ff_mcap_cr, :ltp, :ff_shares, :source)
        ON CONFLICT (symbol, snapshot_date) DO UPDATE SET
            total_mcap_cr = EXCLUDED.total_mcap_cr,
            ff_mcap_cr    = EXCLUDED.ff_mcap_cr,
            ltp           = EXCLUDED.ltp,
            ff_shares     = EXCLUDED.ff_shares,
            captured_at   = now()
    """)
    with eng.begin() as c:
        c.execute(sql, rows)
    with_ff = sum(1 for r in rows if r["ff_mcap_cr"])
    print(f"loaded {len(rows)} symbols ({with_ff} with FF-mcap) @ {snapshot_date}")
    return len(rows)


def snapshot_membership(eng, review_date: date):
    """Snapshot the full current constituent list of each index."""
    total = 0
    for idx in INDICES:
        members = sorted(eligible_at(idx, review_date))
        if not members:
            print(f"  {idx}: 0 members (skipped)"); continue
        rows = [{"index_name": idx, "symbol": s, "review_date": review_date}
                for s in members]
        sql = text("""
            INSERT INTO nifty_index_membership (index_name, symbol, review_date)
            VALUES (:index_name, :symbol, :review_date)
            ON CONFLICT (index_name, symbol, review_date) DO NOTHING
        """)
        with eng.begin() as c:
            c.execute(sql, rows)
        print(f"  {idx}: {len(members)} members @ {review_date}")
        total += len(members)
    return total


def status(eng):
    with eng.connect() as c:
        for tbl, datecol in (("market_cap_history", "snapshot_date"),
                             ("nifty_index_membership", "review_date")):
            try:
                n = c.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
                dates = c.execute(text(
                    f"SELECT DISTINCT {datecol} FROM {tbl} ORDER BY {datecol} DESC LIMIT 5"
                )).fetchall()
                print(f"{tbl}: {n} rows | recent {datecol}s: "
                      + ", ".join(str(d[0]) for d in dates))
            except Exception as e:
                print(f"{tbl}: not created yet ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p_load = sub.add_parser("load-mcap")
    p_load.add_argument("--date", default=None)
    p_load.add_argument("--csv", default=str(MCAP_CSV))
    p_mem = sub.add_parser("snapshot-membership")
    p_mem.add_argument("--date", default=None)
    sub.add_parser("status")
    a = ap.parse_args()

    eng = _get_engine()
    if a.cmd == "init":
        init_tables(eng)
    elif a.cmd == "load-mcap":
        init_tables(eng)
        d = date.fromisoformat(a.date) if a.date else date.today()
        load_mcap(eng, d, Path(a.csv))
    elif a.cmd == "snapshot-membership":
        init_tables(eng)
        d = date.fromisoformat(a.date) if a.date else date.today()
        snapshot_membership(eng, d)
    elif a.cmd == "status":
        status(eng)


if __name__ == "__main__":
    main()
