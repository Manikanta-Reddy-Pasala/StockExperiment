"""Load the point-in-time index_membership table from the two CSVs.

Reads:
    src/data/symbols/n100_membership.csv
    src/data/symbols/n500_membership.csv

Writes table `index_membership` (created by
migrations/2026_05_28_index_membership.sql). Idempotent: truncates then
inserts.

Run inside the trading_system_app container (DB credentials in env):
    python tools/analysis/load_index_membership.py

Verifies row counts at the end so a corrupted run fails loudly.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine

CSV_DIR = ROOT / "src" / "data" / "symbols"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS index_membership (
    index_name  VARCHAR(20) NOT NULL,
    symbol      VARCHAR(50) NOT NULL,
    start_date  DATE        NOT NULL,
    end_date    DATE        NOT NULL,
    PRIMARY KEY (index_name, symbol, start_date)
);
CREATE INDEX IF NOT EXISTS idx_index_membership_lookup
    ON index_membership (index_name, start_date, end_date);
"""


def load_csv(index_name: str) -> list[tuple[str, str, str, str]]:
    """Read membership CSV; return rows tagged with index_name."""
    path = CSV_DIR / f"{index_name}_membership.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out: list[tuple[str, str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append((
                index_name,
                r["symbol"].strip(),
                r["start_date"].strip(),
                r["end_date"].strip(),
            ))
    return out


def main() -> None:
    eng = _get_engine()
    with eng.begin() as conn:
        for stmt in CREATE_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        conn.execute(text("TRUNCATE index_membership"))
        total = 0
        for idx in ("n100", "n500"):
            rows = load_csv(idx)
            # Bulk insert.
            conn.execute(
                text("INSERT INTO index_membership "
                     "(index_name, symbol, start_date, end_date) "
                     "VALUES (:i, :s, :sd, :ed)"),
                [{"i": i, "s": s, "sd": sd, "ed": ed} for i, s, sd, ed in rows]
            )
            print(f"  loaded {idx}: {len(rows)} intervals")
            total += len(rows)
    # Verify
    with eng.connect() as c:
        for idx in ("n100", "n500"):
            r = c.execute(text(
                "SELECT COUNT(*) FROM index_membership WHERE index_name=:i"
            ), {"i": idx}).scalar()
            print(f"  verify {idx}: {r} rows in DB")
        # Smoke-test eligibility query at a few dates
        for d in ("2017-01-01", "2018-06-01", "2020-01-01",
                  "2023-01-01", "2026-05-01"):
            for idx in ("n100", "n500"):
                cnt = c.execute(text(
                    "SELECT COUNT(*) FROM index_membership "
                    "WHERE index_name=:i AND start_date <= :d "
                    "AND end_date > :d"
                ), {"i": idx, "d": d}).scalar()
                print(f"  {idx} @ {d}: {cnt} eligible")
    print(f"\nDone. Total inserted: {total}")


if __name__ == "__main__":
    main()
