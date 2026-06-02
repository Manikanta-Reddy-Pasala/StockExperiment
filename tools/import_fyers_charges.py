"""Import ACTUAL Fyers charges from the account CSV exports → broker_charges_daily.

Fyers has no charges API, so the real per-day charges come from the manual CSV
downloads (Reports → Charges, and Reports → Ledger). This loads them as the
SOURCE OF TRUTH for dates they cover; the formula estimate
(tools/live/broker_charges) is then used ONLY for dates AFTER the last imported
day (e.g. today, before the next export). See model_ledger_service
.get_actual_plus_estimated_charges().

Charges CSV (Reports → Charges) per-day columns:
  Trade Date, Day's turnover (ICAI), Total, Brokerage, STT/CTT, IPFT,
  Stamp duty, GST, Exchange turnover(=exch txn charge), SEBI, CM charge
  → statutory_total = the 'Total' column (brokerage+STT+stamp+GST+exch+SEBI).

Ledger CSV (Reports → Ledger) — pulls the NON-trading broker fees that the
charges report omits (DP transaction charges, Call & Trade / square-off fees,
IGST on those) into other_fees per date.

  total = statutory_total + other_fees.

Idempotent UPSERT by trade_date. Run (inside the app container, CSVs copied in):
  python tools/import_fyers_charges.py --charges <charges.csv> [--ledger <ledger.csv>]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text                              # noqa: E402
from src.models.database import get_database_manager     # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS broker_charges_daily (
    trade_date      DATE PRIMARY KEY,
    brokerage       NUMERIC(14,2) DEFAULT 0,
    stt             NUMERIC(14,2) DEFAULT 0,
    stamp           NUMERIC(14,2) DEFAULT 0,
    gst             NUMERIC(14,2) DEFAULT 0,
    exchange_txn    NUMERIC(14,2) DEFAULT 0,
    sebi            NUMERIC(14,2) DEFAULT 0,
    ipft            NUMERIC(14,2) DEFAULT 0,
    cm              NUMERIC(14,2) DEFAULT 0,
    statutory_total NUMERIC(14,2) DEFAULT 0,
    other_fees      NUMERIC(14,2) DEFAULT 0,
    total           NUMERIC(14,2) DEFAULT 0,
    source          VARCHAR(128),
    imported_at     TIMESTAMP DEFAULT now()
);
"""


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip() or 0)
    except Exception:
        return 0.0


def parse_charges(path: Path) -> dict:
    """charges CSV → {date_iso: {brokerage, stt, stamp, gst, exchange_txn, sebi,
    ipft, cm, statutory_total}}. Finds the per-day table by its header row."""
    out: dict = {}
    rows = list(csv.reader(path.open(encoding="utf-8")))
    hdr_idx = None
    for i, r in enumerate(rows):
        if r and r[0].strip() == "Trade Date":
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("charges CSV: per-day header 'Trade Date' not found")
    # cols: Trade Date, turnover, Total, Brokerage, STT/CTT, IPFT, Stamp, GST,
    #       Exchange txn, SEBI, CM
    for r in rows[hdr_idx + 1:]:
        if not r or not r[0].strip():
            continue
        try:
            d = datetime.strptime(r[0].strip(), "%Y-%m-%d").date().isoformat()
        except Exception:
            continue
        out[d] = {
            "statutory_total": _num(r[2]),
            "brokerage": _num(r[3]),
            "stt": _num(r[4]),
            "ipft": _num(r[5]),
            "stamp": _num(r[6]),
            "gst": _num(r[7]),
            "exchange_txn": _num(r[8]),
            "sebi": _num(r[9]),
            "cm": _num(r[10]) if len(r) > 10 else 0.0,
        }
    return out


def parse_ledger_other_fees(path: Path) -> dict:
    """ledger CSV → {date_iso: other_fees} = DP + Call&Trade/square-off + IGST
    debits (broker fees the charges report omits)."""
    out: dict = {}
    rows = list(csv.reader(path.open(encoding="utf-8")))
    hdr_idx = None
    for i, r in enumerate(rows):
        if r and r[0].strip() == "Date" and "Transaction type" in (r[1] if len(r) > 1 else ""):
            hdr_idx = i
            break
    if hdr_idx is None:
        return out
    FEE_HINTS = ("dp transaction", "call & trade", "square-off", "square off",
                 "igst on call")
    for r in rows[hdr_idx + 1:]:
        if len(r) < 6 or not r[0].strip():
            continue
        desc = (r[2] or "").lower()
        if not any(h in desc for h in FEE_HINTS):
            continue
        try:
            d = datetime.strptime(r[0].strip(), "%d %b %Y").date().isoformat()
        except Exception:
            continue
        out[d] = out.get(d, 0.0) + _num(r[3])  # debit amount
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charges", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    charges = parse_charges(Path(a.charges))
    other = parse_ledger_other_fees(Path(a.ledger)) if a.ledger else {}
    days = sorted(set(charges) | set(other))

    db = get_database_manager()
    with db.get_session() as s:
        s.execute(text(DDL))
        s.commit()
    grand = 0.0
    for d in days:
        c = charges.get(d, {})
        of = round(other.get(d, 0.0), 2)
        stat = round(c.get("statutory_total", 0.0), 2)
        total = round(stat + of, 2)
        grand += total
        print(f"  {d}  statutory={stat:>8}  other_fees={of:>7}  total={total:>8}")
        if a.dry_run:
            continue
        with db.get_session() as s:
            s.execute(text("""
                INSERT INTO broker_charges_daily
                  (trade_date,brokerage,stt,stamp,gst,exchange_txn,sebi,ipft,cm,
                   statutory_total,other_fees,total,source,imported_at)
                VALUES (:d,:brk,:stt,:stamp,:gst,:exch,:sebi,:ipft,:cm,:stat,:of,:tot,:src,now())
                ON CONFLICT (trade_date) DO UPDATE SET
                  brokerage=:brk, stt=:stt, stamp=:stamp, gst=:gst,
                  exchange_txn=:exch, sebi=:sebi, ipft=:ipft, cm=:cm,
                  statutory_total=:stat, other_fees=:of, total=:tot,
                  source=:src, imported_at=now()
            """), {
                "d": d, "brk": c.get("brokerage", 0), "stt": c.get("stt", 0),
                "stamp": c.get("stamp", 0), "gst": c.get("gst", 0),
                "exch": c.get("exchange_txn", 0), "sebi": c.get("sebi", 0),
                "ipft": c.get("ipft", 0), "cm": c.get("cm", 0),
                "stat": stat, "of": of, "tot": total,
                "src": Path(a.charges).name,
            })
            s.commit()

    print(f"{'[DRY-RUN] ' if a.dry_run else ''}imported {len(days)} day(s); "
          f"through {days[-1] if days else '—'}; actual total ₹{round(grand,2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
