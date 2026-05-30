"""Download historical INDEX OHLC (NIFTY 50 / 100 / 500) from niftyindices.com
and load into historical_data. Gives real index levels back to 2016 (our DB's
NSE:NIFTY50-INDEX only had 2023+, and 100/500 indices were missing entirely) —
so regime detection + index benchmarking use the REAL index, not a basket proxy.

Source: niftyindices Backpage API (not WAF-blocked). Pulled in yearly chunks.
Loads NSE:NIFTY{50,100,500}-INDEX rows, data_source='niftyindices'.

Run: python3 tools/analysis/download_nifty_index_history.py [--from 2016-01-01]
       [--to 2026-05-30]    (set DATABASE_URL to target prod)
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
API = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"
INDICES = {"NIFTY 50": "NSE:NIFTY50-INDEX", "NIFTY 100": "NSE:NIFTY100-INDEX",
           "NIFTY 500": "NSE:NIFTY500-INDEX"}
MON = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7,
       "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def _fmt(d: date):
    return d.strftime("%d-%b-%Y")


def fetch(index_name, start, end, tries=3):
    import requests
    body = {"cinfo": json.dumps({"name": index_name, "startDate": _fmt(start),
                                 "endDate": _fmt(end), "indexName": index_name})}
    last = None
    for t in range(tries):
        try:
            r = requests.post(API, json=body, timeout=90, headers={
                "User-Agent": UA, "Content-Type": "application/json; charset=UTF-8",
                "Referer": "https://www.niftyindices.com/reports/historical-data"})
            data = json.loads(r.json()["d"])
            break
        except Exception as e:
            last = e
            if t < tries - 1:
                time.sleep(3)
    else:
        raise last
    out = []
    for row in data:
        hd = row.get("HistoricalDate", "").strip()       # "31 Jan 2024"
        try:
            dd, mm, yy = hd.split()
            d = date(int(yy), MON[mm], int(dd))
            out.append({"date": d, "open": float(row["OPEN"]), "high": float(row["HIGH"]),
                        "low": float(row["LOW"]), "close": float(row["CLOSE"])})
        except (ValueError, KeyError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2016-01-01")
    ap.add_argument("--to", dest="end", default=date.today().isoformat())
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    eng = _get_engine()
    upsert = text("""
        INSERT INTO historical_data (symbol, date, timestamp, open, high, low, close, volume, data_source)
        VALUES (:symbol, :date, :ts, :open, :high, :low, :close, 0, 'niftyindices')
        ON CONFLICT (symbol, date) DO UPDATE SET
          open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close
    """)
    for iname, sym in INDICES.items():
        rows = []
        y = start.year
        while y <= end.year:
            ys = max(start, date(y, 1, 1)); ye = min(end, date(y, 12, 31))
            try:
                chunk = fetch(iname, ys, ye)
                rows.extend(chunk)
                print(f"  {iname} {y}: {len(chunk)} bars")
            except Exception as e:
                print(f"  {iname} {y}: ERR {type(e).__name__}: {e}")
            y += 1
        if rows:
            with eng.begin() as c:
                c.execute(upsert, [{"symbol": sym, "ts": datetime.combine(r["date"], datetime.min.time()), **r}
                                   for r in rows])
            print(f"{sym}: upserted {len(rows)} bars ({rows[-1]['date']}..{rows[0]['date']})")


if __name__ == "__main__":
    main()
