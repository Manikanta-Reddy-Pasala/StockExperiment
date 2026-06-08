#!/usr/bin/env python3
"""
Record LIVE NIFTY intraday (5-min) option bars into historical_options
(interval='5m'), so we accumulate REAL intraday 0DTE data going forward.

Fyers history() serves intraday only for CURRENTLY-LIVE contracts (expired ones
return "Invalid symbol"), so this must run DAILY after close: each run captures
that day's intraday bars for the active weekly strikes -- including the expiry-day
0DTE session while the contract is still live. Over weeks this builds the dataset
the 0DTE backtest (opt_0dte.py) needs for a true walk-forward (vs the daily-OHLC
proxy).

Schedule (host cron, VM is UTC; 15:35 IST = 10:05 UTC), Mon-Fri:
    10 10 * * 1-5  docker exec trading_system_app \
        python /app/tools/options/record_intraday_options.py >> /var/log/rec_opt.log 2>&1

Usage: python record_intraday_options.py [--days N] [--strikes K] [--interval 5m]
"""
import argparse, sys, time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, "/app")
import psycopg
from src.services.brokers.fyers_service import FyersService

DB = "postgresql://trader:trader_password@database:5432/trading_system"
IST = timezone(timedelta(hours=5, minutes=30))
MW = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
      10: "O", 11: "N", 12: "D"}
MON3U = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN", 7: "JUL",
         8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}
STRIKE_STEP = 50
EXPIRY_WD = 1   # NIFTY weekly expiry = Tuesday (post Sep-2025)


def is_monthly(exp):
    """True if exp is the last <EXPIRY_WD> of its month (=> monthly symbol fmt)."""
    return (exp + timedelta(days=7)).month != exp.month


def fyers_symbol(strike, exp, kind):
    yy = exp.strftime("%y")
    if is_monthly(exp):
        return f"NIFTY{yy}{MON3U[exp.month]}{strike}{kind}"
    return f"NIFTY{yy}{MW[exp.month]}{exp.day:02d}{strike}{kind}"


def next_expiries(today, n=2):
    out, d = [], today
    while len(out) < n + 1:
        if d.weekday() == EXPIRY_WD and d >= today:
            out.append(d)
        d += timedelta(days=1)
    return out[:n]


def get_spot(svc, day):
    r = svc.history(user_id=1, symbol="NIFTY50-INDEX", exchange="NSE",
                    interval="5m", start_date=str(day - timedelta(days=4)),
                    end_date=str(day)) or {}
    cs = r.get("candles") or r.get("data", {}).get("candles")
    if not cs:
        return None
    last = cs[-1]
    return float(last["close"]) if isinstance(last, dict) else float(last[4])


def upsert(cur, rows):
    cur.executemany(
        "INSERT INTO historical_options "
        "(symbol,underlying,expiry,strike,opt_type,interval,timestamp,"
        " candle_time,open,high,low,close,volume) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (symbol,interval,timestamp) DO UPDATE SET "
        "open=EXCLUDED.open,high=EXCLUDED.high,low=EXCLUDED.low,"
        "close=EXCLUDED.close,volume=EXCLUDED.volume", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1, help="trailing days to pull")
    ap.add_argument("--strikes", type=int, default=25, help="ATM +/- K strikes")
    ap.add_argument("--interval", default="5m")
    a = ap.parse_args()

    svc = FyersService()
    today = datetime.now(IST).date()
    spot = get_spot(svc, today)
    if not spot:
        print("ERROR: no NIFTY spot; token may be invalid"); sys.exit(1)
    atm = round(spot / STRIKE_STEP) * STRIKE_STEP
    exps = next_expiries(today, 2)
    start = str(today - timedelta(days=a.days - 1))
    end = str(today)
    print(f"{datetime.now(IST):%Y-%m-%d %H:%M} spot={spot} atm={atm} "
          f"expiries={[str(e) for e in exps]} window {start}..{end}")

    conn = psycopg.connect(DB); cur = conn.cursor()
    total = 0
    for exp in exps:
        for i in range(-a.strikes, a.strikes + 1):
            strike = atm + i * STRIKE_STEP
            for kind in ("CE", "PE"):
                sym = fyers_symbol(strike, exp, kind)
                try:
                    r = svc.history(user_id=1, symbol=sym, exchange="NSE",
                                    interval=a.interval, start_date=start,
                                    end_date=end) or {}
                except Exception as e:
                    continue
                cs = r.get("candles") or r.get("data", {}).get("candles")
                if not cs:
                    continue
                rows = []
                for c in cs:
                    if isinstance(c, dict):
                        ts = int(c["timestamp"]); o, h, l, cl = (float(c["open"]),
                            float(c["high"]), float(c["low"]), float(c["close"]))
                        v = int(float(c.get("volume", 0)))
                    else:
                        ts = int(c[0]); o, h, l, cl, v = (float(c[1]), float(c[2]),
                            float(c[3]), float(c[4]), int(float(c[5])))
                    ct = datetime.fromtimestamp(ts, IST).replace(tzinfo=None)
                    rows.append((f"NSE:{sym}", "NIFTY", exp, strike, kind,
                                 a.interval, ts, ct, o, h, l, cl, v))
                if rows:
                    upsert(cur, rows); total += len(rows)
                time.sleep(0.05)
        conn.commit()
    conn.commit()
    print(f"DONE recorded {total} {a.interval} bars across {len(exps)} expiries")
    conn.close()


if __name__ == "__main__":
    main()
