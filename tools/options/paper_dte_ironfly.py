#!/usr/bin/env python3
"""
PAPER-TRADING model: 0DTE NIFTY iron-fly (defined risk).

Strategy (backtest: ~156% CAGR / 77% WR / loss hard-capped at -24%, see
stockexp-options-model memory):
  - Only on NIFTY weekly EXPIRY days (Tuesday post Sep-2025).
  - At OPEN: sell 1.2%-OTM CE + 1.2%-OTM PE; buy wings 2% beyond each (defined
    risk). Net credit collected.
  - Hard stop at 2x credit loss (intraday, via 5m bar highs).
  - Settle at CLOSE.
  - PAPER ONLY: no broker orders. Logs to paper_dte_trades; compares to backtest.

Modes (run by cron on the VM):
  --enter   ~09:20 IST on expiry day: price legs at open, record OPEN paper trade
  --settle  ~15:25 IST on expiry day: mark out at close / stop, record pnl
  --report  print all paper trades + running WR / return
  --force   ignore the expiry-day gate (for testing)

Prices via Fyers history 5m (live contracts). No real capital at risk.
"""
import argparse, sys
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
STEP = 50
EXPIRY_WD = 1            # Tuesday
OTM = 0.012              # short strikes 1.2% OTM
WING = 0.02              # long wings 2% beyond shorts
STOP_MULT = 2.0          # exit if loss >= 2x credit
SLIP = 0.02
MARGIN_FLOOR = 0.25      # floor margin at 25% of wing width


def is_monthly(exp):
    return (exp + timedelta(days=7)).month != exp.month


def sym(strike, exp, kind):
    yy = exp.strftime("%y")
    if is_monthly(exp):
        return f"NIFTY{yy}{MON3U[exp.month]}{strike}{kind}"
    return f"NIFTY{yy}{MW[exp.month]}{exp.day:02d}{strike}{kind}"


def bars(svc, s, day):
    # Fyers needs end > start; [day, day+1) = that day's session.
    r = svc.history(user_id=1, symbol=s, exchange="NSE", interval="5m",
                    start_date=str(day), end_date=str(day + timedelta(days=1))) or {}
    cs = r.get("candles") or r.get("data", {}).get("candles")
    out = []
    for c in (cs or []):
        if isinstance(c, dict):
            out.append((float(c["open"]), float(c["high"]), float(c["low"]),
                        float(c["close"])))
        else:
            out.append((float(c[1]), float(c[2]), float(c[3]), float(c[4])))
    return out


def spot_atm(svc, day):
    b = bars(svc, "NIFTY50-INDEX", day)
    if not b:
        b = bars(svc, "NIFTY50-INDEX", day - timedelta(days=1))
    if not b:
        return None, None
    spot = b[0][0]                      # today's open
    return spot, round(spot / STEP) * STEP


def _is_trading_day(d):
    """NSE trading-day check via the project calendar; fall back to weekday."""
    try:
        from tools.shared.nse_calendar import is_trading_day
        return is_trading_day(d)
    except Exception:
        return d.weekday() < 5


def this_expiry(today):
    """NIFTY weekly expiry = Tuesday; if that Tuesday is an NSE holiday, NSE
    moves the expiry to the PREVIOUS trading day. Returns the actual expiry
    date for the current weekly cycle (>= today's week)."""
    tue = today
    while tue.weekday() != EXPIRY_WD:        # next Tuesday on/after today
        tue += timedelta(days=1)
    exp = tue
    while not _is_trading_day(exp):          # holiday → walk back to prev trading day
        exp -= timedelta(days=1)
    return exp


def ensure_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS paper_dte_trades(
      id serial PRIMARY KEY, trade_date date UNIQUE, expiry date,
      atm int, spot_entry double precision,
      short_ce text, short_pe text, long_ce text, long_pe text,
      entry_credit double precision, margin double precision,
      stop_loss double precision,
      exit_value double precision, pnl double precision,
      ret_margin double precision, reason text, status text,
      entered_at timestamp, settled_at timestamp)""")


def legs(atm, exp):
    sc = round(atm * (1 + OTM) / STEP) * STEP
    sp = round(atm * (1 - OTM) / STEP) * STEP
    bc = round(sc * (1 + WING) / STEP) * STEP
    bp = round(sp * (1 - WING) / STEP) * STEP
    return (sym(sc, exp, "CE"), sym(sp, exp, "PE"),
            sym(bc, exp, "CE"), sym(bp, exp, "PE"), sc, sp, bc, bp)


def do_enter(svc, cur, today, force):
    exp = this_expiry(today)
    if not force and today != exp:
        print(f"{today}: not a NIFTY expiry day (next {exp}) — skip"); return
    spot, atm = spot_atm(svc, today)
    if not atm:
        print("no spot — token?"); return
    sce, spe, lce, lpe, scs, sps, bcs, bps = legs(atm, exp)
    px = {}
    for s in (sce, spe, lce, lpe):
        b = bars(svc, s, today)
        if not b:
            print(f"no open bar for {s} — abort"); return
        px[s] = b[0][0]                 # open
    credit = (px[sce] + px[spe]) * (1 - SLIP) - (px[lce] + px[lpe]) * (1 + SLIP)
    wing_w = min(bcs - scs, sps - bps)
    if credit <= 0 or credit >= 0.95 * wing_w:
        print(f"bad credit {credit:.1f} vs wing {wing_w} — skip"); return
    margin = max(wing_w - credit, MARGIN_FLOOR * wing_w)
    stop_loss = STOP_MULT * credit
    cur.execute("""INSERT INTO paper_dte_trades
      (trade_date,expiry,atm,spot_entry,short_ce,short_pe,long_ce,long_pe,
       entry_credit,margin,stop_loss,status,entered_at)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'OPEN',%s)
      ON CONFLICT (trade_date) DO NOTHING""",
      (today, exp, atm, spot, sce, spe, lce, lpe, round(credit, 2),
       round(margin, 2), round(stop_loss, 2), datetime.now(IST).replace(tzinfo=None)))
    print(f"PAPER ENTER {today} exp={exp} atm={atm} credit={credit:.1f} "
          f"margin={margin:.1f} stop@-{stop_loss:.1f}\n  short {sce}/{spe} "
          f"wings {lce}/{lpe}")


def do_settle(svc, cur, today, force):
    cur.execute("SELECT * FROM paper_dte_trades WHERE trade_date=%s AND status='OPEN'",
                (today,))
    row = cur.fetchone()
    if not row:
        print(f"{today}: no open paper trade"); return
    cols = [d[0] for d in cur.description]
    t = dict(zip(cols, row))
    legs4 = [t["short_ce"], t["short_pe"], t["long_ce"], t["long_pe"]]
    bb = {s: bars(svc, s, today) for s in legs4}
    if any(not bb[s] for s in legs4):
        print("missing settle bars — retry later"); return
    close_val = ((bb[t["short_ce"]][-1][3] + bb[t["short_pe"]][-1][3])
                 - (bb[t["long_ce"]][-1][3] + bb[t["long_pe"]][-1][3]))
    # intraday worst fly value (short highs - long lows) for stop check
    worst_val = ((max(b[1] for b in bb[t["short_ce"]]) + max(b[1] for b in bb[t["short_pe"]]))
                 - (min(b[2] for b in bb[t["long_ce"]]) + min(b[2] for b in bb[t["long_pe"]])))
    credit = t["entry_credit"]
    if (credit - worst_val) <= -t["stop_loss"]:
        pnl = -t["stop_loss"]; reason = "STOP"; exitv = credit + t["stop_loss"]
    else:
        exitv = close_val * (1 + SLIP)
        pnl = credit - exitv; reason = "SETTLE"
    cur.execute("""UPDATE paper_dte_trades SET exit_value=%s,pnl=%s,ret_margin=%s,
      reason=%s,status='CLOSED',settled_at=%s WHERE id=%s""",
      (round(exitv, 2), round(pnl, 2), round(pnl / t["margin"], 4), reason,
       datetime.now(IST).replace(tzinfo=None), t["id"]))
    print(f"PAPER SETTLE {today}: {reason} pnl={pnl:.1f} "
          f"ret_margin={100*pnl/t['margin']:.1f}%")


def do_report(cur):
    cur.execute("SELECT trade_date,reason,entry_credit,pnl,ret_margin,status "
                "FROM paper_dte_trades ORDER BY trade_date")
    rows = cur.fetchall()
    if not rows:
        print("no paper trades yet"); return
    closed = [r for r in rows if r[5] == "CLOSED"]
    print(f"{'date':12} {'reason':7} {'credit':>7} {'pnl':>8} {'ret%':>7} status")
    for r in rows:
        print(f"{str(r[0]):12} {r[1] or '-':7} {r[2] or 0:>7} {r[3] or 0:>8} "
              f"{100*(r[4] or 0):>6.1f}% {r[5]}")
    if closed:
        rets = [r[4] for r in closed]
        wins = sum(1 for r in closed if r[3] > 0)
        cum = 1.0
        for x in rets:
            cum *= (1 + x)
        print(f"\nCLOSED {len(closed)}: WR {100*wins/len(closed):.0f}% "
              f"total ret/margin {100*sum(rets):.1f}% cum x{cum:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enter", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--date", help="override 'today' (YYYY-MM-DD), for testing")
    a = ap.parse_args()
    conn = psycopg.connect(DB); cur = conn.cursor()
    ensure_table(cur); conn.commit()
    today = (date.fromisoformat(a.date) if a.date else datetime.now(IST).date())
    svc = FyersService() if (a.enter or a.settle) else None
    if a.enter:
        do_enter(svc, cur, today, a.force)
    if a.settle:
        do_settle(svc, cur, today, a.force)
    if a.report:
        do_report(cur)
    conn.commit(); conn.close()


if __name__ == "__main__":
    main()
