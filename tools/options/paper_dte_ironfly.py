#!/usr/bin/env python3
"""
PAPER-TRADING: 0DTE iron-fly (defined risk) on index expiry days. PAPER ONLY —
no broker orders; logs to paper_dte_trades; compares to backtest.

Two models (current NSE regime = Tuesday expiries):
  nifty50_weekly_0dte    — NIFTY 50, every weekly expiry (Tuesday)
  banknifty_monthly_0dte — Bank Nifty, monthly expiry only (last Tuesday;
                           BankNifty weeklies discontinued by SEBI Nov-2024)

Both: sell 1.2%-OTM CE+PE at the open, buy 2% wings (defined risk), 2× credit
hard stop, settle at close. Prices via Fyers history 5m (live contracts only,
so this MUST run daily — each run captures that day if it's the model's expiry).

Modes: --enter / --settle / --report / --force / --date / --model <key>
"""
import argparse, json, sys
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
SLIP = 0.02
MARGIN_FLOOR = 0.25
EXPIRY_WD = 1   # Tuesday (current NSE index expiry weekday)

LOTS = 2   # trade 2 lots per leg
MODELS = {
    "nifty50_weekly_0dte": dict(
        title="NIFTY 50 Weekly 0DTE Iron-Fly", underlying="NIFTY",
        fyers_idx="NIFTY50-INDEX", cadence="weekly", step=50, lot=65,
        otm=0.012, wing=0.02, stop=2.0),
    "banknifty_monthly_0dte": dict(
        title="Bank Nifty Monthly 0DTE Iron-Fly", underlying="BANKNIFTY",
        fyers_idx="NIFTYBANK-INDEX", cadence="monthly", step=100, lot=30,
        otm=0.012, wing=0.02, stop=2.0),
}


def _is_trading_day(d):
    try:
        from tools.shared.nse_calendar import is_trading_day
        return is_trading_day(d)
    except Exception:
        return d.weekday() < 5


def _holiday_back(d):
    while not _is_trading_day(d):
        d -= timedelta(days=1)
    return d


def this_expiry(today, cadence):
    """weekly = next Tuesday on/after today; monthly = last Tuesday of the month
    (roll to next month once past it). Holiday-adjusted to prior trading day."""
    if cadence == "weekly":
        tue = today
        while tue.weekday() != EXPIRY_WD:
            tue += timedelta(days=1)
        return _holiday_back(tue)
    # monthly: last Tuesday of today's month
    def last_tue(y, mo):
        d = date(y + (mo == 12), 1 if mo == 12 else mo + 1, 1) - timedelta(days=1)
        while d.weekday() != EXPIRY_WD:
            d -= timedelta(days=1)
        return _holiday_back(d)
    e = last_tue(today.year, today.month)
    if today > e:
        ny, nm = (today.year + (today.month == 12),
                  1 if today.month == 12 else today.month + 1)
        e = last_tue(ny, nm)
    return e


def is_monthly(exp):
    return (exp + timedelta(days=7)).month != exp.month


def sym(underlying, strike, exp, kind):
    yy = exp.strftime("%y")
    if is_monthly(exp):
        return f"{underlying}{yy}{MON3U[exp.month]}{strike}{kind}"
    return f"{underlying}{yy}{MW[exp.month]}{exp.day:02d}{strike}{kind}"


def bars(svc, s, day):
    r = svc.history(user_id=1, symbol=s, exchange="NSE", interval="5m",
                    start_date=str(day), end_date=str(day + timedelta(days=1))) or {}
    cs = r.get("candles") or r.get("data", {}).get("candles")
    out = []
    for c in (cs or []):
        if isinstance(c, dict):
            out.append((float(c["open"]), float(c["high"]), float(c["low"]),
                        float(c["close"]), int(float(c.get("volume", 0)))))
        else:
            out.append((float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                        int(float(c[5])) if len(c) > 5 else 0))
    return out


def spot_atm(svc, day, cfg):
    b = bars(svc, cfg["fyers_idx"], day) or bars(svc, cfg["fyers_idx"], day - timedelta(days=1))
    if not b:
        return None, None
    spot = b[0][0]
    return spot, round(spot / cfg["step"]) * cfg["step"]


def legs(atm, exp, cfg):
    st = cfg["step"]; u = cfg["underlying"]
    sc = round(atm * (1 + cfg["otm"]) / st) * st
    sp = round(atm * (1 - cfg["otm"]) / st) * st
    bc = round(sc * (1 + cfg["wing"]) / st) * st
    bp = round(sp * (1 - cfg["wing"]) / st) * st
    return (sym(u, sc, exp, "CE"), sym(u, sp, exp, "PE"),
            sym(u, bc, exp, "CE"), sym(u, bp, exp, "PE"), sc, sp, bc, bp)


def ensure_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS paper_dte_trades(
      id serial PRIMARY KEY, model text, trade_date date, expiry date,
      atm int, spot_entry double precision,
      short_ce text, short_pe text, long_ce text, long_pe text,
      entry_credit double precision, margin double precision,
      stop_loss double precision,
      exit_value double precision, pnl double precision,
      ret_margin double precision, reason text, status text,
      entered_at timestamp, settled_at timestamp, legs text)""")
    cur.execute("ALTER TABLE paper_dte_trades ADD COLUMN IF NOT EXISTS model text")
    cur.execute("ALTER TABLE paper_dte_trades ADD COLUMN IF NOT EXISTS legs text")
    # legacy rows (pre-multi-model) were NIFTY weekly
    cur.execute("UPDATE paper_dte_trades SET model='nifty50_weekly_0dte' WHERE model IS NULL")
    # old schema had UNIQUE(trade_date); drop it so 2 models can trade same day
    cur.execute("ALTER TABLE paper_dte_trades DROP CONSTRAINT IF EXISTS paper_dte_trades_trade_date_key")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_model_date "
                "ON paper_dte_trades(model, trade_date)")


def do_enter(svc, cur, today, model, cfg, force):
    exp = this_expiry(today, cfg["cadence"])
    if not force and today != exp:
        print(f"[{model}] {today}: not an expiry day (next {exp}) — skip"); return
    spot, atm = spot_atm(svc, today, cfg)
    if not atm:
        print(f"[{model}] no spot — token?"); return
    sce, spe, lce, lpe, scs, sps, bcs, bps = legs(atm, exp, cfg)
    px, vol = {}, {}
    for s in (sce, spe, lce, lpe):
        b = bars(svc, s, today)
        if not b:
            print(f"[{model}] no open bar for {s} — abort"); return
        px[s] = b[0][0]; vol[s] = b[0][4]
    leg_detail = [
        dict(role="short_CE", action="SELL", strike=scs, pct=round(100*(scs/atm-1),2), price=round(px[sce],2), volume=vol[sce], filled=vol[sce] > 0),
        dict(role="short_PE", action="SELL", strike=sps, pct=round(100*(sps/atm-1),2), price=round(px[spe],2), volume=vol[spe], filled=vol[spe] > 0),
        dict(role="long_CE",  action="BUY",  strike=bcs, pct=round(100*(bcs/atm-1),2), price=round(px[lce],2), volume=vol[lce], filled=vol[lce] > 0),
        dict(role="long_PE",  action="BUY",  strike=bps, pct=round(100*(bps/atm-1),2), price=round(px[lpe],2), volume=vol[lpe], filled=vol[lpe] > 0),
    ]
    credit = (px[sce] + px[spe]) * (1 - SLIP) - (px[lce] + px[lpe]) * (1 + SLIP)
    wing_w = min(bcs - scs, sps - bps)
    if credit <= 0 or credit >= 0.95 * wing_w:
        print(f"[{model}] bad credit {credit:.1f} vs wing {wing_w} — skip"); return
    margin = max(wing_w - credit, MARGIN_FLOOR * wing_w)
    stop_loss = cfg["stop"] * credit
    cur.execute("""INSERT INTO paper_dte_trades
      (model,trade_date,expiry,atm,spot_entry,short_ce,short_pe,long_ce,long_pe,
       entry_credit,margin,stop_loss,status,entered_at,legs)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'OPEN',%s,%s)
      ON CONFLICT (model,trade_date) DO NOTHING""",
      (model, today, exp, atm, spot, sce, spe, lce, lpe, round(credit, 2),
       round(margin, 2), round(stop_loss, 2), datetime.now(IST).replace(tzinfo=None),
       json.dumps(leg_detail)))
    qty = LOTS * cfg["lot"]
    print(f"PAPER ENTER [{model}] {today} exp={exp} spot/atm={atm} | "
          f"2 lots = {qty} qty/leg | credit {credit:.1f}pts (₹{credit*qty:,.0f}) "
          f"margin {margin:.1f}pts (₹{margin*qty:,.0f}) maxloss ₹{stop_loss*qty:,.0f}")
    for l in leg_detail:
        print(f"  {l['action']} {l['role']} {l['strike']} ({l['pct']:+.2f}%) x{qty} "
              f"@₹{l['price']} vol={l['volume']:,} {'OK' if l['filled'] else 'THIN'}")


def do_settle(svc, cur, today, model, cfg, force):
    cur.execute("SELECT * FROM paper_dte_trades WHERE model=%s AND trade_date=%s "
                "AND status='OPEN'", (model, today))
    row = cur.fetchone()
    if not row:
        print(f"[{model}] {today}: no open paper trade"); return
    cols = [d[0] for d in cur.description]; t = dict(zip(cols, row))
    legs4 = [t["short_ce"], t["short_pe"], t["long_ce"], t["long_pe"]]
    bb = {s: bars(svc, s, today) for s in legs4}
    if any(not bb[s] for s in legs4):
        print(f"[{model}] missing settle bars — retry later"); return
    close_val = ((bb[t["short_ce"]][-1][3] + bb[t["short_pe"]][-1][3])
                 - (bb[t["long_ce"]][-1][3] + bb[t["long_pe"]][-1][3]))
    worst_val = ((max(b[1] for b in bb[t["short_ce"]]) + max(b[1] for b in bb[t["short_pe"]]))
                 - (min(b[2] for b in bb[t["long_ce"]]) + min(b[2] for b in bb[t["long_pe"]])))
    credit = t["entry_credit"]
    if (credit - worst_val) <= -t["stop_loss"]:
        pnl, reason, exitv = -t["stop_loss"], "STOP", credit + t["stop_loss"]
    else:
        exitv = close_val * (1 + SLIP); pnl = credit - exitv; reason = "SETTLE"
    cur.execute("""UPDATE paper_dte_trades SET exit_value=%s,pnl=%s,ret_margin=%s,
      reason=%s,status='CLOSED',settled_at=%s WHERE id=%s""",
      (round(exitv, 2), round(pnl, 2), round(pnl / t["margin"], 4), reason,
       datetime.now(IST).replace(tzinfo=None), t["id"]))
    print(f"PAPER SETTLE [{model}] {today}: {reason} pnl={pnl:.1f} "
          f"ret_margin={100*pnl/t['margin']:.1f}%")


def do_report(cur, model):
    cur.execute("SELECT trade_date,reason,entry_credit,pnl,ret_margin,status "
                "FROM paper_dte_trades WHERE model=%s ORDER BY trade_date", (model,))
    rows = cur.fetchall()
    print(f"=== {model} ===")
    if not rows:
        print("  no paper trades yet"); return
    for r in rows:
        print(f"  {r[0]} {r[1] or '-':7} credit {r[2] or 0} pnl {r[3] or 0} "
              f"{100*(r[4] or 0):.1f}% {r[5]}")
    closed = [r for r in rows if r[5] == "CLOSED"]
    if closed:
        wins = sum(1 for r in closed if r[3] > 0)
        print(f"  CLOSED {len(closed)} · WR {100*wins/len(closed):.0f}% · "
              f"total ret/margin {100*sum(r[4] for r in closed):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enter", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--model", default="all",
                    help="nifty50_weekly_0dte | banknifty_monthly_0dte | all")
    a = ap.parse_args()
    keys = list(MODELS) if a.model == "all" else [a.model]
    conn = psycopg.connect(DB); cur = conn.cursor()
    ensure_table(cur); conn.commit()
    today = date.fromisoformat(a.date) if a.date else datetime.now(IST).date()
    svc = FyersService() if (a.enter or a.settle) else None
    for k in keys:
        cfg = MODELS[k]
        if a.enter:
            do_enter(svc, cur, today, k, cfg, a.force)
        if a.settle:
            do_settle(svc, cur, today, k, cfg, a.force)
        if a.report:
            do_report(cur, k)
    conn.commit(); conn.close()


if __name__ == "__main__":
    main()
