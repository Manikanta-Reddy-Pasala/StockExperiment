"""Per-trade weekly MTM trace for FinNifty monthly IC backtest.

Reads trades.csv (output of run_winner.py / sweep.py) and for each trade:
  - Identifies each Friday between entry_date and exit_date
  - Fetches CE/PE/WCE/WPE option closes from historical_options
  - Computes IC value at that Friday + cumulative MTM P&L per leg
  - Outputs weekly_mtm.csv + WEEKLY_MTM.md

IC P&L (per unit, no lots/lot_size):
    open_value  = net_credit (received at entry, positive credit)
    current_ic_value (per unit) = -(short_ce_close + short_pe_close)
                                  + (long_wce_close + long_wpe_close)
        (We sold the body, bought the wings — value to close = pay back
         the body, sell the wings. Sign of "current_ic_value" matches
         the signed cash position.)
    mtm_unit = current_ic_value - (-net_credit)
             = current_ic_value + net_credit
        (entry net_credit was POSITIVE cash inflow; current_ic_value is
         what we'd net to close right now — if 0 = max profit.)

For docs: also report max-loss path (worst MTM during trade), recovery
fraction, and exit-vs-trough.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
from datetime import date, datetime, timedelta

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine

TRADES_CSV = "/app/logs/FINNIFTY_monthly_IC_OTM4_w300_lots5/trades.csv"
OUT_DIR    = Path("/app/logs/FINNIFTY_monthly_IC_OTM4_w300_lots5")
UNDERLYING = "FINNIFTY"


def opt_close_on(symbol: str, on_date: date) -> float | None:
    """Return option close on/before given date (most recent <= on_date)."""
    eng = _get_engine()
    q = text(
        "SELECT close FROM historical_options "
        "WHERE symbol=:s AND interval='D' AND candle_time::date <= :d "
        "ORDER BY candle_time DESC LIMIT 1"
    )
    with eng.connect() as conn:
        r = conn.execute(q, {"s": symbol, "d": on_date}).fetchone()
    return float(r.close) if r else None


def opt_symbol(underlying: str, expiry: date, strike: float, opt_type: str) -> str:
    """Standard Fyers option symbol format: NSE:FINNIFTY<YY><M><DD><K><CE/PE>"""
    # Use option_universe table for actual symbol if available
    eng = _get_engine()
    q = text(
        "SELECT symbol FROM option_universe "
        "WHERE underlying=:u AND expiry=:e AND strike=:k AND opt_type=:o LIMIT 1"
    )
    with eng.connect() as conn:
        r = conn.execute(q, {
            "u": underlying, "e": expiry, "k": strike, "o": opt_type,
        }).fetchone()
    return r.symbol if r else ""


def fridays_between(start: date, end: date) -> list:
    out = []
    d = start
    while d <= end:
        if d.weekday() == 4:   # Friday
            out.append(d)
        d += timedelta(days=1)
    if not out or out[-1] != end:
        out.append(end)   # always include exit date
    return out


def trace_trade(trade: dict) -> list:
    """Return list of {date, mtm_unit, mtm_total, breakdown...}."""
    entry_d = datetime.strptime(trade["entry_date"], "%Y-%m-%d").date()
    exit_d  = datetime.strptime(trade["exit_date"],  "%Y-%m-%d").date()
    expiry  = datetime.strptime(trade["expiry"],     "%Y-%m-%d").date()
    ce_k    = float(trade["ce_k"])
    pe_k    = float(trade["pe_k"])
    wce_k   = float(trade["wce_k"])
    wpe_k   = float(trade["wpe_k"])
    net_credit = float(trade["net_credit"])
    lot     = int(trade["lot"])
    lots    = int(trade["lots"])

    sym_ce  = opt_symbol(UNDERLYING, expiry, ce_k, "CE")
    sym_pe  = opt_symbol(UNDERLYING, expiry, pe_k, "PE")
    sym_wce = opt_symbol(UNDERLYING, expiry, wce_k, "CE")
    sym_wpe = opt_symbol(UNDERLYING, expiry, wpe_k, "PE")

    rows = []
    for fri in fridays_between(entry_d, exit_d):
        ce_px  = opt_close_on(sym_ce,  fri) if sym_ce  else None
        pe_px  = opt_close_on(sym_pe,  fri) if sym_pe  else None
        wce_px = opt_close_on(sym_wce, fri) if sym_wce else None
        wpe_px = opt_close_on(sym_wpe, fri) if sym_wpe else None
        if None in (ce_px, pe_px, wce_px, wpe_px):
            rows.append({
                "date": fri.isoformat(),
                "ce_px": ce_px, "pe_px": pe_px,
                "wce_px": wce_px, "wpe_px": wpe_px,
                "mtm_unit": None, "mtm_total": None,
                "note": "missing leg close",
            })
            continue
        # Cost to close = buy back body, sell wings
        close_cost = (ce_px + pe_px) - (wce_px + wpe_px)
        mtm_unit = net_credit - close_cost
        mtm_total = mtm_unit * lot * lots
        rows.append({
            "date": fri.isoformat(),
            "ce_px": round(ce_px, 2), "pe_px": round(pe_px, 2),
            "wce_px": round(wce_px, 2), "wpe_px": round(wpe_px, 2),
            "close_cost": round(close_cost, 2),
            "mtm_unit": round(mtm_unit, 2),
            "mtm_total": round(mtm_total, 2),
            "note": "",
        })
    return rows


def main():
    trades = []
    with open(TRADES_CSV) as f:
        for r in csv.DictReader(f):
            trades.append(r)
    print(f"Loaded {len(trades)} trades from {TRADES_CSV}")

    all_weekly = []
    md_lines = ["# FinNifty IC — Weekly MTM Trace per Trade",
                "",
                f"Source: `{TRADES_CSV}`",
                f"Total trades: {len(trades)}",
                "",
                "Per-trade weekly mark-to-market during life of the iron condor.",
                "Each Friday between entry and exit, recompute the cost-to-close",
                "the position using leg-level option closes.",
                "",
                "Columns:",
                "- **date**: Friday in trade life (or exit date if not Friday)",
                "- **CE px / PE px**: closes of the sold (body) strikes",
                "- **WCE px / WPE px**: closes of the bought (wing) strikes",
                "- **close_cost**: ₹ per unit to close right now (body buy − wings sell)",
                "- **MTM/unit**: net_credit − close_cost (positive = profit so far)",
                "- **MTM total**: MTM/unit × lot × lots (real ₹ P&L)",
                ""]
    for t in trades:
        md_lines.append(f"## Trade {t['entry_date']} → {t['exit_date']}  "
                        f"(expiry {t['expiry']})")
        md_lines.append("")
        md_lines.append(f"- Spot at entry: {t['spot']}")
        md_lines.append(f"- Body strikes: SELL CE {t['ce_k']} / PE {t['pe_k']}")
        md_lines.append(f"- Wing strikes: BUY CE {t['wce_k']} / PE {t['wpe_k']}")
        md_lines.append(f"- Net credit: ₹{t['net_credit']} per unit "
                        f"({t['lot']}×{t['lots']} = {int(t['lot'])*int(t['lots'])} qty)")
        md_lines.append(f"- Final P&L: ₹{t['pnl_total']} ({t['exit_reason']})")
        md_lines.append("")
        rows = trace_trade(t)
        if not rows:
            md_lines.append("_No weekly samples (entry == exit?)._")
            md_lines.append("")
            continue
        md_lines.append("| Date | CE | PE | WCE | WPE | close_cost | MTM/unit | MTM total |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            cc = r.get("close_cost", "—")
            mu = r.get("mtm_unit", "—")
            mt = r.get("mtm_total", "—")
            md_lines.append(f"| {r['date']} | {r['ce_px']} | {r['pe_px']} | "
                            f"{r['wce_px']} | {r['wpe_px']} | {cc} | {mu} | "
                            f"{'₹' + format(mt, ',.0f') if isinstance(mt, (int, float)) else '—'} |")
        # Worst point during trade
        valid = [r for r in rows if r.get("mtm_total") is not None]
        if valid:
            worst = min(valid, key=lambda x: x["mtm_total"])
            best  = max(valid, key=lambda x: x["mtm_total"])
            md_lines.append("")
            md_lines.append(f"**Trade extremes:** worst MTM ₹{worst['mtm_total']:,.0f} "
                            f"on {worst['date']}, best MTM ₹{best['mtm_total']:,.0f} "
                            f"on {best['date']}")
        md_lines.append("")
        # Add to CSV all_weekly
        for r in rows:
            all_weekly.append({**r, "entry_date": t["entry_date"],
                                "exit_date": t["exit_date"]})

    # Write outputs
    out_md = OUT_DIR / "WEEKLY_MTM.md"
    out_csv = OUT_DIR / "weekly_mtm.csv"
    out_md.write_text("\n".join(md_lines))
    with open(out_csv, "w", newline="") as f:
        if all_weekly:
            cols = set()
            for r in all_weekly:
                cols.update(r.keys())
            cols = sorted(cols)
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in all_weekly:
                w.writerow({c: r.get(c, "") for c in cols})
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")
    print(f"Total weekly samples: {len(all_weekly)}")


if __name__ == "__main__":
    main()
