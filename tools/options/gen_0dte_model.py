#!/usr/bin/env python3
"""Generate summary.json + trade_ledger.json for the nifty_0dte_ironfly model
(recommended config), in the same shape as the equity live models."""
import sys, json, statistics as st
sys.path.insert(0, "/app/tools/options")
import opt_0dte as m

START = "2025-03-01"
CFG = dict(structure="ironfly", otm=0.012, stop=2.0, wing=0.02)
OUT = "/app/tools/models/nifty_0dte_ironfly"

tr = m.backtest(CFG["otm"], CFG["stop"], CFG["structure"], CFG["wing"], 0, START, None)
tr.sort(key=lambda t: t["expiry"])

ledger = [dict(expiry=t["expiry"], structure="ironfly", spot=t["spot"],
               credit=t["credit"], pnl=t["pnl"],
               ret_margin_pct=round(100 * t["ret"], 2), reason=t["reason"],
               atm=t["atm"], min_leg_volume=t["minvol"],
               all_legs_filled=t["all_filled"], legs=t["legs"])
          for t in tr]

rets = [t["ret"] for t in tr]
wins = sum(1 for t in tr if t["pnl"] > 0)
eq = peak = mdd = 0.0; ceq = 1.0
for r in rets:
    eq += r; peak = max(peak, eq); mdd = max(mdd, peak - eq); ceq *= (1 + r)
cagr = (ceq ** (52.0 / len(rets)) - 1) if rets else 0
per_year = {}
for t in tr:
    y = t["expiry"][:4]
    per_year.setdefault(y, []).append(t["ret"])
per_year = {y: dict(trades=len(v), ret_margin_pct=round(100 * sum(v), 1),
                    win_rate_pct=round(100 * sum(1 for x in v if x > 0) / len(v), 1))
            for y, v in per_year.items()}

summary = dict(
    model="nifty_0dte_ironfly",
    status="PAPER / RESEARCH — not live, no real capital",
    instrument="NIFTY weekly 0DTE iron-fly (defined risk)",
    config=dict(short_otm_pct=1.2, wing_pct=2.0, stop_x_credit=2.0,
                entry="expiry-day open", exit="expiry-day close or 2x stop"),
    data="historical_options expiry-day OHLC (daily bhavcopy proxy for 0DTE)",
    window=f"{START}..now", trades=len(tr),
    wins=wins, losses=len(tr) - wins,
    win_rate_pct=round(100 * wins / len(tr), 1),
    cagr_pct=round(100 * cagr, 1),
    avg_ret_margin_pct=round(100 * st.mean(rets), 2),
    max_dd_pct=round(100 * mdd, 1),
    worst_trade_pct=round(100 * min(rets), 1),
    max_loss_capped="yes — bought wings (worst day structurally bounded)",
    per_year=per_year,
    caveats=["in-sample single regime (2025-26, seller-friendly)",
             "daily-OHLC proxy not true intraday (recorder accumulating real 5m)",
             "64 trades = thin sample, no walk-forward yet",
             "live execution slippage will reduce returns"])

import os
os.makedirs(OUT, exist_ok=True)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
json.dump(ledger, open(f"{OUT}/trade_ledger.json", "w"), indent=2)

# detailed TRADE_LEDGER.md — shows index level + each leg's strike/%/price/volume/fill
def cell(legs, role):
    l = next((x for x in legs if x["role"] == role), None)
    if not l:
        return "—"
    flag = "" if l["filled"] else " ⚠️"
    return f"{l['action']} {l['strike']} ({l['pct']:+.2f}%) · ₹{l['price']} · vol {l['volume']:,}{flag}"

md = ["# NIFTY 0DTE Iron-Fly — Trade Ledger\n",
      f"{len(ledger)} trades (in-sample backtest, expiry-day OHLC proxy). "
      "Each row: index (spot) at entry, the 4 legs (strike · % from spot · entry "
      "price · day volume), credit, P&L. ⚠️ = leg traded < 100 contracts that day "
      "(thin). All prices are the 9:15 expiry-day open.\n",
      "| Expiry | Spot | Short CE | Short PE | Long CE (wing) | Long PE (wing) | Credit | P&L | Ret/margin | All filled | Exit |",
      "|---|---:|---|---|---|---|---:|---:|---:|:---:|---|"]
for t in ledger:
    L = t["legs"]
    md.append(f"| {t['expiry']} | {t['spot']} | {cell(L,'short_CE')} | "
              f"{cell(L,'short_PE')} | {cell(L,'long_CE')} | {cell(L,'long_PE')} | "
              f"{t['credit']} | {t['pnl']} | {t['ret_margin_pct']}% | "
              f"{'✅' if t['all_legs_filled'] else '⚠️ NO'} | {t['reason']} |")
open(f"{OUT}/TRADE_LEDGER.md", "w").write("\n".join(md))
print(json.dumps(summary, indent=2))
