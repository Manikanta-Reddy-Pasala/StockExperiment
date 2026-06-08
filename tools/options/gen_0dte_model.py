#!/usr/bin/env python3
"""Generate summary.json + trade_ledger.json + TRADE_LEDGER.md for a 0DTE
iron-fly paper model. Parameterized: --model nifty50_weekly_0dte | banknifty_monthly_0dte."""
import sys, json, argparse, statistics as st
sys.path.insert(0, "/app/tools/options")
import opt_0dte as m

START = "2025-03-01"
SPECS = {
    "nifty50_weekly_0dte": dict(underlying="NIFTY", cadence="weekly", otm=0.012,
                                wing=0.02, name="NIFTY 50 Weekly 0DTE Iron-Fly"),
    "banknifty_monthly_0dte": dict(underlying="BANKNIFTY", cadence="monthly",
                                   otm=0.012, wing=0.02,
                                   name="Bank Nifty Monthly 0DTE Iron-Fly"),
}
_ap = argparse.ArgumentParser(); _ap.add_argument("--model", default="nifty50_weekly_0dte")
_A = _ap.parse_args()
SPEC = SPECS[_A.model]
CFG = dict(structure="ironfly", otm=SPEC["otm"], stop=2.0, wing=SPEC["wing"])
OUT = f"/app/tools/models/{_A.model}"

tr = m.backtest(CFG["otm"], CFG["stop"], CFG["structure"], CFG["wing"], 0, START,
                None, underlying=SPEC["underlying"], cadence=SPEC["cadence"])
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
per_yr_count = 52 if SPEC["cadence"] == "weekly" else 12   # expiries/year
cagr = (ceq ** (per_yr_count / len(rets)) - 1) if rets else 0
per_year = {}
for t in tr:
    y = t["expiry"][:4]
    per_year.setdefault(y, []).append(t["ret"])
per_year = {y: dict(trades=len(v), ret_margin_pct=round(100 * sum(v), 1),
                    win_rate_pct=round(100 * sum(1 for x in v if x > 0) / len(v), 1))
            for y, v in per_year.items()}

summary = dict(
    model=_A.model,
    status="PAPER / RESEARCH — not live, no real capital",
    instrument=SPEC["name"] + " (defined risk)",
    config=dict(short_otm_pct=1.2, wing_pct=2.0, stop_x_credit=2.0,
                cadence=SPEC["cadence"],
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
             f"{len(tr)} trades = thin sample, no walk-forward yet",
             "live execution slippage will reduce returns"]
    + ([] if SPEC["cadence"] == "weekly" else
       ["monthly-only (~12/yr); BankNifty options less liquid than NIFTY; diversifier not standalone"]))

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

# ---- ₹2L-per-trade rupee simulation (deploy ₹2,00,000 margin each trade) ----
CAP = 200000
sum_ret = sum(rets)
total_pnl_2L = round(CAP * sum_ret)
avg_pnl_2L = round(CAP * st.mean(rets))
best_inr = round(CAP * max(rets))
worst_inr = round(CAP * min(rets))
def inr(x):
    return f"₹{x:,.0f}"

# SUMMARY.md — consistent report (strategy + entry/exit + regenerated results)
py = "\n".join(f"| {y} | {v['trades']} | {v['ret_margin_pct']}% | {inr(round(CAP*v['ret_margin_pct']/100))} |"
               for y, v in summary["per_year"].items())
exp_word = "weekly expiry (Tuesday)" if SPEC["cadence"] == "weekly" else "monthly expiry (last Tuesday)"
sm = f"""# {SPEC['name']} (`{_A.model}`)

**Status:** PAPER / RESEARCH — not live, no real capital.

Defined-risk 0DTE premium selling on {SPEC['underlying']} {exp_word}.

## Strategy — entry & exit
**Trade only on {exp_word}** — on expiry day the option has one session left; time
value collapses to ~0 by close. Sell at the open, let it decay to settlement.

**ENTRY (9:15 open):** find ATM via put-call parity (spot ≈ median K+CE−PE);
**sell 1.2%-OTM CE + PE**; **buy wings 2% beyond each** (defines max loss).
Net credit = max profit; **max loss = wing width − credit** (fixed at entry, gap-proof).

**EXIT (first of):** hold to close if NIFTY stays between shorts (decays to ~max
profit) · **2× credit hard stop** intraday · expiry settlement.

## Results (backtest {summary['window']}, in-sample, daily-OHLC proxy)
| Metric | Value |
|---|---|
| Trades | {summary['trades']} ({summary['wins']}W / {summary['losses']}L) |
| Win rate | {summary['win_rate_pct']}% |
| CAGR | {summary['cagr_pct']}% |
| Avg return / trade (margin) | {summary['avg_ret_margin_pct']}% |
| Max drawdown | {summary['max_dd_pct']}% |
| Worst trade | {summary['worst_trade_pct']}% (capped by wings) |

## How the % is derived
`return % = P&L ÷ margin deployed`, where **margin = wing width − credit** (the
defined-risk capital locked per iron-fly). Per-unit (lot-size independent).

## Capital simulation — ₹2,00,000 margin per trade
Deploy a fixed **₹2,00,000** of margin on each trade (rupee P&L = ₹2L × return%):
| Metric | Value |
|---|---|
| Margin in / trade | ₹2,00,000 |
| Avg P&L / trade | {inr(avg_pnl_2L)} |
| **Total P&L ({summary['trades']} trades)** | **{inr(total_pnl_2L)}** |
| Best trade | {inr(best_inr)} |
| Worst trade | {inr(worst_inr)} (max loss capped by wings) |

*Fixed ₹2L per trade (profit pocketed, not compounded). Assumes ₹2L fully
deployed as margin; real lots are discrete (NIFTY lot 75, BankNifty 35) so actual
sizing rounds to whole lots.*

## Execution — BASKET / multi-leg order ONLY
The 4 legs are entered as **one basket (multi-leg) order**, never 4 individual
orders — legging in separately risks partial fills + the index moving between
legs, which breaks the defined-risk structure. Backtest/paper price all 4 legs
at the same instant (the basket). **Paper only — no real broker orders.**

### Year-by-year (₹2L/trade)
| Year | Trades | Return % (margin) | P&L (₹2L/trade) |
|---|---:|---:|---:|
{py}

## Caveats
{chr(10).join('- ' + c for c in summary['caveats'])}

## Live paper
Paper-only (no orders). Crons enter 09:20 IST / settle 15:25 IST on its expiry days
→ table `paper_dte_trades` (model=`{_A.model}`).
`python tools/options/paper_dte_ironfly.py --report --model {_A.model}`.
"""
open(f"{OUT}/SUMMARY.md", "w").write(sm)

md = [f"# {SPEC['name']} — Trade Ledger\n",
      f"{len(ledger)} trades (in-sample backtest, expiry-day OHLC proxy). "
      "Each row: index (spot) at entry, the 4 legs (strike · % from spot · entry "
      "price · day volume), then the rupee result on a **fixed ₹2,00,000 margin "
      "deployed per trade** (In → Out). ⚠️ = leg traded < 100 contracts (thin). "
      "Entered as ONE basket order. All prices are the 9:15 expiry-day open.\n",
      "| Expiry | Spot | Short CE | Short PE | Long CE (wing) | Long PE (wing) | "
      "Capital In | P&L ₹ | Capital Out | Ret% | Filled | Exit |",
      "|---|---:|---|---|---|---|---:|---:|---:|---:|:---:|---|"]
for t in ledger:
    L = t["legs"]
    ret = t["ret_margin_pct"] / 100.0
    pnl_inr = round(CAP * ret)
    out_inr = CAP + pnl_inr
    md.append(f"| {t['expiry']} | {t['spot']} | {cell(L,'short_CE')} | "
              f"{cell(L,'short_PE')} | {cell(L,'long_CE')} | {cell(L,'long_PE')} | "
              f"{inr(CAP)} | {inr(pnl_inr)} | {inr(out_inr)} | {t['ret_margin_pct']}% | "
              f"{'✅' if t['all_legs_filled'] else '⚠️ NO'} | {t['reason']} |")
md.append(f"\n**Total on ₹2L/trade: {inr(total_pnl_2L)} P&L across {len(ledger)} "
          f"trades** (avg {inr(avg_pnl_2L)}/trade, best {inr(best_inr)}, worst {inr(worst_inr)}).")
open(f"{OUT}/TRADE_LEDGER.md", "w").write("\n".join(md))
print(json.dumps(summary, indent=2))
