"""Regenerate exports/models/{model}/{SUMMARY.md, TRADE_LEDGER.md} from trade_ledger.json.

Run after backtest produces new ledger. Per-model overrides supplied via MODELS dict.
"""
import json
import csv
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
SYM_N100 = ROOT / "src/data/symbols/nifty100.csv"
SYM_MID  = ROOT / "src/data/symbols/nifty_midcap150.csv"
SYM_SML  = ROOT / "src/data/symbols/nifty_smallcap250.csv"


def _load(path):
    out = set()
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("Series", "").strip() == "EQ":
                out.add(r["Symbol"].strip())
    return out


N100 = _load(SYM_N100)
MID  = _load(SYM_MID)
SML  = _load(SYM_SML)


def cap_segment(sym):
    s = sym.replace("NSE:", "").replace("-EQ", "")
    if s in N100: return ("Large", "Nifty 100")
    if s in MID:  return ("Mid",   "Nifty Midcap 150")
    if s in SML:  return ("Small", "Nifty Smallcap 250")
    return ("Other", "outside top-500")


MODELS = {
    "momentum_n100_top5_max1": {
        "title": "Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret) + MAX_PRICE≤₹3,000 filter.",
        "rebalance": "Monthly (1st trading day)",
        "logic": [
            "Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)",
            "Filter at entry: close ≤ ₹3,000 (skips mega-priced losers BAJAJ-AUTO etc.)",
            "Rank by 30-day return, pick top-1",
            "Rebalance: 1st trading day of month",
            "Exit: rotation only — sell when not rank-1",
        ],
        "data": "Fyers (split-adjusted cont_flag=1)",
        "live": True,
    },
    "momentum_pseudo_n100_adv": {
        "title": "Pseudo-N100 (top-100 ADV from N500 − Smallcap) + uptrend + MAX_PRICE≤₹3,000. Monthly rotation top-1 by 30d ret.",
        "rebalance": "Monthly (1st trading day)",
        "logic": [
            "Universe: top-100 by 20-day ADV from N500 (yearly-PIT, rebuilt at year start)",
            "Drop NSE Smallcap 250 members",
            "Uptrend filter: close > 200-day SMA",
            "Max-price filter: close ≤ ₹3,000 at entry",
            "Rank by 30-day return, pick top-1",
            "Rebalance: 1st trading day of month",
        ],
        "data": "Fyers (split-adjusted cont_flag=1)",
        "live": False,
    },
    "n20_daily_large_only": {
        "title": "Top-20 ADV + uptrend + Nifty 100 + MAX_PRICE≤₹2,500. Daily rotation top-1 by 30d ret.",
        "rebalance": "Daily",
        "logic": [
            "Universe: top-20 by 20-day ADV from N500 (rebuilt daily)",
            "Uptrend filter: close > 200-day SMA",
            "Large-cap filter: stock must be in NSE Nifty 100",
            "Max-price filter: close ≤ ₹2,500 at entry",
            "Rank by 30-day return, pick top-1",
            "Rebalance: every trading day",
        ],
        "data": "Fyers (split-adjusted cont_flag=1)",
        "live": False,
    },
}


def stats(trades):
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    total_pnl = sum(pnls)
    final_cap = trades[-1]["cap_after"] if trades else 1_000_000
    peak = 1_000_000
    mdd = 0
    for t in trades:
        peak = max(peak, t["cap_after"])
        dd = (peak - t["cap_after"]) / peak * 100
        mdd = max(mdd, dd)
    yrs = 3.00
    cagr = ((final_cap / 1_000_000) ** (1 / yrs) - 1) * 100
    return {
        "n": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": wins / max(1, wins + losses) * 100,
        "final": final_cap,
        "total_pnl": total_pnl,
        "cagr": cagr,
        "mdd": mdd,
        "calmar": cagr / max(0.01, mdd),
    }


def cap_breakdown(trades):
    buckets = {}
    for t in trades:
        seg, _ = cap_segment(t.get("sym", "?"))
        b = buckets.setdefault(seg, {"n": 0, "w": 0, "l": 0, "pnl": 0})
        b["n"] += 1
        if t["pnl"] > 0: b["w"] += 1
        if t["pnl"] < 0: b["l"] += 1
        b["pnl"] += t["pnl"]
    return buckets


def write_summary(model_dir, meta, trades):
    s = stats(trades)
    out_path = ROOT / "exports/models" / model_dir / "SUMMARY.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_entry = trades[0].get("entry_date") or trades[0].get("entry", "?")
    last_exit  = trades[-1].get("exit_date") or trades[-1].get("exit", "?")
    md = f"""# {model_dir} — SUMMARY

**{meta['title']}**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | {first_entry} |
| Last exit | {last_exit} |
| Total trades | {s['n']} |
| Trades per year | ~{s['n']/3.0:.1f} |
| Rebalance | {meta['rebalance']} |
| Data source | **{meta['data']}** |

## Stock pick logic

"""
    for i, step in enumerate(meta["logic"], 1):
        md += f"{i}. {step}\n"
    md += f"""
## Headline result

| Metric | Value |
|---|---:|
| Final NAV | **₹{s['final']:,.0f}** |
| Total return | **{(s['final']/1_000_000 - 1)*100:+.2f}%** |
| 3-yr CAGR | **{s['cagr']:+.2f}%** |
| Max DD (rebal cap_after) | **{s['mdd']:.2f}%** |
| Calmar (CAGR / Max DD) | **{s['calmar']:.2f}** |
| Trades | {s['n']} |
| Wins / Losses | {s['wins']} / {s['losses']} |
| Win rate | {s['wr']:.1f}% |
| Live deployment | {'✅ YES' if meta['live'] else '❌ NO'} |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
"""
    for seg in ["Large", "Mid", "Small", "Other"]:
        b = cap_breakdown(trades).get(seg)
        if not b: continue
        md += f"| **{seg}** | {b['n']} | {b['w']} | {b['l']} | {b['w']/max(1,b['w']+b['l'])*100:.0f}% | {b['pnl']:+,.0f} |\n"

    # Top 5 winners + losers
    by_pnl = sorted(trades, key=lambda t: -t["pnl"])
    md += "\n## Top 5 winners\n\n| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |\n|---|---|---:|---:|---:|\n"
    for t in by_pnl[:5]:
        ent = t.get("entry_date") or t.get("entry", "")
        ext = t.get("exit_date") or t.get("exit", "")
        md += f"| {t.get('sym','?'):<12} | {ent} → {ext} | {t.get('entry_px',0):,.2f} | {t.get('ret_pct',0):+.2f}% | {t['pnl']:+,.0f} |\n"
    md += "\n## Top 5 losses\n\n| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |\n|---|---|---:|---:|---:|\n"
    for t in by_pnl[::-1][:5]:
        if t["pnl"] >= 0: break
        ent = t.get("entry_date") or t.get("entry", "")
        ext = t.get("exit_date") or t.get("exit", "")
        md += f"| {t.get('sym','?'):<12} | {ent} → {ext} | {t.get('entry_px',0):,.2f} | {t.get('ret_pct',0):+.2f}% | {t['pnl']:+,.0f} |\n"

    md += f"\nFull trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).\n"
    out_path.write_text(md)
    print(f"  wrote {out_path}")


def write_ledger(model_dir, meta, trades):
    s = stats(trades)
    out_path = ROOT / "exports/models" / model_dir / "TRADE_LEDGER.md"
    md = f"""# {model_dir} — Trade Ledger

₹10L → ₹{s['final']:,.0f} ({(s['final']/1_000_000-1)*100:+.2f}%) · CAGR {s['cagr']:+.2f}% · {s['n']} trades · Max DD {s['mdd']:.2f}% · WR {s['wr']:.1f}%

Data: {meta['data']}. {meta['title']}

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
"""
    for seg in ["Large", "Mid", "Small", "Other"]:
        b = cap_breakdown(trades).get(seg)
        if not b: continue
        md += f"| **{seg}** | {b['n']} | {b['w']} | {b['l']} | {b['w']/max(1,b['w']+b['l'])*100:.0f}% | {b['pnl']:+,.0f} |\n"

    md += f"\n## All {len(trades)} trades\n\n"
    md += "| # | Symbol | Cap | Index | Entry | Entry ₹ | Qty | Invested ₹ | Exit | Exit ₹ | PnL ₹ | Ret % |\n"
    md += "|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|\n"
    for i, t in enumerate(trades, 1):
        ent = t.get("entry_date") or t.get("entry", "")
        ext = t.get("exit_date") or t.get("exit", "")
        sym = t.get("sym", "?")
        seg, idx = cap_segment(sym)
        epx = t.get("entry_px", 0)
        xpx = t.get("exit_px", 0)
        qty = t.get("qty", 0)
        inv = qty * epx
        md += (f"| {i} | {sym} | **{seg}** | {idx} | {ent} | {epx:,.2f} | {qty:,} | ₹{inv:,.0f} | "
               f"{ext} | {xpx:,.2f} | {t['pnl']:+,.0f} | {t.get('ret_pct',0):+.2f}% |\n")
    out_path.write_text(md)
    print(f"  wrote {out_path}")


def main():
    for model, meta in MODELS.items():
        ledger_path = ROOT / "tools/models" / model / "trade_ledger.json"
        if not ledger_path.exists():
            print(f"SKIP {model}: no ledger at {ledger_path}")
            continue
        with open(ledger_path) as f:
            trades = json.load(f)
        if not trades:
            print(f"SKIP {model}: empty ledger")
            continue
        print(f"{model}: {len(trades)} trades")
        write_summary(model, meta, trades)
        write_ledger(model, meta, trades)


if __name__ == "__main__":
    main()
