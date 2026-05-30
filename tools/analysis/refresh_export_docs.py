"""Regenerate exports/models/*/SUMMARY.md + TRADE_LEDGER.md + the top-level index
from the freshly-written summary.json / trade_ledger.json, so the human-readable
docs always match the real backtest data (no stale hand-edited numbers).

Run AFTER regenerating the summary.json files. Pure formatting, no backtest.
Run: python3 tools/analysis/refresh_export_docs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "exports" / "models"

# Per-model descriptors (title, universe, one-line strategy, live status).
DESC = {
    "momentum_pseudo_n100_adv": {
        "title": "Liquid 100 Momentum", "live": "LIVE",
        "universe": "Top-100 by 20d ADV from N500 (yearly-PIT rebuild)",
        "strategy": "Monthly (1st trading day) rotation, single position (rank-1, RET1), 30-trading-day return rank, uptrend (>200d SMA) + ≤₹3K filter. Universe rebuilt yearly at a FIXED mid-May anchor.",
        "note": "⚠️ ADV-ranked pseudo-N100 (not the real index) — selects already-liquid/hot names, so returns are an OPTIMISTIC upper bound vs the real-index sibling momentum_n100_top5_max1. Full-cycle 2021-04→2026-05 (fixed May anchor, PIT N500) ≈ +72.9% CAGR / 28.6% DD / Calmar 2.54 / 76% win. Now PIT (2026-05-31, no survivorship bias) but the ADV-selection bias remains by design.",
    },
    "momentum_n100_top5_max1": {
        "title": "Nifty 100 Momentum", "live": "LIVE",
        "universe": "Real NSE Nifty 100 (PIT membership)",
        "strategy": "Monthly rotation + mid-month check, single position (max 1), 15-trading-day return rank.",
        "note": "True-index version — the trustworthy-clean momentum benchmark.",
    },
    "n40": {
        "title": "Weekly Top-40", "live": "LIVE",
        "universe": "Top-40 by ADV ∩ Nifty 100",
        "strategy": "WEEKLY rotation (first trading day of each ISO week), single position, uptrend gate.",
        "note": "Weekly rebalance cut the daily whipsaw (55% of daily trades held ≤3d). Full-cycle 2021-04→2026-05 on the AUTHORITATIVE PIT Nifty-100 (2026-05-31 rebuild) ≈ +40.3% CAGR / 36.9% DD / Calmar 1.09 — the clean membership (no DUMMYREL/BHEL/IDEA garbage) lifted it from +25%/0.45.",
    },
    "momentum_retest_n500": {
        "title": "Retest Momentum", "live": "DISABLED (₹0)",
        "universe": "Top-120 by 20d ADV from N500 (minus Smallcap-250)",
        "strategy": "Monthly top-2 (K=2), 30d momentum, buy within 20% of 20-EMA, retain top-4 band.",
        "note": "Multi-holding K=2 (2026-05-30 sweep). Now PIT N500 (2026-05-31): full-cycle 2021-04→2026-05 ≈ +64.9% CAGR / 57.1% DD / Calmar 1.14. The earlier +77.8%/2.08 was survivorship-inflated (static current N500); switching to PIT eligible_at deflated CAGR and roughly doubled DD — the honest number. K=2 + wide 20% entry band (vs K3/band-8%) stops missing leaders that never pull back to the EMA.",
    },
    "emerging_momentum": {
        "title": "Emerging Momentum", "live": "LIVE",
        "universe": "Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)",
        "strategy": "Single-position (max-1) momentum rotation: 15d return >0, price ≤₹3000 (no SMA gate); retain top-3; monthly (1st trading day) + mid-month check that rotates only on a ≥5pp lead.",
        "note": "Single-position max-1, emerging mid/small, run_rotation engine + MCAP-CLIMBER filter (keep only rising-mcap-rank names). Full-cycle 2021-04→2026-05 on AUTHORITATIVE PIT membership (2026-05-31) ≈ +46.1% CAGR / 37.7% DD / Calmar 1.22. ⚠ The old +121% headline was a MIRAGE — the buggy Wayback N100 was missing large-cap winners (ADANIGREEN etc), so this 'mid/small' model wrongly held them. Correct N100 exclusion ⇒ genuinely mid/small ⇒ +46%.",
    },
    "midcap_narrow_60d_breakout": {
        "title": "Midcap Breakout", "live": "LIVE",
        "universe": "PIT midcap — top-100 ADV from N500 minus Nifty 100 (excluded at SCAN time)",
        "strategy": "Event-driven single-position breakout: 40d-high + 2× vol + >200DMA. Target +100% / stop −20% / trail −20% off peak / 120d max-hold.",
        "note": "⚠️ Lumpy single-position event model (only ~16 trades/5yr). On AUTHORITATIVE PIT membership (2026-05-31) the full-cycle 2021-04→2026-05 is ≈ +1.65% CAGR / 68% DD / Calmar 0.02 — effectively DEAD. Its earlier +40% was living off large-cap winners that leaked through the buggy Wayback N100 exclusion; with the correct PIT N100 removed it has no edge. Confirms the long-standing 'midcap ignore' call.",
    },
}

WIN = "Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear)."


def f(v, suf="%"):
    return f"{v:+.1f}{suf}" if isinstance(v, (int, float)) else str(v)


def write_summary(model: str, d: dict):
    info = DESC.get(model, {})
    title = info.get("title", model)
    lines = [f"# {title} (`{model}`)", "",
             f"**Status:** {info.get('live','—')}  ", info.get("strategy", ""), "",
             f"**Universe:** {info.get('universe','—')}", "", WIN, "",
             "## Results (net of costs)", "",
             "| Metric | Value |", "|---|---|"]
    if d.get("final_nav"):
        lines.append(f"| Final NAV (₹10L start) | ₹{d['final_nav']:,.0f} |")
    if "total_return_pct" in d:
        lines.append(f"| Total return | {f(d['total_return_pct'])} |")
    lines.append(f"| CAGR (annualized) | {f(d['cagr_pct'])} |")
    lines.append(f"| Max drawdown | {d['max_dd_pct']:.1f}% |")
    lines.append(f"| Calmar | {d['calmar']} |")
    if "trades" in d:
        wl = ""
        if "wins" in d and "losses" in d:
            wl = f" ({d['wins']}W / {d['losses']}L)"
        wr = f" · {d['win_rate_pct']:.0f}% win" if "win_rate_pct" in d else ""
        lines.append(f"| Trades | {d['trades']}{wl}{wr} |")
    if d.get("per_year"):
        py = " · ".join(f"{y} {v:+.1f}%" for y, v in sorted(d["per_year"].items())
                        if isinstance(v, (int, float)))
        if py:
            lines += ["", f"**Per calendar year:** {py}"]
    if info.get("note"):
        lines += ["", "## Note", "", info["note"]]
    # Single open_position OR multi open_positions (e.g. retest K=3). Note the
    # json key is 'sym' (not 'symbol') — fall back to both.
    ops = d.get("open_positions") or ([d["open_position"]] if d.get("open_position") else [])
    for op in ops:
        sym = op.get("sym") or op.get("symbol") or "?"
        cap = op.get("cap")
        cap_s = f" [{cap}]" if cap else ""
        lines += ["", f"**Open position at window end:** {sym}{cap_s} "
                  f"qty {op.get('qty','?')} entry ₹{op.get('entry_px','?')} "
                  f"on {op.get('entry_date','?')} (unrealized {op.get('unrealized_pnl',0):+,.0f})"]
    lines += ["", "---", "*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*", ""]
    (EXPORTS / model / "SUMMARY.md").write_text("\n".join(lines))


def write_ledger(model: str):
    tl_path = EXPORTS / model / "trade_ledger.json"
    if not tl_path.exists():
        return
    trades = json.loads(tl_path.read_text())
    if not isinstance(trades, list) or not trades:
        (EXPORTS / model / "TRADE_LEDGER.md").write_text(f"# {model} — trade ledger\n\nNo closed trades in window.\n")
        return
    # common columns across model schemas
    def g(t, *keys):
        for k in keys:
            if k in t and t[k] is not None:
                return t[k]
        return ""
    def money(v):    # ₹ with 2dp, blank if absent
        return f"{v:,.2f}" if isinstance(v, (int, float)) else (v or "")
    def intg(v):     # integer with thousands, blank if absent
        return f"{v:,.0f}" if isinstance(v, (int, float)) else (v or "")
    rows = ["# " + model + " — trade ledger (2021-04-01 → 2026-05-29)", "",
            "| # | Symbol | Cap | Entry date | Exit date | Entry ₹ | Exit ₹ | Qty | PnL ₹ | Return % | Reason |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for i, t in enumerate(trades, 1):
        rows.append(f"| {i} | {g(t,'sym','symbol')} | {g(t,'cap') or '—'} | {g(t,'entry_date')} | "
                    f"{g(t,'exit_date')} | {money(g(t,'entry_px'))} | {money(g(t,'exit_px'))} | "
                    f"{intg(g(t,'qty'))} | {intg(g(t,'pnl'))} | {g(t,'ret_pct','ret')} | "
                    f"{g(t,'reason','exit_reason')} |")
    rows += ["", f"*{len(trades)} trades. Auto-generated from trade_ledger.json.*", ""]
    (EXPORTS / model / "TRADE_LEDGER.md").write_text("\n".join(rows))


def write_index(summaries: dict):
    rows = ["# Model Backtests — Index", "", WIN, "",
            "All figures net of costs, ₹10L start, true point-in-time universes.", "",
            "| Model | Status | CAGR | maxDD | Calmar | Total |", "|---|---|---|---|---|---|"]
    order = sorted(summaries, key=lambda m: -summaries[m].get("cagr_pct", -999))
    for m in order:
        d = summaries[m]; info = DESC.get(m, {})
        rows.append(f"| [{info.get('title', m)}]({m}/SUMMARY.md) | {info.get('live','—')} | "
                    f"{d['cagr_pct']:+.1f}% | {d['max_dd_pct']:.1f}% | {d['calmar']} | "
                    f"{f(d.get('total_return_pct', 0))} |")
    rows += ["", "**Caveats:**",
             "- `momentum_pseudo_n100_adv` = optimistic upper bound (ADV-selection bias, not real index).",
             "- Single-position models (pseudo / n100 / n20 / midcap) concentrate → bigger window numbers; the N500 models (retest K=3, regime K=5) diversify → smoother, lower.",
             "- `midcap_narrow_60d_breakout` is NEGATIVE full-period 2023-26 on current PIT data — this window flatters it.",
             "", "*Auto-generated by tools/analysis/refresh_export_docs.py.*", ""]
    (EXPORTS / "SUMMARY.md").write_text("\n".join(rows))


def main():
    summaries = {}
    for d in sorted(EXPORTS.iterdir()):
        sj = d / "summary.json"
        if d.is_dir() and sj.exists():
            data = json.loads(sj.read_text())
            summaries[d.name] = data
            write_summary(d.name, data)
            write_ledger(d.name)
            print(f"  refreshed {d.name}/SUMMARY.md + TRADE_LEDGER.md")
    write_index(summaries)
    print(f"  refreshed top-level index ({len(summaries)} models)")


if __name__ == "__main__":
    main()
