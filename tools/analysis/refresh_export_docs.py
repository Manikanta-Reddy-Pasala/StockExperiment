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
        "strategy": "Monthly (1st trading day) rotation, single position (rank-1, RET1), 30-trading-day return rank, uptrend (>200d SMA) + ≤₹3K filter. Universe rebuilt yearly at a FIXED mid-May anchor. + DAILY from-entry ATR×3.0 hard stop (entry − 3×ATR(14)).",
        "note": "⚠️ ADV-ranked pseudo-N100 (not the real index) — selects already-liquid/hot names, an OPTIMISTIC upper bound vs the real-index sibling. NOW with a from-entry ATR×3.0 hard stop (2026-06-02, backtest-validated both windows): full-cycle 2021-03→2026-05 +77.4% CAGR / 43.8% DD / Calmar 1.77 / 74% win; recent 2025-03→2026-05 +209% CAGR / 16% DD. The stop is a FIXED level at entry−3×ATR (cuts genuine breakdowns, winners run to rotation); shared helper tools.shared.stops used by backtest + live --stop-check (no drift). DD is now on the stricter DAILY-MTM (intraday-low) basis — not comparable to the prior rebal-snapshot DD; the stop's gain is the within-basis delta (50.1→43.8). ADV-selection bias remains by design.",
    },
    "momentum_n100_top5_max1": {
        "title": "Nifty 100 Momentum", "live": "LIVE",
        "universe": "Real NSE Nifty 100 (PIT membership)",
        "strategy": "Monthly rotation + mid-month check, single position (max 1), 15-trading-day return rank. + DAILY from-entry FIXED −12% hard stop (entry × 0.88).",
        "note": "True-index version — the trustworthy-clean momentum benchmark. NOW with a from-entry fixed −12% hard stop (2026-06-02, backtest-validated): full-cycle 2021-03→2026-05 +59.9% CAGR / 46.4% DD / Calmar 1.29; recent 2025-03→2026-05 +111% CAGR / 15% DD; 2022-23 crash +95.7% CAGR / 27.8% DD — the stop's big win (was 68.7/42.9). Stop = entry×(1−0.12), checked daily on the low; shared backtest+live helper tools.shared.stops (no drift). Fixed-% fits these large-caps (uniform vol); ATR was DD-only, a price-floor threshold-fragile. DD now DAILY-MTM (stricter than the old rebal-snapshot; within-basis delta 56.8→46.4).",
    },
    "n40": {
        "title": "Weekly Top-40", "live": "LIVE",
        "universe": "Top-40 by ADV ∩ Nifty 100",
        "strategy": "WEEKLY rotation (first trading day of each ISO week), single position, uptrend gate.",
        "note": "Weekly rebalance cut the daily whipsaw (55% of daily trades held ≤3d). Full-cycle 2021-03→2026-05 on the AUTHORITATIVE PIT Nifty-100 (2026-05-31 rebuild) ≈ +41.2% CAGR / 36.9% DD / Calmar 1.12 — the clean membership (no DUMMYREL/BHEL/IDEA garbage) lifted it from +25%/0.45. Recent 2025-03→2026-05 ≈ +96% CAGR / 24% DD. Per-year DD ≤27% every year — the 37% full-cycle is 2021→2022 peak-to-trough chaining.",
    },
    "momentum_retest_n500": {
        "title": "Retest Momentum", "live": "DISABLED (₹0)",
        "universe": "Top-120 by 20d ADV from N500 (minus Smallcap-250)",
        "strategy": "Monthly top-4 (K=4), 30d momentum, buy within 20% of 20-EMA, retain top-4 band.",
        "note": "Multi-holding K=4 (2026-05-31 re-tune, was K2). Full-cycle 2021-03→2026-05 ≈ +57.3% CAGR / 38.8% DD / Calmar 1.48. Recent 2025-03→2026-05 ≈ +53% CAGR / 15% DD. K2→K4 diversified the basket: recent CAGR +38→+53, recent DD 21→15, and full DD 57→39 (per-year DD now ≤32 EVERY year: 2021 D24 / 2022 D32 / 2023 D21 / 2024 D18 / 2025 D22 / 2026 D15) — for only −7pt full CAGR. K-knee: K5/K6 decay. The old K2 (+64/57) concentrated into 2 names and chained a 57% peak-to-trough; K4 is the better risk-adjusted config. Wide 20% entry band keeps leaders that never pull back to the EMA.",
    },
    "emerging_momentum": {
        "title": "Emerging Momentum", "live": "LIVE",
        "universe": "Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)",
        "strategy": "Single-position (max-1) VOL-ADJUSTED momentum: rank by 30d return ÷ 60d return-volatility; ret>0, price ≤₹3000 (no SMA gate); RET1 top-1 rotation; monthly (1st trading day) + mid-month check (≥5pp lead). + DAILY ATR-from-entry hard stop (entry − 2.5× ATR(14)).",
        "note": "Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe, PLUS a 2.5× ATR-from-entry hard stop (2026-06-01, backtest-validated both windows + every year): +119.9% CAGR / 37.9% DD / Calmar 3.16 / 64% win full-cycle 2021-03→2026-05; recent 2023-05→2026-05 +167.7% CAGR / 26.3% DD / 75% win (+27% / +37% net P&L vs the rotation-only baseline). The stop is a FIXED level at entry − 2.5×ATR (NOT trailing) so it cuts genuine breakdowns without whipsawing winners. Per-year: 2021 +20 / 2022 +150 / 2023 +358 / 2024 +171 / 2025 +46 / 2026 +15. Shared helper strategy.atr_stop_hit used by both backtest and the live --stop-check (no drift). The one model that crosses 100% organically (no leverage).",
    },
    "midcap_narrow_60d_breakout": {
        "title": "Midcap Breakout", "live": "LIVE",
        "universe": "PIT midcap — top-100 ADV from N500 minus Nifty 100 (excluded at SCAN time)",
        "strategy": "Event-driven single-position breakout: 40d-high + 2× vol + >200DMA. Target +100% / stop −20% / trail −20% off peak / 120d max-hold.",
        "note": "⚠️ Lumpy single-position event model (only ~16 trades/5yr). On AUTHORITATIVE PIT membership (2026-05-31) the full-cycle 2021-03→2026-05 is ≈ +11.2% CAGR / 57% DD / Calmar 0.2 (only 13 trades/5yr) — effectively DEAD. Its earlier +40% was living off large-cap winners that leaked through the buggy Wayback N100 exclusion; with the correct PIT N100 removed it has no edge. Confirms the long-standing 'midcap ignore' call.",
    },
    "orb_momentum_intraday": {
        "title": "Morning ORB (Intraday)", "live": "OBSERVE (paper)",
        "universe": "Nifty 500 (PIT) — top-3 by 20d momentum, traded intraday",
        "strategy": "DAY-TRADE: each day pick top-3 momentum leaders, LONG the 15-min opening-range breakout if it fires before 10:00, stop=OR-low, target=2×range, FORCED FLAT by 15:10. Zero overnight. Sizing = invested/SELECT_TOP per slot (₹30k/3 = ₹10k per leader); already-held names not re-bought.",
        "note": "The only intraday model. Momentum SELECT + long-only opening-range-breakout EXECUTION (momentum filter is the edge — raw ORB on random names is −13%). MIN_PRICE ≥ ₹100 filter added 2026-06-04 (sub-₹100 pennies — IDEA ~₹8, ALLCARGO ~₹12 — whipsawed the tiny opening range into fake breakouts, both backtest and live; e.g. the 06-04 live IDEA fake signal). 2025-03→2026-05 on PIT N500 (realistic 0.15% slippage + 0.15% round-trip): +275% total / +323% CAGR / 14.1% DD / Calmar 22.92 / Sharpe ~4.04 / 380 trades / WR 54%. TWO 2026-06-04 changes: (1) MIN_PRICE ≥ ₹100 filter (drops sub-₹100 penny whipsaws — IDEA/ALLCARGO — that fake-fired live); (2) EOD square-off moved from the close (15:25 bar) to 15:15 after an EOD-time sweep showed the final 15:15→15:30 window FADES the intraday gains: 15:25 +235%/15.5%DD vs the robust 15:00-15:20 plateau (~+287-323% / ~13.5-14% DD), peak 15:15. Both the filter and EOD time live in shared strategy (rank_momentum / EOD_FLAT_MIN) → backtest and live use identical selection AND exit (no drift); 15:15 is before broker MIS auto-square-off (~15:20). ⚠️ SLIPPAGE-SENSITIVE (degrades to ~+46-90% at 0.25% slip) and validated on ONE bull regime only (Feb-26 −11.9% shows chop-risk; no intraday bear tested). WIRED 2026-06-01: cron breakout scans 09:30-09:55 + 15:10 auto-square-off, multi-holding (cash/3 per slot), INTRADAY/MIS product. Running in OBSERVE (signals_only) — flip to live in Settings after paper fills confirm slippage.",
    },
}

WIN = "Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**."

# Per-model trade rules — transcribed from the model code (strategy.py + backtest
# rank_at / run loop) as the single source of truth. Imported by regen_exports too
# so every model SUMMARY.md carries the same Rebalance / Filters / Entry / Exit block.
RULES = {
    "n40": {
        "Rebalance": "First trading day of each ISO week (WEEKLY).",
        "Universe & filters": "Top-40 by 20d ADV from N500, intersect PIT Nifty 100, and close > 200d SMA (uptrend).",
        "Entry": "On the weekly rebalance, BUY rank-1 by 30-day return among the filtered set (single position, max 1).",
        "Exit": "Rotate: SELL when the held name is no longer rank-1 (RETAIN=1) at the next weekly rebalance, or when it drops out of Nifty 100 / below its 200d SMA.",
        "Source": "Live: niftyindices.com `ind_nifty100list.csv` + `ind_nifty500list.csv` → nifty100.csv/nifty500.csv. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV.",
    },
    "momentum_n100_top5_max1": {
        "Rebalance": "1st trading day of month + a mid-month (day-15) lead check.",
        "Universe & filters": "Real point-in-time NSE Nifty 100 (eligible_at). No price/SMA filter — pure index membership.",
        "Entry": "BUY rank-1 by 15-day return (single position, max 1).",
        "Exit": "Hold while in the top-3 by 15d return (RETAIN=3); rotate out when it drops below rank-3, or leaves the index. Mid-month only rotates if the new rank-1 leads the held name by ≥ 5pp.",
        "Source": "Live: niftyindices.com `ind_nifty100list.csv` → nifty100.csv → n100_current.json. Backtest: PIT `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV.",
    },
    "momentum_pseudo_n100_adv": {
        "Rebalance": "1st trading day of month (mid-month available as opt-in, default OFF).",
        "Universe & filters": "Top-100 by 20d ADV from PIT N500 (yearly fixed mid-May anchor) minus Smallcap-250; close > 200d SMA; price ≤ ₹3000.",
        "Entry": "BUY rank-1 by 30-day return (single position, max 1).",
        "Exit": "Rotate: SELL when the held name is no longer rank-1 (RETAIN=1).",
        "Source": "Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → yearly_universes.json. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV.",
    },
    "emerging_momentum": {
        "Rebalance": "1st trading day of month + a mid-month (day-15) lead check.",
        "Universe & filters": "Top-100 by 20d ADV from (PIT N500 minus PIT N100); 30d return > 0; price ≤ ₹3000; NO SMA gate. MCAP-climber OFF.",
        "Entry": "BUY rank-1 by VOL-ADJUSTED momentum (30d return ÷ 60d return-volatility) — single position, max 1.",
        "Exit": "Rotate when held is no longer rank-1 (RET1). Mid-month only rotates if the new rank-1 leads the held by ≥ 5pp.",
        "Source": "Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Mcap-climber: `exports/nse_mcap.csv`. Prices: Fyers daily OHLCV.",
    },
    "momentum_retest_n500": {
        "Rebalance": "Monthly (1st trading day) re-ranks the leaders; entry is scanned DAILY.",
        "Universe & filters": "Top-120 by 20d ADV from PIT N500 minus Smallcap-250 (incl large+mid, NOT N100-excluded); close > 200d SMA; price ≤ ₹3000; 30d return > 10% (mom floor); 10d return > 0 (accelerating).",
        "Entry": "Each month, watch the top-K=4 leaders; BUY one when its price sits within the retest band of the 20-EMA — between 20EMA×(1−1%) and 20EMA×(1+20%) — checked daily. Holds up to 4 equal-weight positions.",
        "Exit": "Rotate: SELL a holding at the monthly rebalance when it drops out of the top-4 by 30d return (RETAIN=4).",
        "Source": "Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → nifty500.csv. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV.",
    },
    "midcap_narrow_60d_breakout": {
        "Rebalance": "Breakouts scanned DAILY; the eligible band is rebuilt each year-start.",
        "Universe & filters": "Top-100 by 20d ADV from (PIT N500 minus PIT N100, excluded at scan time); close > 200d SMA.",
        "Entry": "BUY (next day's open) on a breakout: close > prior-40-day high AND volume ≥ 2× the 20d average volume. Single position (max 1); the highest volume-ratio breakout wins.",
        "Exit": "Event exits (whichever first): target +100%, hard stop −20%, trailing stop −20% off the peak, or 120-day max hold.",
        "Source": "Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV.",
    },
    "orb_momentum_intraday": {
        "Rebalance": "Intraday, every day. Selection at the open; entries scanned each 5-min bar 09:30–10:00.",
        "Universe & filters": "Nifty 500 (PIT eligible_at), MIN_PRICE ≥ ₹100 (sub-₹100 penny names dropped — their tiny opening ranges whipsaw the ORB into fake breakouts). Each day rank by 20-day return, take top-3 momentum leaders; trade only those.",
        "Entry": "LONG when price breaks above the 15-min opening-range high (first 3× 5-min bars, 09:15–09:30), but ONLY if the breakout fires before 10:00. Long-only (no shorts). Equal-weight across leaders that break out.",
        "Exit": "Stop at opening-range low; target at OR-high + 2×range; else forced flat at 15:15 (EOD_FLAT_MIN — robust peak of the EOD-time sweep; the 15:15→close bar fades hard). Always flat overnight (0-day hold).",
        "Source": "Selection: DB historical_data (daily close, Fyers) + PIT n500. Execution: Fyers 5-min bars (resolution=5), cached under tools/models/orb_momentum_intraday/cache5min/.",
    },
}


def rules_block(model: str) -> list:
    """Markdown lines for the Rebalance / Filters / Entry / Exit / Source block."""
    r = RULES.get(model)
    if not r:
        return []
    out = ["", "## Trade rules", "", "| When | Rule |", "|---|---|"]
    for k in ("Rebalance", "Universe & filters", "Entry", "Exit", "Source"):
        if r.get(k):
            out.append(f"| **{k}** | {r[k]} |")
    return out


def f(v, suf="%"):
    return f"{v:+.1f}{suf}" if isinstance(v, (int, float)) else str(v)


def write_summary(model: str, d: dict):
    info = DESC.get(model, {})
    title = info.get("title", model)
    lines = [f"# {title} (`{model}`)", "",
             f"**Status:** {info.get('live','—')}  ", info.get("strategy", ""), "",
             f"**Universe:** {info.get('universe','—')}", "", WIN]
    lines += rules_block(model)
    lines += ["", "## Results (net of costs)", "",
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
    py = d.get("per_year") or {}
    if py:
        lines += ["", "## Year-by-year breakdown", "", "| Year | Return % | Intra-yr DD % |", "|---|---:|---:|"]
        for y in sorted(py, key=lambda x: int(x)):
            v = py[y]
            if isinstance(v, dict):
                lines.append(f"| {y} | {v.get('ret_pct',0):+.1f}% | {v.get('dd_pct',0):.1f}% |")
            else:
                lines.append(f"| {y} | {v:+.1f}% | — |")
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
    rows = ["# " + model + " — trade ledger (2021-03-01 → 2026-05-29)", "",
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
             "- Single-position models (pseudo / n100 / n40 / emerging / midcap) concentrate → bigger window numbers; the multi-holding retest (K=4, top-120 ADV from N500 minus Smallcap-250 — includes large+mid, NOT N100-excluded) diversifies → smoother.",
             "- `midcap_narrow_60d_breakout` is effectively DEAD full-cycle (+11.2% / 57% DD / Calmar 0.2, only 13 trades/5yr) on authoritative PIT data — only select windows flatter it.",
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
