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
        "note": "Best model. Vol-adjusted momentum (return per unit of volatility) on the mid/small universe, PLUS a 2.5× ATR-from-entry hard stop (2026-06-01, backtest-validated both windows + every year): +121.0% CAGR / 37.9% DD / Calmar 3.19 / 64% win full-cycle 2021-03→2026-05; recent 2023-05→2026-05 +165% CAGR / 26% DD / 75% win. ALL figures UNLEVERED (own cash only — no borrow); rechecked 2026-06-05 against the live core (PIT, no-lookahead, backtest==live). The stop is a FIXED level at entry − 2.5×ATR (NOT trailing) so it cuts genuine breakdowns without whipsawing winners. Per-year: 2021 +20 / 2022 +150 / 2023 +358 / 2024 +171 / 2025 +46 / 2026 +15. Shared helper strategy.atr_stop_hit used by both backtest and the live --stop-check (no drift). The one model that crosses 100% organically (no leverage).",
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
             "- `midcap_narrow_60d_breakout` RETIRED 2026-06-12 (was effectively DEAD full-cycle: +11.2% / 57% DD / Calmar 0.2, 13 trades/5yr); code in tools/models/_archived.",
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
