# Morning ORB (Intraday) (`orb_momentum_intraday`)

**Status:** OBSERVE (paper)  
DAY-TRADE: each day pick top-3 momentum leaders, LONG the 15-min opening-range breakout if it fires before 10:00, stop=OR-low, target=2×range, FORCED FLAT by 15:10. Zero overnight. Sizing = invested/SELECT_TOP per slot (₹30k/3 = ₹10k per leader); already-held names not re-bought.

**Universe:** Nifty 500 (PIT) — top-3 by 20d momentum, traded intraday

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Intraday, every day. Selection at the open; entries scanned each 5-min bar 09:30–10:00. |
| **Universe & filters** | Nifty 500 (PIT eligible_at). Each day rank by 20-day return, take top-3 momentum leaders; trade only those. |
| **Entry** | LONG when price breaks above the 15-min opening-range high (first 3× 5-min bars, 09:15–09:30), but ONLY if the breakout fires before 10:00. Long-only (no shorts). Equal-weight across leaders that break out. |
| **Exit** | Stop at opening-range low; target at OR-high + 2×range; else forced flat at 15:25. Always flat overnight (0-day hold, ~3.5h avg). |
| **Source** | Selection: DB historical_data (daily close, Fyers) + PIT n500. Execution: Fyers 5-min bars (resolution=5), cached under tools/models/orb_momentum_intraday/cache5min/. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹3,163,237 |
| Total return | +216.3% |
| CAGR (annualized) | +251.2% |
| Max drawdown | 17.2% |
| Calmar | 14.59 |
| Trades | 377 (199W / 178L) · 53% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2025 | +200.0% | 8.9% |
| 2026 | +5.5% | 17.2% |

## Note

The only intraday model. Momentum SELECT + long-only opening-range-breakout EXECUTION (momentum filter is the edge — raw ORB on random names is −13%). 2025-03→2026-05 on full PIT N500 (realistic 0.15% slippage + 0.15% round-trip): +216% total / +251% CAGR / 17.2% DD / Sharpe ~3.44 / 377 trades / WR 53% / 13-of-15 months green. ⚠️ SLIPPAGE-SENSITIVE (+251% at 0.15% slip → ~+46-90% at 0.25%) and validated on ONE bull regime only (Feb-26 −11.9% shows chop-risk; no intraday bear tested). WIRED 2026-06-01: cron breakout scans 09:30-09:55 + 15:10 auto-square-off, multi-holding (cash/3 per slot), INTRADAY/MIS product. Running in OBSERVE (signals_only) — flip to live in Settings after paper fills confirm slippage.

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
