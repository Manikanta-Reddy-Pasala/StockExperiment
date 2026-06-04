# Morning ORB (Intraday) (`orb_momentum_intraday`)

**Status:** OBSERVE (paper)  
DAY-TRADE: each day pick top-3 momentum leaders, LONG the 15-min opening-range breakout if it fires before 10:00, stop=OR-low, target=2×range, FORCED FLAT by 15:10. Zero overnight. Sizing = invested/SELECT_TOP per slot (₹30k/3 = ₹10k per leader); already-held names not re-bought.

**Universe:** Nifty 500 (PIT) — top-3 by 20d momentum, traded intraday

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Intraday, every day. Selection at the open; entries scanned each 5-min bar 09:30–10:00. |
| **Universe & filters** | Nifty 500 (PIT eligible_at), MIN_PRICE ≥ ₹100 (sub-₹100 penny names dropped — their tiny opening ranges whipsaw the ORB into fake breakouts). Each day rank by 20-day return, take top-3 momentum leaders; trade only those. |
| **Entry** | LONG when price breaks above the 15-min opening-range high (first 3× 5-min bars, 09:15–09:30), but ONLY if the breakout fires before 10:00. Long-only (no shorts). Equal-weight across leaders that break out. |
| **Exit** | Stop at opening-range low; target at OR-high + 2×range; else forced flat at 15:15 (EOD_FLAT_MIN — robust peak of the EOD-time sweep; the 15:15→close bar fades hard). Always flat overnight (0-day hold). |
| **Source** | Selection: DB historical_data (daily close, Fyers) + PIT n500. Execution: Fyers 5-min bars (resolution=5), cached under tools/models/orb_momentum_intraday/cache5min/. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹3,751,312 |
| Total return | +275.1% |
| CAGR (annualized) | +323.0% |
| Max drawdown | 14.1% |
| Calmar | 22.92 |
| Trades | 380 (205W / 175L) · 54% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2025 | +223.4% | 8.1% |
| 2026 | +16.0% | 14.1% |

## Note

The only intraday model. Momentum SELECT + long-only opening-range-breakout EXECUTION (momentum filter is the edge — raw ORB on random names is −13%). MIN_PRICE ≥ ₹100 filter added 2026-06-04 (sub-₹100 pennies — IDEA ~₹8, ALLCARGO ~₹12 — whipsawed the tiny opening range into fake breakouts, both backtest and live; e.g. the 06-04 live IDEA fake signal). 2025-03→2026-05 on PIT N500 (realistic 0.15% slippage + 0.15% round-trip): +275% total / +323% CAGR / 14.1% DD / Calmar 22.92 / Sharpe ~4.04 / 380 trades / WR 54%. TWO 2026-06-04 changes: (1) MIN_PRICE ≥ ₹100 filter (drops sub-₹100 penny whipsaws — IDEA/ALLCARGO — that fake-fired live); (2) EOD square-off moved from the close (15:25 bar) to 15:15 after an EOD-time sweep showed the final 15:15→15:30 window FADES the intraday gains: 15:25 +235%/15.5%DD vs the robust 15:00-15:20 plateau (~+287-323% / ~13.5-14% DD), peak 15:15. Both the filter and EOD time live in shared strategy (rank_momentum / EOD_FLAT_MIN) → backtest and live use identical selection AND exit (no drift); 15:15 is before broker MIS auto-square-off (~15:20). ⚠️ SLIPPAGE-SENSITIVE (degrades to ~+46-90% at 0.25% slip) and validated on ONE bull regime only (Feb-26 −11.9% shows chop-risk; no intraday bear tested). WIRED 2026-06-01: cron breakout scans 09:30-09:55 + 15:10 auto-square-off, multi-holding (cash/3 per slot), INTRADAY/MIS product. Running in OBSERVE (signals_only) — flip to live in Settings after paper fills confirm slippage.

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
