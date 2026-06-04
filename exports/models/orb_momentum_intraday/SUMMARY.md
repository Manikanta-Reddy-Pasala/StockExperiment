# Morning ORB (Intraday) (`orb_momentum_intraday`)

**Status:** OBSERVE (paper)  
DAY-TRADE: watch top-3 momentum leaders (≥₹100); go ALL-IN (full capital, ONE position) on the FIRST to break above its 15-min opening-range high before 10:00; stop=OR-low, target=2×range, FORCED FLAT by 15:15. Zero overnight. One trade/day, no re-entry.

**Universe:** Nifty 500 (PIT) — top-3 by 20d momentum, traded intraday

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Intraday, every day. Selection at the open; entries scanned each 5-min bar 09:30–10:00. |
| **Universe & filters** | Nifty 500 (PIT eligible_at), MIN_PRICE ≥ ₹100 (sub-₹100 penny names dropped — their tiny opening ranges whipsaw the ORB into fake breakouts). Each day rank by 20-day return, WATCH the top-3 momentum leaders. |
| **Entry** | ALL-IN, single position: commit the FULL capital to the FIRST of the 3 watched leaders to break above its 15-min opening-range high (rank as tiebreak), but ONLY before 10:00. Long-only. One trade/day — no re-entry after the exit. |
| **Exit** | Stop at opening-range low; target at OR-high + 2×range; else forced flat at 15:15 (EOD_FLAT_MIN — robust peak of the EOD-time sweep; the 15:15→close bar fades hard). Always flat overnight (0-day hold). |
| **Source** | Selection: DB historical_data (daily close, Fyers) + PIT n500. Execution: Fyers 5-min bars (resolution=5), cached under tools/models/orb_momentum_intraday/cache5min/. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹4,585,775 |
| Total return | +358.6% |
| CAGR (annualized) | +426.7% |
| Max drawdown | 21.7% |
| Calmar | 19.65 |
| Trades | 231 (124W / 107L) · 54% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2025 | +272.1% | 21.7% |
| 2026 | +23.2% | 13.3% |

## Note

The only intraday model. Momentum SELECT + long-only opening-range-breakout EXECUTION (momentum filter is the edge — raw ORB on random names is −13%). 2025-03→2026-05 on PIT N500 (realistic 0.15% slippage + 0.15% round-trip): +359% total / +427% CAGR / 21.7% DD / Calmar 19.65 / Sharpe ~3.96 / 231 trades / WR 54%. THREE 2026-06-04 changes: (1) MIN_PRICE ≥ ₹100 (drops sub-₹100 penny whipsaws — IDEA ~₹8, ALLCARGO — that fake-fired live); (2) EOD square-off moved off the close to 15:15 (sweep: the 15:15→15:30 bar fades hard, 15:25=+235% vs the 15:00-15:20 plateau peak); (3) ALL-IN SINGLE POSITION — full capital into the one best-momentum breakout, replacing the old 1/SELECT_TOP per-slot split that left ~45% of capital idle (≈49% of days only one of three leaders fires). The single all-in ≈ tripled the deployed return (fixed-slot ~+106% total vs +359%) at a higher DD (5.7%→21.7%) — same trade quality, more capital at work. Selection (rank_momentum+MIN_PRICE+pick_leader), exit (live_exit_reason) and EOD (EOD_FLAT_MIN) all in shared strategy → live == backtest, no drift; the only residual gap is fill PRICE (slippage model vs live LTP). ⚠️ SLIPPAGE-SENSITIVE and validated on ONE bull regime only (Feb-26 −11.9% chop-risk; no intraday bear tested). Cron scans every 5 min 09:30→15:10 (entries pre-10:00, stop/target all session) + 15:15 square-off, INTRADAY/MIS. OBSERVE/paper — flip to live in Settings after paper fills confirm slippage.

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
