# Morning ORB (Intraday) (`orb_momentum_intraday`)

**Status:** OBSERVE — NO EDGE  
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
| Final NAV (₹10L start) | ₹369,110 |
| Total return | -63.1% |
| CAGR (annualized) | -71.7% |
| Max drawdown | 67.3% |
| Calmar | -1.07 |
| Trades | 199 (71W / 128L) · 36% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2025 | -37.2% | 41.3% |
| 2026 | -41.2% | 43.5% |

## Note

⛔ NO DEMONSTRATED EDGE — DO NOT ENABLE LIVE. 2026-06-04 recheck found the old positive results (+251/+323/+427% across the day's iterations) were entirely LOOKAHEAD BIAS: the backtest ranked momentum on the SAME-DAY close (15:30) to pick that day's 09:30 morning breakout — using future information live cannot have (the daily bar isn't pulled until ~20:30, so live correctly ranks on the PRIOR close). Fixing the backtest to rank on the prior close (no lookahead, matching live) collapses it to −63% total / −72% CAGR / 67% DD / Sharpe −3.37 / 36% WR, NEGATIVE every year (2025 −37%, 2026 −41%). The intraday breakout of prior-close momentum leaders has no edge. The engineering work is sound and live==backtest (shared rank_momentum+MIN_PRICE+pick_leader / live_exit_reason / EOD_FLAT_MIN; MIN_PRICE≥₹100 still correctly kills the live penny fake-signals like IDEA) — but the SIGNAL loses money, so ORB stays OBSERVE/paper (zero real money at risk) until a genuinely-predictive selection is found. Other 6 models are unaffected: they rank and transact at the SAME observed close, no lookahead.

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
