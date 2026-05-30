# Retest Momentum (`momentum_retest_n500`)

**Status:** DISABLED (₹0)  
Monthly top-2 (K=2), 30d momentum, buy within 20% of 20-EMA, retain top-4 band.

**Universe:** Top-120 by 20d ADV from N500 (minus Smallcap-250)

Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear).

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹13,195,632 |
| Total return | +1219.6% |
| CAGR (annualized) | +64.9% |
| Max drawdown | 57.1% |
| Calmar | 1.14 |
| Trades | 88 (47W / 41L) · 53% win |

## Note

Multi-holding K=2 (2026-05-30 sweep). Now PIT N500 (2026-05-31): full-cycle 2021-04→2026-05 ≈ +64.9% CAGR / 57.1% DD / Calmar 1.14. The earlier +77.8%/2.08 was survivorship-inflated (static current N500); switching to PIT eligible_at deflated CAGR and roughly doubled DD — the honest number. K=2 + wide 20% entry band (vs K3/band-8%) stops missing leaders that never pull back to the EMA.

**Open position at window end:** ADANIPOWER [large] qty 30962 entry ₹157.11 on 2026-04-01 (unrealized +2,670,782)

**Open position at window end:** ADANIGREEN [large] qty 3841 entry ₹1290.7 on 2026-05-04 (unrealized +709,433)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
