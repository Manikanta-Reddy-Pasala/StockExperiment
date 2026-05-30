# Midcap Breakout (`midcap_narrow_60d_breakout`)

**Status:** LIVE  
Event-driven single-position breakout: 40d-high + 2× vol + >200DMA. Target +100% / stop −20% / trail −20% off peak / 120d max-hold.

**Universe:** PIT midcap — top-100 ADV from N500 minus Nifty 100 (excluded at SCAN time)

Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear).

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹5,673,731 |
| Total return | +467.4% |
| CAGR (annualized) | +40.0% |
| Max drawdown | 22.0% |
| Calmar | 1.82 |
| Trades | 15 (10W / 5L) · 67% win |

## Note

⚠️ Lumpy single-position event model (only ~15 trades/5yr). Full-cycle 2021-04→2026-05 ≈ +40% CAGR / 22% DD / Calmar 1.82 after the 2026-05-31 trade-time PIT-Nifty-100 exclusion fix (a name promoted to large mid-year was leaking in and dragging returns; excluding it at scan time lifted CAGR +13pp and halved DD).

**Open position at window end:** TEJASNET qty 10902 entry ₹538.94 on 2026-05-08 (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
