# Midcap Breakout (`midcap_narrow_60d_breakout`)

**Status:** LIVE  
Event-driven single-position breakout: 40d-high + 2× vol + >200DMA. Target +100% / stop −20% / trail −20% off peak / 120d max-hold.

**Universe:** PIT midcap — top-100 ADV from N500 minus Nifty 100 (excluded at SCAN time)

Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear).

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹1,088,048 |
| Total return | +8.8% |
| CAGR (annualized) | +1.6% |
| Max drawdown | 68.2% |
| Calmar | 0.02 |
| Trades | 16 (6W / 10L) · 38% win |

## Note

⚠️ Lumpy single-position event model (only ~16 trades/5yr). On AUTHORITATIVE PIT membership (2026-05-31) the full-cycle 2021-04→2026-05 is ≈ +1.65% CAGR / 68% DD / Calmar 0.02 — effectively DEAD. Its earlier +40% was living off large-cap winners that leaked through the buggy Wayback N100 exclusion; with the correct PIT N100 removed it has no edge. Confirms the long-standing 'midcap ignore' call.

**Open position at window end:** POWERINDIA qty 28 entry ₹27827.8 on 2026-04-13 (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
