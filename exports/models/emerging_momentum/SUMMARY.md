# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) momentum rotation: 15d return >0, price ≤₹3000 (no SMA gate); retain top-3; monthly (1st trading day) + mid-month check that rotates only on a ≥5pp lead.

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2025-03-01 → 2026-05-12** (recent ~14 months: 2025 chop + 2026 bear).

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹1,817,677 |
| Total return | +81.8% |
| CAGR (annualized) | +64.8% |
| Max drawdown | 13.7% |
| Calmar | 4.71 |
| Trades | 14 (9W / 5L) · 64% win |

## Note

Single-position max-1, emerging mid/small, run_rotation engine + MCAP-CLIMBER filter (keep only rising-mcap-rank names). Full-period 2023-26 ≈ +111% CAGR / 23% DD gross (climber ON; OFF baseline ≈ +98%).

**Open position at window end:** BHEL qty 4639 entry ₹377.05 on 2026-05-04 (unrealized +68,193)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
