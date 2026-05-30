# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) momentum rotation: 15d return >0, price ≤₹3000 (no SMA gate); retain top-3; monthly (1st trading day) + mid-month check that rotates only on a ≥5pp lead.

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear).

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹60,708,657 |
| Total return | +5970.9% |
| CAGR (annualized) | +121.7% |
| Max drawdown | 24.9% |
| Calmar | 4.88 |
| Trades | 59 (39W / 20L) · 66% win |

## Note

Single-position max-1, emerging mid/small, run_rotation engine + MCAP-CLIMBER filter (keep only rising-mcap-rank names). Full-period 2023-26 ≈ +111% CAGR / 23% DD gross (climber ON; OFF baseline ≈ +98%).

**Open position at window end:** BHEL [mid] qty 145671 entry ₹377.05 on 2026-05-04 (unrealized +5,783,139)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
