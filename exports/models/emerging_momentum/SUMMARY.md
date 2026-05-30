# Emerging Momentum (`emerging_momentum`)

**Status:** LIVE  
Single-position (max-1) momentum rotation: 15d return >0, price ≤₹3000 (no SMA gate); retain top-3; monthly (1st trading day) + mid-month check that rotates only on a ≥5pp lead.

**Universe:** Top-100 by 20d ADV from emerging mid/small (PIT N500 minus N100)

Backtest window: **2021-04-01 → 2026-05-29** (full ~5.1-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 bear).

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹7,056,108 |
| Total return | +605.6% |
| CAGR (annualized) | +46.0% |
| Max drawdown | 37.7% |
| Calmar | 1.22 |
| Trades | 62 (37W / 25L) · 60% win |

## Note

Single-position max-1, emerging mid/small, run_rotation engine + MCAP-CLIMBER filter (keep only rising-mcap-rank names). Full-cycle 2021-04→2026-05 on AUTHORITATIVE PIT membership (2026-05-31) ≈ +46.1% CAGR / 37.7% DD / Calmar 1.22. ⚠ The old +121% headline was a MIRAGE — the buggy Wayback N100 was missing large-cap winners (ADANIGREEN etc), so this 'mid/small' model wrongly held them. Correct N100 exclusion ⇒ genuinely mid/small ⇒ +46%.

**Open position at window end:** BHEL [mid] qty 16931 entry ₹377.05 on 2026-05-04 (unrealized +672,161)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
