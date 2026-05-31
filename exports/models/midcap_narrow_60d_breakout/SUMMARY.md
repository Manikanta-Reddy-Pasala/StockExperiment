# Midcap Breakout (`midcap_narrow_60d_breakout`)

**Status:** LIVE  
Event-driven single-position breakout: 40d-high + 2× vol + >200DMA. Target +100% / stop −20% / trail −20% off peak / 120d max-hold.

**Universe:** PIT midcap — top-100 ADV from N500 minus Nifty 100 (excluded at SCAN time)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | Breakouts scanned DAILY; the eligible band is rebuilt each year-start. |
| **Universe & filters** | Top-100 by 20d ADV from (PIT N500 minus PIT N100, excluded at scan time); close > 200d SMA. |
| **Entry** | BUY (next day's open) on a breakout: close > prior-40-day high AND volume ≥ 2× the 20d average volume. Single position (max 1); the highest volume-ratio breakout wins. |
| **Exit** | Event exits (whichever first): target +100%, hard stop −20%, trailing stop −20% off the peak, or 120-day max hold. |
| **Source** | Backtest+live: PIT `n500_membership.csv` MINUS `n100_membership.csv` (factsheet-derived). Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹1,746,287 |
| Total return | +74.6% |
| CAGR (annualized) | +11.2% |
| Max drawdown | 57.2% |
| Calmar | 0.2 |
| Trades | 13 (8W / 5L) · 62% win |

## Note

⚠️ Lumpy single-position event model (only ~16 trades/5yr). On AUTHORITATIVE PIT membership (2026-05-31) the full-cycle 2021-03→2026-05 is ≈ +11.2% CAGR / 57% DD / Calmar 0.2 (only 13 trades/5yr) — effectively DEAD. Its earlier +40% was living off large-cap winners that leaked through the buggy Wayback N100 exclusion; with the correct PIT N100 removed it has no edge. Confirms the long-standing 'midcap ignore' call.

**Open position at window end:** AMBER qty 229 entry ₹7448.44 on 2026-02-11 (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
