# midcap_narrow_60d_breakout (V2: Mid+Small) — SUMMARY

**Mid+Small cap breakout swing. Exclude Large from pseudo-midcap pool. V2 winner of cap-filter sweep.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-17 |
| Last exit | 2026-03-11 |
| Total trades | 11 |
| Trades per year | ~3.7 |
| Rebalance period | Event-driven (daily breakout scan) |

## Headline result (CLEAN DATA, post yfinance restore)

| Metric | Value |
|---|---:|
| Final NAV | **₹3,252,696** |
| Total return | **+225.27%** |
| **3-yr CAGR** | **+48.17%/yr** |
| Max DD (NAV-based) | 15.15% |
| Calmar | 3.18 |
| Trades | 11 |
| WR | 63.6% (7W / 4L) |

## Change vs prior result (dirty data)

| Metric | Prior (dirty) | **Clean** | Δ |
|---|---:|---:|---:|
| CAGR | +86.63% | **+48.17%** | -38.46pp |
| Max DD | 15.15% | **15.15%** | +0.00pp |

Prior result was inflated/distorted by data anomalies in KOTAKBANK, MCX, NUVAMA, VEDL (all jumped 4-5x on 2024-12-23 in raw Fyers data — incremental pull bug). Re-fetched 8 affected stocks via yfinance (split-adjusted) on 2026-05-17.

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹1,909,855 | **+90.99%** | 3 |
| 2024-25 | ₹1,909,855 | ₹2,487,636 | **+30.25%** | 4 |
| 2025-26 | ₹2,487,636 | ₹3,252,696 | **+30.75%** | 4 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Mid** | 3 | 1 | 2 | 33% | +9,593 |
| **Small** | 8 | 6 | 2 | 75% | +2,243,323 |
