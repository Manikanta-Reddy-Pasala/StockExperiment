# momentum_pseudo_n100_adv — SUMMARY

**Pseudo-N100 (top-100 by ADV from N500). Monthly rotation, yearly-PIT universe rebuild.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-04-01 |
| Total trades | 29 |
| Trades per year | ~9.7 |
| Rebalance period | Monthly |

## Headline result (CLEAN DATA, post yfinance restore)

| Metric | Value |
|---|---:|
| Final NAV | **₹7,330,949** |
| Total return | **+633.09%** |
| **3-yr CAGR** | **+113.86%/yr** |
| Max DD (NAV-based) | 36.44% |
| Calmar | 3.12 |
| Trades | 29 |
| WR | 79.3% (23W / 6L) |

## Change vs prior result (dirty data)

| Metric | Prior (dirty) | **Clean** | Δ |
|---|---:|---:|---:|
| CAGR | +136.39% | **+113.86%** | -22.53pp |
| Max DD | 16.15% | **36.44%** | +20.29pp |

Prior result was inflated/distorted by data anomalies in KOTAKBANK, MCX, NUVAMA, VEDL (all jumped 4-5x on 2024-12-23 in raw Fyers data — incremental pull bug). Re-fetched 8 affected stocks via yfinance (split-adjusted) on 2026-05-17.

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,650,158 | **+165.02%** | 9 |
| 2024-25 | ₹2,650,158 | ₹4,286,962 | **+61.76%** | 11 |
| 2025-26 | ₹4,286,962 | ₹7,330,949 | **+71.01%** | 9 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 13 | 12 | 1 | 92% | +4,613,886 |
| **Mid** | 13 | 9 | 4 | 69% | +1,896,138 |
| **Small** | 3 | 2 | 1 | 67% | -179,072 |
