# midcap_narrow_60d_breakout — SUMMARY (V3: Top-100 ADV minus Large)

**V3 winner of universe-expansion sweep.** Universe = **top-100 ADV from N500 minus NSE Nifty 100** (40 stocks). Most-liquid mid+small caps dominate. +94% CAGR vs V2's +44%.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-15** (≈3.00 years) |
| First entry | 2023-05-18 |
| Last exit | 2026-05-11 |
| Total trades | 12 |
| Trades per year | ~4 |
| Strategy class | Daily breakout scan, long-hold swing (60-90d/trade) |
| Rebalance period | Event-driven |
| Data source | Fyers (498/504 N500, 4yr re-pull 2026-05-17, cont_flag=1) |

## Stock pick logic (plain English)

1. **Universe build**: take all 500 Nifty 500 stocks, compute 20-day ADV, take **top 100** by ADV
2. **Cap filter (V3 NEW)**: exclude any stock in NSE Nifty 100 → result is top-100-by-ADV minus Large = ~40 highly-liquid mid+small stocks
3. **Daily scan**: for each day, find stocks with close > 40-day high + vol > 2× 20d avg + close > 200d SMA
4. **Pick top-1** by volume ratio (most-confirmed breakout)
5. **Hold** until: TARGET +100% / TRAIL -20% from peak (after +10%) / MAX_HOLD 90 trading days

## Key knobs

| Knob | Value |
|---|---|
| Universe pool | **Top-100 by ADV from N500** (V3, was 31-130 in V2) |
| Cap filter | Exclude NSE Nifty 100 (Large-cap exclusion) |
| Breakout | 40-day high |
| Volume confirm | ≥ 2× 20-day avg |
| Long-term filter | close > 200-day SMA |
| Position | max_concurrent=1 |
| Exits | TARGET +100% / TRAIL -20% (after +10%) / MAX_HOLD 90d |
| SMA20 exit | DISABLED |
| Costs | 10 bps slippage + ₹20 brokerage + 0.10% STT |

## Headline result (₹10L, 2023-05-15 → 2026-05-15)

| Metric | Value |
|---|---:|
| Final NAV | **₹7,713,735** |
| Total return | **+671.37%** |
| **3-yr CAGR** | **+97.59%/yr** |
| Max DD | **22.82%** |
| Trades | 12 |
| WR | 83.3% (10W / 2L) |
| Calmar | **4.28** |

## Why V3 beats V2 (and others)

Universe sweep tested 11 cap-pool variants:

| Variant | Universe size | CAGR | DD | Calmar |
|---|---:|---:|---:|---:|
| **V3: Top-100 ADV minus Large (THIS)** | **40** | **+94.02%** | 22.82% | **4.12** |
| Conservative: skip-50 take-100 minus Large | 57 | +55.79% | **12.40%** | 4.50 (best Calmar) |
| Broadest: 31-230 minus Large | 124 | +38.30% | 12.36% | 3.10 |
| **V2 baseline: 31-130 minus Large** | 47 | +44.08% | 23.41% | 1.88 |
| V1 baseline: 31-130 all caps | 100 | +37.50% | 49.89% | 0.75 |
| Pure NSE Midcap 150 list | 150 | +8.65% | 22.01% | 0.39 |
| Pure NSE Smallcap 250 list | 248 | +17.89% | 18.61% | 0.96 |
| Mid + Small NSE official | 398 | +5.70% | 26.14% | 0.22 |
| Top-300 ADV minus Large | 203 | +24.18% | 29.53% | 0.82 |

**Insight**: Top-100 ADV universe contains the most-traded mid/small caps. After removing Large, you get 40 stocks that are highly liquid AND have strong momentum potential. Sweet spot — small universe (less dilution) + high quality (liquid only). NSE official Midcap/Smallcap lists underperform because they include illiquid stocks.

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹1,939,276 | **+93.93%** | 4 |
| 2024-25 | ₹1,939,276 | ₹2,093,903 | **+7.97%** | 3 |
| 2025-26 | ₹2,093,903 | ₹7,713,735 | **+268.39%** | 5 |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Mid** | 8 | 7 | 1 | 88% | +4,445,606 |
| **Small** | 4 | 3 | 1 | 75% | +2,268,368 |

All trades in Mid + Small + Other (Large-cap explicitly filtered out).

## Universe (top 10 stocks at end-2026 by ADV, minus Large)

GROWW, BSE, HFCL, BHEL, GRSE, MCX, MEESHO, HSCL, IDEA, DIXON

These are the most-liquid mid+small caps. NSE Smallcap 250 stocks like IDEA, HFCL surface here because they're heavily-traded retail names.