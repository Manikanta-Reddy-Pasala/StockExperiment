# momentum_n100_top5_max1 — SUMMARY

**LIVE production model.** Monthly momentum rotation on REAL NSE Nifty 100.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (≈3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-05-04 |
| Total trades | 31 |
| Trades per year | ~10.3 |
| Strategy class | Monthly rotation (1st trading day) |
| Rebalance period | Monthly |

## Stock pick logic (plain English)

1. **Universe**: take all 104 stocks in official NSE Nifty 100 list (`src/data/symbols/nifty100.csv`)
2. **Rank by return**: for each stock, compute 30-day return = `close_today / close_30_days_ago - 1`, sort highest-first
3. **Pick top-1**: hold the single best-performing stock for the next month (100% of capital)
4. **Rebalance**: 1st trading day of each month, re-rank and rotate to new top-1 if changed

**Unique filter**: real NSE constituent list only. No ADV filter, no mcap filter — NSE already curates the top-100 by free-float market cap. This is the most conservative universe of all 5 models.

| Key knob | Value |
|---|---|
| Universe size | 104 stocks (Nifty 100 + EQ alternates) |
| Lookback | 30 days |
| Position | top-1 (max_concurrent=1) |
| **Rebalance period** | **Monthly (1st trading day)** |
| Exit rule | Rotation only (sell when not rank-1 next month) |
| SL / Target | None |

## Headline result (3-year backtest, ₹10L start)

| Metric | Value |
|---|---:|
| Final NAV | **₹5,868,846** |
| Total return | **+486.88%** |
| **3-yr CAGR** | **+80.38%/yr** |
| Max DD (cash NAV) | 29.71% |
| Trades | 31 (+1 open) |
| WR | 74.2% (23W / 8L) |

## Returns by NSE cap segment

| Cap segment | Trades | Wins | Losses | WR | Total PnL ₹ | Avg PnL/trade ₹ |
|---|---:|---:|---:|---:|---:|---:|
| **Large** | 31 | 23 | 8 | 74% | +4,791,235 | +154,556 |

## Full trade ledger — every entry with price, invested ₹, exit, gain/loss

| # | Symbol | Cap | Index | Entry Date | Entry ₹ | Qty | **Invested** | Exit Date | Exit ₹ | **PnL ₹** | Ret % | Reason |
|--:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | CHOLAFIN | **Large** | Nifty 100 | 2023-05-15 | 1,003.50 | 996 | ₹999,486 | 2023-06-01 | 1,043.90 | +40,238 | +4.03% | rotate |
| 2 | ADANIPOWER | **Large** | Nifty 100 | 2023-06-01 | 50.83 | 20,465 | ₹1,040,236 | 2023-07-03 | 49.69 | -23,330 | -2.24% | rotate |
| 3 | MAZDOCK | **Large** | Nifty 100 | 2023-07-03 | 644.55 | 1,577 | ₹1,016,455 | 2023-09-01 | 943.53 | +471,491 | +46.39% | rotate |
| 4 | IRFC | **Large** | Nifty 100 | 2023-09-01 | 55.75 | 26,697 | ₹1,488,358 | 2023-11-01 | 72.95 | +459,188 | +30.85% | rotate |
| 5 | SOLARINDS | **Large** | Nifty 100 | 2023-11-01 | 5,521.35 | 352 | ₹1,943,515 | 2023-12-01 | 6,188.75 | +234,925 | +12.09% | rotate |
| 6 | PFC | **Large** | Nifty 100 | 2023-12-01 | 365.15 | 5,977 | ₹2,182,502 | 2024-01-01 | 395.05 | +178,712 | +8.19% | rotate |
| 7 | ADANIGREEN | **Large** | Nifty 100 | 2024-01-01 | 1,598.40 | 1,477 | ₹2,360,837 | 2024-02-01 | 1,665.95 | +99,771 | +4.23% | rotate |
| 8 | IRFC | **Large** | Nifty 100 | 2024-02-01 | 169.90 | 14,484 | ₹2,460,832 | 2024-03-01 | 147.40 | -325,890 | -13.24% | rotate |
| 9 | CUMMINSIND | **Large** | Nifty 100 | 2024-03-01 | 2,726.15 | 783 | ₹2,134,575 | 2024-04-01 | 3,003.40 | +217,087 | +10.17% | rotate |
| 10 | ABB | **Large** | Nifty 100 | 2024-04-01 | 6,504.65 | 361 | ₹2,348,179 | 2024-05-02 | 6,682.50 | +64,204 | +2.73% | rotate |
| 11 | VEDL | **Large** | Nifty 100 | 2024-05-02 | 153.86 | 15,705 | ₹2,416,371 | 2024-06-03 | 171.48 | +276,722 | +11.45% | rotate |
| 12 | HINDZINC | **Large** | Nifty 100 | 2024-06-03 | 696.40 | 3,867 | ₹2,692,979 | 2024-07-01 | 656.75 | -153,327 | -5.69% | rotate |
| 13 | MAZDOCK | **Large** | Nifty 100 | 2024-07-01 | 2,196.95 | 1,156 | ₹2,539,674 | 2024-09-02 | 2,099.90 | -112,190 | -4.42% | rotate |
| 14 | TRENT | **Large** | Nifty 100 | 2024-09-02 | 7,148.20 | 339 | ₹2,423,240 | 2024-10-01 | 7,612.70 | +157,466 | +6.50% | rotate |
| 15 | BAJAJ-AUTO | **Large** | Nifty 100 | 2024-10-01 | 12,157.45 | 212 | ₹2,577,379 | 2024-11-01 | 9,875.95 | -483,678 | -18.77% | rotate |
| 16 | HINDZINC | **Large** | Nifty 100 | 2024-11-01 | 558.25 | 3,764 | ₹2,101,253 | 2024-12-02 | 502.85 | -208,526 | -9.92% | rotate |
| 17 | INDHOTEL | **Large** | Nifty 100 | 2024-12-02 | 801.05 | 2,362 | ₹1,892,080 | 2025-01-01 | 873.60 | +171,363 | +9.06% | rotate |
| 18 | KOTAKBANK | **Large** | Nifty 100 | 2025-01-01 | 1,788.40 | 1,154 | ₹2,063,814 | 2025-03-03 | 1,914.60 | +145,635 | +7.06% | rotate |
| 19 | SHRIRAMFIN | **Large** | Nifty 100 | 2025-03-03 | 621.30 | 3,556 | ₹2,209,343 | 2025-04-01 | 637.45 | +57,429 | +2.60% | rotate |
| 20 | SOLARINDS | **Large** | Nifty 100 | 2025-04-01 | 11,131.60 | 203 | ₹2,259,715 | 2025-05-02 | 13,049.00 | +389,232 | +17.22% | rotate |
| 21 | HAL | **Large** | Nifty 100 | 2025-05-02 | 4,492.40 | 591 | ₹2,655,008 | 2025-06-02 | 5,017.10 | +310,098 | +11.68% | rotate |
| 22 | SOLARINDS | **Large** | Nifty 100 | 2025-06-02 | 16,294.00 | 182 | ₹2,965,508 | 2025-07-01 | 17,196.00 | +164,164 | +5.54% | rotate |
| 23 | MUTHOOTFIN | **Large** | Nifty 100 | 2025-07-01 | 2,641.90 | 1,185 | ₹3,130,652 | 2025-08-01 | 2,592.70 | -58,302 | -1.86% | rotate |
| 24 | BOSCHLTD | **Large** | Nifty 100 | 2025-08-01 | 40,390.00 | 76 | ₹3,069,640 | 2025-09-01 | 40,785.00 | +30,020 | +0.98% | rotate |
| 25 | ETERNAL | **Large** | Nifty 100 | 2025-09-01 | 321.10 | 9,662 | ₹3,102,468 | 2025-10-01 | 329.00 | +76,330 | +2.46% | rotate |
| 26 | ADANIPOWER | **Large** | Nifty 100 | 2025-10-01 | 152.51 | 20,843 | ₹3,178,766 | 2025-11-03 | 156.73 | +87,957 | +2.77% | rotate |
| 27 | SHRIRAMFIN | **Large** | Nifty 100 | 2025-11-03 | 796.45 | 4,101 | ₹3,266,241 | 2026-01-01 | 1,019.70 | +915,548 | +28.03% | rotate |
| 28 | TMCV | **Large** | Nifty 100 | 2026-01-01 | 427.75 | 9,777 | ₹4,182,112 | 2026-02-01 | 441.30 | +132,478 | +3.17% | rotate |
| 29 | SHRIRAMFIN | **Large** | Nifty 100 | 2026-02-01 | 997.60 | 4,325 | ₹4,314,620 | 2026-03-02 | 1,052.50 | +237,442 | +5.50% | rotate |
| 30 | ENRIN | **Large** | Nifty 100 | 2026-03-02 | 2,972.70 | 1,531 | ₹4,551,204 | 2026-04-01 | 2,613.90 | -549,323 | -12.07% | rotate |
| 31 | ADANIPOWER | **Large** | Nifty 100 | 2026-04-01 | 157.11 | 25,478 | ₹4,002,849 | 2026-05-04 | 227.30 | +1,788,301 | +44.68% | rotate |
| OPEN | ADANIGREEN | **Large** | Nifty 100 | 2026-05-04 | 1,290.70 | 4,486 | ₹5,790,080 | OPEN | 1,308.00 | +77,608 | +1.34% | open |

**Caveats**:
- Max DD 30% — single-stock concentration with monthly rotation
- Y2 chop: BAJAJ-AUTO -19%, HINDZINC -10%, MAZDOCK -4% consecutive losers
- NSE cap classification uses **current** Nifty index lists. Stock cap may have shifted across 2023-2026.
