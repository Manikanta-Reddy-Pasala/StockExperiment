# State Bank of India (SBIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1071.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -40.45
- **Avg P&L per closed trade:** -6.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 09:15:00 | 807.85 | 837.78 | 837.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 797.65 | 824.29 | 829.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 804.65 | 803.71 | 815.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-23 15:15:00 | 800.05 | 803.66 | 815.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 806.20 | 798.37 | 807.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-14 11:15:00 | 803.80 | 798.43 | 807.54 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 803.50 | 798.67 | 807.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-17 09:15:00 | 811.00 | 799.60 | 807.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 15:15:00 | 855.45 | 810.13 | 810.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 860.45 | 811.89 | 810.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 10:15:00 | 814.95 | 818.93 | 814.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-25 09:15:00 | 844.00 | 814.14 | 813.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-19 09:15:00 | 826.10 | 841.58 | 831.19 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 13:15:00 | 799.15 | 824.44 | 824.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 793.50 | 822.50 | 823.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 10:15:00 | 775.90 | 774.39 | 790.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 11:15:00 | 762.85 | 774.28 | 790.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 746.85 | 732.08 | 748.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-19 14:15:00 | 744.75 | 732.20 | 748.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 744.90 | 732.45 | 748.81 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 14:15:00 | 749.35 | 733.19 | 748.78 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.64 | 757.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.28 | 757.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.35 | 781.81 | 771.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 10:15:00 | 799.65 | 780.74 | 772.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 793.30 | 800.71 | 790.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 786.35 | 800.06 | 790.24 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 1017.35 | 1070.19 | 1070.41 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1107.60 | 1069.70 | 1069.65 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-23 15:15:00 | 800.05 | 2024-10-17 09:15:00 | 811.00 | EXIT_EMA400 | -10.95 |
| SELL | 2024-10-14 11:15:00 | 803.80 | 2024-10-17 09:15:00 | 811.00 | EXIT_EMA400 | -7.20 |
| BUY | 2024-11-25 09:15:00 | 844.00 | 2024-12-19 09:15:00 | 826.10 | EXIT_EMA400 | -17.90 |
| SELL | 2025-02-01 11:15:00 | 762.85 | 2025-03-20 14:15:00 | 749.35 | EXIT_EMA400 | 13.50 |
| SELL | 2025-03-19 14:15:00 | 744.75 | 2025-03-20 14:15:00 | 749.35 | EXIT_EMA400 | -4.60 |
| BUY | 2025-05-12 10:15:00 | 799.65 | 2025-06-16 09:15:00 | 786.35 | EXIT_EMA400 | -13.30 |
