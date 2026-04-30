# State Bank of India (SBIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1068.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -33.73
- **Avg P&L per closed trade:** -4.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 13:15:00 | 604.75 | 583.16 | 583.07 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 12:15:00 | 571.05 | 585.46 | 585.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 567.85 | 584.88 | 585.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 576.90 | 574.43 | 578.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-17 09:15:00 | 566.85 | 577.01 | 579.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-04 09:15:00 | 588.90 | 571.03 | 574.93 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 13:15:00 | 610.65 | 578.59 | 578.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 614.05 | 581.05 | 579.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 624.90 | 625.82 | 610.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-12 13:15:00 | 632.15 | 625.51 | 611.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-23 11:15:00 | 613.10 | 627.48 | 615.28 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 815.30 | 829.61 | 829.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 810.35 | 828.06 | 828.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 804.65 | 803.65 | 813.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-23 15:15:00 | 800.60 | 803.62 | 813.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 797.50 | 800.86 | 809.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-04 13:15:00 | 796.80 | 800.82 | 809.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 805.80 | 798.28 | 806.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-16 09:15:00 | 807.20 | 799.11 | 806.28 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 856.30 | 808.89 | 808.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 860.25 | 811.57 | 810.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 10:15:00 | 814.95 | 818.72 | 814.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-25 09:15:00 | 844.00 | 814.01 | 812.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-19 09:15:00 | 826.10 | 841.57 | 830.92 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 801.25 | 824.16 | 824.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 795.75 | 822.75 | 823.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 10:15:00 | 774.80 | 773.35 | 789.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 11:15:00 | 768.90 | 773.54 | 789.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 746.90 | 732.12 | 749.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-19 14:15:00 | 744.85 | 732.24 | 749.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 745.15 | 732.50 | 749.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 14:15:00 | 749.50 | 733.24 | 749.01 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.67 | 757.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.32 | 757.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.10 | 781.81 | 772.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 10:15:00 | 799.95 | 780.75 | 772.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 793.30 | 800.73 | 790.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 786.35 | 800.07 | 790.28 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 1020.45 | 1069.63 | 1069.75 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1111.00 | 1069.30 | 1069.16 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-17 09:15:00 | 566.85 | 2023-12-04 09:15:00 | 588.90 | EXIT_EMA400 | -22.05 |
| BUY | 2024-01-12 13:15:00 | 632.15 | 2024-01-23 11:15:00 | 613.10 | EXIT_EMA400 | -19.05 |
| SELL | 2024-09-23 15:15:00 | 800.60 | 2024-10-16 09:15:00 | 807.20 | EXIT_EMA400 | -6.60 |
| SELL | 2024-10-04 13:15:00 | 796.80 | 2024-10-16 09:15:00 | 807.20 | EXIT_EMA400 | -10.40 |
| BUY | 2024-11-25 09:15:00 | 844.00 | 2024-12-19 09:15:00 | 826.10 | EXIT_EMA400 | -17.90 |
| SELL | 2025-02-05 11:15:00 | 768.90 | 2025-02-27 09:15:00 | 708.38 | TARGET | 60.52 |
| SELL | 2025-03-19 14:15:00 | 744.85 | 2025-03-20 14:15:00 | 749.50 | EXIT_EMA400 | -4.65 |
| BUY | 2025-05-12 10:15:00 | 799.95 | 2025-06-16 09:15:00 | 786.35 | EXIT_EMA400 | -13.60 |
