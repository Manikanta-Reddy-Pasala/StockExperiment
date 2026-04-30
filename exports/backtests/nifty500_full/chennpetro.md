# Chennai Petroleum Corporation Ltd. (CHENNPETRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1128.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 284.75
- **Avg P&L per closed trade:** 28.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 15:15:00 | 425.20 | 394.33 | 394.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 437.50 | 394.76 | 394.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 11:15:00 | 479.50 | 480.70 | 451.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-06 14:15:00 | 482.40 | 480.68 | 452.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 837.85 | 891.66 | 838.12 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 900.05 | 976.93 | 976.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 09:15:00 | 889.10 | 967.22 | 971.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 15:15:00 | 930.00 | 927.96 | 946.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-26 09:15:00 | 915.80 | 927.84 | 946.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-04 09:15:00 | 946.25 | 927.55 | 943.28 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 607.00 | 567.14 | 567.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 611.35 | 569.51 | 568.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 560.40 | 572.17 | 569.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 10:15:00 | 570.35 | 572.15 | 569.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 570.35 | 572.15 | 569.75 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 11:15:00 | 564.25 | 572.07 | 569.73 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 14:15:00 | 656.25 | 676.94 | 677.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 646.50 | 671.77 | 674.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 11:15:00 | 683.00 | 668.69 | 672.42 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 700.00 | 675.75 | 675.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 702.70 | 676.02 | 675.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 744.75 | 757.25 | 731.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-28 11:15:00 | 803.00 | 757.94 | 737.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 901.95 | 939.82 | 886.66 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-10 09:15:00 | 947.30 | 938.78 | 887.96 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 899.20 | 933.59 | 894.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-18 13:15:00 | 890.75 | 932.08 | 894.83 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 810.40 | 873.90 | 873.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 801.50 | 871.40 | 872.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 860.10 | 854.13 | 863.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 827.00 | 854.60 | 862.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 858.50 | 850.42 | 859.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-22 12:15:00 | 833.20 | 850.09 | 859.59 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-27 09:15:00 | 865.90 | 848.77 | 858.39 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 919.85 | 863.66 | 863.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 933.40 | 867.53 | 865.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 872.25 | 874.79 | 869.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 12:15:00 | 884.10 | 874.91 | 869.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 902.35 | 918.17 | 898.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-12 10:15:00 | 947.70 | 918.46 | 898.43 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 927.15 | 920.07 | 900.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 11:15:00 | 942.50 | 920.40 | 900.90 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-04-13 15:15:00 | 942.00 | 970.59 | 943.81 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-06 14:15:00 | 482.40 | 2023-10-17 09:15:00 | 572.70 | TARGET | 90.30 |
| SELL | 2024-09-26 09:15:00 | 915.80 | 2024-10-04 09:15:00 | 946.25 | EXIT_EMA400 | -30.45 |
| BUY | 2025-04-07 10:15:00 | 570.35 | 2025-04-07 11:15:00 | 564.25 | EXIT_EMA400 | -6.10 |
| BUY | 2025-10-28 11:15:00 | 803.00 | 2025-11-06 09:15:00 | 1000.54 | TARGET | 197.54 |
| BUY | 2025-12-10 09:15:00 | 947.30 | 2025-12-18 13:15:00 | 890.75 | EXIT_EMA400 | -56.55 |
| SELL | 2026-01-20 09:15:00 | 827.00 | 2026-01-27 09:15:00 | 865.90 | EXIT_EMA400 | -38.90 |
| SELL | 2026-01-22 12:15:00 | 833.20 | 2026-01-27 09:15:00 | 865.90 | EXIT_EMA400 | -32.70 |
| BUY | 2026-02-17 12:15:00 | 884.10 | 2026-02-25 09:15:00 | 926.60 | TARGET | 42.50 |
| BUY | 2026-03-16 11:15:00 | 942.50 | 2026-03-17 14:15:00 | 1067.30 | TARGET | 124.80 |
| BUY | 2026-03-12 10:15:00 | 947.70 | 2026-04-13 15:15:00 | 942.00 | EXIT_EMA400 | -5.70 |
