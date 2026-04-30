# JBM Auto Ltd. (JBMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 630.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -251.03
- **Avg P&L per closed trade:** -31.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 625.00 | 670.54 | 670.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 616.50 | 668.57 | 669.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 10:15:00 | 622.00 | 619.18 | 635.88 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 700.78 | 641.62 | 641.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 10:15:00 | 736.33 | 657.27 | 649.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 11:15:00 | 1024.70 | 1031.00 | 946.27 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 09:15:00 | 869.03 | 923.51 | 923.72 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 10:15:00 | 979.25 | 920.79 | 920.77 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 906.00 | 921.81 | 921.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 904.30 | 921.63 | 921.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 936.35 | 921.42 | 921.65 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 960.03 | 921.93 | 921.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 973.40 | 923.91 | 922.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 959.53 | 969.43 | 948.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 1064.68 | 971.26 | 951.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-18 11:15:00 | 1020.00 | 1055.79 | 1022.17 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 925.58 | 1006.91 | 1007.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 909.05 | 1001.29 | 1004.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 14:15:00 | 976.83 | 976.48 | 988.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-29 09:15:00 | 970.53 | 976.42 | 988.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 974.67 | 971.01 | 983.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-06 09:15:00 | 956.05 | 971.09 | 982.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 964.45 | 967.61 | 979.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-11 15:15:00 | 957.80 | 967.52 | 979.47 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-12 09:15:00 | 1003.28 | 967.87 | 979.59 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 685.50 | 647.55 | 647.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 707.35 | 655.59 | 651.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 704.60 | 707.28 | 688.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 14:15:00 | 722.45 | 706.41 | 689.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 695.60 | 705.83 | 690.48 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 690.10 | 705.68 | 690.48 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 643.20 | 679.65 | 679.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 640.00 | 673.67 | 676.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 658.80 | 654.89 | 663.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-28 12:15:00 | 640.85 | 655.30 | 663.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 647.50 | 653.71 | 662.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-31 14:15:00 | 637.15 | 652.92 | 661.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-19 11:15:00 | 647.00 | 631.18 | 645.65 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 754.35 | 644.99 | 644.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 765.80 | 666.15 | 656.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 14:15:00 | 678.10 | 678.39 | 664.50 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 632.00 | 660.79 | 660.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 630.60 | 659.97 | 660.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 626.00 | 593.06 | 614.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 576.50 | 612.21 | 618.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 609.05 | 584.50 | 600.07 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 13:15:00 | 622.40 | 570.97 | 570.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 624.50 | 571.50 | 571.09 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-06 09:15:00 | 1064.68 | 2024-07-18 11:15:00 | 1020.00 | EXIT_EMA400 | -44.68 |
| SELL | 2024-08-29 09:15:00 | 970.53 | 2024-09-12 09:15:00 | 1003.28 | EXIT_EMA400 | -32.75 |
| SELL | 2024-09-06 09:15:00 | 956.05 | 2024-09-12 09:15:00 | 1003.28 | EXIT_EMA400 | -47.23 |
| SELL | 2024-09-11 15:15:00 | 957.80 | 2024-09-12 09:15:00 | 1003.28 | EXIT_EMA400 | -45.48 |
| BUY | 2025-06-16 14:15:00 | 722.45 | 2025-06-19 10:15:00 | 690.10 | EXIT_EMA400 | -32.35 |
| SELL | 2025-07-28 12:15:00 | 640.85 | 2025-08-19 11:15:00 | 647.00 | EXIT_EMA400 | -6.15 |
| SELL | 2025-07-31 14:15:00 | 637.15 | 2025-08-19 11:15:00 | 647.00 | EXIT_EMA400 | -9.85 |
| SELL | 2026-01-20 09:15:00 | 576.50 | 2026-02-04 09:15:00 | 609.05 | EXIT_EMA400 | -32.55 |
