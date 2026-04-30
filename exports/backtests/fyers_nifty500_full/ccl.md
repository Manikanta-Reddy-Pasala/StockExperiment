# CCL Products (I) Ltd. (CCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1130.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 6 |
| ENTRY1 | 3 |
| ENTRY2 | 5 |
| EXIT | 3 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 39.97
- **Avg P&L per closed trade:** 5.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 631.05 | 678.22 | 678.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 628.50 | 676.86 | 677.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 668.70 | 664.43 | 670.70 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 15:15:00 | 718.00 | 675.60 | 675.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 741.15 | 684.53 | 680.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 751.30 | 754.89 | 730.17 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 635.25 | 720.36 | 720.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 627.00 | 703.87 | 711.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 659.05 | 657.69 | 680.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 12:15:00 | 647.80 | 660.21 | 680.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 612.80 | 657.31 | 676.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-14 10:15:00 | 581.95 | 646.28 | 669.11 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-11 10:15:00 | 608.20 | 575.42 | 599.52 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 773.00 | 612.53 | 611.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 13:15:00 | 819.25 | 673.45 | 646.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 803.70 | 803.76 | 747.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 816.65 | 801.57 | 752.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 849.25 | 863.10 | 830.47 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-08 15:15:00 | 870.00 | 862.04 | 833.02 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 880.20 | 894.24 | 871.94 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-19 10:15:00 | 883.60 | 893.41 | 872.18 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 886.45 | 893.03 | 872.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-22 12:15:00 | 897.10 | 893.02 | 872.91 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 878.80 | 895.38 | 876.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-26 14:15:00 | 893.25 | 895.36 | 877.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 877.50 | 895.04 | 877.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 10:15:00 | 867.95 | 894.77 | 876.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 837.85 | 865.64 | 865.68 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 988.15 | 864.26 | 863.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1012.45 | 870.19 | 866.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 970.40 | 971.04 | 934.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-03 09:15:00 | 996.60 | 971.84 | 936.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 12:15:00 | 647.80 | 2025-03-17 12:15:00 | 550.73 | TARGET | 97.07 |
| SELL | 2025-02-14 10:15:00 | 581.95 | 2025-04-11 10:15:00 | 608.20 | EXIT_EMA400 | -26.25 |
| BUY | 2025-09-19 10:15:00 | 883.60 | 2025-09-24 09:15:00 | 917.86 | TARGET | 34.26 |
| BUY | 2025-06-23 09:15:00 | 816.65 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | 51.30 |
| BUY | 2025-08-08 15:15:00 | 870.00 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | -2.05 |
| BUY | 2025-09-22 12:15:00 | 897.10 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | -29.15 |
| BUY | 2025-09-26 14:15:00 | 893.25 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | -25.30 |
| BUY | 2025-12-03 09:15:00 | 996.60 | 2025-12-05 11:15:00 | 936.70 | EXIT_EMA400 | -59.90 |
