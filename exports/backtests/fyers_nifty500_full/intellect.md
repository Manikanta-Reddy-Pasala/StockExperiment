# Intellect Design Arena Ltd. (INTELLECT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 745.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -417.35
- **Avg P&L per closed trade:** -52.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 963.10 | 980.21 | 980.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 13:15:00 | 959.40 | 980.00 | 980.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 10:15:00 | 981.90 | 979.48 | 979.84 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 999.95 | 980.20 | 980.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 10:15:00 | 1004.35 | 981.79 | 981.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 983.00 | 983.17 | 981.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-06 09:15:00 | 1011.05 | 984.51 | 982.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 988.20 | 985.33 | 983.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-09 13:15:00 | 983.20 | 985.35 | 983.31 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 898.30 | 983.77 | 984.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 888.95 | 982.82 | 983.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 764.45 | 762.71 | 819.79 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 999.95 | 837.46 | 837.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 14:15:00 | 1019.35 | 840.95 | 838.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 890.90 | 895.52 | 875.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-20 10:15:00 | 920.40 | 895.60 | 875.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 882.85 | 897.15 | 877.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-22 10:15:00 | 876.15 | 896.94 | 877.91 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 798.55 | 866.85 | 866.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 797.90 | 855.24 | 860.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 12:15:00 | 706.85 | 706.05 | 752.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-28 13:15:00 | 699.50 | 710.31 | 747.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 719.35 | 686.89 | 724.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 13:15:00 | 727.85 | 687.63 | 724.51 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 12:15:00 | 795.60 | 746.72 | 746.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 09:15:00 | 799.60 | 748.35 | 747.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1078.30 | 1106.49 | 1005.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 14:15:00 | 1116.80 | 1101.63 | 1012.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1091.70 | 1149.12 | 1091.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 13:15:00 | 1019.40 | 1147.83 | 1090.70 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 933.50 | 1055.30 | 1055.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 926.90 | 1049.63 | 1052.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 993.40 | 980.32 | 1006.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-14 12:15:00 | 953.55 | 996.71 | 1005.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 997.05 | 995.53 | 1005.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-17 12:15:00 | 957.70 | 992.67 | 1002.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 998.10 | 986.14 | 998.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 1026.15 | 986.64 | 998.33 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1206.40 | 1007.66 | 1007.62 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 1007.50 | 1050.16 | 1050.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 1003.00 | 1049.25 | 1049.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 682.45 | 676.55 | 741.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 12:15:00 | 673.75 | 676.96 | 738.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 707.05 | 676.17 | 730.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 11:15:00 | 697.90 | 676.67 | 730.59 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-28 09:15:00 | 725.05 | 683.74 | 721.74 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-06 09:15:00 | 1011.05 | 2024-09-09 13:15:00 | 983.20 | EXIT_EMA400 | -27.85 |
| BUY | 2025-01-20 10:15:00 | 920.40 | 2025-01-22 10:15:00 | 876.15 | EXIT_EMA400 | -44.25 |
| SELL | 2025-03-28 13:15:00 | 699.50 | 2025-04-15 13:15:00 | 727.85 | EXIT_EMA400 | -28.35 |
| BUY | 2025-06-25 14:15:00 | 1116.80 | 2025-07-25 13:15:00 | 1019.40 | EXIT_EMA400 | -97.40 |
| SELL | 2025-10-14 12:15:00 | 953.55 | 2025-10-27 09:15:00 | 1026.15 | EXIT_EMA400 | -72.60 |
| SELL | 2025-10-17 12:15:00 | 957.70 | 2025-10-27 09:15:00 | 1026.15 | EXIT_EMA400 | -68.45 |
| SELL | 2026-04-09 12:15:00 | 673.75 | 2026-04-28 09:15:00 | 725.05 | EXIT_EMA400 | -51.30 |
| SELL | 2026-04-16 11:15:00 | 697.90 | 2026-04-28 09:15:00 | 725.05 | EXIT_EMA400 | -27.15 |
