# Great Eastern Shipping Co. Ltd. (GESHIP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1576.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 270.94
- **Avg P&L per closed trade:** 45.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 14:15:00 | 773.40 | 803.26 | 803.30 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 14:15:00 | 814.10 | 802.47 | 802.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 833.50 | 803.49 | 802.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 943.10 | 946.13 | 907.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-23 09:15:00 | 981.55 | 948.23 | 911.11 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-12 09:15:00 | 936.70 | 972.22 | 939.51 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 1208.75 | 1271.64 | 1271.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1190.70 | 1267.05 | 1269.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 1251.50 | 1250.05 | 1259.62 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 15:15:00 | 1314.00 | 1266.68 | 1266.58 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 1225.65 | 1266.19 | 1266.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 1196.85 | 1263.47 | 1264.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1261.65 | 1253.28 | 1259.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 09:15:00 | 1208.40 | 1263.01 | 1263.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 934.95 | 894.71 | 933.08 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 931.90 | 917.89 | 917.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 965.35 | 918.36 | 918.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 953.10 | 957.64 | 942.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 961.55 | 957.46 | 942.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 961.55 | 957.46 | 942.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-20 11:15:00 | 966.70 | 957.55 | 942.85 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 975.75 | 988.24 | 973.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 963.80 | 987.99 | 972.96 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 923.00 | 962.37 | 962.51 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 983.35 | 961.97 | 961.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 985.00 | 962.40 | 962.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 961.95 | 963.80 | 962.89 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 942.05 | 961.98 | 962.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 938.20 | 961.74 | 961.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 961.35 | 957.65 | 959.73 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 989.45 | 961.59 | 961.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 998.90 | 966.94 | 964.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 997.45 | 1000.59 | 984.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 09:15:00 | 1034.40 | 999.09 | 985.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1066.40 | 1090.08 | 1066.04 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-08 14:15:00 | 1072.10 | 1089.70 | 1066.08 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 1064.70 | 1089.26 | 1066.10 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-23 09:15:00 | 981.55 | 2024-02-12 09:15:00 | 936.70 | EXIT_EMA400 | -44.85 |
| SELL | 2024-11-08 09:15:00 | 1208.40 | 2024-11-21 09:15:00 | 1043.50 | TARGET | 164.90 |
| BUY | 2025-06-20 10:15:00 | 961.55 | 2025-07-03 09:15:00 | 1018.00 | TARGET | 56.45 |
| BUY | 2025-06-20 11:15:00 | 966.70 | 2025-07-04 12:15:00 | 1038.24 | TARGET | 71.54 |
| BUY | 2025-10-03 09:15:00 | 1034.40 | 2025-12-09 09:15:00 | 1064.70 | EXIT_EMA400 | 30.30 |
| BUY | 2025-12-08 14:15:00 | 1072.10 | 2025-12-09 09:15:00 | 1064.70 | EXIT_EMA400 | -7.40 |
