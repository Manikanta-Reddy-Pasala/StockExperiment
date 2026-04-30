# Jubilant Pharmova Ltd. (JUBLPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 930.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -323.70
- **Avg P&L per closed trade:** -40.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1049.05 | 1114.74 | 1115.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1040.75 | 1114.01 | 1114.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 989.80 | 989.16 | 1032.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 942.45 | 988.54 | 1031.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 10:15:00 | 1031.00 | 984.36 | 1026.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1065.85 | 931.31 | 930.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1120.95 | 937.35 | 933.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 15:15:00 | 1170.00 | 1171.63 | 1122.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-25 14:15:00 | 1181.50 | 1171.64 | 1124.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1150.90 | 1172.75 | 1127.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-30 12:15:00 | 1227.10 | 1174.09 | 1129.57 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-04 09:15:00 | 1123.30 | 1176.04 | 1134.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1054.80 | 1112.91 | 1112.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1044.20 | 1112.22 | 1112.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1133.00 | 1086.52 | 1097.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 13:15:00 | 1069.40 | 1096.32 | 1100.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-01 13:15:00 | 1102.70 | 1089.19 | 1095.88 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1120.00 | 1098.52 | 1098.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1131.00 | 1099.82 | 1099.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1095.50 | 1102.55 | 1100.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 10:15:00 | 1142.50 | 1102.89 | 1100.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1122.20 | 1114.44 | 1107.23 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-10 11:15:00 | 1137.80 | 1114.93 | 1107.59 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1112.00 | 1116.37 | 1108.72 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-12 10:15:00 | 1124.50 | 1116.49 | 1108.86 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-19 14:15:00 | 1107.90 | 1121.89 | 1113.21 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 1080.70 | 1106.38 | 1106.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 1066.20 | 1104.27 | 1105.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 1092.30 | 1092.26 | 1098.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-12 09:15:00 | 1084.80 | 1092.20 | 1098.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 14:15:00 | 1095.00 | 1084.24 | 1093.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-03 09:15:00 | 942.45 | 2025-02-05 10:15:00 | 1031.00 | EXIT_EMA400 | -88.55 |
| BUY | 2025-07-25 14:15:00 | 1181.50 | 2025-08-04 09:15:00 | 1123.30 | EXIT_EMA400 | -58.20 |
| BUY | 2025-07-30 12:15:00 | 1227.10 | 2025-08-04 09:15:00 | 1123.30 | EXIT_EMA400 | -103.80 |
| SELL | 2025-09-25 13:15:00 | 1069.40 | 2025-10-01 13:15:00 | 1102.70 | EXIT_EMA400 | -33.30 |
| BUY | 2025-11-03 10:15:00 | 1142.50 | 2025-11-19 14:15:00 | 1107.90 | EXIT_EMA400 | -34.60 |
| BUY | 2025-11-10 11:15:00 | 1137.80 | 2025-11-19 14:15:00 | 1107.90 | EXIT_EMA400 | -29.90 |
| BUY | 2025-11-12 10:15:00 | 1124.50 | 2025-11-19 14:15:00 | 1107.90 | EXIT_EMA400 | -16.60 |
| SELL | 2025-12-12 09:15:00 | 1084.80 | 2025-12-17 13:15:00 | 1043.55 | TARGET | 41.25 |
