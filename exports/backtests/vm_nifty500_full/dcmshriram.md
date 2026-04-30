# DCM Shriram Ltd. (DCMSHRIRAM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1227.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / EMA400 exits:** 6 / 4
- **Total realized P&L (per unit):** 388.09
- **Avg P&L per closed trade:** 38.81

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 11:15:00 | 887.35 | 971.05 | 971.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 14:15:00 | 883.05 | 968.56 | 969.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 957.40 | 930.04 | 946.28 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 12:15:00 | 975.60 | 955.16 | 955.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 986.45 | 956.06 | 955.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 1019.10 | 1026.40 | 1003.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 14:15:00 | 1034.00 | 1022.86 | 1004.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-23 10:15:00 | 1003.90 | 1022.66 | 1004.54 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 970.70 | 999.25 | 999.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 12:15:00 | 969.35 | 998.95 | 999.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 1016.00 | 994.65 | 996.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-29 10:15:00 | 968.25 | 993.59 | 996.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-04 13:15:00 | 946.70 | 918.77 | 945.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 991.95 | 949.38 | 949.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 12:15:00 | 1004.40 | 957.73 | 953.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 953.00 | 974.26 | 965.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-07 12:15:00 | 989.00 | 965.69 | 961.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 988.50 | 995.45 | 981.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-27 13:15:00 | 980.05 | 995.30 | 981.42 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 988.15 | 1072.06 | 1072.10 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 15:15:00 | 1275.00 | 1067.39 | 1066.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 09:15:00 | 1288.55 | 1069.59 | 1067.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 1154.00 | 1170.04 | 1127.05 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 1079.00 | 1113.72 | 1113.79 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 1147.65 | 1113.37 | 1113.33 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1073.70 | 1113.25 | 1113.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1070.95 | 1112.83 | 1113.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 1104.95 | 1088.35 | 1098.89 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1176.80 | 1106.54 | 1106.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 12:15:00 | 1193.45 | 1109.00 | 1107.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 1112.10 | 1113.83 | 1110.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-31 14:15:00 | 1173.50 | 1114.43 | 1110.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1173.50 | 1114.43 | 1110.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-03 10:15:00 | 1101.25 | 1114.80 | 1110.89 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 1064.40 | 1107.26 | 1107.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 1050.10 | 1103.69 | 1105.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 1032.35 | 1030.35 | 1060.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 12:15:00 | 1022.00 | 1030.27 | 1060.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1056.20 | 1031.26 | 1059.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 09:15:00 | 1032.80 | 1032.16 | 1058.51 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-20 13:15:00 | 1058.45 | 1024.60 | 1048.04 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 1074.00 | 1047.33 | 1047.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1081.10 | 1047.92 | 1047.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 1048.90 | 1049.79 | 1048.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-05 10:15:00 | 1063.60 | 1050.02 | 1048.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1082.20 | 1050.70 | 1049.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-10 10:15:00 | 1093.00 | 1053.90 | 1050.88 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1287.70 | 1348.31 | 1286.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-14 13:15:00 | 1278.90 | 1347.62 | 1286.21 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 1194.10 | 1263.11 | 1263.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 1181.60 | 1259.62 | 1261.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1210.50 | 1209.08 | 1229.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-16 09:15:00 | 1193.60 | 1208.92 | 1229.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1221.10 | 1208.86 | 1228.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-17 11:15:00 | 1240.10 | 1209.41 | 1228.62 | Close above EMA400 |

### Cycle 14 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 1276.10 | 1241.12 | 1241.06 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 1221.00 | 1241.47 | 1241.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 12:15:00 | 1211.60 | 1241.17 | 1241.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 1242.00 | 1229.31 | 1235.03 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1259.40 | 1233.87 | 1233.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1261.10 | 1236.90 | 1235.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 1237.20 | 1241.25 | 1237.89 | EMA200 retest candle locked |

### Cycle 17 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 1180.00 | 1234.67 | 1234.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 1162.80 | 1220.25 | 1227.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 1179.20 | 1178.48 | 1201.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 1147.30 | 1178.54 | 1200.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-25 10:15:00 | 1116.00 | 1061.00 | 1100.33 | Close above EMA400 |

### Cycle 18 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 1200.60 | 1119.68 | 1119.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 15:15:00 | 1202.20 | 1122.86 | 1121.26 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-19 14:15:00 | 1034.00 | 2024-01-23 10:15:00 | 1003.90 | EXIT_EMA400 | -30.10 |
| SELL | 2024-02-29 10:15:00 | 968.25 | 2024-03-13 11:15:00 | 884.54 | TARGET | 83.71 |
| BUY | 2024-06-07 12:15:00 | 989.00 | 2024-06-14 09:15:00 | 1071.13 | TARGET | 82.13 |
| BUY | 2025-01-31 14:15:00 | 1173.50 | 2025-02-03 10:15:00 | 1101.25 | EXIT_EMA400 | -72.25 |
| SELL | 2025-03-10 09:15:00 | 1032.80 | 2025-03-13 09:15:00 | 955.68 | TARGET | 77.12 |
| SELL | 2025-03-05 12:15:00 | 1022.00 | 2025-03-20 13:15:00 | 1058.45 | EXIT_EMA400 | -36.45 |
| BUY | 2025-06-05 10:15:00 | 1063.60 | 2025-06-11 09:15:00 | 1108.21 | TARGET | 44.61 |
| BUY | 2025-06-10 10:15:00 | 1093.00 | 2025-06-30 09:15:00 | 1219.36 | TARGET | 126.36 |
| SELL | 2025-10-16 09:15:00 | 1193.60 | 2025-10-17 11:15:00 | 1240.10 | EXIT_EMA400 | -46.50 |
| SELL | 2026-02-02 09:15:00 | 1147.30 | 2026-03-02 11:15:00 | 987.84 | TARGET | 159.46 |
