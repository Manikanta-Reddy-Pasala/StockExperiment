# Axis Bank Ltd. (AXISBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1268.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 11 |
| ENTRY1 | 6 |
| ENTRY2 | 8 |
| EXIT | 6 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / EMA400 exits:** 7 / 7
- **Total realized P&L (per unit):** 467.52
- **Avg P&L per closed trade:** 33.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 991.45 | 953.18 | 953.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 994.50 | 960.72 | 957.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 13:15:00 | 966.65 | 967.40 | 961.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-07 14:15:00 | 977.45 | 967.50 | 961.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1001.65 | 1004.46 | 991.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-16 10:15:00 | 1004.20 | 1004.45 | 991.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 994.45 | 1004.47 | 992.34 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-10-19 09:15:00 | 989.55 | 1004.22 | 992.34 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 13:15:00 | 1031.90 | 1075.66 | 1075.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 1026.10 | 1065.63 | 1068.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.77 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 10:15:00 | 1142.50 | 1070.75 | 1070.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 11:15:00 | 1150.95 | 1071.55 | 1070.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 13:15:00 | 1171.60 | 1141.28 | 1118.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-25 09:15:00 | 1161.45 | 1259.01 | 1218.65 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1162.95 | 1194.48 | 1194.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1158.95 | 1194.12 | 1194.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 1182.05 | 1181.74 | 1187.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-28 09:15:00 | 1174.50 | 1181.65 | 1186.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1183.00 | 1180.44 | 1185.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-02 13:15:00 | 1191.10 | 1180.57 | 1185.65 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1235.45 | 1187.97 | 1187.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 11:15:00 | 1242.40 | 1190.30 | 1188.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 1158.00 | 1196.57 | 1196.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1150.60 | 1195.33 | 1196.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 12:15:00 | 1181.00 | 1189.91 | 1193.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1181.60 | 1186.55 | 1190.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-25 11:15:00 | 1171.60 | 1186.32 | 1190.82 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 1190.45 | 1186.14 | 1190.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-28 14:15:00 | 1172.50 | 1185.72 | 1190.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1185.65 | 1185.13 | 1189.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-30 09:15:00 | 1173.05 | 1185.02 | 1189.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1181.10 | 1175.37 | 1183.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-12 12:15:00 | 1161.40 | 1175.19 | 1182.84 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1164.00 | 1155.12 | 1167.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-04 13:15:00 | 1156.10 | 1155.25 | 1167.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1167.05 | 1155.47 | 1167.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 1168.60 | 1155.60 | 1167.11 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 1103.55 | 1047.95 | 1046.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 10:15:00 | 1072.15 | 1057.34 | 1052.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 09:15:00 | 1174.60 | 1205.86 | 1180.86 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1170.02 | 1170.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.60 | 1169.30 | 1169.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.30 | 1097.22 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1129.80 | 1110.78 | 1110.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 1133.70 | 1111.00 | 1110.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.00 | 1261.01 | 1230.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-01 10:15:00 | 1274.60 | 1247.92 | 1232.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1258.00 | 1272.96 | 1254.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-27 09:15:00 | 1322.10 | 1273.44 | 1254.95 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1321.40 | 1347.87 | 1318.32 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-06 11:15:00 | 1327.00 | 1347.67 | 1318.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-06 14:15:00 | 1316.10 | 1346.94 | 1318.43 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1210.80 | 1298.57 | 1298.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1204.20 | 1297.63 | 1298.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.40 | 1269.51 | EMA200 retest candle locked |

### Cycle 11 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 1356.50 | 1285.46 | 1285.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 1357.90 | 1286.18 | 1285.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1310.05 | 1299.05 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-07 14:15:00 | 977.45 | 2023-09-15 14:15:00 | 1024.56 | TARGET | 47.11 |
| BUY | 2023-10-16 10:15:00 | 1004.20 | 2023-10-19 09:15:00 | 989.55 | EXIT_EMA400 | -14.65 |
| BUY | 2024-06-05 13:15:00 | 1171.60 | 2024-07-12 10:15:00 | 1330.23 | TARGET | 158.63 |
| SELL | 2024-08-28 09:15:00 | 1174.50 | 2024-09-02 13:15:00 | 1191.10 | EXIT_EMA400 | -16.60 |
| SELL | 2024-10-22 12:15:00 | 1181.00 | 2024-11-04 11:15:00 | 1145.00 | TARGET | 36.00 |
| SELL | 2024-10-30 09:15:00 | 1173.05 | 2024-11-18 10:15:00 | 1123.00 | TARGET | 50.05 |
| SELL | 2024-10-28 14:15:00 | 1172.50 | 2024-11-21 09:15:00 | 1119.12 | TARGET | 53.38 |
| SELL | 2024-10-25 11:15:00 | 1171.60 | 2024-12-05 12:15:00 | 1168.60 | EXIT_EMA400 | 3.00 |
| SELL | 2024-11-12 12:15:00 | 1161.40 | 2024-12-05 12:15:00 | 1168.60 | EXIT_EMA400 | -7.20 |
| SELL | 2024-12-04 13:15:00 | 1156.10 | 2024-12-05 12:15:00 | 1168.60 | EXIT_EMA400 | -12.50 |
| BUY | 2025-04-08 10:15:00 | 1072.15 | 2025-04-16 09:15:00 | 1132.12 | TARGET | 59.97 |
| BUY | 2026-01-01 10:15:00 | 1274.60 | 2026-02-23 10:15:00 | 1401.83 | TARGET | 127.23 |
| BUY | 2026-01-27 09:15:00 | 1322.10 | 2026-03-06 14:15:00 | 1316.10 | EXIT_EMA400 | -6.00 |
| BUY | 2026-03-06 11:15:00 | 1327.00 | 2026-03-06 14:15:00 | 1316.10 | EXIT_EMA400 | -10.90 |
