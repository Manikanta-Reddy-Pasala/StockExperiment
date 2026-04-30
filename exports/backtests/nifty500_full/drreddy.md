# Dr. Reddy's Laboratories Ltd. (DRREDDY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1322.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 13 |
| ALERT2 | 12 |
| ALERT3 | 7 |
| ENTRY1 | 9 |
| ENTRY2 | 2 |
| EXIT | 9 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / EMA400 exits:** 3 / 8
- **Total realized P&L (per unit):** 60.94
- **Avg P&L per closed trade:** 5.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 1056.20 | 1098.21 | 1098.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 13:15:00 | 1054.41 | 1097.77 | 1098.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 1097.78 | 1094.07 | 1096.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-10 09:15:00 | 1081.99 | 1093.99 | 1095.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 1093.19 | 1093.46 | 1095.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-13 10:15:00 | 1083.32 | 1093.35 | 1095.56 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-16 11:15:00 | 1097.25 | 1092.38 | 1094.90 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 10:15:00 | 1134.40 | 1097.44 | 1097.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 1141.00 | 1099.49 | 1098.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 09:15:00 | 1093.71 | 1126.37 | 1114.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-12 09:15:00 | 1124.00 | 1124.37 | 1114.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1124.00 | 1124.37 | 1114.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-12-12 10:15:00 | 1110.02 | 1124.22 | 1114.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 1173.98 | 1221.62 | 1221.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 1165.23 | 1218.03 | 1219.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 1199.84 | 1198.51 | 1208.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-30 09:15:00 | 1183.20 | 1198.37 | 1207.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-07 11:15:00 | 1204.40 | 1187.26 | 1199.74 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 1279.00 | 1206.13 | 1205.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1284.52 | 1207.64 | 1206.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1370.01 | 1371.98 | 1336.83 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1285.95 | 1331.77 | 1331.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 1272.70 | 1331.18 | 1331.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 1320.50 | 1312.03 | 1321.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 1287.10 | 1311.33 | 1320.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-16 12:15:00 | 1270.65 | 1245.42 | 1268.11 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1389.95 | 1283.66 | 1283.48 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 1179.15 | 1300.70 | 1300.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1172.25 | 1240.79 | 1262.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1164.55 | 1162.86 | 1203.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 09:15:00 | 1157.00 | 1169.40 | 1199.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1170.00 | 1164.38 | 1191.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 11:15:00 | 1159.95 | 1164.40 | 1191.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 1173.90 | 1148.89 | 1174.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 14:15:00 | 1177.80 | 1149.17 | 1174.47 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 1231.00 | 1182.99 | 1182.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1240.90 | 1184.75 | 1183.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1278.40 | 1294.47 | 1259.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-04 14:15:00 | 1307.30 | 1291.18 | 1262.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 1256.90 | 1290.07 | 1265.49 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.77 | 1257.92 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.59 | 1255.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.78 | 1255.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-04 11:15:00 | 1266.90 | 1257.81 | 1256.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1258.50 | 1257.93 | 1256.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-04 14:15:00 | 1252.80 | 1257.88 | 1256.81 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1251.50 | 1265.93 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1250.00 | 1265.77 | 1265.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1268.60 | 1259.74 | 1262.60 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.10 | 1265.09 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.01 | 1265.10 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.50 | 1252.56 | 1252.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 1275.20 | 1253.08 | 1252.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.69 | 1256.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.43 | 1255.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.78 | 1239.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-28 10:15:00 | 1224.00 | 1227.47 | 1239.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1221.50 | 1222.04 | 1234.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 1236.20 | 1222.28 | 1234.69 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1242.42 | 1242.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1287.90 | 1246.26 | 1244.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 1283.50 | 1284.04 | 1268.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 10:15:00 | 1298.40 | 1283.44 | 1269.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1276.30 | 1283.96 | 1270.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-19 10:15:00 | 1269.00 | 1283.81 | 1270.27 | Close below EMA400 |

### Cycle 17 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.30 | 1263.39 | 1263.53 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1259.31 | 1259.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.20 | 1260.31 | 1259.71 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-10 09:15:00 | 1081.99 | 2023-11-16 11:15:00 | 1097.25 | EXIT_EMA400 | -15.26 |
| SELL | 2023-11-13 10:15:00 | 1083.32 | 2023-11-16 11:15:00 | 1097.25 | EXIT_EMA400 | -13.93 |
| BUY | 2023-12-12 09:15:00 | 1124.00 | 2023-12-12 10:15:00 | 1110.02 | EXIT_EMA400 | -13.98 |
| SELL | 2024-05-30 09:15:00 | 1183.20 | 2024-06-07 11:15:00 | 1204.40 | EXIT_EMA400 | -21.20 |
| SELL | 2024-11-07 09:15:00 | 1287.10 | 2024-11-18 09:15:00 | 1186.97 | TARGET | 100.13 |
| SELL | 2025-04-03 11:15:00 | 1159.95 | 2025-04-07 09:15:00 | 1065.08 | TARGET | 94.87 |
| SELL | 2025-03-26 09:15:00 | 1157.00 | 2025-04-21 14:15:00 | 1177.80 | EXIT_EMA400 | -20.80 |
| BUY | 2025-07-04 14:15:00 | 1307.30 | 2025-07-10 09:15:00 | 1256.90 | EXIT_EMA400 | -50.40 |
| BUY | 2025-09-04 11:15:00 | 1266.90 | 2025-09-04 14:15:00 | 1252.80 | EXIT_EMA400 | -14.10 |
| SELL | 2026-01-28 10:15:00 | 1224.00 | 2026-02-02 10:15:00 | 1178.99 | TARGET | 45.01 |
| BUY | 2026-03-18 10:15:00 | 1298.40 | 2026-03-19 10:15:00 | 1269.00 | EXIT_EMA400 | -29.40 |
