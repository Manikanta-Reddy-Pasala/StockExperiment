# Mahanagar Gas Ltd. (MGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1135.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 6 |
| EXIT | 7 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / EMA400 exits:** 3 / 10
- **Total realized P&L (per unit):** -174.69
- **Avg P&L per closed trade:** -13.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 11:15:00 | 1109.30 | 1045.73 | 1045.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 12:15:00 | 1113.45 | 1046.41 | 1045.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 09:15:00 | 1044.45 | 1085.18 | 1068.81 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 1002.05 | 1055.91 | 1056.09 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 10:15:00 | 1117.05 | 1051.80 | 1051.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 09:15:00 | 1124.75 | 1055.60 | 1053.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 1341.95 | 1462.34 | 1375.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-26 11:15:00 | 1491.50 | 1409.24 | 1384.33 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-07 09:15:00 | 1374.50 | 1419.38 | 1394.59 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 1310.75 | 1376.45 | 1376.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 11:15:00 | 1298.00 | 1373.62 | 1375.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1374.10 | 1336.45 | 1352.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 10:15:00 | 1273.35 | 1337.81 | 1353.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-06 09:15:00 | 1368.75 | 1332.33 | 1349.21 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 1471.00 | 1362.15 | 1362.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 10:15:00 | 1480.00 | 1370.15 | 1366.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 13:15:00 | 1745.30 | 1748.52 | 1652.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-16 09:15:00 | 1770.70 | 1748.35 | 1653.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1844.70 | 1877.25 | 1812.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-10 09:15:00 | 1878.10 | 1871.94 | 1813.91 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-14 09:15:00 | 1815.90 | 1868.81 | 1816.24 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 1579.55 | 1779.97 | 1780.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 1563.95 | 1775.87 | 1778.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 1285.45 | 1283.02 | 1379.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-07 09:15:00 | 1263.70 | 1285.64 | 1365.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1301.05 | 1282.73 | 1329.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-29 14:15:00 | 1270.50 | 1282.45 | 1328.13 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1317.85 | 1282.65 | 1327.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 09:15:00 | 1344.80 | 1284.67 | 1327.25 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 13:15:00 | 1361.80 | 1318.72 | 1318.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 1379.90 | 1322.14 | 1320.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 1329.85 | 1337.90 | 1329.22 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 11:15:00 | 1250.10 | 1321.77 | 1321.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 1245.20 | 1318.36 | 1320.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 1332.50 | 1312.55 | 1317.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-25 09:15:00 | 1294.30 | 1315.18 | 1318.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1294.30 | 1315.18 | 1318.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-25 10:15:00 | 1278.40 | 1314.82 | 1317.84 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-28 09:15:00 | 1321.00 | 1313.34 | 1317.00 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 14:15:00 | 1360.00 | 1320.14 | 1320.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 15:15:00 | 1360.80 | 1320.55 | 1320.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 12:15:00 | 1330.60 | 1339.46 | 1330.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 13:15:00 | 1362.40 | 1339.69 | 1330.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1362.40 | 1339.69 | 1330.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 09:15:00 | 1393.20 | 1340.67 | 1331.43 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1346.10 | 1361.47 | 1345.95 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-23 12:15:00 | 1345.70 | 1360.87 | 1346.10 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1285.60 | 1407.89 | 1408.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 1274.70 | 1357.24 | 1377.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1331.10 | 1316.47 | 1345.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 10:15:00 | 1288.20 | 1323.05 | 1343.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1302.70 | 1303.01 | 1326.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-08 13:15:00 | 1280.70 | 1302.87 | 1325.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1316.00 | 1300.66 | 1320.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-16 13:15:00 | 1305.90 | 1301.20 | 1320.29 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-06 11:15:00 | 1127.60 | 1086.53 | 1126.64 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 12:15:00 | 1134.35 | 1081.47 | 1081.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 1141.80 | 1082.07 | 1081.72 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-26 11:15:00 | 1491.50 | 2024-05-07 09:15:00 | 1374.50 | EXIT_EMA400 | -117.00 |
| SELL | 2024-06-04 10:15:00 | 1273.35 | 2024-06-06 09:15:00 | 1368.75 | EXIT_EMA400 | -95.40 |
| BUY | 2024-08-16 09:15:00 | 1770.70 | 2024-10-14 09:15:00 | 1815.90 | EXIT_EMA400 | 45.20 |
| BUY | 2024-10-10 09:15:00 | 1878.10 | 2024-10-14 09:15:00 | 1815.90 | EXIT_EMA400 | -62.20 |
| SELL | 2025-01-07 09:15:00 | 1263.70 | 2025-01-31 09:15:00 | 1344.80 | EXIT_EMA400 | -81.10 |
| SELL | 2025-01-29 14:15:00 | 1270.50 | 2025-01-31 09:15:00 | 1344.80 | EXIT_EMA400 | -74.30 |
| SELL | 2025-04-25 09:15:00 | 1294.30 | 2025-04-28 09:15:00 | 1321.00 | EXIT_EMA400 | -26.70 |
| SELL | 2025-04-25 10:15:00 | 1278.40 | 2025-04-28 09:15:00 | 1321.00 | EXIT_EMA400 | -42.60 |
| BUY | 2025-05-09 13:15:00 | 1362.40 | 2025-05-23 12:15:00 | 1345.70 | EXIT_EMA400 | -16.70 |
| BUY | 2025-05-12 09:15:00 | 1393.20 | 2025-05-23 12:15:00 | 1345.70 | EXIT_EMA400 | -47.50 |
| SELL | 2025-10-16 13:15:00 | 1305.90 | 2025-10-30 09:15:00 | 1262.74 | TARGET | 43.16 |
| SELL | 2025-10-08 13:15:00 | 1280.70 | 2025-12-08 11:15:00 | 1145.92 | TARGET | 134.78 |
| SELL | 2025-09-24 10:15:00 | 1288.20 | 2025-12-08 13:15:00 | 1122.53 | TARGET | 165.67 |
