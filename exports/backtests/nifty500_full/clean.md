# Clean Science and Technology Ltd. (CLEAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 819.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -146.82
- **Avg P&L per closed trade:** -14.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 13:15:00 | 1401.50 | 1364.83 | 1364.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 1413.15 | 1366.05 | 1365.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 13:15:00 | 1424.85 | 1427.14 | 1405.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-21 15:15:00 | 1431.00 | 1427.13 | 1405.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 1407.95 | 1426.39 | 1405.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-22 15:15:00 | 1402.50 | 1426.15 | 1405.48 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 1350.25 | 1399.33 | 1399.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 1342.50 | 1397.81 | 1398.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 11:15:00 | 1370.00 | 1369.13 | 1382.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-03 12:15:00 | 1350.00 | 1368.79 | 1381.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-06 11:15:00 | 1380.90 | 1367.93 | 1380.89 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 14:15:00 | 1428.70 | 1380.92 | 1380.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 1453.95 | 1386.25 | 1383.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 1496.15 | 1500.64 | 1459.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 13:15:00 | 1517.90 | 1500.73 | 1461.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 1463.70 | 1501.61 | 1467.62 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 09:15:00 | 1399.90 | 1452.84 | 1452.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1375.00 | 1430.93 | 1438.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 1365.50 | 1356.76 | 1388.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-02 11:15:00 | 1343.00 | 1356.72 | 1388.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1370.40 | 1350.77 | 1378.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-12 14:15:00 | 1356.55 | 1351.41 | 1377.82 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-16 09:15:00 | 1353.35 | 1315.94 | 1340.81 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 1457.05 | 1347.35 | 1347.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 13:15:00 | 1472.00 | 1361.09 | 1354.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1437.75 | 1442.41 | 1409.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-25 09:15:00 | 1541.20 | 1446.53 | 1415.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-29 10:15:00 | 1483.70 | 1545.78 | 1505.19 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1465.00 | 1532.65 | 1532.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1435.95 | 1529.42 | 1531.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 11:15:00 | 1378.50 | 1359.75 | 1419.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 1339.00 | 1418.66 | 1425.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1404.75 | 1401.42 | 1415.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 14:15:00 | 1422.00 | 1401.97 | 1415.36 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1425.20 | 1255.40 | 1254.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1441.20 | 1271.83 | 1263.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1407.20 | 1412.77 | 1360.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 1444.80 | 1413.71 | 1363.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 1350.60 | 1444.09 | 1407.26 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 1233.90 | 1379.10 | 1379.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 1224.40 | 1357.34 | 1367.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 1195.70 | 1192.73 | 1239.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 1178.00 | 1192.67 | 1234.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 747.30 | 721.25 | 758.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 11:15:00 | 744.05 | 723.57 | 758.26 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 766.40 | 724.95 | 758.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-21 15:15:00 | 1431.00 | 2023-09-22 15:15:00 | 1402.50 | EXIT_EMA400 | -28.50 |
| SELL | 2023-11-03 12:15:00 | 1350.00 | 2023-11-06 11:15:00 | 1380.90 | EXIT_EMA400 | -30.90 |
| BUY | 2024-01-11 13:15:00 | 1517.90 | 2024-01-18 09:15:00 | 1463.70 | EXIT_EMA400 | -54.20 |
| SELL | 2024-04-12 14:15:00 | 1356.55 | 2024-04-19 13:15:00 | 1292.75 | TARGET | 63.80 |
| SELL | 2024-04-02 11:15:00 | 1343.00 | 2024-05-16 09:15:00 | 1353.35 | EXIT_EMA400 | -10.35 |
| BUY | 2024-07-25 09:15:00 | 1541.20 | 2024-08-29 10:15:00 | 1483.70 | EXIT_EMA400 | -57.50 |
| SELL | 2025-01-27 09:15:00 | 1339.00 | 2025-01-31 14:15:00 | 1422.00 | EXIT_EMA400 | -83.00 |
| BUY | 2025-06-24 09:15:00 | 1444.80 | 2025-07-18 09:15:00 | 1350.60 | EXIT_EMA400 | -94.20 |
| SELL | 2025-09-19 14:15:00 | 1178.00 | 2025-10-31 15:15:00 | 1007.62 | TARGET | 170.38 |
| SELL | 2026-04-16 11:15:00 | 744.05 | 2026-04-17 09:15:00 | 766.40 | EXIT_EMA400 | -22.35 |
