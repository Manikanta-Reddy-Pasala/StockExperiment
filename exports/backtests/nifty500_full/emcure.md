# Emcure Pharmaceuticals Ltd. (EMCURE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-07-10 09:15:00 → 2026-04-30 15:30:00 (3107 bars)
- **Last close:** 1680.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 9 |
| ENTRY1 | 5 |
| ENTRY2 | 6 |
| EXIT | 5 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -45.25
- **Avg P&L per closed trade:** -4.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1306.85 | 1410.05 | 1410.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1297.05 | 1408.92 | 1409.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 1415.25 | 1388.82 | 1399.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-12 09:15:00 | 1341.75 | 1378.72 | 1389.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1385.75 | 1376.44 | 1387.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-16 13:15:00 | 1397.05 | 1376.56 | 1387.58 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 09:15:00 | 1441.85 | 1394.63 | 1394.42 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 13:15:00 | 1369.35 | 1398.96 | 1399.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 1361.00 | 1397.95 | 1398.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 1019.50 | 1019.39 | 1118.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 12:15:00 | 1008.15 | 1021.95 | 1111.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1068.10 | 1032.33 | 1101.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 10:15:00 | 1065.15 | 1032.66 | 1101.20 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 1065.90 | 1003.00 | 1066.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 13:15:00 | 1073.00 | 1003.70 | 1066.89 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1395.70 | 1072.50 | 1072.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 1412.80 | 1075.89 | 1073.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1285.00 | 1288.55 | 1227.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 09:15:00 | 1315.30 | 1277.43 | 1236.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1368.00 | 1371.99 | 1320.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-11 11:15:00 | 1378.70 | 1372.06 | 1320.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1395.70 | 1419.28 | 1369.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-02 13:15:00 | 1364.50 | 1417.69 | 1369.81 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 14:15:00 | 1278.00 | 1354.52 | 1354.79 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1409.20 | 1354.82 | 1354.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 1428.50 | 1365.29 | 1360.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1356.50 | 1367.48 | 1361.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-20 12:15:00 | 1380.90 | 1363.93 | 1360.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1380.90 | 1363.93 | 1360.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-20 13:15:00 | 1383.40 | 1364.12 | 1360.72 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-24 11:15:00 | 1359.90 | 1366.81 | 1362.34 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 1321.10 | 1358.54 | 1358.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1313.40 | 1356.97 | 1357.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1373.50 | 1352.93 | 1355.72 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1431.50 | 1358.62 | 1358.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1456.90 | 1388.61 | 1377.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1382.00 | 1392.38 | 1380.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-17 14:15:00 | 1391.70 | 1391.29 | 1380.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1386.10 | 1391.25 | 1380.72 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-19 09:15:00 | 1399.40 | 1391.01 | 1380.96 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1382.30 | 1390.84 | 1381.02 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-19 13:15:00 | 1387.00 | 1390.80 | 1381.05 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1382.10 | 1390.55 | 1381.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-22 11:15:00 | 1390.80 | 1390.55 | 1381.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-22 13:15:00 | 1380.10 | 1390.39 | 1381.18 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-12 09:15:00 | 1341.75 | 2024-12-16 13:15:00 | 1397.05 | EXIT_EMA400 | -55.30 |
| SELL | 2025-04-03 10:15:00 | 1065.15 | 2025-04-07 09:15:00 | 957.00 | TARGET | 108.15 |
| SELL | 2025-03-25 12:15:00 | 1008.15 | 2025-04-17 13:15:00 | 1073.00 | EXIT_EMA400 | -64.85 |
| BUY | 2025-07-11 09:15:00 | 1315.30 | 2025-09-02 13:15:00 | 1364.50 | EXIT_EMA400 | 49.20 |
| BUY | 2025-08-11 11:15:00 | 1378.70 | 2025-09-02 13:15:00 | 1364.50 | EXIT_EMA400 | -14.20 |
| BUY | 2025-10-20 12:15:00 | 1380.90 | 2025-10-24 11:15:00 | 1359.90 | EXIT_EMA400 | -21.00 |
| BUY | 2025-10-20 13:15:00 | 1383.40 | 2025-10-24 11:15:00 | 1359.90 | EXIT_EMA400 | -23.50 |
| BUY | 2025-12-19 13:15:00 | 1387.00 | 2025-12-22 09:15:00 | 1404.85 | TARGET | 17.85 |
| BUY | 2025-12-17 14:15:00 | 1391.70 | 2025-12-22 13:15:00 | 1380.10 | EXIT_EMA400 | -11.60 |
| BUY | 2025-12-19 09:15:00 | 1399.40 | 2025-12-22 13:15:00 | 1380.10 | EXIT_EMA400 | -19.30 |
| BUY | 2025-12-22 11:15:00 | 1390.80 | 2025-12-22 13:15:00 | 1380.10 | EXIT_EMA400 | -10.70 |
