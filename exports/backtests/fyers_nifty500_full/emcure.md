# Emcure Pharmaceuticals Ltd. (EMCURE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-07-10 09:15:00 → 2026-04-30 15:15:00 (3126 bars)
- **Last close:** 1695.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -36.52
- **Avg P&L per closed trade:** -5.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1306.85 | 1410.11 | 1410.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1297.05 | 1408.98 | 1409.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 1416.25 | 1388.87 | 1399.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-12 09:15:00 | 1343.00 | 1378.69 | 1389.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1385.75 | 1376.39 | 1387.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-16 13:15:00 | 1397.05 | 1376.52 | 1387.62 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 09:15:00 | 1441.85 | 1394.53 | 1394.40 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 13:15:00 | 1369.35 | 1398.93 | 1399.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 1361.00 | 1397.92 | 1398.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 1019.50 | 1018.85 | 1116.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 12:15:00 | 1008.15 | 1021.52 | 1110.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1068.90 | 1032.06 | 1100.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 10:15:00 | 1065.15 | 1032.39 | 1100.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 1045.10 | 1032.52 | 1100.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 13:15:00 | 1044.65 | 1032.77 | 1099.47 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 1065.90 | 1003.04 | 1066.28 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 13:15:00 | 1073.00 | 1003.74 | 1066.32 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1395.70 | 1072.60 | 1071.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 1412.80 | 1075.98 | 1073.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1285.00 | 1288.52 | 1227.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 09:15:00 | 1315.30 | 1277.41 | 1236.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1368.00 | 1371.88 | 1319.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-11 11:15:00 | 1378.70 | 1371.95 | 1320.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1395.70 | 1419.16 | 1369.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-02 13:15:00 | 1364.50 | 1417.59 | 1369.71 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 14:15:00 | 1278.00 | 1354.51 | 1354.74 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1409.20 | 1354.84 | 1354.71 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 1321.10 | 1358.41 | 1358.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1313.40 | 1356.85 | 1357.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1373.50 | 1352.74 | 1355.58 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1431.20 | 1358.13 | 1358.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1456.90 | 1388.39 | 1377.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1391.40 | 1392.25 | 1380.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-23 10:15:00 | 1411.10 | 1390.30 | 1381.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1406.80 | 1394.49 | 1384.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-30 10:15:00 | 1381.40 | 1394.38 | 1384.61 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-12 09:15:00 | 1343.00 | 2024-12-16 13:15:00 | 1397.05 | EXIT_EMA400 | -54.05 |
| SELL | 2025-04-03 10:15:00 | 1065.15 | 2025-04-07 09:15:00 | 959.72 | TARGET | 105.43 |
| SELL | 2025-03-25 12:15:00 | 1008.15 | 2025-04-17 13:15:00 | 1073.00 | EXIT_EMA400 | -64.85 |
| SELL | 2025-04-03 13:15:00 | 1044.65 | 2025-04-17 13:15:00 | 1073.00 | EXIT_EMA400 | -28.35 |
| BUY | 2025-07-11 09:15:00 | 1315.30 | 2025-09-02 13:15:00 | 1364.50 | EXIT_EMA400 | 49.20 |
| BUY | 2025-08-11 11:15:00 | 1378.70 | 2025-09-02 13:15:00 | 1364.50 | EXIT_EMA400 | -14.20 |
| BUY | 2025-12-23 10:15:00 | 1411.10 | 2025-12-30 10:15:00 | 1381.40 | EXIT_EMA400 | -29.70 |
