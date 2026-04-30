# Kajaria Ceramics Ltd. (KAJARIACER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1187.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** 64.16
- **Avg P&L per closed trade:** 5.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 1336.45 | 1388.92 | 1389.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1302.75 | 1376.85 | 1382.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 11:15:00 | 1302.50 | 1291.20 | 1323.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 13:15:00 | 1283.55 | 1291.11 | 1322.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 12:15:00 | 1296.40 | 1283.70 | 1308.46 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-23 15:15:00 | 1291.50 | 1284.03 | 1308.26 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-24 09:15:00 | 1310.30 | 1284.29 | 1308.27 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 1399.90 | 1323.75 | 1323.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 09:15:00 | 1411.35 | 1340.01 | 1332.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 1351.15 | 1353.70 | 1341.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-08 09:15:00 | 1375.25 | 1338.46 | 1335.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1371.70 | 1360.59 | 1348.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-18 10:15:00 | 1393.95 | 1360.92 | 1349.01 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 1353.35 | 1364.78 | 1351.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-23 13:15:00 | 1337.30 | 1364.50 | 1351.88 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 1255.85 | 1344.79 | 1345.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 15:15:00 | 1249.65 | 1341.18 | 1343.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 1226.45 | 1221.19 | 1257.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 12:15:00 | 1215.35 | 1233.17 | 1255.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1253.35 | 1232.83 | 1253.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-18 11:15:00 | 1254.05 | 1233.04 | 1253.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 15:15:00 | 1317.40 | 1238.48 | 1238.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 1368.85 | 1249.14 | 1243.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 1357.20 | 1381.55 | 1331.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-23 13:15:00 | 1394.55 | 1380.00 | 1339.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1383.40 | 1421.56 | 1382.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-13 12:15:00 | 1396.90 | 1420.95 | 1382.70 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1397.75 | 1420.36 | 1383.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-14 10:15:00 | 1380.65 | 1419.97 | 1383.15 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 1277.70 | 1411.26 | 1411.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 12:15:00 | 1265.60 | 1409.81 | 1411.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 14:15:00 | 1172.75 | 1171.76 | 1219.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-02 11:15:00 | 1150.00 | 1171.15 | 1218.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 855.95 | 829.13 | 870.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-12 10:15:00 | 847.60 | 829.31 | 870.50 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-14 10:15:00 | 883.85 | 832.29 | 869.25 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 1048.40 | 895.00 | 894.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1050.95 | 916.67 | 906.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 11:15:00 | 1223.90 | 1228.80 | 1168.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-28 13:15:00 | 1234.00 | 1228.87 | 1169.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1198.20 | 1227.92 | 1195.93 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-23 12:15:00 | 1188.00 | 1226.42 | 1195.96 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1115.00 | 1198.28 | 1198.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 12:15:00 | 1108.50 | 1195.73 | 1197.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1092.50 | 1091.87 | 1127.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 10:15:00 | 1073.10 | 1091.30 | 1123.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-18 09:15:00 | 1007.60 | 954.81 | 994.64 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 09:15:00 | 1112.20 | 976.49 | 976.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 1117.15 | 977.89 | 976.96 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-07 13:15:00 | 1283.55 | 2023-11-24 09:15:00 | 1310.30 | EXIT_EMA400 | -26.75 |
| SELL | 2023-11-23 15:15:00 | 1291.50 | 2023-11-24 09:15:00 | 1310.30 | EXIT_EMA400 | -18.80 |
| BUY | 2024-01-08 09:15:00 | 1375.25 | 2024-01-23 13:15:00 | 1337.30 | EXIT_EMA400 | -37.95 |
| BUY | 2024-01-18 10:15:00 | 1393.95 | 2024-01-23 13:15:00 | 1337.30 | EXIT_EMA400 | -56.65 |
| SELL | 2024-04-15 12:15:00 | 1215.35 | 2024-04-18 11:15:00 | 1254.05 | EXIT_EMA400 | -38.70 |
| BUY | 2024-07-23 13:15:00 | 1394.55 | 2024-08-14 10:15:00 | 1380.65 | EXIT_EMA400 | -13.90 |
| BUY | 2024-08-13 12:15:00 | 1396.90 | 2024-08-14 10:15:00 | 1380.65 | EXIT_EMA400 | -16.25 |
| SELL | 2025-01-02 11:15:00 | 1150.00 | 2025-02-11 09:15:00 | 945.64 | TARGET | 204.36 |
| SELL | 2025-05-12 10:15:00 | 847.60 | 2025-05-14 10:15:00 | 883.85 | EXIT_EMA400 | -36.25 |
| BUY | 2025-08-28 13:15:00 | 1234.00 | 2025-09-23 12:15:00 | 1188.00 | EXIT_EMA400 | -46.00 |
| SELL | 2025-12-18 10:15:00 | 1073.10 | 2026-01-27 09:15:00 | 922.05 | TARGET | 151.05 |
