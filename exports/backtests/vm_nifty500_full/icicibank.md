# ICICI Bank Ltd. (ICICIBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1263.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** 24.64
- **Avg P&L per closed trade:** 2.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 15:15:00 | 940.00 | 962.84 | 962.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 925.90 | 962.47 | 962.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 959.00 | 956.29 | 959.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-13 09:15:00 | 946.60 | 956.09 | 959.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 958.15 | 955.47 | 958.48 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-10-18 09:15:00 | 947.80 | 955.39 | 958.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-06 15:15:00 | 947.50 | 937.72 | 946.53 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 15:15:00 | 1002.70 | 945.74 | 945.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 1004.05 | 949.75 | 947.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 986.30 | 988.42 | 973.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-05 09:15:00 | 996.00 | 987.82 | 974.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 977.75 | 988.04 | 976.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-10 09:15:00 | 987.55 | 988.04 | 976.33 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 984.90 | 991.82 | 980.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-17 15:15:00 | 979.50 | 991.44 | 980.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1237.50 | 1279.88 | 1279.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 1226.00 | 1274.94 | 1277.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1256.25 | 1252.02 | 1263.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-17 09:15:00 | 1236.00 | 1254.77 | 1261.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-19 14:15:00 | 1262.10 | 1253.47 | 1259.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.86 | 1254.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.24 | 1256.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1287.85 | 1295.44 | 1278.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 09:15:00 | 1347.30 | 1296.85 | 1281.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-17 09:15:00 | 1412.60 | 1428.72 | 1415.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1391.20 | 1429.48 | 1429.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1381.20 | 1416.12 | 1421.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1389.80 | 1388.78 | 1401.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-21 14:15:00 | 1382.00 | 1393.53 | 1402.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-13 10:15:00 | 1385.30 | 1368.48 | 1383.91 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.90 | 1377.35 | 1377.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.48 | 1377.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1347.60 | 1378.54 | 1378.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.20 | 1378.19 | 1378.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-30 10:15:00 | 1364.90 | 1375.41 | 1376.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1364.90 | 1375.41 | 1376.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-30 11:15:00 | 1360.30 | 1375.26 | 1376.78 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 1390.10 | 1372.87 | 1375.46 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1398.30 | 1377.89 | 1377.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1403.60 | 1378.29 | 1378.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-20 09:15:00 | 1398.20 | 1392.01 | 1386.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1398.20 | 1392.01 | 1386.21 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-23 09:15:00 | 1406.70 | 1392.39 | 1386.60 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 1390.00 | 1392.80 | 1387.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-24 14:15:00 | 1384.80 | 1392.71 | 1387.11 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.42 | 1383.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.64 | 1382.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.44 | 1317.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 11:15:00 | 1290.30 | 1283.33 | 1317.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 1321.00 | 1283.72 | 1316.67 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-18 09:15:00 | 947.80 | 2023-10-25 12:15:00 | 916.18 | TARGET | 31.62 |
| SELL | 2023-10-13 09:15:00 | 946.60 | 2023-10-26 09:15:00 | 909.39 | TARGET | 37.21 |
| BUY | 2024-01-05 09:15:00 | 996.00 | 2024-01-17 15:15:00 | 979.50 | EXIT_EMA400 | -16.50 |
| BUY | 2024-01-10 09:15:00 | 987.55 | 2024-01-17 15:15:00 | 979.50 | EXIT_EMA400 | -8.05 |
| SELL | 2025-02-17 09:15:00 | 1236.00 | 2025-02-19 14:15:00 | 1262.10 | EXIT_EMA400 | -26.10 |
| BUY | 2025-04-15 09:15:00 | 1347.30 | 2025-07-17 09:15:00 | 1412.60 | EXIT_EMA400 | 65.30 |
| SELL | 2025-10-21 14:15:00 | 1382.00 | 2025-11-06 11:15:00 | 1319.84 | TARGET | 62.16 |
| SELL | 2026-01-30 10:15:00 | 1364.90 | 2026-02-03 09:15:00 | 1390.10 | EXIT_EMA400 | -25.20 |
| SELL | 2026-01-30 11:15:00 | 1360.30 | 2026-02-03 09:15:00 | 1390.10 | EXIT_EMA400 | -29.80 |
| BUY | 2026-02-20 09:15:00 | 1398.20 | 2026-02-24 14:15:00 | 1384.80 | EXIT_EMA400 | -13.40 |
| BUY | 2026-02-23 09:15:00 | 1406.70 | 2026-02-24 14:15:00 | 1384.80 | EXIT_EMA400 | -21.90 |
| SELL | 2026-04-09 11:15:00 | 1290.30 | 2026-04-10 09:15:00 | 1321.00 | EXIT_EMA400 | -30.70 |
