# Voltas Ltd. (VOLTAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1430.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 16 |
| ALERT2 | 15 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** -32.62
- **Avg P&L per closed trade:** -2.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 15:15:00 | 813.40 | 841.79 | 841.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 12:15:00 | 812.00 | 837.75 | 839.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 846.70 | 834.12 | 837.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-24 10:15:00 | 830.00 | 835.65 | 837.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 831.10 | 832.75 | 836.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-01 12:15:00 | 837.80 | 832.78 | 836.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 858.00 | 838.49 | 838.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 11:15:00 | 862.50 | 841.67 | 840.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 1070.90 | 1071.28 | 1026.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-07 11:15:00 | 1081.75 | 1070.57 | 1028.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 1035.80 | 1069.14 | 1032.28 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-14 09:15:00 | 1051.05 | 1068.06 | 1032.47 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1037.20 | 1066.12 | 1035.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-19 12:15:00 | 1043.75 | 1065.62 | 1035.24 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-03-20 09:15:00 | 1034.75 | 1064.73 | 1035.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 1675.90 | 1729.05 | 1729.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 1664.25 | 1727.34 | 1728.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1721.95 | 1712.90 | 1720.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 09:15:00 | 1706.80 | 1712.93 | 1720.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1706.80 | 1712.93 | 1720.45 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-03 10:15:00 | 1686.80 | 1712.67 | 1720.28 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-06 13:15:00 | 1716.50 | 1706.27 | 1716.04 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 10:15:00 | 1767.80 | 1724.29 | 1724.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 1809.40 | 1726.64 | 1725.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 14:15:00 | 1735.90 | 1738.02 | 1731.78 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 1705.90 | 1726.45 | 1726.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 1702.55 | 1725.73 | 1726.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 1768.90 | 1725.89 | 1726.22 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 09:15:00 | 1755.00 | 1726.63 | 1726.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 10:15:00 | 1766.40 | 1727.02 | 1726.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1725.65 | 1751.49 | 1740.43 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 1634.10 | 1731.51 | 1731.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1621.15 | 1727.63 | 1729.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 1368.35 | 1364.29 | 1467.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-01 14:15:00 | 1354.55 | 1412.68 | 1448.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1295.60 | 1263.86 | 1300.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-10 09:15:00 | 1303.00 | 1265.70 | 1300.32 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 13:15:00 | 1366.10 | 1309.64 | 1309.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1372.20 | 1310.27 | 1309.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1344.60 | 1346.14 | 1331.90 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1241.20 | 1324.69 | 1324.71 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1370.50 | 1323.66 | 1323.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 1373.20 | 1334.67 | 1329.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 10:15:00 | 1386.30 | 1386.73 | 1364.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 09:15:00 | 1410.10 | 1374.15 | 1365.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1372.70 | 1377.07 | 1366.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 15:15:00 | 1380.00 | 1377.04 | 1367.14 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-31 14:15:00 | 1381.50 | 1401.82 | 1385.51 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 1329.00 | 1373.43 | 1373.48 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 1402.00 | 1373.59 | 1373.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 1405.50 | 1374.48 | 1373.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 10:15:00 | 1366.50 | 1377.98 | 1375.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 12:15:00 | 1387.70 | 1375.90 | 1374.86 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-27 09:15:00 | 1369.50 | 1376.16 | 1375.01 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1352.90 | 1373.95 | 1374.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1339.60 | 1373.61 | 1373.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1358.90 | 1358.48 | 1365.24 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 1393.10 | 1370.35 | 1370.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 1406.80 | 1370.71 | 1370.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 09:15:00 | 1364.30 | 1374.33 | 1372.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-01 11:15:00 | 1381.20 | 1372.68 | 1371.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 1404.10 | 1416.53 | 1398.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 14:15:00 | 1398.00 | 1416.18 | 1398.54 | Close below EMA400 |

### Cycle 15 — SELL (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 13:15:00 | 1348.60 | 1383.84 | 1384.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 1340.00 | 1382.96 | 1383.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1406.90 | 1373.02 | 1378.13 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1486.40 | 1383.10 | 1382.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1495.90 | 1384.22 | 1383.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1447.40 | 1480.55 | 1444.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 12:15:00 | 1479.20 | 1477.67 | 1444.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1464.70 | 1477.53 | 1446.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-09 10:15:00 | 1443.60 | 1477.19 | 1446.27 | Close below EMA400 |

### Cycle 17 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1245.00 | 1429.15 | 1429.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 1225.80 | 1394.03 | 1410.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1350.90 | 1344.31 | 1378.76 | EMA200 retest candle locked |

### Cycle 18 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 1511.10 | 1400.89 | 1400.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1516.70 | 1402.04 | 1401.21 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-24 10:15:00 | 830.00 | 2023-12-01 12:15:00 | 837.80 | EXIT_EMA400 | -7.80 |
| BUY | 2024-03-07 11:15:00 | 1081.75 | 2024-03-20 09:15:00 | 1034.75 | EXIT_EMA400 | -47.00 |
| BUY | 2024-03-14 09:15:00 | 1051.05 | 2024-03-20 09:15:00 | 1034.75 | EXIT_EMA400 | -16.30 |
| BUY | 2024-03-19 12:15:00 | 1043.75 | 2024-03-20 09:15:00 | 1034.75 | EXIT_EMA400 | -9.00 |
| SELL | 2024-12-03 09:15:00 | 1706.80 | 2024-12-05 09:15:00 | 1665.85 | TARGET | 40.95 |
| SELL | 2024-12-03 10:15:00 | 1686.80 | 2024-12-06 13:15:00 | 1716.50 | EXIT_EMA400 | -29.70 |
| SELL | 2025-04-01 14:15:00 | 1354.55 | 2025-06-10 09:15:00 | 1303.00 | EXIT_EMA400 | 51.55 |
| BUY | 2025-10-13 15:15:00 | 1380.00 | 2025-10-16 12:15:00 | 1418.57 | TARGET | 38.57 |
| BUY | 2025-10-10 09:15:00 | 1410.10 | 2025-10-31 14:15:00 | 1381.50 | EXIT_EMA400 | -28.60 |
| BUY | 2025-11-26 12:15:00 | 1387.70 | 2025-11-27 09:15:00 | 1369.50 | EXIT_EMA400 | -18.20 |
| BUY | 2026-01-01 11:15:00 | 1381.20 | 2026-01-02 09:15:00 | 1409.71 | TARGET | 28.51 |
| BUY | 2026-03-05 12:15:00 | 1479.20 | 2026-03-09 10:15:00 | 1443.60 | EXIT_EMA400 | -35.60 |
