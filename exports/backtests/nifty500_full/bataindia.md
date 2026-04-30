# Bata India Ltd. (BATAINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 718.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| EXIT | 7 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / EMA400 exits:** 5 / 7
- **Total realized P&L (per unit):** 117.33
- **Avg P&L per closed trade:** 9.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 1606.60 | 1673.46 | 1673.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 1604.05 | 1672.77 | 1673.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 1654.60 | 1650.58 | 1660.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-11 12:15:00 | 1645.00 | 1650.80 | 1660.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 1645.00 | 1650.80 | 1660.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-10-11 13:15:00 | 1641.95 | 1650.71 | 1660.24 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1609.80 | 1591.34 | 1612.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-22 14:15:00 | 1617.20 | 1592.19 | 1612.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 1651.30 | 1622.44 | 1622.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 1660.90 | 1623.18 | 1622.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 1633.25 | 1634.52 | 1628.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-27 14:15:00 | 1639.65 | 1632.00 | 1628.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 1630.70 | 1632.05 | 1628.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-28 11:15:00 | 1647.90 | 1632.28 | 1628.42 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1638.30 | 1632.71 | 1628.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-29 10:15:00 | 1656.00 | 1632.94 | 1628.87 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-02 09:15:00 | 1616.00 | 1634.39 | 1629.88 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 09:15:00 | 1587.95 | 1625.91 | 1626.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 10:15:00 | 1584.70 | 1625.50 | 1625.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 11:15:00 | 1461.00 | 1448.94 | 1493.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-06 10:15:00 | 1423.45 | 1448.51 | 1492.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-21 09:15:00 | 1379.35 | 1346.30 | 1378.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 09:15:00 | 1480.85 | 1388.35 | 1388.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 12:15:00 | 1495.50 | 1426.93 | 1411.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 10:15:00 | 1540.40 | 1545.20 | 1499.37 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1439.05 | 1473.29 | 1473.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1436.05 | 1472.92 | 1473.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 12:15:00 | 1471.50 | 1470.20 | 1471.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-03 14:15:00 | 1463.30 | 1470.12 | 1471.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1446.30 | 1441.72 | 1452.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-27 15:15:00 | 1453.85 | 1441.99 | 1452.69 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1434.50 | 1399.18 | 1399.13 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 1367.90 | 1399.03 | 1399.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 13:15:00 | 1363.65 | 1398.68 | 1398.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 1408.45 | 1386.40 | 1392.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-30 09:15:00 | 1372.20 | 1386.34 | 1392.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 15:15:00 | 1393.00 | 1386.10 | 1391.75 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1427.75 | 1396.22 | 1396.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 11:15:00 | 1435.00 | 1400.01 | 1398.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 1388.35 | 1401.09 | 1398.69 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 14:15:00 | 1342.00 | 1396.03 | 1396.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1319.05 | 1394.79 | 1395.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 1404.05 | 1328.20 | 1355.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 13:15:00 | 1355.90 | 1338.94 | 1357.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1355.90 | 1338.94 | 1357.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-07 09:15:00 | 1344.00 | 1339.34 | 1357.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1345.00 | 1339.33 | 1356.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-11 12:15:00 | 1327.70 | 1339.22 | 1356.06 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 1339.05 | 1339.31 | 1355.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-12 10:15:00 | 1358.70 | 1339.50 | 1355.79 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 1263.80 | 1195.62 | 1195.30 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1164.70 | 1197.98 | 1198.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1158.90 | 1197.59 | 1197.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 932.45 | 893.20 | 939.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-11 11:15:00 | 903.50 | 895.06 | 938.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 771.70 | 715.72 | 764.03 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-11 12:15:00 | 1645.00 | 2023-10-23 09:15:00 | 1599.01 | TARGET | 45.99 |
| SELL | 2023-10-11 13:15:00 | 1641.95 | 2023-10-23 12:15:00 | 1587.09 | TARGET | 54.86 |
| BUY | 2023-12-27 14:15:00 | 1639.65 | 2024-01-02 09:15:00 | 1616.00 | EXIT_EMA400 | -23.65 |
| BUY | 2023-12-28 11:15:00 | 1647.90 | 2024-01-02 09:15:00 | 1616.00 | EXIT_EMA400 | -31.90 |
| BUY | 2023-12-29 10:15:00 | 1656.00 | 2024-01-02 09:15:00 | 1616.00 | EXIT_EMA400 | -40.00 |
| SELL | 2024-03-06 10:15:00 | 1423.45 | 2024-05-21 09:15:00 | 1379.35 | EXIT_EMA400 | 44.10 |
| SELL | 2024-09-03 14:15:00 | 1463.30 | 2024-09-05 15:15:00 | 1438.37 | TARGET | 24.93 |
| SELL | 2024-12-30 09:15:00 | 1372.20 | 2024-12-30 15:15:00 | 1393.00 | EXIT_EMA400 | -20.80 |
| SELL | 2025-02-06 13:15:00 | 1355.90 | 2025-02-07 09:15:00 | 1350.51 | TARGET | 5.39 |
| SELL | 2025-02-07 09:15:00 | 1344.00 | 2025-02-12 10:15:00 | 1358.70 | EXIT_EMA400 | -14.70 |
| SELL | 2025-02-11 12:15:00 | 1327.70 | 2025-02-12 10:15:00 | 1358.70 | EXIT_EMA400 | -31.00 |
| SELL | 2026-02-11 11:15:00 | 903.50 | 2026-02-24 13:15:00 | 799.40 | TARGET | 104.10 |
