# Voltas Ltd. (VOLTAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1438.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -27.09
- **Avg P&L per closed trade:** -3.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 1655.95 | 1729.81 | 1729.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 1639.00 | 1719.30 | 1724.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1721.95 | 1712.06 | 1720.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 09:15:00 | 1706.80 | 1712.10 | 1720.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1706.80 | 1712.10 | 1720.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-03 10:15:00 | 1686.80 | 1711.85 | 1720.04 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-06 13:15:00 | 1716.45 | 1705.62 | 1715.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 1777.10 | 1724.66 | 1724.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 1784.30 | 1725.25 | 1724.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 14:15:00 | 1736.45 | 1737.56 | 1731.56 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 12:15:00 | 1710.25 | 1726.49 | 1726.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 15:15:00 | 1703.00 | 1725.96 | 1726.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 1768.90 | 1725.64 | 1726.07 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 11:15:00 | 1773.80 | 1726.91 | 1726.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 1805.20 | 1728.16 | 1727.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1724.55 | 1751.11 | 1740.18 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 1634.10 | 1731.22 | 1731.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1621.15 | 1727.39 | 1729.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1363.85 | 1359.65 | 1462.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1311.35 | 1401.47 | 1438.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1295.60 | 1263.81 | 1300.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-10 09:15:00 | 1303.00 | 1265.66 | 1299.73 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1367.90 | 1309.03 | 1309.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1372.20 | 1310.22 | 1309.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1344.90 | 1346.05 | 1331.65 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1244.40 | 1323.74 | 1324.08 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1370.50 | 1323.49 | 1323.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 1373.30 | 1334.55 | 1329.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 10:15:00 | 1386.30 | 1386.66 | 1364.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 09:15:00 | 1410.10 | 1373.98 | 1364.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1372.70 | 1376.91 | 1366.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 15:15:00 | 1380.00 | 1376.88 | 1366.98 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-31 14:15:00 | 1381.00 | 1401.80 | 1385.43 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1329.60 | 1373.01 | 1373.21 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 1399.40 | 1373.36 | 1373.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 15:15:00 | 1404.00 | 1374.23 | 1373.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 10:15:00 | 1366.50 | 1378.01 | 1375.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 12:15:00 | 1387.70 | 1376.05 | 1374.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-27 09:15:00 | 1368.90 | 1376.30 | 1375.05 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1352.00 | 1373.97 | 1374.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1339.60 | 1373.63 | 1373.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1358.90 | 1358.42 | 1365.19 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1406.80 | 1370.54 | 1370.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1417.70 | 1373.42 | 1372.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1416.10 | 1416.94 | 1398.15 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 13:15:00 | 1348.60 | 1383.72 | 1383.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 1340.00 | 1382.90 | 1383.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1406.60 | 1368.99 | 1375.79 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1498.60 | 1382.21 | 1381.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1501.70 | 1384.54 | 1383.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1447.40 | 1479.45 | 1443.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 14:15:00 | 1480.80 | 1476.67 | 1443.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1464.70 | 1476.60 | 1445.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-09 10:15:00 | 1443.60 | 1476.27 | 1445.15 | Close below EMA400 |

### Cycle 15 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 1253.80 | 1426.89 | 1427.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 1225.80 | 1393.65 | 1409.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1350.90 | 1344.12 | 1378.23 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 1507.20 | 1399.88 | 1399.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1510.70 | 1402.01 | 1400.80 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-03 09:15:00 | 1706.80 | 2024-12-05 09:15:00 | 1666.57 | TARGET | 40.23 |
| SELL | 2024-12-03 10:15:00 | 1686.80 | 2024-12-06 13:15:00 | 1716.45 | EXIT_EMA400 | -29.65 |
| SELL | 2025-04-04 09:15:00 | 1311.35 | 2025-06-10 09:15:00 | 1303.00 | EXIT_EMA400 | 8.35 |
| BUY | 2025-10-13 15:15:00 | 1380.00 | 2025-10-16 12:15:00 | 1419.07 | TARGET | 39.07 |
| BUY | 2025-10-10 09:15:00 | 1410.10 | 2025-10-31 14:15:00 | 1381.00 | EXIT_EMA400 | -29.10 |
| BUY | 2025-11-26 12:15:00 | 1387.70 | 2025-11-27 09:15:00 | 1368.90 | EXIT_EMA400 | -18.80 |
| BUY | 2026-03-05 14:15:00 | 1480.80 | 2026-03-09 10:15:00 | 1443.60 | EXIT_EMA400 | -37.20 |
