# Pidilite Industries Ltd. (PIDILITIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1375.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 187.10
- **Avg P&L per closed trade:** 18.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 15:15:00 | 1275.00 | 1238.28 | 1238.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 1283.75 | 1238.73 | 1238.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 10:15:00 | 1340.50 | 1341.09 | 1311.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-16 14:15:00 | 1356.25 | 1319.89 | 1309.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-19 09:15:00 | 1411.62 | 1456.24 | 1416.76 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 1545.47 | 1584.83 | 1584.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 1539.10 | 1578.84 | 1581.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1547.68 | 1539.36 | 1556.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 09:15:00 | 1504.95 | 1562.00 | 1564.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1488.15 | 1452.37 | 1490.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-03 10:15:00 | 1481.07 | 1447.98 | 1479.84 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 14:15:00 | 1513.20 | 1425.95 | 1425.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 1521.30 | 1427.76 | 1426.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 1473.05 | 1476.80 | 1457.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 12:15:00 | 1482.20 | 1476.84 | 1457.73 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 1493.75 | 1520.14 | 1499.43 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1452.20 | 1500.33 | 1500.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1443.50 | 1498.84 | 1499.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1482.95 | 1478.38 | 1488.23 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1540.65 | 1495.59 | 1495.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 1551.20 | 1504.16 | 1499.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 1513.90 | 1520.03 | 1509.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-01 09:15:00 | 1549.50 | 1520.56 | 1510.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1526.70 | 1536.07 | 1524.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-19 09:15:00 | 1532.10 | 1535.88 | 1524.41 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1524.95 | 1535.77 | 1524.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-19 12:15:00 | 1522.40 | 1535.53 | 1524.40 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1451.70 | 1516.60 | 1516.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 1445.40 | 1506.03 | 1509.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1489.30 | 1487.60 | 1498.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 14:15:00 | 1481.00 | 1487.52 | 1497.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1491.80 | 1485.54 | 1495.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-18 14:15:00 | 1489.40 | 1485.58 | 1495.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1492.20 | 1485.70 | 1495.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-19 10:15:00 | 1481.60 | 1485.65 | 1495.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1463.00 | 1483.54 | 1493.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-08 13:15:00 | 1451.20 | 1477.94 | 1487.25 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-16 09:15:00 | 1484.30 | 1474.64 | 1483.74 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1497.70 | 1472.59 | 1472.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1508.30 | 1473.46 | 1473.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1464.50 | 1476.75 | 1474.76 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 09:15:00 | 1430.10 | 1472.90 | 1472.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 1427.20 | 1472.44 | 1472.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1371.00 | 1364.35 | 1403.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 14:15:00 | 1355.30 | 1364.29 | 1402.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 1390.00 | 1356.72 | 1391.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 13:15:00 | 1392.60 | 1357.41 | 1391.87 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-16 14:15:00 | 1356.25 | 2024-03-26 09:15:00 | 1495.45 | TARGET | 139.20 |
| SELL | 2024-12-19 09:15:00 | 1504.95 | 2025-02-03 10:15:00 | 1481.07 | EXIT_EMA400 | 23.88 |
| BUY | 2025-05-08 12:15:00 | 1482.20 | 2025-05-13 09:15:00 | 1555.62 | TARGET | 73.42 |
| BUY | 2025-09-01 09:15:00 | 1549.50 | 2025-09-19 12:15:00 | 1522.40 | EXIT_EMA400 | -27.10 |
| BUY | 2025-09-19 09:15:00 | 1532.10 | 2025-09-19 12:15:00 | 1522.40 | EXIT_EMA400 | -9.70 |
| SELL | 2025-11-18 14:15:00 | 1489.40 | 2025-11-21 10:15:00 | 1470.38 | TARGET | 19.02 |
| SELL | 2025-11-19 10:15:00 | 1481.60 | 2025-12-09 09:15:00 | 1439.51 | TARGET | 42.09 |
| SELL | 2025-11-13 14:15:00 | 1481.00 | 2025-12-16 09:15:00 | 1484.30 | EXIT_EMA400 | -3.30 |
| SELL | 2025-12-08 13:15:00 | 1451.20 | 2025-12-16 09:15:00 | 1484.30 | EXIT_EMA400 | -33.10 |
| SELL | 2026-04-08 14:15:00 | 1355.30 | 2026-04-17 13:15:00 | 1392.60 | EXIT_EMA400 | -37.30 |
