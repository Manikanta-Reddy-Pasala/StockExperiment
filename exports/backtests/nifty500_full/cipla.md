# Cipla Ltd. (CIPLA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1309.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 14 |
| ENTRY1 | 7 |
| ENTRY2 | 8 |
| EXIT | 7 |

## P&L

- **Trades closed:** 15
- **Trades open at end:** 0
- **Winners / losers:** 11 / 4
- **Target hits / EMA400 exits:** 10 / 5
- **Total realized P&L (per unit):** 355.33
- **Avg P&L per closed trade:** 23.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 12:15:00 | 1375.50 | 1407.34 | 1407.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 13:15:00 | 1369.50 | 1406.96 | 1407.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 1418.75 | 1406.14 | 1406.83 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 1442.10 | 1407.43 | 1407.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 1459.00 | 1408.27 | 1407.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1438.65 | 1438.99 | 1426.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 09:15:00 | 1500.00 | 1440.35 | 1427.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1469.80 | 1495.10 | 1467.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-27 14:15:00 | 1482.05 | 1494.97 | 1467.80 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1478.85 | 1491.87 | 1469.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-04 10:15:00 | 1481.85 | 1491.77 | 1469.97 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1483.90 | 1498.40 | 1480.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-22 09:15:00 | 1491.00 | 1498.20 | 1480.64 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1485.45 | 1497.84 | 1481.31 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-23 14:15:00 | 1497.90 | 1497.79 | 1481.46 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1623.05 | 1631.99 | 1599.24 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-07 09:15:00 | 1640.25 | 1631.99 | 1599.56 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-11 09:15:00 | 1599.15 | 1635.41 | 1605.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1480.60 | 1586.62 | 1586.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1457.85 | 1574.11 | 1580.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 1559.80 | 1557.80 | 1571.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 12:15:00 | 1549.70 | 1566.04 | 1573.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1539.40 | 1525.74 | 1546.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-29 14:15:00 | 1535.10 | 1526.05 | 1546.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 1540.00 | 1525.70 | 1545.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-03 13:15:00 | 1534.05 | 1525.94 | 1545.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1510.25 | 1490.99 | 1513.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-30 10:15:00 | 1523.20 | 1491.31 | 1513.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 1524.50 | 1472.32 | 1472.17 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 1443.55 | 1472.34 | 1472.45 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 1495.95 | 1472.68 | 1472.60 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1403.70 | 1472.22 | 1472.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1381.20 | 1467.97 | 1470.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1467.80 | 1457.10 | 1464.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-11 10:15:00 | 1454.50 | 1457.08 | 1464.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 1454.50 | 1457.08 | 1464.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1470.90 | 1457.40 | 1464.12 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1526.90 | 1469.84 | 1469.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1550.90 | 1476.00 | 1472.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1501.30 | 1501.98 | 1488.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 1529.60 | 1500.34 | 1489.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1503.60 | 1501.53 | 1490.48 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 1477.30 | 1500.99 | 1490.58 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1475.30 | 1493.13 | 1493.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1472.20 | 1492.26 | 1492.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 1488.20 | 1488.02 | 1490.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-25 09:15:00 | 1482.00 | 1487.95 | 1490.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1486.60 | 1487.88 | 1490.29 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-25 13:15:00 | 1541.60 | 1488.41 | 1490.54 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 1567.30 | 1492.86 | 1492.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 1573.40 | 1493.66 | 1493.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1507.00 | 1507.11 | 1500.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-12 13:15:00 | 1520.40 | 1502.76 | 1499.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1541.20 | 1556.95 | 1541.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-23 09:15:00 | 1535.90 | 1556.59 | 1541.16 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1513.00 | 1531.21 | 1531.21 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.40 | 1531.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1561.80 | 1531.98 | 1531.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1525.00 | 1539.70 | 1539.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1518.00 | 1539.13 | 1539.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.20 | 1529.77 | 1533.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-01 09:15:00 | 1524.60 | 1529.74 | 1533.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1524.60 | 1529.74 | 1533.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-01 13:15:00 | 1521.10 | 1529.56 | 1533.54 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1513.40 | 1509.02 | 1517.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1518.40 | 1509.15 | 1517.48 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-07-04 10:15:00 | 1481.85 | 2024-07-08 09:15:00 | 1517.48 | TARGET | 35.63 |
| BUY | 2024-06-27 14:15:00 | 1482.05 | 2024-07-15 09:15:00 | 1524.79 | TARGET | 42.74 |
| BUY | 2024-07-22 09:15:00 | 1491.00 | 2024-07-26 09:15:00 | 1522.08 | TARGET | 31.08 |
| BUY | 2024-07-23 14:15:00 | 1497.90 | 2024-07-26 14:15:00 | 1547.22 | TARGET | 49.32 |
| BUY | 2024-06-05 09:15:00 | 1500.00 | 2024-10-11 09:15:00 | 1599.15 | EXIT_EMA400 | 99.15 |
| BUY | 2024-10-07 09:15:00 | 1640.25 | 2024-10-11 09:15:00 | 1599.15 | EXIT_EMA400 | -41.10 |
| SELL | 2024-11-11 12:15:00 | 1549.70 | 2024-11-18 09:15:00 | 1478.82 | TARGET | 70.88 |
| SELL | 2024-11-29 14:15:00 | 1535.10 | 2024-12-04 14:15:00 | 1500.39 | TARGET | 34.71 |
| SELL | 2024-12-03 13:15:00 | 1534.05 | 2024-12-04 14:15:00 | 1500.28 | TARGET | 33.77 |
| SELL | 2025-04-11 10:15:00 | 1454.50 | 2025-04-15 09:15:00 | 1470.90 | EXIT_EMA400 | -16.40 |
| BUY | 2025-05-13 09:15:00 | 1529.60 | 2025-05-15 09:15:00 | 1477.30 | EXIT_EMA400 | -52.30 |
| SELL | 2025-07-25 09:15:00 | 1482.00 | 2025-07-25 13:15:00 | 1541.60 | EXIT_EMA400 | -59.60 |
| BUY | 2025-08-12 13:15:00 | 1520.40 | 2025-08-21 14:15:00 | 1583.20 | TARGET | 62.80 |
| SELL | 2025-12-01 09:15:00 | 1524.60 | 2025-12-03 11:15:00 | 1497.27 | TARGET | 27.33 |
| SELL | 2025-12-01 13:15:00 | 1521.10 | 2025-12-30 09:15:00 | 1483.78 | TARGET | 37.32 |
