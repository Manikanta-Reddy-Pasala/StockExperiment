# Infosys (INFY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1181.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** 33.32
- **Avg P&L per closed trade:** 5.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 09:15:00 | 1366.55 | 1422.59 | 1422.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 10:15:00 | 1365.60 | 1422.02 | 1422.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 14:15:00 | 1412.15 | 1406.08 | 1413.21 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 14:15:00 | 1448.80 | 1418.88 | 1418.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 1459.30 | 1421.95 | 1420.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 10:15:00 | 1439.35 | 1445.21 | 1434.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-14 09:15:00 | 1483.45 | 1445.51 | 1434.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-05 09:15:00 | 1611.00 | 1654.89 | 1614.86 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 10:15:00 | 1505.90 | 1597.96 | 1598.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 1490.25 | 1594.19 | 1596.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 1453.75 | 1452.22 | 1489.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 14:15:00 | 1450.45 | 1456.52 | 1485.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1471.65 | 1445.30 | 1473.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 1521.80 | 1446.33 | 1473.64 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 1564.80 | 1489.42 | 1489.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 1568.00 | 1490.20 | 1489.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 11:15:00 | 1733.60 | 1734.33 | 1651.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 13:15:00 | 1753.20 | 1734.55 | 1652.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1881.25 | 1918.32 | 1870.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-21 09:15:00 | 1858.70 | 1916.23 | 1870.42 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1823.65 | 1894.83 | 1895.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 1817.65 | 1874.23 | 1882.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1519.10 | 1516.96 | 1605.78 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 1606.30 | 1591.66 | 1591.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 1641.40 | 1592.60 | 1592.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1594.10 | 1605.27 | 1599.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 12:15:00 | 1608.40 | 1601.14 | 1597.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-17 09:15:00 | 1596.10 | 1601.28 | 1597.91 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.20 | 1594.51 | 1594.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.50 | 1495.72 | 1531.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-29 09:15:00 | 1478.30 | 1500.09 | 1528.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1524.00 | 1489.09 | 1515.15 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.20 | 1500.97 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.57 | 1503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.00 | 1606.82 | 1574.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 13:15:00 | 1639.50 | 1608.38 | 1578.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-04 09:15:00 | 1550.60 | 1635.55 | 1608.84 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.00 | 1587.61 | 1587.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1585.74 | 1586.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.00 | 1311.02 | 1382.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-10 09:15:00 | 1289.30 | 1314.96 | 1375.39 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-14 09:15:00 | 1483.45 | 2024-01-15 09:15:00 | 1629.52 | TARGET | 146.07 |
| SELL | 2024-05-29 14:15:00 | 1450.45 | 2024-06-07 09:15:00 | 1521.80 | EXIT_EMA400 | -71.35 |
| BUY | 2024-08-05 13:15:00 | 1753.20 | 2024-10-21 09:15:00 | 1858.70 | EXIT_EMA400 | 105.50 |
| BUY | 2025-07-16 12:15:00 | 1608.40 | 2025-07-17 09:15:00 | 1596.10 | EXIT_EMA400 | -12.30 |
| SELL | 2025-08-29 09:15:00 | 1478.30 | 2025-09-10 09:15:00 | 1524.00 | EXIT_EMA400 | -45.70 |
| BUY | 2026-01-07 13:15:00 | 1639.50 | 2026-02-04 09:15:00 | 1550.60 | EXIT_EMA400 | -88.90 |
