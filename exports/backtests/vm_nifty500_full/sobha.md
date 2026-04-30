# Sobha Ltd. (SOBHA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1433.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -362.40
- **Avg P&L per closed trade:** -45.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 14:15:00 | 1718.60 | 1854.21 | 1854.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 15:15:00 | 1715.10 | 1852.83 | 1853.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 11:15:00 | 1760.15 | 1750.92 | 1786.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-03 15:15:00 | 1731.00 | 1750.58 | 1786.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1768.95 | 1738.12 | 1772.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-12 09:15:00 | 1739.80 | 1738.69 | 1772.65 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-13 09:15:00 | 1799.15 | 1739.67 | 1771.97 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 2003.95 | 1795.39 | 1794.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 14:15:00 | 2016.90 | 1814.89 | 1804.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1834.80 | 1850.01 | 1825.40 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 1719.00 | 1807.22 | 1807.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 1672.95 | 1787.78 | 1796.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 1689.45 | 1685.63 | 1734.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-06 14:15:00 | 1665.90 | 1685.30 | 1733.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1671.10 | 1632.94 | 1679.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-02 12:15:00 | 1642.60 | 1633.45 | 1678.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1674.45 | 1634.44 | 1678.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 11:15:00 | 1679.00 | 1635.29 | 1678.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1371.20 | 1269.66 | 1269.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1389.00 | 1272.82 | 1270.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1487.50 | 1488.74 | 1413.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 11:15:00 | 1537.00 | 1497.03 | 1450.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1544.80 | 1573.10 | 1524.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-07 13:15:00 | 1548.60 | 1572.58 | 1524.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-08 09:15:00 | 1507.10 | 1571.60 | 1524.97 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1428.50 | 1505.85 | 1506.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1420.60 | 1505.00 | 1505.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 1496.00 | 1490.96 | 1498.08 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1613.80 | 1504.69 | 1504.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1627.00 | 1517.72 | 1511.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 09:15:00 | 1539.20 | 1542.52 | 1526.92 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 1455.40 | 1516.45 | 1516.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 1451.90 | 1515.19 | 1515.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1537.90 | 1503.64 | 1509.70 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1580.30 | 1514.63 | 1514.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 1601.10 | 1526.31 | 1520.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 1575.70 | 1576.50 | 1551.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-17 09:15:00 | 1604.00 | 1576.74 | 1552.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-20 15:15:00 | 1550.00 | 1577.05 | 1555.64 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 1426.90 | 1543.75 | 1544.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 1404.80 | 1531.75 | 1537.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1502.60 | 1497.58 | 1516.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 11:15:00 | 1481.10 | 1497.02 | 1515.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1541.10 | 1487.33 | 1506.32 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-03 15:15:00 | 1731.00 | 2024-09-13 09:15:00 | 1799.15 | EXIT_EMA400 | -68.15 |
| SELL | 2024-09-12 09:15:00 | 1739.80 | 2024-09-13 09:15:00 | 1799.15 | EXIT_EMA400 | -59.35 |
| SELL | 2024-11-06 14:15:00 | 1665.90 | 2024-12-03 11:15:00 | 1679.00 | EXIT_EMA400 | -13.10 |
| SELL | 2024-12-02 12:15:00 | 1642.60 | 2024-12-03 11:15:00 | 1679.00 | EXIT_EMA400 | -36.40 |
| BUY | 2025-07-14 11:15:00 | 1537.00 | 2025-08-08 09:15:00 | 1507.10 | EXIT_EMA400 | -29.90 |
| BUY | 2025-08-07 13:15:00 | 1548.60 | 2025-08-08 09:15:00 | 1507.10 | EXIT_EMA400 | -41.50 |
| BUY | 2025-11-17 09:15:00 | 1604.00 | 2025-11-20 15:15:00 | 1550.00 | EXIT_EMA400 | -54.00 |
| SELL | 2025-12-24 11:15:00 | 1481.10 | 2026-01-05 09:15:00 | 1541.10 | EXIT_EMA400 | -60.00 |
