# Sobha Ltd. (SOBHA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1431.00
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
- **Total realized P&L (per unit):** -385.10
- **Avg P&L per closed trade:** -48.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 1718.40 | 1858.76 | 1859.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1693.50 | 1846.78 | 1852.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 11:15:00 | 1760.15 | 1749.47 | 1786.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-03 15:15:00 | 1731.00 | 1749.18 | 1786.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1767.00 | 1737.15 | 1772.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-12 09:15:00 | 1739.80 | 1737.74 | 1772.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-13 09:15:00 | 1799.15 | 1738.80 | 1771.92 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 2002.20 | 1794.98 | 1794.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 14:15:00 | 2018.85 | 1814.45 | 1804.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1834.80 | 1849.32 | 1825.19 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 1718.60 | 1806.71 | 1807.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 1700.10 | 1788.52 | 1797.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 1683.75 | 1683.60 | 1733.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 10:15:00 | 1629.70 | 1680.07 | 1727.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1671.10 | 1632.15 | 1678.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-02 12:15:00 | 1642.60 | 1632.69 | 1677.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1674.45 | 1633.71 | 1677.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 11:15:00 | 1679.00 | 1634.57 | 1677.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 1377.90 | 1268.27 | 1268.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 1390.50 | 1290.98 | 1280.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1487.80 | 1488.48 | 1413.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 11:15:00 | 1537.00 | 1496.94 | 1449.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1544.80 | 1573.19 | 1524.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-07 13:15:00 | 1548.60 | 1572.67 | 1524.67 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-08 09:15:00 | 1507.80 | 1571.69 | 1524.90 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1428.50 | 1505.88 | 1506.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1420.60 | 1505.03 | 1505.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 1496.00 | 1491.08 | 1498.10 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1613.80 | 1504.81 | 1504.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1627.00 | 1517.83 | 1511.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 09:15:00 | 1539.20 | 1542.64 | 1526.98 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 1454.00 | 1515.92 | 1516.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 1451.90 | 1515.29 | 1515.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1537.20 | 1503.60 | 1509.70 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1580.30 | 1514.49 | 1514.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 1601.10 | 1526.19 | 1520.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 1575.70 | 1576.41 | 1551.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-14 15:15:00 | 1588.00 | 1576.53 | 1552.06 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-20 15:15:00 | 1546.00 | 1577.03 | 1555.61 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 1425.90 | 1543.59 | 1543.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 1404.80 | 1531.59 | 1537.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1502.60 | 1497.67 | 1516.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 11:15:00 | 1481.10 | 1497.12 | 1515.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1541.00 | 1487.42 | 1506.35 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-03 15:15:00 | 1731.00 | 2024-09-13 09:15:00 | 1799.15 | EXIT_EMA400 | -68.15 |
| SELL | 2024-09-12 09:15:00 | 1739.80 | 2024-09-13 09:15:00 | 1799.15 | EXIT_EMA400 | -59.35 |
| SELL | 2024-11-08 10:15:00 | 1629.70 | 2024-12-03 11:15:00 | 1679.00 | EXIT_EMA400 | -49.30 |
| SELL | 2024-12-02 12:15:00 | 1642.60 | 2024-12-03 11:15:00 | 1679.00 | EXIT_EMA400 | -36.40 |
| BUY | 2025-07-14 11:15:00 | 1537.00 | 2025-08-08 09:15:00 | 1507.80 | EXIT_EMA400 | -29.20 |
| BUY | 2025-08-07 13:15:00 | 1548.60 | 2025-08-08 09:15:00 | 1507.80 | EXIT_EMA400 | -40.80 |
| BUY | 2025-11-14 15:15:00 | 1588.00 | 2025-11-20 15:15:00 | 1546.00 | EXIT_EMA400 | -42.00 |
| SELL | 2025-12-24 11:15:00 | 1481.10 | 2026-01-05 09:15:00 | 1541.00 | EXIT_EMA400 | -59.90 |
