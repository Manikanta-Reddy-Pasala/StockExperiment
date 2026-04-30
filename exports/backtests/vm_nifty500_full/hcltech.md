# HCL Technologies Ltd. (HCLTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1199.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 461.16
- **Avg P&L per closed trade:** 92.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 1447.00 | 1557.79 | 1558.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 09:15:00 | 1385.00 | 1531.85 | 1543.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1373.35 | 1369.43 | 1421.38 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 1517.35 | 1438.59 | 1438.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 1521.70 | 1439.41 | 1438.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1555.00 | 1556.05 | 1513.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 1597.40 | 1556.70 | 1515.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-31 14:15:00 | 1762.70 | 1818.74 | 1766.54 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 1723.60 | 1862.22 | 1862.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 1710.55 | 1860.71 | 1861.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 09:15:00 | 1643.55 | 1615.08 | 1681.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 15:15:00 | 1620.00 | 1615.90 | 1679.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1569.30 | 1510.66 | 1588.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 10:15:00 | 1599.00 | 1511.54 | 1588.13 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1650.00 | 1606.86 | 1606.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 15:15:00 | 1657.30 | 1609.21 | 1608.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1691.60 | 1693.05 | 1666.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 12:15:00 | 1714.50 | 1693.38 | 1666.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 1658.40 | 1692.44 | 1667.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.35 | 1649.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.40 | 1647.01 | 1648.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.00 | 1476.57 | 1514.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 12:15:00 | 1470.50 | 1478.10 | 1512.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.00 | 1443.08 | 1478.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-09 10:15:00 | 1481.00 | 1443.45 | 1478.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1535.20 | 1495.77 | 1495.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1539.50 | 1496.20 | 1495.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1498.10 | 1501.14 | 1498.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 09:15:00 | 1532.60 | 1501.97 | 1499.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-05 09:15:00 | 1599.70 | 1637.20 | 1606.64 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1458.00 | 1623.33 | 1624.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 1452.80 | 1616.72 | 1620.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.05 | 1456.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 09:15:00 | 1305.50 | 1416.72 | 1450.05 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-08-06 09:15:00 | 1597.40 | 2024-10-11 09:15:00 | 1844.61 | TARGET | 247.21 |
| SELL | 2025-03-25 15:15:00 | 1620.00 | 2025-04-04 09:15:00 | 1440.04 | TARGET | 179.96 |
| BUY | 2025-07-08 12:15:00 | 1714.50 | 2025-07-10 09:15:00 | 1658.40 | EXIT_EMA400 | -56.10 |
| SELL | 2025-09-19 12:15:00 | 1470.50 | 2025-10-09 10:15:00 | 1481.00 | EXIT_EMA400 | -10.50 |
| BUY | 2025-11-10 09:15:00 | 1532.60 | 2025-11-19 10:15:00 | 1633.19 | TARGET | 100.59 |
