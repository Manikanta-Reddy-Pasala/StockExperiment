# Prestige Estates Projects Ltd. (PRESTIGE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1409.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -284.15
- **Avg P&L per closed trade:** -40.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 1688.00 | 1786.10 | 1786.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1669.00 | 1784.94 | 1785.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1685.10 | 1658.56 | 1704.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 09:15:00 | 1644.00 | 1664.76 | 1704.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1689.05 | 1662.04 | 1698.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 14:15:00 | 1730.75 | 1662.73 | 1698.72 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 1854.25 | 1720.50 | 1720.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 12:15:00 | 1884.85 | 1723.42 | 1721.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 11:15:00 | 1728.10 | 1742.37 | 1731.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-23 14:15:00 | 1758.45 | 1742.50 | 1732.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-26 09:15:00 | 1717.05 | 1743.32 | 1732.96 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 1657.95 | 1725.39 | 1725.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1650.00 | 1722.00 | 1723.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 1212.45 | 1210.49 | 1316.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 14:15:00 | 1197.65 | 1214.47 | 1303.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 1245.40 | 1178.22 | 1246.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 12:15:00 | 1249.00 | 1178.93 | 1246.93 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 1389.40 | 1284.38 | 1284.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1394.40 | 1287.55 | 1285.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 1610.50 | 1612.98 | 1519.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 10:15:00 | 1628.80 | 1612.81 | 1523.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1642.00 | 1683.90 | 1606.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-28 12:15:00 | 1606.10 | 1681.98 | 1606.55 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1555.00 | 1599.25 | 1599.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 1551.60 | 1596.32 | 1597.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1638.80 | 1598.77 | 1598.74 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 1522.90 | 1598.72 | 1598.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1521.10 | 1595.25 | 1597.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1710.00 | 1591.53 | 1591.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 1720.20 | 1594.03 | 1592.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 1695.80 | 1696.71 | 1658.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-13 09:15:00 | 1758.80 | 1697.33 | 1658.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1676.60 | 1707.46 | 1673.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 10:15:00 | 1670.80 | 1707.09 | 1673.24 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1598.80 | 1659.09 | 1659.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1588.90 | 1642.07 | 1649.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1638.00 | 1630.95 | 1642.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 12:15:00 | 1613.80 | 1634.08 | 1643.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1540.00 | 1510.20 | 1561.52 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 10:15:00 | 1509.00 | 1512.25 | 1560.56 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1555.70 | 1514.07 | 1557.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 1591.90 | 1515.26 | 1557.50 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-27 09:15:00 | 1644.00 | 2024-12-02 14:15:00 | 1730.75 | EXIT_EMA400 | -86.75 |
| BUY | 2024-12-23 14:15:00 | 1758.45 | 2024-12-26 09:15:00 | 1717.05 | EXIT_EMA400 | -41.40 |
| SELL | 2025-03-26 14:15:00 | 1197.65 | 2025-04-21 12:15:00 | 1249.00 | EXIT_EMA400 | -51.35 |
| BUY | 2025-07-03 10:15:00 | 1628.80 | 2025-07-28 12:15:00 | 1606.10 | EXIT_EMA400 | -22.70 |
| BUY | 2025-11-13 09:15:00 | 1758.80 | 2025-11-24 10:15:00 | 1670.80 | EXIT_EMA400 | -88.00 |
| SELL | 2026-01-07 12:15:00 | 1613.80 | 2026-01-12 09:15:00 | 1524.85 | TARGET | 88.95 |
| SELL | 2026-02-04 10:15:00 | 1509.00 | 2026-02-09 09:15:00 | 1591.90 | EXIT_EMA400 | -82.90 |
