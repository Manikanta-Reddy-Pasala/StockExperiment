# Sun Pharmaceutical Industries Ltd. (SUNPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1808.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -400.55
- **Avg P&L per closed trade:** -50.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 1446.10 | 1510.41 | 1510.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 12:15:00 | 1443.05 | 1509.73 | 1510.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 12:15:00 | 1498.35 | 1496.88 | 1502.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-20 09:15:00 | 1480.05 | 1502.52 | 1504.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-25 14:15:00 | 1505.50 | 1498.96 | 1502.53 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 1525.40 | 1505.54 | 1505.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1535.65 | 1506.05 | 1505.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 09:15:00 | 1873.30 | 1880.50 | 1824.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-28 13:15:00 | 1896.95 | 1876.18 | 1828.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1837.70 | 1876.01 | 1830.80 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-30 12:15:00 | 1868.80 | 1875.52 | 1831.23 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-11-04 09:15:00 | 1794.65 | 1872.70 | 1832.16 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 1730.00 | 1811.05 | 1811.43 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1850.20 | 1809.01 | 1808.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1859.15 | 1810.34 | 1809.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1831.95 | 1834.06 | 1823.66 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1748.65 | 1815.13 | 1815.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1717.20 | 1806.11 | 1809.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 11:15:00 | 1670.65 | 1668.95 | 1714.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 1649.50 | 1711.89 | 1722.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1695.70 | 1704.68 | 1717.62 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-11 10:15:00 | 1689.15 | 1704.53 | 1717.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1718.20 | 1704.00 | 1716.82 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1818.00 | 1726.01 | 1725.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 10:15:00 | 1829.00 | 1727.04 | 1726.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1681.10 | 1736.05 | 1736.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1670.60 | 1735.40 | 1735.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1719.10 | 1702.91 | 1715.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 13:15:00 | 1691.50 | 1702.91 | 1715.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-30 10:15:00 | 1700.80 | 1683.07 | 1699.30 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1691.90 | 1640.29 | 1640.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1700.80 | 1643.28 | 1641.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 1754.20 | 1771.81 | 1737.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 09:15:00 | 1798.00 | 1749.62 | 1736.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-09 12:15:00 | 1738.60 | 1751.95 | 1738.88 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1613.80 | 1728.49 | 1728.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 13:15:00 | 1605.00 | 1699.82 | 1713.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.10 | 1679.90 | 1701.17 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 1777.80 | 1709.49 | 1709.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 1787.40 | 1712.26 | 1710.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.58 | 1740.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-20 11:15:00 | 1772.30 | 1760.46 | 1741.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-01 12:15:00 | 1740.10 | 1765.97 | 1747.98 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1653.60 | 1734.06 | 1734.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1640.00 | 1713.09 | 1722.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1708.13 | 1719.77 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-20 09:15:00 | 1480.05 | 2024-06-25 14:15:00 | 1505.50 | EXIT_EMA400 | -25.45 |
| BUY | 2024-10-28 13:15:00 | 1896.95 | 2024-11-04 09:15:00 | 1794.65 | EXIT_EMA400 | -102.30 |
| BUY | 2024-10-30 12:15:00 | 1868.80 | 2024-11-04 09:15:00 | 1794.65 | EXIT_EMA400 | -74.15 |
| SELL | 2025-04-07 09:15:00 | 1649.50 | 2025-04-15 09:15:00 | 1718.20 | EXIT_EMA400 | -68.70 |
| SELL | 2025-04-11 10:15:00 | 1689.15 | 2025-04-15 09:15:00 | 1718.20 | EXIT_EMA400 | -29.05 |
| SELL | 2025-06-12 13:15:00 | 1691.50 | 2025-06-30 10:15:00 | 1700.80 | EXIT_EMA400 | -9.30 |
| BUY | 2026-01-07 09:15:00 | 1798.00 | 2026-01-09 12:15:00 | 1738.60 | EXIT_EMA400 | -59.40 |
| BUY | 2026-03-20 11:15:00 | 1772.30 | 2026-04-01 12:15:00 | 1740.10 | EXIT_EMA400 | -32.20 |
