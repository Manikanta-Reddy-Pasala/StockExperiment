# Gland Pharma Ltd. (GLAND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1742.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 6 |
| EXIT | 5 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -395.64
- **Avg P&L per closed trade:** -35.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 1844.45 | 1914.86 | 1914.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 1841.40 | 1910.86 | 1912.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1785.50 | 1696.50 | 1762.58 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1839.70 | 1775.36 | 1775.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1900.05 | 1779.73 | 1777.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1799.60 | 1804.77 | 1791.54 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 1664.90 | 1779.99 | 1780.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1639.10 | 1754.56 | 1766.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 1551.60 | 1545.01 | 1614.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1489.70 | 1582.23 | 1600.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-20 09:15:00 | 1498.60 | 1458.81 | 1497.73 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1624.30 | 1520.47 | 1520.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 1626.80 | 1522.51 | 1521.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 1932.40 | 1935.96 | 1834.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-07 11:15:00 | 1968.70 | 1936.28 | 1835.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1900.80 | 1937.42 | 1874.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-29 12:15:00 | 1873.20 | 1935.68 | 1875.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 1849.00 | 1918.94 | 1919.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1838.10 | 1917.46 | 1918.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1721.50 | 1718.61 | 1776.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-01 09:15:00 | 1696.60 | 1718.43 | 1774.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1734.80 | 1713.25 | 1763.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-08 10:15:00 | 1713.10 | 1713.25 | 1762.77 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1752.10 | 1711.20 | 1754.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-16 09:15:00 | 1736.90 | 1711.81 | 1754.11 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-29 09:15:00 | 1778.00 | 1703.01 | 1738.81 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 1836.40 | 1766.11 | 1766.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1842.60 | 1766.87 | 1766.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-16 15:15:00 | 1815.00 | 1791.14 | 1780.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1800.00 | 1802.66 | 1788.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-24 14:15:00 | 1803.30 | 1802.63 | 1788.86 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 1783.60 | 1808.92 | 1793.67 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 1681.40 | 1780.28 | 1780.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1669.70 | 1767.50 | 1774.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1718.00 | 1701.33 | 1733.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 1681.60 | 1701.57 | 1732.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1707.00 | 1701.88 | 1732.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-30 09:15:00 | 1673.30 | 1701.59 | 1731.70 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1721.60 | 1701.81 | 1730.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-02 09:15:00 | 1679.30 | 1702.32 | 1730.04 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1723.80 | 1701.63 | 1728.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-07 09:15:00 | 1682.10 | 1701.47 | 1727.72 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1724.00 | 1702.13 | 1727.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 10:15:00 | 1731.20 | 1702.42 | 1727.17 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 1489.70 | 2025-05-20 09:15:00 | 1498.60 | EXIT_EMA400 | -8.90 |
| BUY | 2025-08-07 11:15:00 | 1968.70 | 2025-08-29 12:15:00 | 1873.20 | EXIT_EMA400 | -95.50 |
| SELL | 2026-01-16 09:15:00 | 1736.90 | 2026-01-19 13:15:00 | 1685.27 | TARGET | 51.63 |
| SELL | 2026-01-01 09:15:00 | 1696.60 | 2026-01-29 09:15:00 | 1778.00 | EXIT_EMA400 | -81.40 |
| SELL | 2026-01-08 10:15:00 | 1713.10 | 2026-01-29 09:15:00 | 1778.00 | EXIT_EMA400 | -64.90 |
| BUY | 2026-02-24 14:15:00 | 1803.30 | 2026-02-25 14:15:00 | 1846.63 | TARGET | 43.33 |
| BUY | 2026-02-16 15:15:00 | 1815.00 | 2026-03-02 09:15:00 | 1783.60 | EXIT_EMA400 | -31.40 |
| SELL | 2026-03-27 09:15:00 | 1681.60 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -49.60 |
| SELL | 2026-03-30 09:15:00 | 1673.30 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -57.90 |
| SELL | 2026-04-02 09:15:00 | 1679.30 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -51.90 |
| SELL | 2026-04-07 09:15:00 | 1682.10 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -49.10 |
