# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2043.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / EMA400 exits:** 6 / 2
- **Total realized P&L (per unit):** 398.40
- **Avg P&L per closed trade:** 49.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 1686.10 | 1873.84 | 1874.46 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 10:15:00 | 1890.70 | 1867.86 | 1867.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 10:15:00 | 1910.40 | 1868.95 | 1868.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 1856.85 | 1869.28 | 1868.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-30 15:15:00 | 1882.55 | 1868.72 | 1868.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 1882.55 | 1868.72 | 1868.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-31 09:15:00 | 1927.20 | 1869.30 | 1868.56 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1871.50 | 1873.08 | 1870.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-04 12:15:00 | 1864.95 | 1873.00 | 1870.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1771.25 | 1867.92 | 1868.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1737.00 | 1856.76 | 1862.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1810.35 | 1804.49 | 1832.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 1762.00 | 1803.53 | 1831.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1815.05 | 1784.71 | 1815.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 10:15:00 | 1818.95 | 1785.05 | 1815.11 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 1869.40 | 1825.31 | 1825.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1883.95 | 1826.28 | 1825.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 1826.10 | 1830.13 | 1827.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 1837.00 | 1830.08 | 1827.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1845.10 | 1839.87 | 1833.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-10 11:15:00 | 1802.70 | 1839.46 | 1832.93 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 1765.00 | 1827.23 | 1827.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1748.40 | 1821.72 | 1824.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 1842.80 | 1817.34 | 1822.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 15:15:00 | 1790.00 | 1816.43 | 1821.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 1673.80 | 1618.11 | 1670.19 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 1666.10 | 1629.36 | 1629.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1677.50 | 1630.24 | 1629.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1680.80 | 1681.47 | 1661.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 1704.70 | 1681.49 | 1661.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1685.00 | 1713.01 | 1682.93 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-30 10:15:00 | 1680.80 | 1712.69 | 1682.92 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1653.10 | 1699.00 | 1699.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 1650.00 | 1695.51 | 1697.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 1685.00 | 1684.78 | 1690.75 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1803.20 | 1694.31 | 1694.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 15:15:00 | 1815.40 | 1700.85 | 1697.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 1743.60 | 1743.69 | 1723.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 11:15:00 | 1750.40 | 1743.02 | 1723.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1854.50 | 1853.90 | 1820.78 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-30 12:15:00 | 1869.60 | 1854.17 | 1821.41 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1998.00 | 2052.07 | 1991.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 1908.60 | 2049.78 | 1991.07 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-30 15:15:00 | 1882.55 | 2024-10-31 09:15:00 | 1925.41 | TARGET | 42.86 |
| BUY | 2024-10-31 09:15:00 | 1927.20 | 2024-11-04 12:15:00 | 1864.95 | EXIT_EMA400 | -62.25 |
| SELL | 2024-11-25 12:15:00 | 1762.00 | 2024-12-04 10:15:00 | 1818.95 | EXIT_EMA400 | -56.95 |
| BUY | 2025-01-07 09:15:00 | 1837.00 | 2025-01-07 12:15:00 | 1864.94 | TARGET | 27.94 |
| SELL | 2025-01-21 15:15:00 | 1790.00 | 2025-01-28 09:15:00 | 1695.75 | TARGET | 94.25 |
| BUY | 2025-06-20 09:15:00 | 1704.70 | 2025-06-25 09:15:00 | 1833.26 | TARGET | 128.56 |
| BUY | 2025-11-24 11:15:00 | 1750.40 | 2025-12-04 15:15:00 | 1829.80 | TARGET | 79.40 |
| BUY | 2026-01-30 12:15:00 | 1869.60 | 2026-02-23 09:15:00 | 2014.18 | TARGET | 144.58 |
