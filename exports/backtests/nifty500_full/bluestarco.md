# Blue Star Ltd. (BLUESTARCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1781.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -260.15
- **Avg P&L per closed trade:** -43.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 14:15:00 | 807.60 | 755.90 | 755.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 10:15:00 | 823.85 | 766.00 | 761.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 15:15:00 | 860.70 | 871.97 | 836.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-25 14:15:00 | 890.05 | 872.07 | 837.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-20 13:15:00 | 937.10 | 973.48 | 939.00 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 1772.00 | 1957.76 | 1958.40 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 2080.25 | 1948.86 | 1948.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 2126.90 | 1951.92 | 1950.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 2079.95 | 2088.06 | 2035.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-02 14:15:00 | 2122.60 | 2088.91 | 2038.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2064.00 | 2088.61 | 2040.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-04 11:15:00 | 2032.80 | 2088.06 | 2040.95 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 1743.60 | 2013.53 | 2013.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 1731.00 | 2010.72 | 2012.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 13:15:00 | 1635.00 | 1632.03 | 1731.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 11:15:00 | 1620.80 | 1631.99 | 1729.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1688.50 | 1635.43 | 1695.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-01 14:15:00 | 1700.20 | 1636.66 | 1695.23 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1873.60 | 1734.90 | 1734.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1901.10 | 1739.01 | 1736.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1757.30 | 1759.65 | 1747.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-24 13:15:00 | 1769.80 | 1758.81 | 1748.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 14:15:00 | 1747.10 | 1759.28 | 1748.82 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1776.00 | 1879.14 | 1879.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1772.70 | 1877.11 | 1878.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1787.00 | 1784.38 | 1817.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-19 12:15:00 | 1773.00 | 1794.25 | 1817.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1793.90 | 1776.66 | 1801.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 10:15:00 | 1822.00 | 1777.11 | 1801.51 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1954.90 | 1799.98 | 1799.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 11:15:00 | 1996.90 | 1833.42 | 1817.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 1903.50 | 1904.23 | 1865.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 12:15:00 | 1932.20 | 1900.86 | 1866.54 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 11:15:00 | 1865.90 | 1904.34 | 1870.51 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1626.40 | 1852.04 | 1852.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 1611.00 | 1849.64 | 1851.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 10:15:00 | 1726.50 | 1724.00 | 1776.02 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-25 14:15:00 | 890.05 | 2023-12-20 13:15:00 | 937.10 | EXIT_EMA400 | 47.05 |
| BUY | 2025-04-02 14:15:00 | 2122.60 | 2025-04-04 11:15:00 | 2032.80 | EXIT_EMA400 | -89.80 |
| SELL | 2025-06-12 11:15:00 | 1620.80 | 2025-07-01 14:15:00 | 1700.20 | EXIT_EMA400 | -79.40 |
| BUY | 2025-07-24 13:15:00 | 1769.80 | 2025-07-25 14:15:00 | 1747.10 | EXIT_EMA400 | -22.70 |
| SELL | 2025-12-19 12:15:00 | 1773.00 | 2026-01-02 10:15:00 | 1822.00 | EXIT_EMA400 | -49.00 |
| BUY | 2026-03-05 12:15:00 | 1932.20 | 2026-03-09 11:15:00 | 1865.90 | EXIT_EMA400 | -66.30 |
