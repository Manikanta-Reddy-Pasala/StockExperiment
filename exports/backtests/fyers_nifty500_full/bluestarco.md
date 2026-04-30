# Blue Star Ltd. (BLUESTARCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1781.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -310.55
- **Avg P&L per closed trade:** -62.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 1772.00 | 1957.63 | 1958.52 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 2080.25 | 1949.27 | 1948.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 2136.50 | 1954.13 | 1951.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 2079.95 | 2088.20 | 2035.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-02 14:15:00 | 2120.85 | 2089.00 | 2038.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2064.00 | 2088.65 | 2041.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-04 11:15:00 | 2032.80 | 2088.09 | 2040.99 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 1743.60 | 2013.53 | 2013.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 1731.00 | 2010.71 | 2012.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 1636.00 | 1632.28 | 1733.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-13 09:15:00 | 1616.40 | 1631.78 | 1726.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1688.50 | 1635.54 | 1695.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-01 14:15:00 | 1700.20 | 1636.77 | 1695.33 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1873.60 | 1734.88 | 1734.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1901.10 | 1738.99 | 1736.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1748.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-24 13:15:00 | 1769.80 | 1758.78 | 1748.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 14:15:00 | 1747.10 | 1759.26 | 1748.84 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1776.00 | 1879.36 | 1879.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1772.70 | 1877.32 | 1878.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1787.00 | 1784.45 | 1817.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-19 12:15:00 | 1773.00 | 1794.32 | 1817.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1793.90 | 1776.68 | 1801.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-02 10:15:00 | 1822.00 | 1777.13 | 1801.58 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1945.00 | 1800.33 | 1799.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1954.90 | 1801.87 | 1800.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 1903.50 | 1905.01 | 1865.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 12:15:00 | 1933.00 | 1901.49 | 1866.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 11:15:00 | 1866.00 | 1904.95 | 1870.88 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1627.00 | 1852.28 | 1852.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 1602.00 | 1849.79 | 1851.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 10:15:00 | 1726.50 | 1723.92 | 1776.05 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-02 14:15:00 | 2120.85 | 2025-04-04 11:15:00 | 2032.80 | EXIT_EMA400 | -88.05 |
| SELL | 2025-06-13 09:15:00 | 1616.40 | 2025-07-01 14:15:00 | 1700.20 | EXIT_EMA400 | -83.80 |
| BUY | 2025-07-24 13:15:00 | 1769.80 | 2025-07-25 14:15:00 | 1747.10 | EXIT_EMA400 | -22.70 |
| SELL | 2025-12-19 12:15:00 | 1773.00 | 2026-01-02 10:15:00 | 1822.00 | EXIT_EMA400 | -49.00 |
| BUY | 2026-03-05 12:15:00 | 1933.00 | 2026-03-09 11:15:00 | 1866.00 | EXIT_EMA400 | -67.00 |
