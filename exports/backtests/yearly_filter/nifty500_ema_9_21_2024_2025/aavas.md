# Aavas Financiers Ltd. (AAVAS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1446.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 136 |
| ALERT1 | 85 |
| ALERT2 | 84 |
| ALERT2_SKIP | 50 |
| ALERT3 | 222 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 127 |
| PARTIAL | 17 |
| TARGET_HIT | 19 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 146 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 60 / 86
- **Target hits / Stop hits / Partials:** 19 / 110 / 17
- **Avg / median % per leg:** 1.58% / -0.71%
- **Sum % (uncompounded):** 230.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 23 | 37.1% | 14 | 48 | 0 | 1.39% | 86.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.25% | -5.0% |
| BUY @ 3rd Alert (retest2) | 58 | 23 | 39.7% | 14 | 44 | 0 | 1.57% | 91.1% |
| SELL (all) | 84 | 37 | 44.0% | 5 | 62 | 17 | 1.72% | 144.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.26% | -0.3% |
| SELL @ 3rd Alert (retest2) | 83 | 37 | 44.6% | 5 | 61 | 17 | 1.74% | 144.5% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.05% | -5.3% |
| retest2 (combined) | 141 | 60 | 42.6% | 19 | 105 | 17 | 1.67% | 235.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 1574.95 | 1581.96 | 1582.08 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 1596.15 | 1584.80 | 1583.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 1599.00 | 1587.64 | 1584.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 14:15:00 | 1583.95 | 1588.26 | 1585.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 14:15:00 | 1583.95 | 1588.26 | 1585.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1583.95 | 1588.26 | 1585.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 1583.95 | 1588.26 | 1585.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1580.00 | 1586.61 | 1585.13 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 09:15:00 | 1570.90 | 1583.47 | 1583.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 11:15:00 | 1564.70 | 1577.64 | 1581.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 1583.15 | 1571.00 | 1575.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 1583.15 | 1571.00 | 1575.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 1583.15 | 1571.00 | 1575.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:30:00 | 1591.50 | 1571.00 | 1575.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 1570.75 | 1570.95 | 1575.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 11:30:00 | 1565.25 | 1569.05 | 1573.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 12:00:00 | 1561.45 | 1569.05 | 1573.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 10:15:00 | 1582.05 | 1576.91 | 1576.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 10:15:00 | 1582.05 | 1576.91 | 1576.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 11:15:00 | 1593.85 | 1580.30 | 1577.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1582.75 | 1584.98 | 1581.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 1582.75 | 1584.98 | 1581.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1582.75 | 1584.98 | 1581.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:00:00 | 1582.75 | 1584.98 | 1581.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1579.35 | 1583.86 | 1581.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:15:00 | 1585.10 | 1583.86 | 1581.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 1590.00 | 1585.08 | 1582.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 1592.15 | 1585.08 | 1582.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 1598.10 | 1590.88 | 1586.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:15:00 | 1601.85 | 1586.81 | 1585.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 1609.95 | 1618.87 | 1620.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1609.95 | 1618.87 | 1620.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 1591.90 | 1610.59 | 1615.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 11:15:00 | 1579.95 | 1577.16 | 1588.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 12:00:00 | 1579.95 | 1577.16 | 1588.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1565.35 | 1570.90 | 1581.16 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 1626.00 | 1586.24 | 1584.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1633.05 | 1595.60 | 1588.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1592.50 | 1615.02 | 1605.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1592.50 | 1615.02 | 1605.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1592.50 | 1615.02 | 1605.27 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1521.35 | 1596.28 | 1597.64 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1640.05 | 1573.69 | 1572.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 1658.90 | 1630.74 | 1608.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 13:15:00 | 1846.75 | 1881.93 | 1849.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 13:15:00 | 1846.75 | 1881.93 | 1849.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 1846.75 | 1881.93 | 1849.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:45:00 | 1856.75 | 1881.93 | 1849.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1846.00 | 1874.75 | 1849.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:30:00 | 1851.55 | 1874.75 | 1849.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 1844.00 | 1868.60 | 1849.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 1852.95 | 1868.60 | 1849.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 13:00:00 | 1847.95 | 1853.80 | 1847.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 1891.30 | 1898.41 | 1899.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 1891.30 | 1898.41 | 1899.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 1886.20 | 1895.96 | 1898.13 | Break + close below crossover candle low |

### Cycle 10 — BUY (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 14:15:00 | 1917.65 | 1900.28 | 1899.71 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 12:15:00 | 1879.50 | 1896.52 | 1898.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 1869.65 | 1884.44 | 1891.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 1869.45 | 1862.01 | 1872.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 1869.45 | 1862.01 | 1872.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1869.45 | 1862.01 | 1872.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 1869.00 | 1862.01 | 1872.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1870.00 | 1863.61 | 1872.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:30:00 | 1870.85 | 1863.61 | 1872.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1870.30 | 1864.95 | 1871.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:15:00 | 1870.00 | 1864.95 | 1871.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 1874.00 | 1866.76 | 1872.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 1874.00 | 1866.76 | 1872.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 1870.80 | 1867.57 | 1872.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 1900.15 | 1867.57 | 1872.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1876.00 | 1869.25 | 1872.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 1853.00 | 1866.40 | 1870.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:00:00 | 1857.35 | 1864.59 | 1869.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:30:00 | 1856.05 | 1860.83 | 1866.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 1851.15 | 1863.93 | 1867.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1845.60 | 1860.26 | 1865.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 12:15:00 | 1869.20 | 1863.25 | 1862.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1869.20 | 1863.25 | 1862.87 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 1846.30 | 1859.86 | 1861.36 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 1875.00 | 1862.19 | 1861.94 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 14:15:00 | 1858.25 | 1864.29 | 1864.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 09:15:00 | 1819.10 | 1854.78 | 1860.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 10:15:00 | 1804.65 | 1801.89 | 1823.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 11:00:00 | 1804.65 | 1801.89 | 1823.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1836.50 | 1808.81 | 1824.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:00:00 | 1836.50 | 1808.81 | 1824.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1781.90 | 1803.43 | 1820.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:00:00 | 1781.05 | 1795.88 | 1814.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:15:00 | 1780.00 | 1791.73 | 1807.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 1811.65 | 1802.64 | 1802.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 1811.65 | 1802.64 | 1802.55 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 1794.00 | 1802.98 | 1803.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 1786.05 | 1792.64 | 1796.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 14:15:00 | 1785.40 | 1784.78 | 1788.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 1785.40 | 1784.78 | 1788.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1791.25 | 1784.99 | 1788.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 11:00:00 | 1764.45 | 1779.30 | 1783.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 1768.80 | 1776.71 | 1781.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1752.75 | 1776.73 | 1780.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 1792.90 | 1761.71 | 1761.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 1792.90 | 1761.71 | 1761.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 1804.15 | 1775.04 | 1767.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 1782.00 | 1785.81 | 1775.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 1782.00 | 1785.81 | 1775.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1782.00 | 1785.81 | 1775.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1788.60 | 1785.81 | 1775.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1776.35 | 1783.92 | 1775.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1780.55 | 1783.92 | 1775.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1782.35 | 1783.60 | 1775.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 1799.30 | 1782.94 | 1776.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 1761.05 | 1774.99 | 1774.52 | SL hit (close<static) qty=1.00 sl=1762.75 alert=retest2 |

### Cycle 19 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 1759.00 | 1771.79 | 1773.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 1736.05 | 1761.64 | 1767.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 13:15:00 | 1767.90 | 1758.89 | 1764.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 13:15:00 | 1767.90 | 1758.89 | 1764.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 1767.90 | 1758.89 | 1764.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 1767.90 | 1758.89 | 1764.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1774.60 | 1762.03 | 1765.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 1774.60 | 1762.03 | 1765.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1746.30 | 1758.48 | 1762.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 1750.40 | 1758.48 | 1762.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1737.10 | 1746.07 | 1753.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 1728.00 | 1737.44 | 1745.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 1724.70 | 1734.18 | 1742.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 1641.60 | 1672.56 | 1683.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 1638.46 | 1672.56 | 1683.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 1653.50 | 1642.23 | 1658.82 | SL hit (close>ema200) qty=0.50 sl=1642.23 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1654.00 | 1641.42 | 1640.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 1743.00 | 1665.92 | 1653.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1708.80 | 1715.52 | 1695.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:00:00 | 1708.80 | 1715.52 | 1695.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1700.00 | 1709.06 | 1697.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1689.10 | 1709.06 | 1697.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1665.35 | 1700.32 | 1694.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 1665.35 | 1700.32 | 1694.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1686.50 | 1697.55 | 1693.53 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 1680.95 | 1689.35 | 1690.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 1670.15 | 1683.52 | 1687.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 1682.60 | 1679.86 | 1684.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 12:00:00 | 1682.60 | 1679.86 | 1684.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 1697.00 | 1683.29 | 1685.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 1697.00 | 1683.29 | 1685.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 1680.15 | 1682.66 | 1685.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 1689.20 | 1682.66 | 1685.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1669.70 | 1679.24 | 1682.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 1674.95 | 1679.24 | 1682.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1682.55 | 1670.22 | 1674.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 1681.00 | 1670.22 | 1674.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1672.15 | 1670.60 | 1674.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:30:00 | 1685.80 | 1670.60 | 1674.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1677.80 | 1672.04 | 1674.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 1677.40 | 1672.04 | 1674.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 1686.45 | 1674.92 | 1675.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 1685.70 | 1674.92 | 1675.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 13:15:00 | 1685.80 | 1677.10 | 1676.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 1699.90 | 1683.14 | 1679.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 1695.25 | 1697.38 | 1690.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 12:00:00 | 1695.25 | 1697.38 | 1690.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1688.25 | 1694.78 | 1690.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 1688.25 | 1694.78 | 1690.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1691.10 | 1694.04 | 1690.54 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1682.55 | 1688.96 | 1689.57 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1708.95 | 1692.00 | 1690.79 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 14:15:00 | 1710.80 | 1713.71 | 1714.00 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 1723.05 | 1714.96 | 1714.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 1751.05 | 1728.15 | 1721.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 1848.85 | 1854.89 | 1819.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-09 10:00:00 | 1848.85 | 1854.89 | 1819.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1818.20 | 1837.28 | 1827.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 1818.20 | 1837.28 | 1827.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 1819.65 | 1833.76 | 1827.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 1830.00 | 1827.05 | 1825.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 1805.15 | 1824.26 | 1824.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 1805.15 | 1824.26 | 1824.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 10:15:00 | 1796.00 | 1818.61 | 1821.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 1799.65 | 1793.40 | 1804.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 10:15:00 | 1799.65 | 1793.40 | 1804.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1799.65 | 1793.40 | 1804.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 1795.00 | 1793.40 | 1804.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1802.60 | 1795.24 | 1804.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 1804.25 | 1795.24 | 1804.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 1803.50 | 1796.89 | 1804.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 1803.50 | 1796.89 | 1804.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 1803.95 | 1798.31 | 1804.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:30:00 | 1798.20 | 1798.31 | 1804.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1795.00 | 1797.64 | 1803.30 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1843.25 | 1806.82 | 1806.50 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 1802.95 | 1815.97 | 1817.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1784.95 | 1808.65 | 1813.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 1823.60 | 1807.86 | 1812.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 11:15:00 | 1823.60 | 1807.86 | 1812.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 1823.60 | 1807.86 | 1812.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 1823.60 | 1807.86 | 1812.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 1830.60 | 1812.41 | 1813.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:30:00 | 1827.30 | 1812.41 | 1813.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 1829.00 | 1816.98 | 1815.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 10:15:00 | 1834.65 | 1823.29 | 1819.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 1897.75 | 1901.53 | 1885.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 1897.75 | 1901.53 | 1885.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1876.35 | 1895.28 | 1885.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1876.35 | 1895.28 | 1885.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1870.05 | 1890.23 | 1884.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 1871.95 | 1890.23 | 1884.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1880.00 | 1888.94 | 1884.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 1880.00 | 1888.94 | 1884.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1881.30 | 1887.41 | 1884.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 1887.95 | 1887.41 | 1884.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 1889.45 | 1887.82 | 1884.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:00:00 | 1888.90 | 1888.38 | 1885.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:00:00 | 1896.50 | 1889.38 | 1886.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1887.40 | 1888.98 | 1886.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:15:00 | 1882.60 | 1888.98 | 1886.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1882.60 | 1887.70 | 1886.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1883.00 | 1887.70 | 1886.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1883.60 | 1886.88 | 1886.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 1881.45 | 1885.80 | 1885.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 1881.45 | 1885.80 | 1885.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 1867.90 | 1881.45 | 1883.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1853.25 | 1828.17 | 1843.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1853.25 | 1828.17 | 1843.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1853.25 | 1828.17 | 1843.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 1853.90 | 1828.17 | 1843.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1856.00 | 1833.73 | 1844.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 1866.10 | 1833.73 | 1844.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 1843.05 | 1838.65 | 1845.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:30:00 | 1857.00 | 1838.65 | 1845.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 1850.65 | 1841.05 | 1845.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 1850.65 | 1841.05 | 1845.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1851.05 | 1843.05 | 1846.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 1851.05 | 1843.05 | 1846.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1847.00 | 1843.84 | 1846.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1811.40 | 1843.84 | 1846.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1720.83 | 1765.93 | 1788.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1736.75 | 1736.02 | 1759.10 | SL hit (close>ema200) qty=0.50 sl=1736.02 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 1733.05 | 1728.50 | 1728.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 12:15:00 | 1753.50 | 1739.06 | 1733.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 1737.20 | 1752.24 | 1743.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 1737.20 | 1752.24 | 1743.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1737.20 | 1752.24 | 1743.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 1735.40 | 1752.24 | 1743.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 1738.10 | 1749.41 | 1742.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:30:00 | 1736.70 | 1749.41 | 1742.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 1738.45 | 1745.59 | 1742.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:30:00 | 1737.10 | 1745.59 | 1742.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1742.05 | 1744.88 | 1742.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 1748.75 | 1745.66 | 1742.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 1730.15 | 1741.25 | 1741.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 1730.15 | 1741.25 | 1741.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1721.00 | 1737.20 | 1739.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 1730.05 | 1729.51 | 1733.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 1730.05 | 1729.51 | 1733.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1730.05 | 1729.51 | 1733.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:30:00 | 1731.30 | 1729.51 | 1733.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1729.00 | 1729.16 | 1732.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 1733.00 | 1729.16 | 1732.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1723.15 | 1727.96 | 1731.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:45:00 | 1715.95 | 1729.02 | 1731.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:45:00 | 1716.20 | 1725.18 | 1729.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 1710.00 | 1720.68 | 1726.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 1668.20 | 1664.07 | 1663.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1668.20 | 1664.07 | 1663.77 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 1661.30 | 1663.49 | 1663.59 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 15:15:00 | 1670.85 | 1664.96 | 1664.25 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 11:15:00 | 1659.95 | 1663.25 | 1663.63 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 1670.25 | 1663.85 | 1663.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1680.50 | 1668.17 | 1665.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 1674.00 | 1682.99 | 1677.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1674.00 | 1682.99 | 1677.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1674.00 | 1682.99 | 1677.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 1671.80 | 1682.99 | 1677.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1666.05 | 1679.60 | 1676.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 1666.05 | 1679.60 | 1676.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1663.50 | 1676.38 | 1675.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:30:00 | 1659.35 | 1676.38 | 1675.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1676.15 | 1676.35 | 1675.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 1676.15 | 1676.35 | 1675.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 1683.60 | 1677.80 | 1676.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1697.90 | 1677.80 | 1676.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 15:00:00 | 1686.00 | 1686.72 | 1682.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:30:00 | 1693.90 | 1686.71 | 1683.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 10:45:00 | 1691.00 | 1687.35 | 1683.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1684.40 | 1686.76 | 1683.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1684.40 | 1686.76 | 1683.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1685.00 | 1686.41 | 1683.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:30:00 | 1684.75 | 1686.41 | 1683.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1684.60 | 1686.05 | 1684.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 15:15:00 | 1714.50 | 1684.77 | 1683.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 1660.20 | 1684.61 | 1684.07 | SL hit (close<static) qty=1.00 sl=1675.65 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1653.75 | 1678.44 | 1681.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1641.00 | 1653.56 | 1659.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 1649.35 | 1648.36 | 1654.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 13:15:00 | 1649.35 | 1648.36 | 1654.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 1649.35 | 1648.36 | 1654.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:45:00 | 1654.45 | 1648.36 | 1654.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 1658.25 | 1650.34 | 1655.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 15:00:00 | 1658.25 | 1650.34 | 1655.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 1659.00 | 1652.07 | 1655.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:15:00 | 1676.70 | 1652.07 | 1655.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1677.45 | 1657.15 | 1657.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:30:00 | 1681.95 | 1657.15 | 1657.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 10:15:00 | 1668.05 | 1659.33 | 1658.42 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 1648.95 | 1657.18 | 1657.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 1641.65 | 1652.92 | 1655.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1651.50 | 1645.36 | 1649.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1651.50 | 1645.36 | 1649.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1651.50 | 1645.36 | 1649.30 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1659.05 | 1652.96 | 1652.25 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1640.45 | 1650.94 | 1651.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1640.00 | 1648.76 | 1650.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1654.60 | 1642.42 | 1643.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 1654.60 | 1642.42 | 1643.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1654.60 | 1642.42 | 1643.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 1661.85 | 1642.42 | 1643.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1664.80 | 1646.90 | 1645.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 10:15:00 | 1677.05 | 1663.01 | 1657.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 1668.05 | 1670.39 | 1665.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:00:00 | 1668.05 | 1670.39 | 1665.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1670.85 | 1670.17 | 1666.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1675.40 | 1670.61 | 1667.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1658.95 | 1671.14 | 1669.99 | SL hit (close<static) qty=1.00 sl=1666.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 1660.80 | 1669.08 | 1669.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 11:15:00 | 1650.65 | 1665.39 | 1667.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 13:15:00 | 1666.95 | 1664.89 | 1666.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 13:15:00 | 1666.95 | 1664.89 | 1666.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1666.95 | 1664.89 | 1666.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 1666.95 | 1664.89 | 1666.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1661.00 | 1664.11 | 1666.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:45:00 | 1669.85 | 1664.11 | 1666.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1661.50 | 1663.73 | 1665.76 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 1680.00 | 1668.68 | 1667.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 1682.45 | 1671.43 | 1668.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1670.00 | 1672.42 | 1669.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 1670.00 | 1672.42 | 1669.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1670.00 | 1672.42 | 1669.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1670.00 | 1672.42 | 1669.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1668.80 | 1671.70 | 1669.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 1670.00 | 1671.70 | 1669.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1668.90 | 1671.14 | 1669.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:00:00 | 1668.90 | 1671.14 | 1669.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 1655.00 | 1667.91 | 1668.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 13:15:00 | 1653.05 | 1658.00 | 1661.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 1656.90 | 1656.17 | 1659.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 10:15:00 | 1656.90 | 1656.17 | 1659.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1656.90 | 1656.17 | 1659.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:30:00 | 1660.10 | 1656.17 | 1659.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1660.45 | 1657.03 | 1659.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:30:00 | 1664.85 | 1657.03 | 1659.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1664.80 | 1658.58 | 1659.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:30:00 | 1663.90 | 1658.58 | 1659.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1661.95 | 1659.92 | 1660.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 1659.90 | 1659.92 | 1660.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1658.20 | 1659.58 | 1660.07 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 1673.35 | 1662.51 | 1661.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 1685.85 | 1667.18 | 1663.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 1666.40 | 1669.84 | 1665.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1666.40 | 1669.84 | 1665.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1666.40 | 1669.84 | 1665.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 1666.40 | 1669.84 | 1665.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1668.80 | 1669.64 | 1665.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 15:00:00 | 1677.50 | 1670.27 | 1667.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1707.70 | 1670.64 | 1667.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 1657.20 | 1674.19 | 1672.78 | SL hit (close<static) qty=1.00 sl=1665.50 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1659.05 | 1671.17 | 1671.53 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 1686.35 | 1668.51 | 1667.10 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 1660.00 | 1670.86 | 1671.70 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 15:15:00 | 1682.00 | 1673.99 | 1672.95 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 1665.00 | 1673.52 | 1673.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 09:15:00 | 1663.75 | 1667.26 | 1669.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 1668.65 | 1666.90 | 1668.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 1668.65 | 1666.90 | 1668.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1668.65 | 1666.90 | 1668.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 1668.65 | 1666.90 | 1668.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1669.35 | 1667.39 | 1668.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 1665.40 | 1668.61 | 1669.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:30:00 | 1665.40 | 1666.97 | 1668.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:00:00 | 1665.20 | 1666.62 | 1667.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 1670.10 | 1662.47 | 1662.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 1670.10 | 1662.47 | 1662.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 11:15:00 | 1677.35 | 1665.45 | 1663.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 1702.00 | 1702.35 | 1692.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 1702.00 | 1702.35 | 1692.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1704.45 | 1702.25 | 1694.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 1701.40 | 1702.25 | 1694.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1684.50 | 1698.70 | 1693.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:30:00 | 1688.45 | 1698.70 | 1693.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1686.20 | 1696.20 | 1693.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1684.10 | 1696.20 | 1693.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 1678.05 | 1690.82 | 1691.13 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 1695.15 | 1691.64 | 1691.42 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1686.40 | 1691.86 | 1691.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 1678.80 | 1683.59 | 1686.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1650.60 | 1642.75 | 1653.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 1650.60 | 1642.75 | 1653.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1650.60 | 1642.75 | 1653.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1650.60 | 1642.75 | 1653.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1655.85 | 1645.37 | 1653.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:30:00 | 1651.75 | 1645.37 | 1653.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 1651.60 | 1646.61 | 1653.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 1648.55 | 1646.61 | 1653.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 15:00:00 | 1637.05 | 1645.47 | 1651.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:45:00 | 1644.00 | 1640.59 | 1641.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 1660.70 | 1644.62 | 1642.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 1660.70 | 1644.62 | 1642.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 14:15:00 | 1667.45 | 1656.29 | 1649.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 15:15:00 | 1655.00 | 1656.03 | 1650.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1693.05 | 1656.03 | 1650.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1687.00 | 1688.52 | 1681.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 1700.00 | 1688.52 | 1681.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1710.15 | 1692.85 | 1683.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 1718.75 | 1692.85 | 1683.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 1715.00 | 1713.39 | 1703.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 13:00:00 | 1721.25 | 1714.96 | 1704.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 1673.45 | 1710.83 | 1706.71 | SL hit (close<ema400) qty=1.00 sl=1706.71 alert=retest1 |

### Cycle 59 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 1670.25 | 1702.71 | 1703.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 1665.00 | 1695.17 | 1699.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1674.50 | 1668.93 | 1681.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 1674.50 | 1668.93 | 1681.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1688.50 | 1672.85 | 1682.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 1688.50 | 1672.85 | 1682.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1685.05 | 1675.29 | 1682.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 1685.05 | 1675.29 | 1682.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1672.55 | 1674.74 | 1681.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 1687.50 | 1674.74 | 1681.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1692.65 | 1678.52 | 1681.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1692.65 | 1678.52 | 1681.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1685.25 | 1679.87 | 1682.28 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1695.90 | 1684.13 | 1683.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1707.60 | 1691.66 | 1687.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 10:15:00 | 1695.45 | 1701.17 | 1696.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 10:15:00 | 1695.45 | 1701.17 | 1696.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1695.45 | 1701.17 | 1696.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 1695.45 | 1701.17 | 1696.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 1709.90 | 1702.92 | 1697.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:45:00 | 1716.55 | 1707.51 | 1701.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:30:00 | 1714.45 | 1709.47 | 1703.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:15:00 | 1716.00 | 1709.47 | 1703.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:45:00 | 1718.00 | 1713.51 | 1705.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1712.95 | 1713.40 | 1706.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1709.00 | 1713.40 | 1706.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1691.90 | 1709.10 | 1704.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 1691.90 | 1709.10 | 1704.94 | SL hit (close<static) qty=1.00 sl=1694.25 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 1689.00 | 1701.34 | 1701.91 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 09:15:00 | 1724.50 | 1702.72 | 1702.23 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 10:15:00 | 1696.80 | 1708.97 | 1710.22 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 15:15:00 | 1717.00 | 1710.86 | 1710.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 10:15:00 | 1722.35 | 1713.69 | 1711.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 10:15:00 | 1718.35 | 1721.82 | 1717.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 1718.35 | 1721.82 | 1717.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1718.35 | 1721.82 | 1717.70 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 1712.00 | 1716.16 | 1716.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 1707.00 | 1713.61 | 1715.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1685.95 | 1682.45 | 1692.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:30:00 | 1685.95 | 1682.45 | 1692.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1696.90 | 1685.34 | 1692.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 1699.00 | 1685.34 | 1692.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1691.90 | 1686.65 | 1692.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:45:00 | 1691.05 | 1686.65 | 1692.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 1701.75 | 1689.67 | 1693.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 1701.75 | 1689.67 | 1693.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1700.00 | 1691.74 | 1693.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 1699.30 | 1691.74 | 1693.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1697.55 | 1692.90 | 1694.27 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 1704.30 | 1696.11 | 1695.54 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1685.00 | 1694.49 | 1695.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 1681.60 | 1688.65 | 1692.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 15:15:00 | 1690.00 | 1688.18 | 1691.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-17 09:15:00 | 1680.30 | 1688.18 | 1691.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1684.10 | 1687.37 | 1690.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:45:00 | 1691.55 | 1687.37 | 1690.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1681.00 | 1686.09 | 1689.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 1688.75 | 1686.09 | 1689.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1680.90 | 1683.62 | 1687.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1687.05 | 1683.62 | 1687.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1685.75 | 1684.04 | 1687.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1685.75 | 1684.04 | 1687.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1686.00 | 1684.44 | 1687.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1683.00 | 1684.44 | 1687.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1683.00 | 1684.15 | 1686.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:30:00 | 1677.55 | 1683.78 | 1686.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:30:00 | 1679.00 | 1682.66 | 1685.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:00:00 | 1680.00 | 1681.53 | 1684.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:30:00 | 1680.00 | 1681.00 | 1683.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1682.95 | 1681.39 | 1683.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:00:00 | 1682.95 | 1681.39 | 1683.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1681.00 | 1681.31 | 1683.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:00:00 | 1681.00 | 1681.31 | 1683.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 1682.00 | 1681.45 | 1683.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:45:00 | 1680.70 | 1681.75 | 1683.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 1684.60 | 1682.32 | 1683.31 | SL hit (close>ema400) qty=1.00 sl=1683.31 alert=retest1 |

### Cycle 68 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 1693.50 | 1685.19 | 1684.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 14:15:00 | 1695.25 | 1689.61 | 1687.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 1684.75 | 1688.86 | 1687.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 1684.75 | 1688.86 | 1687.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1684.75 | 1688.86 | 1687.82 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 1682.20 | 1686.77 | 1687.00 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1701.40 | 1687.98 | 1687.22 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 1686.10 | 1690.84 | 1691.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 1680.00 | 1685.87 | 1688.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 1691.95 | 1681.06 | 1682.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 1691.95 | 1681.06 | 1682.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1691.95 | 1681.06 | 1682.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 1691.95 | 1681.06 | 1682.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1700.05 | 1684.86 | 1683.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1708.00 | 1698.38 | 1692.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1692.05 | 1702.19 | 1697.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 1692.05 | 1702.19 | 1697.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1692.05 | 1702.19 | 1697.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 1692.80 | 1702.19 | 1697.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1702.80 | 1702.31 | 1698.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:15:00 | 1703.25 | 1702.31 | 1698.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 12:15:00 | 1705.20 | 1702.24 | 1698.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:00:00 | 1703.65 | 1702.52 | 1699.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:30:00 | 1706.00 | 1703.89 | 1700.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 1700.05 | 1703.12 | 1700.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:45:00 | 1700.30 | 1703.12 | 1700.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1701.05 | 1702.71 | 1700.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1754.70 | 1702.71 | 1700.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 1703.10 | 1740.23 | 1734.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:00:00 | 1707.00 | 1733.58 | 1732.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-13 10:15:00 | 1873.58 | 1817.53 | 1787.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 1955.85 | 1967.74 | 1967.91 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 2001.95 | 1972.78 | 1970.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 12:15:00 | 2026.55 | 1993.88 | 1981.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 09:15:00 | 2001.10 | 2002.84 | 1990.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 2001.10 | 2002.84 | 1990.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 2001.10 | 2002.84 | 1990.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:15:00 | 2031.00 | 2002.84 | 1990.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 2020.25 | 2036.65 | 2021.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 10:15:00 | 2050.95 | 2078.54 | 2081.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 2050.95 | 2078.54 | 2081.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1904.05 | 2031.95 | 2055.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 2024.95 | 1991.47 | 2019.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 2024.95 | 1991.47 | 2019.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 2024.95 | 1991.47 | 2019.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 2043.05 | 1991.47 | 2019.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2036.80 | 2000.53 | 2020.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:15:00 | 2079.25 | 2000.53 | 2020.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 2085.75 | 2031.82 | 2031.43 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 09:15:00 | 1957.00 | 2025.38 | 2033.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 10:15:00 | 1938.75 | 2008.05 | 2025.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 13:15:00 | 2005.80 | 2001.36 | 2017.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 2005.80 | 2001.36 | 2017.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 2034.75 | 2008.04 | 2018.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 2034.75 | 2008.04 | 2018.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 2013.50 | 2009.13 | 2018.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 1991.90 | 2009.13 | 2018.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 13:30:00 | 2007.30 | 2007.76 | 2013.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-16 12:15:00 | 2021.90 | 2015.10 | 2014.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 2021.90 | 2015.10 | 2014.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 2027.80 | 2017.64 | 2015.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2016.50 | 2020.71 | 2018.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 2016.50 | 2020.71 | 2018.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2016.50 | 2020.71 | 2018.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 2068.70 | 2019.28 | 2018.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 2113.70 | 2142.84 | 2145.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 2113.70 | 2142.84 | 2145.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 14:15:00 | 2098.00 | 2133.87 | 2141.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 2055.00 | 2022.05 | 2047.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 2055.00 | 2022.05 | 2047.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2055.00 | 2022.05 | 2047.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:45:00 | 1993.10 | 2013.06 | 2033.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 1893.44 | 1942.78 | 1978.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-05-05 09:15:00 | 1793.79 | 1866.76 | 1916.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 80 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1855.00 | 1783.35 | 1777.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1872.90 | 1801.26 | 1786.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 1819.20 | 1833.64 | 1813.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:00:00 | 1819.20 | 1833.64 | 1813.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1810.10 | 1828.93 | 1813.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1810.10 | 1828.93 | 1813.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1797.90 | 1822.72 | 1811.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1797.90 | 1822.72 | 1811.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1805.00 | 1809.92 | 1808.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:45:00 | 1816.10 | 1810.28 | 1808.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:00:00 | 1809.50 | 1815.16 | 1812.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 1804.40 | 1810.31 | 1810.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 12:15:00 | 1804.40 | 1810.31 | 1810.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 1788.00 | 1802.04 | 1806.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 15:15:00 | 1794.00 | 1792.18 | 1798.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:15:00 | 1796.00 | 1792.18 | 1798.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1798.50 | 1793.44 | 1798.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 13:45:00 | 1785.00 | 1789.57 | 1794.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 1821.20 | 1792.21 | 1794.43 | SL hit (close>static) qty=1.00 sl=1806.90 alert=retest2 |

### Cycle 82 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1833.00 | 1800.36 | 1797.94 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1781.00 | 1799.09 | 1800.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 12:15:00 | 1779.70 | 1795.21 | 1798.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 1781.00 | 1779.85 | 1788.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 1781.00 | 1779.85 | 1788.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1781.00 | 1779.85 | 1788.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1781.00 | 1779.85 | 1788.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1772.00 | 1777.84 | 1786.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:00:00 | 1765.10 | 1778.96 | 1782.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1765.50 | 1776.27 | 1781.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 1762.80 | 1767.66 | 1774.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:45:00 | 1763.60 | 1767.15 | 1771.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1779.00 | 1770.02 | 1772.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1794.30 | 1770.02 | 1772.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1801.00 | 1776.22 | 1774.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1801.00 | 1776.22 | 1774.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1816.80 | 1784.34 | 1778.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1799.70 | 1821.06 | 1813.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 1799.70 | 1821.06 | 1813.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1799.70 | 1821.06 | 1813.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 1800.00 | 1821.06 | 1813.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1790.20 | 1814.88 | 1811.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:45:00 | 1789.60 | 1814.88 | 1811.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 1790.20 | 1806.17 | 1807.56 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1818.00 | 1806.76 | 1806.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 1829.60 | 1818.64 | 1813.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 1921.70 | 1922.74 | 1891.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 13:15:00 | 1899.60 | 1911.78 | 1895.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1899.60 | 1911.78 | 1895.65 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1847.80 | 1884.34 | 1888.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1832.20 | 1856.31 | 1871.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1826.40 | 1825.90 | 1842.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 1826.40 | 1825.90 | 1842.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1846.60 | 1830.04 | 1843.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1846.60 | 1830.04 | 1843.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1855.00 | 1835.03 | 1844.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1837.80 | 1835.03 | 1844.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 1839.10 | 1834.52 | 1841.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1845.09 | 1845.21 | SL hit (close>static) qty=1.00 sl=1861.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1893.80 | 1854.83 | 1849.63 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 1830.10 | 1845.58 | 1846.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 1825.10 | 1837.27 | 1842.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1850.70 | 1830.79 | 1834.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1850.70 | 1830.79 | 1834.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1850.70 | 1830.79 | 1834.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 1827.00 | 1833.27 | 1834.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 1826.50 | 1833.27 | 1834.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1827.30 | 1825.02 | 1829.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:00:00 | 1830.00 | 1826.02 | 1829.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1830.10 | 1826.83 | 1829.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 1830.10 | 1826.83 | 1829.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1828.40 | 1822.49 | 1826.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1828.40 | 1822.49 | 1826.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1830.80 | 1824.15 | 1826.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 1828.90 | 1824.15 | 1826.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1863.80 | 1832.08 | 1830.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1863.80 | 1832.08 | 1830.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 1872.10 | 1851.94 | 1842.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 1883.00 | 1892.87 | 1875.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 1883.00 | 1892.87 | 1875.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1879.00 | 1890.09 | 1875.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1893.00 | 1893.93 | 1878.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 2082.30 | 2001.41 | 1958.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 1992.30 | 2027.91 | 2029.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1964.80 | 2004.01 | 2016.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1928.90 | 1925.31 | 1950.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:00:00 | 1928.90 | 1925.31 | 1950.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1961.90 | 1910.40 | 1927.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1961.90 | 1910.40 | 1927.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1971.90 | 1922.70 | 1931.15 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1992.20 | 1946.25 | 1940.99 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 1946.00 | 1954.61 | 1955.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 1931.70 | 1950.03 | 1953.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1951.00 | 1935.32 | 1943.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 1951.00 | 1935.32 | 1943.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1951.00 | 1935.32 | 1943.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 1951.00 | 1935.32 | 1943.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 2006.10 | 1949.48 | 1949.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 2006.10 | 1949.48 | 1949.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 15:15:00 | 1999.00 | 1959.38 | 1954.05 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1947.10 | 1960.72 | 1961.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 1931.00 | 1951.76 | 1957.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1939.20 | 1927.63 | 1938.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1939.20 | 1927.63 | 1938.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1939.20 | 1927.63 | 1938.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1918.00 | 1928.76 | 1933.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1915.40 | 1918.77 | 1928.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 1917.20 | 1916.32 | 1919.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 1917.20 | 1916.32 | 1919.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1932.30 | 1919.52 | 1920.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1932.30 | 1919.52 | 1920.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1923.40 | 1920.29 | 1920.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1919.30 | 1920.29 | 1920.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1923.00 | 1920.83 | 1920.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:45:00 | 1905.50 | 1912.89 | 1916.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1900.30 | 1907.21 | 1912.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1822.10 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1819.63 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1821.34 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1821.34 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1810.22 | 1833.25 | 1851.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 1805.28 | 1823.29 | 1843.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 1778.20 | 1768.64 | 1792.26 | SL hit (close>ema200) qty=0.50 sl=1768.64 alert=retest2 |

### Cycle 96 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 1710.00 | 1694.95 | 1693.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1734.00 | 1705.33 | 1698.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 14:15:00 | 1710.90 | 1714.11 | 1706.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 15:00:00 | 1710.90 | 1714.11 | 1706.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1714.00 | 1714.09 | 1707.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1725.10 | 1714.09 | 1707.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 1726.30 | 1715.23 | 1710.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 1680.30 | 1711.85 | 1712.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 1680.30 | 1711.85 | 1712.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 1666.00 | 1692.97 | 1702.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1700.50 | 1675.49 | 1687.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1700.50 | 1675.49 | 1687.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1700.50 | 1675.49 | 1687.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1700.50 | 1675.49 | 1687.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1672.90 | 1674.97 | 1685.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 1670.90 | 1674.83 | 1682.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1702.50 | 1677.83 | 1676.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1702.50 | 1677.83 | 1676.77 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 1660.80 | 1675.44 | 1677.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 1654.60 | 1671.27 | 1675.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 1651.80 | 1650.39 | 1659.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 12:00:00 | 1651.80 | 1650.39 | 1659.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1659.90 | 1652.95 | 1659.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 1659.90 | 1652.95 | 1659.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1648.80 | 1652.12 | 1658.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:45:00 | 1635.50 | 1648.53 | 1655.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1553.72 | 1592.38 | 1616.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1535.70 | 1532.80 | 1554.18 | SL hit (close>ema200) qty=0.50 sl=1532.80 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1609.00 | 1552.89 | 1545.43 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 1583.10 | 1586.10 | 1586.41 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 1589.40 | 1586.76 | 1586.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1596.00 | 1588.81 | 1587.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 1586.50 | 1589.88 | 1588.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 1586.50 | 1589.88 | 1588.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1586.50 | 1589.88 | 1588.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1586.50 | 1589.88 | 1588.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1600.70 | 1592.04 | 1589.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 1613.70 | 1601.15 | 1594.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 1640.90 | 1650.37 | 1651.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1640.90 | 1650.37 | 1651.37 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1661.30 | 1653.35 | 1652.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1681.80 | 1667.63 | 1661.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1664.40 | 1672.18 | 1666.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1664.40 | 1672.18 | 1666.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1664.40 | 1672.18 | 1666.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 1665.30 | 1672.18 | 1666.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1664.20 | 1670.58 | 1666.48 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1655.00 | 1662.95 | 1663.82 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 12:15:00 | 1670.40 | 1664.25 | 1663.88 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1640.40 | 1661.45 | 1662.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 1632.20 | 1651.91 | 1658.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1620.70 | 1593.93 | 1613.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1620.70 | 1593.93 | 1613.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1620.70 | 1593.93 | 1613.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:45:00 | 1593.90 | 1595.30 | 1611.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1597.00 | 1595.44 | 1610.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 1640.40 | 1618.94 | 1616.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 1640.40 | 1618.94 | 1616.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1652.40 | 1635.76 | 1629.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 1667.00 | 1667.00 | 1657.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:30:00 | 1667.00 | 1667.00 | 1657.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1672.00 | 1667.28 | 1659.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:30:00 | 1672.60 | 1668.55 | 1660.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1676.40 | 1669.14 | 1661.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1637.10 | 1661.59 | 1662.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1637.10 | 1661.59 | 1662.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 1632.80 | 1655.83 | 1659.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1636.60 | 1635.96 | 1645.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 10:15:00 | 1638.70 | 1635.96 | 1645.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1622.10 | 1628.71 | 1637.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1637.00 | 1628.71 | 1637.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1621.80 | 1627.33 | 1636.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 1611.00 | 1619.64 | 1627.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 1606.90 | 1602.46 | 1601.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1606.90 | 1602.46 | 1601.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1630.50 | 1608.91 | 1605.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 1688.60 | 1690.38 | 1668.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:30:00 | 1687.40 | 1690.38 | 1668.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1672.10 | 1686.34 | 1670.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 1672.10 | 1686.34 | 1670.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1658.90 | 1680.86 | 1669.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 1658.90 | 1680.86 | 1669.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1659.00 | 1676.48 | 1668.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 1653.10 | 1676.48 | 1668.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 1659.20 | 1663.90 | 1664.09 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1665.70 | 1661.52 | 1661.19 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1651.00 | 1660.05 | 1660.62 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1673.20 | 1660.76 | 1660.44 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1650.00 | 1662.36 | 1662.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 1646.60 | 1658.46 | 1660.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 1654.00 | 1646.41 | 1651.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 15:15:00 | 1654.00 | 1646.41 | 1651.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1654.00 | 1646.41 | 1651.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1622.60 | 1646.41 | 1651.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1659.90 | 1600.21 | 1608.04 | SL hit (close>static) qty=1.00 sl=1654.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1666.10 | 1613.39 | 1613.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1683.40 | 1639.05 | 1632.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1716.50 | 1727.73 | 1705.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 1716.50 | 1727.73 | 1705.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1704.90 | 1723.16 | 1705.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 1703.40 | 1723.16 | 1705.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1718.20 | 1722.17 | 1706.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 1719.90 | 1722.17 | 1706.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1700.80 | 1713.76 | 1708.01 | SL hit (close<static) qty=1.00 sl=1702.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 1672.60 | 1699.53 | 1702.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1657.30 | 1685.27 | 1693.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 14:15:00 | 1674.10 | 1673.76 | 1683.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:45:00 | 1677.30 | 1673.76 | 1683.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1664.00 | 1633.51 | 1635.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1664.00 | 1633.51 | 1635.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1650.00 | 1636.81 | 1637.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 1635.70 | 1634.25 | 1635.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 12:15:00 | 1553.91 | 1573.62 | 1590.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-03 12:15:00 | 1472.13 | 1493.72 | 1516.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 13:15:00 | 1498.00 | 1491.66 | 1491.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 14:15:00 | 1499.30 | 1493.19 | 1492.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1515.50 | 1532.92 | 1523.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 1515.50 | 1532.92 | 1523.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1515.50 | 1532.92 | 1523.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1537.90 | 1532.92 | 1523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1541.90 | 1534.71 | 1525.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 1547.00 | 1534.71 | 1525.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:30:00 | 1550.40 | 1556.12 | 1551.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1501.00 | 1542.30 | 1546.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1501.00 | 1542.30 | 1546.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 1476.40 | 1522.35 | 1536.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1457.00 | 1450.29 | 1463.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 1457.00 | 1450.29 | 1463.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1465.00 | 1453.24 | 1463.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1463.40 | 1453.24 | 1463.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1460.80 | 1454.75 | 1463.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 1455.50 | 1455.42 | 1462.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:00:00 | 1456.00 | 1455.79 | 1461.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:00:00 | 1455.60 | 1455.26 | 1460.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 12:15:00 | 1475.30 | 1464.47 | 1463.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 12:15:00 | 1475.30 | 1464.47 | 1463.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 1497.10 | 1478.37 | 1471.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 11:15:00 | 1496.00 | 1499.55 | 1489.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:45:00 | 1498.30 | 1499.55 | 1489.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1480.00 | 1494.81 | 1489.81 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1457.10 | 1484.90 | 1486.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1445.10 | 1458.19 | 1463.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1445.00 | 1443.16 | 1451.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 1439.00 | 1443.16 | 1451.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1440.90 | 1439.90 | 1444.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 1443.80 | 1439.90 | 1444.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1451.90 | 1442.74 | 1445.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:45:00 | 1452.60 | 1442.74 | 1445.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1456.00 | 1445.40 | 1446.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 1456.00 | 1445.40 | 1446.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 13:15:00 | 1465.50 | 1449.42 | 1448.10 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1441.50 | 1454.22 | 1454.76 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1456.20 | 1448.36 | 1447.65 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 1443.00 | 1446.56 | 1446.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 1435.80 | 1443.60 | 1445.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 13:15:00 | 1391.40 | 1384.73 | 1397.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 14:00:00 | 1391.40 | 1384.73 | 1397.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1388.00 | 1385.39 | 1396.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:15:00 | 1380.00 | 1385.39 | 1396.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1380.00 | 1384.31 | 1394.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1402.50 | 1384.31 | 1394.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1407.80 | 1389.01 | 1396.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1404.10 | 1389.01 | 1396.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1405.70 | 1392.35 | 1396.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 1410.00 | 1392.35 | 1396.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 1431.00 | 1402.57 | 1400.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 1436.90 | 1409.43 | 1404.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 1465.00 | 1469.35 | 1447.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1495.40 | 1469.35 | 1447.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 11:00:00 | 1486.40 | 1476.71 | 1454.58 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 14:30:00 | 1488.70 | 1482.79 | 1464.84 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1483.20 | 1483.23 | 1468.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 1489.80 | 1484.52 | 1470.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:00:00 | 1489.70 | 1484.52 | 1470.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 1471.00 | 1489.02 | 1480.15 | SL hit (close<ema400) qty=1.00 sl=1480.15 alert=retest1 |

### Cycle 127 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 1462.50 | 1473.56 | 1474.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 1450.00 | 1465.35 | 1469.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 1466.30 | 1463.68 | 1467.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1466.30 | 1463.68 | 1467.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1466.30 | 1463.68 | 1467.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1466.30 | 1463.68 | 1467.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1465.00 | 1463.94 | 1467.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1473.50 | 1463.94 | 1467.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1457.60 | 1462.68 | 1466.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 1453.40 | 1462.68 | 1466.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1456.10 | 1442.63 | 1449.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:45:00 | 1453.00 | 1442.64 | 1449.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 11:15:00 | 1383.29 | 1405.17 | 1418.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 12:15:00 | 1380.73 | 1400.56 | 1415.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 12:15:00 | 1380.35 | 1400.56 | 1415.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-06 09:15:00 | 1308.06 | 1375.16 | 1398.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 128 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1304.40 | 1301.46 | 1301.24 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1295.80 | 1301.83 | 1302.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1292.10 | 1299.89 | 1301.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 1287.70 | 1277.95 | 1283.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1287.70 | 1277.95 | 1283.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1287.70 | 1277.95 | 1283.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 1287.70 | 1277.95 | 1283.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1280.00 | 1278.36 | 1283.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1261.00 | 1278.36 | 1283.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1293.40 | 1274.46 | 1275.18 | SL hit (close>static) qty=1.00 sl=1288.10 alert=retest2 |

### Cycle 130 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 1282.90 | 1276.15 | 1275.88 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1244.70 | 1277.86 | 1280.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 1216.40 | 1231.05 | 1243.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 1226.20 | 1226.13 | 1235.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 1233.00 | 1226.13 | 1235.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1234.40 | 1229.57 | 1234.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 1234.40 | 1229.57 | 1234.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1234.00 | 1230.45 | 1234.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 1232.40 | 1230.45 | 1234.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1228.50 | 1230.06 | 1233.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1169.50 | 1230.06 | 1233.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1212.10 | 1212.69 | 1213.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:15:00 | 1151.49 | 1169.98 | 1185.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 1172.40 | 1165.43 | 1178.94 | SL hit (close>ema200) qty=0.50 sl=1165.43 alert=retest2 |

### Cycle 132 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 1117.50 | 1096.34 | 1094.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1152.50 | 1113.06 | 1103.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1113.90 | 1126.57 | 1117.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1113.90 | 1126.57 | 1117.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1113.90 | 1126.57 | 1117.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1113.90 | 1126.57 | 1117.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1118.80 | 1125.02 | 1117.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 1111.60 | 1125.02 | 1117.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1112.70 | 1122.55 | 1117.05 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1080.80 | 1108.31 | 1111.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 1076.40 | 1092.91 | 1101.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1118.70 | 1098.07 | 1103.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1118.70 | 1098.07 | 1103.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1118.70 | 1098.07 | 1103.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 1117.00 | 1098.07 | 1103.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1117.20 | 1101.89 | 1104.59 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1126.00 | 1109.72 | 1107.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 1135.40 | 1116.92 | 1111.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1117.40 | 1117.62 | 1112.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1117.40 | 1117.62 | 1112.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1117.40 | 1117.62 | 1112.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 1133.10 | 1120.36 | 1115.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:45:00 | 1140.90 | 1127.63 | 1121.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 1134.60 | 1136.95 | 1132.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1145.20 | 1131.58 | 1130.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 15:15:00 | 1246.41 | 1233.32 | 1212.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 1387.90 | 1403.68 | 1403.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 1371.10 | 1387.43 | 1394.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 14:15:00 | 1377.30 | 1370.89 | 1382.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 1377.30 | 1370.89 | 1382.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1365.00 | 1370.74 | 1380.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1358.30 | 1377.54 | 1379.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 1361.20 | 1369.15 | 1374.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 1381.10 | 1377.20 | 1376.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1381.10 | 1377.20 | 1376.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1395.30 | 1384.96 | 1381.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 1422.50 | 1432.55 | 1418.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 15:15:00 | 1422.50 | 1432.55 | 1418.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1422.50 | 1432.55 | 1418.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1440.20 | 1432.55 | 1418.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 1416.60 | 1427.51 | 1418.18 | SL hit (close<static) qty=1.00 sl=1417.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 11:30:00 | 1565.25 | 2024-05-16 10:15:00 | 1582.05 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-05-15 12:00:00 | 1561.45 | 2024-05-16 10:15:00 | 1582.05 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-05-17 12:15:00 | 1592.15 | 2024-05-28 12:15:00 | 1609.95 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-05-18 09:15:00 | 1598.10 | 2024-05-28 12:15:00 | 1609.95 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-05-21 12:15:00 | 1601.85 | 2024-05-28 12:15:00 | 1609.95 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2024-06-14 09:15:00 | 1852.95 | 2024-06-21 11:15:00 | 1891.30 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2024-06-14 13:00:00 | 1847.95 | 2024-06-21 11:15:00 | 1891.30 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2024-06-27 10:30:00 | 1853.00 | 2024-07-01 12:15:00 | 1869.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-27 12:00:00 | 1857.35 | 2024-07-01 12:15:00 | 1869.20 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-06-27 13:30:00 | 1856.05 | 2024-07-01 12:15:00 | 1869.20 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-06-28 09:15:00 | 1851.15 | 2024-07-01 12:15:00 | 1869.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-07-05 15:00:00 | 1781.05 | 2024-07-10 11:15:00 | 1811.65 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-07-08 11:15:00 | 1780.00 | 2024-07-10 11:15:00 | 1811.65 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-18 11:00:00 | 1764.45 | 2024-07-22 14:15:00 | 1792.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-07-18 14:15:00 | 1768.80 | 2024-07-22 14:15:00 | 1792.90 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1752.75 | 2024-07-22 14:15:00 | 1792.90 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-07-24 09:15:00 | 1799.30 | 2024-07-24 12:15:00 | 1761.05 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-07-29 15:15:00 | 1728.00 | 2024-08-05 09:15:00 | 1641.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 10:15:00 | 1724.70 | 2024-08-05 09:15:00 | 1638.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-29 15:15:00 | 1728.00 | 2024-08-06 09:15:00 | 1653.50 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2024-07-30 10:15:00 | 1724.70 | 2024-08-06 09:15:00 | 1653.50 | STOP_HIT | 0.50 | 4.13% |
| BUY | retest2 | 2024-09-10 15:00:00 | 1830.00 | 2024-09-11 09:15:00 | 1805.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-09-25 14:15:00 | 1887.95 | 2024-09-27 10:15:00 | 1881.45 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-09-25 15:00:00 | 1889.45 | 2024-09-27 10:15:00 | 1881.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-09-26 10:00:00 | 1888.90 | 2024-09-27 10:15:00 | 1881.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-09-26 14:00:00 | 1896.50 | 2024-09-27 10:15:00 | 1881.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1811.40 | 2024-10-07 10:15:00 | 1720.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1811.40 | 2024-10-08 10:15:00 | 1736.75 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2024-10-16 15:00:00 | 1748.75 | 2024-10-17 10:15:00 | 1730.15 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-10-21 09:45:00 | 1715.95 | 2024-10-30 11:15:00 | 1668.20 | STOP_HIT | 1.00 | 2.78% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1716.20 | 2024-10-30 11:15:00 | 1668.20 | STOP_HIT | 1.00 | 2.80% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1710.00 | 2024-10-30 11:15:00 | 1668.20 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1697.90 | 2024-11-08 09:15:00 | 1660.20 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-11-06 15:00:00 | 1686.00 | 2024-11-08 09:15:00 | 1660.20 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-11-07 09:30:00 | 1693.90 | 2024-11-08 09:15:00 | 1660.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-07 10:45:00 | 1691.00 | 2024-11-08 09:15:00 | 1660.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-11-07 15:15:00 | 1714.50 | 2024-11-08 09:15:00 | 1660.20 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-11-29 09:30:00 | 1675.40 | 2024-12-02 09:15:00 | 1658.95 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-12-11 15:00:00 | 1677.50 | 2024-12-13 09:15:00 | 1657.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-12-12 09:15:00 | 1707.70 | 2024-12-13 09:15:00 | 1657.20 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-12-26 09:15:00 | 1665.40 | 2024-12-31 10:15:00 | 1670.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-12-26 11:30:00 | 1665.40 | 2024-12-31 10:15:00 | 1670.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-12-26 13:00:00 | 1665.20 | 2024-12-31 10:15:00 | 1670.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-01-14 13:15:00 | 1648.55 | 2025-01-20 10:15:00 | 1660.70 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-01-14 15:00:00 | 1637.05 | 2025-01-20 10:15:00 | 1660.70 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-01-20 09:45:00 | 1644.00 | 2025-01-20 10:15:00 | 1660.70 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest1 | 2025-01-21 09:15:00 | 1693.05 | 2025-01-27 09:15:00 | 1673.45 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-01-23 10:15:00 | 1718.75 | 2025-01-27 10:15:00 | 1670.25 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-01-24 12:00:00 | 1715.00 | 2025-01-27 10:15:00 | 1670.25 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-01-24 13:00:00 | 1721.25 | 2025-01-27 10:15:00 | 1670.25 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-01-31 14:45:00 | 1716.55 | 2025-02-01 12:15:00 | 1691.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-01 09:30:00 | 1714.45 | 2025-02-01 12:15:00 | 1691.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-02-01 10:15:00 | 1716.00 | 2025-02-01 12:15:00 | 1691.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-01 10:45:00 | 1718.00 | 2025-02-01 12:15:00 | 1691.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest1 | 2025-02-17 09:15:00 | 1680.30 | 2025-02-19 14:15:00 | 1684.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-02-18 10:30:00 | 1677.55 | 2025-02-19 14:15:00 | 1684.60 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-02-18 11:30:00 | 1679.00 | 2025-02-20 10:15:00 | 1693.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-02-18 15:00:00 | 1680.00 | 2025-02-20 10:15:00 | 1693.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-02-19 09:30:00 | 1680.00 | 2025-02-20 10:15:00 | 1693.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-02-19 13:45:00 | 1680.70 | 2025-02-20 10:15:00 | 1693.50 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-03-07 11:15:00 | 1703.25 | 2025-03-13 10:15:00 | 1873.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-07 12:15:00 | 1705.20 | 2025-03-13 10:15:00 | 1874.02 | TARGET_HIT | 1.00 | 9.90% |
| BUY | retest2 | 2025-03-07 13:00:00 | 1703.65 | 2025-03-13 10:15:00 | 1873.41 | TARGET_HIT | 1.00 | 9.96% |
| BUY | retest2 | 2025-03-07 13:30:00 | 1706.00 | 2025-03-13 11:15:00 | 1875.72 | TARGET_HIT | 1.00 | 9.95% |
| BUY | retest2 | 2025-03-10 09:15:00 | 1754.70 | 2025-03-13 11:15:00 | 1876.60 | TARGET_HIT | 1.00 | 6.95% |
| BUY | retest2 | 2025-03-11 10:30:00 | 1703.10 | 2025-03-13 11:15:00 | 1877.70 | TARGET_HIT | 1.00 | 10.25% |
| BUY | retest2 | 2025-03-11 12:00:00 | 1707.00 | 2025-03-19 13:15:00 | 1930.17 | TARGET_HIT | 1.00 | 13.07% |
| BUY | retest2 | 2025-03-27 10:15:00 | 2031.00 | 2025-04-04 10:15:00 | 2050.95 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-03-28 12:15:00 | 2020.25 | 2025-04-04 10:15:00 | 2050.95 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-04-15 09:15:00 | 1991.90 | 2025-04-16 12:15:00 | 2021.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-04-15 13:30:00 | 2007.30 | 2025-04-16 12:15:00 | 2021.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-04-21 09:15:00 | 2068.70 | 2025-04-24 13:15:00 | 2113.70 | STOP_HIT | 1.00 | 2.18% |
| SELL | retest2 | 2025-04-29 14:45:00 | 1993.10 | 2025-05-02 09:15:00 | 1893.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 14:45:00 | 1993.10 | 2025-05-05 09:15:00 | 1793.79 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-14 12:45:00 | 1816.10 | 2025-05-15 12:15:00 | 1804.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-15 10:00:00 | 1809.50 | 2025-05-15 12:15:00 | 1804.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-05-19 13:45:00 | 1785.00 | 2025-05-20 09:15:00 | 1821.20 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-26 10:00:00 | 1765.10 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1765.50 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-05-27 09:45:00 | 1762.80 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-05-27 13:45:00 | 1763.60 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1837.80 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-06-16 12:00:00 | 1839.10 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-19 12:45:00 | 1827.00 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-19 13:15:00 | 1826.50 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1827.30 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-20 13:00:00 | 1830.00 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-06-26 09:30:00 | 1893.00 | 2025-06-30 09:15:00 | 2082.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1918.00 | 2025-07-28 12:15:00 | 1822.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1915.40 | 2025-07-28 12:15:00 | 1819.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:30:00 | 1917.20 | 2025-07-28 12:15:00 | 1821.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1917.20 | 2025-07-28 12:15:00 | 1821.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:45:00 | 1905.50 | 2025-07-28 13:15:00 | 1810.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:30:00 | 1900.30 | 2025-07-28 15:15:00 | 1805.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1918.00 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.29% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1915.40 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.16% |
| SELL | retest2 | 2025-07-21 13:30:00 | 1917.20 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.25% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1917.20 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.25% |
| SELL | retest2 | 2025-07-22 14:45:00 | 1905.50 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2025-07-23 10:30:00 | 1900.30 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 6.43% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1725.10 | 2025-08-13 14:15:00 | 1680.30 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-12 13:45:00 | 1726.30 | 2025-08-13 14:15:00 | 1680.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-08-18 14:45:00 | 1670.90 | 2025-08-20 09:15:00 | 1702.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-08-25 09:45:00 | 1635.50 | 2025-08-28 09:15:00 | 1553.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 09:45:00 | 1635.50 | 2025-09-01 09:15:00 | 1535.70 | STOP_HIT | 0.50 | 6.10% |
| BUY | retest2 | 2025-09-11 09:30:00 | 1613.70 | 2025-09-18 12:15:00 | 1640.90 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-09-29 10:45:00 | 1593.90 | 2025-09-30 10:15:00 | 1640.40 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1597.00 | 2025-09-30 10:15:00 | 1640.40 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-10-07 14:30:00 | 1672.60 | 2025-10-09 09:15:00 | 1637.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1676.40 | 2025-10-09 09:15:00 | 1637.10 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-10-14 09:45:00 | 1611.00 | 2025-10-17 13:15:00 | 1606.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1622.60 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-14 12:15:00 | 1719.90 | 2025-11-17 09:15:00 | 1700.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-17 09:45:00 | 1719.30 | 2025-11-17 10:15:00 | 1676.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-11-28 12:15:00 | 1553.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-12-03 12:15:00 | 1472.13 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-11 10:15:00 | 1547.00 | 2025-12-16 09:15:00 | 1501.00 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-12-15 11:30:00 | 1550.40 | 2025-12-16 09:15:00 | 1501.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-12-22 11:15:00 | 1455.50 | 2025-12-23 12:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-22 13:00:00 | 1456.00 | 2025-12-23 12:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-22 15:00:00 | 1455.60 | 2025-12-23 12:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2026-01-27 09:15:00 | 1495.40 | 2026-01-29 09:15:00 | 1471.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2026-01-27 11:00:00 | 1486.40 | 2026-01-29 09:15:00 | 1471.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest1 | 2026-01-27 14:30:00 | 1488.70 | 2026-01-29 09:15:00 | 1471.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-28 10:30:00 | 1489.80 | 2026-01-29 13:15:00 | 1462.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-01-28 11:00:00 | 1489.70 | 2026-01-29 13:15:00 | 1462.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-01 14:15:00 | 1453.40 | 2026-02-05 11:15:00 | 1383.29 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-02-03 09:15:00 | 1456.10 | 2026-02-05 12:15:00 | 1380.73 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-02-03 09:45:00 | 1453.00 | 2026-02-05 12:15:00 | 1380.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 14:15:00 | 1453.40 | 2026-02-06 09:15:00 | 1308.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 09:15:00 | 1456.10 | 2026-02-06 09:15:00 | 1310.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 09:45:00 | 1453.00 | 2026-02-06 09:15:00 | 1307.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1261.00 | 2026-02-25 14:15:00 | 1293.40 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1169.50 | 2026-03-13 11:15:00 | 1151.49 | PARTIAL | 0.50 | 1.54% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1169.50 | 2026-03-13 14:15:00 | 1172.40 | STOP_HIT | 0.50 | -0.25% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1212.10 | 2026-03-17 09:15:00 | 1111.02 | PARTIAL | 0.50 | 8.34% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1212.10 | 2026-03-18 14:15:00 | 1124.60 | STOP_HIT | 0.50 | 7.22% |
| BUY | retest2 | 2026-04-02 14:15:00 | 1133.10 | 2026-04-10 15:15:00 | 1246.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:45:00 | 1140.90 | 2026-04-13 09:15:00 | 1254.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 12:30:00 | 1134.60 | 2026-04-13 09:15:00 | 1248.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1145.20 | 2026-04-13 09:15:00 | 1259.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:00:00 | 1285.20 | 2026-04-21 10:15:00 | 1413.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 10:00:00 | 1289.00 | 2026-04-21 11:15:00 | 1417.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1358.30 | 2026-05-04 10:15:00 | 1381.10 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-30 11:30:00 | 1361.20 | 2026-05-04 10:15:00 | 1381.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-05-07 09:15:00 | 1440.20 | 2026-05-07 10:15:00 | 1416.60 | STOP_HIT | 1.00 | -1.64% |
