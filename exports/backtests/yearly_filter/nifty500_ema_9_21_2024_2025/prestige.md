# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1495.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 136 |
| ALERT1 | 94 |
| ALERT2 | 91 |
| ALERT2_SKIP | 41 |
| ALERT3 | 238 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 101 |
| PARTIAL | 19 |
| TARGET_HIT | 9 |
| STOP_HIT | 97 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 51 / 74
- **Target hits / Stop hits / Partials:** 9 / 97 / 19
- **Avg / median % per leg:** 1.19% / -0.55%
- **Sum % (uncompounded):** 148.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 11 | 25.6% | 6 | 37 | 0 | 0.88% | 37.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.36% | -7.1% |
| BUY @ 3rd Alert (retest2) | 40 | 11 | 27.5% | 6 | 34 | 0 | 1.13% | 45.0% |
| SELL (all) | 82 | 40 | 48.8% | 3 | 60 | 19 | 1.35% | 110.4% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.23% | -4.9% |
| SELL @ 3rd Alert (retest2) | 78 | 40 | 51.3% | 3 | 56 | 19 | 1.48% | 115.3% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.71% | -12.0% |
| retest2 (combined) | 118 | 51 | 43.2% | 9 | 90 | 19 | 1.36% | 160.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 1461.15 | 1486.76 | 1488.85 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1530.75 | 1489.18 | 1485.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 1549.90 | 1506.99 | 1497.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 1527.60 | 1533.24 | 1518.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 1527.60 | 1533.24 | 1518.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 1527.60 | 1533.24 | 1518.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 1521.00 | 1533.24 | 1518.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 1520.65 | 1530.72 | 1518.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:15:00 | 1520.65 | 1530.72 | 1518.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1540.95 | 1532.77 | 1520.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:30:00 | 1545.00 | 1534.45 | 1525.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 15:15:00 | 1607.10 | 1615.18 | 1615.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 1607.10 | 1615.18 | 1615.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 10:15:00 | 1593.60 | 1609.83 | 1613.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 1525.60 | 1522.57 | 1542.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:45:00 | 1524.70 | 1522.57 | 1542.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1536.60 | 1525.38 | 1542.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:30:00 | 1537.15 | 1525.38 | 1542.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1549.00 | 1530.10 | 1542.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 1597.30 | 1530.10 | 1542.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1574.00 | 1538.88 | 1545.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1605.95 | 1538.88 | 1545.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 1598.30 | 1550.77 | 1550.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1689.75 | 1601.17 | 1577.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1631.00 | 1678.47 | 1637.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1631.00 | 1678.47 | 1637.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1631.00 | 1678.47 | 1637.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:00:00 | 1631.00 | 1678.47 | 1637.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1521.50 | 1647.07 | 1627.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1521.50 | 1647.07 | 1627.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1450.05 | 1607.67 | 1611.09 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 1705.00 | 1589.85 | 1581.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 1751.60 | 1690.26 | 1643.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 1835.75 | 1854.64 | 1818.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 1835.75 | 1854.64 | 1818.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1835.75 | 1854.64 | 1818.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 1810.95 | 1854.64 | 1818.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1831.05 | 1845.31 | 1822.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 1831.05 | 1845.31 | 1822.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 1914.85 | 1939.60 | 1920.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 1857.75 | 1939.60 | 1920.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1863.50 | 1924.38 | 1915.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1872.05 | 1924.38 | 1915.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1873.95 | 1914.29 | 1911.39 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 1871.95 | 1905.83 | 1907.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1867.80 | 1887.92 | 1898.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1898.85 | 1887.56 | 1896.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1898.85 | 1887.56 | 1896.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1898.85 | 1887.56 | 1896.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1898.85 | 1887.56 | 1896.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1899.00 | 1889.84 | 1896.36 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 1932.40 | 1903.72 | 1901.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 13:15:00 | 1964.30 | 1915.84 | 1907.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 10:15:00 | 1968.15 | 1975.93 | 1957.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 11:15:00 | 1968.55 | 1975.93 | 1957.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 2039.35 | 2024.56 | 1995.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:30:00 | 2028.95 | 2024.56 | 1995.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1987.40 | 2015.63 | 1998.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 1998.95 | 2015.63 | 1998.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1994.70 | 2011.45 | 1998.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 2011.30 | 2006.64 | 1997.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 1961.80 | 1987.76 | 1990.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 1961.80 | 1987.76 | 1990.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 1929.50 | 1976.11 | 1985.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 12:15:00 | 1889.00 | 1880.59 | 1910.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 13:00:00 | 1889.00 | 1880.59 | 1910.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1898.95 | 1884.26 | 1909.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:30:00 | 1900.00 | 1884.26 | 1909.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1899.30 | 1887.27 | 1908.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:45:00 | 1900.00 | 1887.27 | 1908.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1881.65 | 1858.12 | 1875.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 1881.65 | 1858.12 | 1875.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1857.75 | 1858.04 | 1874.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:45:00 | 1845.95 | 1855.65 | 1871.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:30:00 | 1841.00 | 1849.47 | 1867.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 15:00:00 | 1843.45 | 1846.59 | 1862.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1753.65 | 1775.85 | 1785.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1751.28 | 1775.85 | 1785.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 12:15:00 | 1748.95 | 1774.23 | 1784.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-10 14:15:00 | 1799.65 | 1778.83 | 1784.38 | SL hit (close>ema200) qty=0.50 sl=1778.83 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 1795.85 | 1780.50 | 1779.67 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 1761.50 | 1777.25 | 1778.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 1756.00 | 1769.34 | 1774.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 12:15:00 | 1775.00 | 1766.69 | 1771.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 12:15:00 | 1775.00 | 1766.69 | 1771.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1775.00 | 1766.69 | 1771.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:45:00 | 1780.70 | 1766.69 | 1771.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1760.60 | 1765.47 | 1770.40 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 1832.20 | 1782.91 | 1776.81 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 13:15:00 | 1772.15 | 1786.06 | 1786.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 1758.00 | 1780.45 | 1783.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 1781.20 | 1759.99 | 1768.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 13:15:00 | 1781.20 | 1759.99 | 1768.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 1781.20 | 1759.99 | 1768.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 1778.55 | 1759.99 | 1768.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1783.35 | 1764.66 | 1770.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:45:00 | 1793.00 | 1764.66 | 1770.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1762.70 | 1765.78 | 1769.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1763.25 | 1765.78 | 1769.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1774.00 | 1763.13 | 1766.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 1774.00 | 1763.13 | 1766.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1785.00 | 1767.51 | 1768.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1796.55 | 1767.51 | 1768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 1790.45 | 1772.10 | 1770.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 1824.95 | 1788.32 | 1778.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 13:15:00 | 1734.25 | 1778.03 | 1775.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 1734.25 | 1778.03 | 1775.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1734.25 | 1778.03 | 1775.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1724.20 | 1778.03 | 1775.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1760.05 | 1774.44 | 1774.06 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 1760.00 | 1771.55 | 1772.78 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1840.00 | 1785.24 | 1778.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 1859.70 | 1800.13 | 1786.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1846.45 | 1846.94 | 1819.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 09:45:00 | 1849.20 | 1846.94 | 1819.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1845.50 | 1852.42 | 1833.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 1845.50 | 1852.42 | 1833.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 1837.15 | 1852.32 | 1841.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:30:00 | 1830.10 | 1852.32 | 1841.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 1830.10 | 1847.88 | 1840.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 1824.00 | 1847.88 | 1840.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 1827.70 | 1841.54 | 1838.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 1852.40 | 1841.54 | 1838.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1857.60 | 1870.24 | 1858.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:45:00 | 1857.00 | 1870.24 | 1858.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1850.15 | 1866.22 | 1858.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 1847.30 | 1866.22 | 1858.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 1842.00 | 1861.38 | 1856.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 1842.00 | 1861.38 | 1856.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1847.00 | 1858.50 | 1855.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:15:00 | 1852.40 | 1858.50 | 1855.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 1851.30 | 1857.06 | 1855.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 15:15:00 | 1849.50 | 1855.36 | 1854.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 1849.50 | 1854.19 | 1854.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 1849.50 | 1854.19 | 1854.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 1835.80 | 1850.51 | 1852.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1663.35 | 1654.10 | 1692.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1663.35 | 1654.10 | 1692.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1663.35 | 1654.10 | 1692.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:45:00 | 1655.35 | 1654.43 | 1689.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1647.65 | 1658.79 | 1683.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:45:00 | 1645.90 | 1654.51 | 1678.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:30:00 | 1640.60 | 1650.47 | 1672.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1678.05 | 1654.62 | 1663.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:45:00 | 1679.65 | 1654.62 | 1663.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 1685.00 | 1660.69 | 1665.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:45:00 | 1679.85 | 1660.69 | 1665.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 1679.00 | 1669.36 | 1668.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 1679.00 | 1669.36 | 1668.86 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 1663.20 | 1668.10 | 1668.40 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1692.40 | 1672.96 | 1670.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 1706.55 | 1691.37 | 1681.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1738.95 | 1744.05 | 1726.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 12:15:00 | 1738.95 | 1744.05 | 1726.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1738.95 | 1744.05 | 1726.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 1734.80 | 1744.05 | 1726.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1793.15 | 1826.34 | 1812.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 1793.15 | 1826.34 | 1812.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1780.00 | 1817.07 | 1809.56 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 1775.90 | 1802.19 | 1803.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 09:15:00 | 1749.60 | 1780.44 | 1792.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 12:15:00 | 1750.20 | 1746.10 | 1761.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-22 12:45:00 | 1750.85 | 1746.10 | 1761.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1722.20 | 1702.37 | 1718.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 1731.10 | 1702.37 | 1718.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 1771.00 | 1716.09 | 1722.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 1771.00 | 1716.09 | 1722.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 1750.00 | 1722.87 | 1725.31 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 1748.80 | 1728.06 | 1727.45 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 1699.50 | 1722.38 | 1725.10 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 1749.50 | 1728.61 | 1726.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 1761.55 | 1743.10 | 1735.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 1790.00 | 1790.00 | 1767.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 12:15:00 | 1781.80 | 1797.90 | 1779.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1781.80 | 1797.90 | 1779.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 1781.80 | 1797.90 | 1779.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1783.10 | 1794.94 | 1780.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:15:00 | 1778.45 | 1794.94 | 1780.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1785.00 | 1792.95 | 1780.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1818.45 | 1790.36 | 1780.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 10:45:00 | 1800.25 | 1797.25 | 1785.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 1763.20 | 1786.12 | 1785.13 | SL hit (close<static) qty=1.00 sl=1777.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 1771.15 | 1783.13 | 1783.86 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 1819.90 | 1789.70 | 1786.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 1847.40 | 1801.24 | 1792.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 1825.00 | 1826.61 | 1809.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 10:30:00 | 1824.55 | 1826.61 | 1809.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1817.95 | 1824.88 | 1810.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:30:00 | 1808.70 | 1824.88 | 1810.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 1794.50 | 1818.80 | 1808.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:00:00 | 1794.50 | 1818.80 | 1808.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 1774.95 | 1810.03 | 1805.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 1774.95 | 1810.03 | 1805.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 1772.00 | 1802.43 | 1802.67 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 15:15:00 | 1812.00 | 1800.92 | 1799.70 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 1759.00 | 1792.54 | 1796.00 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 1804.45 | 1789.32 | 1788.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 1846.90 | 1800.83 | 1793.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 1800.60 | 1804.16 | 1796.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:30:00 | 1809.55 | 1804.16 | 1796.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 1805.55 | 1804.44 | 1797.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:15:00 | 1828.05 | 1803.63 | 1800.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 1819.10 | 1805.55 | 1802.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:15:00 | 1824.15 | 1805.55 | 1802.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:45:00 | 1831.85 | 1847.44 | 1843.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1833.00 | 1844.55 | 1842.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 1804.85 | 1844.55 | 1842.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 1814.10 | 1838.46 | 1840.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 1814.10 | 1838.46 | 1840.18 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 1931.75 | 1855.01 | 1846.43 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 1862.00 | 1882.15 | 1883.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 1826.05 | 1865.76 | 1874.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 15:15:00 | 1826.60 | 1825.45 | 1838.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 09:15:00 | 1805.95 | 1825.45 | 1838.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1877.35 | 1837.05 | 1838.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1877.35 | 1837.05 | 1838.31 | SL hit (close>ema400) qty=1.00 sl=1838.31 alert=retest1 |

### Cycle 34 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 1867.00 | 1843.04 | 1840.92 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1823.05 | 1839.04 | 1839.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 11:15:00 | 1803.05 | 1829.12 | 1834.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 1840.00 | 1826.23 | 1831.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 14:15:00 | 1840.00 | 1826.23 | 1831.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 1840.00 | 1826.23 | 1831.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 1840.00 | 1826.23 | 1831.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 1840.00 | 1828.98 | 1832.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 1824.05 | 1828.98 | 1832.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1732.85 | 1764.13 | 1789.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 1740.55 | 1730.65 | 1748.89 | SL hit (close>ema200) qty=0.50 sl=1730.65 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 1802.00 | 1755.12 | 1753.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 1817.85 | 1773.91 | 1762.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 1856.95 | 1858.57 | 1840.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:30:00 | 1858.50 | 1858.57 | 1840.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1856.10 | 1855.94 | 1843.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:45:00 | 1845.25 | 1855.94 | 1843.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1883.55 | 1860.53 | 1848.08 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1797.50 | 1857.04 | 1862.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1743.90 | 1799.13 | 1825.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 1776.10 | 1771.67 | 1794.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 12:15:00 | 1766.90 | 1770.40 | 1788.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 1766.90 | 1770.40 | 1788.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 1753.35 | 1766.99 | 1785.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1665.68 | 1716.62 | 1740.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 1713.95 | 1713.64 | 1732.88 | SL hit (close>ema200) qty=0.50 sl=1713.64 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1653.95 | 1616.24 | 1613.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 1701.75 | 1633.34 | 1621.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 1655.00 | 1660.37 | 1643.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 11:00:00 | 1655.00 | 1660.37 | 1643.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1649.65 | 1655.19 | 1646.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:45:00 | 1647.65 | 1655.19 | 1646.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1647.00 | 1653.55 | 1646.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1642.55 | 1653.55 | 1646.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1636.15 | 1650.07 | 1645.49 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1614.05 | 1642.03 | 1642.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 12:15:00 | 1607.00 | 1635.03 | 1639.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1638.95 | 1611.82 | 1623.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 1638.95 | 1611.82 | 1623.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1638.95 | 1611.82 | 1623.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 1638.95 | 1611.82 | 1623.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1634.95 | 1616.45 | 1624.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:45:00 | 1610.10 | 1620.72 | 1625.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 1613.10 | 1617.87 | 1622.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 12:15:00 | 1529.59 | 1563.62 | 1589.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 12:15:00 | 1532.44 | 1563.62 | 1589.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 1543.60 | 1540.24 | 1563.94 | SL hit (close>ema200) qty=0.50 sl=1540.24 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1588.15 | 1555.49 | 1553.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 13:15:00 | 1592.75 | 1566.75 | 1559.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 12:15:00 | 1629.55 | 1630.09 | 1610.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:45:00 | 1627.95 | 1630.09 | 1610.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1644.00 | 1695.39 | 1682.17 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 1631.30 | 1672.38 | 1673.38 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 14:15:00 | 1674.40 | 1663.42 | 1663.33 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 1656.15 | 1662.98 | 1663.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 1643.10 | 1656.81 | 1660.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 1653.20 | 1652.67 | 1657.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 14:00:00 | 1653.20 | 1652.67 | 1657.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1651.55 | 1652.45 | 1657.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 15:15:00 | 1641.75 | 1652.45 | 1657.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 11:15:00 | 1659.55 | 1652.44 | 1655.25 | SL hit (close>static) qty=1.00 sl=1658.95 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 1689.05 | 1660.95 | 1658.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 1730.75 | 1674.91 | 1665.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 1736.80 | 1745.21 | 1726.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 11:00:00 | 1736.80 | 1745.21 | 1726.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1735.50 | 1739.31 | 1729.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 1723.20 | 1739.31 | 1729.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1737.90 | 1737.54 | 1730.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:15:00 | 1743.95 | 1737.54 | 1730.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1750.20 | 1740.07 | 1732.01 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1725.50 | 1735.39 | 1736.37 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 1750.35 | 1738.39 | 1737.64 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 1735.25 | 1741.11 | 1741.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 1727.40 | 1737.69 | 1739.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1757.80 | 1739.37 | 1739.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 1757.80 | 1739.37 | 1739.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1757.80 | 1739.37 | 1739.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1757.80 | 1739.37 | 1739.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 1748.95 | 1741.29 | 1740.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 1797.75 | 1752.58 | 1745.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 1843.30 | 1848.95 | 1822.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 13:00:00 | 1884.85 | 1857.15 | 1833.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1865.75 | 1870.12 | 1848.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 1849.70 | 1864.59 | 1852.59 | SL hit (close<ema400) qty=1.00 sl=1852.59 alert=retest1 |

### Cycle 49 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1804.00 | 1840.95 | 1845.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1800.95 | 1832.95 | 1841.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1775.00 | 1765.44 | 1789.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 1775.00 | 1765.44 | 1789.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1757.25 | 1736.33 | 1752.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1757.25 | 1736.33 | 1752.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1752.80 | 1739.62 | 1752.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1754.85 | 1739.62 | 1752.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1747.85 | 1741.27 | 1752.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 1735.70 | 1741.76 | 1751.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 1762.70 | 1750.82 | 1751.61 | SL hit (close>static) qty=1.00 sl=1760.40 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1486.00 | 1453.65 | 1453.44 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1433.80 | 1460.04 | 1461.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1427.65 | 1453.56 | 1458.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1354.20 | 1340.99 | 1375.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1354.20 | 1340.99 | 1375.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1282.40 | 1254.93 | 1273.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 1282.40 | 1254.93 | 1273.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1301.40 | 1264.22 | 1276.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 1301.40 | 1264.22 | 1276.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1291.15 | 1271.13 | 1276.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1291.15 | 1271.13 | 1276.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1276.45 | 1272.20 | 1276.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 1271.75 | 1272.20 | 1276.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 13:15:00 | 1311.65 | 1284.48 | 1281.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1311.65 | 1284.48 | 1281.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1320.05 | 1291.59 | 1284.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 10:15:00 | 1330.00 | 1332.13 | 1317.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 10:45:00 | 1332.00 | 1332.13 | 1317.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 1353.95 | 1336.49 | 1320.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 1387.55 | 1342.33 | 1327.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 1383.95 | 1404.23 | 1406.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 1383.95 | 1404.23 | 1406.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 1363.05 | 1385.78 | 1395.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 12:15:00 | 1378.25 | 1377.71 | 1389.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:45:00 | 1374.00 | 1377.71 | 1389.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1224.85 | 1224.81 | 1238.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 1239.10 | 1224.81 | 1238.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1229.00 | 1226.68 | 1237.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 1229.00 | 1226.68 | 1237.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1229.90 | 1227.32 | 1236.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1243.95 | 1227.32 | 1236.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1239.75 | 1229.81 | 1236.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1239.75 | 1229.81 | 1236.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1244.00 | 1232.65 | 1237.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1224.60 | 1232.65 | 1237.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1234.05 | 1221.45 | 1226.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 1235.50 | 1221.45 | 1226.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1235.00 | 1224.16 | 1227.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1240.45 | 1224.16 | 1227.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1231.55 | 1225.64 | 1227.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:45:00 | 1244.55 | 1225.64 | 1227.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1238.20 | 1230.45 | 1229.60 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 09:15:00 | 1205.05 | 1225.40 | 1227.53 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 1234.50 | 1229.67 | 1229.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 1256.10 | 1234.95 | 1231.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1228.75 | 1238.91 | 1234.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1228.75 | 1238.91 | 1234.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1228.75 | 1238.91 | 1234.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1228.75 | 1238.91 | 1234.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1225.85 | 1236.30 | 1233.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1225.85 | 1236.30 | 1233.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1232.10 | 1235.46 | 1233.78 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 1210.00 | 1228.76 | 1230.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 10:15:00 | 1183.35 | 1212.30 | 1221.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 15:15:00 | 1196.20 | 1193.01 | 1207.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1194.30 | 1193.27 | 1205.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1194.30 | 1193.27 | 1205.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 1199.90 | 1193.27 | 1205.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1111.10 | 1118.94 | 1138.61 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 1185.20 | 1149.92 | 1147.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1201.15 | 1182.80 | 1169.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 12:15:00 | 1187.00 | 1190.50 | 1176.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 13:00:00 | 1187.00 | 1190.50 | 1176.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 1175.75 | 1187.55 | 1176.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:00:00 | 1175.75 | 1187.55 | 1176.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 1179.40 | 1185.92 | 1176.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 09:30:00 | 1183.45 | 1182.53 | 1176.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 11:15:00 | 1163.25 | 1177.69 | 1175.53 | SL hit (close<static) qty=1.00 sl=1170.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 1143.35 | 1168.84 | 1172.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 1133.05 | 1158.03 | 1166.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 1126.80 | 1124.43 | 1135.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 1126.80 | 1124.43 | 1135.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1122.80 | 1124.98 | 1133.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 1137.00 | 1124.98 | 1133.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1139.35 | 1125.43 | 1128.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:30:00 | 1146.00 | 1125.43 | 1128.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1141.40 | 1128.62 | 1129.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 1143.65 | 1128.62 | 1129.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1130.00 | 1129.46 | 1130.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 1122.00 | 1129.61 | 1130.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 11:45:00 | 1119.85 | 1117.71 | 1120.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 1132.60 | 1124.43 | 1123.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1132.60 | 1124.43 | 1123.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1165.95 | 1134.42 | 1128.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 1217.05 | 1221.13 | 1201.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 14:00:00 | 1217.05 | 1221.13 | 1201.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1217.05 | 1227.02 | 1216.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:30:00 | 1217.45 | 1227.02 | 1216.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1220.00 | 1225.62 | 1217.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 1230.40 | 1225.62 | 1217.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:45:00 | 1226.85 | 1229.75 | 1221.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 1226.05 | 1228.04 | 1221.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1254.00 | 1227.47 | 1222.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1230.95 | 1228.16 | 1223.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 1230.95 | 1228.16 | 1223.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1230.00 | 1228.53 | 1224.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 1230.00 | 1228.53 | 1224.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 1218.05 | 1226.43 | 1223.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 1218.05 | 1226.43 | 1223.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1223.00 | 1225.75 | 1223.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 13:15:00 | 1215.35 | 1223.67 | 1222.81 | SL hit (close<static) qty=1.00 sl=1216.30 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1197.65 | 1218.46 | 1220.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 1192.10 | 1205.38 | 1212.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 1207.10 | 1203.43 | 1209.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 15:15:00 | 1207.10 | 1203.43 | 1209.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1207.10 | 1203.43 | 1209.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1197.35 | 1203.43 | 1209.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1206.65 | 1204.08 | 1209.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 1189.85 | 1203.23 | 1208.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 1188.30 | 1200.24 | 1206.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 10:15:00 | 1130.36 | 1173.94 | 1190.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:15:00 | 1128.88 | 1167.55 | 1186.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 1161.50 | 1153.89 | 1170.94 | SL hit (close>ema200) qty=0.50 sl=1153.89 alert=retest2 |

### Cycle 62 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 1183.05 | 1174.42 | 1174.13 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1144.85 | 1172.41 | 1173.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1141.75 | 1161.77 | 1168.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1084.00 | 1077.96 | 1104.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1084.00 | 1077.96 | 1104.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1096.90 | 1084.02 | 1089.87 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1120.40 | 1095.69 | 1094.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1168.90 | 1122.49 | 1108.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 15:15:00 | 1318.90 | 1319.29 | 1293.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 09:15:00 | 1301.50 | 1319.29 | 1293.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1292.30 | 1313.89 | 1292.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 1286.80 | 1313.89 | 1292.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1296.20 | 1310.36 | 1293.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:30:00 | 1294.50 | 1310.36 | 1293.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1294.10 | 1307.10 | 1293.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:00:00 | 1294.10 | 1307.10 | 1293.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1302.00 | 1306.08 | 1294.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 1303.60 | 1306.08 | 1294.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 1290.10 | 1302.89 | 1293.75 | SL hit (close<static) qty=1.00 sl=1294.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1246.30 | 1284.79 | 1287.81 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 1297.10 | 1283.74 | 1283.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 1307.70 | 1291.08 | 1286.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 1385.20 | 1388.27 | 1359.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:00:00 | 1385.20 | 1388.27 | 1359.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1363.90 | 1377.72 | 1364.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 1363.90 | 1377.72 | 1364.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1355.00 | 1373.17 | 1363.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 1355.00 | 1373.17 | 1363.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1364.00 | 1371.34 | 1363.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:45:00 | 1375.80 | 1370.83 | 1365.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 1354.60 | 1370.88 | 1370.59 | SL hit (close<static) qty=1.00 sl=1355.10 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 1346.90 | 1366.08 | 1368.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1326.90 | 1358.25 | 1364.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1340.20 | 1335.09 | 1345.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 1340.20 | 1335.09 | 1345.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1330.20 | 1334.47 | 1343.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 1318.30 | 1328.92 | 1339.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 1313.80 | 1325.11 | 1335.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 1338.50 | 1318.23 | 1316.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1338.50 | 1318.23 | 1316.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1371.00 | 1328.78 | 1321.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1347.80 | 1350.91 | 1339.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 1347.80 | 1350.91 | 1339.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1348.00 | 1350.05 | 1341.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1365.00 | 1350.05 | 1341.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 1415.00 | 1431.92 | 1432.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 1415.00 | 1431.92 | 1432.30 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1431.60 | 1430.36 | 1430.33 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 1426.00 | 1430.44 | 1430.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 1414.70 | 1427.29 | 1429.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 13:15:00 | 1434.40 | 1426.61 | 1428.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 1434.40 | 1426.61 | 1428.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1434.40 | 1426.61 | 1428.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1434.40 | 1426.61 | 1428.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1438.90 | 1429.07 | 1429.05 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 1425.30 | 1428.61 | 1428.95 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 1432.90 | 1429.55 | 1429.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 1446.20 | 1432.88 | 1430.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 1450.70 | 1451.02 | 1442.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:30:00 | 1451.50 | 1451.02 | 1442.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1462.10 | 1482.14 | 1476.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1462.10 | 1482.14 | 1476.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1484.00 | 1482.51 | 1477.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1524.10 | 1482.51 | 1477.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 10:15:00 | 1676.51 | 1636.23 | 1606.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 1664.20 | 1672.93 | 1673.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1661.40 | 1667.10 | 1669.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1660.30 | 1640.51 | 1648.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 1660.30 | 1640.51 | 1648.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1660.30 | 1640.51 | 1648.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1660.30 | 1640.51 | 1648.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1656.20 | 1643.65 | 1648.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1628.60 | 1643.65 | 1648.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 1666.00 | 1647.77 | 1649.86 | SL hit (close>static) qty=1.00 sl=1660.40 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1688.50 | 1655.92 | 1653.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1701.40 | 1672.92 | 1662.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 1695.00 | 1698.38 | 1683.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 14:45:00 | 1695.10 | 1698.38 | 1683.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1701.90 | 1698.39 | 1685.96 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 1675.30 | 1686.87 | 1687.30 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1713.80 | 1691.18 | 1688.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1724.70 | 1706.50 | 1699.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1728.00 | 1730.16 | 1717.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 1731.80 | 1730.16 | 1717.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1723.40 | 1726.83 | 1719.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:30:00 | 1719.20 | 1726.83 | 1719.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1709.50 | 1723.36 | 1718.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 1709.50 | 1723.36 | 1718.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1718.00 | 1722.29 | 1718.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 13:30:00 | 1721.70 | 1722.33 | 1719.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 1707.70 | 1723.99 | 1721.64 | SL hit (close<static) qty=1.00 sl=1709.40 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1700.00 | 1719.19 | 1719.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 09:15:00 | 1678.80 | 1702.35 | 1710.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1620.70 | 1614.71 | 1632.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 1620.70 | 1614.71 | 1632.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1614.60 | 1602.98 | 1613.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1614.60 | 1602.98 | 1613.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1618.00 | 1605.99 | 1613.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1613.00 | 1605.99 | 1613.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1608.20 | 1606.43 | 1613.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1613.90 | 1606.43 | 1613.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1612.90 | 1607.72 | 1613.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1612.90 | 1607.72 | 1613.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1614.50 | 1609.08 | 1613.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1616.60 | 1609.08 | 1613.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1613.60 | 1609.98 | 1613.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 1611.90 | 1609.98 | 1613.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1619.70 | 1611.93 | 1613.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 1619.70 | 1611.93 | 1613.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 1631.40 | 1615.82 | 1615.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 1632.00 | 1624.78 | 1620.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 10:15:00 | 1640.30 | 1642.92 | 1632.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:00:00 | 1640.30 | 1642.92 | 1632.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1647.90 | 1643.46 | 1634.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 1633.80 | 1643.46 | 1634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1637.30 | 1642.23 | 1635.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1667.80 | 1644.54 | 1637.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 1741.00 | 1775.47 | 1779.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1741.00 | 1775.47 | 1779.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1729.80 | 1752.59 | 1764.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1633.90 | 1628.15 | 1657.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 1634.00 | 1628.15 | 1657.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 1626.70 | 1628.34 | 1641.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 1639.30 | 1628.34 | 1641.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1621.60 | 1613.39 | 1624.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 1621.60 | 1613.39 | 1624.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1626.40 | 1615.99 | 1624.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:30:00 | 1631.50 | 1615.99 | 1624.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1625.00 | 1617.79 | 1624.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1616.90 | 1617.79 | 1624.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1651.20 | 1624.47 | 1627.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1651.20 | 1624.47 | 1627.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1641.40 | 1627.86 | 1628.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 1652.50 | 1627.86 | 1628.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1627.60 | 1627.22 | 1627.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 1623.30 | 1627.22 | 1627.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1618.00 | 1625.38 | 1627.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 1604.30 | 1625.38 | 1627.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 1631.40 | 1618.27 | 1620.37 | SL hit (close>static) qty=1.00 sl=1630.30 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1628.00 | 1622.69 | 1622.16 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1608.30 | 1619.81 | 1620.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 1602.20 | 1614.41 | 1617.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 1616.70 | 1599.15 | 1607.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 12:15:00 | 1616.70 | 1599.15 | 1607.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1616.70 | 1599.15 | 1607.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 1616.70 | 1599.15 | 1607.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1607.40 | 1600.80 | 1607.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:15:00 | 1604.40 | 1600.80 | 1607.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1600.00 | 1601.84 | 1607.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:45:00 | 1604.00 | 1600.04 | 1605.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1630.00 | 1611.56 | 1609.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1630.00 | 1611.56 | 1609.25 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 1600.60 | 1607.79 | 1608.54 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1614.70 | 1609.01 | 1608.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 1626.50 | 1612.51 | 1610.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 1628.80 | 1634.23 | 1624.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 1628.80 | 1634.23 | 1624.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1634.80 | 1633.51 | 1627.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 1645.80 | 1633.51 | 1627.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1635.00 | 1633.81 | 1628.46 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1611.90 | 1628.56 | 1628.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 1602.80 | 1620.76 | 1625.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1639.10 | 1615.55 | 1619.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1639.10 | 1615.55 | 1619.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1639.10 | 1615.55 | 1619.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1639.10 | 1615.55 | 1619.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1634.40 | 1619.32 | 1621.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 1623.20 | 1620.10 | 1621.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 1633.90 | 1624.33 | 1623.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 1633.90 | 1624.33 | 1623.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 1651.90 | 1631.09 | 1627.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 1636.20 | 1639.05 | 1633.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 1636.20 | 1639.05 | 1633.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1636.20 | 1639.05 | 1633.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 1636.20 | 1639.05 | 1633.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1628.00 | 1636.84 | 1632.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1628.00 | 1636.84 | 1632.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1637.00 | 1636.87 | 1633.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1663.70 | 1639.62 | 1636.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 1660.50 | 1646.11 | 1640.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 1656.30 | 1648.23 | 1641.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1625.20 | 1645.27 | 1642.56 | SL hit (close<static) qty=1.00 sl=1626.60 alert=retest2 |

### Cycle 89 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1618.10 | 1639.84 | 1640.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1613.00 | 1628.76 | 1634.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 1590.00 | 1589.87 | 1606.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:15:00 | 1566.20 | 1589.87 | 1606.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 11:30:00 | 1576.10 | 1580.75 | 1597.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1572.40 | 1580.75 | 1597.22 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1576.60 | 1566.73 | 1576.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1576.60 | 1566.73 | 1576.54 | SL hit (close>ema400) qty=1.00 sl=1576.54 alert=retest1 |

### Cycle 90 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 1585.10 | 1575.40 | 1575.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 1589.60 | 1578.24 | 1576.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 1578.00 | 1583.82 | 1580.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 11:15:00 | 1578.00 | 1583.82 | 1580.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1578.00 | 1583.82 | 1580.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1578.00 | 1583.82 | 1580.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1568.00 | 1580.65 | 1578.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 1569.90 | 1580.65 | 1578.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1556.00 | 1575.72 | 1576.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1553.80 | 1571.34 | 1574.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1540.80 | 1528.33 | 1544.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1540.80 | 1528.33 | 1544.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1540.80 | 1528.33 | 1544.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1540.00 | 1528.33 | 1544.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1539.20 | 1530.50 | 1544.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1532.00 | 1537.36 | 1543.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1548.60 | 1543.63 | 1543.85 | SL hit (close>static) qty=1.00 sl=1548.40 alert=retest2 |

### Cycle 92 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1562.00 | 1547.31 | 1545.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 1570.40 | 1558.27 | 1552.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 1551.60 | 1558.82 | 1554.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 1551.60 | 1558.82 | 1554.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1551.60 | 1558.82 | 1554.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1551.60 | 1558.82 | 1554.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1557.60 | 1558.58 | 1555.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1564.50 | 1559.06 | 1555.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:00:00 | 1566.90 | 1560.63 | 1556.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1544.00 | 1556.82 | 1555.59 | SL hit (close<static) qty=1.00 sl=1550.30 alert=retest2 |

### Cycle 93 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1536.90 | 1552.84 | 1553.89 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 1573.80 | 1556.40 | 1554.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1593.60 | 1563.84 | 1558.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 1638.80 | 1646.10 | 1627.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:45:00 | 1636.50 | 1646.10 | 1627.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1635.70 | 1644.02 | 1628.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 1634.80 | 1644.02 | 1628.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1628.20 | 1640.85 | 1628.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 1628.20 | 1640.85 | 1628.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1631.10 | 1638.90 | 1628.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 1627.10 | 1638.90 | 1628.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1627.10 | 1636.54 | 1628.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1628.00 | 1636.54 | 1628.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1620.70 | 1633.37 | 1627.96 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1601.30 | 1622.77 | 1623.82 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 1635.40 | 1625.43 | 1624.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1640.00 | 1628.34 | 1625.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1621.50 | 1630.43 | 1627.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 1621.50 | 1630.43 | 1627.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1621.50 | 1630.43 | 1627.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 1619.40 | 1630.43 | 1627.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1614.00 | 1627.14 | 1626.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 1612.60 | 1627.14 | 1626.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1609.70 | 1623.66 | 1625.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 1580.00 | 1603.99 | 1613.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 1526.60 | 1521.56 | 1540.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:45:00 | 1527.70 | 1521.56 | 1540.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1524.60 | 1522.41 | 1534.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1514.60 | 1522.11 | 1533.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 1518.80 | 1521.55 | 1532.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 1518.80 | 1510.99 | 1520.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 1518.60 | 1512.91 | 1520.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1519.40 | 1514.21 | 1520.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:45:00 | 1514.50 | 1514.71 | 1519.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1526.40 | 1517.04 | 1520.36 | SL hit (close>static) qty=1.00 sl=1523.90 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1542.00 | 1526.25 | 1524.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1548.10 | 1536.48 | 1532.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1530.70 | 1535.98 | 1532.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 1530.70 | 1535.98 | 1532.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1530.70 | 1535.98 | 1532.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1530.70 | 1535.98 | 1532.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1528.00 | 1534.39 | 1532.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 1528.00 | 1534.39 | 1532.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1529.00 | 1533.31 | 1531.97 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1530.00 | 1531.09 | 1531.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1516.70 | 1528.21 | 1529.84 | Break + close below crossover candle low |

### Cycle 100 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1569.40 | 1526.64 | 1526.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 1626.20 | 1584.23 | 1560.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1616.60 | 1620.31 | 1598.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 12:30:00 | 1615.80 | 1620.31 | 1598.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1603.30 | 1613.65 | 1601.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1601.80 | 1613.65 | 1601.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1600.10 | 1610.94 | 1601.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 1602.40 | 1610.94 | 1601.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1588.80 | 1606.51 | 1600.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 1588.80 | 1606.51 | 1600.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1600.30 | 1604.23 | 1600.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 1606.00 | 1604.10 | 1600.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 1604.40 | 1604.10 | 1600.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-24 12:15:00 | 1764.84 | 1744.47 | 1730.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1747.30 | 1757.32 | 1758.18 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1763.70 | 1756.32 | 1756.09 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 1751.90 | 1755.33 | 1755.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1740.20 | 1752.30 | 1754.32 | Break + close below crossover candle low |

### Cycle 104 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1772.50 | 1756.29 | 1755.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 1774.20 | 1759.87 | 1757.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1774.20 | 1776.52 | 1768.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:00:00 | 1774.20 | 1776.52 | 1768.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1765.50 | 1774.31 | 1768.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 1767.80 | 1774.31 | 1768.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1759.50 | 1771.35 | 1767.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:30:00 | 1759.30 | 1771.35 | 1767.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1763.30 | 1768.96 | 1767.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1763.20 | 1768.96 | 1767.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1742.90 | 1763.75 | 1765.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1739.80 | 1756.12 | 1761.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1740.10 | 1731.23 | 1742.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 1740.10 | 1731.23 | 1742.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1728.80 | 1730.74 | 1740.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1728.80 | 1730.74 | 1740.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1737.30 | 1733.06 | 1740.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:15:00 | 1736.00 | 1733.06 | 1740.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1736.00 | 1733.65 | 1739.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1737.90 | 1733.65 | 1739.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1736.80 | 1734.28 | 1739.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1753.30 | 1734.28 | 1739.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1747.10 | 1736.84 | 1740.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1743.30 | 1736.84 | 1740.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1742.60 | 1737.99 | 1740.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 1751.60 | 1737.99 | 1740.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1755.70 | 1742.98 | 1742.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 1765.00 | 1753.73 | 1748.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 10:15:00 | 1739.70 | 1751.76 | 1748.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 1739.70 | 1751.76 | 1748.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1739.70 | 1751.76 | 1748.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 1739.70 | 1751.76 | 1748.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1732.00 | 1747.81 | 1747.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 1732.00 | 1747.81 | 1747.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 1726.70 | 1743.59 | 1745.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1712.70 | 1737.41 | 1742.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1758.80 | 1730.55 | 1737.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1758.80 | 1730.55 | 1737.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1758.80 | 1730.55 | 1737.08 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 1748.20 | 1741.60 | 1741.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 14:15:00 | 1756.70 | 1745.00 | 1742.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1725.60 | 1742.22 | 1742.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1725.60 | 1742.22 | 1742.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1725.60 | 1742.22 | 1742.04 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1721.20 | 1738.02 | 1740.14 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 13:15:00 | 1749.00 | 1741.42 | 1741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1760.00 | 1749.77 | 1745.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1724.60 | 1745.74 | 1745.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1724.60 | 1745.74 | 1745.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1724.60 | 1745.74 | 1745.15 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1734.70 | 1743.54 | 1744.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1720.00 | 1735.27 | 1739.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 1715.40 | 1714.36 | 1722.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 1715.40 | 1714.36 | 1722.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1720.80 | 1715.64 | 1721.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1720.80 | 1715.64 | 1721.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1724.00 | 1717.32 | 1722.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1708.60 | 1717.32 | 1722.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1703.00 | 1714.45 | 1720.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 1683.00 | 1708.16 | 1716.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:15:00 | 1691.00 | 1698.26 | 1709.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 1675.80 | 1671.15 | 1671.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 1675.80 | 1671.15 | 1671.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 1677.10 | 1672.34 | 1671.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 1674.40 | 1675.18 | 1673.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 1674.40 | 1675.18 | 1673.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1674.40 | 1675.18 | 1673.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1674.40 | 1675.18 | 1673.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 1651.50 | 1670.45 | 1671.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1643.50 | 1652.61 | 1658.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 1648.40 | 1648.13 | 1654.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 1648.40 | 1648.13 | 1654.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1646.50 | 1643.83 | 1649.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1646.50 | 1643.83 | 1649.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1661.70 | 1647.41 | 1651.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1661.70 | 1647.41 | 1651.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1662.90 | 1650.50 | 1652.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 1663.60 | 1650.50 | 1652.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 1661.10 | 1654.40 | 1653.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 1691.90 | 1662.38 | 1657.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1640.50 | 1675.08 | 1669.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1640.50 | 1675.08 | 1669.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1640.50 | 1675.08 | 1669.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1640.50 | 1675.08 | 1669.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1625.40 | 1665.15 | 1665.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1621.10 | 1656.34 | 1661.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1643.10 | 1626.14 | 1639.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 1643.10 | 1626.14 | 1639.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1643.10 | 1626.14 | 1639.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1643.10 | 1626.14 | 1639.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1631.00 | 1627.11 | 1638.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:45:00 | 1629.10 | 1629.37 | 1637.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 1618.40 | 1632.23 | 1636.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1650.50 | 1631.34 | 1634.41 | SL hit (close>static) qty=1.00 sl=1648.80 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 1645.40 | 1637.30 | 1636.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1670.60 | 1649.44 | 1643.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 1648.80 | 1649.31 | 1643.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:45:00 | 1645.20 | 1649.31 | 1643.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1645.90 | 1648.63 | 1644.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 1654.60 | 1647.79 | 1644.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1641.80 | 1650.98 | 1647.19 | SL hit (close<static) qty=1.00 sl=1643.70 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 1635.50 | 1644.66 | 1644.93 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1656.20 | 1646.28 | 1645.58 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1630.00 | 1643.57 | 1644.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 1619.70 | 1636.77 | 1641.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1614.70 | 1605.37 | 1612.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1614.70 | 1605.37 | 1612.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1614.70 | 1605.37 | 1612.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1619.80 | 1605.37 | 1612.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1607.90 | 1605.87 | 1611.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 1625.00 | 1605.87 | 1611.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1604.60 | 1605.24 | 1610.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1606.80 | 1605.24 | 1610.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1618.00 | 1607.79 | 1611.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1618.00 | 1607.79 | 1611.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1625.90 | 1611.41 | 1612.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1617.50 | 1611.41 | 1612.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1628.30 | 1615.76 | 1614.40 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 14:15:00 | 1605.00 | 1614.14 | 1614.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 1599.20 | 1610.86 | 1612.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1610.60 | 1607.25 | 1609.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1610.60 | 1607.25 | 1609.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1610.60 | 1607.25 | 1609.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:00:00 | 1595.80 | 1605.85 | 1608.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 1619.90 | 1611.52 | 1610.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 15:15:00 | 1619.90 | 1611.52 | 1610.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1635.80 | 1616.38 | 1613.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 1614.70 | 1621.25 | 1616.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 12:15:00 | 1614.70 | 1621.25 | 1616.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1614.70 | 1621.25 | 1616.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 1614.70 | 1621.25 | 1616.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1624.60 | 1621.92 | 1617.30 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1605.20 | 1614.07 | 1614.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 1593.30 | 1604.41 | 1609.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1576.00 | 1575.97 | 1588.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1574.30 | 1575.97 | 1588.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1588.10 | 1577.17 | 1586.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1588.10 | 1577.17 | 1586.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1589.80 | 1579.70 | 1587.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 1589.70 | 1579.70 | 1587.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1595.70 | 1582.90 | 1587.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 1587.10 | 1583.98 | 1587.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 1597.50 | 1588.25 | 1589.25 | SL hit (close>static) qty=1.00 sl=1597.10 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1606.00 | 1592.69 | 1591.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 1608.50 | 1599.48 | 1594.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1659.20 | 1659.79 | 1642.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1659.20 | 1659.79 | 1642.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1649.00 | 1656.71 | 1645.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 1636.00 | 1656.71 | 1645.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1635.80 | 1652.53 | 1644.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 1635.80 | 1652.53 | 1644.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1622.30 | 1646.48 | 1642.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 1622.30 | 1646.48 | 1642.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 1619.10 | 1635.78 | 1638.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 1601.70 | 1621.89 | 1630.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1518.60 | 1511.15 | 1531.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1518.60 | 1511.15 | 1531.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1549.80 | 1507.10 | 1514.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1549.80 | 1507.10 | 1514.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1539.80 | 1513.64 | 1516.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 1533.00 | 1517.91 | 1518.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 1533.30 | 1517.91 | 1518.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 12:15:00 | 1533.40 | 1521.01 | 1519.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 1533.40 | 1521.01 | 1519.92 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1510.90 | 1519.70 | 1519.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 1498.60 | 1515.48 | 1517.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1443.10 | 1430.98 | 1450.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1443.10 | 1430.98 | 1450.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1439.60 | 1432.59 | 1448.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 1430.10 | 1432.59 | 1448.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1358.59 | 1393.55 | 1410.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 1389.60 | 1378.54 | 1395.17 | SL hit (close>ema200) qty=0.50 sl=1378.54 alert=retest2 |

### Cycle 128 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1410.00 | 1401.84 | 1401.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 1421.90 | 1405.85 | 1403.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 1410.30 | 1411.50 | 1407.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 1410.30 | 1411.50 | 1407.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1410.30 | 1411.50 | 1407.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1408.90 | 1411.50 | 1407.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 1406.80 | 1410.56 | 1407.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 1406.80 | 1410.56 | 1407.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 1418.90 | 1412.23 | 1408.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:15:00 | 1426.50 | 1412.23 | 1408.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1426.50 | 1422.18 | 1414.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 1453.80 | 1455.69 | 1440.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 1569.15 | 1483.41 | 1465.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1570.80 | 1576.03 | 1576.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1491.10 | 1558.21 | 1567.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1527.10 | 1522.52 | 1534.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 1527.10 | 1522.52 | 1534.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1527.00 | 1523.42 | 1533.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1516.50 | 1523.42 | 1533.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 1542.40 | 1527.18 | 1532.14 | SL hit (close>static) qty=1.00 sl=1534.50 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1272.00 | 1253.09 | 1252.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1293.30 | 1264.00 | 1257.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1272.50 | 1294.83 | 1279.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1272.50 | 1294.83 | 1279.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1272.50 | 1294.83 | 1279.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1272.50 | 1294.83 | 1279.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1279.60 | 1291.79 | 1279.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1290.40 | 1278.71 | 1276.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 1261.60 | 1274.21 | 1275.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 1261.60 | 1274.21 | 1275.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1246.90 | 1268.75 | 1273.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1200.30 | 1198.21 | 1218.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 1208.50 | 1198.21 | 1218.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1238.00 | 1208.84 | 1217.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1238.60 | 1208.84 | 1217.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1231.70 | 1213.41 | 1218.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 1225.90 | 1213.41 | 1218.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:00:00 | 1227.20 | 1219.09 | 1220.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 1228.50 | 1220.97 | 1220.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 1228.50 | 1220.97 | 1220.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 1234.00 | 1223.58 | 1222.13 | Break + close above crossover candle high |

### Cycle 133 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1178.10 | 1214.48 | 1218.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1174.90 | 1206.57 | 1214.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1158.60 | 1143.05 | 1163.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1158.60 | 1143.05 | 1163.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1158.60 | 1143.05 | 1163.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 1140.00 | 1142.52 | 1161.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 1141.80 | 1146.53 | 1157.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1101.50 | 1146.23 | 1156.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1139.40 | 1138.55 | 1144.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1149.20 | 1144.14 | 1146.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 1157.40 | 1144.14 | 1146.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1193.40 | 1160.69 | 1153.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1310.40 | 1315.17 | 1286.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1329.50 | 1315.17 | 1286.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 1322.60 | 1323.31 | 1304.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1291.50 | 1317.32 | 1304.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1291.50 | 1317.32 | 1304.90 | SL hit (close<ema400) qty=1.00 sl=1304.90 alert=retest1 |

### Cycle 135 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 1380.00 | 1393.68 | 1393.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1360.40 | 1384.90 | 1389.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 1375.00 | 1373.04 | 1380.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1390.70 | 1373.04 | 1380.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1392.40 | 1376.91 | 1381.74 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1399.60 | 1385.05 | 1384.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 1408.00 | 1389.64 | 1386.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 1394.30 | 1400.01 | 1395.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 1394.30 | 1400.01 | 1395.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1394.30 | 1400.01 | 1395.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1394.30 | 1400.01 | 1395.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1402.90 | 1400.59 | 1396.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1413.50 | 1401.67 | 1397.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:00:00 | 1408.30 | 1413.79 | 1411.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 10:30:00 | 1545.00 | 2024-05-27 15:15:00 | 1607.10 | STOP_HIT | 1.00 | 4.02% |
| BUY | retest2 | 2024-06-26 09:15:00 | 2011.30 | 2024-06-26 13:15:00 | 1961.80 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-07-02 11:45:00 | 1845.95 | 2024-07-10 11:15:00 | 1753.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-02 12:30:00 | 1841.00 | 2024-07-10 11:15:00 | 1751.28 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-07-02 15:00:00 | 1843.45 | 2024-07-10 12:15:00 | 1748.95 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2024-07-02 11:45:00 | 1845.95 | 2024-07-10 14:15:00 | 1799.65 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2024-07-02 12:30:00 | 1841.00 | 2024-07-10 14:15:00 | 1799.65 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2024-07-02 15:00:00 | 1843.45 | 2024-07-10 14:15:00 | 1799.65 | STOP_HIT | 0.50 | 2.38% |
| BUY | retest2 | 2024-07-30 13:15:00 | 1852.40 | 2024-07-30 15:15:00 | 1849.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-07-30 14:00:00 | 1851.30 | 2024-07-30 15:15:00 | 1849.50 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-07-30 15:15:00 | 1849.50 | 2024-07-30 15:15:00 | 1849.50 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-06 10:45:00 | 1655.35 | 2024-08-08 12:15:00 | 1679.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-08-06 14:00:00 | 1647.65 | 2024-08-08 12:15:00 | 1679.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-08-06 14:45:00 | 1645.90 | 2024-08-08 12:15:00 | 1679.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-08-07 09:30:00 | 1640.60 | 2024-08-08 12:15:00 | 1679.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1818.45 | 2024-09-04 09:15:00 | 1763.20 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-09-03 10:45:00 | 1800.25 | 2024-09-04 09:15:00 | 1763.20 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-09-12 10:15:00 | 1828.05 | 2024-09-17 09:15:00 | 1814.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-09-12 13:45:00 | 1819.10 | 2024-09-17 09:15:00 | 1814.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-09-12 14:15:00 | 1824.15 | 2024-09-17 09:15:00 | 1814.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-09-16 14:45:00 | 1831.85 | 2024-09-17 09:15:00 | 1814.10 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest1 | 2024-09-27 09:15:00 | 1805.95 | 2024-09-27 14:15:00 | 1877.35 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2024-10-01 09:15:00 | 1824.05 | 2024-10-04 09:15:00 | 1732.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:15:00 | 1824.05 | 2024-10-07 14:15:00 | 1740.55 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1753.35 | 2024-10-23 09:15:00 | 1665.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1753.35 | 2024-10-23 12:15:00 | 1713.95 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2024-11-11 14:45:00 | 1610.10 | 2024-11-13 12:15:00 | 1529.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:30:00 | 1613.10 | 2024-11-13 12:15:00 | 1532.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 14:45:00 | 1610.10 | 2024-11-14 11:15:00 | 1543.60 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2024-11-12 09:30:00 | 1613.10 | 2024-11-14 11:15:00 | 1543.60 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2024-11-29 15:15:00 | 1641.75 | 2024-12-02 11:15:00 | 1659.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest1 | 2024-12-18 13:00:00 | 1884.85 | 2024-12-19 13:15:00 | 1849.70 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-12-27 11:15:00 | 1735.70 | 2024-12-30 10:15:00 | 1762.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-12-30 13:15:00 | 1739.25 | 2024-12-30 15:15:00 | 1652.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 13:15:00 | 1739.25 | 2024-12-31 15:15:00 | 1699.00 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2025-01-29 11:15:00 | 1271.75 | 2025-01-29 13:15:00 | 1311.65 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-01-31 15:15:00 | 1387.55 | 2025-02-05 12:15:00 | 1383.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-03-06 09:30:00 | 1183.45 | 2025-03-06 11:15:00 | 1163.25 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-03-06 13:45:00 | 1183.95 | 2025-03-06 15:15:00 | 1169.80 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-03-13 15:15:00 | 1122.00 | 2025-03-18 14:15:00 | 1132.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-03-18 11:45:00 | 1119.85 | 2025-03-18 14:15:00 | 1132.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-25 09:15:00 | 1230.40 | 2025-03-26 13:15:00 | 1215.35 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-03-25 11:45:00 | 1226.85 | 2025-03-26 13:15:00 | 1215.35 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-03-25 13:15:00 | 1226.05 | 2025-03-26 13:15:00 | 1215.35 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1254.00 | 2025-03-26 13:15:00 | 1215.35 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1189.85 | 2025-04-01 10:15:00 | 1130.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:00:00 | 1188.30 | 2025-04-01 11:15:00 | 1128.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1189.85 | 2025-04-02 09:15:00 | 1161.50 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2025-03-28 13:00:00 | 1188.30 | 2025-04-02 09:15:00 | 1161.50 | STOP_HIT | 0.50 | 2.26% |
| BUY | retest2 | 2025-04-24 13:15:00 | 1303.60 | 2025-04-24 13:15:00 | 1290.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-04-25 09:15:00 | 1305.20 | 2025-04-25 09:15:00 | 1271.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-05-05 09:45:00 | 1375.80 | 2025-05-06 11:15:00 | 1354.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-05-08 11:30:00 | 1318.30 | 2025-05-12 12:15:00 | 1338.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1313.80 | 2025-05-12 12:15:00 | 1338.50 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1365.00 | 2025-05-22 12:15:00 | 1415.00 | STOP_HIT | 1.00 | 3.66% |
| BUY | retest2 | 2025-06-02 09:15:00 | 1524.10 | 2025-06-06 10:15:00 | 1676.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1628.60 | 2025-06-16 10:15:00 | 1666.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-06-25 13:30:00 | 1721.70 | 2025-06-26 10:15:00 | 1707.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-26 10:30:00 | 1719.90 | 2025-06-26 11:15:00 | 1700.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-10 09:15:00 | 1667.80 | 2025-07-23 10:15:00 | 1741.00 | STOP_HIT | 1.00 | 4.39% |
| SELL | retest2 | 2025-08-01 14:15:00 | 1604.30 | 2025-08-04 13:15:00 | 1631.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-06 14:15:00 | 1604.40 | 2025-08-07 14:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-06 15:15:00 | 1600.00 | 2025-08-07 14:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-08-07 09:45:00 | 1604.00 | 2025-08-07 14:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-18 12:00:00 | 1623.20 | 2025-08-18 13:15:00 | 1633.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-25 10:15:00 | 1663.70 | 2025-08-26 09:15:00 | 1625.20 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-25 11:45:00 | 1660.50 | 2025-08-26 09:15:00 | 1625.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-25 12:45:00 | 1656.30 | 2025-08-26 09:15:00 | 1625.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2025-08-29 09:15:00 | 1566.20 | 2025-09-01 14:15:00 | 1576.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-08-29 11:30:00 | 1576.10 | 2025-09-01 14:15:00 | 1576.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest1 | 2025-08-29 12:15:00 | 1572.40 | 2025-09-01 14:15:00 | 1576.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-02 14:00:00 | 1562.50 | 2025-09-03 14:15:00 | 1585.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1532.00 | 2025-09-10 09:15:00 | 1548.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-12 09:15:00 | 1564.50 | 2025-09-12 11:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-12 10:00:00 | 1566.90 | 2025-09-12 11:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1514.60 | 2025-10-01 13:15:00 | 1526.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-09-30 09:45:00 | 1518.80 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-01 10:00:00 | 1518.80 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-01 10:45:00 | 1518.60 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-10-01 12:45:00 | 1514.50 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-10-14 14:30:00 | 1606.00 | 2025-10-24 12:15:00 | 1764.84 | TARGET_HIT | 1.00 | 9.89% |
| BUY | retest2 | 2025-10-14 15:15:00 | 1604.40 | 2025-10-27 09:15:00 | 1766.60 | TARGET_HIT | 1.00 | 10.11% |
| SELL | retest2 | 2025-11-21 11:00:00 | 1683.00 | 2025-11-28 13:15:00 | 1675.80 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-11-21 14:15:00 | 1691.00 | 2025-11-28 13:15:00 | 1675.80 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-12-09 14:45:00 | 1629.10 | 2025-12-11 09:15:00 | 1650.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1618.40 | 2025-12-11 09:15:00 | 1650.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1627.40 | 2025-12-11 10:15:00 | 1651.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-12 13:15:00 | 1654.60 | 2025-12-15 09:15:00 | 1641.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-24 13:00:00 | 1595.80 | 2025-12-24 15:15:00 | 1619.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-31 13:30:00 | 1587.10 | 2025-12-31 15:15:00 | 1597.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-16 11:45:00 | 1533.00 | 2026-01-16 12:15:00 | 1533.40 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-01-16 12:15:00 | 1533.30 | 2026-01-16 12:15:00 | 1533.40 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-22 10:15:00 | 1430.10 | 2026-01-27 09:15:00 | 1358.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 1430.10 | 2026-01-27 14:15:00 | 1389.60 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest2 | 2026-01-29 13:15:00 | 1426.50 | 2026-02-03 09:15:00 | 1569.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 1426.50 | 2026-02-03 09:15:00 | 1569.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 12:30:00 | 1453.80 | 2026-02-03 09:15:00 | 1599.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1516.50 | 2026-02-17 12:15:00 | 1542.40 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-17 15:15:00 | 1520.90 | 2026-02-24 11:15:00 | 1444.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1514.00 | 2026-02-24 12:15:00 | 1438.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1514.50 | 2026-02-24 12:15:00 | 1438.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:00:00 | 1482.40 | 2026-02-24 14:15:00 | 1408.47 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-02-17 15:15:00 | 1520.90 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1514.00 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1514.50 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2026-02-23 12:00:00 | 1482.40 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2026-02-23 12:45:00 | 1482.60 | 2026-02-27 09:15:00 | 1408.28 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1482.20 | 2026-02-27 09:15:00 | 1408.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1467.30 | 2026-02-27 11:15:00 | 1393.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:45:00 | 1482.60 | 2026-03-02 09:15:00 | 1334.16 | TARGET_HIT | 0.50 | 10.01% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1482.20 | 2026-03-02 09:15:00 | 1333.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1445.90 | 2026-03-02 09:15:00 | 1373.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1467.30 | 2026-03-04 10:15:00 | 1320.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1445.90 | 2026-03-04 14:15:00 | 1352.90 | STOP_HIT | 0.50 | 6.43% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1290.40 | 2026-03-20 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-25 11:15:00 | 1225.90 | 2026-03-25 14:15:00 | 1228.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1227.20 | 2026-03-25 14:15:00 | 1228.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1140.00 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1141.80 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1101.50 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest2 | 2026-04-06 09:15:00 | 1139.40 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1329.50 | 2026-04-13 09:15:00 | 1291.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest1 | 2026-04-10 15:00:00 | 1322.60 | 2026-04-13 09:15:00 | 1291.50 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1303.50 | 2026-04-23 15:15:00 | 1380.00 | STOP_HIT | 1.00 | 5.87% |
| BUY | retest2 | 2026-04-13 11:15:00 | 1303.00 | 2026-04-23 15:15:00 | 1380.00 | STOP_HIT | 1.00 | 5.91% |
