# Concord Biotech Ltd. (CONCORDBIO)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1168.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 40 |
| ALERT2 | 39 |
| ALERT2_SKIP | 19 |
| ALERT3 | 103 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 50 |
| PARTIAL | 22 |
| TARGET_HIT | 3 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 21
- **Target hits / Stop hits / Partials:** 3 / 48 / 22
- **Avg / median % per leg:** 2.76% / 3.30%
- **Sum % (uncompounded):** 201.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 1 | 9 | 0 | 1.59% | 15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 1 | 9 | 0 | 1.59% | 15.9% |
| SELL (all) | 63 | 46 | 73.0% | 2 | 39 | 22 | 2.95% | 185.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.51% | -1.5% |
| SELL @ 3rd Alert (retest2) | 62 | 46 | 74.2% | 2 | 38 | 22 | 3.02% | 187.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.51% | -1.5% |
| retest2 (combined) | 72 | 52 | 72.2% | 3 | 47 | 22 | 2.82% | 203.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 14:15:00 | 1497.50 | 1513.94 | 1514.16 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1531.80 | 1515.73 | 1514.83 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1511.90 | 1519.30 | 1519.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 13:15:00 | 1508.40 | 1514.38 | 1516.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 1519.40 | 1512.26 | 1514.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 1519.40 | 1512.26 | 1514.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1519.40 | 1512.26 | 1514.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1519.40 | 1512.26 | 1514.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1523.00 | 1514.41 | 1515.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 1523.00 | 1514.41 | 1515.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 1536.80 | 1519.64 | 1517.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1538.30 | 1531.58 | 1525.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 1536.00 | 1536.04 | 1530.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 13:00:00 | 1536.00 | 1536.04 | 1530.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1530.80 | 1534.99 | 1530.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1530.80 | 1534.99 | 1530.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1535.70 | 1535.13 | 1531.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:15:00 | 1534.00 | 1535.13 | 1531.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1534.00 | 1534.90 | 1531.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1534.00 | 1534.90 | 1531.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1540.10 | 1535.94 | 1532.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 1557.00 | 1536.91 | 1534.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 09:15:00 | 1712.70 | 1679.78 | 1634.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 2074.50 | 2093.01 | 2093.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 09:15:00 | 2019.30 | 2074.56 | 2084.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 15:15:00 | 2035.00 | 2026.63 | 2051.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 2036.70 | 2028.64 | 2050.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2036.70 | 2028.64 | 2050.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 2036.70 | 2028.64 | 2050.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1940.90 | 1984.46 | 2015.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1935.10 | 1984.46 | 2015.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 1935.70 | 1974.71 | 2007.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1838.34 | 1918.85 | 1968.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1838.91 | 1918.85 | 1968.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 1821.70 | 1799.40 | 1836.93 | SL hit (close>ema200) qty=0.50 sl=1799.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 1821.70 | 1799.40 | 1836.93 | SL hit (close>ema200) qty=0.50 sl=1799.40 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 14:15:00 | 1839.10 | 1823.45 | 1822.37 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1817.30 | 1822.51 | 1822.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 1800.00 | 1815.14 | 1818.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 1760.90 | 1760.57 | 1777.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:30:00 | 1760.60 | 1760.57 | 1777.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1778.60 | 1751.84 | 1760.09 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 1774.00 | 1765.05 | 1764.59 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 1760.90 | 1764.22 | 1764.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 1756.60 | 1762.69 | 1763.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1753.30 | 1749.70 | 1755.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1753.30 | 1749.70 | 1755.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1753.30 | 1749.70 | 1755.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 1755.80 | 1749.70 | 1755.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1777.70 | 1755.30 | 1757.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 1777.70 | 1755.30 | 1757.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 1789.90 | 1762.22 | 1760.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1827.30 | 1788.20 | 1774.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1813.60 | 1818.27 | 1799.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:30:00 | 1804.30 | 1818.27 | 1799.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1796.90 | 1813.99 | 1798.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1796.90 | 1813.99 | 1798.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1800.00 | 1811.19 | 1799.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 1801.90 | 1811.19 | 1799.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1811.60 | 1811.28 | 1800.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 1816.10 | 1811.28 | 1800.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1803.00 | 1809.62 | 1800.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1804.90 | 1809.62 | 1800.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1798.80 | 1806.76 | 1801.38 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1790.80 | 1797.84 | 1798.15 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 1805.30 | 1799.34 | 1798.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 1837.70 | 1808.17 | 1803.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 1833.00 | 1835.74 | 1824.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 13:00:00 | 1833.00 | 1835.74 | 1824.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1828.20 | 1834.58 | 1826.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1828.20 | 1834.58 | 1826.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1846.70 | 1837.00 | 1827.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 1849.00 | 1837.44 | 1829.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 1849.90 | 1840.45 | 1832.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 1848.10 | 1841.48 | 1833.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1848.90 | 1839.95 | 1834.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1847.80 | 1841.52 | 1835.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:45:00 | 1876.70 | 1845.44 | 1837.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 1898.30 | 1856.01 | 1843.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 1896.00 | 1918.89 | 1923.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 13:15:00 | 1698.70 | 1698.00 | 1724.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 13:45:00 | 1706.20 | 1698.00 | 1724.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1656.50 | 1689.40 | 1713.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 1654.10 | 1689.40 | 1713.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1650.00 | 1663.54 | 1689.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-11 09:15:00 | 1488.69 | 1599.52 | 1626.98 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-11 09:15:00 | 1485.00 | 1599.52 | 1626.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1671.10 | 1627.13 | 1623.26 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 1631.70 | 1636.16 | 1636.34 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 1647.00 | 1637.99 | 1637.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1653.30 | 1645.88 | 1642.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 1743.00 | 1757.54 | 1735.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:15:00 | 1734.90 | 1757.54 | 1735.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1731.00 | 1752.23 | 1735.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 1730.90 | 1752.23 | 1735.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1728.20 | 1747.43 | 1734.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1728.20 | 1747.43 | 1734.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1734.00 | 1744.74 | 1734.40 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1703.60 | 1726.58 | 1728.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1689.20 | 1709.65 | 1718.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 14:15:00 | 1700.20 | 1697.96 | 1708.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 14:45:00 | 1701.00 | 1697.96 | 1708.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1702.20 | 1695.11 | 1704.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 1713.60 | 1695.11 | 1704.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1708.30 | 1697.74 | 1705.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1708.30 | 1697.74 | 1705.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1701.70 | 1698.54 | 1704.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1703.40 | 1698.54 | 1704.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1698.70 | 1698.57 | 1704.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 1692.90 | 1697.43 | 1703.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 1692.00 | 1691.98 | 1698.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 1685.50 | 1692.82 | 1698.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1682.00 | 1689.71 | 1694.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1695.80 | 1690.93 | 1694.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:45:00 | 1674.10 | 1686.55 | 1691.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1657.30 | 1687.25 | 1690.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 1671.40 | 1675.18 | 1683.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 1672.50 | 1681.97 | 1684.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1672.60 | 1678.55 | 1682.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 1672.60 | 1678.55 | 1682.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1687.00 | 1677.07 | 1679.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1687.00 | 1677.07 | 1679.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1679.30 | 1677.51 | 1679.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:00:00 | 1665.90 | 1675.19 | 1678.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 1663.30 | 1664.86 | 1671.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 15:15:00 | 1608.26 | 1633.11 | 1650.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 15:15:00 | 1607.40 | 1633.11 | 1650.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1601.22 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1597.90 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1590.39 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1574.43 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1587.83 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1588.88 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1582.61 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1580.13 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 1673.50 | 1640.79 | 1637.71 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1630.00 | 1637.85 | 1638.74 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1655.50 | 1642.43 | 1640.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 1664.00 | 1646.74 | 1642.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1646.70 | 1651.56 | 1647.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1646.70 | 1651.56 | 1647.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1646.70 | 1651.56 | 1647.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 1644.40 | 1651.56 | 1647.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1654.00 | 1652.05 | 1648.10 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1624.00 | 1642.08 | 1644.45 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 1670.00 | 1642.23 | 1638.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1679.30 | 1649.65 | 1642.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1660.70 | 1674.40 | 1661.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1660.70 | 1674.40 | 1661.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1660.70 | 1674.40 | 1661.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:15:00 | 1656.00 | 1674.40 | 1661.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1647.00 | 1668.92 | 1660.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1647.00 | 1668.92 | 1660.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1630.30 | 1661.20 | 1657.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 1630.30 | 1661.20 | 1657.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 1625.00 | 1653.96 | 1654.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1620.60 | 1642.86 | 1649.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1631.40 | 1630.35 | 1640.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:45:00 | 1629.80 | 1630.35 | 1640.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1630.90 | 1629.18 | 1635.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1630.90 | 1629.18 | 1635.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1634.40 | 1627.99 | 1632.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 1631.00 | 1627.99 | 1632.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1626.60 | 1627.72 | 1632.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:45:00 | 1632.30 | 1627.72 | 1632.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1630.60 | 1628.29 | 1632.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1618.80 | 1628.29 | 1632.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1618.70 | 1626.37 | 1630.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:45:00 | 1589.50 | 1615.51 | 1624.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 1599.60 | 1590.09 | 1589.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 1599.60 | 1590.09 | 1589.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 1610.70 | 1594.22 | 1591.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 15:15:00 | 1630.30 | 1631.65 | 1620.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 1627.20 | 1631.65 | 1620.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 1604.40 | 1624.52 | 1621.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 1604.40 | 1624.52 | 1621.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 14:15:00 | 1592.90 | 1618.20 | 1618.78 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1630.70 | 1615.46 | 1614.73 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1605.90 | 1618.69 | 1619.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1593.20 | 1611.72 | 1616.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1609.80 | 1601.83 | 1607.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1609.80 | 1601.83 | 1607.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1609.80 | 1601.83 | 1607.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 1609.80 | 1601.83 | 1607.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1590.10 | 1599.49 | 1606.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1585.50 | 1597.29 | 1604.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 1587.30 | 1595.00 | 1601.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 1586.00 | 1594.72 | 1600.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 1507.93 | 1530.73 | 1553.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 1506.22 | 1524.76 | 1549.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 1506.70 | 1524.76 | 1549.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1509.60 | 1509.43 | 1529.26 | SL hit (close>ema200) qty=0.50 sl=1509.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1509.60 | 1509.43 | 1529.26 | SL hit (close>ema200) qty=0.50 sl=1509.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1509.60 | 1509.43 | 1529.26 | SL hit (close>ema200) qty=0.50 sl=1509.43 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 1516.80 | 1514.43 | 1514.41 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1505.90 | 1512.73 | 1513.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1497.00 | 1509.47 | 1511.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1444.40 | 1442.73 | 1457.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:15:00 | 1435.20 | 1442.73 | 1457.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1450.00 | 1447.58 | 1454.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 1453.30 | 1447.58 | 1454.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1456.80 | 1449.42 | 1454.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 1456.80 | 1449.42 | 1454.91 | SL hit (close>ema400) qty=1.00 sl=1454.91 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 1453.30 | 1449.42 | 1454.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1446.40 | 1448.82 | 1454.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 1437.50 | 1448.57 | 1451.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1468.00 | 1452.62 | 1453.15 | SL hit (close>static) qty=1.00 sl=1463.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1463.20 | 1454.99 | 1454.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1467.00 | 1458.51 | 1455.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1458.00 | 1461.19 | 1458.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1458.00 | 1461.19 | 1458.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1458.00 | 1461.19 | 1458.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1458.00 | 1461.19 | 1458.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1459.30 | 1460.82 | 1458.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 1458.00 | 1460.82 | 1458.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1465.00 | 1461.65 | 1458.81 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1432.30 | 1453.26 | 1455.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1424.00 | 1447.41 | 1452.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1402.20 | 1399.38 | 1415.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 15:15:00 | 1402.20 | 1399.38 | 1415.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1402.20 | 1399.38 | 1415.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1421.00 | 1403.70 | 1416.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1432.00 | 1409.36 | 1417.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1432.00 | 1409.36 | 1417.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1435.20 | 1421.96 | 1421.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 1439.50 | 1425.47 | 1423.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1422.30 | 1426.41 | 1424.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1422.30 | 1426.41 | 1424.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1422.30 | 1426.41 | 1424.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1416.00 | 1426.41 | 1424.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1416.90 | 1424.51 | 1423.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 1415.00 | 1424.51 | 1423.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1457.40 | 1471.67 | 1460.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1457.40 | 1471.67 | 1460.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1463.80 | 1470.09 | 1461.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1464.30 | 1470.09 | 1461.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1460.90 | 1468.26 | 1461.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1488.10 | 1471.54 | 1463.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1455.20 | 1485.67 | 1487.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1455.20 | 1485.67 | 1487.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 1448.50 | 1466.87 | 1477.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 1438.10 | 1432.69 | 1441.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:30:00 | 1437.80 | 1432.69 | 1441.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1440.20 | 1413.84 | 1421.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1440.20 | 1413.84 | 1421.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1436.10 | 1418.29 | 1422.61 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1439.90 | 1426.25 | 1425.71 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 1413.60 | 1425.49 | 1426.95 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1440.00 | 1425.96 | 1425.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 1442.00 | 1431.06 | 1428.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 15:15:00 | 1430.00 | 1431.05 | 1428.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 1425.70 | 1431.05 | 1428.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1431.90 | 1431.22 | 1429.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1426.10 | 1431.22 | 1429.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1439.90 | 1432.62 | 1430.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 1432.10 | 1432.62 | 1430.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1419.60 | 1431.75 | 1430.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 1419.60 | 1431.75 | 1430.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 1421.00 | 1429.60 | 1429.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 1407.00 | 1423.61 | 1427.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1417.10 | 1415.42 | 1421.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1417.10 | 1415.42 | 1421.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1417.10 | 1415.42 | 1421.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 1417.10 | 1415.42 | 1421.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1415.20 | 1415.38 | 1420.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 1410.30 | 1415.28 | 1419.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1426.90 | 1414.53 | 1417.32 | SL hit (close>static) qty=1.00 sl=1421.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 1406.90 | 1414.62 | 1416.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:15:00 | 1336.56 | 1367.44 | 1386.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 1365.30 | 1360.59 | 1377.97 | SL hit (close>ema200) qty=0.50 sl=1360.59 alert=retest2 |

### Cycle 38 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 1394.60 | 1385.05 | 1384.42 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1379.90 | 1383.79 | 1384.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1365.80 | 1380.19 | 1382.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1374.70 | 1373.77 | 1377.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1374.70 | 1373.77 | 1377.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1374.70 | 1373.77 | 1377.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1366.00 | 1372.75 | 1376.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 1367.60 | 1371.58 | 1375.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1359.10 | 1370.79 | 1373.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 1380.00 | 1375.74 | 1375.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 1380.00 | 1375.74 | 1375.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 1380.00 | 1375.74 | 1375.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1380.00 | 1375.74 | 1375.34 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 1365.00 | 1373.52 | 1374.39 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 14:15:00 | 1384.00 | 1374.27 | 1373.66 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1357.20 | 1370.17 | 1371.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1348.00 | 1362.64 | 1367.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 1340.30 | 1337.20 | 1347.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 1340.30 | 1337.20 | 1347.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1348.00 | 1339.36 | 1347.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1361.00 | 1339.36 | 1347.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1352.40 | 1341.97 | 1347.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 1343.80 | 1342.34 | 1347.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 1336.00 | 1345.38 | 1345.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:30:00 | 1343.00 | 1339.97 | 1339.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1351.60 | 1342.29 | 1341.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1351.60 | 1342.29 | 1341.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1351.60 | 1342.29 | 1341.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1351.60 | 1342.29 | 1341.04 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1340.00 | 1342.58 | 1342.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1331.20 | 1340.20 | 1341.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1344.60 | 1338.23 | 1339.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 1344.60 | 1338.23 | 1339.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1344.60 | 1338.23 | 1339.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1344.60 | 1338.23 | 1339.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1336.00 | 1337.78 | 1339.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 1334.00 | 1337.78 | 1339.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 1368.00 | 1343.22 | 1341.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 1368.00 | 1343.22 | 1341.65 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 1339.20 | 1345.03 | 1345.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1324.50 | 1338.85 | 1342.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1329.60 | 1319.71 | 1327.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1329.60 | 1319.71 | 1327.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1329.60 | 1319.71 | 1327.95 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 14:15:00 | 1339.20 | 1332.80 | 1332.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 12:15:00 | 1347.20 | 1336.61 | 1334.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 14:15:00 | 1367.90 | 1377.46 | 1361.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 1367.90 | 1377.46 | 1361.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1370.00 | 1375.96 | 1362.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1353.30 | 1375.96 | 1362.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1343.30 | 1369.43 | 1360.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1343.30 | 1369.43 | 1360.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1331.20 | 1361.79 | 1358.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1331.20 | 1361.79 | 1358.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1337.20 | 1353.14 | 1354.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1332.10 | 1345.34 | 1350.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1332.00 | 1322.17 | 1331.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 1332.00 | 1322.17 | 1331.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1332.00 | 1322.17 | 1331.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 1332.00 | 1322.17 | 1331.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1330.40 | 1323.82 | 1331.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1335.80 | 1323.82 | 1331.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1331.70 | 1325.40 | 1331.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 1331.90 | 1325.40 | 1331.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1336.90 | 1327.70 | 1332.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1341.00 | 1327.70 | 1332.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1334.00 | 1328.96 | 1332.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1341.90 | 1328.96 | 1332.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1356.60 | 1337.28 | 1335.73 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 1326.60 | 1340.77 | 1342.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1324.60 | 1337.53 | 1340.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 14:15:00 | 1346.20 | 1339.27 | 1341.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 1346.20 | 1339.27 | 1341.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1346.20 | 1339.27 | 1341.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1346.20 | 1339.27 | 1341.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1356.30 | 1342.67 | 1342.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1342.50 | 1342.67 | 1342.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1346.20 | 1343.38 | 1343.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 1346.20 | 1343.38 | 1343.11 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1325.10 | 1339.72 | 1341.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1313.00 | 1334.38 | 1338.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1253.50 | 1246.40 | 1268.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 1253.50 | 1246.40 | 1268.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1259.20 | 1251.50 | 1263.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1259.20 | 1251.50 | 1263.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1262.20 | 1253.64 | 1263.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1263.50 | 1253.64 | 1263.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 1250.40 | 1252.99 | 1262.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 1249.80 | 1252.54 | 1260.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 1247.20 | 1252.54 | 1260.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:15:00 | 1187.31 | 1226.86 | 1240.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:15:00 | 1184.84 | 1226.86 | 1240.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1208.50 | 1205.12 | 1219.26 | SL hit (close>ema200) qty=0.50 sl=1205.12 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1208.50 | 1205.12 | 1219.26 | SL hit (close>ema200) qty=0.50 sl=1205.12 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 1169.60 | 1154.62 | 1153.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 1190.60 | 1164.92 | 1158.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1164.40 | 1169.18 | 1162.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1164.40 | 1169.18 | 1162.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1164.40 | 1169.18 | 1162.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1165.70 | 1169.18 | 1162.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1163.40 | 1168.02 | 1162.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:30:00 | 1162.50 | 1168.02 | 1162.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1152.00 | 1164.82 | 1161.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1152.00 | 1164.82 | 1161.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1149.20 | 1161.69 | 1160.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 1147.30 | 1161.69 | 1160.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1147.20 | 1158.79 | 1159.37 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1177.70 | 1162.58 | 1161.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1243.00 | 1177.69 | 1168.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 1267.50 | 1275.71 | 1250.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:15:00 | 1265.00 | 1275.71 | 1250.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1257.10 | 1274.75 | 1263.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1257.50 | 1274.75 | 1263.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1252.60 | 1270.32 | 1262.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 1252.70 | 1270.32 | 1262.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1246.50 | 1265.56 | 1261.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 1249.00 | 1265.56 | 1261.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 1240.80 | 1257.68 | 1258.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1235.00 | 1249.55 | 1254.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 1150.90 | 1149.55 | 1162.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:00:00 | 1150.90 | 1149.55 | 1162.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1146.60 | 1148.70 | 1158.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 1143.60 | 1148.70 | 1158.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 1143.00 | 1146.47 | 1155.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 1143.60 | 1143.71 | 1152.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1086.42 | 1102.13 | 1118.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1085.85 | 1102.13 | 1118.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1086.42 | 1102.13 | 1118.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1097.80 | 1096.18 | 1111.09 | SL hit (close>ema200) qty=0.50 sl=1096.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1097.80 | 1096.18 | 1111.09 | SL hit (close>ema200) qty=0.50 sl=1096.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1097.80 | 1096.18 | 1111.09 | SL hit (close>ema200) qty=0.50 sl=1096.18 alert=retest2 |

### Cycle 58 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1209.50 | 1121.61 | 1110.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 12:15:00 | 1229.90 | 1143.27 | 1121.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 1226.10 | 1232.83 | 1204.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:45:00 | 1228.10 | 1232.83 | 1204.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1213.10 | 1223.11 | 1209.88 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1175.60 | 1201.97 | 1204.47 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 1213.20 | 1203.12 | 1202.30 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1169.00 | 1198.86 | 1201.48 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1207.40 | 1193.79 | 1192.80 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1165.90 | 1188.00 | 1190.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1127.10 | 1168.70 | 1180.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1173.80 | 1167.08 | 1176.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1173.80 | 1167.08 | 1176.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1173.20 | 1168.30 | 1176.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1156.00 | 1171.38 | 1176.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 13:15:00 | 1188.50 | 1170.17 | 1173.19 | SL hit (close>static) qty=1.00 sl=1182.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 1167.80 | 1170.73 | 1172.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 14:15:00 | 1109.41 | 1123.28 | 1134.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 1112.00 | 1110.66 | 1120.75 | SL hit (close>ema200) qty=0.50 sl=1110.66 alert=retest2 |

### Cycle 64 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1036.50 | 1025.55 | 1024.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1057.80 | 1034.73 | 1029.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 1058.90 | 1065.52 | 1054.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 1058.90 | 1065.52 | 1054.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1058.90 | 1065.52 | 1054.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 1058.90 | 1065.52 | 1054.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1060.00 | 1064.42 | 1055.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1069.50 | 1064.42 | 1055.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1046.50 | 1060.86 | 1059.08 | SL hit (close<static) qty=1.00 sl=1052.10 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 1048.00 | 1056.43 | 1057.36 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1060.40 | 1057.60 | 1057.40 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 11:15:00 | 1049.00 | 1055.88 | 1056.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 12:15:00 | 1044.60 | 1053.62 | 1055.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1048.00 | 1047.54 | 1051.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1048.00 | 1047.54 | 1051.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1048.00 | 1047.54 | 1051.61 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 1094.50 | 1057.62 | 1053.11 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 1056.70 | 1062.36 | 1062.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 1048.90 | 1059.67 | 1061.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 14:15:00 | 1052.00 | 1050.46 | 1054.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 14:15:00 | 1052.00 | 1050.46 | 1054.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1052.00 | 1050.46 | 1054.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1046.90 | 1050.89 | 1054.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1065.30 | 1053.77 | 1055.39 | SL hit (close>static) qty=1.00 sl=1056.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:15:00 | 1048.00 | 1054.37 | 1055.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 1060.10 | 1055.06 | 1055.53 | SL hit (close>static) qty=1.00 sl=1056.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:45:00 | 1044.90 | 1053.36 | 1054.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 1044.30 | 1053.36 | 1054.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1061.50 | 1035.33 | 1042.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 1061.50 | 1035.33 | 1042.03 | SL hit (close>static) qty=1.00 sl=1056.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 1061.50 | 1035.33 | 1042.03 | SL hit (close>static) qty=1.00 sl=1056.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1061.50 | 1035.33 | 1042.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1063.90 | 1041.05 | 1044.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1071.40 | 1041.05 | 1044.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1067.60 | 1046.36 | 1046.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1071.70 | 1057.77 | 1052.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 1071.30 | 1075.36 | 1064.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 1071.30 | 1075.36 | 1064.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1173.60 | 1179.71 | 1159.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 1153.00 | 1179.71 | 1159.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1200.00 | 1210.22 | 1195.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 1198.00 | 1210.22 | 1195.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1197.90 | 1205.93 | 1195.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 1197.90 | 1205.93 | 1195.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1199.90 | 1204.72 | 1196.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1201.50 | 1201.53 | 1196.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 1183.50 | 1195.21 | 1194.60 | SL hit (close<static) qty=1.00 sl=1191.60 alert=retest2 |

### Cycle 71 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 1174.50 | 1191.07 | 1192.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 1168.40 | 1184.25 | 1189.24 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-28 09:30:00 | 1557.00 | 2025-05-30 09:15:00 | 1712.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-20 10:15:00 | 1935.10 | 2025-06-20 14:15:00 | 1838.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-20 11:00:00 | 1935.70 | 2025-06-20 14:15:00 | 1838.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-20 10:15:00 | 1935.10 | 2025-06-25 09:15:00 | 1821.70 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2025-06-20 11:00:00 | 1935.70 | 2025-06-25 09:15:00 | 1821.70 | STOP_HIT | 0.50 | 5.89% |
| BUY | retest2 | 2025-07-16 11:15:00 | 1849.00 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2025-07-16 12:30:00 | 1849.90 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2025-07-16 14:15:00 | 1848.10 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1848.90 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2025-07-17 10:45:00 | 1876.70 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-07-17 12:00:00 | 1898.30 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-08-06 10:15:00 | 1654.10 | 2025-08-11 09:15:00 | 1488.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-06 15:15:00 | 1650.00 | 2025-08-11 09:15:00 | 1485.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 14:00:00 | 1692.90 | 2025-09-08 15:15:00 | 1608.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 10:00:00 | 1692.00 | 2025-09-08 15:15:00 | 1607.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 11:15:00 | 1685.50 | 2025-09-09 14:15:00 | 1601.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1682.00 | 2025-09-09 14:15:00 | 1597.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-02 12:45:00 | 1674.10 | 2025-09-09 14:15:00 | 1590.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-03 09:15:00 | 1657.30 | 2025-09-09 14:15:00 | 1574.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-03 12:15:00 | 1671.40 | 2025-09-09 14:15:00 | 1587.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1672.50 | 2025-09-09 14:15:00 | 1588.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1665.90 | 2025-09-09 14:15:00 | 1582.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 09:45:00 | 1663.30 | 2025-09-09 14:15:00 | 1580.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 14:00:00 | 1692.90 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-09-01 10:00:00 | 1692.00 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-09-01 11:15:00 | 1685.50 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1682.00 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-09-02 12:45:00 | 1674.10 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2025-09-03 09:15:00 | 1657.30 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-09-03 12:15:00 | 1671.40 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1672.50 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1665.90 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2025-09-08 09:45:00 | 1663.30 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2025-09-25 14:45:00 | 1589.50 | 2025-09-30 10:15:00 | 1599.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1585.50 | 2025-10-15 09:15:00 | 1507.93 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-10-10 15:00:00 | 1587.30 | 2025-10-15 10:15:00 | 1506.22 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-10-13 10:00:00 | 1586.00 | 2025-10-15 10:15:00 | 1506.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1585.50 | 2025-10-16 09:15:00 | 1509.60 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-10-10 15:00:00 | 1587.30 | 2025-10-16 09:15:00 | 1509.60 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-13 10:00:00 | 1586.00 | 2025-10-16 09:15:00 | 1509.60 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest1 | 2025-10-29 10:15:00 | 1435.20 | 2025-10-29 15:15:00 | 1456.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-31 15:00:00 | 1437.50 | 2025-11-03 09:15:00 | 1468.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1488.10 | 2025-11-19 10:15:00 | 1455.20 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-04 14:15:00 | 1410.30 | 2025-12-05 10:15:00 | 1426.90 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-05 14:45:00 | 1406.90 | 2025-12-09 10:15:00 | 1336.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:45:00 | 1406.90 | 2025-12-09 13:15:00 | 1365.30 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1366.00 | 2025-12-15 12:15:00 | 1380.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-12 11:45:00 | 1367.60 | 2025-12-15 12:15:00 | 1380.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1359.10 | 2025-12-15 12:15:00 | 1380.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-19 11:00:00 | 1343.80 | 2025-12-24 10:15:00 | 1351.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-22 12:15:00 | 1336.00 | 2025-12-24 10:15:00 | 1351.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-12-24 09:30:00 | 1343.00 | 2025-12-24 10:15:00 | 1351.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-29 15:15:00 | 1334.00 | 2025-12-30 09:15:00 | 1368.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1342.50 | 2026-01-19 09:15:00 | 1346.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-01-23 10:45:00 | 1249.80 | 2026-01-28 09:15:00 | 1187.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 1247.20 | 2026-01-28 09:15:00 | 1184.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 1249.80 | 2026-01-29 10:15:00 | 1208.50 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2026-01-23 11:15:00 | 1247.20 | 2026-01-29 10:15:00 | 1208.50 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1143.60 | 2026-02-23 11:15:00 | 1086.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 1143.00 | 2026-02-23 11:15:00 | 1085.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:30:00 | 1143.60 | 2026-02-23 11:15:00 | 1086.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1143.60 | 2026-02-23 14:15:00 | 1097.80 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 1143.00 | 2026-02-23 14:15:00 | 1097.80 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2026-02-19 13:30:00 | 1143.60 | 2026-02-23 14:15:00 | 1097.80 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1156.00 | 2026-03-13 13:15:00 | 1188.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-16 09:30:00 | 1167.80 | 2026-03-19 14:15:00 | 1109.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:30:00 | 1167.80 | 2026-03-20 15:15:00 | 1112.00 | STOP_HIT | 0.50 | 4.78% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1069.50 | 2026-04-13 09:15:00 | 1046.50 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1046.90 | 2026-04-23 09:15:00 | 1065.30 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-23 12:15:00 | 1048.00 | 2026-04-23 13:15:00 | 1060.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-04-24 09:45:00 | 1044.90 | 2026-04-27 09:15:00 | 1061.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-24 10:15:00 | 1044.30 | 2026-04-27 09:15:00 | 1061.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-05-08 09:15:00 | 1201.50 | 2026-05-08 12:15:00 | 1183.50 | STOP_HIT | 1.00 | -1.50% |
