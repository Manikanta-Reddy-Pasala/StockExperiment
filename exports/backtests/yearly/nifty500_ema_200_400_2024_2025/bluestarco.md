# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1691.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 4 / 34
- **Target hits / Stop hits / Partials:** 1 / 36 / 1
- **Avg / median % per leg:** -2.15% / -2.08%
- **Sum % (uncompounded):** -81.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 1 | 4.0% | 1 | 24 | 0 | -2.66% | -66.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 1 | 4.0% | 1 | 24 | 0 | -2.66% | -66.4% |
| SELL (all) | 13 | 3 | 23.1% | 0 | 12 | 1 | -1.19% | -15.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 0 | 12 | 1 | -1.19% | -15.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 4 | 10.5% | 1 | 36 | 1 | -2.15% | -81.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 1772.00 | 1957.63 | 1958.37 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 2080.25 | 1949.27 | 1948.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 2136.50 | 1954.13 | 1951.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 2079.95 | 2088.20 | 2035.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 11:00:00 | 2079.95 | 2088.20 | 2035.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2064.00 | 2088.65 | 2041.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 2055.00 | 2088.65 | 2041.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 2032.80 | 2088.09 | 2040.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 2032.80 | 2088.09 | 2040.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 2039.30 | 2087.61 | 2040.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:15:00 | 2050.80 | 2049.20 | 2028.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:45:00 | 2050.30 | 2049.22 | 2028.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 2023.60 | 2049.02 | 2028.67 | SL hit (close<static) qty=1.00 sl=2031.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 1743.60 | 2013.53 | 2013.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 1731.00 | 2010.71 | 2012.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 1636.00 | 1632.28 | 1733.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:00:00 | 1636.00 | 1632.28 | 1733.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1688.50 | 1635.54 | 1695.30 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1873.60 | 1734.88 | 1734.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1901.10 | 1738.99 | 1736.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1748.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1748.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1748.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 1770.40 | 1758.78 | 1748.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1779.60 | 1758.92 | 1748.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 1721.90 | 1757.30 | 1748.35 | SL hit (close<static) qty=1.00 sl=1730.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1776.00 | 1879.36 | 1879.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1772.70 | 1877.32 | 1878.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1787.00 | 1784.45 | 1817.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 1787.00 | 1784.45 | 1817.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1815.00 | 1785.19 | 1816.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 1815.00 | 1785.19 | 1816.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1809.40 | 1785.43 | 1816.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 12:30:00 | 1806.30 | 1785.67 | 1816.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:30:00 | 1805.60 | 1786.08 | 1816.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 1805.70 | 1786.08 | 1816.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:45:00 | 1805.80 | 1787.38 | 1816.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1815.50 | 1787.66 | 1816.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1815.50 | 1787.66 | 1816.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1814.00 | 1787.92 | 1816.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1814.00 | 1787.92 | 1816.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1815.30 | 1788.20 | 1816.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1815.90 | 1788.20 | 1816.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | SL hit (close>static) qty=1.00 sl=1817.60 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1945.00 | 1800.33 | 1799.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1954.90 | 1801.87 | 1800.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 1903.50 | 1905.01 | 1865.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 1903.50 | 1905.01 | 1865.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1881.00 | 1904.47 | 1866.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 1889.10 | 1901.21 | 1866.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:30:00 | 1884.80 | 1904.65 | 1870.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 1900.60 | 1903.65 | 1870.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 1909.20 | 1903.71 | 1871.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1869.50 | 1909.00 | 1877.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 1862.40 | 1909.00 | 1877.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1880.90 | 1908.72 | 1877.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 1851.00 | 1908.72 | 1877.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 1864.20 | 1908.28 | 1877.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 1864.20 | 1908.28 | 1877.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | SL hit (close<static) qty=1.00 sl=1841.50 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1627.00 | 1852.28 | 1852.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 1602.00 | 1849.79 | 1851.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 10:15:00 | 1726.50 | 1723.92 | 1776.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 11:00:00 | 1726.50 | 1723.92 | 1776.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1773.60 | 1724.45 | 1774.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 1784.60 | 1724.45 | 1774.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 1772.00 | 1724.92 | 1774.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:45:00 | 1762.00 | 1797.60 | 1801.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 12:00:00 | 1767.60 | 1797.30 | 1801.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1803.10 | 1796.71 | 1801.37 | SL hit (close>static) qty=1.00 sl=1793.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-16 14:15:00 | 2050.80 | 2025-04-17 09:15:00 | 2023.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-04-16 14:45:00 | 2050.30 | 2025-04-17 09:15:00 | 2023.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-04-21 14:30:00 | 2049.50 | 2025-04-23 09:15:00 | 2020.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-04-22 13:15:00 | 2080.00 | 2025-04-23 09:15:00 | 2020.80 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-07-24 13:45:00 | 1770.40 | 2025-07-29 10:15:00 | 1721.90 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1779.60 | 2025-07-29 10:15:00 | 1721.90 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-08-05 10:00:00 | 1773.00 | 2025-08-12 10:15:00 | 1727.70 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-08-05 11:00:00 | 1770.00 | 2025-08-12 10:15:00 | 1727.70 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-08-06 15:15:00 | 1786.30 | 2025-08-12 11:15:00 | 1717.70 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-08-12 09:45:00 | 1751.90 | 2025-08-12 11:15:00 | 1717.70 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-08-12 10:30:00 | 1750.00 | 2025-08-12 11:15:00 | 1717.70 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-08-12 15:15:00 | 1750.00 | 2025-08-18 09:15:00 | 1925.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 10:15:00 | 1891.80 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-29 11:00:00 | 1894.40 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-29 12:15:00 | 1892.10 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-29 14:00:00 | 1894.70 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-10-10 09:15:00 | 1924.80 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -5.56% |
| BUY | retest2 | 2025-10-13 13:45:00 | 1921.80 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1937.10 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -6.16% |
| BUY | retest2 | 2025-11-04 15:00:00 | 1929.00 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -5.76% |
| BUY | retest2 | 2025-11-06 09:15:00 | 1961.90 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2025-12-15 12:30:00 | 1806.30 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-15 14:30:00 | 1805.60 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-15 15:00:00 | 1805.70 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-16 13:45:00 | 1805.80 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-01-09 11:30:00 | 1821.20 | 2026-01-20 12:15:00 | 1730.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 1821.20 | 2026-01-30 09:15:00 | 1786.50 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2026-02-01 09:30:00 | 1819.90 | 2026-02-02 13:15:00 | 1812.40 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2026-02-01 10:00:00 | 1822.10 | 2026-02-03 09:15:00 | 1883.30 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-02-01 10:30:00 | 1820.60 | 2026-02-03 09:15:00 | 1883.30 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-02-02 09:15:00 | 1769.50 | 2026-02-03 09:15:00 | 1883.30 | STOP_HIT | 1.00 | -6.43% |
| BUY | retest2 | 2026-03-05 11:00:00 | 1889.10 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-09 12:30:00 | 1884.80 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-03-10 09:15:00 | 1900.60 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-03-10 10:00:00 | 1909.20 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2026-04-30 10:45:00 | 1762.00 | 2026-05-04 09:15:00 | 1803.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-04-30 12:00:00 | 1767.60 | 2026-05-04 09:15:00 | 1803.10 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-05-07 09:30:00 | 1761.60 | 2026-05-07 11:15:00 | 1794.00 | STOP_HIT | 1.00 | -1.84% |
