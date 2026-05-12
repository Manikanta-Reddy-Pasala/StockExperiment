# Netweb Technologies India Ltd. (NETWEB)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 4305.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 55 |
| ALERT2 | 55 |
| ALERT2_SKIP | 33 |
| ALERT3 | 136 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 53 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 48
- **Target hits / Stop hits / Partials:** 8 / 56 / 7
- **Avg / median % per leg:** -0.22% / -1.68%
- **Sum % (uncompounded):** -15.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 10 | 28.6% | 8 | 26 | 1 | 0.01% | 0.4% |
| BUY @ 2nd Alert (retest1) | 12 | 2 | 16.7% | 0 | 11 | 1 | -2.09% | -25.0% |
| BUY @ 3rd Alert (retest2) | 23 | 8 | 34.8% | 8 | 15 | 0 | 1.11% | 25.4% |
| SELL (all) | 36 | 13 | 36.1% | 0 | 30 | 6 | -0.44% | -15.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 13 | 36.1% | 0 | 30 | 6 | -0.44% | -15.9% |
| retest1 (combined) | 12 | 2 | 16.7% | 0 | 11 | 1 | -2.09% | -25.0% |
| retest2 (combined) | 59 | 21 | 35.6% | 8 | 45 | 6 | 0.16% | 9.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 1807.00 | 1817.39 | 1818.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1783.80 | 1810.68 | 1815.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 1822.60 | 1808.28 | 1811.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 1822.60 | 1808.28 | 1811.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1822.60 | 1808.28 | 1811.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 1822.60 | 1808.28 | 1811.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1815.50 | 1809.72 | 1812.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:30:00 | 1811.30 | 1811.36 | 1812.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 1813.30 | 1810.89 | 1812.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 1806.30 | 1808.07 | 1810.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1929.90 | 1827.25 | 1817.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1929.90 | 1827.25 | 1817.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1987.90 | 1929.85 | 1899.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1991.60 | 2009.49 | 1981.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1991.60 | 2009.49 | 1981.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1991.60 | 2009.49 | 1981.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1992.00 | 2009.49 | 1981.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1988.90 | 2005.37 | 1982.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 1992.60 | 2005.37 | 1982.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 1994.00 | 2001.05 | 1984.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1971.60 | 1993.39 | 1985.79 | SL hit (close<static) qty=1.00 sl=1980.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 1963.90 | 1980.04 | 1981.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 1951.00 | 1974.23 | 1978.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 1989.10 | 1973.81 | 1977.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 1989.10 | 1973.81 | 1977.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1989.10 | 1973.81 | 1977.34 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1995.90 | 1974.40 | 1973.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1996.00 | 1982.16 | 1977.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1987.10 | 2004.37 | 1994.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 1987.10 | 2004.37 | 1994.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1987.10 | 2004.37 | 1994.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1987.10 | 2004.37 | 1994.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1952.00 | 1993.90 | 1990.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1923.70 | 1993.90 | 1990.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1931.40 | 1981.40 | 1985.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1841.50 | 1896.92 | 1925.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1769.30 | 1768.66 | 1805.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 1769.30 | 1768.66 | 1805.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1823.70 | 1783.34 | 1805.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1823.90 | 1783.34 | 1805.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1818.00 | 1790.27 | 1807.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1794.30 | 1790.27 | 1807.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1796.20 | 1793.35 | 1805.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 1807.30 | 1793.35 | 1805.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1769.00 | 1761.55 | 1775.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1774.40 | 1761.55 | 1775.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1751.70 | 1739.52 | 1754.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1741.60 | 1739.52 | 1754.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1757.90 | 1743.20 | 1755.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1757.90 | 1743.20 | 1755.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1757.50 | 1746.06 | 1755.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1746.00 | 1746.06 | 1755.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1762.90 | 1750.81 | 1755.33 | SL hit (close>static) qty=1.00 sl=1760.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1765.80 | 1758.90 | 1758.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1803.70 | 1769.63 | 1763.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1786.10 | 1786.61 | 1775.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 1786.10 | 1786.61 | 1775.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1770.00 | 1783.29 | 1774.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1770.00 | 1783.29 | 1774.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1775.30 | 1781.69 | 1774.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1836.20 | 1781.69 | 1774.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 1831.60 | 1851.94 | 1854.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1831.60 | 1851.94 | 1854.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1819.00 | 1845.36 | 1851.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 1804.80 | 1796.65 | 1810.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:45:00 | 1809.00 | 1796.65 | 1810.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1809.80 | 1799.78 | 1809.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1820.20 | 1799.78 | 1809.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1801.40 | 1800.11 | 1808.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1810.80 | 1800.11 | 1808.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1803.00 | 1800.68 | 1808.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 1797.80 | 1800.27 | 1807.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1821.40 | 1805.03 | 1807.25 | SL hit (close>static) qty=1.00 sl=1808.20 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 1818.00 | 1810.56 | 1809.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1929.80 | 1837.34 | 1822.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 1947.80 | 1962.57 | 1927.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:30:00 | 1955.30 | 1962.57 | 1927.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1959.70 | 1964.68 | 1953.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1960.70 | 1964.68 | 1953.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 1942.10 | 1960.17 | 1952.63 | SL hit (close<static) qty=1.00 sl=1950.10 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1931.00 | 1947.50 | 1948.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1910.30 | 1932.81 | 1939.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1911.60 | 1909.32 | 1921.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1911.60 | 1909.32 | 1921.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1911.60 | 1909.32 | 1921.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 1910.20 | 1909.32 | 1921.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1882.90 | 1896.85 | 1908.32 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 1959.30 | 1917.86 | 1915.50 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 1923.00 | 1935.59 | 1935.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1900.00 | 1927.55 | 1931.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1920.90 | 1875.20 | 1896.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 1920.90 | 1875.20 | 1896.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1920.90 | 1875.20 | 1896.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 1920.90 | 1875.20 | 1896.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1931.10 | 1886.38 | 1899.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 1931.10 | 1886.38 | 1899.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1952.60 | 1914.20 | 1910.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 1971.70 | 1925.70 | 1915.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 2117.80 | 2142.65 | 2079.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 2117.80 | 2142.65 | 2079.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 2199.00 | 2222.90 | 2192.41 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 2137.80 | 2173.19 | 2177.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 2130.30 | 2164.61 | 2172.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 2149.40 | 2148.94 | 2162.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 2149.40 | 2148.94 | 2162.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2149.40 | 2148.94 | 2162.63 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 2150.00 | 2138.51 | 2137.46 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2113.50 | 2132.82 | 2135.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 2089.00 | 2116.73 | 2126.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 2125.10 | 2104.59 | 2110.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 2125.10 | 2104.59 | 2110.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2125.10 | 2104.59 | 2110.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 2125.10 | 2104.59 | 2110.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2106.00 | 2104.88 | 2109.80 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 2129.10 | 2114.99 | 2113.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 2157.40 | 2123.47 | 2117.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 2125.00 | 2126.36 | 2119.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 2125.00 | 2126.36 | 2119.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 2139.40 | 2142.71 | 2135.76 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 2067.30 | 2123.15 | 2128.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 2051.90 | 2083.26 | 2104.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 11:15:00 | 2076.20 | 2073.25 | 2091.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 2076.20 | 2073.25 | 2091.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 2076.20 | 2073.25 | 2091.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 2081.60 | 2073.25 | 2091.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2130.80 | 2083.15 | 2089.09 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 2132.60 | 2100.09 | 2096.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 2171.20 | 2114.31 | 2103.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 2323.40 | 2327.61 | 2267.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 10:30:00 | 2327.20 | 2327.61 | 2267.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 2289.90 | 2310.80 | 2281.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 2279.20 | 2310.80 | 2281.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 2300.60 | 2308.76 | 2283.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 2301.90 | 2308.76 | 2283.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2281.10 | 2301.75 | 2284.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 2281.10 | 2301.75 | 2284.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 2277.20 | 2296.84 | 2283.57 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 2240.00 | 2273.61 | 2275.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 13:15:00 | 2208.60 | 2250.05 | 2262.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 2240.60 | 2231.01 | 2244.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 12:15:00 | 2240.60 | 2231.01 | 2244.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 2240.60 | 2231.01 | 2244.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 2240.60 | 2231.01 | 2244.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2248.00 | 2234.41 | 2245.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 2231.00 | 2233.73 | 2243.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 2277.30 | 2246.46 | 2247.54 | SL hit (close>static) qty=1.00 sl=2265.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 2287.30 | 2254.62 | 2251.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 2382.20 | 2287.26 | 2268.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 2978.80 | 3047.61 | 2917.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 2981.00 | 3047.61 | 2917.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 3044.60 | 3087.80 | 3046.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 3044.60 | 3087.80 | 3046.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 3056.30 | 3081.50 | 3047.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 3039.10 | 3081.50 | 3047.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 3025.60 | 3070.32 | 3045.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 3025.60 | 3070.32 | 3045.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 3034.30 | 3063.12 | 3044.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 3034.30 | 3063.12 | 3044.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 3008.40 | 3052.17 | 3040.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 3008.40 | 3052.17 | 3040.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 2989.00 | 3039.54 | 3036.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 2955.70 | 3039.54 | 3036.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 2941.40 | 3019.91 | 3027.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 2902.00 | 2951.01 | 2986.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 2941.00 | 2919.63 | 2955.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 2941.00 | 2919.63 | 2955.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 2941.00 | 2919.63 | 2955.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 3003.30 | 2919.63 | 2955.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2884.60 | 2898.99 | 2932.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 2869.00 | 2891.35 | 2923.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:45:00 | 2864.90 | 2859.05 | 2883.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 2962.20 | 2901.30 | 2898.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 2962.20 | 2901.30 | 2898.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 3027.80 | 2926.60 | 2910.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 3410.00 | 3427.91 | 3294.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 11:00:00 | 3555.90 | 3453.51 | 3318.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 11:30:00 | 3565.80 | 3466.41 | 3336.68 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 12:45:00 | 3544.10 | 3478.52 | 3353.98 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 13:15:00 | 3547.50 | 3478.52 | 3353.98 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 3409.00 | 3468.40 | 3425.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 3409.00 | 3468.40 | 3425.42 | SL hit (close<ema400) qty=1.00 sl=3425.42 alert=retest1 |

### Cycle 23 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 4082.10 | 4248.20 | 4252.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 3996.50 | 4151.31 | 4203.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 3945.00 | 3882.29 | 3919.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 3945.00 | 3882.29 | 3919.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 3945.00 | 3882.29 | 3919.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 3945.00 | 3882.29 | 3919.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 3930.00 | 3891.83 | 3920.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 3900.20 | 3891.83 | 3920.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 4000.60 | 3863.25 | 3859.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 4000.60 | 3863.25 | 3859.84 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 3828.00 | 3877.69 | 3880.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 3809.50 | 3864.05 | 3873.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 3805.70 | 3757.10 | 3789.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 3805.70 | 3757.10 | 3789.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3805.70 | 3757.10 | 3789.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 3805.70 | 3757.10 | 3789.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3818.00 | 3769.28 | 3792.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 3818.00 | 3769.28 | 3792.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 3791.00 | 3773.62 | 3792.05 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 3890.00 | 3813.06 | 3806.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 3911.00 | 3832.64 | 3816.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 4010.80 | 4073.28 | 3988.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 4010.80 | 4073.28 | 3988.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 4010.80 | 4073.28 | 3988.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 4006.00 | 4073.28 | 3988.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 3920.50 | 4042.72 | 3982.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 3920.50 | 4042.72 | 3982.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 3945.60 | 4023.30 | 3979.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:45:00 | 3927.50 | 4023.30 | 3979.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 4095.40 | 4002.42 | 3979.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 4178.20 | 4002.42 | 3979.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 4123.40 | 4026.62 | 3992.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 4110.10 | 4066.16 | 4021.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3815.90 | 3993.30 | 4008.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 3815.90 | 3993.30 | 4008.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 3739.00 | 3804.53 | 3877.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 3469.30 | 3420.11 | 3543.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 3469.30 | 3420.11 | 3543.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3500.00 | 3436.09 | 3539.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3534.90 | 3436.09 | 3539.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 3490.00 | 3446.87 | 3534.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 3407.00 | 3438.90 | 3523.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 3461.90 | 3425.71 | 3451.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 3477.70 | 3446.56 | 3457.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 3500.00 | 3466.76 | 3464.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 3500.00 | 3466.76 | 3464.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 3582.50 | 3489.91 | 3475.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 3508.40 | 3528.65 | 3506.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 3508.40 | 3528.65 | 3506.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 3508.40 | 3528.65 | 3506.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 3478.90 | 3528.65 | 3506.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 3497.40 | 3522.40 | 3505.66 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 3472.40 | 3495.93 | 3497.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 3467.90 | 3490.32 | 3494.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 3353.90 | 3309.04 | 3354.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 3353.90 | 3309.04 | 3354.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 3353.90 | 3309.04 | 3354.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 3353.90 | 3309.04 | 3354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 3314.30 | 3310.09 | 3350.79 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 3444.20 | 3364.91 | 3358.64 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 3331.00 | 3368.82 | 3370.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 3310.50 | 3346.71 | 3359.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 3293.80 | 3279.18 | 3312.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 3293.80 | 3279.18 | 3312.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 3304.00 | 3284.14 | 3311.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 3300.00 | 3284.14 | 3311.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3291.80 | 3285.67 | 3309.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 3263.30 | 3283.62 | 3306.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 3271.20 | 3280.83 | 3303.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 3326.00 | 3305.20 | 3303.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 3326.00 | 3305.20 | 3303.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 3347.50 | 3313.66 | 3307.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 3298.50 | 3319.04 | 3312.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 3298.50 | 3319.04 | 3312.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 3298.50 | 3319.04 | 3312.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 3298.50 | 3319.04 | 3312.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 3292.50 | 3313.73 | 3311.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 3297.70 | 3313.73 | 3311.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 3298.90 | 3307.43 | 3308.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 3288.80 | 3301.64 | 3305.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 3215.00 | 3206.87 | 3237.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 3215.00 | 3206.87 | 3237.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 3240.00 | 3214.40 | 3235.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 3239.60 | 3214.40 | 3235.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 3201.50 | 3211.82 | 3232.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 3180.50 | 3205.07 | 3227.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:00:00 | 3178.10 | 3205.07 | 3227.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 3113.90 | 3199.31 | 3215.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 3187.00 | 3194.83 | 3203.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 3187.00 | 3193.26 | 3202.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 3159.80 | 3193.26 | 3202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3150.40 | 3184.69 | 3197.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 3133.00 | 3184.69 | 3197.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:00:00 | 3143.00 | 3176.35 | 3192.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 3021.47 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 3019.19 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2958.20 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 3027.65 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2976.35 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2985.85 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 3081.00 | 3053.48 | 3093.14 | SL hit (close>ema200) qty=0.50 sl=3053.48 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 3109.80 | 3103.66 | 3103.02 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 3079.00 | 3098.32 | 3100.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 3051.00 | 3087.20 | 3095.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 3109.00 | 3091.56 | 3096.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 3109.00 | 3091.56 | 3096.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 3109.00 | 3091.56 | 3096.37 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 3149.10 | 3105.03 | 3101.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 3188.40 | 3155.79 | 3133.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3258.90 | 3275.03 | 3226.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 3258.90 | 3275.03 | 3226.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3222.10 | 3263.63 | 3244.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 3224.30 | 3263.63 | 3244.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 3215.50 | 3254.00 | 3242.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 3209.70 | 3254.00 | 3242.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 3216.20 | 3241.80 | 3238.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 3214.70 | 3241.80 | 3238.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 14:15:00 | 3206.00 | 3230.34 | 3233.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 3175.00 | 3219.27 | 3228.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 3224.00 | 3167.99 | 3188.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 3224.00 | 3167.99 | 3188.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3224.00 | 3167.99 | 3188.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 3222.00 | 3167.99 | 3188.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 3227.00 | 3179.80 | 3192.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 3227.00 | 3179.80 | 3192.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 3235.80 | 3200.33 | 3199.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 3259.90 | 3212.24 | 3204.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3248.50 | 3257.34 | 3239.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 3248.50 | 3257.34 | 3239.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3248.50 | 3257.34 | 3239.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 3257.00 | 3257.34 | 3239.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3232.00 | 3250.70 | 3239.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 3236.10 | 3250.70 | 3239.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 3232.70 | 3247.10 | 3238.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 3232.70 | 3247.10 | 3238.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 3225.80 | 3242.84 | 3237.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 3225.80 | 3242.84 | 3237.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 3234.00 | 3238.26 | 3236.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 3243.40 | 3238.26 | 3236.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3243.00 | 3239.21 | 3236.79 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 3206.80 | 3230.60 | 3233.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 3197.20 | 3216.98 | 3225.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 3106.90 | 3101.05 | 3126.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 3106.90 | 3101.05 | 3126.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3090.60 | 3097.62 | 3118.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 3078.60 | 3094.04 | 3105.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 3079.30 | 3091.09 | 3103.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 3077.10 | 3087.14 | 3099.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 3161.30 | 3066.40 | 3073.82 | SL hit (close>static) qty=1.00 sl=3123.80 alert=retest2 |

### Cycle 40 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 3252.80 | 3103.68 | 3090.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 3347.60 | 3152.46 | 3113.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 3311.60 | 3325.25 | 3265.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 12:15:00 | 3391.40 | 3344.42 | 3284.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 3396.40 | 3363.09 | 3322.13 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 13:45:00 | 3415.50 | 3380.66 | 3341.34 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 14:30:00 | 3405.30 | 3383.35 | 3346.14 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3403.50 | 3390.06 | 3355.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 3366.80 | 3390.06 | 3355.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 3309.00 | 3374.31 | 3362.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 3309.00 | 3374.31 | 3362.04 | SL hit (close<ema400) qty=1.00 sl=3362.04 alert=retest1 |

### Cycle 41 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 3294.30 | 3347.81 | 3351.62 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 3359.10 | 3350.83 | 3350.25 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 3322.30 | 3345.12 | 3347.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 3315.90 | 3339.28 | 3344.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 3306.00 | 3253.04 | 3283.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 3306.00 | 3253.04 | 3283.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3306.00 | 3253.04 | 3283.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 3283.40 | 3253.04 | 3283.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 3295.00 | 3261.43 | 3284.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 3312.60 | 3261.43 | 3284.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 3294.00 | 3270.96 | 3285.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 3294.00 | 3270.96 | 3285.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 3273.30 | 3271.43 | 3283.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 3276.90 | 3271.43 | 3283.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 3348.10 | 3286.77 | 3289.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 3348.10 | 3286.77 | 3289.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 15:15:00 | 3377.90 | 3304.99 | 3297.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 09:15:00 | 3544.90 | 3352.97 | 3320.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 14:15:00 | 3413.90 | 3424.09 | 3375.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 3413.90 | 3424.09 | 3375.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 3412.00 | 3421.67 | 3378.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 3339.80 | 3421.67 | 3378.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 3311.50 | 3399.64 | 3372.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 3266.20 | 3399.64 | 3372.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 3276.80 | 3375.07 | 3363.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 3276.80 | 3375.07 | 3363.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 3290.50 | 3345.66 | 3351.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 3260.00 | 3328.53 | 3343.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 3156.90 | 3145.95 | 3187.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 09:15:00 | 3167.90 | 3145.95 | 3187.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 3149.30 | 3146.62 | 3183.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 3159.60 | 3146.62 | 3183.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3147.40 | 3072.98 | 3100.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 3147.40 | 3072.98 | 3100.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 3162.70 | 3090.92 | 3106.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 3162.70 | 3090.92 | 3106.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 3147.00 | 3121.10 | 3117.79 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3079.50 | 3111.68 | 3114.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 13:15:00 | 3073.50 | 3093.39 | 3104.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 3161.50 | 3098.38 | 3103.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 3161.50 | 3098.38 | 3103.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 3161.50 | 3098.38 | 3103.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 3158.90 | 3098.38 | 3103.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 3150.40 | 3108.78 | 3107.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 3217.10 | 3147.78 | 3128.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 3195.00 | 3264.35 | 3214.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 3195.00 | 3264.35 | 3214.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3195.00 | 3264.35 | 3214.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 3195.00 | 3264.35 | 3214.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 3180.80 | 3247.64 | 3211.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 3180.80 | 3247.64 | 3211.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 3152.50 | 3228.62 | 3205.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 3161.00 | 3228.62 | 3205.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 3205.00 | 3208.61 | 3200.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 3245.00 | 3208.61 | 3200.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 3229.00 | 3219.80 | 3207.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 3232.00 | 3212.87 | 3211.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 3144.00 | 3202.16 | 3207.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 3144.00 | 3202.16 | 3207.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 3102.20 | 3182.17 | 3197.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 3105.90 | 3100.50 | 3133.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 3105.90 | 3100.50 | 3133.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3105.90 | 3100.50 | 3133.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 3105.90 | 3100.50 | 3133.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3225.00 | 3129.88 | 3138.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 3244.00 | 3129.88 | 3138.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 3212.30 | 3146.36 | 3145.65 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 3153.80 | 3172.23 | 3173.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 3066.70 | 3150.45 | 3163.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 3184.40 | 3142.04 | 3154.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 12:15:00 | 3184.40 | 3142.04 | 3154.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 3184.40 | 3142.04 | 3154.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:45:00 | 3179.40 | 3142.04 | 3154.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 3182.30 | 3150.09 | 3157.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 3184.00 | 3150.09 | 3157.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 3216.00 | 3163.27 | 3162.70 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3120.10 | 3156.51 | 3159.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3105.00 | 3131.82 | 3145.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 3087.00 | 3085.65 | 3107.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 3087.00 | 3085.65 | 3107.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 3087.00 | 3085.65 | 3107.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:45:00 | 3068.00 | 3081.92 | 3103.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:00:00 | 3073.50 | 3080.24 | 3100.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 3146.90 | 3101.10 | 3103.62 | SL hit (close>static) qty=1.00 sl=3119.90 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 3289.10 | 3138.70 | 3120.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 3440.90 | 3199.14 | 3149.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 09:15:00 | 3579.00 | 3601.98 | 3512.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 3548.20 | 3585.04 | 3538.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 3548.20 | 3585.04 | 3538.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 3548.20 | 3585.04 | 3538.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 3558.00 | 3579.63 | 3540.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 3506.70 | 3579.63 | 3540.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 3477.00 | 3559.11 | 3534.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:45:00 | 3469.50 | 3559.11 | 3534.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 3485.00 | 3544.28 | 3529.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 3518.30 | 3539.09 | 3528.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 3522.70 | 3531.61 | 3526.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 10:15:00 | 3870.13 | 3724.69 | 3667.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 3624.50 | 3719.24 | 3729.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 3532.50 | 3681.89 | 3711.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 3195.00 | 3188.17 | 3271.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:45:00 | 3214.20 | 3188.17 | 3271.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3275.10 | 3209.82 | 3267.17 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 3338.80 | 3287.60 | 3287.13 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 3258.40 | 3285.66 | 3286.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 3248.00 | 3270.93 | 3279.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 14:15:00 | 3276.00 | 3271.95 | 3278.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 3276.00 | 3271.95 | 3278.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 3276.00 | 3271.95 | 3278.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 3276.00 | 3271.95 | 3278.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 3285.00 | 3274.56 | 3279.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 3243.30 | 3274.56 | 3279.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 3290.00 | 3223.28 | 3216.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 3290.00 | 3223.28 | 3216.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 3304.70 | 3265.92 | 3242.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 3299.00 | 3310.83 | 3282.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 15:00:00 | 3299.00 | 3310.83 | 3282.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 3290.00 | 3306.67 | 3282.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 3254.50 | 3306.67 | 3282.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3256.80 | 3296.69 | 3280.41 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 3256.80 | 3271.71 | 3272.80 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 3283.30 | 3272.88 | 3272.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 3314.50 | 3281.21 | 3276.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 3282.00 | 3286.35 | 3280.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 3282.00 | 3286.35 | 3280.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 3282.00 | 3286.35 | 3280.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 3282.00 | 3286.35 | 3280.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3285.00 | 3286.08 | 3281.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 3197.00 | 3286.08 | 3281.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 3139.00 | 3256.66 | 3268.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 3109.10 | 3227.15 | 3253.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3156.30 | 3137.80 | 3187.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3156.30 | 3137.80 | 3187.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3156.30 | 3137.80 | 3187.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 3143.30 | 3137.80 | 3187.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 3248.60 | 3187.63 | 3190.08 | SL hit (close>static) qty=1.00 sl=3232.30 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3250.00 | 3200.11 | 3195.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 3292.50 | 3218.59 | 3204.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3160.00 | 3227.62 | 3217.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3160.00 | 3227.62 | 3217.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3160.00 | 3227.62 | 3217.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3160.00 | 3227.62 | 3217.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3153.00 | 3212.69 | 3211.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 3153.00 | 3212.69 | 3211.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 3176.60 | 3205.47 | 3208.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 3135.80 | 3171.58 | 3189.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3186.00 | 3144.58 | 3164.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3186.00 | 3144.58 | 3164.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3186.00 | 3144.58 | 3164.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 3161.90 | 3155.20 | 3166.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 3147.90 | 3160.21 | 3166.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 3182.80 | 3152.82 | 3149.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 3182.80 | 3152.82 | 3149.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 3213.00 | 3171.75 | 3159.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 3348.10 | 3374.37 | 3342.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 14:15:00 | 3348.10 | 3374.37 | 3342.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 3348.10 | 3374.37 | 3342.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 3357.00 | 3374.37 | 3342.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 3343.00 | 3368.09 | 3342.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 3290.30 | 3368.09 | 3342.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3285.00 | 3351.47 | 3337.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 3308.80 | 3342.38 | 3334.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 3308.80 | 3325.93 | 3328.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 3308.80 | 3325.93 | 3328.11 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 3460.00 | 3348.85 | 3337.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 3508.30 | 3380.74 | 3353.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 3773.00 | 3780.55 | 3695.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 3891.00 | 3806.42 | 3750.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 4085.55 | 3945.03 | 3862.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 3985.90 | 4005.18 | 3942.82 | SL hit (close<ema200) qty=0.50 sl=4005.18 alert=retest1 |

### Cycle 67 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 3795.00 | 3910.64 | 3921.54 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 3917.00 | 3900.08 | 3898.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 4065.00 | 3937.53 | 3916.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 3994.70 | 4016.16 | 3985.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:45:00 | 4002.60 | 4016.16 | 3985.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 4017.50 | 4016.43 | 3988.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 4060.00 | 4016.43 | 3988.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 10:30:00 | 4032.20 | 4019.27 | 3997.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 4041.60 | 4026.30 | 4002.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 3808.10 | 3998.14 | 4001.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 3808.10 | 3998.14 | 4001.01 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 4139.10 | 3988.68 | 3977.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 4174.00 | 4084.00 | 4031.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 13:15:00 | 4305.80 | 4330.56 | 4253.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:15:00 | 4436.20 | 4322.36 | 4262.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 4372.20 | 4348.60 | 4285.78 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 4381.90 | 4391.76 | 4339.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 09:45:00 | 4351.60 | 4391.76 | 4339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 13:15:00 | 4358.90 | 4386.23 | 4353.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 14:00:00 | 4358.90 | 4386.23 | 4353.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 14:15:00 | 4288.00 | 4366.59 | 4347.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-11 14:15:00 | 4288.00 | 4366.59 | 4347.93 | SL hit (close<ema400) qty=1.00 sl=4347.93 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 1672.30 | 2025-05-15 09:15:00 | 1839.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-22 11:30:00 | 1811.30 | 2025-05-26 09:15:00 | 1929.90 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2025-05-22 12:30:00 | 1813.30 | 2025-05-26 09:15:00 | 1929.90 | STOP_HIT | 1.00 | -6.43% |
| SELL | retest2 | 2025-05-23 09:45:00 | 1806.30 | 2025-05-26 09:15:00 | 1929.90 | STOP_HIT | 1.00 | -6.84% |
| BUY | retest2 | 2025-05-30 11:15:00 | 1992.60 | 2025-06-02 09:15:00 | 1971.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-30 13:00:00 | 1994.00 | 2025-06-02 09:15:00 | 1971.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1746.00 | 2025-06-20 14:15:00 | 1762.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1836.20 | 2025-07-04 10:15:00 | 1831.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-09 12:30:00 | 1797.80 | 2025-07-10 09:15:00 | 1821.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1960.70 | 2025-07-17 09:15:00 | 1942.10 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-01 15:00:00 | 2231.00 | 2025-09-02 10:15:00 | 2277.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-16 12:15:00 | 2869.00 | 2025-09-18 09:15:00 | 2962.20 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-09-17 13:45:00 | 2864.90 | 2025-09-18 09:15:00 | 2962.20 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest1 | 2025-09-23 11:00:00 | 3555.90 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest1 | 2025-09-23 11:30:00 | 3565.80 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest1 | 2025-09-23 12:45:00 | 3544.10 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest1 | 2025-09-23 13:15:00 | 3547.50 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-09-25 11:15:00 | 3564.80 | 2025-10-01 09:15:00 | 3921.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-25 11:45:00 | 3586.30 | 2025-10-01 09:15:00 | 3944.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 11:00:00 | 3560.70 | 2025-10-01 09:15:00 | 3916.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 12:15:00 | 3564.80 | 2025-10-01 09:15:00 | 3921.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 09:15:00 | 3733.10 | 2025-10-01 10:15:00 | 4106.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-16 09:15:00 | 3900.20 | 2025-10-20 09:15:00 | 4000.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-10-30 11:15:00 | 4178.20 | 2025-11-03 09:15:00 | 3815.90 | STOP_HIT | 1.00 | -8.67% |
| BUY | retest2 | 2025-10-30 12:00:00 | 4123.40 | 2025-11-03 09:15:00 | 3815.90 | STOP_HIT | 1.00 | -7.46% |
| BUY | retest2 | 2025-10-30 14:45:00 | 4110.10 | 2025-11-03 09:15:00 | 3815.90 | STOP_HIT | 1.00 | -7.16% |
| SELL | retest2 | 2025-11-07 15:00:00 | 3407.00 | 2025-11-11 15:15:00 | 3500.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-11-11 11:15:00 | 3461.90 | 2025-11-11 15:15:00 | 3500.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-11 12:45:00 | 3477.70 | 2025-11-11 15:15:00 | 3500.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-25 10:45:00 | 3263.30 | 2025-11-27 11:15:00 | 3326.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-11-25 11:30:00 | 3271.20 | 2025-11-27 11:15:00 | 3326.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-12-03 10:30:00 | 3180.50 | 2025-12-05 15:15:00 | 3021.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 11:00:00 | 3178.10 | 2025-12-05 15:15:00 | 3019.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 09:15:00 | 3113.90 | 2025-12-05 15:15:00 | 2958.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 3187.00 | 2025-12-05 15:15:00 | 3027.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 10:15:00 | 3133.00 | 2025-12-05 15:15:00 | 2976.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 11:00:00 | 3143.00 | 2025-12-05 15:15:00 | 2985.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 10:30:00 | 3180.50 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-12-03 11:00:00 | 3178.10 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-12-04 09:15:00 | 3113.90 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2025-12-04 15:15:00 | 3187.00 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-12-05 10:15:00 | 3133.00 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2025-12-05 11:00:00 | 3143.00 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-12-10 09:45:00 | 3141.00 | 2025-12-10 11:15:00 | 3109.80 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2026-01-01 12:00:00 | 3078.60 | 2026-01-05 09:15:00 | 3161.30 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-01-01 13:00:00 | 3079.30 | 2026-01-05 09:15:00 | 3161.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-01-01 15:00:00 | 3077.10 | 2026-01-05 09:15:00 | 3161.30 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest1 | 2026-01-07 12:15:00 | 3391.40 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest1 | 2026-01-08 11:15:00 | 3396.40 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest1 | 2026-01-08 13:45:00 | 3415.50 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2026-01-08 14:30:00 | 3405.30 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-12 09:15:00 | 3358.60 | 2026-01-12 10:15:00 | 3294.30 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-02-03 09:15:00 | 3245.00 | 2026-02-05 09:15:00 | 3144.00 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-02-03 10:30:00 | 3229.00 | 2026-02-05 09:15:00 | 3144.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-02-04 15:15:00 | 3232.00 | 2026-02-05 09:15:00 | 3144.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-02-17 10:45:00 | 3068.00 | 2026-02-18 09:15:00 | 3146.90 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-02-17 12:00:00 | 3073.50 | 2026-02-18 09:15:00 | 3146.90 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-02-24 12:00:00 | 3518.30 | 2026-02-27 10:15:00 | 3870.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-24 13:15:00 | 3522.70 | 2026-02-27 10:15:00 | 3874.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 3243.30 | 2026-03-17 09:15:00 | 3290.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-03-24 10:15:00 | 3143.30 | 2026-03-25 09:15:00 | 3248.60 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-04-01 11:45:00 | 3161.90 | 2026-04-06 10:15:00 | 3182.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-01 13:30:00 | 3147.90 | 2026-04-06 10:15:00 | 3182.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-04-13 10:45:00 | 3308.80 | 2026-04-13 13:15:00 | 3308.80 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 3891.00 | 2026-04-22 10:15:00 | 4085.55 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 3891.00 | 2026-04-23 10:15:00 | 3985.90 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest2 | 2026-04-29 15:15:00 | 4060.00 | 2026-05-04 09:15:00 | 3808.10 | STOP_HIT | 1.00 | -6.20% |
| BUY | retest2 | 2026-04-30 10:30:00 | 4032.20 | 2026-05-04 09:15:00 | 3808.10 | STOP_HIT | 1.00 | -5.56% |
| BUY | retest2 | 2026-04-30 11:30:00 | 4041.60 | 2026-05-04 09:15:00 | 3808.10 | STOP_HIT | 1.00 | -5.78% |
| BUY | retest1 | 2026-05-08 09:15:00 | 4436.20 | 2026-05-11 14:15:00 | 4288.00 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest1 | 2026-05-08 10:45:00 | 4372.20 | 2026-05-11 14:15:00 | 4288.00 | STOP_HIT | 1.00 | -1.93% |
