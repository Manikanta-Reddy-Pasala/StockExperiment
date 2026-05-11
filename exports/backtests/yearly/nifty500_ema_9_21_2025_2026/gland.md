# Gland Pharma Ltd. (GLAND)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1540 bars)
- **Last close:** 1906.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 54 |
| ALERT1 | 39 |
| ALERT2 | 39 |
| ALERT2_SKIP | 18 |
| ALERT3 | 107 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 61 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 25 / 46
- **Target hits / Stop hits / Partials:** 1 / 63 / 7
- **Avg / median % per leg:** 0.08% / -1.00%
- **Sum % (uncompounded):** 6.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 10 | 27.8% | 1 | 34 | 1 | -0.55% | -19.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.16% | 4.6% |
| BUY @ 3rd Alert (retest2) | 32 | 8 | 25.0% | 1 | 31 | 0 | -0.76% | -24.4% |
| SELL (all) | 35 | 15 | 42.9% | 0 | 29 | 6 | 0.74% | 25.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.39% | -1.4% |
| SELL @ 3rd Alert (retest2) | 34 | 15 | 44.1% | 0 | 28 | 6 | 0.80% | 27.1% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.65% | 3.2% |
| retest2 (combined) | 66 | 23 | 34.8% | 1 | 59 | 6 | 0.04% | 2.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1524.60 | 1534.19 | 1535.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 1518.60 | 1526.40 | 1530.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 1550.30 | 1531.18 | 1532.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 1550.30 | 1531.18 | 1532.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1550.30 | 1531.18 | 1532.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 1550.30 | 1531.18 | 1532.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 1542.00 | 1533.34 | 1532.92 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 1529.00 | 1532.98 | 1532.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1521.40 | 1529.71 | 1531.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 14:15:00 | 1526.00 | 1525.76 | 1528.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 1526.00 | 1525.76 | 1528.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1526.00 | 1525.76 | 1528.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 1526.00 | 1525.76 | 1528.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1517.90 | 1524.19 | 1527.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 1522.60 | 1524.19 | 1527.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1523.60 | 1524.07 | 1527.11 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 1551.00 | 1532.88 | 1530.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 1591.30 | 1544.57 | 1536.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1607.90 | 1613.59 | 1597.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 1607.90 | 1613.59 | 1597.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1597.30 | 1610.33 | 1597.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:15:00 | 1596.90 | 1610.33 | 1597.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1589.90 | 1606.25 | 1597.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 1589.90 | 1606.25 | 1597.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1584.00 | 1601.80 | 1595.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1614.30 | 1604.40 | 1597.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-16 14:15:00 | 1775.73 | 1735.58 | 1707.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 1719.30 | 1736.81 | 1738.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 1707.20 | 1730.89 | 1735.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1733.40 | 1718.92 | 1726.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1733.40 | 1718.92 | 1726.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1733.40 | 1718.92 | 1726.29 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1741.30 | 1728.36 | 1727.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1779.50 | 1743.41 | 1735.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 11:15:00 | 1783.40 | 1784.78 | 1777.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 11:45:00 | 1783.50 | 1784.78 | 1777.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1782.70 | 1784.30 | 1778.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1782.70 | 1784.30 | 1778.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1785.10 | 1784.46 | 1779.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1799.70 | 1784.46 | 1779.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 1863.80 | 1875.24 | 1876.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1863.80 | 1875.24 | 1876.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 13:15:00 | 1849.20 | 1863.73 | 1870.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 1875.50 | 1860.46 | 1866.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 1875.50 | 1860.46 | 1866.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1875.50 | 1860.46 | 1866.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 1874.00 | 1860.46 | 1866.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1863.90 | 1861.14 | 1866.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 1854.00 | 1859.54 | 1865.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 1880.00 | 1866.43 | 1865.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1880.00 | 1866.43 | 1865.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 1909.70 | 1875.09 | 1869.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 15:15:00 | 1891.30 | 1892.70 | 1884.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1914.10 | 1892.70 | 1884.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1967.00 | 1907.56 | 1891.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 1983.90 | 1931.08 | 1905.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 1972.90 | 1938.88 | 1911.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:00:00 | 1971.90 | 1951.91 | 1922.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 1973.00 | 1959.00 | 1931.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 12:15:00 | 2009.81 | 1978.02 | 1947.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 2000.00 | 2001.56 | 1981.30 | SL hit (close<ema200) qty=0.50 sl=2001.56 alert=retest1 |

### Cycle 9 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 1973.10 | 1988.03 | 1988.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1961.90 | 1979.93 | 1984.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1991.50 | 1979.54 | 1983.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1991.50 | 1979.54 | 1983.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1991.50 | 1979.54 | 1983.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 1989.00 | 1979.54 | 1983.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 2029.80 | 1989.59 | 1987.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 2039.90 | 2011.94 | 2000.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 2019.30 | 2025.08 | 2014.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:30:00 | 2025.00 | 2025.08 | 2014.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 2017.00 | 2023.47 | 2014.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 2009.80 | 2023.47 | 2014.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2024.50 | 2023.67 | 2015.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 2047.00 | 2023.67 | 2015.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:30:00 | 2027.00 | 2020.29 | 2016.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:45:00 | 2026.00 | 2020.64 | 2016.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 2033.40 | 2020.07 | 2016.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2059.60 | 2027.97 | 2020.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:45:00 | 2062.90 | 2037.69 | 2028.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 2064.40 | 2041.13 | 2031.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 2067.20 | 2065.89 | 2061.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 2068.10 | 2062.62 | 2060.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 2061.80 | 2062.46 | 2060.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 2062.20 | 2062.46 | 2060.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2066.40 | 2063.25 | 2061.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 2065.60 | 2063.25 | 2061.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 2063.70 | 2064.75 | 2062.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 2042.70 | 2064.75 | 2062.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 2008.80 | 2053.56 | 2057.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 2008.80 | 2053.56 | 2057.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2004.50 | 2038.06 | 2049.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1983.00 | 1982.89 | 2004.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:45:00 | 1977.50 | 1982.89 | 2004.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1964.70 | 1980.04 | 1999.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 1962.20 | 1980.04 | 1999.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 1963.00 | 1972.29 | 1994.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 1958.00 | 1962.63 | 1981.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 1953.80 | 1972.21 | 1980.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1961.00 | 1962.61 | 1973.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1950.00 | 1957.45 | 1969.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1968.70 | 1955.69 | 1966.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 1968.70 | 1955.69 | 1966.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 1961.00 | 1956.75 | 1966.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 1965.00 | 1956.75 | 1966.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1959.30 | 1958.69 | 1965.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 1955.00 | 1958.69 | 1965.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1949.30 | 1958.12 | 1963.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 1955.00 | 1938.52 | 1943.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 1957.00 | 1941.92 | 1940.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1957.00 | 1941.92 | 1940.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 13:15:00 | 1988.80 | 1964.56 | 1954.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 10:15:00 | 1968.50 | 1971.56 | 1961.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:00:00 | 1968.50 | 1971.56 | 1961.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1961.30 | 1969.51 | 1961.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 1961.30 | 1969.51 | 1961.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1973.70 | 1970.35 | 1962.57 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 1949.40 | 1960.86 | 1961.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 13:15:00 | 1933.40 | 1953.45 | 1957.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1940.00 | 1932.13 | 1941.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 1940.00 | 1932.13 | 1941.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1940.00 | 1932.13 | 1941.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 1940.00 | 1932.13 | 1941.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1945.40 | 1934.79 | 1941.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 1946.00 | 1934.79 | 1941.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1945.10 | 1936.85 | 1942.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1949.00 | 1936.85 | 1942.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1957.30 | 1940.94 | 1943.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1958.60 | 1940.94 | 1943.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1950.90 | 1943.35 | 1944.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 1950.90 | 1943.35 | 1944.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 14:15:00 | 1950.00 | 1944.68 | 1944.66 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 1919.10 | 1939.69 | 1942.41 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 1960.70 | 1943.12 | 1942.30 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 1932.50 | 1945.14 | 1945.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1927.90 | 1938.99 | 1941.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 1900.80 | 1896.54 | 1912.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 1900.80 | 1896.54 | 1912.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1877.50 | 1868.61 | 1878.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 1862.30 | 1871.89 | 1877.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1893.80 | 1879.55 | 1878.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1893.80 | 1879.55 | 1878.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1925.40 | 1893.14 | 1886.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 13:15:00 | 1887.10 | 1901.91 | 1893.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 1887.10 | 1901.91 | 1893.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1887.10 | 1901.91 | 1893.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1887.10 | 1901.91 | 1893.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1891.50 | 1899.83 | 1893.61 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1872.80 | 1889.96 | 1890.42 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1914.90 | 1891.45 | 1889.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1920.80 | 1904.76 | 1897.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1905.80 | 1908.20 | 1900.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 1905.80 | 1908.20 | 1900.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1894.40 | 1905.44 | 1900.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1921.90 | 1902.05 | 1899.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1915.00 | 1903.44 | 1900.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1978.00 | 2001.93 | 2002.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 1978.00 | 2001.93 | 2002.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 09:15:00 | 1686.40 | 1932.80 | 1970.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 1662.90 | 1662.71 | 1694.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:30:00 | 1663.20 | 1662.71 | 1694.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1669.20 | 1651.11 | 1659.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1669.20 | 1651.11 | 1659.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1672.00 | 1655.28 | 1660.89 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 1689.90 | 1666.77 | 1665.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 1715.20 | 1676.46 | 1669.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 1696.60 | 1701.40 | 1686.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 1696.60 | 1701.40 | 1686.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1687.20 | 1702.28 | 1694.76 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 1688.50 | 1696.73 | 1697.56 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 1706.40 | 1698.08 | 1697.88 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 1692.70 | 1697.15 | 1697.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 15:15:00 | 1689.00 | 1695.52 | 1696.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1699.00 | 1696.22 | 1696.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 1699.00 | 1696.22 | 1696.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1699.00 | 1696.22 | 1696.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 1699.00 | 1696.22 | 1696.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1691.90 | 1695.35 | 1696.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1689.40 | 1695.35 | 1696.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 1688.50 | 1694.32 | 1695.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 1707.90 | 1697.80 | 1697.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 1707.90 | 1697.80 | 1697.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 15:15:00 | 1710.00 | 1700.24 | 1698.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 1707.00 | 1708.07 | 1703.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 1707.00 | 1708.07 | 1703.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1707.00 | 1708.07 | 1703.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1707.00 | 1708.07 | 1703.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1708.00 | 1708.06 | 1703.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:30:00 | 1706.60 | 1708.06 | 1703.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1702.50 | 1706.95 | 1703.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 1702.50 | 1706.95 | 1703.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1701.20 | 1705.80 | 1703.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 1701.20 | 1705.80 | 1703.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1690.00 | 1702.64 | 1702.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1674.90 | 1702.64 | 1702.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1679.40 | 1697.99 | 1699.96 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1703.10 | 1693.00 | 1692.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 1710.10 | 1696.42 | 1694.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 1731.90 | 1734.63 | 1723.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:30:00 | 1734.20 | 1734.63 | 1723.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1722.50 | 1731.11 | 1723.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 1722.50 | 1731.11 | 1723.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1715.40 | 1727.97 | 1722.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 1715.40 | 1727.97 | 1722.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1707.00 | 1723.77 | 1721.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1707.00 | 1723.77 | 1721.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1701.00 | 1719.22 | 1719.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1683.80 | 1704.52 | 1711.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 1685.40 | 1676.25 | 1685.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 11:15:00 | 1685.40 | 1676.25 | 1685.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 1685.40 | 1676.25 | 1685.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 1685.40 | 1676.25 | 1685.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1688.90 | 1678.78 | 1686.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1688.90 | 1678.78 | 1686.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1682.10 | 1679.44 | 1685.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:15:00 | 1680.10 | 1679.44 | 1685.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1707.00 | 1687.50 | 1688.15 | SL hit (close>static) qty=1.00 sl=1692.40 alert=retest2 |

### Cycle 30 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1698.80 | 1689.76 | 1689.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1723.70 | 1701.45 | 1695.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1693.80 | 1701.91 | 1696.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1693.80 | 1701.91 | 1696.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1693.80 | 1701.91 | 1696.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 1680.60 | 1701.91 | 1696.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1691.20 | 1699.77 | 1696.14 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1683.40 | 1693.96 | 1693.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1669.60 | 1683.55 | 1688.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 1682.00 | 1680.37 | 1685.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 1669.10 | 1680.37 | 1685.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1685.40 | 1681.37 | 1685.31 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 1778.00 | 1703.41 | 1693.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1863.10 | 1795.70 | 1754.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1830.00 | 1830.59 | 1793.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:15:00 | 1808.90 | 1830.59 | 1793.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1804.30 | 1829.65 | 1805.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1818.70 | 1829.65 | 1805.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1850.00 | 1833.72 | 1809.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1827.70 | 1833.72 | 1809.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1860.00 | 1838.03 | 1817.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 1865.90 | 1845.69 | 1826.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 1868.50 | 1878.70 | 1870.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 1849.80 | 1869.50 | 1870.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1849.80 | 1869.50 | 1870.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1834.50 | 1856.59 | 1864.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 1848.80 | 1846.65 | 1855.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 1848.80 | 1846.65 | 1855.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1848.80 | 1846.65 | 1855.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1848.80 | 1846.65 | 1855.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1847.70 | 1846.32 | 1853.68 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1879.70 | 1861.13 | 1859.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1886.30 | 1869.02 | 1863.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 1865.40 | 1873.05 | 1867.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 10:15:00 | 1865.40 | 1873.05 | 1867.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1865.40 | 1873.05 | 1867.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1865.40 | 1873.05 | 1867.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1871.60 | 1872.76 | 1867.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 1880.90 | 1873.64 | 1868.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 15:00:00 | 1887.20 | 1876.35 | 1870.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1858.20 | 1874.38 | 1870.74 | SL hit (close<static) qty=1.00 sl=1863.50 alert=retest2 |

### Cycle 35 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1867.70 | 1869.23 | 1869.34 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 1872.00 | 1869.79 | 1869.58 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1849.70 | 1865.77 | 1867.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1815.00 | 1841.19 | 1854.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 1802.00 | 1794.70 | 1811.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 1802.00 | 1794.70 | 1811.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1808.20 | 1797.91 | 1808.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1808.20 | 1797.91 | 1808.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1815.00 | 1801.33 | 1809.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1811.90 | 1801.33 | 1809.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1812.00 | 1803.47 | 1809.65 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1826.90 | 1813.84 | 1813.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 1837.10 | 1818.49 | 1815.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1842.80 | 1850.10 | 1842.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1842.80 | 1850.10 | 1842.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1842.80 | 1850.10 | 1842.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:30:00 | 1849.80 | 1850.10 | 1842.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1825.10 | 1845.10 | 1840.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1816.60 | 1845.10 | 1840.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1823.60 | 1840.80 | 1839.41 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 1822.90 | 1837.22 | 1837.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 1820.20 | 1833.82 | 1836.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 1810.00 | 1805.18 | 1811.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:15:00 | 1795.30 | 1805.18 | 1811.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1809.00 | 1805.95 | 1811.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1808.70 | 1805.95 | 1811.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1807.00 | 1806.16 | 1810.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1807.30 | 1806.16 | 1810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1820.20 | 1808.97 | 1811.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1820.20 | 1808.97 | 1811.68 | SL hit (close>ema400) qty=1.00 sl=1811.68 alert=retest1 |

### Cycle 40 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1846.50 | 1818.90 | 1815.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1863.90 | 1836.46 | 1825.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 1846.10 | 1847.90 | 1835.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:15:00 | 1828.50 | 1847.90 | 1835.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1836.00 | 1845.52 | 1835.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 1838.80 | 1845.52 | 1835.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1836.20 | 1843.66 | 1835.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 1835.10 | 1843.66 | 1835.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1835.00 | 1841.93 | 1835.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 1835.10 | 1841.93 | 1835.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1847.50 | 1843.04 | 1836.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1842.30 | 1843.04 | 1836.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1834.90 | 1841.41 | 1836.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 1834.90 | 1841.41 | 1836.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1815.70 | 1836.27 | 1834.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1815.70 | 1836.27 | 1834.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1822.00 | 1833.42 | 1833.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1783.60 | 1823.45 | 1829.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1795.00 | 1792.51 | 1808.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 1795.00 | 1792.51 | 1808.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1692.90 | 1680.01 | 1690.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1681.00 | 1680.46 | 1689.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:00:00 | 1681.80 | 1681.09 | 1688.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:15:00 | 1673.70 | 1684.01 | 1688.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 1679.80 | 1686.72 | 1688.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1677.50 | 1684.88 | 1687.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 14:45:00 | 1668.70 | 1678.35 | 1683.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:15:00 | 1662.50 | 1678.35 | 1683.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1638.20 | 1662.23 | 1669.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1596.95 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1597.71 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1590.01 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1595.81 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1585.26 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1579.38 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 1607.80 | 1604.80 | 1623.13 | SL hit (close>ema200) qty=0.50 sl=1604.80 alert=retest2 |

### Cycle 42 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1647.90 | 1628.50 | 1626.52 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1620.70 | 1633.15 | 1633.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1607.00 | 1627.92 | 1630.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 1607.00 | 1606.72 | 1616.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 13:00:00 | 1607.00 | 1606.72 | 1616.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 1615.00 | 1608.38 | 1616.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 1615.00 | 1608.38 | 1616.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1627.20 | 1612.14 | 1617.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1627.20 | 1612.14 | 1617.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1627.60 | 1615.23 | 1618.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1619.10 | 1615.23 | 1618.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1644.80 | 1621.42 | 1618.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1644.80 | 1621.42 | 1618.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 1653.80 | 1627.89 | 1621.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1681.60 | 1694.57 | 1673.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:30:00 | 1707.00 | 1697.49 | 1678.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1723.00 | 1704.05 | 1686.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1673.30 | 1698.37 | 1686.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1673.30 | 1698.37 | 1686.96 | SL hit (close<ema400) qty=1.00 sl=1686.96 alert=retest1 |

### Cycle 45 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1668.40 | 1698.19 | 1698.86 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 1723.80 | 1699.43 | 1698.38 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 1682.10 | 1697.32 | 1698.18 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 1707.70 | 1698.67 | 1698.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 15:15:00 | 1720.00 | 1707.96 | 1703.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 11:15:00 | 1741.60 | 1745.30 | 1735.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:45:00 | 1743.00 | 1745.30 | 1735.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1744.70 | 1746.47 | 1738.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 1740.00 | 1746.47 | 1738.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 1749.80 | 1747.14 | 1739.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 1731.90 | 1747.14 | 1739.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1730.40 | 1743.79 | 1738.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1740.00 | 1743.79 | 1738.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 1729.20 | 1736.53 | 1736.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 1729.20 | 1736.53 | 1736.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 1718.20 | 1732.87 | 1735.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 15:15:00 | 1733.90 | 1733.07 | 1734.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 15:15:00 | 1733.90 | 1733.07 | 1734.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1733.90 | 1733.07 | 1734.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 1757.30 | 1733.07 | 1734.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1765.60 | 1739.58 | 1737.73 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 1736.90 | 1742.41 | 1742.78 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 1771.30 | 1747.85 | 1744.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 12:15:00 | 1810.00 | 1760.28 | 1750.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 1784.70 | 1785.37 | 1775.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1784.70 | 1785.37 | 1775.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1784.70 | 1785.37 | 1775.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1798.60 | 1781.04 | 1776.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1795.00 | 1781.04 | 1776.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1814.70 | 1784.40 | 1778.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 1816.00 | 1799.12 | 1788.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1797.90 | 1798.00 | 1792.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 1790.70 | 1798.00 | 1792.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1795.00 | 1797.40 | 1792.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1771.30 | 1797.40 | 1792.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1771.30 | 1792.18 | 1790.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 1752.60 | 1784.26 | 1787.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1752.60 | 1784.26 | 1787.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1734.80 | 1774.37 | 1782.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1753.00 | 1740.26 | 1759.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1753.00 | 1740.26 | 1759.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1753.00 | 1740.26 | 1759.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1761.00 | 1740.26 | 1759.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1751.60 | 1742.53 | 1758.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1752.30 | 1742.53 | 1758.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1758.00 | 1748.91 | 1757.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1758.00 | 1748.91 | 1757.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1776.10 | 1754.35 | 1759.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 1776.10 | 1754.35 | 1759.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 1775.30 | 1758.54 | 1760.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 1765.10 | 1758.54 | 1760.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:00:00 | 1764.10 | 1759.65 | 1761.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 1763.90 | 1760.38 | 1761.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 1763.70 | 1757.87 | 1759.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1755.20 | 1758.11 | 1758.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 1748.10 | 1756.11 | 1757.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1746.30 | 1752.73 | 1756.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1742.00 | 1753.20 | 1756.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 1748.30 | 1747.42 | 1751.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1748.40 | 1747.62 | 1750.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 1755.10 | 1747.62 | 1750.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1742.80 | 1746.65 | 1750.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1772.70 | 1746.65 | 1750.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1793.60 | 1756.04 | 1754.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1793.60 | 1756.04 | 1754.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1813.20 | 1767.48 | 1759.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 1795.10 | 1801.45 | 1784.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:30:00 | 1790.90 | 1801.45 | 1784.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1787.40 | 1798.64 | 1785.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 1785.00 | 1798.64 | 1785.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1769.70 | 1792.85 | 1783.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 1769.70 | 1792.85 | 1783.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1771.10 | 1788.50 | 1782.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1785.00 | 1783.53 | 1781.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 14:30:00 | 1456.70 | 2025-05-26 10:15:00 | 1524.60 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2025-06-04 09:30:00 | 1614.30 | 2025-06-16 14:15:00 | 1775.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1799.70 | 2025-07-10 10:15:00 | 1863.80 | STOP_HIT | 1.00 | 3.56% |
| SELL | retest2 | 2025-07-11 11:30:00 | 1854.00 | 2025-07-14 12:15:00 | 1880.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest1 | 2025-07-16 09:15:00 | 1914.10 | 2025-07-17 12:15:00 | 2009.81 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-16 09:15:00 | 1914.10 | 2025-07-18 14:15:00 | 2000.00 | STOP_HIT | 0.50 | 4.49% |
| BUY | retest2 | 2025-07-16 11:30:00 | 1983.90 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-16 13:15:00 | 1972.90 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-16 15:00:00 | 1971.90 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-07-17 09:30:00 | 1973.00 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-25 10:15:00 | 2047.00 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-25 13:30:00 | 2027.00 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-25 14:45:00 | 2026.00 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-28 09:15:00 | 2033.40 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-28 14:45:00 | 2062.90 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-29 09:15:00 | 2064.40 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-07-31 09:15:00 | 2067.20 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-07-31 10:45:00 | 2068.10 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-08-05 10:15:00 | 1962.20 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-08-05 10:45:00 | 1963.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-05 15:15:00 | 1958.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-08-06 12:30:00 | 1953.80 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-07 15:15:00 | 1955.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-08-08 10:15:00 | 1949.30 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-08-12 09:30:00 | 1955.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-02 15:15:00 | 1862.30 | 2025-09-04 09:15:00 | 1893.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1921.90 | 2025-09-17 12:15:00 | 1978.00 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-09-11 10:15:00 | 1915.00 | 2025-09-17 12:15:00 | 1978.00 | STOP_HIT | 1.00 | 3.29% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1689.40 | 2026-01-07 14:15:00 | 1707.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-07 12:15:00 | 1688.50 | 2026-01-07 14:15:00 | 1707.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-01-21 14:15:00 | 1680.10 | 2026-01-22 09:15:00 | 1707.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-02 13:00:00 | 1865.90 | 2026-02-05 13:15:00 | 1849.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-04 12:30:00 | 1868.50 | 2026-02-05 13:15:00 | 1849.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-10 14:15:00 | 1880.90 | 2026-02-11 09:15:00 | 1858.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-10 15:00:00 | 1887.20 | 2026-02-11 09:15:00 | 1858.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2026-02-25 09:15:00 | 1795.30 | 2026-02-25 11:15:00 | 1820.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-10 10:45:00 | 1681.00 | 2026-03-16 10:15:00 | 1596.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 13:00:00 | 1681.80 | 2026-03-16 10:15:00 | 1597.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 15:15:00 | 1673.70 | 2026-03-16 10:15:00 | 1590.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:15:00 | 1679.80 | 2026-03-16 10:15:00 | 1595.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:45:00 | 1668.70 | 2026-03-16 10:15:00 | 1585.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:15:00 | 1662.50 | 2026-03-16 10:15:00 | 1579.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 10:45:00 | 1681.00 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-03-10 13:00:00 | 1681.80 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2026-03-10 15:15:00 | 1673.70 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2026-03-11 11:15:00 | 1679.80 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2026-03-11 14:45:00 | 1668.70 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2026-03-11 15:15:00 | 1662.50 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1638.20 | 2026-03-18 09:15:00 | 1647.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1619.10 | 2026-03-24 10:15:00 | 1644.80 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest1 | 2026-03-27 11:30:00 | 1707.00 | 2026-03-30 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2026-03-27 15:00:00 | 1723.00 | 2026-03-30 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-03-30 11:15:00 | 1710.60 | 2026-04-02 09:15:00 | 1679.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-03-30 12:30:00 | 1704.60 | 2026-04-02 09:15:00 | 1679.30 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-03-30 13:45:00 | 1704.40 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1713.30 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-04-01 10:30:00 | 1731.00 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2026-04-01 12:15:00 | 1733.90 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1740.00 | 2026-04-13 13:15:00 | 1729.20 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-22 09:30:00 | 1798.60 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-22 10:15:00 | 1795.00 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-04-22 12:15:00 | 1814.70 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-04-23 09:30:00 | 1816.00 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1765.10 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-28 10:00:00 | 1764.10 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-28 10:30:00 | 1763.90 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-29 09:45:00 | 1763.70 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-29 14:00:00 | 1748.10 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1746.30 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1742.00 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1748.30 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.59% |
