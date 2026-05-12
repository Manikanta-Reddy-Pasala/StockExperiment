# Bombay Burmah Trading Corporation Ltd. (BBTC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1563.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 27 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 58 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 32 / 35
- **Target hits / Stop hits / Partials:** 4 / 52 / 11
- **Avg / median % per leg:** 1.53% / -0.44%
- **Sum % (uncompounded):** 102.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 10 | 41.7% | 4 | 20 | 0 | 1.67% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 10 | 41.7% | 4 | 20 | 0 | 1.67% | 40.0% |
| SELL (all) | 43 | 22 | 51.2% | 0 | 32 | 11 | 1.45% | 62.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 22 | 51.2% | 0 | 32 | 11 | 1.45% | 62.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 67 | 32 | 47.8% | 4 | 52 | 11 | 1.53% | 102.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1800.00 | 1771.55 | 1768.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1808.40 | 1783.58 | 1775.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1777.60 | 1787.29 | 1779.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1777.60 | 1787.29 | 1779.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1777.60 | 1787.29 | 1779.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1777.60 | 1787.29 | 1779.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1780.00 | 1785.83 | 1779.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1780.00 | 1785.83 | 1779.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1796.30 | 1787.92 | 1781.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1819.00 | 1786.90 | 1781.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 09:15:00 | 2000.90 | 1922.58 | 1876.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 2014.00 | 2028.74 | 2029.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 2010.00 | 2024.99 | 2027.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 15:15:00 | 1985.00 | 1983.73 | 1992.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 09:15:00 | 1991.40 | 1983.73 | 1992.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 2022.20 | 1991.42 | 1994.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 2022.20 | 1991.42 | 1994.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 2014.90 | 1996.12 | 1996.63 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 2017.90 | 2000.47 | 1998.56 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 1989.20 | 1999.98 | 2000.20 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2020.70 | 2003.21 | 2000.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 2072.60 | 2021.79 | 2011.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 2009.90 | 2023.94 | 2016.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 2009.90 | 2023.94 | 2016.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2009.90 | 2023.94 | 2016.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2009.90 | 2023.94 | 2016.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1993.60 | 2017.87 | 2014.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 1993.60 | 2017.87 | 2014.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1965.00 | 2007.30 | 2009.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1927.90 | 1969.75 | 1985.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1955.50 | 1953.95 | 1971.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:30:00 | 1953.00 | 1953.95 | 1971.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1961.00 | 1957.11 | 1970.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1960.80 | 1957.11 | 1970.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1978.40 | 1961.37 | 1971.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1977.60 | 1961.37 | 1971.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1974.80 | 1964.06 | 1971.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 1982.40 | 1964.06 | 1971.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1960.30 | 1963.31 | 1970.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 1941.90 | 1963.31 | 1970.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 1955.70 | 1956.63 | 1965.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 1948.70 | 1955.27 | 1962.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1844.81 | 1879.45 | 1902.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1857.91 | 1879.45 | 1902.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1851.26 | 1879.45 | 1902.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 1880.30 | 1864.86 | 1880.88 | SL hit (close>ema200) qty=0.50 sl=1864.86 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 1936.00 | 1895.00 | 1889.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1956.20 | 1929.28 | 1917.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 1979.30 | 1994.65 | 1976.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 1979.30 | 1994.65 | 1976.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1978.80 | 1994.40 | 1984.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1978.80 | 1994.40 | 1984.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2007.00 | 1996.92 | 1986.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 2007.50 | 1999.94 | 1988.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:30:00 | 2015.90 | 2001.35 | 1990.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 1980.00 | 1988.43 | 1989.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 1980.00 | 1988.43 | 1989.58 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 1991.00 | 1988.80 | 1988.75 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 1978.20 | 1986.68 | 1987.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1960.50 | 1979.44 | 1983.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 1971.30 | 1963.42 | 1971.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 1971.30 | 1963.42 | 1971.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1971.30 | 1963.42 | 1971.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1971.30 | 1963.42 | 1971.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1960.80 | 1962.90 | 1970.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 1955.90 | 1961.70 | 1969.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 1956.20 | 1961.94 | 1968.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 1955.10 | 1955.19 | 1963.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1974.30 | 1960.84 | 1962.95 | SL hit (close>static) qty=1.00 sl=1971.50 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1967.50 | 1964.82 | 1964.56 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1951.10 | 1962.07 | 1963.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 1942.30 | 1956.46 | 1960.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1957.80 | 1956.73 | 1960.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 1957.80 | 1956.73 | 1960.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1959.00 | 1957.18 | 1960.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 1959.00 | 1957.18 | 1960.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1940.00 | 1953.75 | 1958.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 1931.10 | 1945.05 | 1951.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1968.00 | 1948.10 | 1950.87 | SL hit (close>static) qty=1.00 sl=1961.70 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1972.70 | 1955.60 | 1953.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2004.40 | 1965.36 | 1958.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 2013.10 | 2023.27 | 2007.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 2013.10 | 2023.27 | 2007.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2016.30 | 2021.88 | 2008.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:30:00 | 2019.80 | 2019.26 | 2009.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 2004.00 | 2014.53 | 2008.89 | SL hit (close<static) qty=1.00 sl=2005.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1978.20 | 2005.42 | 2005.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 1967.90 | 1992.27 | 1999.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1990.00 | 1962.86 | 1972.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1990.00 | 1962.86 | 1972.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1990.00 | 1962.86 | 1972.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 2002.00 | 1962.86 | 1972.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2017.90 | 1973.87 | 1976.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 2017.90 | 1973.87 | 1976.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 2012.60 | 1981.62 | 1979.99 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 1958.90 | 1980.34 | 1982.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1945.70 | 1966.02 | 1973.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 15:15:00 | 1931.00 | 1927.19 | 1943.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 15:15:00 | 1931.00 | 1927.19 | 1943.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1931.00 | 1927.19 | 1943.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1969.00 | 1927.19 | 1943.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1952.70 | 1932.29 | 1944.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 1970.00 | 1932.29 | 1944.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1941.90 | 1934.21 | 1944.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 1941.00 | 1934.21 | 1944.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1940.40 | 1935.45 | 1943.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 1942.10 | 1935.45 | 1943.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1944.00 | 1937.16 | 1943.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 1944.00 | 1937.16 | 1943.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1940.00 | 1937.73 | 1943.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:15:00 | 1935.10 | 1939.08 | 1943.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1952.40 | 1941.11 | 1943.69 | SL hit (close>static) qty=1.00 sl=1944.30 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 1863.00 | 1846.98 | 1846.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 15:15:00 | 1878.00 | 1853.18 | 1849.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1855.10 | 1856.65 | 1853.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 1855.10 | 1856.65 | 1853.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1860.00 | 1857.32 | 1853.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1847.00 | 1857.32 | 1853.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1846.00 | 1855.06 | 1853.15 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1833.20 | 1850.69 | 1851.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1831.00 | 1838.86 | 1844.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1851.10 | 1841.31 | 1845.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1851.10 | 1841.31 | 1845.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1851.10 | 1841.31 | 1845.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 1832.80 | 1840.69 | 1844.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 14:15:00 | 1852.40 | 1845.94 | 1845.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 1852.40 | 1845.94 | 1845.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 1859.90 | 1848.73 | 1847.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1847.70 | 1848.53 | 1847.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1847.70 | 1848.53 | 1847.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1847.70 | 1848.53 | 1847.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 1869.30 | 1859.30 | 1853.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 1840.00 | 1854.33 | 1855.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1840.00 | 1854.33 | 1855.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1835.10 | 1848.19 | 1852.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1818.80 | 1791.48 | 1803.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 1818.80 | 1791.48 | 1803.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1818.80 | 1791.48 | 1803.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 1818.80 | 1791.48 | 1803.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1831.20 | 1799.42 | 1806.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:30:00 | 1840.20 | 1799.42 | 1806.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1806.10 | 1802.79 | 1806.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:45:00 | 1809.60 | 1802.79 | 1806.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1810.00 | 1804.23 | 1806.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 1810.00 | 1804.23 | 1806.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1800.10 | 1803.40 | 1806.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 1800.10 | 1803.40 | 1806.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 1804.50 | 1803.62 | 1806.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 1824.30 | 1803.62 | 1806.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1790.00 | 1800.90 | 1804.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1783.40 | 1797.87 | 1802.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 1787.10 | 1780.91 | 1790.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1782.00 | 1784.79 | 1788.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 1822.80 | 1797.26 | 1793.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1822.80 | 1797.26 | 1793.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1837.30 | 1826.91 | 1815.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 1867.50 | 1867.78 | 1849.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 11:15:00 | 1848.20 | 1860.95 | 1849.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1848.20 | 1860.95 | 1849.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1848.20 | 1860.95 | 1849.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1846.00 | 1857.96 | 1848.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 1842.90 | 1857.96 | 1848.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1852.40 | 1856.85 | 1849.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1862.80 | 1855.32 | 1851.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 1858.70 | 1857.60 | 1854.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1860.30 | 1856.77 | 1854.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1867.00 | 1858.47 | 1855.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1893.00 | 1865.37 | 1859.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1911.00 | 1883.83 | 1880.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1937.90 | 1947.60 | 1948.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1937.90 | 1947.60 | 1948.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 1931.10 | 1939.69 | 1943.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 1890.00 | 1885.69 | 1900.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 1890.00 | 1885.69 | 1900.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1890.00 | 1885.69 | 1900.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 1897.10 | 1885.69 | 1900.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1769.70 | 1767.05 | 1784.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1784.60 | 1767.05 | 1784.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1782.40 | 1773.37 | 1783.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 1782.40 | 1773.37 | 1783.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1789.30 | 1776.56 | 1783.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1789.30 | 1776.56 | 1783.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1808.30 | 1782.91 | 1785.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 1808.30 | 1782.91 | 1785.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1809.00 | 1788.13 | 1788.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1817.00 | 1793.90 | 1790.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1887.90 | 1891.02 | 1863.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 1887.90 | 1891.02 | 1863.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1869.50 | 1880.97 | 1869.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 1884.60 | 1880.97 | 1869.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1844.20 | 1862.85 | 1864.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1844.20 | 1862.85 | 1864.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1840.30 | 1855.34 | 1860.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1860.90 | 1849.04 | 1853.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1860.90 | 1849.04 | 1853.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1860.90 | 1849.04 | 1853.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 1860.90 | 1849.04 | 1853.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1852.40 | 1849.71 | 1853.11 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1865.80 | 1855.67 | 1855.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1875.10 | 1859.55 | 1857.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1864.70 | 1865.29 | 1860.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1864.70 | 1865.29 | 1860.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1864.70 | 1865.29 | 1860.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 1861.00 | 1865.29 | 1860.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1857.40 | 1863.72 | 1860.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1851.40 | 1863.72 | 1860.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1873.50 | 1865.67 | 1861.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 1880.80 | 1865.67 | 1861.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:30:00 | 1882.00 | 1871.51 | 1864.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 15:00:00 | 1907.80 | 1888.52 | 1884.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-17 12:15:00 | 2068.88 | 1951.52 | 1921.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 2039.40 | 2054.40 | 2056.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 2031.60 | 2046.50 | 2052.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 2015.30 | 2011.28 | 2024.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:00:00 | 2015.30 | 2011.28 | 2024.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 2020.30 | 2012.99 | 2022.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:30:00 | 2025.00 | 2012.99 | 2022.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1997.00 | 2011.43 | 2019.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 1988.30 | 2003.99 | 2013.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1984.50 | 2000.22 | 2009.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 1977.90 | 1986.92 | 1995.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1888.88 | 1915.21 | 1939.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1885.27 | 1915.21 | 1939.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1879.01 | 1915.21 | 1939.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 1905.40 | 1901.28 | 1925.92 | SL hit (close>ema200) qty=0.50 sl=1901.28 alert=retest2 |

### Cycle 27 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 1939.00 | 1926.78 | 1925.87 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1875.40 | 1916.50 | 1921.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1853.30 | 1869.76 | 1880.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1855.40 | 1854.96 | 1869.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:15:00 | 1853.80 | 1854.96 | 1869.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 29 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 1954.70 | 1867.76 | 1867.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 1982.80 | 1890.77 | 1877.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1964.80 | 1971.26 | 1936.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 11:00:00 | 1964.80 | 1971.26 | 1936.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1933.60 | 1963.77 | 1948.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1933.60 | 1963.77 | 1948.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1935.00 | 1958.02 | 1946.92 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 1920.00 | 1939.91 | 1940.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1896.20 | 1924.64 | 1932.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 1863.70 | 1863.43 | 1884.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 1863.70 | 1863.43 | 1884.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1844.60 | 1843.58 | 1851.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1834.90 | 1844.25 | 1848.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1835.10 | 1843.38 | 1847.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1834.80 | 1843.38 | 1847.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 1833.00 | 1840.24 | 1845.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1850.00 | 1839.09 | 1842.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 1851.30 | 1839.09 | 1842.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1846.90 | 1840.65 | 1842.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1842.00 | 1841.52 | 1843.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1837.20 | 1841.51 | 1842.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 1885.70 | 1835.77 | 1835.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1885.70 | 1835.77 | 1835.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 12:15:00 | 1914.30 | 1851.48 | 1842.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 1863.00 | 1867.59 | 1858.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 1863.00 | 1867.59 | 1858.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1863.00 | 1867.59 | 1858.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1863.00 | 1867.59 | 1858.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1860.00 | 1866.07 | 1858.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1860.00 | 1866.07 | 1858.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1854.00 | 1863.65 | 1858.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1840.10 | 1863.65 | 1858.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1834.70 | 1857.86 | 1856.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1834.70 | 1857.86 | 1856.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 10:15:00 | 1830.00 | 1852.29 | 1853.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 11:15:00 | 1825.80 | 1846.99 | 1851.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1852.00 | 1844.55 | 1848.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 1852.00 | 1844.55 | 1848.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1852.00 | 1844.55 | 1848.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 1852.00 | 1844.55 | 1848.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1843.80 | 1844.40 | 1848.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1836.00 | 1844.40 | 1848.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1858.70 | 1835.73 | 1837.34 | SL hit (close>static) qty=1.00 sl=1857.30 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1858.00 | 1840.18 | 1839.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 1870.20 | 1846.18 | 1842.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1864.80 | 1877.73 | 1865.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1864.80 | 1877.73 | 1865.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1864.80 | 1877.73 | 1865.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1864.80 | 1877.73 | 1865.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1854.00 | 1872.98 | 1864.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1872.50 | 1872.98 | 1864.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 1872.00 | 1872.79 | 1864.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1863.70 | 1870.56 | 1871.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1863.70 | 1870.56 | 1871.34 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1877.80 | 1871.77 | 1871.65 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 1867.00 | 1871.52 | 1871.60 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 1872.50 | 1871.72 | 1871.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 1918.10 | 1880.99 | 1875.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 1858.50 | 1881.41 | 1878.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 1858.50 | 1881.41 | 1878.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1858.50 | 1881.41 | 1878.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1858.50 | 1881.41 | 1878.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1855.00 | 1876.12 | 1876.76 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 1890.70 | 1879.04 | 1878.03 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 1877.60 | 1880.95 | 1881.00 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 1899.00 | 1884.56 | 1882.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 1902.20 | 1888.96 | 1885.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 13:15:00 | 1907.90 | 1912.96 | 1903.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:00:00 | 1907.90 | 1912.96 | 1903.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1902.90 | 1910.95 | 1903.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:30:00 | 1901.20 | 1910.95 | 1903.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1909.00 | 1910.56 | 1903.76 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 1886.00 | 1900.79 | 1902.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 1882.90 | 1897.22 | 1900.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 1899.00 | 1847.03 | 1852.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 1899.00 | 1847.03 | 1852.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1899.00 | 1847.03 | 1852.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 1883.60 | 1847.03 | 1852.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 1905.20 | 1858.66 | 1857.15 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1862.30 | 1871.58 | 1872.63 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1878.10 | 1873.44 | 1873.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 1883.60 | 1876.59 | 1874.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 1874.70 | 1876.21 | 1874.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 14:15:00 | 1874.70 | 1876.21 | 1874.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1874.70 | 1876.21 | 1874.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 1874.70 | 1876.21 | 1874.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1884.00 | 1877.77 | 1875.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 1894.50 | 1880.41 | 1876.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:45:00 | 1889.80 | 1881.49 | 1877.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 1872.80 | 1878.70 | 1877.10 | SL hit (close<static) qty=1.00 sl=1873.50 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 1870.80 | 1875.58 | 1875.86 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1886.10 | 1876.59 | 1876.20 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1873.10 | 1875.89 | 1875.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 1870.20 | 1874.75 | 1875.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 1875.60 | 1871.84 | 1873.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 1875.60 | 1871.84 | 1873.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1875.60 | 1871.84 | 1873.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1875.60 | 1871.84 | 1873.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 1879.00 | 1873.27 | 1874.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 1878.20 | 1873.27 | 1874.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1887.50 | 1876.12 | 1875.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 1910.00 | 1882.90 | 1878.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 15:15:00 | 1896.10 | 1899.35 | 1889.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:15:00 | 1890.20 | 1899.35 | 1889.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1884.60 | 1896.40 | 1889.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1884.60 | 1896.40 | 1889.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1872.60 | 1891.64 | 1887.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1872.60 | 1891.64 | 1887.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1867.40 | 1882.10 | 1883.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1832.20 | 1869.58 | 1877.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1808.00 | 1803.88 | 1822.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1825.40 | 1803.88 | 1822.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1828.80 | 1808.86 | 1823.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1811.00 | 1814.24 | 1821.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1841.90 | 1825.00 | 1824.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1841.90 | 1825.00 | 1824.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 1848.60 | 1837.78 | 1831.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1835.60 | 1837.34 | 1832.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 1835.60 | 1837.34 | 1832.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1835.60 | 1837.34 | 1832.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 1829.40 | 1837.34 | 1832.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1822.50 | 1834.37 | 1831.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 1822.50 | 1834.37 | 1831.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1826.40 | 1832.78 | 1830.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 1823.20 | 1832.78 | 1830.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1826.90 | 1831.60 | 1830.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 1823.10 | 1831.60 | 1830.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 1821.00 | 1829.48 | 1829.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1812.80 | 1824.63 | 1827.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1758.00 | 1753.29 | 1777.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 1758.00 | 1753.29 | 1777.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1768.80 | 1747.55 | 1762.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1771.30 | 1747.55 | 1762.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1769.00 | 1751.84 | 1762.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 1768.70 | 1751.84 | 1762.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1761.10 | 1755.01 | 1762.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 1765.00 | 1755.01 | 1762.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1776.00 | 1759.21 | 1763.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1776.00 | 1759.21 | 1763.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1777.20 | 1762.81 | 1764.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:15:00 | 1790.00 | 1762.81 | 1764.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1790.00 | 1768.25 | 1767.18 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1735.10 | 1762.25 | 1765.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1729.80 | 1755.76 | 1762.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 1719.60 | 1715.27 | 1731.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1719.60 | 1715.27 | 1731.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1719.60 | 1715.27 | 1731.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1728.00 | 1715.27 | 1731.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1724.20 | 1716.32 | 1725.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 1727.20 | 1716.32 | 1725.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1728.80 | 1718.82 | 1725.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 1726.20 | 1718.82 | 1725.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1715.00 | 1718.06 | 1724.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 1708.20 | 1718.06 | 1724.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1710.50 | 1715.86 | 1720.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1692.00 | 1716.05 | 1720.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 1732.70 | 1723.24 | 1722.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1732.70 | 1723.24 | 1722.14 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1679.00 | 1713.01 | 1717.59 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1742.60 | 1704.77 | 1703.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1786.30 | 1776.62 | 1766.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1793.30 | 1793.75 | 1783.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1793.30 | 1793.75 | 1783.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1827.10 | 1836.10 | 1818.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 1822.60 | 1836.10 | 1818.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1816.80 | 1826.86 | 1820.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1816.80 | 1826.86 | 1820.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1808.00 | 1823.09 | 1819.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 1785.90 | 1823.09 | 1819.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1776.80 | 1813.83 | 1815.32 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1819.30 | 1807.76 | 1807.24 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1798.20 | 1810.42 | 1811.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1793.80 | 1807.10 | 1809.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 1782.60 | 1778.27 | 1790.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 1782.60 | 1778.27 | 1790.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1796.90 | 1783.02 | 1790.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 1800.50 | 1783.02 | 1790.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1799.50 | 1786.32 | 1791.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1799.50 | 1786.32 | 1791.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1766.40 | 1766.75 | 1774.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 1748.30 | 1762.87 | 1770.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1751.90 | 1760.05 | 1767.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1753.00 | 1758.90 | 1765.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1660.88 | 1697.87 | 1719.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1664.31 | 1697.87 | 1719.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1665.35 | 1697.87 | 1719.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 1668.00 | 1667.54 | 1692.35 | SL hit (close>ema200) qty=0.50 sl=1667.54 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 1592.80 | 1575.80 | 1574.07 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1527.60 | 1565.38 | 1570.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1515.10 | 1544.72 | 1558.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1506.00 | 1504.01 | 1521.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 14:15:00 | 1508.70 | 1506.84 | 1516.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1508.70 | 1506.84 | 1516.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 1516.80 | 1506.84 | 1516.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1521.00 | 1509.67 | 1516.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1534.80 | 1509.67 | 1516.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1533.60 | 1514.46 | 1518.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1537.60 | 1514.46 | 1518.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1541.10 | 1519.79 | 1520.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1541.10 | 1519.79 | 1520.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1543.90 | 1524.61 | 1522.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 1561.70 | 1532.03 | 1526.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1518.00 | 1534.35 | 1529.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1518.00 | 1534.35 | 1529.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1518.00 | 1534.35 | 1529.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1518.00 | 1534.35 | 1529.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1522.90 | 1532.06 | 1529.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 1531.50 | 1529.48 | 1528.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 1517.10 | 1527.01 | 1527.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1517.10 | 1527.01 | 1527.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1513.30 | 1524.27 | 1526.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1532.00 | 1524.43 | 1525.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1532.00 | 1524.43 | 1525.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1532.00 | 1524.43 | 1525.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1532.00 | 1524.43 | 1525.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1522.40 | 1524.02 | 1525.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1518.80 | 1524.02 | 1525.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 1442.86 | 1487.25 | 1506.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1440.50 | 1425.43 | 1444.94 | SL hit (close>ema200) qty=0.50 sl=1425.43 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1405.80 | 1387.58 | 1385.24 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1364.00 | 1381.57 | 1383.17 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1399.00 | 1384.88 | 1384.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1413.30 | 1390.57 | 1387.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1422.70 | 1423.47 | 1411.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1422.70 | 1423.47 | 1411.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1448.80 | 1456.58 | 1447.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 1448.00 | 1456.58 | 1447.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1445.30 | 1454.32 | 1447.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 1443.80 | 1454.32 | 1447.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1445.90 | 1452.64 | 1447.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 1466.20 | 1452.64 | 1447.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1485.10 | 1459.13 | 1450.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:45:00 | 1495.60 | 1469.46 | 1457.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:00:00 | 1490.10 | 1481.61 | 1466.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 1458.30 | 1464.92 | 1464.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1458.30 | 1464.92 | 1464.97 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1483.80 | 1468.70 | 1466.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1507.20 | 1479.34 | 1472.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1477.20 | 1486.79 | 1479.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1477.20 | 1486.79 | 1479.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1477.20 | 1486.79 | 1479.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1477.20 | 1486.79 | 1479.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 1500.50 | 1489.53 | 1481.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 1505.00 | 1489.53 | 1481.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1529.70 | 1495.32 | 1487.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 1513.30 | 1518.20 | 1506.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 1508.20 | 1532.31 | 1533.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1508.20 | 1532.31 | 1533.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1503.00 | 1526.45 | 1530.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1512.00 | 1506.77 | 1517.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1512.00 | 1506.77 | 1517.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1512.00 | 1506.77 | 1517.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1516.90 | 1506.77 | 1517.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1512.00 | 1507.82 | 1517.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1516.70 | 1507.82 | 1517.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1522.90 | 1511.33 | 1517.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1522.90 | 1511.33 | 1517.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1522.10 | 1513.48 | 1517.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 1517.20 | 1513.87 | 1517.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1528.50 | 1505.39 | 1505.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1528.50 | 1505.39 | 1505.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1547.30 | 1513.77 | 1509.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1519.80 | 1523.37 | 1517.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 1519.80 | 1523.37 | 1517.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1519.80 | 1523.37 | 1517.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:15:00 | 1513.00 | 1523.37 | 1517.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1515.50 | 1521.80 | 1517.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 1511.60 | 1521.80 | 1517.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1511.20 | 1519.68 | 1516.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 1505.50 | 1519.68 | 1516.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1516.60 | 1519.06 | 1516.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:45:00 | 1517.00 | 1516.91 | 1515.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1525.80 | 1516.19 | 1515.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 1819.00 | 2025-05-19 09:15:00 | 2000.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 12:15:00 | 1941.90 | 2025-06-20 14:15:00 | 1844.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1955.70 | 2025-06-20 14:15:00 | 1857.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 10:45:00 | 1948.70 | 2025-06-20 14:15:00 | 1851.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:15:00 | 1941.90 | 2025-06-23 14:15:00 | 1880.30 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1955.70 | 2025-06-23 14:15:00 | 1880.30 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-06-18 10:45:00 | 1948.70 | 2025-06-23 14:15:00 | 1880.30 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-07-02 11:45:00 | 2007.50 | 2025-07-03 15:15:00 | 1980.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-02 12:30:00 | 2015.90 | 2025-07-03 15:15:00 | 1980.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-09 12:45:00 | 1955.90 | 2025-07-10 14:15:00 | 1974.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-09 14:15:00 | 1956.20 | 2025-07-10 14:15:00 | 1974.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-10 09:30:00 | 1955.10 | 2025-07-10 14:15:00 | 1974.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-14 15:15:00 | 1931.10 | 2025-07-15 10:15:00 | 1968.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-18 12:30:00 | 2019.80 | 2025-07-18 14:15:00 | 2004.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-29 15:15:00 | 1935.10 | 2025-07-30 09:15:00 | 1952.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1933.20 | 2025-08-06 15:15:00 | 1836.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1933.20 | 2025-08-07 12:15:00 | 1855.80 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-08-18 12:00:00 | 1832.80 | 2025-08-18 14:15:00 | 1852.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-08-20 10:30:00 | 1869.30 | 2025-08-21 14:15:00 | 1840.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-08-29 12:15:00 | 1783.40 | 2025-09-02 10:15:00 | 1822.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-01 10:45:00 | 1787.10 | 2025-09-02 10:15:00 | 1822.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1782.00 | 2025-09-02 10:15:00 | 1822.80 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-09-08 15:00:00 | 1862.80 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2025-09-09 11:00:00 | 1858.70 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 4.26% |
| BUY | retest2 | 2025-09-09 13:00:00 | 1860.30 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 4.17% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1867.00 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 3.80% |
| BUY | retest2 | 2025-09-15 10:15:00 | 1911.00 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-10-08 10:15:00 | 1884.60 | 2025-10-08 14:15:00 | 1844.20 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-13 12:15:00 | 1880.80 | 2025-10-17 12:15:00 | 2068.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-13 13:30:00 | 1882.00 | 2025-10-17 12:15:00 | 2070.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 15:00:00 | 1907.80 | 2025-10-23 09:15:00 | 2098.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 1988.30 | 2025-11-07 09:15:00 | 1888.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1984.50 | 2025-11-07 09:15:00 | 1885.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:15:00 | 1977.90 | 2025-11-07 09:15:00 | 1879.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 1988.30 | 2025-11-07 12:15:00 | 1905.40 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1984.50 | 2025-11-07 12:15:00 | 1905.40 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-11-03 10:15:00 | 1977.90 | 2025-11-07 12:15:00 | 1905.40 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1834.90 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-11-28 09:45:00 | 1835.10 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-11-28 10:15:00 | 1834.80 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-11-28 12:00:00 | 1833.00 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1842.00 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-12-01 14:15:00 | 1837.20 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1836.00 | 2025-12-09 11:15:00 | 1858.70 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1872.50 | 2025-12-15 09:15:00 | 1863.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-11 10:00:00 | 1872.00 | 2025-12-15 09:15:00 | 1863.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-01-05 09:30:00 | 1894.50 | 2026-01-05 12:15:00 | 1872.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-05 10:45:00 | 1889.80 | 2026-01-05 12:15:00 | 1872.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1811.00 | 2026-01-14 10:15:00 | 1841.90 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-29 10:15:00 | 1708.20 | 2026-02-01 10:15:00 | 1732.70 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1710.50 | 2026-02-01 10:15:00 | 1732.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1692.00 | 2026-02-01 10:15:00 | 1732.70 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1748.30 | 2026-03-02 09:15:00 | 1660.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1751.90 | 2026-03-02 09:15:00 | 1664.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:30:00 | 1753.00 | 2026-03-02 09:15:00 | 1665.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1748.30 | 2026-03-02 15:15:00 | 1668.00 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1751.90 | 2026-03-02 15:15:00 | 1668.00 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2026-02-26 10:30:00 | 1753.00 | 2026-03-02 15:15:00 | 1668.00 | STOP_HIT | 0.50 | 4.85% |
| BUY | retest2 | 2026-03-19 12:45:00 | 1531.50 | 2026-03-19 13:15:00 | 1517.10 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1518.80 | 2026-03-23 10:15:00 | 1442.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1518.80 | 2026-03-25 09:15:00 | 1440.50 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2026-04-10 11:45:00 | 1495.60 | 2026-04-13 15:15:00 | 1458.30 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-04-10 15:00:00 | 1490.10 | 2026-04-13 15:15:00 | 1458.30 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-16 11:15:00 | 1505.00 | 2026-04-24 10:15:00 | 1508.20 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2026-04-17 09:15:00 | 1529.70 | 2026-04-24 10:15:00 | 1508.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-20 09:45:00 | 1513.30 | 2026-04-24 10:15:00 | 1508.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-04-27 14:30:00 | 1517.20 | 2026-05-04 09:15:00 | 1528.50 | STOP_HIT | 1.00 | -0.74% |
