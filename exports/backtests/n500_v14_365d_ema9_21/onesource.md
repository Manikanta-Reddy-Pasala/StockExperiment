# Onesource Specialty Pharma Ltd. (ONESOURCE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1836.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 89 |
| ALERT1 | 58 |
| ALERT2 | 56 |
| ALERT2_SKIP | 38 |
| ALERT3 | 130 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 68 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 56
- **Target hits / Stop hits / Partials:** 6 / 65 / 4
- **Avg / median % per leg:** -0.21% / -0.93%
- **Sum % (uncompounded):** -16.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 11 | 39.3% | 5 | 23 | 0 | 1.28% | 35.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.35% | -8.7% |
| BUY @ 3rd Alert (retest2) | 26 | 11 | 42.3% | 5 | 21 | 0 | 1.71% | 44.5% |
| SELL (all) | 47 | 8 | 17.0% | 1 | 42 | 4 | -1.10% | -51.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.26% | -2.3% |
| SELL @ 3rd Alert (retest2) | 46 | 8 | 17.4% | 1 | 41 | 4 | -1.08% | -49.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.65% | -11.0% |
| retest2 (combined) | 72 | 19 | 26.4% | 6 | 62 | 4 | -0.07% | -5.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 1653.70 | 1613.37 | 1608.74 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 1606.60 | 1612.07 | 1612.56 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1616.10 | 1602.11 | 1601.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 14:15:00 | 1655.00 | 1623.90 | 1615.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 1668.10 | 1671.30 | 1655.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:45:00 | 1666.80 | 1671.30 | 1655.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1676.00 | 1671.01 | 1660.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 1694.40 | 1675.69 | 1663.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1709.40 | 1684.73 | 1673.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 1693.90 | 1692.93 | 1680.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 10:15:00 | 1863.84 | 1821.92 | 1794.50 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-30 10:15:00 | 1880.34 | 1821.92 | 1794.50 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-30 10:15:00 | 1863.29 | 1821.92 | 1794.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1960.70 | 1999.16 | 2000.97 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 2028.00 | 2000.21 | 1997.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 2041.30 | 2008.43 | 2001.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 2086.40 | 2115.21 | 2072.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:00:00 | 2086.40 | 2115.21 | 2072.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 2089.30 | 2105.45 | 2075.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 2101.60 | 2105.45 | 2075.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2085.00 | 2132.00 | 2136.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 2085.00 | 2132.00 | 2136.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 2018.60 | 2102.16 | 2119.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 14:15:00 | 1993.00 | 1954.07 | 1987.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 1993.00 | 1954.07 | 1987.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1993.00 | 1954.07 | 1987.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1993.00 | 1954.07 | 1987.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1981.00 | 1959.45 | 1986.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 2015.10 | 1959.45 | 1986.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 2004.50 | 1968.46 | 1988.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 2003.20 | 1968.46 | 1988.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1965.10 | 1967.79 | 1986.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 1961.00 | 1966.43 | 1983.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 15:15:00 | 1996.00 | 1983.31 | 1982.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 1996.00 | 1983.31 | 1982.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 2037.50 | 1997.28 | 1989.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 2147.70 | 2161.41 | 2113.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:45:00 | 2149.20 | 2161.41 | 2113.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 2119.30 | 2143.65 | 2119.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 2119.30 | 2143.65 | 2119.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 2112.20 | 2137.36 | 2118.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:45:00 | 2119.00 | 2137.36 | 2118.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 2102.00 | 2130.29 | 2117.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 2104.20 | 2130.29 | 2117.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2101.00 | 2124.43 | 2115.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 2100.10 | 2124.43 | 2115.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2090.60 | 2117.66 | 2113.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 2089.50 | 2117.66 | 2113.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 2071.00 | 2103.12 | 2107.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 2047.00 | 2091.90 | 2101.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 2013.00 | 2007.57 | 2036.52 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:45:00 | 1990.00 | 2004.46 | 2030.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2005.00 | 2004.57 | 2021.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 1992.40 | 2004.57 | 2021.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 2035.00 | 2010.66 | 2022.93 | SL hit (close>ema400) qty=1.00 sl=2022.93 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 2035.00 | 2010.66 | 2022.93 | SL hit (close>static) qty=1.00 sl=2033.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1995.80 | 2012.22 | 2020.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 1999.80 | 2006.79 | 2014.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 1999.10 | 2002.65 | 2004.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1981.10 | 1994.67 | 1999.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 2004.50 | 1994.94 | 1999.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2001.00 | 1996.15 | 1999.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 1999.70 | 1996.15 | 1999.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1996.10 | 1996.14 | 1998.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 1991.00 | 1997.48 | 1999.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 1991.90 | 1991.33 | 1995.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:30:00 | 1996.00 | 1991.06 | 1994.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 2009.90 | 1998.42 | 1997.32 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 10:15:00 | 1984.20 | 1995.67 | 1996.27 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 2002.00 | 1997.35 | 1996.73 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 1987.20 | 1995.32 | 1995.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 1983.00 | 1992.86 | 1994.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1996.70 | 1988.00 | 1991.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1996.70 | 1988.00 | 1991.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1996.70 | 1988.00 | 1991.51 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 2001.60 | 1994.91 | 1994.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 14:15:00 | 2009.20 | 1997.78 | 1995.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1987.90 | 1996.78 | 1995.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 1987.90 | 1996.78 | 1995.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1987.90 | 1996.78 | 1995.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1990.00 | 1996.78 | 1995.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1995.70 | 1996.57 | 1995.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:15:00 | 1988.60 | 1996.57 | 1995.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1984.40 | 1994.13 | 1994.57 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 1995.00 | 1994.78 | 1994.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 1999.00 | 1995.63 | 1995.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 1991.20 | 1995.75 | 1995.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 1991.20 | 1995.75 | 1995.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1991.20 | 1995.75 | 1995.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 1991.20 | 1995.75 | 1995.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1997.70 | 1996.14 | 1995.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1999.80 | 1996.57 | 1995.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1998.80 | 1996.26 | 1995.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 1985.80 | 1994.06 | 1994.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 1985.80 | 1994.06 | 1994.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 1985.80 | 1994.06 | 1994.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 1967.10 | 1985.22 | 1990.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1973.30 | 1971.36 | 1981.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 1990.00 | 1975.09 | 1982.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1990.00 | 1975.09 | 1982.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 1990.00 | 1975.09 | 1982.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1988.90 | 1977.85 | 1983.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:30:00 | 1980.50 | 1981.88 | 1984.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 1998.30 | 1985.16 | 1985.69 | SL hit (close>static) qty=1.00 sl=1997.90 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 1998.70 | 1987.87 | 1986.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 2005.50 | 1995.71 | 1991.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1989.00 | 1996.74 | 1993.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1989.00 | 1996.74 | 1993.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1989.00 | 1996.74 | 1993.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1989.00 | 1996.74 | 1993.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1970.00 | 1991.39 | 1990.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1995.00 | 1991.39 | 1990.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1985.90 | 1990.39 | 1990.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1985.90 | 1990.39 | 1990.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 1982.00 | 1988.71 | 1989.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 1996.70 | 1965.35 | 1972.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 14:15:00 | 1996.70 | 1965.35 | 1972.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1996.70 | 1965.35 | 1972.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 1996.70 | 1965.35 | 1972.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1993.90 | 1971.06 | 1974.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 1959.40 | 1971.06 | 1974.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:45:00 | 1976.30 | 1968.54 | 1970.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 1991.30 | 1973.04 | 1972.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 1991.30 | 1973.04 | 1972.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1991.30 | 1973.04 | 1972.18 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1952.30 | 1969.69 | 1971.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 1944.80 | 1964.71 | 1968.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 13:15:00 | 1964.90 | 1958.76 | 1964.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 13:15:00 | 1964.90 | 1958.76 | 1964.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1964.90 | 1958.76 | 1964.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:45:00 | 1961.40 | 1958.76 | 1964.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1976.50 | 1962.31 | 1965.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:45:00 | 1974.80 | 1962.31 | 1965.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1973.90 | 1964.63 | 1966.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1943.50 | 1964.63 | 1966.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 14:15:00 | 1846.32 | 1891.28 | 1923.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 1905.70 | 1873.60 | 1900.23 | SL hit (close>ema200) qty=0.50 sl=1873.60 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 1883.80 | 1878.04 | 1877.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1901.80 | 1890.63 | 1885.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 1886.60 | 1895.05 | 1890.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 1886.60 | 1895.05 | 1890.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1886.60 | 1895.05 | 1890.86 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 1886.90 | 1890.55 | 1890.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 1876.20 | 1887.58 | 1889.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1832.50 | 1825.65 | 1836.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1832.50 | 1825.65 | 1836.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1832.50 | 1825.65 | 1836.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 1832.50 | 1825.65 | 1836.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1847.00 | 1829.92 | 1837.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1847.00 | 1829.92 | 1837.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1832.10 | 1830.36 | 1836.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 1828.00 | 1830.36 | 1836.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1818.00 | 1834.43 | 1836.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1861.00 | 1831.81 | 1832.99 | SL hit (close>static) qty=1.00 sl=1848.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1861.00 | 1831.81 | 1832.99 | SL hit (close>static) qty=1.00 sl=1848.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 15:15:00 | 1850.00 | 1835.44 | 1834.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 1883.10 | 1850.90 | 1843.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1844.80 | 1852.42 | 1845.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1844.80 | 1852.42 | 1845.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1844.80 | 1852.42 | 1845.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1831.30 | 1852.42 | 1845.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1850.00 | 1851.93 | 1845.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 1850.00 | 1851.93 | 1845.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1869.10 | 1868.69 | 1860.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 1869.10 | 1868.69 | 1860.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 1851.10 | 1865.17 | 1859.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 1892.90 | 1865.17 | 1859.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 1872.00 | 1881.30 | 1873.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 1873.60 | 1879.06 | 1873.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:30:00 | 1871.80 | 1873.75 | 1873.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1867.80 | 1872.56 | 1872.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1867.80 | 1872.56 | 1872.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1867.80 | 1872.56 | 1872.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1867.80 | 1872.56 | 1872.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 1867.80 | 1872.56 | 1872.78 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 13:15:00 | 1887.50 | 1875.02 | 1873.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 1895.60 | 1879.14 | 1875.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 1881.10 | 1881.11 | 1877.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:30:00 | 1881.90 | 1881.11 | 1877.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1881.70 | 1881.23 | 1877.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 1894.10 | 1883.25 | 1879.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:00:00 | 1886.20 | 1887.90 | 1884.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 1875.10 | 1885.34 | 1883.41 | SL hit (close<static) qty=1.00 sl=1876.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 1875.10 | 1885.34 | 1883.41 | SL hit (close<static) qty=1.00 sl=1876.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1880.00 | 1882.33 | 1882.35 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1888.50 | 1883.57 | 1882.91 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1870.40 | 1881.82 | 1882.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1861.00 | 1877.66 | 1880.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1847.30 | 1843.80 | 1856.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1847.30 | 1843.80 | 1856.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1847.30 | 1843.80 | 1856.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 1832.10 | 1838.48 | 1851.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 1833.90 | 1838.61 | 1849.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 1832.70 | 1837.43 | 1847.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1827.20 | 1840.34 | 1848.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1841.30 | 1840.53 | 1847.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 1820.20 | 1833.39 | 1842.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1851.00 | 1843.45 | 1842.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1851.00 | 1843.45 | 1842.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1851.00 | 1843.45 | 1842.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1851.00 | 1843.45 | 1842.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1851.00 | 1843.45 | 1842.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 14:15:00 | 1851.00 | 1843.45 | 1842.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 1867.10 | 1849.92 | 1846.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 1848.10 | 1851.17 | 1848.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 1848.10 | 1851.17 | 1848.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1848.10 | 1851.17 | 1848.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 1842.10 | 1851.17 | 1848.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1848.20 | 1850.58 | 1848.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 1846.50 | 1850.58 | 1848.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1835.30 | 1847.52 | 1846.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 1835.30 | 1847.52 | 1846.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1847.80 | 1847.58 | 1847.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 1849.50 | 1846.79 | 1846.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 1839.70 | 1845.37 | 1846.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 1839.70 | 1845.37 | 1846.12 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 1851.10 | 1846.92 | 1846.72 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1841.00 | 1845.78 | 1846.33 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1857.00 | 1847.46 | 1846.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1858.00 | 1849.57 | 1847.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 1848.00 | 1850.50 | 1848.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 1848.00 | 1850.50 | 1848.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1848.00 | 1850.50 | 1848.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1848.00 | 1850.50 | 1848.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 1834.00 | 1847.20 | 1847.46 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 1870.00 | 1851.34 | 1848.90 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1836.10 | 1845.38 | 1846.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 15:15:00 | 1831.50 | 1840.93 | 1843.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 1737.10 | 1731.88 | 1755.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 1737.10 | 1731.88 | 1755.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1780.80 | 1741.66 | 1757.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1691.00 | 1741.66 | 1757.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 1728.50 | 1739.03 | 1754.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:45:00 | 1729.60 | 1736.62 | 1752.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1799.00 | 1763.44 | 1759.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1799.00 | 1763.44 | 1759.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1799.00 | 1763.44 | 1759.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 1799.00 | 1763.44 | 1759.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 1839.50 | 1790.37 | 1779.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1790.50 | 1795.94 | 1783.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:00:00 | 1790.50 | 1795.94 | 1783.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1793.40 | 1797.26 | 1788.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 1787.90 | 1797.26 | 1788.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1786.00 | 1795.01 | 1788.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 1786.00 | 1795.01 | 1788.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1799.00 | 1795.81 | 1789.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:30:00 | 1802.90 | 1793.84 | 1789.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:45:00 | 1801.10 | 1795.79 | 1791.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:00:00 | 1804.70 | 1797.57 | 1792.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:45:00 | 1801.00 | 1805.40 | 1798.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1841.60 | 1853.51 | 1836.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 1875.00 | 1851.97 | 1841.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 1882.10 | 1852.48 | 1846.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:45:00 | 1874.00 | 1862.30 | 1855.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1840.40 | 1854.21 | 1855.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 1831.20 | 1843.99 | 1848.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 1839.80 | 1836.14 | 1841.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 12:15:00 | 1839.80 | 1836.14 | 1841.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1839.80 | 1836.14 | 1841.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 1841.00 | 1836.14 | 1841.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1843.20 | 1837.55 | 1841.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 1843.20 | 1837.55 | 1841.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1840.80 | 1838.20 | 1841.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1840.80 | 1838.20 | 1841.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1855.00 | 1841.56 | 1842.82 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1850.00 | 1843.80 | 1843.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 1851.60 | 1847.14 | 1845.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 14:15:00 | 1844.30 | 1846.58 | 1845.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 14:15:00 | 1844.30 | 1846.58 | 1845.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1844.30 | 1846.58 | 1845.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1844.30 | 1846.58 | 1845.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1846.90 | 1846.64 | 1845.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1836.20 | 1846.64 | 1845.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1828.00 | 1842.91 | 1843.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 1827.00 | 1839.73 | 1842.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 1826.20 | 1820.51 | 1827.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 1826.20 | 1820.51 | 1827.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1826.20 | 1820.51 | 1827.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 1826.20 | 1820.51 | 1827.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1857.90 | 1827.99 | 1830.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 1857.90 | 1827.99 | 1830.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 1860.00 | 1834.39 | 1832.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 1860.40 | 1848.50 | 1841.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1847.90 | 1849.26 | 1842.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1836.80 | 1846.77 | 1842.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1836.80 | 1846.77 | 1842.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1836.80 | 1846.77 | 1842.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1836.10 | 1844.63 | 1841.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1833.80 | 1844.63 | 1841.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1845.80 | 1844.87 | 1842.12 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 1827.40 | 1838.75 | 1840.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 1810.00 | 1829.06 | 1834.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 1835.00 | 1801.64 | 1811.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 1835.00 | 1801.64 | 1811.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1835.00 | 1801.64 | 1811.94 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1849.30 | 1821.26 | 1817.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1860.70 | 1829.15 | 1821.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 1833.30 | 1835.66 | 1827.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 1833.30 | 1835.66 | 1827.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1850.00 | 1847.95 | 1840.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:15:00 | 1881.90 | 1848.47 | 1842.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 1837.50 | 1855.97 | 1851.69 | SL hit (close<static) qty=1.00 sl=1840.80 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 1831.40 | 1849.16 | 1849.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1806.70 | 1835.77 | 1842.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1779.10 | 1776.43 | 1798.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 1787.00 | 1776.43 | 1798.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1780.60 | 1773.41 | 1785.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 1780.60 | 1773.41 | 1785.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1779.00 | 1774.53 | 1784.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1779.00 | 1774.53 | 1784.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1779.00 | 1775.42 | 1784.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1773.00 | 1775.42 | 1784.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 1786.80 | 1776.66 | 1775.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 1786.80 | 1776.66 | 1775.82 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 1768.30 | 1774.99 | 1775.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1761.90 | 1772.37 | 1773.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1784.10 | 1770.85 | 1772.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1784.10 | 1770.85 | 1772.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1784.10 | 1770.85 | 1772.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1784.10 | 1770.85 | 1772.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1780.10 | 1772.70 | 1773.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 1787.40 | 1772.70 | 1773.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1807.30 | 1749.81 | 1752.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 1816.00 | 1749.81 | 1752.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1840.00 | 1767.85 | 1760.13 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1734.20 | 1768.00 | 1768.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1722.80 | 1747.56 | 1757.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1613.70 | 1607.56 | 1633.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 1613.70 | 1607.56 | 1633.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1606.70 | 1607.42 | 1620.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:30:00 | 1601.50 | 1613.09 | 1618.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1628.30 | 1617.54 | 1619.65 | SL hit (close>static) qty=1.00 sl=1622.90 alert=retest2 |

### Cycle 49 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 1631.00 | 1621.95 | 1621.40 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 1593.80 | 1618.79 | 1620.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1577.50 | 1608.15 | 1614.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 10:15:00 | 1562.30 | 1561.88 | 1584.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:45:00 | 1563.70 | 1561.88 | 1584.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1596.00 | 1569.49 | 1584.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 1596.00 | 1569.49 | 1584.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1602.80 | 1576.15 | 1585.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 1603.80 | 1576.15 | 1585.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 1602.90 | 1591.02 | 1591.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 1630.00 | 1602.10 | 1596.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 1652.00 | 1658.53 | 1643.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 1652.00 | 1658.53 | 1643.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1651.70 | 1664.17 | 1652.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1651.70 | 1664.17 | 1652.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1636.10 | 1658.56 | 1650.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1636.10 | 1658.56 | 1650.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1671.70 | 1661.18 | 1652.72 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 1622.30 | 1647.47 | 1649.63 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 1652.60 | 1647.17 | 1646.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 1684.10 | 1654.56 | 1650.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 10:15:00 | 1650.80 | 1662.45 | 1656.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 1650.80 | 1662.45 | 1656.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1650.80 | 1662.45 | 1656.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 1648.30 | 1662.45 | 1656.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1659.70 | 1661.90 | 1656.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 1653.50 | 1661.90 | 1656.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1656.80 | 1660.88 | 1656.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 1657.90 | 1660.88 | 1656.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1650.20 | 1658.74 | 1656.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 1650.20 | 1658.74 | 1656.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1658.20 | 1658.64 | 1656.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 1643.40 | 1658.64 | 1656.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 1637.00 | 1654.31 | 1654.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 09:15:00 | 1615.20 | 1646.49 | 1651.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 1635.30 | 1626.65 | 1637.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 1635.30 | 1626.65 | 1637.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1635.30 | 1626.65 | 1637.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1635.30 | 1626.65 | 1637.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 1613.10 | 1623.94 | 1635.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1612.20 | 1619.75 | 1627.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 1612.60 | 1606.30 | 1612.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:00:00 | 1613.00 | 1607.64 | 1612.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 1613.00 | 1609.85 | 1613.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1618.50 | 1611.58 | 1613.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1604.80 | 1612.87 | 1614.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1680.00 | 1619.40 | 1614.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1680.00 | 1619.40 | 1614.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1680.00 | 1619.40 | 1614.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1680.00 | 1619.40 | 1614.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1680.00 | 1619.40 | 1614.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 1680.00 | 1619.40 | 1614.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 1698.00 | 1635.12 | 1622.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 1696.00 | 1696.32 | 1678.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 1681.90 | 1696.32 | 1678.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1700.00 | 1697.06 | 1680.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1697.10 | 1697.06 | 1680.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1701.70 | 1708.56 | 1695.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:45:00 | 1740.10 | 1711.81 | 1703.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 1738.20 | 1716.10 | 1710.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 1748.90 | 1770.09 | 1770.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 1748.90 | 1770.09 | 1770.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1748.90 | 1770.09 | 1770.25 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 1799.10 | 1770.84 | 1770.10 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1769.00 | 1779.57 | 1780.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1765.30 | 1773.47 | 1777.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 14:15:00 | 1783.50 | 1765.55 | 1770.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 14:15:00 | 1783.50 | 1765.55 | 1770.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1783.50 | 1765.55 | 1770.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 1783.50 | 1765.55 | 1770.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1793.00 | 1771.04 | 1772.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1766.50 | 1771.04 | 1772.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1770.20 | 1770.42 | 1772.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:30:00 | 1763.00 | 1768.53 | 1771.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:00:00 | 1760.50 | 1768.53 | 1771.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:45:00 | 1758.60 | 1766.49 | 1769.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 1733.00 | 1765.99 | 1769.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1737.90 | 1730.89 | 1746.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1747.20 | 1730.89 | 1746.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1779.40 | 1735.75 | 1739.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1779.40 | 1735.75 | 1739.28 | SL hit (close>static) qty=1.00 sl=1778.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1779.40 | 1735.75 | 1739.28 | SL hit (close>static) qty=1.00 sl=1778.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1779.40 | 1735.75 | 1739.28 | SL hit (close>static) qty=1.00 sl=1778.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1779.40 | 1735.75 | 1739.28 | SL hit (close>static) qty=1.00 sl=1778.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 1778.40 | 1735.75 | 1739.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 1778.00 | 1744.20 | 1742.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 1800.00 | 1762.67 | 1752.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 15:15:00 | 1765.00 | 1776.85 | 1763.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 15:15:00 | 1765.00 | 1776.85 | 1763.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1765.00 | 1776.85 | 1763.71 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1724.00 | 1753.75 | 1757.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 1706.30 | 1744.26 | 1752.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1629.90 | 1618.48 | 1657.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1629.90 | 1618.48 | 1657.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1653.90 | 1625.57 | 1657.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:00:00 | 1621.20 | 1624.69 | 1653.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 14:15:00 | 1540.14 | 1585.32 | 1621.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 10:15:00 | 1459.08 | 1534.92 | 1588.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 1214.60 | 1189.85 | 1188.98 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1152.80 | 1188.44 | 1189.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 12:15:00 | 1126.80 | 1166.35 | 1178.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1173.00 | 1162.99 | 1174.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1173.00 | 1162.99 | 1174.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1173.00 | 1162.99 | 1174.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1173.00 | 1162.99 | 1174.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1210.00 | 1172.39 | 1177.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1203.70 | 1172.39 | 1177.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1187.20 | 1175.35 | 1178.53 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1204.10 | 1181.10 | 1180.86 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 1168.00 | 1182.70 | 1182.94 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 1198.00 | 1183.28 | 1182.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 1222.00 | 1191.02 | 1186.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1178.50 | 1191.11 | 1187.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1178.50 | 1191.11 | 1187.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1178.50 | 1191.11 | 1187.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1175.10 | 1191.11 | 1187.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1167.40 | 1186.37 | 1185.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1167.40 | 1186.37 | 1185.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 1179.30 | 1184.95 | 1184.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 1164.00 | 1180.47 | 1182.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1147.50 | 1134.89 | 1151.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1147.50 | 1134.89 | 1151.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1147.50 | 1134.89 | 1151.46 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1168.00 | 1157.42 | 1156.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 1191.80 | 1166.80 | 1161.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 1186.10 | 1187.63 | 1177.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:30:00 | 1183.10 | 1187.63 | 1177.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1188.30 | 1189.01 | 1181.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1188.80 | 1189.01 | 1181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1186.70 | 1188.55 | 1181.57 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 1159.70 | 1175.48 | 1176.64 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 1190.70 | 1178.52 | 1177.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 14:15:00 | 1206.80 | 1184.18 | 1180.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 11:15:00 | 1194.40 | 1200.12 | 1190.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 1194.40 | 1200.12 | 1190.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 1194.40 | 1200.12 | 1190.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:00:00 | 1241.10 | 1201.72 | 1195.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-18 10:15:00 | 1365.21 | 1297.50 | 1254.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 1267.20 | 1297.75 | 1298.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 1260.50 | 1290.30 | 1295.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 1282.50 | 1281.36 | 1288.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 1282.50 | 1281.36 | 1288.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1282.50 | 1281.36 | 1288.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1282.50 | 1281.36 | 1288.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1303.00 | 1285.69 | 1289.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 1300.10 | 1285.69 | 1289.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 1303.40 | 1289.23 | 1291.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 1303.00 | 1289.23 | 1291.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1324.00 | 1298.28 | 1295.08 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 14:15:00 | 1295.00 | 1300.30 | 1300.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 15:15:00 | 1292.50 | 1298.74 | 1299.78 | Break + close below crossover candle low |

### Cycle 73 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1327.70 | 1299.77 | 1298.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 1337.00 | 1307.21 | 1302.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1320.00 | 1325.17 | 1315.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1320.00 | 1325.17 | 1315.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1320.00 | 1325.17 | 1315.29 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1280.00 | 1305.73 | 1307.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1265.00 | 1294.19 | 1301.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 1293.10 | 1288.78 | 1296.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 12:15:00 | 1293.10 | 1288.78 | 1296.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1293.10 | 1288.78 | 1296.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:45:00 | 1291.00 | 1288.78 | 1296.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1294.50 | 1289.93 | 1296.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 1294.50 | 1289.93 | 1296.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1335.00 | 1298.94 | 1299.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 1337.20 | 1298.94 | 1299.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 15:15:00 | 1334.00 | 1305.95 | 1303.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 11:15:00 | 1345.10 | 1323.29 | 1312.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1438.90 | 1451.34 | 1408.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1510.00 | 1463.13 | 1429.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 1488.60 | 1476.48 | 1444.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 1455.50 | 1472.85 | 1448.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:45:00 | 1457.10 | 1472.85 | 1448.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1470.70 | 1472.30 | 1455.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 1434.00 | 1468.72 | 1463.34 | SL hit (close<ema400) qty=1.00 sl=1463.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 1434.00 | 1468.72 | 1463.34 | SL hit (close<ema400) qty=1.00 sl=1463.34 alert=retest1 |

### Cycle 76 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1432.30 | 1461.16 | 1462.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1431.00 | 1455.13 | 1459.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 1494.40 | 1447.36 | 1452.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 14:15:00 | 1494.40 | 1447.36 | 1452.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 1494.40 | 1447.36 | 1452.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 1494.40 | 1447.36 | 1452.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 1500.00 | 1457.89 | 1457.23 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 1432.70 | 1452.85 | 1455.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 1423.00 | 1446.88 | 1452.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 1449.10 | 1446.53 | 1450.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 12:15:00 | 1449.10 | 1446.53 | 1450.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1449.10 | 1446.53 | 1450.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 1449.10 | 1446.53 | 1450.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 1455.00 | 1448.22 | 1451.34 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 1500.00 | 1458.58 | 1455.77 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 10:15:00 | 1432.80 | 1452.21 | 1454.80 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 1475.00 | 1454.76 | 1454.14 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1446.30 | 1453.04 | 1453.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1444.00 | 1449.33 | 1451.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 14:15:00 | 1457.60 | 1450.99 | 1452.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 14:15:00 | 1457.60 | 1450.99 | 1452.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1457.60 | 1450.99 | 1452.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1457.60 | 1450.99 | 1452.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1445.00 | 1449.79 | 1451.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:15:00 | 1460.00 | 1449.79 | 1451.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1478.30 | 1455.49 | 1453.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 1529.90 | 1476.50 | 1464.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 1452.90 | 1484.28 | 1471.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 1452.90 | 1484.28 | 1471.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1452.90 | 1484.28 | 1471.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1452.90 | 1484.28 | 1471.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1429.00 | 1473.22 | 1468.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1403.40 | 1473.22 | 1468.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1381.30 | 1454.84 | 1460.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1365.10 | 1436.89 | 1451.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 14:15:00 | 1356.20 | 1292.10 | 1327.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 1356.20 | 1292.10 | 1327.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 1356.20 | 1292.10 | 1327.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 1356.20 | 1292.10 | 1327.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1413.00 | 1316.28 | 1334.85 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 11:15:00 | 1379.70 | 1347.10 | 1345.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 13:15:00 | 1390.00 | 1362.30 | 1353.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 14:15:00 | 1432.80 | 1452.12 | 1425.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-01 15:00:00 | 1432.80 | 1452.12 | 1425.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 1418.00 | 1445.29 | 1425.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 1391.20 | 1445.29 | 1425.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1387.10 | 1433.65 | 1421.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 1377.80 | 1433.65 | 1421.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1427.60 | 1423.21 | 1419.41 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 1393.00 | 1413.61 | 1415.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 1376.10 | 1406.11 | 1411.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 1407.70 | 1403.85 | 1409.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 11:15:00 | 1407.70 | 1403.85 | 1409.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1407.70 | 1403.85 | 1409.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:30:00 | 1406.50 | 1403.85 | 1409.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1419.00 | 1406.60 | 1409.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 1419.00 | 1406.60 | 1409.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 1410.70 | 1407.42 | 1409.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 1392.10 | 1406.88 | 1409.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 12:00:00 | 1389.50 | 1403.23 | 1407.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 15:15:00 | 1322.49 | 1358.01 | 1375.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 15:15:00 | 1320.02 | 1358.01 | 1375.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 1362.00 | 1358.81 | 1374.61 | SL hit (close>ema200) qty=0.50 sl=1358.81 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 1362.00 | 1358.81 | 1374.61 | SL hit (close>ema200) qty=0.50 sl=1358.81 alert=retest2 |

### Cycle 87 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 1422.60 | 1381.58 | 1379.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 1426.80 | 1390.62 | 1383.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 11:15:00 | 1358.60 | 1384.22 | 1381.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 11:15:00 | 1358.60 | 1384.22 | 1381.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1358.60 | 1384.22 | 1381.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 1358.60 | 1384.22 | 1381.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 1361.90 | 1379.75 | 1379.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:30:00 | 1374.70 | 1380.80 | 1380.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-13 13:15:00 | 1512.17 | 1464.14 | 1427.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 1760.00 | 1785.27 | 1785.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 1753.70 | 1774.88 | 1780.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1751.80 | 1743.10 | 1758.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 1735.20 | 1743.10 | 1758.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1720.50 | 1738.58 | 1754.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 1715.60 | 1734.07 | 1751.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 1717.60 | 1729.47 | 1747.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1772.80 | 1747.78 | 1746.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1772.80 | 1747.78 | 1746.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1772.80 | 1747.78 | 1746.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 1822.00 | 1769.88 | 1757.49 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 11:15:00 | 1602.20 | 2025-05-13 14:15:00 | 1653.70 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-05-12 12:00:00 | 1601.20 | 2025-05-13 14:15:00 | 1653.70 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-05-12 13:30:00 | 1602.00 | 2025-05-13 14:15:00 | 1653.70 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-05-12 14:30:00 | 1596.10 | 2025-05-13 14:15:00 | 1653.70 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-05-23 11:00:00 | 1694.40 | 2025-05-30 10:15:00 | 1863.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1709.40 | 2025-05-30 10:15:00 | 1880.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 12:00:00 | 1693.90 | 2025-05-30 10:15:00 | 1863.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-17 14:15:00 | 2101.60 | 2025-06-20 09:15:00 | 2085.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-25 12:00:00 | 1961.00 | 2025-06-26 15:15:00 | 1996.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2025-07-04 11:45:00 | 1990.00 | 2025-07-07 09:15:00 | 2035.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-07-07 09:15:00 | 1992.40 | 2025-07-07 09:15:00 | 2035.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1995.80 | 2025-07-14 15:15:00 | 2009.90 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-08 11:15:00 | 1999.80 | 2025-07-14 15:15:00 | 2009.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-07-10 10:30:00 | 1999.10 | 2025-07-14 15:15:00 | 2009.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-11 14:15:00 | 1991.00 | 2025-07-14 15:15:00 | 2009.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-07-14 10:00:00 | 1991.90 | 2025-07-14 15:15:00 | 2009.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-07-14 10:30:00 | 1996.00 | 2025-07-14 15:15:00 | 2009.90 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-07-21 14:45:00 | 1999.80 | 2025-07-22 10:15:00 | 1985.80 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-07-22 09:15:00 | 1998.80 | 2025-07-22 10:15:00 | 1985.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-23 12:30:00 | 1980.50 | 2025-07-23 13:15:00 | 1998.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1995.00 | 2025-07-25 11:15:00 | 1985.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-29 09:15:00 | 1959.40 | 2025-07-30 11:15:00 | 1991.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-07-30 09:45:00 | 1976.30 | 2025-07-30 11:15:00 | 1991.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-01 09:15:00 | 1943.50 | 2025-08-01 14:15:00 | 1846.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-01 09:15:00 | 1943.50 | 2025-08-04 12:15:00 | 1905.70 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2025-08-21 12:15:00 | 1828.00 | 2025-08-22 14:15:00 | 1861.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1818.00 | 2025-08-22 14:15:00 | 1861.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-08-29 09:15:00 | 1892.90 | 2025-09-02 11:15:00 | 1867.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-01 10:15:00 | 1872.00 | 2025-09-02 11:15:00 | 1867.80 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-09-01 10:45:00 | 1873.60 | 2025-09-02 11:15:00 | 1867.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-09-02 10:30:00 | 1871.80 | 2025-09-02 11:15:00 | 1867.80 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-09-03 12:45:00 | 1894.10 | 2025-09-04 12:15:00 | 1875.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-04 12:00:00 | 1886.20 | 2025-09-04 12:15:00 | 1875.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-09 11:30:00 | 1832.10 | 2025-09-11 14:15:00 | 1851.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-09 14:15:00 | 1833.90 | 2025-09-11 14:15:00 | 1851.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-09 15:00:00 | 1832.70 | 2025-09-11 14:15:00 | 1851.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-10 09:15:00 | 1827.20 | 2025-09-11 14:15:00 | 1851.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-10 13:00:00 | 1820.20 | 2025-09-11 14:15:00 | 1851.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-15 14:30:00 | 1849.50 | 2025-09-15 15:15:00 | 1839.70 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1691.00 | 2025-09-29 09:15:00 | 1799.00 | STOP_HIT | 1.00 | -6.39% |
| SELL | retest2 | 2025-09-26 10:00:00 | 1728.50 | 2025-09-29 09:15:00 | 1799.00 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-09-26 10:45:00 | 1729.60 | 2025-09-29 09:15:00 | 1799.00 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-10-03 09:30:00 | 1802.90 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2025-10-03 13:45:00 | 1801.10 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2025-10-03 15:00:00 | 1804.70 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2025-10-06 14:45:00 | 1801.00 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-10-08 15:15:00 | 1875.00 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-09 14:15:00 | 1882.10 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-10-10 12:45:00 | 1874.00 | 2025-10-13 12:15:00 | 1840.40 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-03 12:15:00 | 1881.90 | 2025-11-04 11:15:00 | 1837.50 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1773.00 | 2025-11-12 12:15:00 | 1786.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-26 14:30:00 | 1601.50 | 2025-11-27 09:15:00 | 1628.30 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1612.20 | 2025-12-18 15:15:00 | 1680.00 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-12-17 12:15:00 | 1612.60 | 2025-12-18 15:15:00 | 1680.00 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-12-17 13:00:00 | 1613.00 | 2025-12-18 15:15:00 | 1680.00 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-12-17 13:30:00 | 1613.00 | 2025-12-18 15:15:00 | 1680.00 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1604.80 | 2025-12-18 15:15:00 | 1680.00 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2025-12-26 14:45:00 | 1740.10 | 2026-01-02 11:15:00 | 1748.90 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-12-29 14:15:00 | 1738.20 | 2026-01-02 11:15:00 | 1748.90 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2026-01-09 12:30:00 | 1763.00 | 2026-01-13 14:15:00 | 1779.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-09 13:00:00 | 1760.50 | 2026-01-13 14:15:00 | 1779.40 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-09 14:45:00 | 1758.60 | 2026-01-13 14:15:00 | 1779.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1733.00 | 2026-01-13 14:15:00 | 1779.40 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-01-22 10:00:00 | 1621.20 | 2026-01-22 14:15:00 | 1540.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:00:00 | 1621.20 | 2026-01-23 10:15:00 | 1459.08 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-17 10:00:00 | 1241.10 | 2026-02-18 10:15:00 | 1365.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2026-03-09 14:30:00 | 1510.00 | 2026-03-12 09:15:00 | 1434.00 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest1 | 2026-03-10 10:30:00 | 1488.60 | 2026-03-12 09:15:00 | 1434.00 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1392.10 | 2026-04-08 15:15:00 | 1322.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-07 12:00:00 | 1389.50 | 2026-04-08 15:15:00 | 1320.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1392.10 | 2026-04-09 09:15:00 | 1362.00 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2026-04-07 12:00:00 | 1389.50 | 2026-04-09 09:15:00 | 1362.00 | STOP_HIT | 0.50 | 1.98% |
| BUY | retest2 | 2026-04-10 13:30:00 | 1374.70 | 2026-04-13 13:15:00 | 1512.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 10:45:00 | 1715.60 | 2026-05-07 09:15:00 | 1772.80 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-05-04 11:45:00 | 1717.60 | 2026-05-07 09:15:00 | 1772.80 | STOP_HIT | 1.00 | -3.21% |
