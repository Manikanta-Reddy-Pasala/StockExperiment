# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1874.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 19 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 17
- **Target hits / Stop hits / Partials:** 0 / 21 / 4
- **Avg / median % per leg:** -0.30% / -1.29%
- **Sum % (uncompounded):** -7.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.00% | -24.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.02% | -6.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.79% | -17.9% |
| SELL (all) | 13 | 8 | 61.5% | 0 | 9 | 4 | 1.26% | 16.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 0 | 9 | 4 | 1.26% | 16.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.02% | -6.0% |
| retest2 (combined) | 23 | 8 | 34.8% | 0 | 19 | 4 | -0.07% | -1.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 2234.00 | 2157.84 | 2157.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 12:15:00 | 2248.00 | 2158.73 | 2157.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 2339.00 | 2342.41 | 2281.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2352.50 | 2342.41 | 2281.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:45:00 | 2349.60 | 2342.42 | 2282.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2297.50 | 2339.15 | 2283.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2290.00 | 2339.15 | 2283.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2280.10 | 2338.07 | 2283.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 2280.10 | 2338.07 | 2283.84 | SL hit (close<ema400) qty=1.00 sl=2283.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 2280.10 | 2338.07 | 2283.84 | SL hit (close<ema400) qty=1.00 sl=2283.84 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 2299.80 | 2337.65 | 2283.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 2273.30 | 2335.92 | 2283.83 | SL hit (close<static) qty=1.00 sl=2280.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 2301.20 | 2333.65 | 2283.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 2277.10 | 2332.31 | 2283.78 | SL hit (close<static) qty=1.00 sl=2280.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:15:00 | 2295.20 | 2331.88 | 2283.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 2276.70 | 2329.31 | 2284.62 | SL hit (close<static) qty=1.00 sl=2280.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:30:00 | 2295.50 | 2327.42 | 2284.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2271.10 | 2326.22 | 2284.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 2271.10 | 2326.22 | 2284.79 | SL hit (close<static) qty=1.00 sl=2280.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 2285.00 | 2305.83 | 2279.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 2287.10 | 2302.99 | 2279.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 2245.00 | 2315.23 | 2291.18 | SL hit (close<static) qty=1.00 sl=2245.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 2245.00 | 2315.23 | 2291.18 | SL hit (close<static) qty=1.00 sl=2245.30 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 2101.40 | 2270.00 | 2270.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2083.00 | 2263.29 | 2267.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 2047.00 | 2032.86 | 2097.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 2047.00 | 2032.86 | 2097.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2079.00 | 2037.64 | 2094.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 2083.00 | 2037.64 | 2094.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 2094.50 | 2038.74 | 2094.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:30:00 | 2079.90 | 2039.13 | 2094.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 2085.80 | 2040.94 | 2094.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 2112.80 | 2042.13 | 2094.61 | SL hit (close>static) qty=1.00 sl=2106.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 2112.80 | 2042.13 | 2094.61 | SL hit (close>static) qty=1.00 sl=2106.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 2087.20 | 2055.00 | 2096.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 1982.84 | 2051.85 | 2093.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 2039.50 | 2035.23 | 2077.80 | SL hit (close>ema200) qty=0.50 sl=2035.23 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:15:00 | 2087.30 | 2037.95 | 2071.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 2080.50 | 2040.34 | 2072.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 2085.50 | 2040.34 | 2072.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 2098.40 | 2040.91 | 2072.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 2098.40 | 2040.91 | 2072.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2070.90 | 2043.53 | 2072.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 2060.70 | 2043.53 | 2072.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2129.80 | 2045.07 | 2072.71 | SL hit (close>static) qty=1.00 sl=2106.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2129.80 | 2045.07 | 2072.71 | SL hit (close>static) qty=1.00 sl=2079.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 2313.50 | 2097.01 | 2096.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2329.00 | 2118.83 | 2107.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 2190.50 | 2192.59 | 2152.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 15:00:00 | 2190.50 | 2192.59 | 2152.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 2144.50 | 2192.12 | 2152.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 2154.40 | 2192.12 | 2152.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 2183.80 | 2192.04 | 2152.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 2190.90 | 2185.86 | 2152.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:00:00 | 2188.30 | 2185.67 | 2153.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 2195.00 | 2187.97 | 2156.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 2200.50 | 2188.19 | 2157.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2156.30 | 2188.01 | 2158.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2156.30 | 2188.01 | 2158.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 2159.10 | 2187.72 | 2158.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 2154.80 | 2187.72 | 2158.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 2165.00 | 2187.50 | 2158.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 2073.00 | 2139.59 | 2139.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 2062.30 | 2138.82 | 2139.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 2063.00 | 2049.00 | 2080.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:00:00 | 2063.00 | 2049.00 | 2080.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2072.50 | 2050.03 | 2079.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 2076.40 | 2050.03 | 2079.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2098.10 | 2050.51 | 2079.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 2090.80 | 2050.51 | 2079.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2100.70 | 2051.01 | 2079.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 2107.80 | 2051.01 | 2079.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1849.10 | 1786.01 | 1867.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1846.30 | 1793.14 | 1867.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 1844.00 | 1793.64 | 1867.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 1841.40 | 1795.84 | 1864.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1753.98 | 1796.61 | 1860.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1751.80 | 1796.61 | 1860.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1749.33 | 1796.61 | 1860.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1809.00 | 1796.59 | 1860.17 | SL hit (close>ema200) qty=0.50 sl=1796.59 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1809.00 | 1796.59 | 1860.17 | SL hit (close>ema200) qty=0.50 sl=1796.59 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1809.00 | 1796.59 | 1860.17 | SL hit (close>ema200) qty=0.50 sl=1796.59 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1828.20 | 1725.14 | 1732.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1889.20 | 1734.07 | 1737.04 | SL hit (close>static) qty=1.00 sl=1873.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 1876.50 | 1740.27 | 1740.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 1918.90 | 1743.56 | 1741.76 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-01 09:15:00 | 2352.50 | 2025-07-02 15:15:00 | 2280.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2025-07-01 09:45:00 | 2349.60 | 2025-07-02 15:15:00 | 2280.10 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-07-03 09:45:00 | 2299.80 | 2025-07-03 12:15:00 | 2273.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-04 09:30:00 | 2301.20 | 2025-07-04 12:15:00 | 2277.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-04 14:15:00 | 2295.20 | 2025-07-07 15:15:00 | 2276.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-08 12:30:00 | 2295.50 | 2025-07-09 09:15:00 | 2271.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-15 11:15:00 | 2285.00 | 2025-07-25 10:15:00 | 2245.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-16 13:30:00 | 2287.10 | 2025-07-25 10:15:00 | 2245.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2079.90 | 2025-09-19 11:15:00 | 2112.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-19 09:45:00 | 2085.80 | 2025-09-19 11:15:00 | 2112.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-24 09:30:00 | 2087.20 | 2025-09-25 11:15:00 | 1982.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:30:00 | 2087.20 | 2025-10-03 09:15:00 | 2039.50 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-10-10 13:15:00 | 2087.30 | 2025-10-15 09:15:00 | 2129.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-10-14 11:15:00 | 2060.70 | 2025-10-15 09:15:00 | 2129.80 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-11-12 09:15:00 | 2190.90 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-11-12 13:00:00 | 2188.30 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-11-14 10:45:00 | 2195.00 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-11-14 15:00:00 | 2200.50 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1846.30 | 2026-02-24 14:15:00 | 1753.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:45:00 | 1844.00 | 2026-02-24 14:15:00 | 1751.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1841.40 | 2026-02-24 14:15:00 | 1749.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1846.30 | 2026-02-25 09:15:00 | 1809.00 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2026-02-19 12:45:00 | 1844.00 | 2026-02-25 09:15:00 | 1809.00 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1841.40 | 2026-02-25 09:15:00 | 1809.00 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1828.20 | 2026-05-04 09:15:00 | 1889.20 | STOP_HIT | 1.00 | -3.34% |
