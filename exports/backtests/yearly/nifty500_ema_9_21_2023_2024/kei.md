# KEI Industries Ltd. (KEI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5117.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 204 |
| ALERT1 | 146 |
| ALERT2 | 145 |
| ALERT2_SKIP | 100 |
| ALERT3 | 293 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 113 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 103 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 32 / 86
- **Target hits / Stop hits / Partials:** 8 / 103 / 7
- **Avg / median % per leg:** 0.03% / -0.84%
- **Sum % (uncompounded):** 3.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 11 | 19.6% | 8 | 48 | 0 | 0.10% | 5.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.29% | -1.3% |
| BUY @ 3rd Alert (retest2) | 55 | 11 | 20.0% | 8 | 47 | 0 | 0.12% | 6.8% |
| SELL (all) | 62 | 21 | 33.9% | 0 | 55 | 7 | -0.04% | -2.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 21 | 33.9% | 0 | 55 | 7 | -0.04% | -2.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.29% | -1.3% |
| retest2 (combined) | 117 | 32 | 27.4% | 8 | 102 | 7 | 0.04% | 4.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 13:15:00 | 2000.85 | 2014.65 | 2016.32 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 2041.05 | 2018.29 | 2017.38 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 2005.00 | 2015.75 | 2017.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 09:15:00 | 1973.55 | 2007.31 | 2013.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 09:15:00 | 1951.75 | 1943.51 | 1963.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 1951.75 | 1943.51 | 1963.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 1951.75 | 1943.51 | 1963.26 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 13:15:00 | 2016.55 | 1980.37 | 1975.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 15:15:00 | 2029.00 | 1996.49 | 1984.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 11:15:00 | 1992.55 | 1996.87 | 1987.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 11:15:00 | 1992.55 | 1996.87 | 1987.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 1992.55 | 1996.87 | 1987.71 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 11:15:00 | 1956.40 | 1990.30 | 1993.03 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 2009.75 | 1979.89 | 1979.78 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 1991.20 | 1998.73 | 1999.25 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 2057.45 | 2010.26 | 2004.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 14:15:00 | 2061.60 | 2041.54 | 2023.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 15:15:00 | 2086.95 | 2090.66 | 2064.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 14:15:00 | 2092.10 | 2108.56 | 2099.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 2092.10 | 2108.56 | 2099.00 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 11:15:00 | 2105.70 | 2128.18 | 2129.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 09:15:00 | 2070.75 | 2106.87 | 2117.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 13:15:00 | 2093.90 | 2084.67 | 2095.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 14:15:00 | 2098.20 | 2087.37 | 2095.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 2098.20 | 2087.37 | 2095.33 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 09:15:00 | 2169.65 | 2106.97 | 2103.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 10:15:00 | 2209.95 | 2127.57 | 2112.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 13:15:00 | 2254.80 | 2270.71 | 2235.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 09:15:00 | 2217.85 | 2252.58 | 2235.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 2217.85 | 2252.58 | 2235.06 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 13:15:00 | 2306.00 | 2331.67 | 2333.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 13:15:00 | 2291.55 | 2312.86 | 2322.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 14:15:00 | 2314.50 | 2313.18 | 2321.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 15:15:00 | 2310.70 | 2312.69 | 2320.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 15:15:00 | 2310.70 | 2312.69 | 2320.64 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 10:15:00 | 2347.55 | 2321.68 | 2319.53 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 11:15:00 | 2315.40 | 2328.05 | 2328.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 13:15:00 | 2302.00 | 2320.37 | 2324.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 14:15:00 | 2322.90 | 2320.88 | 2324.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 2326.85 | 2320.33 | 2323.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 2326.85 | 2320.33 | 2323.53 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 2331.90 | 2318.90 | 2318.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 2334.65 | 2322.05 | 2319.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 12:15:00 | 2442.95 | 2443.95 | 2421.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 11:15:00 | 2428.90 | 2443.02 | 2431.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 2428.90 | 2443.02 | 2431.49 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 09:15:00 | 2389.45 | 2422.39 | 2424.78 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 2562.35 | 2443.50 | 2429.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 10:15:00 | 2705.35 | 2495.87 | 2454.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 15:15:00 | 2570.00 | 2573.04 | 2540.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 15:15:00 | 2580.00 | 2598.70 | 2593.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 2580.00 | 2598.70 | 2593.52 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 09:15:00 | 2477.10 | 2574.38 | 2582.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 09:15:00 | 2290.00 | 2406.84 | 2443.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 09:15:00 | 2346.05 | 2314.37 | 2365.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 2346.05 | 2314.37 | 2365.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 2346.05 | 2314.37 | 2365.51 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 2410.45 | 2368.39 | 2364.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 2430.35 | 2380.78 | 2370.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 14:15:00 | 2375.05 | 2404.00 | 2394.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 14:15:00 | 2375.05 | 2404.00 | 2394.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 2375.05 | 2404.00 | 2394.29 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 2390.00 | 2409.33 | 2409.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 11:15:00 | 2378.90 | 2403.25 | 2406.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 2288.05 | 2287.97 | 2312.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 14:15:00 | 2342.10 | 2299.12 | 2313.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 2342.10 | 2299.12 | 2313.07 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 2406.00 | 2330.80 | 2325.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 10:15:00 | 2416.00 | 2347.84 | 2333.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 11:15:00 | 2607.00 | 2607.08 | 2583.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 14:15:00 | 2603.50 | 2610.89 | 2601.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 2603.50 | 2610.89 | 2601.16 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 2604.50 | 2659.68 | 2666.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 10:15:00 | 2598.45 | 2623.13 | 2635.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 2580.00 | 2536.92 | 2554.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 2580.00 | 2536.92 | 2554.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 2580.00 | 2536.92 | 2554.12 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 13:15:00 | 2580.60 | 2564.92 | 2563.71 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 2551.00 | 2566.41 | 2566.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 2515.20 | 2556.17 | 2561.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 2577.00 | 2557.91 | 2561.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 2577.00 | 2557.91 | 2561.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 2577.00 | 2557.91 | 2561.59 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 2585.90 | 2566.88 | 2565.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 2623.00 | 2576.69 | 2569.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 2610.30 | 2615.51 | 2601.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 12:15:00 | 2610.30 | 2615.51 | 2601.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 12:15:00 | 2610.30 | 2615.51 | 2601.42 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 12:15:00 | 2593.80 | 2610.18 | 2612.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 15:15:00 | 2588.00 | 2603.62 | 2608.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 2532.55 | 2445.09 | 2478.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 2532.55 | 2445.09 | 2478.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 2532.55 | 2445.09 | 2478.12 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 2588.35 | 2502.61 | 2493.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 11:15:00 | 2602.90 | 2537.56 | 2511.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 09:15:00 | 2626.45 | 2631.94 | 2595.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 2576.80 | 2630.22 | 2613.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 2576.80 | 2630.22 | 2613.70 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 2670.05 | 2683.39 | 2685.20 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 2689.55 | 2684.52 | 2684.51 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 10:15:00 | 2679.85 | 2683.96 | 2684.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 12:15:00 | 2663.75 | 2679.45 | 2682.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 2686.00 | 2675.81 | 2679.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 2686.00 | 2675.81 | 2679.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 2686.00 | 2675.81 | 2679.22 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 14:15:00 | 2700.95 | 2683.02 | 2681.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 09:15:00 | 2713.10 | 2691.75 | 2685.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 11:15:00 | 2712.90 | 2715.93 | 2703.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 12:15:00 | 2693.20 | 2711.38 | 2702.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 12:15:00 | 2693.20 | 2711.38 | 2702.94 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 10:15:00 | 2684.20 | 2695.97 | 2697.43 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 09:15:00 | 2721.55 | 2696.15 | 2695.92 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 2674.20 | 2700.15 | 2701.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 13:15:00 | 2667.25 | 2687.08 | 2694.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 2678.85 | 2677.77 | 2687.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 2678.85 | 2677.77 | 2687.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 2678.85 | 2677.77 | 2687.77 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 14:15:00 | 2471.00 | 2453.13 | 2451.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 09:15:00 | 2564.10 | 2477.78 | 2463.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 2567.35 | 2569.69 | 2544.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 10:15:00 | 2572.95 | 2579.79 | 2566.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 2572.95 | 2579.79 | 2566.15 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 10:15:00 | 2548.55 | 2561.70 | 2562.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 09:15:00 | 2538.65 | 2552.61 | 2557.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 11:15:00 | 2565.65 | 2552.19 | 2556.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 11:15:00 | 2565.65 | 2552.19 | 2556.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 2565.65 | 2552.19 | 2556.04 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 14:15:00 | 2569.45 | 2550.23 | 2549.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 15:15:00 | 2579.00 | 2555.99 | 2551.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 11:15:00 | 2563.10 | 2567.90 | 2559.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 12:15:00 | 2585.60 | 2571.44 | 2562.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 12:15:00 | 2585.60 | 2571.44 | 2562.06 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 2552.35 | 2562.45 | 2562.93 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 2573.30 | 2563.27 | 2563.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 10:15:00 | 2608.60 | 2572.33 | 2567.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 10:15:00 | 2708.10 | 2716.96 | 2677.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 12:15:00 | 2825.90 | 2869.34 | 2827.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 2825.90 | 2869.34 | 2827.38 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 13:15:00 | 2791.55 | 2814.64 | 2817.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 2764.05 | 2801.73 | 2809.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 12:15:00 | 2765.00 | 2763.69 | 2778.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 13:15:00 | 2782.90 | 2767.53 | 2779.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 2782.90 | 2767.53 | 2779.07 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 2808.00 | 2786.75 | 2785.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 2842.85 | 2802.95 | 2793.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 2875.95 | 2876.49 | 2849.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 09:15:00 | 2876.85 | 2873.93 | 2852.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 2876.85 | 2873.93 | 2852.49 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 2925.55 | 2955.07 | 2956.73 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 2980.00 | 2955.74 | 2954.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 12:15:00 | 2993.25 | 2974.54 | 2966.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 14:15:00 | 3042.00 | 3068.43 | 3030.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 15:15:00 | 3045.00 | 3063.74 | 3031.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 15:15:00 | 3045.00 | 3063.74 | 3031.57 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 13:15:00 | 3015.65 | 3034.16 | 3035.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 14:15:00 | 2987.00 | 3024.72 | 3030.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 3017.45 | 3013.42 | 3023.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 3017.45 | 3013.42 | 3023.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 3017.45 | 3013.42 | 3023.38 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 3050.15 | 2987.72 | 2987.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 3117.15 | 3065.79 | 3041.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 14:15:00 | 3248.00 | 3251.60 | 3206.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 3210.30 | 3242.48 | 3210.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 3210.30 | 3242.48 | 3210.12 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 3160.10 | 3197.70 | 3199.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 3127.95 | 3183.75 | 3193.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 14:15:00 | 3190.10 | 3171.62 | 3182.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 14:15:00 | 3190.10 | 3171.62 | 3182.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 3190.10 | 3171.62 | 3182.97 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 10:15:00 | 3240.00 | 3196.88 | 3192.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 3340.00 | 3244.68 | 3219.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 3341.10 | 3359.88 | 3327.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 3341.10 | 3359.88 | 3327.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 3341.10 | 3359.88 | 3327.91 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 10:15:00 | 3275.10 | 3317.22 | 3320.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 12:15:00 | 3248.70 | 3296.76 | 3309.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 3013.55 | 3003.87 | 3049.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 14:15:00 | 3012.15 | 3010.96 | 3036.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 3012.15 | 3010.96 | 3036.45 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 15:15:00 | 3050.60 | 3044.27 | 3043.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-17 09:15:00 | 3112.50 | 3057.91 | 3049.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 3105.10 | 3107.88 | 3083.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 3105.10 | 3107.88 | 3083.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 3105.10 | 3107.88 | 3083.54 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 3051.40 | 3100.76 | 3106.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 12:15:00 | 3047.75 | 3090.15 | 3101.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 3153.95 | 3053.83 | 3064.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 3153.95 | 3053.83 | 3064.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 3153.95 | 3053.83 | 3064.78 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 10:15:00 | 3252.10 | 3093.49 | 3081.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 3286.25 | 3176.46 | 3126.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 09:15:00 | 3204.95 | 3220.59 | 3162.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 15:15:00 | 3182.00 | 3196.42 | 3173.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 3182.00 | 3196.42 | 3173.81 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 10:15:00 | 3126.60 | 3163.35 | 3165.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 13:15:00 | 3103.00 | 3141.31 | 3154.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 3139.20 | 3118.39 | 3138.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 3139.20 | 3118.39 | 3138.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 3139.20 | 3118.39 | 3138.46 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 3193.50 | 3154.20 | 3150.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 14:15:00 | 3204.00 | 3164.16 | 3155.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 3151.50 | 3164.15 | 3157.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 3151.50 | 3164.15 | 3157.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 3151.50 | 3164.15 | 3157.01 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 3237.70 | 3269.20 | 3270.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 3230.00 | 3256.69 | 3264.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 12:15:00 | 3264.50 | 3239.22 | 3250.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 12:15:00 | 3264.50 | 3239.22 | 3250.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 3264.50 | 3239.22 | 3250.85 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 3309.05 | 3259.95 | 3258.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 15:15:00 | 3327.20 | 3314.34 | 3300.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 3275.30 | 3314.40 | 3305.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 12:15:00 | 3275.30 | 3314.40 | 3305.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 3275.30 | 3314.40 | 3305.90 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 15:15:00 | 3280.00 | 3298.84 | 3300.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 09:15:00 | 3211.45 | 3281.36 | 3292.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 3275.60 | 3270.43 | 3284.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 12:15:00 | 3279.00 | 3272.15 | 3284.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 3279.00 | 3272.15 | 3284.21 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 3237.70 | 3190.99 | 3188.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 13:15:00 | 3280.65 | 3228.55 | 3208.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 3240.00 | 3253.46 | 3232.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 09:15:00 | 3217.90 | 3252.03 | 3239.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 3217.90 | 3252.03 | 3239.15 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 14:15:00 | 3196.00 | 3231.53 | 3233.59 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 09:15:00 | 3278.25 | 3230.83 | 3228.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 10:15:00 | 3336.10 | 3251.88 | 3237.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 15:15:00 | 3372.00 | 3372.49 | 3335.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 3284.90 | 3354.97 | 3330.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 3284.90 | 3354.97 | 3330.80 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 3273.60 | 3309.45 | 3313.63 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 14:15:00 | 3364.90 | 3320.00 | 3317.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 14:15:00 | 3401.55 | 3357.01 | 3338.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 13:15:00 | 3379.95 | 3390.22 | 3367.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 14:15:00 | 3361.10 | 3384.39 | 3366.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 3361.10 | 3384.39 | 3366.65 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 3306.45 | 3355.12 | 3356.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 3305.35 | 3345.17 | 3351.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 3101.90 | 3076.14 | 3161.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 15:15:00 | 3090.00 | 3088.06 | 3141.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 3090.00 | 3088.06 | 3141.40 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 3211.70 | 3165.48 | 3159.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 14:15:00 | 3300.00 | 3223.21 | 3196.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 3414.90 | 3416.77 | 3382.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 3413.70 | 3416.99 | 3388.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 3413.70 | 3416.99 | 3388.37 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 14:15:00 | 3473.00 | 3506.63 | 3508.66 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 13:15:00 | 3559.80 | 3515.70 | 3510.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 14:15:00 | 3597.45 | 3532.05 | 3518.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 14:15:00 | 3567.55 | 3589.71 | 3562.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 15:15:00 | 3585.00 | 3588.77 | 3564.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 15:15:00 | 3585.00 | 3588.77 | 3564.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 3899.85 | 3774.95 | 3733.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 15:15:00 | 3843.45 | 3886.58 | 3888.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 15:15:00 | 3843.45 | 3886.58 | 3888.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 3777.25 | 3854.19 | 3871.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 10:15:00 | 3817.05 | 3794.19 | 3820.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 10:15:00 | 3817.05 | 3794.19 | 3820.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 3817.05 | 3794.19 | 3820.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 11:00:00 | 3817.05 | 3794.19 | 3820.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 11:15:00 | 3815.10 | 3798.37 | 3819.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 11:45:00 | 3849.10 | 3798.37 | 3819.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 3816.45 | 3801.99 | 3819.35 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 3910.70 | 3840.78 | 3832.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 3947.00 | 3862.02 | 3842.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 3958.60 | 3977.87 | 3927.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 13:00:00 | 3958.60 | 3977.87 | 3927.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 3941.10 | 3966.86 | 3930.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:30:00 | 3944.65 | 3966.86 | 3930.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 3942.55 | 3961.63 | 3934.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:30:00 | 3934.15 | 3961.63 | 3934.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 3949.70 | 3959.25 | 3936.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:30:00 | 3935.05 | 3959.25 | 3936.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 3945.00 | 3954.91 | 3938.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:30:00 | 3938.60 | 3954.91 | 3938.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 3936.80 | 3951.29 | 3937.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:00:00 | 3936.80 | 3951.29 | 3937.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 3914.95 | 3944.02 | 3935.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:45:00 | 3918.50 | 3944.02 | 3935.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 3934.00 | 3942.02 | 3935.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:30:00 | 3906.10 | 3934.00 | 3932.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 10:15:00 | 3878.50 | 3922.90 | 3927.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 13:15:00 | 3863.55 | 3899.26 | 3914.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 14:15:00 | 3902.65 | 3899.94 | 3913.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 3902.65 | 3899.94 | 3913.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 3907.00 | 3901.35 | 3912.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:15:00 | 3947.00 | 3901.35 | 3912.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 3909.85 | 3903.05 | 3912.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 15:00:00 | 3859.25 | 3894.92 | 3905.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 09:15:00 | 3968.50 | 3908.04 | 3909.49 | SL hit (close>static) qty=1.00 sl=3964.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 10:15:00 | 3989.05 | 3924.24 | 3916.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 11:15:00 | 4008.55 | 3941.10 | 3925.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 14:15:00 | 4002.00 | 4003.45 | 3978.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 14:15:00 | 4002.00 | 4003.45 | 3978.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 4002.00 | 4003.45 | 3978.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 15:00:00 | 4002.00 | 4003.45 | 3978.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 4000.00 | 4002.76 | 3980.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 4023.00 | 4002.76 | 3980.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 3963.40 | 3999.99 | 3984.90 | SL hit (close<static) qty=1.00 sl=3976.25 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 3919.00 | 3970.28 | 3973.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 3803.00 | 3915.25 | 3945.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 10:15:00 | 3846.70 | 3834.36 | 3875.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 11:15:00 | 3853.85 | 3834.36 | 3875.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 3777.80 | 3792.84 | 3827.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 3777.80 | 3792.84 | 3827.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 3826.25 | 3789.71 | 3813.82 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-05-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 09:15:00 | 3849.35 | 3822.30 | 3821.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 13:15:00 | 3944.30 | 3855.21 | 3837.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 3840.00 | 3871.41 | 3851.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 3840.00 | 3871.41 | 3851.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 3840.00 | 3871.41 | 3851.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:00:00 | 3840.00 | 3871.41 | 3851.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 3861.00 | 3869.33 | 3852.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 11:30:00 | 3874.60 | 3862.18 | 3850.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 12:30:00 | 3868.30 | 3865.12 | 3852.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-16 14:15:00 | 4262.06 | 4169.36 | 4087.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 4135.95 | 4145.58 | 4145.90 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 4157.90 | 4147.79 | 4146.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 4247.40 | 4184.14 | 4167.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 4214.95 | 4261.09 | 4234.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 4214.95 | 4261.09 | 4234.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 4214.95 | 4261.09 | 4234.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 4214.95 | 4261.09 | 4234.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 4193.00 | 4247.48 | 4230.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 4193.00 | 4247.48 | 4230.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 4152.05 | 4217.83 | 4219.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 4138.15 | 4183.97 | 4197.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 4178.90 | 4177.68 | 4192.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 4178.90 | 4177.68 | 4192.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 4178.90 | 4177.68 | 4192.17 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 4213.20 | 4194.46 | 4194.30 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 4184.70 | 4192.51 | 4193.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 12:15:00 | 4155.00 | 4185.01 | 4189.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 4179.75 | 4119.83 | 4136.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 4179.75 | 4119.83 | 4136.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 4179.75 | 4119.83 | 4136.88 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 4212.15 | 4148.26 | 4147.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 4215.80 | 4171.31 | 4158.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4167.65 | 4194.05 | 4175.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 4167.65 | 4194.05 | 4175.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 4167.65 | 4194.05 | 4175.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 4167.65 | 4194.05 | 4175.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 3935.10 | 4142.26 | 4153.55 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 4160.00 | 4107.10 | 4103.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 4253.45 | 4154.89 | 4130.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 15:15:00 | 4780.00 | 4781.73 | 4673.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 09:15:00 | 4938.00 | 4781.73 | 4673.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 4796.80 | 4853.74 | 4770.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:30:00 | 4728.00 | 4853.74 | 4770.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 4764.15 | 4831.86 | 4774.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 4785.20 | 4831.86 | 4774.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 4760.00 | 4817.49 | 4773.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:15:00 | 4753.80 | 4817.49 | 4773.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 4696.00 | 4746.33 | 4749.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 4662.95 | 4719.84 | 4736.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 4420.00 | 4411.19 | 4512.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:30:00 | 4444.55 | 4411.19 | 4512.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 4565.90 | 4443.86 | 4509.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 4518.75 | 4443.86 | 4509.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 4559.60 | 4467.01 | 4514.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 4545.00 | 4467.01 | 4514.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 4551.25 | 4509.44 | 4524.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:45:00 | 4550.00 | 4509.44 | 4524.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 4596.25 | 4539.77 | 4535.66 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 4478.05 | 4534.95 | 4536.40 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 4550.00 | 4537.48 | 4536.67 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 4524.25 | 4534.57 | 4535.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 4493.15 | 4526.28 | 4531.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 4499.00 | 4467.30 | 4491.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 4499.00 | 4467.30 | 4491.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 4499.00 | 4467.30 | 4491.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:45:00 | 4457.45 | 4468.93 | 4490.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:15:00 | 4470.25 | 4470.03 | 4488.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:15:00 | 4471.85 | 4471.71 | 4487.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:45:00 | 4447.05 | 4464.39 | 4479.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 4483.20 | 4456.51 | 4471.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 4476.80 | 4456.51 | 4471.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 4410.65 | 4447.34 | 4465.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 15:15:00 | 4471.00 | 4469.39 | 4469.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 4471.00 | 4469.39 | 4469.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 4532.00 | 4481.92 | 4474.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 11:15:00 | 4533.55 | 4550.52 | 4524.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 11:15:00 | 4533.55 | 4550.52 | 4524.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 4533.55 | 4550.52 | 4524.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:45:00 | 4559.60 | 4550.52 | 4524.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 4510.55 | 4542.52 | 4523.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:45:00 | 4518.80 | 4542.52 | 4523.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 4492.65 | 4532.55 | 4520.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 4482.00 | 4532.55 | 4520.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 4681.85 | 4641.75 | 4594.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:45:00 | 4740.30 | 4659.98 | 4606.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 15:15:00 | 4604.00 | 4630.11 | 4632.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 15:15:00 | 4604.00 | 4630.11 | 4632.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 4536.80 | 4608.76 | 4622.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 4502.95 | 4495.37 | 4526.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 4502.95 | 4495.37 | 4526.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 4587.50 | 4515.98 | 4530.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 4587.50 | 4515.98 | 4530.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 4584.80 | 4529.74 | 4535.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 4590.80 | 4529.74 | 4535.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 4596.10 | 4543.01 | 4540.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 4639.00 | 4582.98 | 4563.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 4592.30 | 4611.64 | 4591.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 4592.30 | 4611.64 | 4591.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 4592.30 | 4611.64 | 4591.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 4584.25 | 4611.64 | 4591.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 4583.40 | 4605.99 | 4590.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:15:00 | 4572.60 | 4605.99 | 4590.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 4543.95 | 4593.58 | 4586.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 4543.95 | 4593.58 | 4586.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 12:15:00 | 4523.35 | 4579.54 | 4580.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 4509.55 | 4565.54 | 4574.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 4119.15 | 4083.07 | 4170.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 15:00:00 | 4119.15 | 4083.07 | 4170.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 4159.80 | 4102.58 | 4164.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 4041.95 | 4124.71 | 4164.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 4093.30 | 4124.13 | 4157.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 4274.85 | 4170.41 | 4171.89 | SL hit (close>static) qty=1.00 sl=4199.85 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 4262.85 | 4188.90 | 4180.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 4301.05 | 4236.24 | 4215.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 4257.85 | 4303.27 | 4267.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 4257.85 | 4303.27 | 4267.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 4257.85 | 4303.27 | 4267.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:45:00 | 4259.55 | 4303.27 | 4267.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 4275.90 | 4297.79 | 4268.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 13:45:00 | 4285.40 | 4288.43 | 4270.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 4308.25 | 4324.54 | 4324.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 4308.25 | 4324.54 | 4324.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 4268.05 | 4308.64 | 4317.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 4060.60 | 4038.08 | 4095.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 4069.80 | 4038.08 | 4095.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 4113.35 | 4053.14 | 4097.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 4113.35 | 4053.14 | 4097.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 4125.00 | 4067.51 | 4099.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 4126.40 | 4067.51 | 4099.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 4125.70 | 4112.20 | 4112.13 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 4094.35 | 4109.40 | 4110.92 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 14:15:00 | 4138.00 | 4115.12 | 4113.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 15:15:00 | 4140.00 | 4120.10 | 4115.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 4221.20 | 4307.89 | 4280.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 4221.20 | 4307.89 | 4280.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 4221.20 | 4307.89 | 4280.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 4205.60 | 4307.89 | 4280.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 4237.00 | 4293.71 | 4276.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:30:00 | 4252.45 | 4285.57 | 4274.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-20 10:15:00 | 4677.70 | 4448.03 | 4377.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 4576.40 | 4626.02 | 4631.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 4545.50 | 4609.92 | 4623.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 15:15:00 | 4554.00 | 4548.62 | 4567.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 09:15:00 | 4557.95 | 4548.62 | 4567.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 4548.85 | 4548.66 | 4566.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:45:00 | 4557.50 | 4548.66 | 4566.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 4561.30 | 4544.65 | 4556.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 4561.30 | 4544.65 | 4556.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 4575.00 | 4550.72 | 4558.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 4530.40 | 4550.72 | 4558.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:30:00 | 4557.95 | 4547.79 | 4555.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 4619.15 | 4559.47 | 4558.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 4619.15 | 4559.47 | 4558.18 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 4540.65 | 4557.38 | 4559.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 14:15:00 | 4509.95 | 4544.81 | 4552.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 4552.00 | 4502.31 | 4519.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 4552.00 | 4502.31 | 4519.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 4552.00 | 4502.31 | 4519.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:45:00 | 4568.60 | 4502.31 | 4519.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 4524.05 | 4506.66 | 4520.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:00:00 | 4489.50 | 4503.23 | 4517.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 4488.15 | 4487.64 | 4503.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 4430.55 | 4405.34 | 4404.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 4430.55 | 4405.34 | 4404.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 4454.95 | 4439.82 | 4426.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 4419.20 | 4437.72 | 4429.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 14:15:00 | 4419.20 | 4437.72 | 4429.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 4419.20 | 4437.72 | 4429.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:30:00 | 4419.10 | 4437.72 | 4429.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 4430.00 | 4436.17 | 4429.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 4416.60 | 4436.17 | 4429.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 4440.45 | 4437.03 | 4430.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 4421.40 | 4437.03 | 4430.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 4436.60 | 4436.94 | 4430.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 4437.95 | 4436.94 | 4430.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 4422.45 | 4434.04 | 4430.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 4422.45 | 4434.04 | 4430.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 4429.80 | 4433.20 | 4430.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 15:15:00 | 4440.00 | 4428.69 | 4428.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 12:00:00 | 4437.80 | 4432.81 | 4430.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 13:15:00 | 4441.55 | 4431.25 | 4430.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 4408.10 | 4453.74 | 4444.66 | SL hit (close<static) qty=1.00 sl=4412.25 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 4409.60 | 4437.75 | 4438.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 4400.00 | 4430.20 | 4435.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 12:15:00 | 4363.70 | 4355.60 | 4390.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 13:00:00 | 4363.70 | 4355.60 | 4390.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 4288.00 | 4318.26 | 4359.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:00:00 | 4270.00 | 4308.61 | 4351.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:00:00 | 4257.55 | 4266.03 | 4281.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 14:15:00 | 4299.85 | 4235.17 | 4228.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 4299.85 | 4235.17 | 4228.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 13:15:00 | 4304.00 | 4264.12 | 4247.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 4278.25 | 4282.34 | 4262.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 4278.25 | 4282.34 | 4262.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 4278.25 | 4282.34 | 4262.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:30:00 | 4262.05 | 4282.34 | 4262.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 4268.60 | 4276.83 | 4264.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 4264.50 | 4276.83 | 4264.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 4257.75 | 4273.02 | 4264.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 4257.75 | 4273.02 | 4264.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 4270.00 | 4272.41 | 4264.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 4187.75 | 4272.41 | 4264.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 4241.10 | 4266.15 | 4262.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 4203.05 | 4266.15 | 4262.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 4222.60 | 4257.44 | 4259.04 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 13:15:00 | 4305.35 | 4263.56 | 4261.15 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 4188.35 | 4248.72 | 4255.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 4099.25 | 4210.47 | 4236.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 4192.85 | 4160.03 | 4196.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 4192.85 | 4160.03 | 4196.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 4111.45 | 4150.32 | 4188.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 4142.00 | 4150.32 | 4188.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 4193.00 | 4159.94 | 4183.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 4193.00 | 4159.94 | 4183.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 4190.00 | 4165.95 | 4184.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 4274.90 | 4165.95 | 4184.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 4322.80 | 4197.32 | 4196.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 4345.00 | 4226.86 | 4210.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 4349.90 | 4573.60 | 4559.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 4349.90 | 4573.60 | 4559.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 4349.90 | 4573.60 | 4559.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 4349.90 | 4573.60 | 4559.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 4284.45 | 4515.77 | 4534.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 4166.00 | 4445.81 | 4501.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4100.85 | 4093.39 | 4176.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 11:00:00 | 4100.85 | 4093.39 | 4176.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 4119.00 | 4045.01 | 4078.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 4119.00 | 4045.01 | 4078.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 4017.75 | 4039.56 | 4073.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 12:15:00 | 4011.10 | 4039.56 | 4073.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 4003.90 | 4042.11 | 4063.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 3810.54 | 3933.78 | 3990.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 3803.70 | 3933.78 | 3990.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 3894.95 | 3882.01 | 3930.61 | SL hit (close>ema200) qty=0.50 sl=3882.01 alert=retest2 |

### Cycle 104 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 3981.20 | 3861.57 | 3851.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 13:15:00 | 4004.00 | 3939.84 | 3896.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 3964.00 | 3999.56 | 3949.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 3964.00 | 3999.56 | 3949.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 3964.00 | 3999.56 | 3949.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 4021.85 | 4003.45 | 3959.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 4031.95 | 4023.66 | 3980.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 3915.00 | 3995.81 | 3978.59 | SL hit (close<static) qty=1.00 sl=3932.60 alert=retest2 |

### Cycle 105 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 3895.00 | 3959.60 | 3964.13 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 4070.55 | 3978.05 | 3966.94 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 3966.30 | 3996.21 | 3997.16 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 4000.00 | 3998.06 | 3997.87 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 3986.80 | 3996.45 | 3997.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 3970.00 | 3991.16 | 3994.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 4017.05 | 3996.34 | 3996.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 4017.05 | 3996.34 | 3996.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 4017.05 | 3996.34 | 3996.77 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 4019.55 | 4000.98 | 3998.84 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 3982.00 | 3997.81 | 3998.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 3871.80 | 3970.98 | 3985.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 3777.60 | 3757.95 | 3807.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 10:15:00 | 3789.90 | 3757.95 | 3807.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 3780.30 | 3762.42 | 3805.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:45:00 | 3744.60 | 3796.51 | 3808.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 10:15:00 | 3822.60 | 3801.73 | 3809.40 | SL hit (close>static) qty=1.00 sl=3807.95 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 3886.30 | 3815.41 | 3811.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 10:15:00 | 3920.75 | 3836.48 | 3821.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 3997.25 | 3997.88 | 3951.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 11:15:00 | 3955.50 | 3987.34 | 3954.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 3955.50 | 3987.34 | 3954.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:00:00 | 3955.50 | 3987.34 | 3954.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 4002.90 | 3990.45 | 3959.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 13:15:00 | 4019.90 | 3990.45 | 3959.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 11:15:00 | 4421.89 | 4328.65 | 4283.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 4491.45 | 4525.26 | 4527.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 12:15:00 | 4474.80 | 4515.17 | 4522.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 4490.00 | 4487.35 | 4504.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 4490.00 | 4487.35 | 4504.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 4490.00 | 4487.35 | 4504.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:30:00 | 4437.45 | 4464.79 | 4483.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 4215.58 | 4312.97 | 4369.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 4295.55 | 4275.00 | 4330.19 | SL hit (close>ema200) qty=0.50 sl=4275.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 4306.20 | 4234.51 | 4226.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 4318.10 | 4262.03 | 4241.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 15:15:00 | 4345.00 | 4348.52 | 4322.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:15:00 | 4375.00 | 4348.52 | 4322.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 4318.75 | 4342.57 | 4321.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 4318.75 | 4342.57 | 4321.91 | SL hit (close<ema400) qty=1.00 sl=4321.91 alert=retest1 |

### Cycle 115 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 4365.65 | 4422.15 | 4428.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 4349.75 | 4407.67 | 4420.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 4287.80 | 4287.08 | 4327.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 13:15:00 | 4318.85 | 4297.07 | 4325.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 4318.85 | 4297.07 | 4325.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 4326.20 | 4297.07 | 4325.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 4320.05 | 4304.21 | 4323.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 4264.90 | 4304.21 | 4323.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 4051.65 | 4175.56 | 4216.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 4025.10 | 4007.16 | 4070.83 | SL hit (close>ema200) qty=0.50 sl=4007.16 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 4123.00 | 4075.97 | 4075.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 4174.45 | 4125.94 | 4107.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 4067.05 | 4125.76 | 4114.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 4067.05 | 4125.76 | 4114.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 4067.05 | 4125.76 | 4114.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 4067.05 | 4125.76 | 4114.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 4065.05 | 4113.62 | 4110.21 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 4085.20 | 4107.93 | 4107.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 4042.10 | 4085.44 | 4096.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 13:15:00 | 4125.70 | 4047.35 | 4066.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 13:15:00 | 4125.70 | 4047.35 | 4066.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 4125.70 | 4047.35 | 4066.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:00:00 | 4125.70 | 4047.35 | 4066.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 4122.65 | 4062.41 | 4071.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 4122.65 | 4062.41 | 4071.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 4414.15 | 4140.28 | 4105.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 4511.55 | 4214.53 | 4142.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 10:15:00 | 4403.80 | 4405.78 | 4300.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:00:00 | 4403.80 | 4405.78 | 4300.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 4297.25 | 4378.38 | 4314.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 4297.25 | 4378.38 | 4314.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 4255.85 | 4353.87 | 4308.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 4249.35 | 4353.87 | 4308.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 4072.90 | 4280.34 | 4282.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 4034.70 | 4231.21 | 4259.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 4012.85 | 3965.77 | 4041.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 3989.00 | 3965.77 | 4041.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 4016.10 | 3978.82 | 4019.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:45:00 | 4021.75 | 3978.82 | 4019.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 4020.00 | 3987.05 | 4019.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 4031.20 | 3987.05 | 4019.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 4021.55 | 3993.95 | 4019.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 4046.60 | 3993.95 | 4019.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 4001.85 | 3995.53 | 4017.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 3961.50 | 3991.05 | 4011.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 4026.75 | 3986.24 | 4000.87 | SL hit (close>static) qty=1.00 sl=4026.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 4082.75 | 4021.53 | 4015.36 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 3990.00 | 4014.43 | 4016.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 3900.70 | 3991.68 | 4006.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 13:15:00 | 3887.35 | 3877.39 | 3912.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 14:00:00 | 3887.35 | 3877.39 | 3912.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 3882.55 | 3877.17 | 3903.27 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 14:15:00 | 3918.00 | 3907.68 | 3907.34 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 3874.00 | 3904.04 | 3907.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 3831.35 | 3889.50 | 3900.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 3411.45 | 3393.80 | 3458.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 14:30:00 | 3424.15 | 3393.80 | 3458.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 3405.40 | 3399.87 | 3445.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 3432.85 | 3399.87 | 3445.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 3436.85 | 3412.73 | 3440.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 3452.95 | 3412.73 | 3440.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 3453.15 | 3420.82 | 3441.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 3453.15 | 3420.82 | 3441.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 3444.70 | 3425.59 | 3441.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 3433.45 | 3425.59 | 3441.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 3430.50 | 3426.57 | 3440.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 3406.70 | 3426.57 | 3440.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 3397.85 | 3420.83 | 3436.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:00:00 | 3404.00 | 3403.27 | 3421.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 10:15:00 | 3468.95 | 3426.30 | 3428.42 | SL hit (close>static) qty=1.00 sl=3468.75 alert=retest2 |

### Cycle 124 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 3529.75 | 3446.99 | 3437.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 3548.80 | 3479.64 | 3454.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 3752.80 | 3802.74 | 3719.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 10:00:00 | 3752.80 | 3802.74 | 3719.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 3738.45 | 3789.89 | 3721.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 3727.55 | 3789.89 | 3721.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 3749.90 | 3773.80 | 3739.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 3780.70 | 3773.80 | 3739.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 3099.95 | 3650.11 | 3705.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 3099.95 | 3650.11 | 3705.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 3020.50 | 3362.87 | 3546.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 3110.00 | 3070.16 | 3229.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 3064.90 | 3070.16 | 3229.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 3036.95 | 3063.52 | 3211.54 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 3271.25 | 3174.87 | 3170.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 3311.70 | 3255.08 | 3217.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 3230.60 | 3265.33 | 3244.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 3230.60 | 3265.33 | 3244.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 3230.60 | 3265.33 | 3244.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 3231.95 | 3265.33 | 3244.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 3232.05 | 3258.67 | 3243.03 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 3222.75 | 3234.84 | 3234.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 15:15:00 | 3219.50 | 3231.77 | 3233.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 3049.75 | 3034.56 | 3084.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 3063.80 | 3034.56 | 3084.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 3065.85 | 3040.82 | 3082.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 3077.35 | 3040.82 | 3082.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 3062.90 | 3046.89 | 3072.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 3057.70 | 3046.89 | 3072.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 3069.85 | 3051.48 | 3072.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 3069.85 | 3051.48 | 3072.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 3060.75 | 3055.18 | 3070.45 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 3113.00 | 3072.44 | 3071.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 3132.20 | 3099.74 | 3085.25 | Break + close above crossover candle high |

### Cycle 129 — SELL (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 09:15:00 | 2834.20 | 3187.66 | 3189.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-20 10:15:00 | 2790.75 | 3108.28 | 3153.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 2897.00 | 2893.63 | 2955.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 2908.95 | 2899.92 | 2929.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 2908.95 | 2899.92 | 2929.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:45:00 | 2924.30 | 2899.92 | 2929.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 2863.50 | 2892.64 | 2923.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 2860.80 | 2892.64 | 2923.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 2857.75 | 2859.88 | 2887.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:45:00 | 2851.50 | 2857.31 | 2884.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 2910.20 | 2878.70 | 2875.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 2910.20 | 2878.70 | 2875.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 11:15:00 | 2938.20 | 2897.88 | 2884.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 2892.00 | 2899.44 | 2888.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 13:15:00 | 2892.00 | 2899.44 | 2888.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 2892.00 | 2899.44 | 2888.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 2892.00 | 2899.44 | 2888.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 2892.25 | 2898.00 | 2888.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:15:00 | 2877.30 | 2898.00 | 2888.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 2877.30 | 2893.86 | 2887.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 2884.90 | 2893.86 | 2887.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2862.00 | 2887.49 | 2885.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:15:00 | 2841.10 | 2887.49 | 2885.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 2841.90 | 2878.37 | 2881.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 2819.45 | 2854.66 | 2868.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 2861.70 | 2831.43 | 2847.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 2861.70 | 2831.43 | 2847.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 2861.70 | 2831.43 | 2847.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 2861.70 | 2831.43 | 2847.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 2873.85 | 2839.92 | 2849.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 2864.20 | 2839.92 | 2849.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2677.65 | 2803.43 | 2828.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 12:00:00 | 2667.00 | 2760.39 | 2803.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2533.65 | 2653.00 | 2729.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 2575.00 | 2543.30 | 2621.52 | SL hit (close>ema200) qty=0.50 sl=2543.30 alert=retest2 |

### Cycle 132 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 2648.10 | 2613.33 | 2610.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 2694.60 | 2642.70 | 2625.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 2939.00 | 2939.01 | 2893.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 2939.00 | 2939.01 | 2893.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2933.40 | 2973.34 | 2957.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2933.40 | 2973.34 | 2957.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2928.80 | 2964.43 | 2954.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 2917.00 | 2964.43 | 2954.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 2960.00 | 2965.51 | 2958.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 2980.00 | 2965.51 | 2958.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2972.30 | 2966.87 | 2960.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 12:00:00 | 3008.70 | 2978.28 | 2966.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-07 09:15:00 | 3309.57 | 3202.19 | 3161.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 3186.00 | 3246.93 | 3254.18 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3359.90 | 3257.11 | 3253.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 3430.00 | 3366.78 | 3320.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 3410.00 | 3410.07 | 3370.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 3410.00 | 3410.07 | 3370.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 3380.50 | 3399.85 | 3375.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 3369.60 | 3399.85 | 3375.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 3386.00 | 3397.08 | 3376.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 13:45:00 | 3370.80 | 3397.08 | 3376.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 3401.90 | 3398.04 | 3378.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 3429.70 | 3398.43 | 3380.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 3437.10 | 3453.85 | 3454.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 3437.10 | 3453.85 | 3454.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 3431.70 | 3449.42 | 3452.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 3476.00 | 3453.23 | 3453.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 3476.00 | 3453.23 | 3453.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 3476.00 | 3453.23 | 3453.33 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 3468.90 | 3455.29 | 3454.18 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 3436.60 | 3450.54 | 3452.16 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 3465.00 | 3454.61 | 3453.80 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 3435.00 | 3450.69 | 3452.09 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 3483.10 | 3453.25 | 3450.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 3497.80 | 3467.40 | 3457.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 3627.60 | 3637.67 | 3602.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 3627.60 | 3637.67 | 3602.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 3584.00 | 3623.90 | 3601.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 3584.00 | 3623.90 | 3601.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 3570.80 | 3613.28 | 3599.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 3570.80 | 3613.28 | 3599.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 3606.00 | 3608.64 | 3600.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 3616.40 | 3608.64 | 3600.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 3614.60 | 3609.83 | 3601.41 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 3594.00 | 3597.71 | 3597.83 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 3606.40 | 3599.45 | 3598.61 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3577.00 | 3594.96 | 3596.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 3570.20 | 3590.01 | 3594.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 3586.10 | 3563.06 | 3576.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 3586.10 | 3563.06 | 3576.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3586.10 | 3563.06 | 3576.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 3586.10 | 3563.06 | 3576.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 3589.00 | 3568.25 | 3577.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 3612.40 | 3568.25 | 3577.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 3613.70 | 3586.79 | 3584.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 3628.70 | 3595.17 | 3588.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 3798.60 | 3805.23 | 3769.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:00:00 | 3798.60 | 3805.23 | 3769.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 3781.70 | 3803.38 | 3786.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 3782.10 | 3803.38 | 3786.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 3788.70 | 3800.45 | 3786.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 3809.80 | 3800.45 | 3786.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 3747.50 | 3791.35 | 3784.91 | SL hit (close<static) qty=1.00 sl=3774.10 alert=retest2 |

### Cycle 145 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 3734.00 | 3779.88 | 3780.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 3679.40 | 3748.24 | 3764.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 3683.00 | 3663.29 | 3687.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 3683.00 | 3663.29 | 3687.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3691.70 | 3668.97 | 3687.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 3696.00 | 3668.97 | 3687.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 3680.40 | 3671.26 | 3686.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 3678.50 | 3671.26 | 3686.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3679.30 | 3672.87 | 3686.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 3691.00 | 3672.87 | 3686.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 3657.50 | 3669.79 | 3683.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 3650.00 | 3669.79 | 3683.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 3678.00 | 3605.78 | 3605.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 3678.00 | 3605.78 | 3605.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 3707.70 | 3626.17 | 3614.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 3803.20 | 3807.69 | 3751.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 3803.20 | 3807.69 | 3751.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 3770.80 | 3796.86 | 3777.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 3770.80 | 3796.86 | 3777.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 3759.70 | 3789.43 | 3776.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:45:00 | 3754.90 | 3789.43 | 3776.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 3765.20 | 3782.80 | 3775.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 3784.00 | 3782.80 | 3775.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 3790.00 | 3794.93 | 3786.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 3786.00 | 3796.37 | 3791.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 3786.70 | 3794.43 | 3790.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 3803.20 | 3795.37 | 3791.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 3859.30 | 3795.37 | 3791.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:15:00 | 3817.40 | 3802.18 | 3795.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 3812.60 | 3804.26 | 3797.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 3811.70 | 3805.73 | 3799.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 3746.00 | 3794.55 | 3795.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 3746.00 | 3794.55 | 3795.62 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 3842.30 | 3798.79 | 3793.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 3857.10 | 3810.45 | 3799.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 3827.00 | 3833.04 | 3816.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 3827.00 | 3833.04 | 3816.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 3827.00 | 3833.04 | 3816.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 3827.00 | 3833.04 | 3816.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 3833.00 | 3833.03 | 3818.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 3818.10 | 3833.03 | 3818.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 3822.00 | 3830.83 | 3818.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 3822.00 | 3830.83 | 3818.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 3768.00 | 3818.26 | 3814.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 3768.00 | 3818.26 | 3814.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 3755.00 | 3805.61 | 3808.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 3729.20 | 3770.47 | 3785.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 3735.80 | 3659.68 | 3682.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 3735.80 | 3659.68 | 3682.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3735.80 | 3659.68 | 3682.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 3748.00 | 3659.68 | 3682.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 3719.80 | 3671.70 | 3686.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 3706.70 | 3671.70 | 3686.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 3720.60 | 3695.51 | 3694.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 3720.60 | 3695.51 | 3694.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 3731.50 | 3702.71 | 3698.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 3950.00 | 3954.95 | 3913.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:00:00 | 3950.00 | 3954.95 | 3913.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 3918.50 | 3942.86 | 3915.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 3904.00 | 3942.86 | 3915.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 3909.60 | 3936.21 | 3914.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 3931.90 | 3936.21 | 3914.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 3933.30 | 3971.57 | 3962.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:30:00 | 3939.30 | 3958.25 | 3957.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:30:00 | 3937.50 | 3962.65 | 3961.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 3911.30 | 3952.38 | 3956.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 3911.30 | 3952.38 | 3956.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 3901.30 | 3923.37 | 3939.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 3890.00 | 3881.46 | 3905.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 15:00:00 | 3890.00 | 3881.46 | 3905.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 3938.60 | 3891.01 | 3905.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 3926.70 | 3891.01 | 3905.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 3917.50 | 3896.31 | 3906.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 3894.20 | 3897.05 | 3906.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 3903.00 | 3906.58 | 3907.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 3902.00 | 3905.66 | 3906.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 3926.40 | 3906.43 | 3906.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 3926.40 | 3906.43 | 3906.03 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 3876.30 | 3904.63 | 3905.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3810.00 | 3879.76 | 3891.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 13:15:00 | 3874.60 | 3864.46 | 3878.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 13:15:00 | 3874.60 | 3864.46 | 3878.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 3874.60 | 3864.46 | 3878.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:45:00 | 3866.70 | 3864.46 | 3878.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3850.50 | 3861.66 | 3876.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 3826.30 | 3851.17 | 3866.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 3828.30 | 3840.91 | 3857.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 3883.80 | 3832.70 | 3840.80 | SL hit (close>static) qty=1.00 sl=3880.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 3863.00 | 3847.14 | 3846.48 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 3823.40 | 3845.83 | 3846.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 3818.00 | 3837.49 | 3842.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 13:15:00 | 3791.90 | 3790.31 | 3812.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:45:00 | 3800.00 | 3790.31 | 3812.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 3808.30 | 3793.91 | 3811.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 3808.30 | 3793.91 | 3811.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 3800.00 | 3795.13 | 3810.88 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 3854.80 | 3821.36 | 3817.50 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 3807.50 | 3826.43 | 3826.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 3731.60 | 3800.10 | 3813.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 3808.30 | 3779.30 | 3791.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 3808.30 | 3779.30 | 3791.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 3808.30 | 3779.30 | 3791.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 3812.70 | 3779.30 | 3791.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 3820.40 | 3787.52 | 3794.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 3820.40 | 3787.52 | 3794.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 3820.60 | 3796.13 | 3797.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 3820.60 | 3796.13 | 3797.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 3820.90 | 3801.09 | 3799.47 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 3773.40 | 3795.20 | 3797.14 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 3812.80 | 3800.80 | 3799.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 3817.50 | 3805.64 | 3801.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 3980.00 | 3980.68 | 3941.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 3952.00 | 3980.68 | 3941.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 3964.40 | 3977.43 | 3943.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 4008.00 | 3959.55 | 3947.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:45:00 | 4001.50 | 3973.93 | 3956.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:30:00 | 4001.00 | 3980.27 | 3960.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 3935.50 | 3969.49 | 3966.16 | SL hit (close<static) qty=1.00 sl=3939.70 alert=retest2 |

### Cycle 161 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 3918.30 | 3954.84 | 3959.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 3860.60 | 3914.62 | 3937.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 3849.40 | 3830.30 | 3858.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 3849.40 | 3830.30 | 3858.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3849.40 | 3830.30 | 3858.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 3861.40 | 3830.30 | 3858.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 3866.00 | 3837.44 | 3858.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 3866.00 | 3837.44 | 3858.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 3868.40 | 3843.63 | 3859.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 3868.40 | 3843.63 | 3859.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3893.00 | 3853.51 | 3862.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 3893.00 | 3853.51 | 3862.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 3911.70 | 3865.15 | 3867.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 3911.70 | 3865.15 | 3867.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 3915.90 | 3875.30 | 3871.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 3943.80 | 3903.36 | 3886.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 4075.40 | 4087.99 | 4041.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 11:00:00 | 4075.40 | 4087.99 | 4041.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 4043.50 | 4079.09 | 4041.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 4042.10 | 4079.09 | 4041.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 4030.60 | 4069.40 | 4040.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 4030.60 | 4069.40 | 4040.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 4050.00 | 4065.52 | 4041.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 4068.90 | 4054.43 | 4040.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 4064.00 | 4055.63 | 4043.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 4062.90 | 4057.17 | 4047.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 4030.00 | 4044.55 | 4044.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 4030.00 | 4044.55 | 4044.73 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 4070.30 | 4046.25 | 4044.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 4096.00 | 4066.58 | 4055.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 4079.70 | 4085.87 | 4071.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 4079.70 | 4085.87 | 4071.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 4090.00 | 4084.62 | 4073.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 4096.00 | 4084.62 | 4073.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 4107.00 | 4091.85 | 4079.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 4106.10 | 4094.86 | 4082.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 4036.20 | 4091.39 | 4093.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 4036.20 | 4091.39 | 4093.89 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 4111.20 | 4095.12 | 4094.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 4186.00 | 4113.29 | 4102.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 4138.00 | 4156.84 | 4136.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 4138.00 | 4156.84 | 4136.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 4138.00 | 4156.84 | 4136.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 4138.00 | 4156.84 | 4136.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 4144.80 | 4154.43 | 4136.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 4135.40 | 4154.43 | 4136.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 4138.10 | 4151.16 | 4136.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 4161.50 | 4151.16 | 4136.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 4145.10 | 4149.95 | 4137.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:15:00 | 4147.60 | 4149.95 | 4137.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 4152.90 | 4150.54 | 4139.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 4178.50 | 4150.62 | 4143.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 4164.20 | 4151.54 | 4144.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 4183.00 | 4188.80 | 4168.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 4131.00 | 4171.24 | 4167.74 | SL hit (close<static) qty=1.00 sl=4135.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 4120.00 | 4160.99 | 4163.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 4102.10 | 4149.21 | 4157.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 14:15:00 | 4195.70 | 4134.90 | 4143.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 4195.70 | 4134.90 | 4143.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 4195.70 | 4134.90 | 4143.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 4195.70 | 4134.90 | 4143.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 4172.00 | 4142.32 | 4146.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 4087.40 | 4142.32 | 4146.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 4117.80 | 4140.95 | 4145.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 4122.70 | 4115.60 | 4127.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 4163.90 | 4133.99 | 4133.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 12:15:00 | 4163.90 | 4133.99 | 4133.61 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 4104.30 | 4135.54 | 4135.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 4088.50 | 4126.13 | 4131.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 4069.00 | 4054.25 | 4081.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 4069.00 | 4054.25 | 4081.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 4048.40 | 4053.08 | 4078.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 4005.00 | 4052.44 | 4076.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 4025.10 | 4034.87 | 4050.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:15:00 | 4020.30 | 4032.16 | 4043.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 4087.60 | 4039.84 | 4037.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 4087.60 | 4039.84 | 4037.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 4181.50 | 4116.54 | 4082.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 4250.50 | 4255.96 | 4223.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:45:00 | 4253.10 | 4255.96 | 4223.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4250.60 | 4268.20 | 4250.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 4268.50 | 4264.33 | 4251.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 4269.50 | 4264.33 | 4251.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 4283.10 | 4308.35 | 4287.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 4208.60 | 4355.49 | 4337.88 | SL hit (close<static) qty=1.00 sl=4227.30 alert=retest2 |

### Cycle 171 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 4195.90 | 4323.57 | 4324.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 11:15:00 | 4181.10 | 4295.08 | 4311.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 4197.40 | 4157.56 | 4195.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 4197.40 | 4157.56 | 4195.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 4197.40 | 4157.56 | 4195.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 4197.40 | 4157.56 | 4195.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 4155.50 | 4157.15 | 4191.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 4140.00 | 4156.20 | 4188.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:00:00 | 4144.10 | 4154.31 | 4181.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:00:00 | 4141.00 | 4150.64 | 4175.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:30:00 | 4135.60 | 4142.75 | 4167.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 4133.00 | 4112.83 | 4136.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 4144.00 | 4112.83 | 4136.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 4135.30 | 4117.32 | 4136.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 4135.90 | 4117.32 | 4136.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 4112.00 | 4116.26 | 4134.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 4092.00 | 4114.72 | 4125.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 4095.50 | 4109.26 | 4121.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 12:15:00 | 4116.50 | 4088.59 | 4088.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 4116.50 | 4088.59 | 4088.10 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 4055.50 | 4086.41 | 4088.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 4035.20 | 4069.16 | 4078.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 4021.00 | 4011.70 | 4033.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:45:00 | 4028.00 | 4011.70 | 4033.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 4004.00 | 4011.85 | 4030.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 4021.60 | 4011.85 | 4030.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 3935.40 | 3924.59 | 3952.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 3946.00 | 3924.59 | 3952.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 3934.60 | 3926.59 | 3951.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 3952.00 | 3926.59 | 3951.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 3960.00 | 3933.27 | 3951.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 3970.30 | 3933.27 | 3951.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 3981.40 | 3942.90 | 3954.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 3991.90 | 3942.90 | 3954.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 4003.60 | 3955.04 | 3959.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 4003.60 | 3955.04 | 3959.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 3995.50 | 3963.13 | 3962.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 4026.00 | 3981.81 | 3971.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 4067.30 | 4067.56 | 4042.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 15:00:00 | 4067.30 | 4067.56 | 4042.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 4103.70 | 4101.62 | 4081.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 4077.20 | 4101.62 | 4081.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4113.00 | 4135.66 | 4119.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 4114.20 | 4135.66 | 4119.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 4100.00 | 4128.53 | 4117.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 4093.90 | 4128.53 | 4117.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 4120.00 | 4125.49 | 4118.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 4110.70 | 4125.49 | 4118.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 4142.10 | 4128.81 | 4120.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 4147.50 | 4127.50 | 4123.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 4148.50 | 4131.70 | 4126.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 4103.50 | 4141.56 | 4134.87 | SL hit (close<static) qty=1.00 sl=4120.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 4058.60 | 4124.97 | 4127.94 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 4115.80 | 4101.85 | 4101.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 4132.80 | 4108.04 | 4104.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 4102.60 | 4109.19 | 4105.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 4102.60 | 4109.19 | 4105.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 4102.60 | 4109.19 | 4105.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 4103.00 | 4109.19 | 4105.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 4125.40 | 4112.43 | 4107.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 4102.20 | 4112.43 | 4107.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 4175.10 | 4135.59 | 4121.40 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 4109.60 | 4126.34 | 4128.08 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 4163.70 | 4129.32 | 4127.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 4183.00 | 4140.06 | 4132.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 15:15:00 | 4147.00 | 4156.83 | 4147.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 15:15:00 | 4147.00 | 4156.83 | 4147.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 4147.00 | 4156.83 | 4147.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 4168.90 | 4156.83 | 4147.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 4175.10 | 4160.48 | 4150.46 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 4117.30 | 4149.99 | 4152.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 4110.00 | 4136.79 | 4145.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 4070.70 | 4067.14 | 4092.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 4070.70 | 4067.14 | 4092.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 4081.20 | 4070.87 | 4090.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 4042.80 | 4064.26 | 4085.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:45:00 | 4035.00 | 4010.27 | 4034.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:00:00 | 4061.00 | 4036.49 | 4040.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 4078.10 | 4044.81 | 4044.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 4078.10 | 4044.81 | 4044.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 4081.80 | 4063.51 | 4054.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 4143.90 | 4144.72 | 4117.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:30:00 | 4145.30 | 4144.72 | 4117.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 4124.00 | 4144.24 | 4126.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 4067.90 | 4144.24 | 4126.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 4087.00 | 4132.79 | 4122.61 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 4097.00 | 4114.76 | 4115.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 4050.30 | 4099.09 | 4108.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 4099.00 | 4087.73 | 4096.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 15:15:00 | 4099.00 | 4087.73 | 4096.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 4099.00 | 4087.73 | 4096.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 4154.40 | 4087.73 | 4096.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4150.00 | 4100.19 | 4101.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 4175.50 | 4100.19 | 4101.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 4160.20 | 4112.19 | 4106.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 4182.40 | 4134.20 | 4118.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 4407.50 | 4409.17 | 4347.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 4408.90 | 4409.17 | 4347.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 4381.00 | 4407.12 | 4383.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 4381.00 | 4407.12 | 4383.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 4362.00 | 4398.10 | 4381.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 4362.00 | 4398.10 | 4381.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 4385.00 | 4395.48 | 4381.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 4382.20 | 4395.48 | 4381.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 4383.90 | 4393.16 | 4381.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 4378.00 | 4393.16 | 4381.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 4386.00 | 4391.73 | 4382.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 4399.00 | 4381.49 | 4379.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 4393.50 | 4395.05 | 4387.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 4400.00 | 4395.70 | 4389.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 4399.10 | 4397.77 | 4391.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 4388.30 | 4397.96 | 4392.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 4388.30 | 4397.96 | 4392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 4364.80 | 4391.33 | 4390.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 4364.80 | 4391.33 | 4390.25 | SL hit (close<static) qty=1.00 sl=4376.40 alert=retest2 |

### Cycle 183 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 4367.20 | 4386.50 | 4388.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 4344.70 | 4378.14 | 4384.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4418.50 | 4383.63 | 4385.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4418.50 | 4383.63 | 4385.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4418.50 | 4383.63 | 4385.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 4413.00 | 4383.63 | 4385.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 4429.00 | 4392.70 | 4389.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 4462.70 | 4412.90 | 4399.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 15:15:00 | 4510.00 | 4520.14 | 4491.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 4514.20 | 4518.95 | 4493.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 4514.20 | 4518.95 | 4493.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 4514.20 | 4518.95 | 4493.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 4521.70 | 4535.13 | 4523.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 4524.30 | 4535.13 | 4523.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 4515.10 | 4531.12 | 4522.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 4572.00 | 4531.12 | 4522.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 4569.50 | 4538.80 | 4526.70 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 4470.70 | 4521.48 | 4525.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 4441.10 | 4492.49 | 4510.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 4346.00 | 4345.91 | 4387.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 4346.00 | 4345.91 | 4387.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 4275.00 | 4335.72 | 4373.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 4256.70 | 4320.66 | 4362.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 4273.50 | 4310.27 | 4354.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 4268.70 | 4310.27 | 4354.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 4263.60 | 4298.21 | 4344.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 4330.00 | 4298.59 | 4332.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 4313.90 | 4298.59 | 4332.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 4280.40 | 4294.95 | 4327.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 4385.00 | 4340.59 | 4337.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 4385.00 | 4340.59 | 4337.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 4405.40 | 4353.55 | 4343.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 4351.00 | 4385.36 | 4368.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 4351.00 | 4385.36 | 4368.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 4351.00 | 4385.36 | 4368.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 4331.40 | 4383.29 | 4369.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 4359.50 | 4378.53 | 4368.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 4364.50 | 4378.53 | 4368.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 4320.60 | 4366.95 | 4364.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 4320.60 | 4366.95 | 4364.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 4332.30 | 4360.02 | 4361.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 4299.80 | 4335.54 | 4348.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 3928.40 | 3887.64 | 3967.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 09:30:00 | 3933.60 | 3887.64 | 3967.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3847.20 | 3824.08 | 3860.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 3842.00 | 3824.08 | 3860.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 3867.90 | 3832.84 | 3861.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 3867.90 | 3832.84 | 3861.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 3857.00 | 3837.68 | 3860.89 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 3895.90 | 3872.27 | 3869.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 3917.40 | 3881.30 | 3874.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 3981.30 | 3983.92 | 3944.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 3981.30 | 3983.92 | 3944.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 4000.50 | 3996.81 | 3964.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:45:00 | 4044.70 | 4000.23 | 3980.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 4449.17 | 4081.08 | 4021.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 4541.30 | 4565.44 | 4566.71 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 4590.20 | 4570.40 | 4568.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 4605.00 | 4577.32 | 4572.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 14:15:00 | 4563.20 | 4576.23 | 4572.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 14:15:00 | 4563.20 | 4576.23 | 4572.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 4563.20 | 4576.23 | 4572.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 4563.20 | 4576.23 | 4572.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 4568.00 | 4574.59 | 4572.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 4562.00 | 4574.59 | 4572.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 4556.00 | 4570.87 | 4570.75 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 4521.70 | 4561.04 | 4566.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 4491.30 | 4547.09 | 4559.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 13:15:00 | 4543.30 | 4501.11 | 4519.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 4543.30 | 4501.11 | 4519.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 4543.30 | 4501.11 | 4519.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 4543.30 | 4501.11 | 4519.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 4568.90 | 4514.67 | 4524.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 4568.90 | 4514.67 | 4524.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 4581.50 | 4536.41 | 4533.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 4671.60 | 4610.41 | 4592.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 4704.40 | 4716.36 | 4661.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:00:00 | 4704.40 | 4716.36 | 4661.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5221.00 | 5083.47 | 5005.90 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 4990.00 | 5051.13 | 5056.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 4938.00 | 5014.72 | 5038.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 4963.00 | 4950.57 | 4988.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 4963.00 | 4950.57 | 4988.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 4963.00 | 4950.57 | 4988.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:15:00 | 5000.00 | 4950.57 | 4988.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 4958.50 | 4952.15 | 4985.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:30:00 | 4955.50 | 4955.98 | 4981.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 4955.00 | 4955.98 | 4981.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 4707.72 | 4902.81 | 4948.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 4707.25 | 4902.81 | 4948.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 4793.00 | 4784.37 | 4862.22 | SL hit (close>ema200) qty=0.50 sl=4784.37 alert=retest2 |

### Cycle 194 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 4305.00 | 4221.67 | 4219.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 4391.00 | 4267.27 | 4241.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4252.00 | 4314.44 | 4277.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4252.00 | 4314.44 | 4277.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4252.00 | 4314.44 | 4277.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 4252.00 | 4314.44 | 4277.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 4242.00 | 4299.95 | 4274.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 4242.00 | 4299.95 | 4274.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 4218.00 | 4271.62 | 4266.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 4218.00 | 4271.62 | 4266.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 4217.50 | 4260.80 | 4262.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 4000.50 | 4173.37 | 4214.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 4057.00 | 4031.64 | 4104.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 4057.00 | 4031.64 | 4104.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4057.00 | 4031.64 | 4104.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:45:00 | 4028.00 | 4028.01 | 4095.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 4125.00 | 4063.10 | 4096.25 | SL hit (close>static) qty=1.00 sl=4123.50 alert=retest2 |

### Cycle 196 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 4244.00 | 4113.11 | 4112.15 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4089.00 | 4139.87 | 4146.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4015.50 | 4115.00 | 4134.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4198.20 | 4076.86 | 4097.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4198.20 | 4076.86 | 4097.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4198.20 | 4076.86 | 4097.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 4192.30 | 4076.86 | 4097.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 4188.00 | 4114.44 | 4111.84 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 3990.90 | 4101.42 | 4109.62 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 4121.50 | 4088.88 | 4086.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 4172.00 | 4116.43 | 4101.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 4435.70 | 4441.82 | 4333.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 4435.70 | 4441.82 | 4333.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4379.00 | 4437.83 | 4410.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4442.70 | 4431.46 | 4409.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 4436.80 | 4438.02 | 4416.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 09:15:00 | 4886.97 | 4793.15 | 4706.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 4781.70 | 4852.99 | 4861.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 4775.00 | 4837.39 | 4853.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 4830.00 | 4811.97 | 4833.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 15:15:00 | 4830.00 | 4811.97 | 4833.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 4830.00 | 4811.97 | 4833.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 4920.00 | 4811.97 | 4833.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 4942.00 | 4837.98 | 4843.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 5000.00 | 4837.98 | 4843.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 4991.60 | 4868.70 | 4856.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 4997.00 | 4939.26 | 4912.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 4927.60 | 4946.83 | 4928.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 4927.60 | 4946.83 | 4928.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 4927.60 | 4946.83 | 4928.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 4927.60 | 4946.83 | 4928.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 4932.80 | 4944.03 | 4928.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:15:00 | 4959.00 | 4944.03 | 4928.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 4959.00 | 4947.02 | 4931.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 4864.00 | 4947.02 | 4931.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 4818.90 | 4921.40 | 4921.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 4818.90 | 4921.40 | 4921.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 4785.00 | 4894.12 | 4908.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 12:15:00 | 4757.00 | 4847.76 | 4883.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 4863.00 | 4848.77 | 4877.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 4863.00 | 4848.77 | 4877.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 4863.00 | 4848.77 | 4877.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 4893.20 | 4848.77 | 4877.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4892.80 | 4861.77 | 4879.05 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 4980.60 | 4906.86 | 4897.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 5040.00 | 4933.49 | 4910.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 4862.70 | 4951.54 | 4927.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 4862.70 | 4951.54 | 4927.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 4862.70 | 4951.54 | 4927.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:00:00 | 5000.00 | 4958.06 | 4934.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:45:00 | 5016.60 | 4973.55 | 4943.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 5022.90 | 5105.16 | 5056.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 3899.85 | 2024-04-16 15:15:00 | 3843.45 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-04-29 15:00:00 | 3859.25 | 2024-04-30 09:15:00 | 3968.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-05-03 09:15:00 | 4023.00 | 2024-05-03 11:15:00 | 3963.40 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-05-13 11:30:00 | 3874.60 | 2024-05-16 14:15:00 | 4262.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 12:30:00 | 3868.30 | 2024-05-16 14:15:00 | 4255.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-27 10:45:00 | 4457.45 | 2024-07-01 15:15:00 | 4471.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-06-27 12:15:00 | 4470.25 | 2024-07-01 15:15:00 | 4471.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-06-27 13:15:00 | 4471.85 | 2024-07-01 15:15:00 | 4471.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-06-28 10:45:00 | 4447.05 | 2024-07-01 15:15:00 | 4471.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-05 10:45:00 | 4740.30 | 2024-07-08 15:15:00 | 4604.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-07-23 12:15:00 | 4041.95 | 2024-07-24 09:15:00 | 4274.85 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest2 | 2024-07-23 13:30:00 | 4093.30 | 2024-07-24 09:15:00 | 4274.85 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2024-07-29 13:45:00 | 4285.40 | 2024-08-02 10:15:00 | 4308.25 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2024-08-14 11:30:00 | 4252.45 | 2024-08-20 10:15:00 | 4677.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-30 09:15:00 | 4530.40 | 2024-08-30 14:15:00 | 4619.15 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-08-30 10:30:00 | 4557.95 | 2024-08-30 14:15:00 | 4619.15 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-09-04 12:00:00 | 4489.50 | 2024-09-13 09:15:00 | 4430.55 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2024-09-05 09:30:00 | 4488.15 | 2024-09-13 09:15:00 | 4430.55 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2024-09-17 15:15:00 | 4440.00 | 2024-09-19 10:15:00 | 4408.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-18 12:00:00 | 4437.80 | 2024-09-19 10:15:00 | 4408.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-09-18 13:15:00 | 4441.55 | 2024-09-19 10:15:00 | 4408.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-09-23 11:00:00 | 4270.00 | 2024-09-30 14:15:00 | 4299.85 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-09-25 10:00:00 | 4257.55 | 2024-09-30 14:15:00 | 4299.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-23 12:15:00 | 4011.10 | 2024-10-25 10:15:00 | 3810.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 09:15:00 | 4003.90 | 2024-10-25 10:15:00 | 3803.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 12:15:00 | 4011.10 | 2024-10-28 10:15:00 | 3894.95 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2024-10-24 09:15:00 | 4003.90 | 2024-10-28 10:15:00 | 3894.95 | STOP_HIT | 0.50 | 2.72% |
| BUY | retest2 | 2024-11-04 11:45:00 | 4021.85 | 2024-11-05 10:15:00 | 3915.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-11-04 14:30:00 | 4031.95 | 2024-11-05 10:15:00 | 3915.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-11-21 09:45:00 | 3744.60 | 2024-11-21 10:15:00 | 3822.60 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-11-26 13:15:00 | 4019.90 | 2024-12-02 11:15:00 | 4421.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-17 10:30:00 | 4437.45 | 2024-12-19 09:15:00 | 4215.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:30:00 | 4437.45 | 2024-12-19 13:15:00 | 4295.55 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest1 | 2024-12-31 09:15:00 | 4375.00 | 2024-12-31 09:15:00 | 4318.75 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-12-31 11:15:00 | 4362.50 | 2025-01-03 14:15:00 | 4365.65 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-12-31 12:15:00 | 4382.00 | 2025-01-03 14:15:00 | 4365.65 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-01-08 09:15:00 | 4264.90 | 2025-01-13 09:15:00 | 4051.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 4264.90 | 2025-01-14 13:15:00 | 4025.10 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2025-01-30 13:15:00 | 3961.50 | 2025-01-31 09:15:00 | 4026.75 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-02-18 10:15:00 | 3406.70 | 2025-02-19 10:15:00 | 3468.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-02-18 11:00:00 | 3397.85 | 2025-02-19 10:15:00 | 3468.95 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-02-18 15:00:00 | 3404.00 | 2025-02-19 10:15:00 | 3468.95 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-02-25 09:15:00 | 3780.70 | 2025-02-27 09:15:00 | 3099.95 | STOP_HIT | 1.00 | -18.01% |
| SELL | retest2 | 2025-03-25 11:15:00 | 2860.80 | 2025-03-28 09:15:00 | 2910.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-03-26 10:45:00 | 2857.75 | 2025-03-28 09:15:00 | 2910.20 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-03-26 11:45:00 | 2851.50 | 2025-03-28 09:15:00 | 2910.20 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-04-04 12:00:00 | 2667.00 | 2025-04-07 09:15:00 | 2533.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 12:00:00 | 2667.00 | 2025-04-08 09:15:00 | 2575.00 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2025-04-28 12:00:00 | 3008.70 | 2025-05-07 09:15:00 | 3309.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-15 09:15:00 | 3429.70 | 2025-05-21 13:15:00 | 3437.10 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-06-11 15:15:00 | 3809.80 | 2025-06-12 09:15:00 | 3747.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-06-17 11:15:00 | 3650.00 | 2025-06-23 10:15:00 | 3678.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-27 09:15:00 | 3784.00 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-27 15:15:00 | 3790.00 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-30 13:15:00 | 3786.00 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-30 14:00:00 | 3786.70 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-01 09:15:00 | 3859.30 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-07-01 11:15:00 | 3817.40 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-01 12:00:00 | 3812.60 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-01 15:00:00 | 3811.70 | 2025-07-02 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-07-14 11:15:00 | 3706.70 | 2025-07-14 13:15:00 | 3720.60 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-21 10:15:00 | 3931.90 | 2025-07-24 10:15:00 | 3911.30 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-23 09:45:00 | 3933.30 | 2025-07-24 10:15:00 | 3911.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-23 11:30:00 | 3939.30 | 2025-07-24 10:15:00 | 3911.30 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-24 09:30:00 | 3937.50 | 2025-07-24 10:15:00 | 3911.30 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-28 11:45:00 | 3894.20 | 2025-07-29 13:15:00 | 3926.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-29 09:15:00 | 3903.00 | 2025-07-29 13:15:00 | 3926.40 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-29 10:00:00 | 3902.00 | 2025-07-29 13:15:00 | 3926.40 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-08-01 11:30:00 | 3826.30 | 2025-08-04 14:15:00 | 3883.80 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-01 14:30:00 | 3828.30 | 2025-08-04 14:15:00 | 3883.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-08-22 10:15:00 | 4008.00 | 2025-08-25 14:15:00 | 3935.50 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-08-22 11:45:00 | 4001.50 | 2025-08-25 14:15:00 | 3935.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-08-22 12:30:00 | 4001.00 | 2025-08-25 14:15:00 | 3935.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-08 09:15:00 | 4068.90 | 2025-09-09 11:15:00 | 4030.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-08 11:15:00 | 4064.00 | 2025-09-09 11:15:00 | 4030.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-08 14:15:00 | 4062.90 | 2025-09-09 11:15:00 | 4030.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-12 09:15:00 | 4096.00 | 2025-09-16 09:15:00 | 4036.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-09-12 11:45:00 | 4107.00 | 2025-09-16 09:15:00 | 4036.20 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-09-12 12:30:00 | 4106.10 | 2025-09-16 09:15:00 | 4036.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-09-19 09:30:00 | 4178.50 | 2025-09-22 14:15:00 | 4131.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-19 11:15:00 | 4164.20 | 2025-09-22 14:15:00 | 4131.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-22 09:30:00 | 4183.00 | 2025-09-22 14:15:00 | 4131.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-24 09:15:00 | 4087.40 | 2025-09-25 12:15:00 | 4163.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-24 11:15:00 | 4117.80 | 2025-09-25 12:15:00 | 4163.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-25 09:30:00 | 4122.70 | 2025-09-25 12:15:00 | 4163.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 4005.00 | 2025-10-06 09:15:00 | 4087.60 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-10-01 09:15:00 | 4025.10 | 2025-10-06 09:15:00 | 4087.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-10-01 13:15:00 | 4020.30 | 2025-10-06 09:15:00 | 4087.60 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-10-13 11:45:00 | 4268.50 | 2025-10-16 09:15:00 | 4208.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-13 12:15:00 | 4269.50 | 2025-10-16 09:15:00 | 4208.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-14 13:45:00 | 4283.10 | 2025-10-16 09:15:00 | 4208.60 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-10-20 12:30:00 | 4140.00 | 2025-10-30 12:15:00 | 4116.50 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-10-20 15:00:00 | 4144.10 | 2025-10-30 12:15:00 | 4116.50 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-10-21 14:00:00 | 4141.00 | 2025-10-30 12:15:00 | 4116.50 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-10-23 09:30:00 | 4135.60 | 2025-10-30 12:15:00 | 4116.50 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-10-27 12:00:00 | 4092.00 | 2025-10-30 12:15:00 | 4116.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-10-27 14:00:00 | 4095.50 | 2025-10-30 12:15:00 | 4116.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-20 11:15:00 | 4147.50 | 2025-11-21 09:15:00 | 4103.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-20 12:00:00 | 4148.50 | 2025-11-21 09:15:00 | 4103.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-10 10:45:00 | 4042.80 | 2025-12-12 10:15:00 | 4078.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-11 12:45:00 | 4035.00 | 2025-12-12 10:15:00 | 4078.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-12-12 10:00:00 | 4061.00 | 2025-12-12 10:15:00 | 4078.10 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-12-29 09:15:00 | 4399.00 | 2025-12-30 12:15:00 | 4364.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-29 13:15:00 | 4393.50 | 2025-12-30 12:15:00 | 4364.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-29 15:15:00 | 4400.00 | 2025-12-30 12:15:00 | 4364.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-12-30 09:30:00 | 4399.10 | 2025-12-30 12:15:00 | 4364.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-13 10:45:00 | 4256.70 | 2026-01-14 15:15:00 | 4385.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-01-13 11:30:00 | 4273.50 | 2026-01-14 15:15:00 | 4385.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-01-13 12:00:00 | 4268.70 | 2026-01-14 15:15:00 | 4385.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-13 12:30:00 | 4263.60 | 2026-01-14 15:15:00 | 4385.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-02-02 14:45:00 | 4044.70 | 2026-02-03 09:15:00 | 4449.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-06 12:30:00 | 4955.50 | 2026-03-09 09:15:00 | 4707.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:00:00 | 4955.00 | 2026-03-09 09:15:00 | 4707.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:30:00 | 4955.50 | 2026-03-09 14:15:00 | 4793.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2026-03-06 13:00:00 | 4955.00 | 2026-03-09 14:15:00 | 4793.00 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2026-03-24 10:45:00 | 4028.00 | 2026-03-24 13:15:00 | 4125.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4442.70 | 2026-04-20 09:15:00 | 4886.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 12:30:00 | 4436.80 | 2026-04-20 09:15:00 | 4880.48 | TARGET_HIT | 1.00 | 10.00% |
