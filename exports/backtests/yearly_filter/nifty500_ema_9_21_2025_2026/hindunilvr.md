# Hindustan Unilever Ltd. (HINDUNILVR)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2286.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 76 |
| ALERT1 | 54 |
| ALERT2 | 53 |
| ALERT2_SKIP | 21 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 69 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 22 / 53
- **Target hits / Stop hits / Partials:** 0 / 71 / 4
- **Avg / median % per leg:** 0.13% / -0.57%
- **Sum % (uncompounded):** 9.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 7 | 18.9% | 0 | 36 | 1 | -0.13% | -4.7% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.43% | 10.3% |
| BUY @ 3rd Alert (retest2) | 34 | 4 | 11.8% | 0 | 34 | 0 | -0.44% | -15.0% |
| SELL (all) | 38 | 15 | 39.5% | 0 | 35 | 3 | 0.37% | 14.2% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.21% | 1.2% |
| SELL @ 3rd Alert (retest2) | 37 | 14 | 37.8% | 0 | 34 | 3 | 0.35% | 13.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 3 | 1 | 2.87% | 11.5% |
| retest2 (combined) | 71 | 18 | 25.4% | 0 | 68 | 3 | -0.03% | -2.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 2340.55 | 2313.91 | 2312.63 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 09:15:00 | 2309.56 | 2322.83 | 2323.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 2303.07 | 2316.06 | 2319.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 14:15:00 | 2314.77 | 2313.91 | 2318.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:30:00 | 2311.53 | 2313.91 | 2318.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 2307.79 | 2312.56 | 2316.70 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 2323.53 | 2317.00 | 2316.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 2336.22 | 2320.84 | 2318.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 2342.02 | 2342.98 | 2334.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 2342.02 | 2342.98 | 2334.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2335.24 | 2341.39 | 2335.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 2335.24 | 2341.39 | 2335.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2333.27 | 2339.76 | 2335.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 2333.27 | 2339.76 | 2335.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2327.66 | 2337.34 | 2334.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 2326.68 | 2337.34 | 2334.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2312.51 | 2329.43 | 2331.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 2301.99 | 2323.94 | 2328.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2327.76 | 2322.42 | 2327.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 2327.76 | 2322.42 | 2327.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2327.76 | 2322.42 | 2327.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 2324.41 | 2322.42 | 2327.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2330.51 | 2324.04 | 2327.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2333.27 | 2324.04 | 2327.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2322.84 | 2323.80 | 2327.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 2287.04 | 2324.68 | 2326.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:45:00 | 2320.09 | 2307.04 | 2311.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 2342.12 | 2318.52 | 2315.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 2342.12 | 2318.52 | 2315.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 2350.09 | 2324.83 | 2318.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 2335.43 | 2345.10 | 2336.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 12:15:00 | 2335.43 | 2345.10 | 2336.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 2335.43 | 2345.10 | 2336.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 2335.43 | 2345.10 | 2336.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 2327.46 | 2341.57 | 2335.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 2327.46 | 2341.57 | 2335.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 2342.22 | 2341.70 | 2335.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 2344.09 | 2341.70 | 2335.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 2324.91 | 2333.05 | 2333.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 2324.91 | 2333.05 | 2333.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 2320.97 | 2330.08 | 2332.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 2332.68 | 2330.60 | 2332.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 11:15:00 | 2332.68 | 2330.60 | 2332.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 2332.68 | 2330.60 | 2332.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 2332.68 | 2330.60 | 2332.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 2323.63 | 2329.21 | 2331.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 2316.64 | 2324.18 | 2327.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 2316.64 | 2322.57 | 2326.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 2315.07 | 2321.07 | 2325.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 2306.12 | 2318.08 | 2323.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2328.35 | 2318.47 | 2322.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 2332.87 | 2321.35 | 2323.89 | SL hit (close>static) qty=1.00 sl=2332.19 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 2330.51 | 2325.58 | 2325.39 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2314.68 | 2324.71 | 2325.18 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 2327.27 | 2320.76 | 2319.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 2337.01 | 2324.01 | 2321.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 2325.00 | 2325.41 | 2322.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 14:00:00 | 2325.00 | 2325.41 | 2322.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 2336.91 | 2327.71 | 2323.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 2342.22 | 2329.61 | 2325.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:30:00 | 2341.14 | 2337.43 | 2331.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 2339.46 | 2347.59 | 2347.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 2333.46 | 2344.76 | 2346.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 2333.46 | 2344.76 | 2346.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 2311.73 | 2338.15 | 2342.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 2290.77 | 2286.52 | 2301.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:30:00 | 2290.48 | 2286.52 | 2301.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2290.18 | 2287.95 | 2292.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 2290.18 | 2287.95 | 2292.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2288.61 | 2288.41 | 2291.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 2280.64 | 2285.77 | 2290.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 2256.15 | 2245.49 | 2244.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 2256.15 | 2245.49 | 2244.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 2257.43 | 2249.40 | 2246.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 2252.71 | 2262.23 | 2256.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 2252.71 | 2262.23 | 2256.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 2252.71 | 2262.23 | 2256.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 2252.71 | 2262.23 | 2256.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2254.87 | 2260.76 | 2256.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:30:00 | 2251.03 | 2260.76 | 2256.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 2258.41 | 2259.60 | 2256.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 2259.79 | 2259.60 | 2256.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 2264.41 | 2260.56 | 2257.38 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 2252.80 | 2256.61 | 2256.64 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 2261.46 | 2257.58 | 2257.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 2269.92 | 2261.55 | 2259.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 2353.73 | 2356.96 | 2336.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 15:15:00 | 2360.81 | 2356.76 | 2338.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 2365.63 | 2376.34 | 2366.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 2365.63 | 2376.34 | 2366.23 | SL hit (close<ema400) qty=1.00 sl=2366.23 alert=retest1 |

### Cycle 14 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 2448.46 | 2467.37 | 2469.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 2444.62 | 2457.16 | 2464.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 2430.75 | 2428.91 | 2439.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:45:00 | 2430.85 | 2428.91 | 2439.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 2439.11 | 2432.47 | 2439.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 2439.90 | 2432.47 | 2439.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 2441.96 | 2434.37 | 2439.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 2430.45 | 2434.37 | 2439.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2431.83 | 2433.86 | 2438.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 2424.55 | 2431.05 | 2437.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 2409.99 | 2399.10 | 2397.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 2409.99 | 2399.10 | 2397.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 2413.34 | 2401.94 | 2399.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 2395.14 | 2402.88 | 2400.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 2395.14 | 2402.88 | 2400.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 2395.14 | 2402.88 | 2400.14 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 2374.29 | 2393.89 | 2396.33 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 2483.87 | 2413.52 | 2404.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 2488.88 | 2446.94 | 2423.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 12:15:00 | 2505.71 | 2507.46 | 2487.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 2505.71 | 2507.46 | 2487.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2481.61 | 2500.23 | 2490.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:15:00 | 2485.24 | 2500.23 | 2490.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2480.62 | 2496.31 | 2489.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 2480.62 | 2496.31 | 2489.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 2490.95 | 2495.24 | 2489.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 2499.02 | 2492.50 | 2489.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 11:15:00 | 2502.16 | 2495.48 | 2491.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:30:00 | 2501.48 | 2493.72 | 2491.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 2498.52 | 2493.75 | 2491.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 2498.52 | 2494.71 | 2492.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 2485.15 | 2494.71 | 2492.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2481.21 | 2492.01 | 2491.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 2481.21 | 2492.01 | 2491.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 2477.97 | 2489.20 | 2490.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 2477.97 | 2489.20 | 2490.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2457.11 | 2480.21 | 2485.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 2468.23 | 2461.84 | 2468.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 2468.23 | 2461.84 | 2468.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2468.23 | 2461.84 | 2468.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 2466.75 | 2461.84 | 2468.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 2460.55 | 2461.58 | 2468.20 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 2490.66 | 2471.56 | 2471.16 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2468.03 | 2470.85 | 2470.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 2453.47 | 2467.38 | 2469.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 2454.46 | 2448.78 | 2455.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 14:00:00 | 2454.46 | 2448.78 | 2455.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 2452.19 | 2449.47 | 2455.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 2446.29 | 2453.65 | 2455.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 2520.56 | 2462.72 | 2458.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2520.56 | 2462.72 | 2458.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 2552.92 | 2500.89 | 2479.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 2597.19 | 2602.96 | 2569.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 2592.56 | 2602.96 | 2569.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 2586.86 | 2597.97 | 2585.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 2583.71 | 2597.97 | 2585.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 2585.68 | 2595.51 | 2585.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 2585.68 | 2595.51 | 2585.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2585.97 | 2593.60 | 2585.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 2578.69 | 2593.60 | 2585.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 2582.63 | 2591.41 | 2585.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 2582.63 | 2591.41 | 2585.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 2587.74 | 2590.67 | 2585.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 2583.81 | 2590.67 | 2585.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 2587.05 | 2589.95 | 2585.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 2578.20 | 2589.95 | 2585.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2578.20 | 2587.60 | 2584.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 2575.45 | 2587.60 | 2584.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 2584.10 | 2586.90 | 2584.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 2589.02 | 2587.32 | 2585.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2602.11 | 2587.85 | 2586.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 2596.89 | 2613.53 | 2615.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 09:15:00 | 2596.89 | 2613.53 | 2615.01 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 2637.32 | 2616.06 | 2615.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 2656.31 | 2631.35 | 2627.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 2627.09 | 2630.50 | 2627.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 2627.09 | 2630.50 | 2627.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2627.09 | 2630.50 | 2627.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2627.09 | 2630.50 | 2627.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2633.48 | 2631.10 | 2628.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 2643.22 | 2633.50 | 2629.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 2623.16 | 2632.66 | 2629.86 | SL hit (close<static) qty=1.00 sl=2627.09 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2611.65 | 2626.67 | 2627.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 2601.12 | 2621.56 | 2625.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 2589.51 | 2588.15 | 2597.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:00:00 | 2589.51 | 2588.15 | 2597.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 2601.22 | 2590.68 | 2595.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 2601.22 | 2590.68 | 2595.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 2597.09 | 2591.96 | 2595.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 2607.71 | 2591.96 | 2595.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2590.10 | 2591.59 | 2595.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 2587.35 | 2591.59 | 2595.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 2587.84 | 2589.23 | 2593.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:45:00 | 2583.02 | 2591.20 | 2592.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2457.98 | 2472.20 | 2486.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2458.45 | 2472.20 | 2486.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2453.87 | 2472.20 | 2486.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 2469.70 | 2468.82 | 2481.12 | SL hit (close>ema200) qty=0.50 sl=2468.82 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 2488.39 | 2478.21 | 2477.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 2492.23 | 2481.02 | 2478.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 2484.85 | 2490.58 | 2486.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 2484.85 | 2490.58 | 2486.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 2484.85 | 2490.58 | 2486.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 2491.54 | 2490.58 | 2486.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 2493.41 | 2491.14 | 2486.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 2495.48 | 2491.64 | 2487.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 2494.29 | 2492.39 | 2488.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:00:00 | 2495.38 | 2492.39 | 2488.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 2482.39 | 2494.58 | 2491.32 | SL hit (close<static) qty=1.00 sl=2483.57 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 2477.67 | 2487.66 | 2488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 2473.54 | 2484.83 | 2487.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 2488.69 | 2485.60 | 2487.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 15:15:00 | 2488.69 | 2485.60 | 2487.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2488.69 | 2485.60 | 2487.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2466.65 | 2485.60 | 2487.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 2470.59 | 2472.61 | 2479.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:45:00 | 2472.56 | 2468.64 | 2474.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:00:00 | 2472.16 | 2469.35 | 2474.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2472.95 | 2470.07 | 2474.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 2472.85 | 2470.07 | 2474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 2471.38 | 2470.33 | 2473.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:45:00 | 2472.56 | 2470.33 | 2473.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2476.69 | 2471.60 | 2474.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 2476.69 | 2471.60 | 2474.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2481.21 | 2473.52 | 2474.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2476.79 | 2473.52 | 2474.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2479.83 | 2474.79 | 2475.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 2481.41 | 2474.79 | 2475.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 2480.82 | 2475.99 | 2475.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 2480.82 | 2475.99 | 2475.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 2484.75 | 2480.09 | 2477.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 2470.00 | 2478.82 | 2477.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 2470.00 | 2478.82 | 2477.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2470.00 | 2478.82 | 2477.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 2470.00 | 2478.82 | 2477.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2457.41 | 2474.54 | 2475.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 2452.49 | 2468.06 | 2472.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 11:15:00 | 2462.23 | 2458.94 | 2464.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 2462.23 | 2458.94 | 2464.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2462.23 | 2458.94 | 2464.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 2462.23 | 2458.94 | 2464.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2459.67 | 2459.09 | 2464.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:15:00 | 2463.90 | 2459.09 | 2464.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 2459.96 | 2459.26 | 2464.05 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 2483.77 | 2468.23 | 2466.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 2503.05 | 2481.94 | 2475.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 2554.50 | 2557.85 | 2539.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 2554.50 | 2557.85 | 2539.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 2572.10 | 2570.06 | 2554.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 2572.79 | 2570.06 | 2554.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2559.12 | 2567.87 | 2555.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 2559.12 | 2567.87 | 2555.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2555.77 | 2565.45 | 2555.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2555.77 | 2565.45 | 2555.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 2469.01 | 2545.29 | 2547.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 10:15:00 | 2433.21 | 2446.16 | 2458.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2419.44 | 2417.93 | 2426.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:15:00 | 2412.16 | 2418.31 | 2426.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 2417.37 | 2415.25 | 2421.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 2419.34 | 2415.25 | 2421.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2411.57 | 2410.86 | 2417.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:30:00 | 2404.68 | 2410.97 | 2415.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 2407.14 | 2410.97 | 2415.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:30:00 | 2401.53 | 2408.51 | 2414.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 2382.94 | 2375.98 | 2381.33 | SL hit (close>ema400) qty=1.00 sl=2381.33 alert=retest1 |

### Cycle 31 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 2386.68 | 2383.81 | 2383.55 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 2377.14 | 2383.19 | 2383.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 2372.32 | 2381.02 | 2382.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 2379.40 | 2375.65 | 2378.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 2379.40 | 2375.65 | 2378.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2379.40 | 2375.65 | 2378.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 2379.40 | 2375.65 | 2378.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 2379.11 | 2376.34 | 2378.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 2379.11 | 2376.34 | 2378.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 2376.65 | 2376.40 | 2378.71 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 2386.39 | 2380.38 | 2380.06 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2369.76 | 2380.59 | 2380.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 2363.66 | 2372.50 | 2376.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 2392.29 | 2375.23 | 2376.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 2392.29 | 2375.23 | 2376.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2392.29 | 2375.23 | 2376.83 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 2399.96 | 2380.18 | 2378.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 2407.63 | 2392.81 | 2385.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 2397.11 | 2401.07 | 2393.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:00:00 | 2397.11 | 2401.07 | 2393.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 2391.50 | 2398.85 | 2394.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 2391.50 | 2398.85 | 2394.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 2389.34 | 2396.95 | 2393.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 2384.42 | 2394.19 | 2392.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2385.30 | 2392.41 | 2391.99 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 2381.57 | 2390.24 | 2391.04 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 2401.73 | 2392.13 | 2391.74 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 2384.42 | 2391.09 | 2391.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 2381.37 | 2388.24 | 2390.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 2382.45 | 2382.13 | 2386.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 2382.45 | 2382.13 | 2386.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2383.73 | 2379.88 | 2383.79 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2404.78 | 2387.93 | 2385.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 2416.98 | 2393.74 | 2388.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 12:15:00 | 2428.19 | 2430.85 | 2419.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:00:00 | 2428.19 | 2430.85 | 2419.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 2421.80 | 2429.57 | 2420.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 2421.80 | 2429.57 | 2420.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 2418.85 | 2427.43 | 2420.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 2432.82 | 2427.43 | 2420.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2414.81 | 2429.24 | 2425.83 | SL hit (close<static) qty=1.00 sl=2418.85 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 2409.80 | 2423.66 | 2424.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 2401.63 | 2419.26 | 2422.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 2419.44 | 2417.08 | 2420.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 2419.44 | 2417.08 | 2420.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 2420.32 | 2417.73 | 2420.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 2431.63 | 2417.73 | 2420.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 2425.83 | 2419.35 | 2420.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 2432.52 | 2419.35 | 2420.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 2422.29 | 2419.93 | 2420.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 2433.60 | 2419.93 | 2420.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 2423.86 | 2420.72 | 2421.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 2462.19 | 2408.90 | 2415.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2309.00 | 2305.67 | 2311.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 2300.90 | 2305.67 | 2311.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 2296.90 | 2289.59 | 2289.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 2296.90 | 2289.59 | 2289.22 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2286.20 | 2288.91 | 2288.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 2281.40 | 2287.02 | 2288.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 2285.20 | 2284.85 | 2286.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:30:00 | 2281.50 | 2284.85 | 2286.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 2268.30 | 2281.54 | 2284.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 2267.50 | 2277.92 | 2281.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:00:00 | 2265.80 | 2273.12 | 2278.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 2268.10 | 2272.12 | 2277.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 2265.00 | 2269.70 | 2275.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2271.40 | 2269.37 | 2274.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2275.30 | 2269.37 | 2274.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2270.00 | 2269.50 | 2273.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 2273.20 | 2269.50 | 2273.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2274.90 | 2270.58 | 2273.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2274.90 | 2270.58 | 2273.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2280.20 | 2272.50 | 2274.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 2280.20 | 2272.50 | 2274.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2281.60 | 2274.32 | 2275.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 2281.60 | 2274.32 | 2275.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 2280.90 | 2275.64 | 2275.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2280.90 | 2275.64 | 2275.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2285.90 | 2278.55 | 2276.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 2285.20 | 2285.67 | 2281.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 2283.30 | 2285.67 | 2281.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2298.50 | 2288.23 | 2283.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:30:00 | 2300.20 | 2290.69 | 2284.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:00:00 | 2300.50 | 2290.69 | 2284.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 2300.10 | 2291.14 | 2286.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 2283.00 | 2288.28 | 2288.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 2283.00 | 2288.28 | 2288.40 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 2291.00 | 2287.45 | 2287.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 2295.90 | 2289.57 | 2288.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 12:15:00 | 2285.00 | 2290.02 | 2289.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 2285.00 | 2290.02 | 2289.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 2285.00 | 2290.02 | 2289.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 2285.00 | 2290.02 | 2289.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2285.90 | 2289.20 | 2288.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2285.90 | 2289.20 | 2288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2291.00 | 2289.56 | 2288.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 2286.90 | 2289.56 | 2288.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2316.00 | 2294.92 | 2291.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 2320.50 | 2301.79 | 2295.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 2318.50 | 2307.10 | 2299.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:45:00 | 2320.20 | 2312.68 | 2306.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 2374.90 | 2391.06 | 2391.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 2374.90 | 2391.06 | 2391.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 2364.10 | 2377.99 | 2384.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 2378.00 | 2377.15 | 2382.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 2386.20 | 2377.15 | 2382.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2381.00 | 2377.92 | 2382.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 2388.00 | 2377.92 | 2382.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 2381.90 | 2378.72 | 2382.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:00:00 | 2381.90 | 2378.72 | 2382.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 2386.70 | 2380.31 | 2382.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 2386.70 | 2380.31 | 2382.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 2390.90 | 2382.43 | 2383.62 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 2402.20 | 2386.38 | 2385.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 2407.90 | 2390.69 | 2387.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 10:15:00 | 2395.70 | 2396.23 | 2391.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:00:00 | 2395.70 | 2396.23 | 2391.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2382.60 | 2393.50 | 2390.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 2382.60 | 2393.50 | 2390.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 2369.80 | 2388.76 | 2388.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 2369.80 | 2388.76 | 2388.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 13:15:00 | 2372.80 | 2385.57 | 2387.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 2351.40 | 2377.15 | 2382.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 2363.70 | 2360.78 | 2370.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 2363.70 | 2360.78 | 2370.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2363.70 | 2360.78 | 2370.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 2369.60 | 2360.78 | 2370.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2365.00 | 2361.63 | 2369.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 2358.00 | 2362.88 | 2369.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 2355.50 | 2362.71 | 2369.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2365.82 | 2368.31 | SL hit (close>static) qty=1.00 sl=2372.90 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 2392.30 | 2371.11 | 2370.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 11:15:00 | 2397.50 | 2376.39 | 2372.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 2413.40 | 2414.09 | 2399.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 14:00:00 | 2413.40 | 2414.09 | 2399.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 2377.10 | 2406.69 | 2397.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 2377.10 | 2406.69 | 2397.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2382.00 | 2401.76 | 2396.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 2388.20 | 2401.76 | 2396.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 10:45:00 | 2387.70 | 2396.12 | 2394.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 2385.70 | 2392.19 | 2392.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 2385.70 | 2392.19 | 2392.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 2364.90 | 2386.73 | 2390.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2381.20 | 2380.01 | 2385.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2381.20 | 2380.01 | 2385.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2381.20 | 2380.01 | 2385.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 2367.80 | 2377.14 | 2383.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2407.50 | 2386.91 | 2385.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 2407.50 | 2386.91 | 2385.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 2424.70 | 2394.47 | 2389.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 2406.00 | 2407.76 | 2399.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 2405.90 | 2407.76 | 2399.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2409.00 | 2408.01 | 2400.17 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 2362.60 | 2394.05 | 2396.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 13:15:00 | 2348.90 | 2373.55 | 2385.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 2378.60 | 2374.56 | 2384.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 2378.60 | 2374.56 | 2384.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 2386.00 | 2376.85 | 2384.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 2344.60 | 2376.85 | 2384.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 2382.40 | 2356.64 | 2353.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 2382.40 | 2356.64 | 2353.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 2385.80 | 2366.27 | 2358.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 2364.80 | 2365.98 | 2359.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 2364.80 | 2365.98 | 2359.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2378.70 | 2369.09 | 2362.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2412.50 | 2369.86 | 2365.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 2353.90 | 2371.94 | 2369.47 | SL hit (close<static) qty=1.00 sl=2360.60 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 2355.00 | 2365.68 | 2366.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2339.00 | 2360.34 | 2364.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 2362.40 | 2360.76 | 2364.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 2362.40 | 2360.76 | 2364.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2377.30 | 2364.06 | 2365.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 2377.30 | 2364.06 | 2365.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 2381.50 | 2367.55 | 2366.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 2402.00 | 2374.44 | 2370.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2442.33 | 2437.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 2376.70 | 2442.33 | 2437.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2376.70 | 2442.33 | 2437.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 2376.70 | 2442.33 | 2437.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 2414.50 | 2436.76 | 2435.67 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 2418.60 | 2433.13 | 2434.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 2407.00 | 2421.94 | 2428.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 2312.90 | 2312.17 | 2330.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:45:00 | 2311.00 | 2312.17 | 2330.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 2321.00 | 2314.04 | 2324.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 2321.00 | 2314.04 | 2324.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 2325.10 | 2316.25 | 2324.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 2325.10 | 2316.25 | 2324.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2323.00 | 2317.60 | 2324.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:00:00 | 2319.00 | 2321.28 | 2324.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 2329.00 | 2307.58 | 2312.43 | SL hit (close>static) qty=1.00 sl=2328.90 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 2334.40 | 2317.02 | 2315.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 2345.70 | 2333.17 | 2325.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 2352.20 | 2356.96 | 2346.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 2352.20 | 2356.96 | 2346.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 2364.40 | 2365.14 | 2355.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 2357.30 | 2365.14 | 2355.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2343.30 | 2367.56 | 2362.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 2343.30 | 2367.56 | 2362.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2351.30 | 2364.30 | 2361.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 2354.30 | 2360.86 | 2360.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:00:00 | 2360.90 | 2360.87 | 2360.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 2339.50 | 2356.04 | 2357.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 2339.50 | 2356.04 | 2357.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 2315.30 | 2345.33 | 2352.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2324.50 | 2323.62 | 2337.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:30:00 | 2325.40 | 2323.62 | 2337.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2204.50 | 2199.62 | 2220.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 2195.10 | 2199.88 | 2218.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:45:00 | 2192.30 | 2197.10 | 2215.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2166.70 | 2163.16 | 2163.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 11:15:00 | 2166.70 | 2163.16 | 2163.15 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 2160.80 | 2162.69 | 2162.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 2157.70 | 2161.69 | 2162.46 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 2177.40 | 2164.83 | 2163.82 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 2151.70 | 2162.94 | 2163.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 12:15:00 | 2145.00 | 2153.97 | 2158.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 2097.30 | 2095.39 | 2114.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:30:00 | 2098.80 | 2095.39 | 2114.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 2080.30 | 2068.05 | 2085.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:45:00 | 2090.90 | 2068.05 | 2085.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 2052.40 | 2064.92 | 2082.91 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2134.70 | 2094.75 | 2089.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 2141.30 | 2104.06 | 2094.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2098.80 | 2122.10 | 2110.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2098.80 | 2122.10 | 2110.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2098.80 | 2122.10 | 2110.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 2098.80 | 2122.10 | 2110.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2096.40 | 2116.96 | 2109.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 2096.40 | 2116.96 | 2109.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 2079.50 | 2102.85 | 2104.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 2071.30 | 2096.54 | 2101.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2084.20 | 2067.77 | 2078.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2084.20 | 2067.77 | 2078.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2084.20 | 2067.77 | 2078.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 2068.90 | 2068.07 | 2077.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 2083.30 | 2062.32 | 2061.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2083.30 | 2062.32 | 2061.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 2107.00 | 2074.05 | 2068.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 2136.60 | 2137.10 | 2114.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 2134.00 | 2137.10 | 2114.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 2121.10 | 2131.98 | 2119.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 2117.80 | 2131.98 | 2119.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 2133.90 | 2132.36 | 2120.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2151.60 | 2131.99 | 2121.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2136.30 | 2143.19 | 2136.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 2121.00 | 2131.45 | 2132.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 2121.00 | 2131.45 | 2132.53 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2162.10 | 2137.58 | 2135.21 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 2140.70 | 2142.68 | 2142.90 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2204.60 | 2154.49 | 2148.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 2226.10 | 2168.81 | 2155.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 2227.20 | 2232.62 | 2212.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:15:00 | 2239.40 | 2232.62 | 2212.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 2351.37 | 2286.49 | 2252.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 2353.00 | 2364.49 | 2337.19 | SL hit (close<ema200) qty=0.50 sl=2364.49 alert=retest1 |

### Cycle 70 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 2310.90 | 2328.44 | 2330.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 2291.80 | 2317.99 | 2324.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 2305.70 | 2305.63 | 2315.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:30:00 | 2301.70 | 2305.63 | 2315.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 2334.20 | 2311.55 | 2316.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 2334.20 | 2311.55 | 2316.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 2347.90 | 2318.82 | 2319.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 2347.90 | 2318.82 | 2319.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 2328.00 | 2320.65 | 2320.32 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 2313.80 | 2319.28 | 2319.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2300.80 | 2315.24 | 2317.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 2346.90 | 2283.07 | 2293.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 2346.90 | 2283.07 | 2293.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2346.90 | 2283.07 | 2293.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 2346.90 | 2283.07 | 2293.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 2348.60 | 2296.18 | 2298.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 2354.80 | 2296.18 | 2298.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 2338.80 | 2304.70 | 2301.92 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 2298.40 | 2305.17 | 2306.02 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 2317.50 | 2307.00 | 2306.66 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 2290.30 | 2305.07 | 2305.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 2285.80 | 2301.21 | 2304.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 11:15:00 | 2293.00 | 2283.44 | 2290.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 11:15:00 | 2293.00 | 2283.44 | 2290.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 2293.00 | 2283.44 | 2290.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 2295.00 | 2283.44 | 2290.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 2285.00 | 2283.75 | 2290.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 13:30:00 | 2284.40 | 2285.48 | 2290.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 09:15:00 | 2287.04 | 2025-05-26 09:15:00 | 2342.12 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-05-23 11:45:00 | 2320.09 | 2025-05-26 09:15:00 | 2342.12 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-27 15:15:00 | 2344.09 | 2025-05-28 14:15:00 | 2324.91 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-30 11:45:00 | 2316.64 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-30 12:30:00 | 2316.64 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-30 14:00:00 | 2315.07 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-05-30 15:00:00 | 2306.12 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-06 09:15:00 | 2342.22 | 2025-06-11 15:15:00 | 2333.46 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-06-06 13:30:00 | 2341.14 | 2025-06-11 15:15:00 | 2333.46 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-06-11 15:00:00 | 2339.46 | 2025-06-11 15:15:00 | 2333.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-18 10:30:00 | 2280.64 | 2025-06-27 09:15:00 | 2256.15 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest1 | 2025-07-08 15:15:00 | 2360.81 | 2025-07-10 12:15:00 | 2365.63 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-07-11 09:15:00 | 2467.54 | 2025-07-18 10:15:00 | 2448.46 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-07-23 10:30:00 | 2424.55 | 2025-07-29 13:15:00 | 2409.99 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-08-06 09:15:00 | 2499.02 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-06 11:15:00 | 2502.16 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-08-06 13:30:00 | 2501.48 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-08-06 15:15:00 | 2498.52 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-14 13:30:00 | 2446.29 | 2025-08-18 09:15:00 | 2520.56 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-08-25 11:45:00 | 2589.02 | 2025-09-01 09:15:00 | 2596.89 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-08-26 09:15:00 | 2602.11 | 2025-09-01 09:15:00 | 2596.89 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-09-04 12:45:00 | 2643.22 | 2025-09-04 14:15:00 | 2623.16 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2587.35 | 2025-09-29 09:15:00 | 2457.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 11:30:00 | 2587.84 | 2025-09-29 09:15:00 | 2458.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 14:45:00 | 2583.02 | 2025-09-29 09:15:00 | 2453.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2587.35 | 2025-09-29 12:15:00 | 2469.70 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-09-10 11:30:00 | 2587.84 | 2025-09-29 12:15:00 | 2469.70 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2025-09-11 14:45:00 | 2583.02 | 2025-09-29 12:15:00 | 2469.70 | STOP_HIT | 0.50 | 4.39% |
| BUY | retest2 | 2025-10-06 11:30:00 | 2495.48 | 2025-10-07 10:15:00 | 2482.39 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-10-06 12:30:00 | 2494.29 | 2025-10-07 10:15:00 | 2482.39 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-06 13:00:00 | 2495.38 | 2025-10-07 10:15:00 | 2482.39 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2466.65 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-08 14:00:00 | 2470.59 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 10:45:00 | 2472.56 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-10-09 12:00:00 | 2472.16 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-04 09:15:00 | 2412.16 | 2025-11-11 13:15:00 | 2382.94 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2025-11-06 12:30:00 | 2404.68 | 2025-11-12 14:15:00 | 2386.68 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-11-06 13:00:00 | 2407.14 | 2025-11-12 14:15:00 | 2386.68 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2025-11-06 13:30:00 | 2401.53 | 2025-11-12 14:15:00 | 2386.68 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-12-02 09:15:00 | 2432.82 | 2025-12-03 09:15:00 | 2414.81 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-03 10:15:00 | 2427.80 | 2025-12-03 14:15:00 | 2409.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-03 10:45:00 | 2426.72 | 2025-12-03 14:15:00 | 2409.80 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-03 14:00:00 | 2425.73 | 2025-12-03 14:15:00 | 2409.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-12 09:15:00 | 2300.90 | 2025-12-16 11:15:00 | 2296.90 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-12-18 09:15:00 | 2267.50 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-18 11:00:00 | 2265.80 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-18 12:00:00 | 2268.10 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-18 13:30:00 | 2265.00 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-23 10:30:00 | 2300.20 | 2025-12-24 15:15:00 | 2283.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-23 11:00:00 | 2300.50 | 2025-12-24 15:15:00 | 2283.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-23 14:15:00 | 2300.10 | 2025-12-24 15:15:00 | 2283.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-31 12:15:00 | 2320.50 | 2026-01-09 09:15:00 | 2374.90 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-12-31 13:30:00 | 2318.50 | 2026-01-09 09:15:00 | 2374.90 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2026-01-01 14:45:00 | 2320.20 | 2026-01-09 09:15:00 | 2374.90 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2026-01-16 12:15:00 | 2358.00 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-16 13:15:00 | 2355.50 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-01-21 09:15:00 | 2388.20 | 2026-01-21 12:15:00 | 2385.70 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-01-21 10:45:00 | 2387.70 | 2026-01-21 12:15:00 | 2385.70 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2367.80 | 2026-01-23 09:15:00 | 2407.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2344.60 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-05 09:15:00 | 2412.50 | 2026-02-05 13:15:00 | 2353.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-02-19 11:00:00 | 2319.00 | 2026-02-20 11:15:00 | 2329.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-02-20 11:45:00 | 2321.10 | 2026-02-23 09:15:00 | 2334.40 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-20 13:30:00 | 2313.00 | 2026-02-23 09:15:00 | 2334.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-27 12:15:00 | 2354.30 | 2026-02-27 14:15:00 | 2339.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-27 13:00:00 | 2360.90 | 2026-02-27 14:15:00 | 2339.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-10 11:15:00 | 2195.10 | 2026-03-16 11:15:00 | 2166.70 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2026-03-10 11:45:00 | 2192.30 | 2026-03-16 11:15:00 | 2166.70 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2026-04-01 10:45:00 | 2068.90 | 2026-04-06 14:15:00 | 2083.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-04-10 09:15:00 | 2151.60 | 2026-04-13 15:15:00 | 2121.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2136.30 | 2026-04-13 15:15:00 | 2121.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2026-04-21 09:15:00 | 2239.40 | 2026-04-22 09:15:00 | 2351.37 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-21 09:15:00 | 2239.40 | 2026-04-23 15:15:00 | 2353.00 | STOP_HIT | 0.50 | 5.07% |
| BUY | retest2 | 2026-04-27 09:15:00 | 2343.70 | 2026-04-28 09:15:00 | 2310.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-27 12:15:00 | 2335.50 | 2026-04-28 09:15:00 | 2310.90 | STOP_HIT | 1.00 | -1.05% |
