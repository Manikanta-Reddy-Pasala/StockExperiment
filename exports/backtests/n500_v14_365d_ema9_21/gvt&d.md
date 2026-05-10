# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 4630.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 51 |
| ALERT2 | 52 |
| ALERT2_SKIP | 34 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 50
- **Target hits / Stop hits / Partials:** 6 / 60 / 2
- **Avg / median % per leg:** -0.24% / -1.26%
- **Sum % (uncompounded):** -16.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 9 | 34.6% | 6 | 20 | 0 | 1.13% | 29.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 6 | 20 | 0 | 1.13% | 29.3% |
| SELL (all) | 42 | 9 | 21.4% | 0 | 40 | 2 | -1.09% | -45.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 9 | 21.4% | 0 | 40 | 2 | -1.09% | -45.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 18 | 26.5% | 6 | 60 | 2 | -0.24% | -16.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1794.00 | 1822.77 | 1824.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 1785.00 | 1815.22 | 1820.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1845.00 | 1820.98 | 1822.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1826.40 | 1822.06 | 1822.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1823.90 | 1822.71 | 1823.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 1849.90 | 1828.45 | 1825.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 1849.90 | 1828.45 | 1825.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1870.00 | 1843.37 | 1834.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2260.30 | 2274.98 | 2226.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 2246.00 | 2271.15 | 2242.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 2246.00 | 2271.15 | 2242.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 2246.00 | 2271.15 | 2242.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 2234.80 | 2263.88 | 2242.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 2309.20 | 2263.88 | 2242.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 2332.40 | 2364.69 | 2367.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 2332.40 | 2364.69 | 2367.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 2314.90 | 2354.74 | 2362.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 15:15:00 | 2313.60 | 2311.22 | 2330.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 2320.40 | 2311.22 | 2330.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2319.80 | 2312.93 | 2329.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 2319.80 | 2312.93 | 2329.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2397.20 | 2323.23 | 2325.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 2370.70 | 2323.23 | 2325.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2403.00 | 2339.18 | 2332.18 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 2290.50 | 2336.67 | 2341.04 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 2366.30 | 2319.90 | 2316.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 2387.00 | 2353.32 | 2336.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 2350.00 | 2367.75 | 2349.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 2370.00 | 2368.20 | 2351.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 2392.40 | 2365.06 | 2354.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2345.00 | 2354.60 | 2353.00 | SL hit (close<static) qty=1.00 sl=2347.60 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2326.40 | 2348.96 | 2350.58 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 2408.80 | 2350.72 | 2346.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 13:15:00 | 2434.00 | 2382.42 | 2363.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 2398.30 | 2406.20 | 2383.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 2398.30 | 2406.20 | 2383.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 2395.10 | 2403.98 | 2384.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:30:00 | 2413.90 | 2403.98 | 2384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 2382.60 | 2399.70 | 2383.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 2378.00 | 2399.70 | 2383.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 2387.00 | 2397.16 | 2384.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 2371.00 | 2397.16 | 2384.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2394.90 | 2396.71 | 2385.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 2390.00 | 2396.71 | 2385.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 2390.00 | 2395.37 | 2385.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 2415.60 | 2395.37 | 2385.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:45:00 | 2412.50 | 2395.96 | 2387.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 2400.00 | 2395.96 | 2387.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 2375.50 | 2391.87 | 2386.50 | SL hit (close<static) qty=1.00 sl=2381.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 2375.50 | 2391.87 | 2386.50 | SL hit (close<static) qty=1.00 sl=2381.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 2375.50 | 2391.87 | 2386.50 | SL hit (close<static) qty=1.00 sl=2381.20 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 2354.60 | 2381.65 | 2382.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 14:15:00 | 2347.60 | 2374.84 | 2379.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 2362.70 | 2368.87 | 2375.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2362.70 | 2367.64 | 2373.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:45:00 | 2370.20 | 2367.64 | 2373.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2362.70 | 2354.14 | 2363.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 2362.70 | 2354.14 | 2363.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 2353.00 | 2353.91 | 2362.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2349.50 | 2357.26 | 2361.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 2348.90 | 2358.84 | 2361.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 2348.70 | 2357.07 | 2360.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 2345.60 | 2354.21 | 2358.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2355.00 | 2354.37 | 2357.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 2387.20 | 2370.18 | 2365.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 2344.30 | 2360.53 | 2362.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 2332.60 | 2349.77 | 2355.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 2357.60 | 2347.10 | 2353.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 2358.70 | 2349.42 | 2353.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 2370.00 | 2349.42 | 2353.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2359.10 | 2351.35 | 2354.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 2345.20 | 2351.35 | 2354.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 2350.00 | 2351.35 | 2353.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 2349.40 | 2350.60 | 2353.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2372.20 | 2350.31 | 2350.72 | SL hit (close>static) qty=1.00 sl=2364.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2372.20 | 2350.31 | 2350.72 | SL hit (close>static) qty=1.00 sl=2364.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2372.20 | 2350.31 | 2350.72 | SL hit (close>static) qty=1.00 sl=2364.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2327.80 | 2350.27 | 2350.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2350.00 | 2350.21 | 2350.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 2350.00 | 2350.21 | 2350.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-08 12:15:00 | 2383.90 | 2356.95 | 2353.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 2383.90 | 2356.95 | 2353.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2424.10 | 2381.47 | 2367.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 2387.30 | 2388.40 | 2375.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:45:00 | 2394.00 | 2388.40 | 2375.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2372.00 | 2384.73 | 2377.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 2372.00 | 2384.73 | 2377.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2400.00 | 2387.79 | 2379.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 2407.20 | 2390.93 | 2381.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 2367.00 | 2383.56 | 2379.63 | SL hit (close<static) qty=1.00 sl=2370.90 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 2332.50 | 2373.31 | 2375.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 2320.30 | 2356.51 | 2367.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 2322.80 | 2299.84 | 2319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2340.40 | 2307.95 | 2321.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2340.40 | 2307.95 | 2321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2322.10 | 2310.78 | 2321.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 2332.90 | 2310.78 | 2321.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2322.00 | 2313.02 | 2321.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 2325.40 | 2313.02 | 2321.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2305.60 | 2311.54 | 2320.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 2319.40 | 2311.54 | 2320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2329.00 | 2311.39 | 2317.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2319.20 | 2311.39 | 2317.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2323.50 | 2313.81 | 2318.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 2325.50 | 2313.81 | 2318.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 2333.10 | 2313.36 | 2316.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 2333.10 | 2313.36 | 2316.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 2340.30 | 2318.75 | 2318.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2360.60 | 2328.82 | 2323.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 2375.00 | 2388.87 | 2363.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:45:00 | 2392.20 | 2388.87 | 2363.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2393.50 | 2389.79 | 2366.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 2407.90 | 2377.77 | 2368.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 2440.00 | 2474.00 | 2475.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 2440.00 | 2474.00 | 2475.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 2434.50 | 2460.66 | 2468.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 2456.40 | 2418.81 | 2430.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2476.80 | 2430.41 | 2434.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 2485.40 | 2430.41 | 2434.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 2475.20 | 2439.37 | 2437.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 2597.00 | 2476.76 | 2455.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 2795.50 | 2820.49 | 2742.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 2795.50 | 2820.49 | 2742.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 2759.80 | 2791.82 | 2747.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 15:15:00 | 2810.00 | 2782.87 | 2750.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 2787.90 | 2797.37 | 2798.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 2787.90 | 2797.37 | 2798.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 2751.90 | 2788.28 | 2794.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 2789.20 | 2786.85 | 2792.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 2799.90 | 2789.46 | 2793.08 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 2822.10 | 2795.99 | 2795.72 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 2790.00 | 2795.43 | 2795.55 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 2815.40 | 2799.43 | 2797.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 2914.60 | 2821.96 | 2809.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 2848.90 | 2885.70 | 2857.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2890.00 | 2886.56 | 2860.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 2911.90 | 2856.92 | 2853.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 2839.70 | 2852.35 | 2851.55 | SL hit (close<static) qty=1.00 sl=2848.90 alert=retest2 |

### Cycle 21 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 2840.00 | 2849.88 | 2850.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 14:15:00 | 2829.90 | 2844.29 | 2847.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 2827.10 | 2817.73 | 2828.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2792.00 | 2812.58 | 2825.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 2788.60 | 2807.07 | 2821.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:00:00 | 2785.00 | 2807.07 | 2821.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 2787.10 | 2801.61 | 2814.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 2789.80 | 2793.57 | 2807.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 2791.80 | 2793.21 | 2805.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 2812.50 | 2793.21 | 2805.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2787.20 | 2775.48 | 2790.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2718.60 | 2766.63 | 2776.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 2709.60 | 2745.17 | 2764.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 2723.70 | 2698.55 | 2712.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 2819.90 | 2778.26 | 2757.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 2763.60 | 2775.32 | 2757.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 2763.60 | 2775.32 | 2757.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 2783.20 | 2776.90 | 2760.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 2809.20 | 2773.48 | 2764.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 2756.10 | 2791.28 | 2783.49 | SL hit (close<static) qty=1.00 sl=2760.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 2729.10 | 2772.93 | 2776.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 2706.80 | 2752.35 | 2765.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 2756.20 | 2748.03 | 2761.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2761.20 | 2750.66 | 2761.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 2761.20 | 2750.66 | 2761.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 2777.40 | 2756.01 | 2762.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 2776.70 | 2756.01 | 2762.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 2760.00 | 2756.81 | 2762.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 15:00:00 | 2749.20 | 2755.30 | 2760.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 2752.10 | 2751.98 | 2758.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 2745.00 | 2749.90 | 2756.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 2751.90 | 2751.01 | 2755.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 2752.00 | 2751.21 | 2755.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 2737.10 | 2748.89 | 2754.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 2730.70 | 2749.24 | 2753.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 2828.00 | 2771.22 | 2757.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 2760.20 | 2786.45 | 2771.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2743.30 | 2777.82 | 2768.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 2743.30 | 2777.82 | 2768.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2760.50 | 2767.14 | 2765.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 2752.10 | 2767.14 | 2765.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2781.70 | 2770.05 | 2766.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 2789.40 | 2770.94 | 2767.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 2794.00 | 2773.87 | 2769.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 2786.00 | 2778.24 | 2772.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 10:15:00 | 3068.34 | 3001.25 | 2959.17 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-19 10:15:00 | 3064.60 | 3001.25 | 2959.17 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-19 11:15:00 | 3073.40 | 3016.80 | 2970.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 3006.30 | 3011.44 | 3011.94 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 3030.00 | 3010.39 | 3009.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 3030.30 | 3016.74 | 3012.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 2998.20 | 3013.03 | 3011.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 2978.10 | 3006.05 | 3008.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 2963.00 | 2997.44 | 3004.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 2946.60 | 2944.84 | 2966.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 2989.70 | 2944.84 | 2966.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2959.60 | 2947.79 | 2965.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 2997.40 | 2947.79 | 2965.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2966.30 | 2951.50 | 2965.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 2953.70 | 2951.50 | 2965.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 2986.00 | 2958.40 | 2967.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 2988.80 | 2958.40 | 2967.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 3000.00 | 2966.72 | 2970.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 3000.00 | 2966.72 | 2970.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 2988.70 | 2974.84 | 2973.63 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 2953.40 | 2972.30 | 2973.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 2945.40 | 2966.92 | 2970.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 2967.80 | 2967.09 | 2970.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 2967.80 | 2967.09 | 2970.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 2961.10 | 2965.90 | 2969.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 2944.60 | 2964.84 | 2968.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 3034.40 | 2975.51 | 2972.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3034.40 | 2975.51 | 2972.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 3070.00 | 3015.42 | 2995.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 13:15:00 | 3133.30 | 3146.13 | 3104.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:30:00 | 3129.10 | 3146.13 | 3104.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3135.30 | 3142.89 | 3113.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 3137.70 | 3142.89 | 3113.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 3135.00 | 3139.85 | 3119.45 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3065.10 | 3109.00 | 3110.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 3030.50 | 3082.73 | 3097.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 3049.50 | 3037.88 | 3061.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 3049.50 | 3037.88 | 3061.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3042.90 | 3039.29 | 3058.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 3049.00 | 3039.29 | 3058.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3044.50 | 3025.59 | 3039.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3033.60 | 3025.59 | 3039.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 3040.00 | 3028.47 | 3039.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 3022.50 | 3034.38 | 3040.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2968.20 | 3031.79 | 3038.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 13:15:00 | 2962.30 | 2967.16 | 2967.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 2956.00 | 2964.93 | 2966.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 2910.00 | 2897.88 | 2927.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 2910.10 | 2900.32 | 2925.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 2893.50 | 2900.04 | 2923.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 2900.50 | 2903.98 | 2921.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:45:00 | 2902.00 | 2903.52 | 2919.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 2873.20 | 2897.12 | 2913.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2927.70 | 2871.15 | 2886.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 2932.70 | 2871.15 | 2886.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | SL hit (close>static) qty=1.00 sl=2929.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | SL hit (close>static) qty=1.00 sl=2929.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | SL hit (close>static) qty=1.00 sl=2929.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | SL hit (close>static) qty=1.00 sl=2929.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 2958.60 | 2888.64 | 2893.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 2950.00 | 2900.91 | 2898.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 13:15:00 | 2968.60 | 2922.27 | 2909.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 2933.50 | 2943.78 | 2925.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 2933.50 | 2943.78 | 2925.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 3034.30 | 3051.27 | 3025.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 3112.80 | 3044.26 | 3028.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 3073.60 | 3117.32 | 3106.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 3039.30 | 3086.15 | 3093.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3093.90 | 3058.60 | 3075.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3071.70 | 3061.22 | 3074.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 3087.20 | 3061.22 | 3074.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3098.60 | 3068.70 | 3077.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 3098.60 | 3068.70 | 3077.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3104.10 | 3075.78 | 3079.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 3090.50 | 3075.42 | 3079.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 3117.50 | 3084.07 | 3082.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3117.50 | 3084.07 | 3082.14 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 3080.40 | 3114.32 | 3116.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 3051.60 | 3092.90 | 3105.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 3110.00 | 3069.70 | 3088.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3065.60 | 3068.88 | 3086.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:15:00 | 3108.90 | 3068.88 | 3086.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 3069.30 | 3068.96 | 3084.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 3074.90 | 3068.96 | 3084.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2997.80 | 3008.57 | 3034.31 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 3068.30 | 3043.25 | 3040.79 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2976.70 | 3044.39 | 3045.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 2943.00 | 3024.11 | 3036.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 3002.30 | 2956.27 | 2988.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 2974.80 | 2959.98 | 2986.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 3003.00 | 2959.98 | 2986.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 2972.00 | 2962.38 | 2985.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 2987.40 | 2962.38 | 2985.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 2989.30 | 2969.34 | 2984.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:45:00 | 2988.80 | 2969.34 | 2984.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2867.60 | 2948.99 | 2974.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2843.00 | 2948.99 | 2974.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 2998.30 | 2942.99 | 2937.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 2998.30 | 2942.99 | 2937.22 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 2884.50 | 2928.96 | 2932.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2880.00 | 2901.42 | 2915.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 2902.00 | 2897.96 | 2911.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:30:00 | 2907.40 | 2897.96 | 2911.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2875.30 | 2890.96 | 2903.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 2892.20 | 2890.96 | 2903.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2873.80 | 2886.59 | 2899.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 2832.00 | 2874.82 | 2891.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2841.80 | 2834.40 | 2863.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 2827.90 | 2868.28 | 2870.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 2819.50 | 2853.25 | 2863.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 2760.00 | 2744.59 | 2774.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 2836.30 | 2764.67 | 2776.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 2836.30 | 2764.67 | 2776.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2865.10 | 2784.75 | 2784.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 2895.50 | 2806.90 | 2794.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 2899.10 | 2903.68 | 2863.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 2899.10 | 2903.68 | 2863.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2914.20 | 2905.78 | 2867.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 2935.80 | 2913.51 | 2886.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 2959.60 | 2988.00 | 2989.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 2959.60 | 2988.00 | 2989.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 2846.40 | 2950.60 | 2970.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 2931.30 | 2894.08 | 2920.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2897.50 | 2894.76 | 2918.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 2880.60 | 2894.76 | 2918.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 3085.30 | 2939.40 | 2930.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3085.30 | 2939.40 | 2930.77 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 3065.00 | 3096.24 | 3097.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 3035.40 | 3070.09 | 3081.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 3074.60 | 3073.08 | 3081.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 3071.10 | 3072.68 | 3080.85 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 3135.60 | 3089.60 | 3086.84 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 3081.40 | 3124.30 | 3125.53 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 3171.20 | 3106.31 | 3102.35 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3087.10 | 3118.26 | 3118.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3009.10 | 3096.43 | 3108.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 2813.80 | 2755.09 | 2803.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2802.70 | 2764.61 | 2803.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 2787.30 | 2771.31 | 2803.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 2747.90 | 2778.83 | 2796.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 2647.93 | 2693.49 | 2742.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 2610.51 | 2675.01 | 2724.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 2575.10 | 2571.98 | 2607.73 | SL hit (close>ema200) qty=0.50 sl=2571.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 2575.10 | 2571.98 | 2607.73 | SL hit (close>ema200) qty=0.50 sl=2571.98 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 2646.70 | 2622.79 | 2619.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 2746.40 | 2652.67 | 2634.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 2700.00 | 2704.70 | 2674.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 2704.90 | 2704.70 | 2674.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2720.00 | 2707.76 | 2678.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:00:00 | 2733.40 | 2708.66 | 2689.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-29 09:15:00 | 3006.74 | 2880.93 | 2804.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3564.20 | 3627.06 | 3632.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3552.30 | 3595.39 | 3613.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 3694.60 | 3610.48 | 3616.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 3687.10 | 3625.80 | 3623.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 3702.60 | 3660.83 | 3643.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 3643.20 | 3657.31 | 3643.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 3641.00 | 3654.05 | 3643.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:15:00 | 3626.60 | 3654.05 | 3643.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 3636.00 | 3650.44 | 3642.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 3621.70 | 3650.44 | 3642.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 3628.80 | 3646.11 | 3641.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:45:00 | 3626.90 | 3646.11 | 3641.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 3647.80 | 3646.45 | 3642.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 3655.00 | 3646.45 | 3642.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:45:00 | 3665.50 | 3650.96 | 3645.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 3566.50 | 3674.29 | 3665.72 | SL hit (close<static) qty=1.00 sl=3625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 3566.50 | 3674.29 | 3665.72 | SL hit (close<static) qty=1.00 sl=3625.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 3563.00 | 3652.04 | 3656.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 3542.40 | 3630.11 | 3646.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 3579.60 | 3569.33 | 3601.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 3579.60 | 3569.33 | 3601.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 3622.40 | 3579.94 | 3603.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 3642.90 | 3579.94 | 3603.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3661.10 | 3596.17 | 3608.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 3662.00 | 3596.17 | 3608.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 3729.40 | 3637.54 | 3625.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3824.10 | 3754.01 | 3715.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 3842.30 | 3849.22 | 3808.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 3865.10 | 3852.48 | 3829.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 3865.10 | 3852.48 | 3829.81 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 3775.00 | 3820.33 | 3821.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3742.10 | 3791.38 | 3806.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 3818.20 | 3775.32 | 3770.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 3838.00 | 3796.43 | 3781.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 3697.00 | 3815.78 | 3816.43 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3878.60 | 3809.72 | 3807.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 3885.60 | 3824.89 | 3814.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 3794.80 | 3835.93 | 3827.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 3779.90 | 3824.73 | 3823.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 3779.90 | 3824.73 | 3823.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 3756.90 | 3811.16 | 3817.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 3749.90 | 3782.49 | 3800.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3619.00 | 3543.16 | 3597.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3668.30 | 3568.18 | 3604.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 3668.30 | 3568.18 | 3604.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 3663.20 | 3625.82 | 3625.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3769.70 | 3668.41 | 3646.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 3766.80 | 3779.22 | 3739.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 3766.80 | 3779.22 | 3739.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 3768.40 | 3777.05 | 3742.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 3734.40 | 3777.05 | 3742.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 3760.00 | 3773.64 | 3743.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 3846.30 | 3773.64 | 3743.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 3722.70 | 3763.45 | 3741.85 | SL hit (close<static) qty=1.00 sl=3742.50 alert=retest2 |

### Cycle 63 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 3670.20 | 3729.12 | 3732.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3519.00 | 3680.04 | 3709.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 3506.70 | 3515.97 | 3585.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 3501.10 | 3506.74 | 3563.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 3694.40 | 3548.93 | 3568.80 | SL hit (close>static) qty=1.00 sl=3627.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 3694.40 | 3548.93 | 3568.80 | SL hit (close>static) qty=1.00 sl=3627.70 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 3691.40 | 3595.19 | 3587.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 3720.60 | 3620.28 | 3599.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 3859.80 | 3685.63 | 3682.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 11:15:00 | 4245.78 | 3815.17 | 3749.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 15:00:00 | 3799.80 | 3810.53 | 3763.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 3820.00 | 3759.62 | 3752.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 3792.50 | 3783.65 | 3765.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 3771.00 | 3781.12 | 3766.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 3884.00 | 3781.12 | 3766.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 10:15:00 | 3686.90 | 3732.95 | 3747.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 3793.90 | 3740.79 | 3735.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 3879.80 | 3766.44 | 3748.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 4097.30 | 4099.18 | 4020.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 4097.30 | 4099.18 | 4020.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 4067.80 | 4116.33 | 4079.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 4067.80 | 4116.33 | 4079.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 4063.40 | 4105.74 | 4077.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:15:00 | 4060.30 | 4105.74 | 4077.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 4065.00 | 4097.59 | 4076.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 4060.00 | 4097.59 | 4076.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 4071.20 | 4084.30 | 4074.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 4071.20 | 4084.30 | 4074.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 4070.00 | 4081.44 | 4074.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 4135.20 | 4081.44 | 4074.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 4548.72 | 4308.09 | 4256.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 4494.70 | 4520.85 | 4521.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 4483.40 | 4508.06 | 4515.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 4518.90 | 4504.90 | 4512.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 4538.00 | 4511.52 | 4514.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 4550.10 | 4511.52 | 4514.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 4529.10 | 4514.78 | 4515.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 4543.80 | 4514.78 | 4515.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4500.00 | 4511.82 | 4514.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 4490.70 | 4511.82 | 4514.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 4478.50 | 4485.72 | 4494.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 4718.20 | 4603.35 | 4572.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 4668.00 | 4707.98 | 4653.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 4633.80 | 4693.14 | 4651.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 4633.80 | 4693.14 | 4651.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 4630.00 | 4680.51 | 4649.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 4636.00 | 4680.51 | 4649.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 4638.60 | 4672.13 | 4648.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 4634.90 | 4672.13 | 4648.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 13:15:00 | 1823.90 | 2025-05-21 14:15:00 | 1849.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-02 09:15:00 | 2309.20 | 2025-06-06 13:15:00 | 2332.40 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-06-20 09:30:00 | 2392.40 | 2025-06-20 15:15:00 | 2345.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-26 09:15:00 | 2415.60 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-26 10:45:00 | 2412.50 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-26 11:15:00 | 2400.00 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-01 09:15:00 | 2349.50 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2348.90 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-01 14:15:00 | 2348.70 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-02 09:15:00 | 2345.60 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-04 15:15:00 | 2345.20 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-07 10:30:00 | 2350.00 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-07 11:45:00 | 2349.40 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2327.80 | 2025-07-08 12:15:00 | 2383.90 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-10 12:15:00 | 2407.20 | 2025-07-10 13:15:00 | 2367.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-21 10:00:00 | 2407.90 | 2025-07-25 12:15:00 | 2440.00 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-08-04 15:15:00 | 2810.00 | 2025-08-07 15:15:00 | 2787.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-18 09:30:00 | 2911.90 | 2025-08-18 11:15:00 | 2839.70 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-20 10:30:00 | 2788.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-08-20 11:00:00 | 2785.00 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-08-20 14:30:00 | 2787.10 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-21 10:45:00 | 2789.80 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-08-26 09:15:00 | 2718.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-08-26 11:00:00 | 2709.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-08-29 10:45:00 | 2723.70 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-09-03 09:45:00 | 2809.20 | 2025-09-04 10:15:00 | 2756.10 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-05 15:00:00 | 2749.20 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-08 09:45:00 | 2752.10 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-08 10:30:00 | 2745.00 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-08 13:15:00 | 2751.90 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-08 14:45:00 | 2737.10 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-09 10:15:00 | 2730.70 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-12 13:45:00 | 2789.40 | 2025-09-19 10:15:00 | 3068.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 14:45:00 | 2794.00 | 2025-09-19 10:15:00 | 3064.60 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2025-09-15 10:00:00 | 2786.00 | 2025-09-19 11:15:00 | 3073.40 | TARGET_HIT | 1.00 | 10.32% |
| SELL | retest2 | 2025-09-30 15:15:00 | 2944.60 | 2025-10-01 09:15:00 | 3034.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-10-13 14:00:00 | 3022.50 | 2025-10-20 09:15:00 | 3004.10 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-10-14 09:15:00 | 2968.20 | 2025-10-20 09:15:00 | 3004.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-24 11:30:00 | 2893.50 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-10-24 13:30:00 | 2900.50 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-24 14:45:00 | 2902.00 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-27 09:45:00 | 2873.20 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-11-03 09:15:00 | 3112.80 | 2025-11-06 11:15:00 | 3059.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-06 09:30:00 | 3073.60 | 2025-11-06 11:15:00 | 3059.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-07 13:30:00 | 3090.50 | 2025-11-10 09:15:00 | 3117.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-24 15:15:00 | 2843.00 | 2025-11-26 14:15:00 | 2998.30 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2025-12-01 11:45:00 | 2832.00 | 2025-12-03 10:15:00 | 2900.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2841.80 | 2025-12-03 10:15:00 | 2900.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-12-11 13:30:00 | 2935.80 | 2025-12-17 11:15:00 | 2959.60 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-12-19 12:15:00 | 2880.60 | 2025-12-22 09:15:00 | 3085.30 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2787.30 | 2026-01-16 14:15:00 | 2647.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2747.90 | 2026-01-19 09:15:00 | 2610.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2787.30 | 2026-01-21 14:15:00 | 2575.10 | STOP_HIT | 0.50 | 7.61% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2747.90 | 2026-01-21 14:15:00 | 2575.10 | STOP_HIT | 0.50 | 6.29% |
| BUY | retest2 | 2026-01-27 15:00:00 | 2733.40 | 2026-01-29 09:15:00 | 3006.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-17 15:15:00 | 3655.00 | 2026-02-19 09:15:00 | 3566.50 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-02-18 10:45:00 | 3665.50 | 2026-02-19 09:15:00 | 3566.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-03-20 09:15:00 | 3846.30 | 2026-03-20 09:15:00 | 3722.70 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2026-03-24 10:30:00 | 3506.70 | 2026-03-25 09:15:00 | 3694.40 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-03-24 13:45:00 | 3501.10 | 2026-03-25 09:15:00 | 3694.40 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3859.80 | 2026-04-01 11:15:00 | 4245.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-01 15:00:00 | 3799.80 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-02 13:15:00 | 3820.00 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-04-02 14:30:00 | 3792.50 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-04-06 09:15:00 | 3884.00 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-04-17 09:15:00 | 4135.20 | 2026-04-23 09:15:00 | 4548.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 14:15:00 | 4490.70 | 2026-05-05 09:15:00 | 4575.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-05-04 15:00:00 | 4478.50 | 2026-05-05 09:15:00 | 4575.50 | STOP_HIT | 1.00 | -2.17% |
