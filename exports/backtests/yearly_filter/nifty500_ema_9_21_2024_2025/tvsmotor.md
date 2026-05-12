# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 3701.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 152 |
| ALERT1 | 114 |
| ALERT2 | 113 |
| ALERT2_SKIP | 66 |
| ALERT3 | 296 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 123 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 39 / 93
- **Target hits / Stop hits / Partials:** 1 / 127 / 4
- **Avg / median % per leg:** -0.39% / -0.95%
- **Sum % (uncompounded):** -51.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 19 | 27.5% | 1 | 68 | 0 | -0.48% | -33.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.87% | -2.6% |
| BUY @ 3rd Alert (retest2) | 66 | 19 | 28.8% | 1 | 65 | 0 | -0.47% | -30.7% |
| SELL (all) | 63 | 20 | 31.7% | 0 | 59 | 4 | -0.29% | -18.5% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.64% | 3.2% |
| SELL @ 3rd Alert (retest2) | 58 | 18 | 31.0% | 0 | 55 | 3 | -0.37% | -21.7% |
| retest1 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.08% | 0.6% |
| retest2 (combined) | 124 | 37 | 29.8% | 1 | 120 | 3 | -0.42% | -52.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 2129.50 | 2149.77 | 2150.37 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 2160.00 | 2148.80 | 2148.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 2168.85 | 2156.17 | 2152.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 2238.70 | 2239.71 | 2217.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:45:00 | 2267.80 | 2245.29 | 2222.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 2231.00 | 2245.02 | 2232.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 2231.00 | 2245.02 | 2232.64 | SL hit (close<ema400) qty=1.00 sl=2232.64 alert=retest1 |

### Cycle 3 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 2204.45 | 2231.39 | 2234.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 10:15:00 | 2186.90 | 2222.49 | 2230.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 2242.50 | 2204.62 | 2214.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 2242.50 | 2204.62 | 2214.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2242.50 | 2204.62 | 2214.61 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 2242.70 | 2220.47 | 2220.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 2246.80 | 2225.74 | 2222.51 | Break + close above crossover candle high |

### Cycle 5 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2163.70 | 2221.20 | 2222.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 2107.00 | 2198.36 | 2212.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 2210.25 | 2199.77 | 2210.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 2210.25 | 2199.77 | 2210.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 2210.25 | 2199.77 | 2210.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:00:00 | 2210.25 | 2199.77 | 2210.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 2221.95 | 2204.20 | 2211.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 2221.95 | 2204.20 | 2211.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 2225.05 | 2208.37 | 2212.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 2220.15 | 2208.37 | 2212.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 2296.90 | 2226.08 | 2220.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 2342.55 | 2249.37 | 2231.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 2426.65 | 2432.53 | 2417.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 2426.65 | 2432.53 | 2417.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 2426.65 | 2432.53 | 2417.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 2456.00 | 2434.84 | 2426.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 2452.30 | 2441.39 | 2432.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 2461.45 | 2466.20 | 2462.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:00:00 | 2449.90 | 2461.31 | 2461.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 2434.80 | 2456.00 | 2458.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 2434.80 | 2456.00 | 2458.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 10:15:00 | 2427.00 | 2442.69 | 2450.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 2428.20 | 2422.78 | 2437.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 15:00:00 | 2428.20 | 2422.78 | 2437.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 2437.00 | 2425.62 | 2437.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 2458.00 | 2425.62 | 2437.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 2452.35 | 2430.97 | 2438.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 2449.05 | 2430.97 | 2438.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 2439.30 | 2432.63 | 2438.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 2432.25 | 2437.38 | 2439.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 2444.20 | 2440.75 | 2440.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 2444.20 | 2440.75 | 2440.51 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 2432.50 | 2438.95 | 2439.73 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 2482.50 | 2447.66 | 2443.62 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 2433.70 | 2441.49 | 2441.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 2412.85 | 2433.31 | 2437.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 2354.00 | 2351.28 | 2369.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 14:00:00 | 2354.00 | 2351.28 | 2369.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 2360.95 | 2353.21 | 2368.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 2360.95 | 2353.21 | 2368.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 2372.70 | 2358.20 | 2368.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:15:00 | 2384.00 | 2358.20 | 2368.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 2366.95 | 2359.95 | 2368.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 2384.00 | 2359.95 | 2368.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 2375.00 | 2362.96 | 2368.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 2375.00 | 2362.96 | 2368.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 2371.00 | 2364.57 | 2368.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:45:00 | 2374.70 | 2364.57 | 2368.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 2366.00 | 2362.26 | 2366.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 2366.00 | 2362.26 | 2366.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 2360.00 | 2361.81 | 2365.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 2359.20 | 2358.05 | 2363.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 2372.00 | 2350.46 | 2356.35 | SL hit (close>static) qty=1.00 sl=2369.40 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 2365.45 | 2346.47 | 2345.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 2404.00 | 2361.58 | 2352.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 2406.05 | 2407.59 | 2387.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 2406.05 | 2407.59 | 2387.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 2391.80 | 2404.44 | 2387.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 2391.80 | 2404.44 | 2387.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 2392.70 | 2403.90 | 2391.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 2392.70 | 2403.90 | 2391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 2400.50 | 2403.22 | 2392.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 10:30:00 | 2419.00 | 2407.40 | 2396.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 13:15:00 | 2438.60 | 2443.96 | 2444.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 13:15:00 | 2438.60 | 2443.96 | 2444.21 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 2450.80 | 2445.33 | 2444.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 2462.00 | 2452.31 | 2448.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 2436.50 | 2452.28 | 2450.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 2436.50 | 2452.28 | 2450.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2436.50 | 2452.28 | 2450.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 2409.00 | 2452.28 | 2450.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 2404.40 | 2442.70 | 2446.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 11:15:00 | 2400.00 | 2434.16 | 2441.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 15:15:00 | 2422.15 | 2417.53 | 2429.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:15:00 | 2391.20 | 2417.53 | 2429.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:45:00 | 2396.00 | 2411.06 | 2425.88 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 15:15:00 | 2398.00 | 2401.40 | 2414.22 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2436.00 | 2407.77 | 2414.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 2436.00 | 2407.77 | 2414.86 | SL hit (close>ema400) qty=1.00 sl=2414.86 alert=retest1 |

### Cycle 16 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 2439.00 | 2419.27 | 2417.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 2461.85 | 2427.78 | 2421.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 2441.90 | 2448.57 | 2435.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 2441.90 | 2448.57 | 2435.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2441.90 | 2448.57 | 2435.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 2438.25 | 2448.57 | 2435.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2427.90 | 2444.44 | 2435.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 2437.55 | 2444.44 | 2435.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 2439.60 | 2443.47 | 2435.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:15:00 | 2426.00 | 2443.47 | 2435.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 2437.20 | 2442.22 | 2435.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:30:00 | 2422.65 | 2442.22 | 2435.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 2440.00 | 2441.77 | 2436.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:00:00 | 2440.00 | 2441.77 | 2436.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 2449.50 | 2443.32 | 2437.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 15:15:00 | 2456.25 | 2443.32 | 2437.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 2431.95 | 2443.11 | 2438.42 | SL hit (close<static) qty=1.00 sl=2437.10 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 2479.05 | 2542.12 | 2546.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 2446.40 | 2496.01 | 2512.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 2548.10 | 2499.61 | 2509.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 2548.10 | 2499.61 | 2509.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 2548.10 | 2499.61 | 2509.37 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 2555.20 | 2522.24 | 2518.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 2576.45 | 2545.93 | 2531.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 2541.95 | 2549.50 | 2535.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 2541.95 | 2549.50 | 2535.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 2541.95 | 2549.50 | 2535.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 2541.95 | 2549.50 | 2535.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 2526.95 | 2544.99 | 2535.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 2526.95 | 2544.99 | 2535.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 2512.60 | 2538.51 | 2533.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 2512.60 | 2538.51 | 2533.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 2523.25 | 2535.46 | 2532.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 2553.40 | 2531.46 | 2530.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 2558.15 | 2563.28 | 2553.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 2589.40 | 2598.51 | 2599.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 10:15:00 | 2589.40 | 2598.51 | 2599.05 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 2605.00 | 2599.81 | 2599.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 2612.80 | 2605.85 | 2602.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 2611.00 | 2624.70 | 2617.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 2611.00 | 2624.70 | 2617.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 2611.00 | 2624.70 | 2617.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 2638.70 | 2625.50 | 2620.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 2734.25 | 2743.50 | 2743.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 2734.25 | 2743.50 | 2743.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 2733.10 | 2741.42 | 2742.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 2743.00 | 2741.74 | 2742.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 2743.00 | 2741.74 | 2742.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 2743.00 | 2741.74 | 2742.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 11:00:00 | 2737.95 | 2740.98 | 2742.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:45:00 | 2738.00 | 2737.56 | 2740.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 15:15:00 | 2762.35 | 2743.37 | 2742.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 2762.35 | 2743.37 | 2742.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 2779.60 | 2750.61 | 2745.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 2784.30 | 2786.75 | 2771.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 2784.30 | 2786.75 | 2771.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 2795.60 | 2789.85 | 2778.89 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 2756.55 | 2778.68 | 2779.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 11:15:00 | 2754.45 | 2773.84 | 2776.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 2782.90 | 2772.19 | 2775.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 14:15:00 | 2782.90 | 2772.19 | 2775.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 2782.90 | 2772.19 | 2775.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 2782.90 | 2772.19 | 2775.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 2780.95 | 2773.94 | 2775.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:30:00 | 2767.30 | 2774.00 | 2775.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:15:00 | 2766.15 | 2774.92 | 2775.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:45:00 | 2767.25 | 2752.30 | 2755.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 2769.00 | 2758.76 | 2757.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 2769.00 | 2758.76 | 2757.62 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 2755.70 | 2756.86 | 2756.92 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 2764.15 | 2758.32 | 2757.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 2775.00 | 2762.95 | 2759.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 2751.10 | 2760.58 | 2759.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 2751.10 | 2760.58 | 2759.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 2751.10 | 2760.58 | 2759.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 2751.10 | 2760.58 | 2759.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 2760.90 | 2760.64 | 2759.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 2760.90 | 2760.64 | 2759.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 2759.90 | 2760.49 | 2759.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:30:00 | 2776.50 | 2767.48 | 2762.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 2791.00 | 2809.41 | 2810.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 2791.00 | 2809.41 | 2810.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2780.40 | 2803.61 | 2808.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2772.65 | 2768.31 | 2784.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 2772.65 | 2768.31 | 2784.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 2770.70 | 2768.79 | 2783.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 2783.80 | 2768.79 | 2783.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2788.40 | 2772.71 | 2783.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 2797.75 | 2772.71 | 2783.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 2803.65 | 2778.90 | 2785.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 2808.00 | 2778.90 | 2785.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 2817.10 | 2786.54 | 2788.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 2817.10 | 2786.54 | 2788.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 2806.65 | 2790.56 | 2790.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 2821.00 | 2805.32 | 2798.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 2852.05 | 2859.32 | 2846.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 2852.05 | 2859.32 | 2846.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 2858.00 | 2857.47 | 2847.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 2874.50 | 2860.88 | 2849.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:00:00 | 2879.00 | 2860.73 | 2854.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 10:00:00 | 2883.20 | 2903.17 | 2885.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 2817.10 | 2866.68 | 2872.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 2817.10 | 2866.68 | 2872.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 2815.10 | 2847.92 | 2860.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 15:15:00 | 2844.45 | 2839.00 | 2850.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:15:00 | 2764.00 | 2839.00 | 2850.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 2625.80 | 2689.26 | 2733.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 2671.45 | 2658.00 | 2691.73 | SL hit (close>ema200) qty=0.50 sl=2658.00 alert=retest1 |

### Cycle 30 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 2745.20 | 2707.77 | 2705.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 2775.25 | 2721.27 | 2712.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 2787.00 | 2795.38 | 2774.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 2783.45 | 2795.38 | 2774.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2779.95 | 2792.29 | 2774.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:15:00 | 2766.55 | 2792.29 | 2774.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2781.70 | 2790.18 | 2775.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 2774.30 | 2790.18 | 2775.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 2825.50 | 2816.97 | 2803.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:30:00 | 2826.90 | 2819.66 | 2806.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:30:00 | 2834.60 | 2823.34 | 2810.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 2793.20 | 2818.56 | 2812.62 | SL hit (close<static) qty=1.00 sl=2800.65 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 2774.15 | 2803.27 | 2806.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 2618.05 | 2754.05 | 2781.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 2719.75 | 2702.43 | 2734.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:45:00 | 2722.00 | 2702.43 | 2734.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 2747.15 | 2711.38 | 2735.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 2744.05 | 2711.38 | 2735.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 2727.00 | 2714.50 | 2734.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 2710.80 | 2715.04 | 2730.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 2692.15 | 2715.63 | 2728.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 2751.55 | 2720.55 | 2728.73 | SL hit (close>static) qty=1.00 sl=2748.15 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 2749.30 | 2723.96 | 2720.52 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 2526.80 | 2686.18 | 2704.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 2482.00 | 2542.48 | 2610.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 2449.90 | 2443.05 | 2515.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 15:00:00 | 2449.90 | 2443.05 | 2515.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 2467.40 | 2447.48 | 2462.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:30:00 | 2481.20 | 2447.48 | 2462.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 2473.30 | 2452.64 | 2463.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 2473.30 | 2452.64 | 2463.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 2467.55 | 2455.62 | 2463.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 15:00:00 | 2457.65 | 2460.28 | 2464.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 10:15:00 | 2483.85 | 2465.82 | 2465.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 2483.85 | 2465.82 | 2465.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 2497.75 | 2477.71 | 2471.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 2444.40 | 2482.53 | 2477.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 2444.40 | 2482.53 | 2477.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2444.40 | 2482.53 | 2477.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 2450.95 | 2482.53 | 2477.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2419.00 | 2469.83 | 2471.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 2414.10 | 2449.88 | 2461.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 10:15:00 | 2437.15 | 2433.49 | 2447.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 11:00:00 | 2437.15 | 2433.49 | 2447.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 2444.50 | 2433.22 | 2443.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 2444.50 | 2433.22 | 2443.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 2456.75 | 2437.93 | 2444.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 2456.75 | 2437.93 | 2444.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 2461.10 | 2442.56 | 2446.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 2492.00 | 2442.56 | 2446.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 2502.35 | 2454.52 | 2451.50 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 2454.95 | 2470.47 | 2470.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 2448.55 | 2460.86 | 2465.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 2469.90 | 2462.67 | 2466.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 11:15:00 | 2469.90 | 2462.67 | 2466.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 2469.90 | 2462.67 | 2466.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:00:00 | 2469.90 | 2462.67 | 2466.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2468.35 | 2463.80 | 2466.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:30:00 | 2460.60 | 2462.87 | 2465.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:00:00 | 2459.15 | 2462.87 | 2465.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 2426.85 | 2417.41 | 2416.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 2426.85 | 2417.41 | 2416.84 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 2405.55 | 2414.80 | 2415.74 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 2446.20 | 2421.08 | 2418.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 2454.90 | 2427.85 | 2421.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 2424.65 | 2439.53 | 2430.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 2424.65 | 2439.53 | 2430.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 2424.65 | 2439.53 | 2430.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 2424.65 | 2439.53 | 2430.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 2421.00 | 2435.82 | 2429.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 2417.70 | 2435.82 | 2429.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2410.90 | 2430.84 | 2428.07 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 2412.15 | 2425.21 | 2425.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 12:15:00 | 2400.75 | 2420.32 | 2423.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 2398.25 | 2397.90 | 2408.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:00:00 | 2398.25 | 2397.90 | 2408.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 2408.25 | 2399.97 | 2408.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 2408.25 | 2399.97 | 2408.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 2421.95 | 2404.37 | 2410.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 2421.95 | 2404.37 | 2410.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 2415.75 | 2406.64 | 2410.59 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 2465.00 | 2420.02 | 2416.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 2485.85 | 2433.19 | 2422.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 2425.00 | 2450.18 | 2438.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 2425.00 | 2450.18 | 2438.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 2425.00 | 2450.18 | 2438.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 2413.05 | 2450.18 | 2438.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 2432.75 | 2446.70 | 2438.26 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 2424.95 | 2433.02 | 2433.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 10:15:00 | 2422.15 | 2429.48 | 2432.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 2431.15 | 2428.39 | 2430.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 12:15:00 | 2431.15 | 2428.39 | 2430.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 2431.15 | 2428.39 | 2430.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:00:00 | 2431.15 | 2428.39 | 2430.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 2431.00 | 2428.91 | 2430.99 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 2442.45 | 2433.76 | 2432.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 2449.55 | 2436.92 | 2434.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 2419.90 | 2434.14 | 2433.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 12:15:00 | 2419.90 | 2434.14 | 2433.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 2419.90 | 2434.14 | 2433.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 2419.90 | 2434.14 | 2433.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 2419.40 | 2431.19 | 2432.56 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 15:15:00 | 2442.05 | 2432.08 | 2431.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 2463.60 | 2438.39 | 2434.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 2518.95 | 2533.10 | 2509.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 2518.95 | 2533.10 | 2509.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2518.95 | 2533.10 | 2509.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 2518.95 | 2533.10 | 2509.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2490.05 | 2519.43 | 2511.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 2486.00 | 2519.43 | 2511.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2491.85 | 2513.92 | 2509.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:15:00 | 2508.55 | 2507.30 | 2507.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 2514.90 | 2508.82 | 2507.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 13:15:00 | 2495.80 | 2510.28 | 2511.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 2495.80 | 2510.28 | 2511.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 2489.50 | 2506.13 | 2509.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 10:15:00 | 2511.95 | 2503.79 | 2507.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 10:15:00 | 2511.95 | 2503.79 | 2507.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 2511.95 | 2503.79 | 2507.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 2511.95 | 2503.79 | 2507.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 2505.60 | 2504.15 | 2507.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:30:00 | 2516.10 | 2504.15 | 2507.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 2520.40 | 2507.16 | 2508.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 2520.40 | 2507.16 | 2508.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 2520.15 | 2509.76 | 2509.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 2525.00 | 2513.65 | 2511.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 2506.90 | 2521.11 | 2517.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 2506.90 | 2521.11 | 2517.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 2506.90 | 2521.11 | 2517.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 2506.90 | 2521.11 | 2517.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2502.25 | 2517.34 | 2515.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 2502.25 | 2517.34 | 2515.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 2524.75 | 2518.12 | 2516.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:45:00 | 2521.15 | 2518.12 | 2516.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 2482.90 | 2512.18 | 2514.14 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 2521.70 | 2514.45 | 2513.93 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 2494.45 | 2516.26 | 2516.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 2466.30 | 2502.45 | 2509.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 11:15:00 | 2476.75 | 2471.00 | 2486.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 12:15:00 | 2477.05 | 2471.00 | 2486.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 2478.20 | 2472.06 | 2483.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:30:00 | 2484.60 | 2472.06 | 2483.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 2484.00 | 2474.45 | 2483.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 2442.60 | 2474.45 | 2483.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 11:15:00 | 2438.20 | 2424.73 | 2423.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 2438.20 | 2424.73 | 2423.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 2477.00 | 2440.99 | 2432.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 2436.25 | 2442.45 | 2434.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 2436.25 | 2442.45 | 2434.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 2430.30 | 2440.02 | 2434.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 2430.30 | 2440.02 | 2434.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 2443.70 | 2440.75 | 2435.03 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 2385.35 | 2424.12 | 2428.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 2371.75 | 2396.94 | 2412.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 2369.00 | 2368.23 | 2386.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 2369.00 | 2368.23 | 2386.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 2376.05 | 2368.88 | 2381.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 2375.15 | 2368.88 | 2381.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 2380.50 | 2372.90 | 2380.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:15:00 | 2385.05 | 2372.90 | 2380.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 2394.15 | 2377.15 | 2382.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:00:00 | 2394.15 | 2377.15 | 2382.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 2406.55 | 2383.03 | 2384.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:30:00 | 2415.00 | 2383.03 | 2384.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 2402.10 | 2386.84 | 2385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 2427.10 | 2394.89 | 2389.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 2442.00 | 2472.81 | 2458.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 2442.00 | 2472.81 | 2458.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2442.00 | 2472.81 | 2458.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2442.00 | 2472.81 | 2458.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2446.60 | 2467.57 | 2457.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 2429.95 | 2467.57 | 2457.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 2411.90 | 2445.46 | 2448.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 10:15:00 | 2403.00 | 2427.14 | 2438.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 2221.70 | 2217.23 | 2260.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:30:00 | 2229.70 | 2217.23 | 2260.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 2255.85 | 2224.95 | 2259.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 2255.85 | 2224.95 | 2259.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2241.25 | 2232.68 | 2248.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:45:00 | 2234.10 | 2232.68 | 2248.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 2249.05 | 2235.96 | 2248.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 2252.10 | 2235.96 | 2248.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 2250.15 | 2238.79 | 2249.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 2250.15 | 2238.79 | 2249.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 2241.05 | 2239.25 | 2248.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:45:00 | 2247.15 | 2239.25 | 2248.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 2250.00 | 2241.28 | 2247.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:30:00 | 2252.75 | 2241.28 | 2247.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 2253.00 | 2243.62 | 2248.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 2290.55 | 2243.62 | 2248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 2297.80 | 2254.46 | 2252.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 15:15:00 | 2304.00 | 2292.45 | 2279.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 2250.20 | 2284.00 | 2276.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 2250.20 | 2284.00 | 2276.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 2250.20 | 2284.00 | 2276.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 2250.20 | 2284.00 | 2276.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 2285.50 | 2284.30 | 2277.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 2298.75 | 2284.30 | 2277.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:45:00 | 2296.25 | 2298.96 | 2292.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 12:15:00 | 2271.05 | 2288.31 | 2289.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 2271.05 | 2288.31 | 2289.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 2257.35 | 2282.12 | 2286.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2290.90 | 2280.16 | 2284.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 2290.90 | 2280.16 | 2284.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2290.90 | 2280.16 | 2284.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 2299.45 | 2280.16 | 2284.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2292.00 | 2282.53 | 2284.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 2296.15 | 2282.53 | 2284.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 2300.95 | 2288.45 | 2287.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 2304.85 | 2291.73 | 2288.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 15:15:00 | 2292.15 | 2293.24 | 2290.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 09:15:00 | 2286.50 | 2293.24 | 2290.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2270.80 | 2288.75 | 2288.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2270.80 | 2288.75 | 2288.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 2272.70 | 2285.54 | 2286.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 2251.70 | 2269.43 | 2277.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 2288.30 | 2249.20 | 2257.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 2288.30 | 2249.20 | 2257.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 2288.30 | 2249.20 | 2257.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 2288.30 | 2249.20 | 2257.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 2264.40 | 2252.24 | 2257.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 2264.40 | 2252.24 | 2257.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 2357.40 | 2273.27 | 2267.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 2514.20 | 2341.28 | 2301.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 2458.80 | 2458.86 | 2409.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:00:00 | 2458.80 | 2458.86 | 2409.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2461.70 | 2463.79 | 2445.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 2446.35 | 2463.79 | 2445.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2480.80 | 2469.69 | 2451.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 2480.80 | 2469.69 | 2451.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 2610.00 | 2638.43 | 2611.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 2610.00 | 2638.43 | 2611.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 2614.95 | 2633.73 | 2611.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:45:00 | 2613.70 | 2633.73 | 2611.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 2616.75 | 2627.38 | 2614.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 2616.75 | 2627.38 | 2614.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 2611.00 | 2624.11 | 2613.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 2605.00 | 2624.11 | 2613.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2590.95 | 2617.47 | 2611.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 2597.65 | 2617.47 | 2611.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 2612.60 | 2616.50 | 2611.94 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 2595.00 | 2609.36 | 2609.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 14:15:00 | 2592.95 | 2606.08 | 2608.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 2604.85 | 2603.17 | 2605.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 2604.85 | 2603.17 | 2605.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 2602.80 | 2603.10 | 2605.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:15:00 | 2595.80 | 2602.42 | 2604.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 2575.00 | 2602.74 | 2604.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 2466.01 | 2492.12 | 2529.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 2446.25 | 2492.12 | 2529.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 2487.25 | 2484.14 | 2515.70 | SL hit (close>ema200) qty=0.50 sl=2484.14 alert=retest2 |

### Cycle 62 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 2389.95 | 2384.49 | 2383.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 2402.05 | 2388.00 | 2385.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2343.45 | 2387.19 | 2386.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 2343.45 | 2387.19 | 2386.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2343.45 | 2387.19 | 2386.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2343.45 | 2387.19 | 2386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 2330.25 | 2375.80 | 2381.44 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 2365.00 | 2360.65 | 2360.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 13:15:00 | 2369.85 | 2362.49 | 2361.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 14:15:00 | 2361.90 | 2362.37 | 2361.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 14:15:00 | 2361.90 | 2362.37 | 2361.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 2361.90 | 2362.37 | 2361.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 2361.90 | 2362.37 | 2361.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 15:15:00 | 2350.00 | 2359.90 | 2360.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 2341.15 | 2356.15 | 2358.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 2342.20 | 2335.03 | 2345.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 2342.20 | 2335.03 | 2345.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 2342.20 | 2335.03 | 2345.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 2342.20 | 2335.03 | 2345.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 2257.00 | 2254.68 | 2287.92 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 2308.50 | 2296.44 | 2295.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 2322.60 | 2301.68 | 2297.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 11:15:00 | 2296.65 | 2300.67 | 2297.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 11:15:00 | 2296.65 | 2300.67 | 2297.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 2296.65 | 2300.67 | 2297.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:00:00 | 2296.65 | 2300.67 | 2297.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 2291.05 | 2298.75 | 2297.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:45:00 | 2284.85 | 2298.75 | 2297.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 2295.25 | 2298.05 | 2297.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:30:00 | 2285.50 | 2298.05 | 2297.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 2294.05 | 2297.25 | 2296.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 2294.05 | 2297.25 | 2296.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 15:15:00 | 2291.15 | 2296.03 | 2296.23 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 2346.60 | 2306.14 | 2300.81 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 2300.00 | 2321.24 | 2321.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 2272.55 | 2308.46 | 2315.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 2284.60 | 2279.44 | 2293.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 2284.60 | 2279.44 | 2293.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 2284.90 | 2280.53 | 2292.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 2285.10 | 2280.53 | 2292.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 2295.75 | 2283.57 | 2292.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 2258.35 | 2283.57 | 2292.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2250.85 | 2277.03 | 2288.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 2246.55 | 2269.67 | 2283.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 2247.65 | 2269.67 | 2283.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:45:00 | 2249.25 | 2264.32 | 2278.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 2250.10 | 2260.22 | 2269.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 2282.15 | 2263.68 | 2269.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 2293.30 | 2263.68 | 2269.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 2274.90 | 2265.92 | 2269.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:45:00 | 2270.40 | 2268.31 | 2270.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:30:00 | 2265.00 | 2270.29 | 2270.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 2279.90 | 2271.52 | 2271.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 2279.90 | 2271.52 | 2271.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 2317.25 | 2285.94 | 2278.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 2318.05 | 2318.84 | 2304.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 2318.05 | 2318.84 | 2304.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 2421.30 | 2434.82 | 2418.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 2421.30 | 2434.82 | 2418.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 2432.00 | 2434.26 | 2419.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 2432.75 | 2434.26 | 2419.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2412.75 | 2429.96 | 2418.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 2442.75 | 2431.20 | 2422.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:00:00 | 2449.20 | 2433.54 | 2425.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 2443.90 | 2439.57 | 2432.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 10:00:00 | 2442.05 | 2441.82 | 2434.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 2427.25 | 2444.87 | 2438.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 2427.25 | 2444.87 | 2438.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 2410.35 | 2437.97 | 2435.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 2410.35 | 2437.97 | 2435.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 15:15:00 | 2418.95 | 2431.61 | 2433.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 2418.95 | 2431.61 | 2433.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 2408.85 | 2427.85 | 2431.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 2444.05 | 2425.93 | 2428.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 2444.05 | 2425.93 | 2428.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 2444.05 | 2425.93 | 2428.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 2444.05 | 2425.93 | 2428.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 2439.90 | 2428.72 | 2429.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:45:00 | 2441.70 | 2428.72 | 2429.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 2449.45 | 2432.87 | 2431.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 2507.40 | 2447.78 | 2438.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 2459.60 | 2475.62 | 2460.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 2459.60 | 2475.62 | 2460.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 2459.60 | 2475.62 | 2460.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 2459.60 | 2475.62 | 2460.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 2468.45 | 2474.19 | 2461.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:15:00 | 2478.25 | 2474.19 | 2461.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 2494.35 | 2477.23 | 2469.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:45:00 | 2475.25 | 2477.73 | 2471.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 2358.90 | 2450.52 | 2460.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2358.90 | 2450.52 | 2460.40 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 2433.30 | 2419.43 | 2419.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 2464.80 | 2428.51 | 2423.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 2733.10 | 2733.50 | 2702.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 2761.50 | 2733.50 | 2702.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 11:30:00 | 2750.30 | 2738.14 | 2712.57 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 2777.60 | 2775.36 | 2754.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 2759.00 | 2775.36 | 2754.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2742.40 | 2769.83 | 2755.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 2742.40 | 2769.83 | 2755.55 | SL hit (close<ema400) qty=1.00 sl=2755.55 alert=retest1 |

### Cycle 75 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 2736.10 | 2745.45 | 2746.64 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 2778.10 | 2745.62 | 2744.24 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 2711.50 | 2738.80 | 2741.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 10:15:00 | 2699.80 | 2731.00 | 2737.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 2737.20 | 2688.95 | 2701.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 2737.20 | 2688.95 | 2701.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 2737.20 | 2688.95 | 2701.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 2741.70 | 2688.95 | 2701.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 2715.20 | 2694.20 | 2702.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:30:00 | 2711.40 | 2696.46 | 2702.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 13:15:00 | 2730.50 | 2708.20 | 2707.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 2730.50 | 2708.20 | 2707.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 2760.60 | 2719.83 | 2713.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 2777.40 | 2779.78 | 2761.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 2777.40 | 2779.78 | 2761.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 2777.40 | 2779.78 | 2761.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 2759.30 | 2779.78 | 2761.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2762.90 | 2774.82 | 2762.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:30:00 | 2792.50 | 2781.62 | 2766.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:00:00 | 2794.70 | 2787.58 | 2774.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 2721.70 | 2765.54 | 2769.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 2721.70 | 2765.54 | 2769.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 2695.50 | 2751.53 | 2762.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 2749.00 | 2694.71 | 2715.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 2749.00 | 2694.71 | 2715.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2749.00 | 2694.71 | 2715.02 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 2753.90 | 2729.02 | 2727.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 2755.50 | 2734.32 | 2729.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 2730.90 | 2738.93 | 2733.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 2730.90 | 2738.93 | 2733.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 2730.90 | 2738.93 | 2733.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 2730.90 | 2738.93 | 2733.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 2732.90 | 2737.73 | 2733.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:45:00 | 2726.50 | 2737.73 | 2733.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 2716.40 | 2733.46 | 2731.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 2716.40 | 2733.46 | 2731.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 2709.40 | 2728.65 | 2729.85 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 2748.60 | 2731.30 | 2729.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 2770.70 | 2746.16 | 2737.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 2830.00 | 2831.05 | 2804.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 2830.00 | 2831.05 | 2804.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2783.70 | 2818.75 | 2806.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 2783.70 | 2818.75 | 2806.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2791.60 | 2813.32 | 2805.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 2788.80 | 2813.32 | 2805.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2772.30 | 2798.68 | 2800.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 2752.20 | 2789.38 | 2795.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2785.00 | 2782.36 | 2791.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 2785.00 | 2782.36 | 2791.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2797.20 | 2785.33 | 2791.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2797.50 | 2785.33 | 2791.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2766.00 | 2781.46 | 2789.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 14:45:00 | 2765.30 | 2775.77 | 2784.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 2744.20 | 2773.81 | 2782.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 2802.50 | 2776.98 | 2778.63 | SL hit (close>static) qty=1.00 sl=2798.80 alert=retest2 |

### Cycle 84 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 2805.00 | 2782.58 | 2781.02 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 12:15:00 | 2771.70 | 2784.43 | 2784.53 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 2803.40 | 2785.30 | 2784.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 2830.50 | 2803.82 | 2795.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 2801.70 | 2807.27 | 2800.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 13:15:00 | 2801.70 | 2807.27 | 2800.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 2801.70 | 2807.27 | 2800.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 2800.20 | 2807.27 | 2800.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 2811.30 | 2808.07 | 2801.41 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 2782.30 | 2795.25 | 2796.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 2776.00 | 2788.77 | 2793.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 2800.00 | 2789.58 | 2792.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 15:15:00 | 2800.00 | 2789.58 | 2792.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 2800.00 | 2789.58 | 2792.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 2760.80 | 2789.58 | 2792.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 2755.80 | 2742.84 | 2742.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 2755.80 | 2742.84 | 2742.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 2803.40 | 2755.98 | 2748.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 2754.40 | 2766.40 | 2760.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 2754.40 | 2766.40 | 2760.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 2754.40 | 2766.40 | 2760.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 2754.40 | 2766.40 | 2760.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 2765.90 | 2766.30 | 2761.01 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 2730.00 | 2752.54 | 2755.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2716.10 | 2745.25 | 2751.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 2730.10 | 2729.43 | 2740.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 2730.10 | 2729.43 | 2740.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 2730.10 | 2729.43 | 2740.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 2730.50 | 2729.43 | 2740.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 2740.90 | 2731.73 | 2740.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 2740.90 | 2731.73 | 2740.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 2744.50 | 2734.28 | 2741.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 2734.90 | 2734.28 | 2741.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2742.60 | 2735.94 | 2741.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 2759.70 | 2735.94 | 2741.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2774.10 | 2743.58 | 2744.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2774.10 | 2743.58 | 2744.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 2778.80 | 2750.62 | 2747.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 2796.20 | 2771.86 | 2759.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 2777.40 | 2779.13 | 2767.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 11:30:00 | 2773.90 | 2779.13 | 2767.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2774.00 | 2778.13 | 2770.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 2806.40 | 2778.13 | 2770.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2820.00 | 2786.50 | 2774.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:45:00 | 2829.50 | 2801.04 | 2792.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 2779.90 | 2789.24 | 2789.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 2779.90 | 2789.24 | 2789.39 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 2842.30 | 2796.19 | 2792.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 2853.60 | 2807.68 | 2797.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 2920.00 | 2929.93 | 2909.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 09:15:00 | 2907.80 | 2929.93 | 2909.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2912.00 | 2926.34 | 2909.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 10:45:00 | 2949.30 | 2930.23 | 2912.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 2893.80 | 2911.47 | 2912.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 2893.80 | 2911.47 | 2912.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 2886.90 | 2904.08 | 2909.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 2899.70 | 2896.43 | 2902.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 2899.70 | 2896.43 | 2902.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2899.70 | 2896.43 | 2902.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 2899.70 | 2896.43 | 2902.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2938.70 | 2905.71 | 2905.82 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 2923.10 | 2909.19 | 2907.39 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 2894.00 | 2905.51 | 2906.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 2877.00 | 2898.54 | 2903.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 2881.90 | 2880.95 | 2888.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:30:00 | 2885.60 | 2880.95 | 2888.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2840.60 | 2870.30 | 2880.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 2830.80 | 2859.38 | 2874.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 2834.50 | 2838.44 | 2853.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 2835.00 | 2836.93 | 2849.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 2836.40 | 2837.10 | 2848.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 2783.00 | 2781.95 | 2798.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:30:00 | 2806.50 | 2781.95 | 2798.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 2799.50 | 2785.46 | 2798.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 2799.50 | 2785.46 | 2798.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 2803.20 | 2789.00 | 2798.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 2803.20 | 2789.00 | 2798.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 2807.00 | 2792.60 | 2799.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 2830.70 | 2792.60 | 2799.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 2873.70 | 2815.77 | 2809.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 2873.70 | 2815.77 | 2809.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 2879.70 | 2828.55 | 2815.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 2878.00 | 2885.70 | 2868.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 2878.00 | 2885.70 | 2868.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2872.50 | 2879.87 | 2872.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 2870.40 | 2879.87 | 2872.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 2858.60 | 2875.61 | 2871.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 2858.00 | 2875.61 | 2871.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2860.90 | 2872.67 | 2870.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 2862.00 | 2872.67 | 2870.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 2860.10 | 2870.16 | 2869.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:15:00 | 2861.30 | 2870.16 | 2869.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 2850.00 | 2866.13 | 2867.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 2837.10 | 2858.51 | 2863.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2826.90 | 2810.72 | 2827.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2826.90 | 2810.72 | 2827.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2826.90 | 2810.72 | 2827.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 2824.70 | 2810.72 | 2827.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2802.90 | 2809.16 | 2825.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 2801.50 | 2809.16 | 2825.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:30:00 | 2801.50 | 2807.99 | 2822.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:30:00 | 2796.40 | 2806.15 | 2820.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 2796.00 | 2804.06 | 2814.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 2773.00 | 2767.16 | 2782.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 2793.80 | 2767.16 | 2782.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2815.80 | 2776.89 | 2785.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 2815.80 | 2776.89 | 2785.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2803.20 | 2782.15 | 2786.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 2817.40 | 2782.15 | 2786.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 2796.10 | 2789.58 | 2789.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 2796.10 | 2789.58 | 2789.44 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 2759.80 | 2791.07 | 2793.21 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 2810.40 | 2795.66 | 2794.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 2860.00 | 2813.99 | 2804.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 2816.20 | 2816.36 | 2807.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 11:15:00 | 2816.20 | 2816.36 | 2807.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 2816.20 | 2816.36 | 2807.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 2808.40 | 2816.36 | 2807.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 2851.40 | 2823.36 | 2811.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 2871.30 | 2832.41 | 2816.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 2870.00 | 2837.33 | 2819.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 13:15:00 | 2959.40 | 2973.56 | 2973.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 2959.40 | 2973.56 | 2973.61 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 3009.10 | 2977.71 | 2975.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 3038.50 | 2989.87 | 2980.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2992.50 | 3008.99 | 2997.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 2992.50 | 3008.99 | 2997.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2992.50 | 3008.99 | 2997.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 2992.50 | 3008.99 | 2997.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2999.50 | 3007.09 | 2997.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:30:00 | 3012.10 | 3009.91 | 2999.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-25 09:15:00 | 3313.31 | 3285.81 | 3265.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 3266.80 | 3273.52 | 3273.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 3256.50 | 3269.42 | 3271.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 3294.00 | 3272.81 | 3272.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 3294.00 | 3272.81 | 3272.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3294.00 | 3272.81 | 3272.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 3294.00 | 3272.81 | 3272.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 3298.30 | 3277.91 | 3275.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 3305.80 | 3283.49 | 3278.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 14:15:00 | 3273.90 | 3285.88 | 3280.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 14:15:00 | 3273.90 | 3285.88 | 3280.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 3273.90 | 3285.88 | 3280.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 3273.90 | 3285.88 | 3280.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 3280.00 | 3284.70 | 3280.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 3285.00 | 3284.70 | 3280.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:45:00 | 3293.70 | 3286.98 | 3282.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 3493.60 | 3517.40 | 3520.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 3493.60 | 3517.40 | 3520.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 3470.00 | 3504.37 | 3513.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 12:15:00 | 3504.40 | 3496.97 | 3507.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 3504.40 | 3496.97 | 3507.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 3517.30 | 3501.03 | 3508.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:30:00 | 3521.20 | 3501.03 | 3508.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 3510.70 | 3502.97 | 3508.41 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 3527.70 | 3514.24 | 3512.66 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 3492.00 | 3509.46 | 3510.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 11:15:00 | 3474.90 | 3492.19 | 3501.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 3514.80 | 3489.79 | 3495.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 3514.80 | 3489.79 | 3495.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3514.80 | 3489.79 | 3495.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 3514.80 | 3489.79 | 3495.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 3513.70 | 3494.57 | 3497.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 3512.90 | 3494.57 | 3497.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 3516.50 | 3501.06 | 3499.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 3524.40 | 3508.57 | 3503.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 3500.00 | 3508.24 | 3504.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 3500.00 | 3508.24 | 3504.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 3500.00 | 3508.24 | 3504.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 3500.00 | 3508.24 | 3504.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 3499.50 | 3506.49 | 3504.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 3499.40 | 3506.49 | 3504.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 3497.90 | 3504.77 | 3503.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 3510.40 | 3504.77 | 3503.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 3509.90 | 3504.37 | 3503.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 3514.90 | 3531.52 | 3532.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 3514.90 | 3531.52 | 3532.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 3501.40 | 3525.50 | 3529.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 10:15:00 | 3438.90 | 3429.84 | 3456.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:45:00 | 3449.00 | 3429.84 | 3456.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 3437.00 | 3419.15 | 3432.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 3437.00 | 3419.15 | 3432.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 3439.00 | 3423.12 | 3433.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 3439.00 | 3423.12 | 3433.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 3446.60 | 3427.82 | 3434.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 3433.20 | 3427.82 | 3434.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 3410.40 | 3424.06 | 3431.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 3420.10 | 3424.06 | 3431.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 3436.10 | 3425.85 | 3431.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 3436.10 | 3425.85 | 3431.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 3439.30 | 3428.54 | 3431.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 3450.00 | 3428.54 | 3431.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 3446.00 | 3433.55 | 3433.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 3470.10 | 3433.55 | 3433.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3442.90 | 3435.42 | 3434.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 3491.90 | 3454.44 | 3446.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 3510.00 | 3512.75 | 3496.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 3521.00 | 3512.75 | 3496.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3504.30 | 3511.06 | 3496.89 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 3474.40 | 3494.29 | 3495.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 12:15:00 | 3451.30 | 3481.11 | 3488.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 3486.40 | 3478.98 | 3486.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 3486.40 | 3478.98 | 3486.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 3486.40 | 3478.98 | 3486.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 3486.40 | 3478.98 | 3486.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 3481.00 | 3479.38 | 3485.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 3484.90 | 3479.38 | 3485.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3489.30 | 3481.37 | 3485.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 3485.20 | 3481.37 | 3485.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3498.70 | 3484.83 | 3487.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 3498.70 | 3484.83 | 3487.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 3508.70 | 3489.61 | 3489.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 3512.50 | 3494.19 | 3491.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 3490.80 | 3494.60 | 3491.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 3490.80 | 3494.60 | 3491.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 3490.80 | 3494.60 | 3491.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 3487.00 | 3494.60 | 3491.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 3490.20 | 3493.72 | 3491.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 3496.40 | 3493.72 | 3491.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3509.70 | 3496.92 | 3493.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 3538.00 | 3509.15 | 3502.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 3607.00 | 3626.39 | 3628.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 3607.00 | 3626.39 | 3628.25 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 3635.70 | 3621.43 | 3619.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 3643.90 | 3625.93 | 3622.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 3628.00 | 3635.16 | 3628.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 3628.00 | 3635.16 | 3628.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 3628.00 | 3635.16 | 3628.43 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3550.40 | 3614.01 | 3620.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 3523.60 | 3585.35 | 3605.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 3534.30 | 3508.68 | 3530.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 3534.30 | 3508.68 | 3530.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 3534.30 | 3508.68 | 3530.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 3543.00 | 3508.68 | 3530.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 3519.00 | 3510.74 | 3529.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:30:00 | 3511.70 | 3513.02 | 3528.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 3513.20 | 3516.68 | 3527.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 3510.40 | 3513.60 | 3522.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 3510.40 | 3508.38 | 3515.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 3526.50 | 3511.94 | 3515.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 3526.50 | 3511.94 | 3515.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 3512.00 | 3511.96 | 3515.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 3500.20 | 3511.96 | 3515.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 3499.00 | 3468.26 | 3467.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 3499.00 | 3468.26 | 3467.22 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 3458.60 | 3470.08 | 3471.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 3445.40 | 3461.81 | 3465.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 3444.80 | 3413.34 | 3429.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 3444.80 | 3413.34 | 3429.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3444.80 | 3413.34 | 3429.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 3444.80 | 3413.34 | 3429.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 3460.70 | 3422.82 | 3432.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 3460.70 | 3422.82 | 3432.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 3473.90 | 3433.03 | 3436.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 3473.90 | 3433.03 | 3436.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 3474.30 | 3441.29 | 3439.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 3475.60 | 3448.15 | 3442.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 3465.00 | 3474.59 | 3463.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 3481.60 | 3474.59 | 3463.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 3481.90 | 3476.05 | 3464.85 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 13:15:00 | 3468.70 | 3473.15 | 3473.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 3441.90 | 3466.90 | 3470.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 3472.50 | 3454.33 | 3459.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 3472.50 | 3454.33 | 3459.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3472.50 | 3454.33 | 3459.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 3472.50 | 3454.33 | 3459.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 3448.00 | 3453.06 | 3458.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 3455.00 | 3453.06 | 3458.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 3459.00 | 3454.25 | 3458.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 3459.00 | 3454.25 | 3458.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 3465.90 | 3456.58 | 3459.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 3465.90 | 3456.58 | 3459.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 3455.80 | 3456.42 | 3458.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 3451.00 | 3455.28 | 3458.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 3484.00 | 3458.59 | 3459.00 | SL hit (close>static) qty=1.00 sl=3466.40 alert=retest2 |

### Cycle 120 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 3485.40 | 3463.96 | 3461.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 3499.90 | 3471.14 | 3464.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 3509.80 | 3515.03 | 3495.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:00:00 | 3509.80 | 3515.03 | 3495.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 3623.50 | 3645.04 | 3625.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 3623.50 | 3645.04 | 3625.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 3621.00 | 3640.23 | 3624.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 3632.10 | 3635.91 | 3625.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 3611.80 | 3648.77 | 3645.20 | SL hit (close<static) qty=1.00 sl=3613.80 alert=retest2 |

### Cycle 121 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 3606.40 | 3640.30 | 3641.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 3601.10 | 3632.46 | 3637.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 3611.40 | 3609.69 | 3619.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:30:00 | 3611.60 | 3609.69 | 3619.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 3612.10 | 3610.17 | 3618.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 3612.10 | 3610.17 | 3618.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 3613.00 | 3610.74 | 3618.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 3633.00 | 3610.74 | 3618.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 3647.00 | 3617.99 | 3620.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 3647.00 | 3617.99 | 3620.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 3624.90 | 3620.24 | 3621.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:45:00 | 3609.60 | 3618.12 | 3620.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 3630.60 | 3618.98 | 3620.10 | SL hit (close>static) qty=1.00 sl=3629.80 alert=retest2 |

### Cycle 122 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 3636.70 | 3621.01 | 3620.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 3649.90 | 3628.05 | 3623.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 3629.30 | 3630.05 | 3625.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 12:00:00 | 3629.30 | 3630.05 | 3625.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 3643.50 | 3632.73 | 3627.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 14:30:00 | 3647.60 | 3637.79 | 3630.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 3615.60 | 3636.34 | 3631.10 | SL hit (close<static) qty=1.00 sl=3623.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 3617.30 | 3630.67 | 3630.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 3602.40 | 3625.02 | 3628.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 3639.30 | 3626.05 | 3627.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 3639.30 | 3626.05 | 3627.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3639.30 | 3626.05 | 3627.53 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 3647.90 | 3630.42 | 3629.38 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 3576.90 | 3624.46 | 3628.10 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 3656.60 | 3619.60 | 3618.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 3666.90 | 3640.30 | 3629.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3692.10 | 3694.98 | 3672.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 3688.00 | 3690.51 | 3678.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 3688.00 | 3690.51 | 3678.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 3686.50 | 3690.51 | 3678.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 3685.20 | 3689.44 | 3679.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 3679.30 | 3689.44 | 3679.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3686.00 | 3688.76 | 3679.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 3675.60 | 3688.76 | 3679.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 3682.20 | 3691.29 | 3683.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 3682.20 | 3691.29 | 3683.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 3674.60 | 3687.96 | 3682.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 3674.60 | 3687.96 | 3682.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 3671.80 | 3684.72 | 3681.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 3671.30 | 3684.72 | 3681.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 3653.80 | 3678.54 | 3679.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 3647.70 | 3672.37 | 3676.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 3598.60 | 3597.17 | 3621.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:00:00 | 3598.60 | 3597.17 | 3621.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 3597.70 | 3597.27 | 3619.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 3619.00 | 3597.27 | 3619.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 3615.10 | 3599.70 | 3615.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 3626.00 | 3599.70 | 3615.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 3636.70 | 3607.10 | 3616.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 3646.60 | 3607.10 | 3616.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3639.00 | 3613.48 | 3618.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 3665.70 | 3613.48 | 3618.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 3660.00 | 3622.78 | 3622.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 3686.40 | 3643.09 | 3632.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 3846.60 | 3863.93 | 3822.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 3846.60 | 3863.93 | 3822.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 3856.70 | 3858.43 | 3835.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 3878.50 | 3862.45 | 3839.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 3879.40 | 3865.84 | 3842.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 13:15:00 | 3825.20 | 3850.35 | 3846.31 | SL hit (close<static) qty=1.00 sl=3835.10 alert=retest2 |

### Cycle 129 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 3833.50 | 3844.18 | 3844.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 3806.70 | 3826.94 | 3835.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 3829.00 | 3820.69 | 3829.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 3829.00 | 3820.69 | 3829.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3829.00 | 3820.69 | 3829.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:30:00 | 3773.90 | 3804.57 | 3819.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 3585.20 | 3625.46 | 3655.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 3597.20 | 3597.04 | 3627.45 | SL hit (close>ema200) qty=0.50 sl=3597.04 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 3665.10 | 3587.23 | 3576.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 3724.20 | 3614.62 | 3590.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 3621.50 | 3639.05 | 3610.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 3621.50 | 3639.05 | 3610.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3621.50 | 3639.05 | 3610.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:00:00 | 3621.50 | 3639.05 | 3610.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 3624.80 | 3636.20 | 3611.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:30:00 | 3630.40 | 3636.20 | 3611.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 3632.00 | 3635.36 | 3613.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 3617.90 | 3635.36 | 3613.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 3667.40 | 3651.49 | 3635.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:30:00 | 3677.50 | 3656.25 | 3638.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 3671.30 | 3658.76 | 3641.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 3686.60 | 3663.01 | 3646.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 3595.00 | 3645.47 | 3642.21 | SL hit (close<static) qty=1.00 sl=3630.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 3595.00 | 3635.37 | 3637.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 3569.30 | 3615.80 | 3627.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 3633.50 | 3618.92 | 3627.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 3633.50 | 3618.92 | 3627.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 3633.50 | 3618.92 | 3627.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 3633.50 | 3618.92 | 3627.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 3632.10 | 3621.56 | 3627.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:15:00 | 3641.30 | 3621.56 | 3627.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 3644.40 | 3626.13 | 3629.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 3644.40 | 3626.13 | 3629.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 3655.00 | 3631.90 | 3631.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 3702.90 | 3646.10 | 3637.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 3727.90 | 3729.41 | 3703.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 14:45:00 | 3718.30 | 3729.41 | 3703.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3702.40 | 3722.14 | 3704.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 3699.40 | 3722.14 | 3704.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3707.40 | 3719.19 | 3704.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:45:00 | 3730.00 | 3716.63 | 3705.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 3727.30 | 3715.77 | 3706.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 3681.30 | 3708.38 | 3705.37 | SL hit (close<static) qty=1.00 sl=3700.60 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 13:15:00 | 3822.90 | 3847.50 | 3848.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 3808.90 | 3839.78 | 3844.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 12:15:00 | 3869.00 | 3838.77 | 3841.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 12:15:00 | 3869.00 | 3838.77 | 3841.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 3869.00 | 3838.77 | 3841.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:00:00 | 3869.00 | 3838.77 | 3841.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 3857.00 | 3842.42 | 3842.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 3857.00 | 3842.42 | 3842.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 3878.00 | 3849.53 | 3845.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 3882.50 | 3871.73 | 3861.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 3860.00 | 3871.99 | 3863.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 3860.00 | 3871.99 | 3863.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 3860.00 | 3871.99 | 3863.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 3860.00 | 3871.99 | 3863.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 3852.80 | 3868.15 | 3862.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 3861.10 | 3868.15 | 3862.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 3874.30 | 3869.38 | 3863.37 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 3829.60 | 3860.78 | 3861.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 3810.10 | 3845.72 | 3853.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 15:15:00 | 3829.40 | 3824.02 | 3837.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 3870.30 | 3824.02 | 3837.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 3863.70 | 3831.95 | 3839.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 3861.90 | 3831.95 | 3839.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 3858.80 | 3837.32 | 3841.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 3844.20 | 3840.46 | 3842.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 3939.70 | 3854.00 | 3842.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3939.70 | 3854.00 | 3842.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 3956.20 | 3914.28 | 3882.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 3928.90 | 3936.53 | 3911.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 3928.90 | 3936.53 | 3911.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 3906.00 | 3930.42 | 3910.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 3906.00 | 3930.42 | 3910.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 3905.10 | 3925.36 | 3910.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 3920.00 | 3910.40 | 3906.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 3866.00 | 3901.52 | 3903.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 3866.00 | 3901.52 | 3903.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 10:15:00 | 3819.00 | 3882.47 | 3894.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 3757.60 | 3754.71 | 3803.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 3757.60 | 3754.71 | 3803.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3774.90 | 3756.83 | 3792.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:30:00 | 3744.00 | 3756.47 | 3785.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 3748.50 | 3755.06 | 3782.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 3816.70 | 3775.74 | 3785.73 | SL hit (close>static) qty=1.00 sl=3814.90 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 3781.20 | 3733.98 | 3729.37 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 3673.40 | 3718.61 | 3724.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 3634.60 | 3701.81 | 3716.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3380.10 | 3343.90 | 3401.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 3378.80 | 3343.90 | 3401.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3429.70 | 3362.21 | 3399.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3429.70 | 3362.21 | 3399.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3423.80 | 3374.53 | 3401.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 3423.80 | 3374.53 | 3401.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 3423.00 | 3385.67 | 3402.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 3423.00 | 3385.67 | 3402.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 3457.50 | 3400.04 | 3407.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 3457.50 | 3400.04 | 3407.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 3492.00 | 3418.43 | 3414.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3544.00 | 3454.12 | 3432.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3478.90 | 3523.40 | 3488.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3478.90 | 3523.40 | 3488.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3478.90 | 3523.40 | 3488.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 3484.00 | 3523.40 | 3488.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3466.90 | 3512.10 | 3486.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 3466.90 | 3512.10 | 3486.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 3443.90 | 3498.46 | 3482.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 3443.90 | 3498.46 | 3482.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 3452.40 | 3470.46 | 3471.84 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 3483.00 | 3474.03 | 3473.29 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 3458.90 | 3472.61 | 3473.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3386.60 | 3456.25 | 3465.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3428.60 | 3419.34 | 3437.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3428.60 | 3419.34 | 3437.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3428.60 | 3419.34 | 3437.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 3414.00 | 3418.21 | 3434.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 3495.70 | 3443.91 | 3443.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 3495.70 | 3443.91 | 3443.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 3555.70 | 3477.87 | 3459.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3447.70 | 3517.49 | 3496.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3447.70 | 3517.49 | 3496.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3447.70 | 3517.49 | 3496.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3447.70 | 3517.49 | 3496.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3447.00 | 3503.39 | 3492.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 3447.00 | 3503.39 | 3492.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 3438.70 | 3480.84 | 3484.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3383.80 | 3455.36 | 3471.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3455.20 | 3407.21 | 3431.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3455.20 | 3407.21 | 3431.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3455.20 | 3407.21 | 3431.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 3485.50 | 3407.21 | 3431.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 3400.60 | 3405.89 | 3429.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 3331.40 | 3431.23 | 3435.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 3390.30 | 3379.72 | 3403.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 3496.00 | 3406.55 | 3409.85 | SL hit (close>static) qty=1.00 sl=3464.80 alert=retest2 |

### Cycle 146 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 3464.90 | 3418.22 | 3414.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 3690.90 | 3507.78 | 3471.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 3730.00 | 3782.19 | 3726.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 3730.00 | 3782.19 | 3726.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3730.00 | 3782.19 | 3726.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 3783.70 | 3755.23 | 3736.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:15:00 | 3780.80 | 3759.98 | 3740.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 3781.40 | 3764.30 | 3744.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:30:00 | 3783.80 | 3770.07 | 3750.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 3744.90 | 3770.22 | 3760.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 3744.90 | 3770.22 | 3760.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 3755.70 | 3767.31 | 3759.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 3768.80 | 3762.99 | 3758.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 3770.00 | 3762.99 | 3758.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:00:00 | 3768.00 | 3765.11 | 3760.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 3731.20 | 3754.32 | 3756.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 12:15:00 | 3731.20 | 3754.32 | 3756.38 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 3761.30 | 3754.56 | 3753.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 3780.20 | 3761.94 | 3757.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 3751.50 | 3763.36 | 3760.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 3751.50 | 3763.36 | 3760.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 3751.50 | 3763.36 | 3760.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 3751.50 | 3763.36 | 3760.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 3752.00 | 3761.09 | 3759.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 3751.10 | 3761.09 | 3759.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 3748.60 | 3758.59 | 3758.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 3717.80 | 3750.43 | 3754.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 3505.00 | 3501.58 | 3544.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 3505.00 | 3501.58 | 3544.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 3552.60 | 3511.79 | 3545.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 3541.20 | 3511.79 | 3545.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 3556.00 | 3520.63 | 3546.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:15:00 | 3562.40 | 3520.63 | 3546.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 3548.20 | 3526.14 | 3546.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 3529.70 | 3532.85 | 3545.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 3570.10 | 3520.70 | 3530.51 | SL hit (close>static) qty=1.00 sl=3564.10 alert=retest2 |

### Cycle 150 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 3583.70 | 3541.36 | 3538.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 3587.10 | 3550.51 | 3543.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 3553.00 | 3553.05 | 3546.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 3450.00 | 3553.05 | 3546.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 151 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 3423.50 | 3527.14 | 3535.17 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 3542.10 | 3506.18 | 3503.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 3566.70 | 3527.46 | 3515.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 3519.00 | 3525.77 | 3515.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 3519.00 | 3525.77 | 3515.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 3519.00 | 3525.77 | 3515.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 3519.00 | 3525.77 | 3515.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 3558.10 | 3532.23 | 3519.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 3571.00 | 3532.23 | 3519.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 3592.20 | 3543.13 | 3526.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 2121.00 | 2024-05-22 11:15:00 | 2129.50 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2024-05-16 11:30:00 | 2104.85 | 2024-05-22 11:15:00 | 2129.50 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-05-16 13:45:00 | 2108.80 | 2024-05-22 11:15:00 | 2129.50 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest1 | 2024-05-28 10:45:00 | 2267.80 | 2024-05-29 09:15:00 | 2231.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-05-29 10:30:00 | 2240.90 | 2024-05-30 13:15:00 | 2237.60 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-05-29 12:15:00 | 2239.40 | 2024-05-31 09:15:00 | 2204.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-05-29 13:00:00 | 2249.95 | 2024-05-31 09:15:00 | 2204.45 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-05-30 10:00:00 | 2240.00 | 2024-05-31 09:15:00 | 2204.45 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-05-30 12:30:00 | 2248.85 | 2024-05-31 09:15:00 | 2204.45 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-06-13 13:00:00 | 2456.00 | 2024-06-19 13:15:00 | 2434.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-06-14 09:45:00 | 2452.30 | 2024-06-19 13:15:00 | 2434.80 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-06-19 09:30:00 | 2461.45 | 2024-06-19 13:15:00 | 2434.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-06-19 13:00:00 | 2449.90 | 2024-06-19 13:15:00 | 2434.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-06-21 15:00:00 | 2432.25 | 2024-06-24 12:15:00 | 2444.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-07-02 11:30:00 | 2359.20 | 2024-07-03 09:15:00 | 2372.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-07-03 11:00:00 | 2354.15 | 2024-07-04 14:15:00 | 2365.45 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-07-09 10:30:00 | 2419.00 | 2024-07-15 13:15:00 | 2438.60 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest1 | 2024-07-19 09:15:00 | 2391.20 | 2024-07-22 09:15:00 | 2436.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest1 | 2024-07-19 09:45:00 | 2396.00 | 2024-07-22 09:15:00 | 2436.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest1 | 2024-07-19 15:15:00 | 2398.00 | 2024-07-22 09:15:00 | 2436.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-07-24 15:15:00 | 2456.25 | 2024-07-25 09:15:00 | 2431.95 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-07-25 11:30:00 | 2457.30 | 2024-08-05 10:15:00 | 2479.05 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2024-07-26 09:30:00 | 2468.95 | 2024-08-05 10:15:00 | 2479.05 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-08-09 09:15:00 | 2553.40 | 2024-08-19 10:15:00 | 2589.40 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2024-08-12 11:15:00 | 2558.15 | 2024-08-19 10:15:00 | 2589.40 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2024-08-22 09:15:00 | 2638.70 | 2024-08-28 14:15:00 | 2734.25 | STOP_HIT | 1.00 | 3.62% |
| SELL | retest2 | 2024-08-29 11:00:00 | 2737.95 | 2024-08-29 15:15:00 | 2762.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-08-29 13:45:00 | 2738.00 | 2024-08-29 15:15:00 | 2762.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-09-05 11:30:00 | 2767.30 | 2024-09-10 13:15:00 | 2769.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-09-05 14:15:00 | 2766.15 | 2024-09-10 13:15:00 | 2769.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-09-10 09:45:00 | 2767.25 | 2024-09-10 13:15:00 | 2769.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-09-12 09:30:00 | 2776.50 | 2024-09-18 12:15:00 | 2791.00 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2024-09-25 15:00:00 | 2874.50 | 2024-09-30 13:15:00 | 2817.10 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-09-26 14:00:00 | 2879.00 | 2024-09-30 13:15:00 | 2817.10 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-09-30 10:00:00 | 2883.20 | 2024-09-30 13:15:00 | 2817.10 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2024-10-03 09:15:00 | 2764.00 | 2024-10-07 10:15:00 | 2625.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-03 09:15:00 | 2764.00 | 2024-10-08 10:15:00 | 2671.45 | STOP_HIT | 0.50 | 3.35% |
| BUY | retest2 | 2024-10-15 11:30:00 | 2826.90 | 2024-10-16 10:15:00 | 2793.20 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-10-15 13:30:00 | 2834.60 | 2024-10-16 10:15:00 | 2793.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-10-18 15:00:00 | 2710.80 | 2024-10-21 10:15:00 | 2751.55 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-10-21 09:15:00 | 2692.15 | 2024-10-21 10:15:00 | 2751.55 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-10-22 10:00:00 | 2708.10 | 2024-10-23 10:15:00 | 2749.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-10-22 11:15:00 | 2710.20 | 2024-10-23 10:15:00 | 2749.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-10-30 15:00:00 | 2457.65 | 2024-10-31 10:15:00 | 2483.85 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-12 13:30:00 | 2460.60 | 2024-11-18 13:15:00 | 2426.85 | STOP_HIT | 1.00 | 1.37% |
| SELL | retest2 | 2024-11-12 14:00:00 | 2459.15 | 2024-11-18 13:15:00 | 2426.85 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2024-12-05 14:15:00 | 2508.55 | 2024-12-09 13:15:00 | 2495.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-12-05 15:00:00 | 2514.90 | 2024-12-09 13:15:00 | 2495.80 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-19 09:15:00 | 2442.60 | 2024-12-26 11:15:00 | 2438.20 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-01-20 11:15:00 | 2298.75 | 2025-01-22 12:15:00 | 2271.05 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-01-21 14:45:00 | 2296.25 | 2025-01-22 12:15:00 | 2271.05 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-02-07 14:15:00 | 2595.80 | 2025-02-12 09:15:00 | 2466.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 2575.00 | 2025-02-12 09:15:00 | 2446.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 14:15:00 | 2595.80 | 2025-02-12 12:15:00 | 2487.25 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-02-10 09:15:00 | 2575.00 | 2025-02-12 12:15:00 | 2487.25 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-03-12 11:30:00 | 2246.55 | 2025-03-18 09:15:00 | 2279.90 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-03-12 12:15:00 | 2247.65 | 2025-03-18 09:15:00 | 2279.90 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-03-12 13:45:00 | 2249.25 | 2025-03-18 09:15:00 | 2279.90 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-13 15:00:00 | 2250.10 | 2025-03-18 09:15:00 | 2279.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-03-17 12:45:00 | 2270.40 | 2025-03-18 09:15:00 | 2279.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-03-17 14:30:00 | 2265.00 | 2025-03-18 09:15:00 | 2279.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-03-26 14:15:00 | 2442.75 | 2025-03-28 15:15:00 | 2418.95 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-03-27 10:00:00 | 2449.20 | 2025-03-28 15:15:00 | 2418.95 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-03-27 14:45:00 | 2443.90 | 2025-03-28 15:15:00 | 2418.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-03-28 10:00:00 | 2442.05 | 2025-03-28 15:15:00 | 2418.95 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-04-03 11:15:00 | 2478.25 | 2025-04-07 09:15:00 | 2358.90 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2025-04-04 10:30:00 | 2494.35 | 2025-04-07 09:15:00 | 2358.90 | STOP_HIT | 1.00 | -5.43% |
| BUY | retest2 | 2025-04-04 12:45:00 | 2475.25 | 2025-04-07 09:15:00 | 2358.90 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest1 | 2025-04-23 09:15:00 | 2761.50 | 2025-04-25 09:15:00 | 2742.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-04-23 11:30:00 | 2750.30 | 2025-04-25 09:15:00 | 2742.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-05-02 11:30:00 | 2711.40 | 2025-05-02 13:15:00 | 2730.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-05-07 10:30:00 | 2792.50 | 2025-05-08 13:15:00 | 2721.70 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-05-07 15:00:00 | 2794.70 | 2025-05-08 13:15:00 | 2721.70 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-05-21 14:45:00 | 2765.30 | 2025-05-22 14:15:00 | 2802.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-05-22 09:15:00 | 2744.20 | 2025-05-22 14:15:00 | 2802.50 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-06-02 09:15:00 | 2760.80 | 2025-06-10 10:15:00 | 2755.80 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-06-20 14:45:00 | 2829.50 | 2025-06-23 12:15:00 | 2779.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-30 10:45:00 | 2949.30 | 2025-07-01 14:15:00 | 2893.80 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-07-08 10:45:00 | 2830.80 | 2025-07-15 10:15:00 | 2873.70 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-07-09 11:15:00 | 2834.50 | 2025-07-15 10:15:00 | 2873.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-07-09 13:15:00 | 2835.00 | 2025-07-15 10:15:00 | 2873.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-07-09 14:15:00 | 2836.40 | 2025-07-15 10:15:00 | 2873.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-23 11:15:00 | 2801.50 | 2025-07-28 13:15:00 | 2796.10 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-07-23 12:30:00 | 2801.50 | 2025-07-28 13:15:00 | 2796.10 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-07-23 13:30:00 | 2796.40 | 2025-07-28 13:15:00 | 2796.10 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-07-24 11:00:00 | 2796.00 | 2025-07-28 13:15:00 | 2796.10 | STOP_HIT | 1.00 | -0.00% |
| BUY | retest2 | 2025-08-01 14:15:00 | 2871.30 | 2025-08-12 13:15:00 | 2959.40 | STOP_HIT | 1.00 | 3.07% |
| BUY | retest2 | 2025-08-01 15:15:00 | 2870.00 | 2025-08-12 13:15:00 | 2959.40 | STOP_HIT | 1.00 | 3.11% |
| BUY | retest2 | 2025-08-14 11:30:00 | 3012.10 | 2025-08-25 09:15:00 | 3313.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:15:00 | 3285.00 | 2025-09-10 14:15:00 | 3493.60 | STOP_HIT | 1.00 | 6.35% |
| BUY | retest2 | 2025-09-01 09:45:00 | 3293.70 | 2025-09-10 14:15:00 | 3493.60 | STOP_HIT | 1.00 | 6.07% |
| BUY | retest2 | 2025-09-18 09:15:00 | 3510.40 | 2025-09-23 14:15:00 | 3514.90 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-18 13:15:00 | 3509.90 | 2025-09-23 14:15:00 | 3514.90 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-10-15 10:00:00 | 3538.00 | 2025-10-23 15:15:00 | 3607.00 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-10-31 11:30:00 | 3511.70 | 2025-11-10 11:15:00 | 3499.00 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-10-31 14:15:00 | 3513.20 | 2025-11-10 11:15:00 | 3499.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-11-03 11:00:00 | 3510.40 | 2025-11-10 11:15:00 | 3499.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-04 10:30:00 | 3510.40 | 2025-11-10 11:15:00 | 3499.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-04 14:15:00 | 3500.20 | 2025-11-10 11:15:00 | 3499.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-11-25 14:30:00 | 3451.00 | 2025-11-26 09:15:00 | 3484.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-03 15:00:00 | 3632.10 | 2025-12-08 09:15:00 | 3611.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-10 14:45:00 | 3609.60 | 2025-12-11 09:15:00 | 3630.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-11 10:15:00 | 3609.10 | 2025-12-11 14:15:00 | 3636.70 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-11 10:45:00 | 3608.60 | 2025-12-11 14:15:00 | 3636.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-12 14:30:00 | 3647.60 | 2025-12-15 09:15:00 | 3615.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-06 13:00:00 | 3878.50 | 2026-01-07 13:15:00 | 3825.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-01-06 14:00:00 | 3879.40 | 2026-01-07 13:15:00 | 3825.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-09 12:30:00 | 3773.90 | 2026-01-21 09:15:00 | 3585.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 12:30:00 | 3773.90 | 2026-01-21 14:15:00 | 3597.20 | STOP_HIT | 0.50 | 4.68% |
| BUY | retest2 | 2026-01-30 14:30:00 | 3677.50 | 2026-02-01 13:15:00 | 3595.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-02-01 09:15:00 | 3671.30 | 2026-02-01 13:15:00 | 3595.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-02-01 10:30:00 | 3686.60 | 2026-02-01 13:15:00 | 3595.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-02-05 12:45:00 | 3730.00 | 2026-02-06 09:15:00 | 3681.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-02-05 13:30:00 | 3727.30 | 2026-02-06 09:15:00 | 3681.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-02-06 14:30:00 | 3730.30 | 2026-02-16 13:15:00 | 3822.90 | STOP_HIT | 1.00 | 2.48% |
| SELL | retest2 | 2026-02-23 12:15:00 | 3844.20 | 2026-02-25 10:15:00 | 3939.70 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-02-27 14:30:00 | 3920.00 | 2026-02-27 15:15:00 | 3866.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-03-05 11:30:00 | 3744.00 | 2026-03-05 15:15:00 | 3816.70 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-05 12:30:00 | 3748.50 | 2026-03-05 15:15:00 | 3816.70 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-03-09 09:15:00 | 3621.20 | 2026-03-10 14:15:00 | 3781.20 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2026-03-10 11:45:00 | 3749.90 | 2026-03-10 14:15:00 | 3781.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-24 10:30:00 | 3414.00 | 2026-03-24 13:15:00 | 3495.70 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-04-02 09:15:00 | 3331.40 | 2026-04-06 09:15:00 | 3496.00 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2026-04-02 14:15:00 | 3390.30 | 2026-04-06 09:15:00 | 3496.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-04-15 10:15:00 | 3783.70 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-15 11:15:00 | 3780.80 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-04-15 11:45:00 | 3781.40 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-04-15 13:30:00 | 3783.80 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-16 14:30:00 | 3768.80 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-04-16 15:15:00 | 3770.00 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-04-17 10:00:00 | 3768.00 | 2026-04-17 12:15:00 | 3731.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-04-28 10:30:00 | 3529.70 | 2026-04-29 09:15:00 | 3570.10 | STOP_HIT | 1.00 | -1.14% |
