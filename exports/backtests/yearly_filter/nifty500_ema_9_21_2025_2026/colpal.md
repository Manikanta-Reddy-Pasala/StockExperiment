# Colgate Palmolive (India) Ltd. (COLPAL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2193.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 50 |
| ALERT2 | 48 |
| ALERT2_SKIP | 20 |
| ALERT3 | 156 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 65 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 31 / 33
- **Target hits / Stop hits / Partials:** 4 / 58 / 2
- **Avg / median % per leg:** 0.93% / -0.03%
- **Sum % (uncompounded):** 59.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 15 | 65.2% | 4 | 19 | 0 | 1.85% | 42.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 15 | 65.2% | 4 | 19 | 0 | 1.85% | 42.6% |
| SELL (all) | 41 | 16 | 39.0% | 0 | 39 | 2 | 0.41% | 16.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.48% | -0.5% |
| SELL @ 3rd Alert (retest2) | 40 | 16 | 40.0% | 0 | 38 | 2 | 0.43% | 17.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.48% | -0.5% |
| retest2 (combined) | 63 | 31 | 49.2% | 4 | 57 | 2 | 0.95% | 59.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2613.60 | 2579.09 | 2574.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 2642.10 | 2612.36 | 2603.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 2684.90 | 2700.18 | 2679.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 10:00:00 | 2684.90 | 2700.18 | 2679.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2691.90 | 2698.52 | 2680.84 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 2627.40 | 2672.18 | 2673.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 2510.20 | 2628.23 | 2648.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2499.50 | 2493.06 | 2531.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 2512.00 | 2505.27 | 2517.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2512.00 | 2505.27 | 2517.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 2477.40 | 2509.00 | 2515.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:45:00 | 2503.40 | 2504.17 | 2509.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 2503.80 | 2505.37 | 2509.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 2482.10 | 2478.31 | 2477.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 2482.10 | 2478.31 | 2477.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 2468.80 | 2476.68 | 2477.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 10:15:00 | 2467.00 | 2474.74 | 2476.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 2443.00 | 2439.79 | 2450.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 2443.00 | 2439.79 | 2450.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 2449.40 | 2442.18 | 2448.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 2449.40 | 2442.18 | 2448.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 2450.00 | 2443.75 | 2448.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 2450.70 | 2443.75 | 2448.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2451.40 | 2445.28 | 2449.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 2447.00 | 2445.28 | 2449.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2450.10 | 2447.45 | 2449.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:15:00 | 2448.50 | 2447.45 | 2449.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:00:00 | 2447.90 | 2447.54 | 2449.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:45:00 | 2444.00 | 2447.03 | 2449.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 2444.80 | 2448.33 | 2449.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2438.50 | 2446.37 | 2448.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 2434.50 | 2444.19 | 2447.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 2436.50 | 2442.66 | 2446.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 2434.90 | 2441.20 | 2445.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 2397.00 | 2385.00 | 2384.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 2397.00 | 2385.00 | 2384.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 09:15:00 | 2404.70 | 2395.59 | 2392.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 13:15:00 | 2398.20 | 2399.25 | 2395.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 14:00:00 | 2398.20 | 2399.25 | 2395.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 2405.70 | 2400.54 | 2396.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 2405.70 | 2400.54 | 2396.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 2401.10 | 2400.37 | 2396.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 2413.70 | 2405.70 | 2399.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 2358.90 | 2413.06 | 2417.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 2358.90 | 2413.06 | 2417.03 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 2405.20 | 2396.39 | 2395.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 2430.50 | 2409.03 | 2402.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 2398.00 | 2406.82 | 2402.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 10:15:00 | 2398.00 | 2406.82 | 2402.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 2398.00 | 2406.82 | 2402.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 2398.00 | 2406.82 | 2402.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 2404.50 | 2406.36 | 2402.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 2408.00 | 2406.36 | 2402.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 2409.60 | 2406.09 | 2403.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 2406.70 | 2406.15 | 2403.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 2410.70 | 2405.98 | 2403.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 2416.50 | 2408.08 | 2405.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 2456.80 | 2408.08 | 2405.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 12:15:00 | 2439.00 | 2447.42 | 2447.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 2439.00 | 2447.42 | 2447.70 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 2454.00 | 2448.63 | 2448.05 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 15:15:00 | 2434.00 | 2445.43 | 2446.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 09:15:00 | 2426.50 | 2441.65 | 2445.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 15:15:00 | 2384.50 | 2382.29 | 2395.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:15:00 | 2399.60 | 2382.29 | 2395.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2401.00 | 2386.03 | 2395.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 2401.00 | 2386.03 | 2395.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2407.80 | 2390.38 | 2396.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2407.80 | 2390.38 | 2396.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2403.70 | 2393.05 | 2397.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:45:00 | 2397.20 | 2394.96 | 2397.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 2398.60 | 2398.81 | 2399.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 2397.20 | 2396.04 | 2397.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 2399.10 | 2391.69 | 2392.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2390.10 | 2387.36 | 2389.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 2390.90 | 2387.36 | 2389.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 2381.00 | 2386.08 | 2389.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 2406.20 | 2392.67 | 2391.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 2406.20 | 2392.67 | 2391.55 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 2364.90 | 2389.00 | 2391.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 2313.00 | 2368.48 | 2379.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2243.10 | 2237.34 | 2264.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 2245.20 | 2237.34 | 2264.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 2230.00 | 2218.81 | 2226.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 2230.00 | 2218.81 | 2226.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 2237.90 | 2222.63 | 2227.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 2238.50 | 2222.63 | 2227.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 2238.70 | 2225.84 | 2228.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 2238.70 | 2225.84 | 2228.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 2244.50 | 2230.42 | 2229.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 2251.60 | 2238.01 | 2233.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 2243.90 | 2247.81 | 2240.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 12:00:00 | 2243.90 | 2247.81 | 2240.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 2254.20 | 2249.09 | 2242.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 2242.20 | 2249.09 | 2242.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2243.00 | 2250.09 | 2245.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 2241.80 | 2250.09 | 2245.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 2250.90 | 2250.25 | 2245.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 2247.00 | 2250.25 | 2245.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 2261.00 | 2252.40 | 2247.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 2254.40 | 2252.40 | 2247.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2242.30 | 2252.51 | 2249.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 2242.00 | 2252.51 | 2249.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2243.40 | 2250.68 | 2248.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 2240.80 | 2250.68 | 2248.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 2241.00 | 2249.01 | 2248.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 2241.00 | 2249.01 | 2248.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 2232.60 | 2245.72 | 2247.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 2231.80 | 2240.95 | 2244.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 2231.90 | 2231.61 | 2238.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:00:00 | 2231.90 | 2231.61 | 2238.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 2240.20 | 2233.33 | 2238.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 2240.20 | 2233.33 | 2238.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 2240.20 | 2234.70 | 2238.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 2243.10 | 2234.70 | 2238.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 2243.00 | 2236.36 | 2238.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 2241.50 | 2236.36 | 2238.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 2235.60 | 2236.39 | 2238.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 2235.60 | 2236.39 | 2238.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2238.80 | 2232.08 | 2235.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2238.80 | 2232.08 | 2235.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2243.00 | 2234.27 | 2236.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 2233.20 | 2234.27 | 2236.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 2230.90 | 2232.81 | 2235.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 2227.50 | 2232.81 | 2235.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 2210.40 | 2211.63 | 2219.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 2217.30 | 2211.63 | 2219.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2224.40 | 2214.40 | 2219.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:45:00 | 2231.20 | 2214.40 | 2219.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2206.20 | 2212.76 | 2217.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:30:00 | 2204.80 | 2211.11 | 2216.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:45:00 | 2205.20 | 2210.09 | 2215.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:15:00 | 2202.70 | 2209.27 | 2214.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 2225.90 | 2191.34 | 2187.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 2225.90 | 2191.34 | 2187.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 2241.50 | 2201.37 | 2192.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 2321.50 | 2340.57 | 2317.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 2321.50 | 2340.57 | 2317.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 2307.50 | 2333.96 | 2316.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 2307.50 | 2333.96 | 2316.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 2321.50 | 2331.47 | 2316.85 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 2290.20 | 2307.73 | 2309.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 2282.80 | 2302.75 | 2307.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 2301.00 | 2291.88 | 2298.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 2301.00 | 2291.88 | 2298.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 2301.00 | 2291.88 | 2298.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 2311.90 | 2291.88 | 2298.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2310.20 | 2295.55 | 2299.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 2310.20 | 2295.55 | 2299.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 2313.90 | 2299.22 | 2301.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:45:00 | 2287.30 | 2298.39 | 2300.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:00:00 | 2299.40 | 2298.60 | 2300.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 2308.60 | 2280.12 | 2286.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 2323.90 | 2288.88 | 2289.61 | SL hit (close>static) qty=1.00 sl=2321.90 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 2348.90 | 2300.88 | 2295.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 2363.30 | 2332.73 | 2315.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 2384.90 | 2398.51 | 2380.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 12:15:00 | 2384.90 | 2398.51 | 2380.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2384.90 | 2398.51 | 2380.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 2384.90 | 2398.51 | 2380.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2378.00 | 2394.41 | 2380.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 2377.90 | 2394.41 | 2380.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 2384.30 | 2392.39 | 2380.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 2473.00 | 2389.71 | 2380.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 2395.50 | 2416.22 | 2418.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 2395.50 | 2416.22 | 2418.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 15:15:00 | 2388.00 | 2402.88 | 2411.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 2400.50 | 2399.38 | 2406.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 2400.50 | 2399.38 | 2406.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 2402.40 | 2399.99 | 2406.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 2406.00 | 2399.99 | 2406.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2402.10 | 2400.42 | 2404.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 2405.90 | 2400.42 | 2404.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2400.10 | 2400.36 | 2404.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 2400.90 | 2400.36 | 2404.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 2400.00 | 2400.29 | 2404.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 2400.00 | 2400.29 | 2404.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 2400.80 | 2400.39 | 2403.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:45:00 | 2402.50 | 2400.39 | 2403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 2403.60 | 2401.03 | 2403.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 2403.60 | 2401.03 | 2403.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 2410.50 | 2402.92 | 2404.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 2410.50 | 2402.92 | 2404.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 2412.40 | 2404.82 | 2405.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 2395.60 | 2404.82 | 2405.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 2364.80 | 2357.81 | 2357.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 2364.80 | 2357.81 | 2357.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 15:15:00 | 2368.80 | 2360.01 | 2358.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 2359.50 | 2359.90 | 2358.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 2359.50 | 2359.90 | 2358.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2359.50 | 2359.90 | 2358.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 2359.50 | 2359.90 | 2358.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 2358.90 | 2359.70 | 2358.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 2358.90 | 2359.70 | 2358.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 2365.90 | 2360.94 | 2359.50 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 2351.50 | 2358.58 | 2358.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 2336.90 | 2354.25 | 2356.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 2351.90 | 2351.80 | 2355.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:30:00 | 2348.60 | 2351.80 | 2355.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 2355.30 | 2351.97 | 2354.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 2355.30 | 2351.97 | 2354.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 2353.80 | 2352.34 | 2354.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 2344.40 | 2351.27 | 2353.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 2227.18 | 2280.47 | 2303.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 2224.30 | 2224.08 | 2242.63 | SL hit (close>ema200) qty=0.50 sl=2224.08 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 2232.30 | 2221.06 | 2220.37 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2203.30 | 2220.86 | 2222.30 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 2217.30 | 2215.56 | 2215.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 2225.60 | 2218.16 | 2216.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 2218.00 | 2219.39 | 2217.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 2218.00 | 2219.39 | 2217.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2218.00 | 2219.39 | 2217.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 2220.80 | 2219.39 | 2217.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 2207.80 | 2217.07 | 2216.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 2207.90 | 2217.07 | 2216.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 2206.60 | 2214.97 | 2215.80 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 2220.40 | 2216.91 | 2216.46 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 2207.50 | 2215.03 | 2215.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 2201.00 | 2209.97 | 2212.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2219.20 | 2207.88 | 2210.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 2219.20 | 2207.88 | 2210.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2219.20 | 2207.88 | 2210.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 2219.60 | 2207.88 | 2210.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 2232.50 | 2212.81 | 2212.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 2271.10 | 2235.88 | 2225.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 2287.20 | 2294.59 | 2273.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 2286.10 | 2294.59 | 2273.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 2275.20 | 2290.71 | 2273.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 2275.20 | 2290.71 | 2273.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 2280.50 | 2288.67 | 2274.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 2270.80 | 2288.67 | 2274.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 2263.00 | 2281.88 | 2274.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 2263.00 | 2281.88 | 2274.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 2243.10 | 2274.12 | 2271.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 2243.10 | 2274.12 | 2271.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 2249.20 | 2269.14 | 2269.78 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 2285.30 | 2271.15 | 2270.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 15:15:00 | 2300.00 | 2285.05 | 2278.30 | Break + close above crossover candle high |

### Cycle 30 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 2209.60 | 2269.96 | 2272.05 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 2262.00 | 2237.65 | 2234.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 2266.60 | 2247.71 | 2240.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2251.00 | 2252.25 | 2244.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2251.00 | 2252.25 | 2244.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2251.00 | 2252.25 | 2244.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2251.00 | 2252.25 | 2244.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 2255.40 | 2252.96 | 2246.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 2247.80 | 2252.96 | 2246.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 2254.00 | 2254.16 | 2248.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 2248.80 | 2254.16 | 2248.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2249.50 | 2253.23 | 2248.84 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 2211.70 | 2241.98 | 2245.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 2195.50 | 2226.99 | 2237.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 2184.90 | 2184.01 | 2196.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 15:00:00 | 2167.70 | 2180.75 | 2193.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2178.00 | 2170.39 | 2177.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 2178.00 | 2170.39 | 2177.74 | SL hit (close>ema400) qty=1.00 sl=2177.74 alert=retest1 |

### Cycle 33 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2190.20 | 2176.06 | 2174.73 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 2176.10 | 2178.09 | 2178.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2168.30 | 2176.13 | 2177.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 2175.00 | 2172.84 | 2174.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 2175.00 | 2172.84 | 2174.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 2175.00 | 2172.84 | 2174.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 2175.00 | 2172.84 | 2174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 2175.40 | 2173.35 | 2175.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 2188.10 | 2173.35 | 2175.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 2187.90 | 2176.26 | 2176.17 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 2176.60 | 2178.58 | 2178.74 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 2182.60 | 2179.39 | 2179.09 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 2175.10 | 2178.24 | 2178.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 2172.90 | 2176.50 | 2177.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 2177.90 | 2176.78 | 2177.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 2177.90 | 2176.78 | 2177.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2177.90 | 2176.78 | 2177.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 2175.50 | 2176.78 | 2177.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 2173.00 | 2176.02 | 2177.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 2176.00 | 2176.02 | 2177.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 2182.30 | 2177.28 | 2177.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 2182.30 | 2177.28 | 2177.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 2182.90 | 2178.40 | 2178.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 2192.10 | 2182.18 | 2180.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 2190.10 | 2190.50 | 2185.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:00:00 | 2190.10 | 2190.50 | 2185.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 2178.60 | 2188.26 | 2184.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 2178.60 | 2188.26 | 2184.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 2180.00 | 2186.61 | 2184.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 2176.40 | 2186.61 | 2184.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2175.10 | 2184.31 | 2183.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 10:45:00 | 2188.40 | 2184.83 | 2183.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 15:15:00 | 2179.00 | 2182.98 | 2183.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 2179.00 | 2182.98 | 2183.50 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 2188.80 | 2184.15 | 2183.98 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 2176.90 | 2182.70 | 2183.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 2171.60 | 2180.48 | 2182.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 2166.00 | 2161.55 | 2168.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 13:30:00 | 2165.00 | 2161.55 | 2168.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2168.40 | 2162.92 | 2168.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2168.40 | 2162.92 | 2168.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2165.00 | 2163.33 | 2168.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 2174.00 | 2163.33 | 2168.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2182.00 | 2167.07 | 2169.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 2182.00 | 2167.07 | 2169.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2177.80 | 2169.21 | 2170.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:00:00 | 2173.70 | 2170.11 | 2170.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 2176.00 | 2171.29 | 2171.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 2176.00 | 2171.29 | 2171.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 2185.40 | 2174.69 | 2172.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 2178.50 | 2179.95 | 2175.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 2178.50 | 2179.95 | 2175.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 2171.50 | 2178.26 | 2175.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 2171.50 | 2178.26 | 2175.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 2160.70 | 2174.75 | 2174.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 2160.70 | 2174.75 | 2174.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 2168.30 | 2173.46 | 2173.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 2153.30 | 2165.24 | 2168.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 11:15:00 | 2156.40 | 2155.68 | 2160.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 2156.40 | 2155.68 | 2160.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2160.60 | 2156.82 | 2159.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 2160.60 | 2156.82 | 2159.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 2156.00 | 2156.66 | 2159.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2127.40 | 2156.66 | 2159.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 2150.70 | 2136.71 | 2134.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 2150.70 | 2136.71 | 2134.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 13:15:00 | 2160.90 | 2143.29 | 2138.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 2151.10 | 2151.99 | 2145.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:15:00 | 2150.70 | 2151.99 | 2145.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 2143.20 | 2150.23 | 2145.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:45:00 | 2140.50 | 2150.23 | 2145.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 2139.10 | 2148.01 | 2144.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 2141.50 | 2148.01 | 2144.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 2146.80 | 2147.51 | 2145.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 2144.90 | 2147.51 | 2145.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 2152.80 | 2148.57 | 2145.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 2139.60 | 2148.57 | 2145.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 2160.90 | 2151.03 | 2147.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 2151.10 | 2151.03 | 2147.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2154.70 | 2153.91 | 2149.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:45:00 | 2150.70 | 2153.91 | 2149.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 2152.00 | 2157.89 | 2153.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 2150.60 | 2157.89 | 2153.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 2147.50 | 2155.81 | 2152.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 2147.50 | 2155.81 | 2152.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 2142.00 | 2153.05 | 2151.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:45:00 | 2142.20 | 2153.05 | 2151.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 2139.80 | 2150.40 | 2150.63 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 2162.30 | 2150.09 | 2149.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 2166.40 | 2158.26 | 2155.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 2163.80 | 2166.44 | 2161.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 2163.80 | 2166.44 | 2161.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2163.80 | 2166.44 | 2161.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 2163.80 | 2166.44 | 2161.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2162.80 | 2165.71 | 2162.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:30:00 | 2159.90 | 2165.71 | 2162.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2163.10 | 2165.19 | 2162.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 2163.10 | 2165.19 | 2162.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 2161.80 | 2164.51 | 2162.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 2161.90 | 2164.51 | 2162.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 2158.00 | 2163.21 | 2161.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 2132.80 | 2163.21 | 2161.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 2123.70 | 2155.31 | 2158.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 2112.40 | 2141.40 | 2151.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 2101.10 | 2090.98 | 2102.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 2101.10 | 2090.98 | 2102.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2101.10 | 2090.98 | 2102.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 2097.60 | 2090.98 | 2102.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 2112.80 | 2095.34 | 2103.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 2112.80 | 2095.34 | 2103.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 2111.80 | 2098.63 | 2103.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 2106.70 | 2098.63 | 2103.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:30:00 | 2107.80 | 2104.78 | 2104.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 2108.40 | 2105.50 | 2105.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 2108.40 | 2105.50 | 2105.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 2118.70 | 2108.14 | 2106.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 2106.20 | 2108.23 | 2106.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 2106.20 | 2108.23 | 2106.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2106.20 | 2108.23 | 2106.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 2107.30 | 2108.23 | 2106.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2102.80 | 2107.14 | 2106.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 2102.80 | 2107.14 | 2106.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2103.80 | 2106.47 | 2106.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 2105.80 | 2106.47 | 2106.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 2105.00 | 2106.02 | 2106.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 2104.30 | 2105.67 | 2105.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 2104.30 | 2105.67 | 2105.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 2103.00 | 2105.14 | 2105.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 2108.50 | 2105.73 | 2105.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 11:15:00 | 2108.50 | 2105.73 | 2105.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 2108.50 | 2105.73 | 2105.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 2108.10 | 2105.73 | 2105.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 2101.50 | 2104.88 | 2105.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 2106.80 | 2104.88 | 2105.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2094.00 | 2097.78 | 2101.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 2087.10 | 2098.63 | 2101.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 2088.60 | 2073.33 | 2072.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 2088.60 | 2073.33 | 2072.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 2102.10 | 2091.23 | 2086.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 14:15:00 | 2093.20 | 2095.55 | 2090.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 2093.20 | 2095.55 | 2090.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 2095.00 | 2095.44 | 2091.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 2076.10 | 2095.44 | 2091.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2077.90 | 2091.93 | 2089.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 2076.60 | 2091.93 | 2089.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2078.00 | 2089.14 | 2088.78 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 2083.00 | 2087.92 | 2088.26 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 2091.40 | 2088.85 | 2088.51 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 2072.70 | 2085.73 | 2087.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 2069.30 | 2082.44 | 2085.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 2068.60 | 2055.16 | 2064.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 2068.60 | 2055.16 | 2064.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2068.60 | 2055.16 | 2064.51 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 2074.90 | 2063.66 | 2063.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 2082.80 | 2067.49 | 2064.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 13:15:00 | 2093.70 | 2098.53 | 2090.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 14:00:00 | 2093.70 | 2098.53 | 2090.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 2090.30 | 2096.88 | 2090.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 2091.90 | 2096.88 | 2090.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 2097.80 | 2097.06 | 2091.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 2102.20 | 2097.06 | 2091.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 11:30:00 | 2099.00 | 2098.23 | 2093.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 2101.30 | 2098.68 | 2093.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 2116.10 | 2122.40 | 2123.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 2116.10 | 2122.40 | 2123.16 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 2140.00 | 2124.44 | 2123.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 10:15:00 | 2159.10 | 2131.37 | 2126.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 2167.60 | 2168.81 | 2156.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:45:00 | 2168.50 | 2168.81 | 2156.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2149.00 | 2165.02 | 2157.11 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 2138.50 | 2150.22 | 2151.57 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2161.70 | 2152.80 | 2152.52 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 2127.30 | 2147.70 | 2150.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 2114.30 | 2138.90 | 2143.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 2112.90 | 2103.94 | 2116.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 2112.90 | 2103.94 | 2116.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2112.90 | 2103.94 | 2116.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 2112.90 | 2103.94 | 2116.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2112.00 | 2105.55 | 2115.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 2100.50 | 2104.37 | 2112.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 2096.20 | 2104.77 | 2110.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:45:00 | 2097.00 | 2102.55 | 2108.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 2127.70 | 2113.18 | 2111.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2127.70 | 2113.18 | 2111.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 2128.90 | 2116.32 | 2113.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 10:15:00 | 2109.00 | 2115.06 | 2113.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 10:15:00 | 2109.00 | 2115.06 | 2113.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 2109.00 | 2115.06 | 2113.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:45:00 | 2108.00 | 2115.06 | 2113.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 2125.00 | 2117.05 | 2114.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 13:30:00 | 2135.00 | 2123.14 | 2117.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2144.60 | 2123.48 | 2122.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 14:15:00 | 2113.70 | 2123.06 | 2123.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 14:15:00 | 2113.70 | 2123.06 | 2123.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 2111.00 | 2120.65 | 2122.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 2121.90 | 2116.08 | 2119.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 2121.90 | 2116.08 | 2119.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2121.90 | 2116.08 | 2119.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 2121.90 | 2116.08 | 2119.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 2126.80 | 2118.22 | 2120.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 2126.00 | 2118.22 | 2120.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 2133.00 | 2122.10 | 2121.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 2137.70 | 2125.22 | 2123.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 2143.10 | 2146.06 | 2137.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:45:00 | 2142.40 | 2146.06 | 2137.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 2140.20 | 2144.67 | 2138.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 2139.10 | 2144.67 | 2138.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 2158.50 | 2147.44 | 2140.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 2142.00 | 2147.44 | 2140.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2158.20 | 2168.41 | 2160.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:15:00 | 2151.50 | 2168.41 | 2160.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2144.50 | 2163.63 | 2159.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 2144.50 | 2163.63 | 2159.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 2141.70 | 2154.68 | 2156.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 2138.00 | 2151.34 | 2154.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 2122.50 | 2119.32 | 2128.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 2122.50 | 2119.32 | 2128.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2137.40 | 2122.56 | 2128.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 2134.50 | 2122.56 | 2128.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 2157.80 | 2129.61 | 2130.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 2157.80 | 2129.61 | 2130.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2142.50 | 2132.19 | 2131.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 2170.00 | 2146.57 | 2139.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 2183.00 | 2191.86 | 2172.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 2183.00 | 2191.86 | 2172.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 2179.80 | 2189.45 | 2173.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 2176.00 | 2189.45 | 2173.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 2168.00 | 2185.16 | 2172.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 2168.00 | 2185.16 | 2172.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 2160.10 | 2180.15 | 2171.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 2160.10 | 2180.15 | 2171.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 2172.30 | 2178.58 | 2171.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 2192.90 | 2182.46 | 2174.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2231.50 | 2254.05 | 2254.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 2231.50 | 2254.05 | 2254.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 2198.50 | 2222.55 | 2235.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 2199.10 | 2190.70 | 2203.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 2199.10 | 2190.70 | 2203.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2199.10 | 2190.70 | 2203.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 2212.20 | 2190.70 | 2203.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2205.90 | 2193.74 | 2203.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 2219.00 | 2193.74 | 2203.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2217.70 | 2198.53 | 2204.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2217.00 | 2198.53 | 2204.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2207.80 | 2200.39 | 2205.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 2159.90 | 2201.90 | 2204.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 10:15:00 | 2195.80 | 2171.43 | 2181.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 10:45:00 | 2193.40 | 2176.38 | 2182.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:00:00 | 2189.10 | 2178.93 | 2183.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 2199.00 | 2185.45 | 2185.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 2199.00 | 2185.45 | 2185.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 2192.20 | 2186.80 | 2186.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 2192.20 | 2186.80 | 2186.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 2210.40 | 2191.52 | 2188.50 | Break + close above crossover candle high |

### Cycle 68 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 2112.00 | 2175.62 | 2181.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 2069.00 | 2131.93 | 2158.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 1949.00 | 1944.81 | 1971.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:45:00 | 1953.60 | 1944.81 | 1971.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 1951.00 | 1944.02 | 1953.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:00:00 | 1951.00 | 1944.02 | 1953.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1942.90 | 1943.80 | 1952.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:15:00 | 1942.00 | 1943.80 | 1952.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 1844.90 | 1885.52 | 1905.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1862.50 | 1858.71 | 1877.22 | SL hit (close>ema200) qty=0.50 sl=1858.71 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1907.20 | 1884.79 | 1883.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1916.50 | 1897.13 | 1890.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1900.00 | 1903.94 | 1895.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1900.00 | 1903.94 | 1895.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1900.00 | 1903.94 | 1895.46 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1876.30 | 1892.86 | 1892.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1840.30 | 1881.09 | 1887.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1840.40 | 1824.59 | 1849.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:45:00 | 1839.20 | 1824.59 | 1849.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1828.00 | 1815.46 | 1827.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 1828.00 | 1815.46 | 1827.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 1829.00 | 1818.17 | 1828.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 1829.00 | 1818.17 | 1828.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1835.30 | 1821.59 | 1828.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1816.20 | 1821.59 | 1828.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1825.30 | 1822.33 | 1828.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 1805.90 | 1825.24 | 1827.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 1839.50 | 1830.76 | 1829.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 1839.50 | 1830.76 | 1829.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 1851.20 | 1836.29 | 1832.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1896.60 | 1903.47 | 1882.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 1896.60 | 1903.47 | 1882.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1902.90 | 1924.02 | 1909.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1923.50 | 1924.10 | 1911.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1919.70 | 1922.59 | 1912.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1920.70 | 1922.37 | 1913.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1929.70 | 1919.90 | 1913.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 10:15:00 | 2115.85 | 2014.83 | 1975.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 2109.10 | 2140.16 | 2142.00 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 2162.00 | 2140.22 | 2137.67 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 2122.00 | 2135.66 | 2136.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 2097.20 | 2127.97 | 2132.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 2110.00 | 2107.78 | 2119.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 2153.70 | 2107.78 | 2119.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2165.50 | 2119.32 | 2123.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 2165.50 | 2119.32 | 2123.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 2168.50 | 2129.16 | 2127.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 2176.00 | 2154.74 | 2142.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 11:15:00 | 2165.90 | 2169.88 | 2160.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:00:00 | 2165.90 | 2169.88 | 2160.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 2160.60 | 2168.03 | 2160.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 2160.60 | 2168.03 | 2160.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 2163.00 | 2167.02 | 2160.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:00:00 | 2163.00 | 2167.02 | 2160.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 2159.10 | 2165.44 | 2160.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:45:00 | 2158.70 | 2165.44 | 2160.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 2162.00 | 2164.75 | 2160.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 2158.40 | 2164.75 | 2160.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 2152.40 | 2162.28 | 2159.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:45:00 | 2149.50 | 2162.28 | 2159.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 2167.20 | 2163.26 | 2160.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 2175.70 | 2165.01 | 2161.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 2170.00 | 2168.02 | 2164.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:00:00 | 2170.00 | 2168.73 | 2165.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:30:00 | 2171.10 | 2172.00 | 2167.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-28 09:15:00 | 2477.40 | 2025-06-04 14:15:00 | 2482.10 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-05-28 13:45:00 | 2503.40 | 2025-06-04 14:15:00 | 2482.10 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2025-05-29 09:15:00 | 2503.80 | 2025-06-04 14:15:00 | 2482.10 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-06-10 11:15:00 | 2448.50 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2025-06-10 12:00:00 | 2447.90 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-06-10 12:45:00 | 2444.00 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2025-06-11 09:15:00 | 2444.80 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 1.96% |
| SELL | retest2 | 2025-06-11 10:30:00 | 2434.50 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-06-11 12:00:00 | 2436.50 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2025-06-11 12:30:00 | 2434.90 | 2025-06-18 10:15:00 | 2397.00 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-06-23 10:30:00 | 2413.70 | 2025-06-26 09:15:00 | 2358.90 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-07-01 12:15:00 | 2408.00 | 2025-07-08 12:15:00 | 2439.00 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-07-01 15:00:00 | 2409.60 | 2025-07-08 12:15:00 | 2439.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-07-02 10:00:00 | 2406.70 | 2025-07-08 12:15:00 | 2439.00 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-07-02 11:15:00 | 2410.70 | 2025-07-08 12:15:00 | 2439.00 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2025-07-02 12:15:00 | 2456.80 | 2025-07-08 12:15:00 | 2439.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-07-15 12:45:00 | 2397.20 | 2025-07-21 09:15:00 | 2406.20 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-07-16 09:15:00 | 2398.60 | 2025-07-21 09:15:00 | 2406.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-16 11:30:00 | 2397.20 | 2025-07-21 09:15:00 | 2406.20 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-07-17 14:30:00 | 2399.10 | 2025-07-21 09:15:00 | 2406.20 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-12 11:30:00 | 2204.80 | 2025-08-18 11:15:00 | 2225.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-08-12 12:45:00 | 2205.20 | 2025-08-18 11:15:00 | 2225.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-12 14:15:00 | 2202.70 | 2025-08-18 11:15:00 | 2225.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-08-26 12:45:00 | 2287.30 | 2025-08-29 10:15:00 | 2323.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-26 14:00:00 | 2299.40 | 2025-08-29 10:15:00 | 2323.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-29 09:45:00 | 2308.60 | 2025-08-29 10:15:00 | 2323.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-04 09:15:00 | 2473.00 | 2025-09-08 12:15:00 | 2395.50 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-09-11 09:15:00 | 2395.60 | 2025-09-18 14:15:00 | 2364.80 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-09-22 13:45:00 | 2344.40 | 2025-09-26 09:15:00 | 2227.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 2344.40 | 2025-09-30 10:15:00 | 2224.30 | STOP_HIT | 0.50 | 5.12% |
| SELL | retest1 | 2025-11-06 15:00:00 | 2167.70 | 2025-11-10 10:15:00 | 2178.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-10 12:15:00 | 2172.00 | 2025-11-11 14:15:00 | 2181.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-10 15:00:00 | 2169.80 | 2025-11-11 14:15:00 | 2181.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-11-21 10:45:00 | 2188.40 | 2025-11-21 15:15:00 | 2179.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-11-26 12:00:00 | 2173.70 | 2025-11-26 12:15:00 | 2176.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2127.40 | 2025-12-05 11:15:00 | 2150.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-22 09:15:00 | 2106.70 | 2025-12-22 15:15:00 | 2108.40 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-12-22 14:30:00 | 2107.80 | 2025-12-22 15:15:00 | 2108.40 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-12-23 14:15:00 | 2105.80 | 2025-12-23 15:15:00 | 2104.30 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-23 14:45:00 | 2105.00 | 2025-12-23 15:15:00 | 2104.30 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-12-26 15:00:00 | 2087.10 | 2026-01-01 09:15:00 | 2088.60 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-01-16 09:15:00 | 2102.20 | 2026-01-21 11:15:00 | 2116.10 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-01-16 11:30:00 | 2099.00 | 2026-01-21 11:15:00 | 2116.10 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2026-01-16 12:45:00 | 2101.30 | 2026-01-21 11:15:00 | 2116.10 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-02-01 11:45:00 | 2100.50 | 2026-02-02 14:15:00 | 2127.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-02-01 15:15:00 | 2096.20 | 2026-02-02 14:15:00 | 2127.70 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-02 10:45:00 | 2097.00 | 2026-02-02 14:15:00 | 2127.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-02-03 13:30:00 | 2135.00 | 2026-02-05 14:15:00 | 2113.70 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-05 09:15:00 | 2144.60 | 2026-02-05 14:15:00 | 2113.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-02-20 09:30:00 | 2192.90 | 2026-03-02 09:15:00 | 2231.50 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2026-03-09 09:15:00 | 2159.90 | 2026-03-10 14:15:00 | 2192.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-03-10 10:15:00 | 2195.80 | 2026-03-10 14:15:00 | 2192.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2026-03-10 10:45:00 | 2193.40 | 2026-03-10 14:15:00 | 2192.20 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2026-03-10 12:00:00 | 2189.10 | 2026-03-10 14:15:00 | 2192.20 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-03-18 15:15:00 | 1942.00 | 2026-03-23 10:15:00 | 1844.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 15:15:00 | 1942.00 | 2026-03-24 11:15:00 | 1862.50 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1805.90 | 2026-04-07 12:15:00 | 1839.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1923.50 | 2026-04-17 10:15:00 | 2115.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1919.70 | 2026-04-17 10:15:00 | 2111.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1920.70 | 2026-04-17 10:15:00 | 2112.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1929.70 | 2026-04-20 09:15:00 | 2122.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-21 14:15:00 | 2091.90 | 2026-04-28 10:15:00 | 2109.10 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2026-04-21 14:45:00 | 2090.30 | 2026-04-28 10:15:00 | 2109.10 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2026-04-23 09:45:00 | 2087.30 | 2026-04-28 10:15:00 | 2109.10 | STOP_HIT | 1.00 | 1.04% |
