# Grasim Industries Ltd. (GRASIM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 2965.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 167 |
| ALERT1 | 109 |
| ALERT2 | 108 |
| ALERT2_SKIP | 62 |
| ALERT3 | 314 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 129 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 129 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 138 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 95
- **Target hits / Stop hits / Partials:** 2 / 129 / 7
- **Avg / median % per leg:** 0.01% / -0.70%
- **Sum % (uncompounded):** 1.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 17 | 27.9% | 2 | 59 | 0 | -0.10% | -5.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.21% | -4.4% |
| BUY @ 3rd Alert (retest2) | 59 | 17 | 28.8% | 2 | 57 | 0 | -0.02% | -1.4% |
| SELL (all) | 77 | 26 | 33.8% | 0 | 70 | 7 | 0.10% | 7.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 77 | 26 | 33.8% | 0 | 70 | 7 | 0.10% | 7.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.21% | -4.4% |
| retest2 (combined) | 136 | 43 | 31.6% | 2 | 127 | 7 | 0.05% | 6.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 2392.00 | 2376.11 | 2374.12 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 15:15:00 | 2370.00 | 2375.20 | 2375.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 2341.15 | 2367.73 | 2371.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 2370.65 | 2355.13 | 2363.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 2370.65 | 2355.13 | 2363.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2370.65 | 2355.13 | 2363.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2370.65 | 2355.13 | 2363.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 2379.00 | 2359.90 | 2365.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 2409.25 | 2359.90 | 2365.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 2415.60 | 2377.16 | 2372.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 14:15:00 | 2421.15 | 2399.06 | 2385.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 2430.55 | 2435.68 | 2421.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 09:45:00 | 2438.80 | 2435.68 | 2421.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 2424.00 | 2433.35 | 2421.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 2424.00 | 2433.35 | 2421.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 2431.95 | 2433.07 | 2422.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:30:00 | 2437.40 | 2432.87 | 2424.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 2404.00 | 2427.38 | 2423.75 | SL hit (close<static) qty=1.00 sl=2420.95 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 2413.05 | 2420.96 | 2421.24 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 2434.55 | 2423.68 | 2422.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 2446.20 | 2428.18 | 2424.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 2440.05 | 2449.92 | 2440.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 2440.05 | 2449.92 | 2440.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 2440.05 | 2449.92 | 2440.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 2440.05 | 2449.92 | 2440.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 2443.65 | 2448.66 | 2441.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 2437.85 | 2448.66 | 2441.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2420.20 | 2442.97 | 2439.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 2420.20 | 2442.97 | 2439.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 2407.00 | 2435.78 | 2436.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 12:15:00 | 2392.10 | 2422.00 | 2429.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 11:15:00 | 2427.30 | 2410.42 | 2418.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 11:15:00 | 2427.30 | 2410.42 | 2418.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 2427.30 | 2410.42 | 2418.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 2421.45 | 2410.42 | 2418.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 2428.45 | 2414.02 | 2419.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:00:00 | 2428.45 | 2414.02 | 2419.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 2441.00 | 2419.42 | 2421.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:30:00 | 2441.45 | 2419.42 | 2421.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 2440.50 | 2423.63 | 2423.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 2455.80 | 2435.08 | 2428.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 13:15:00 | 2423.80 | 2436.59 | 2430.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 2423.80 | 2436.59 | 2430.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 2423.80 | 2436.59 | 2430.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 2423.80 | 2436.59 | 2430.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 2399.40 | 2429.16 | 2428.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 2399.40 | 2429.16 | 2428.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 15:15:00 | 2399.00 | 2423.12 | 2425.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 2366.85 | 2411.87 | 2420.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 2404.85 | 2351.08 | 2365.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 2404.85 | 2351.08 | 2365.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2404.85 | 2351.08 | 2365.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 2404.85 | 2351.08 | 2365.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 2390.45 | 2358.95 | 2367.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:45:00 | 2384.95 | 2363.81 | 2369.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 2265.70 | 2329.30 | 2351.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 12:15:00 | 2262.20 | 2258.96 | 2295.57 | SL hit (close>ema200) qty=0.50 sl=2258.96 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 2322.40 | 2307.72 | 2306.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 2335.30 | 2313.23 | 2309.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 2448.10 | 2452.04 | 2424.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 09:15:00 | 2448.50 | 2452.04 | 2424.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 2436.10 | 2451.07 | 2440.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 2436.10 | 2451.07 | 2440.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 2433.15 | 2447.48 | 2439.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 2433.15 | 2447.48 | 2439.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 2437.95 | 2445.58 | 2439.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:30:00 | 2433.60 | 2445.58 | 2439.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 2446.00 | 2445.66 | 2440.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 12:45:00 | 2441.85 | 2445.66 | 2440.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 2445.85 | 2445.70 | 2440.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:45:00 | 2447.80 | 2445.70 | 2440.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 2460.85 | 2448.73 | 2442.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:30:00 | 2450.70 | 2448.73 | 2442.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 2518.00 | 2463.75 | 2450.43 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 2422.00 | 2452.86 | 2456.75 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 2467.00 | 2454.70 | 2453.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 13:15:00 | 2497.95 | 2463.35 | 2457.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 2476.30 | 2479.36 | 2469.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 12:00:00 | 2476.30 | 2479.36 | 2469.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 2464.00 | 2476.29 | 2468.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 2464.00 | 2476.29 | 2468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 2481.30 | 2477.29 | 2470.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:30:00 | 2469.80 | 2477.29 | 2470.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 2462.40 | 2474.31 | 2469.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 2462.40 | 2474.31 | 2469.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 2468.00 | 2473.05 | 2469.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 2461.45 | 2473.05 | 2469.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 2482.20 | 2474.88 | 2470.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:30:00 | 2483.45 | 2477.42 | 2471.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:00:00 | 2487.60 | 2477.42 | 2471.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-01 14:15:00 | 2731.80 | 2695.86 | 2655.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 2720.00 | 2732.64 | 2734.02 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 15:15:00 | 2738.30 | 2734.95 | 2734.91 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 2732.20 | 2734.40 | 2734.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 2723.00 | 2731.95 | 2733.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 2736.25 | 2731.92 | 2733.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 2736.25 | 2731.92 | 2733.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 2736.25 | 2731.92 | 2733.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 2736.25 | 2731.92 | 2733.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 2752.10 | 2735.95 | 2734.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 15:15:00 | 2784.80 | 2745.72 | 2739.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 12:15:00 | 2793.75 | 2801.82 | 2780.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 13:00:00 | 2793.75 | 2801.82 | 2780.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2809.00 | 2827.43 | 2811.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2796.00 | 2827.43 | 2811.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 2804.00 | 2822.74 | 2810.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 2805.00 | 2822.74 | 2810.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 2795.05 | 2817.20 | 2809.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:45:00 | 2793.45 | 2817.20 | 2809.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 2798.50 | 2812.76 | 2808.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 2798.50 | 2812.76 | 2808.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2803.80 | 2810.97 | 2808.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 2822.25 | 2809.26 | 2807.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 11:15:00 | 2792.60 | 2815.61 | 2815.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 2792.60 | 2815.61 | 2815.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 2762.00 | 2794.82 | 2804.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2796.00 | 2769.65 | 2783.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2796.00 | 2769.65 | 2783.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2796.00 | 2769.65 | 2783.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 2796.00 | 2769.65 | 2783.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2797.00 | 2775.12 | 2784.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 2797.00 | 2775.12 | 2784.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2811.95 | 2784.05 | 2786.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 2816.40 | 2784.05 | 2786.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 2803.65 | 2787.97 | 2788.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 2817.40 | 2787.97 | 2788.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 2813.90 | 2793.16 | 2790.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 2841.50 | 2806.74 | 2797.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 2782.00 | 2811.40 | 2805.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 2782.00 | 2811.40 | 2805.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2782.00 | 2811.40 | 2805.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 2789.10 | 2811.40 | 2805.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2794.10 | 2807.94 | 2804.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 2775.10 | 2807.94 | 2804.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 2782.70 | 2799.99 | 2801.44 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 2821.00 | 2805.65 | 2803.85 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 2796.40 | 2802.23 | 2802.70 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 2829.90 | 2806.68 | 2804.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 2851.75 | 2815.69 | 2808.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 2833.00 | 2837.80 | 2825.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 2838.90 | 2837.80 | 2825.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 2837.85 | 2837.81 | 2826.51 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 2807.30 | 2822.89 | 2824.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 12:15:00 | 2796.55 | 2817.62 | 2821.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2660.00 | 2645.48 | 2686.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:45:00 | 2661.35 | 2645.48 | 2686.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 2610.95 | 2583.41 | 2610.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:00:00 | 2610.95 | 2583.41 | 2610.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 2608.35 | 2588.40 | 2610.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 2608.35 | 2588.40 | 2610.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 2628.45 | 2596.41 | 2612.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 2628.45 | 2596.41 | 2612.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 2623.55 | 2601.84 | 2613.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:30:00 | 2624.90 | 2601.84 | 2613.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 2589.60 | 2599.39 | 2611.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 2603.75 | 2599.39 | 2611.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 2592.25 | 2590.84 | 2603.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 2602.90 | 2590.84 | 2603.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 2598.80 | 2592.43 | 2603.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 2595.00 | 2592.43 | 2603.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 2592.35 | 2592.42 | 2602.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:30:00 | 2600.50 | 2592.42 | 2602.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2554.40 | 2525.74 | 2540.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 2554.40 | 2525.74 | 2540.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 2527.25 | 2526.04 | 2539.14 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 2589.30 | 2547.66 | 2546.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 2602.00 | 2558.53 | 2551.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 2713.85 | 2731.36 | 2693.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 2713.85 | 2731.36 | 2693.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 2735.65 | 2738.33 | 2716.56 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 2703.70 | 2719.09 | 2720.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 2700.00 | 2715.28 | 2718.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 11:15:00 | 2712.70 | 2709.63 | 2714.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 11:15:00 | 2712.70 | 2709.63 | 2714.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 2712.70 | 2709.63 | 2714.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:45:00 | 2711.35 | 2709.63 | 2714.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 2705.60 | 2708.82 | 2713.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 2702.40 | 2709.76 | 2713.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 2720.00 | 2711.81 | 2714.01 | SL hit (close>static) qty=1.00 sl=2719.95 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 2704.00 | 2697.22 | 2696.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 2722.20 | 2702.22 | 2698.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 2696.05 | 2700.98 | 2698.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 2696.05 | 2700.98 | 2698.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 2696.05 | 2700.98 | 2698.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 2696.05 | 2700.98 | 2698.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 2686.05 | 2698.00 | 2697.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 2678.65 | 2698.00 | 2697.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 2683.50 | 2695.10 | 2696.20 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 2704.55 | 2695.76 | 2695.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 2740.70 | 2705.59 | 2699.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 2738.55 | 2744.98 | 2732.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:45:00 | 2741.10 | 2744.98 | 2732.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2704.35 | 2736.06 | 2730.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2704.35 | 2736.06 | 2730.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2692.85 | 2727.42 | 2727.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 2692.85 | 2727.42 | 2727.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 11:15:00 | 2704.35 | 2722.80 | 2725.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 2681.50 | 2714.54 | 2721.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 2704.20 | 2699.83 | 2708.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 13:00:00 | 2704.20 | 2699.83 | 2708.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 2695.00 | 2698.86 | 2707.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 14:15:00 | 2694.50 | 2698.86 | 2707.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:45:00 | 2693.40 | 2701.67 | 2706.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 2682.90 | 2701.67 | 2706.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 2692.70 | 2701.08 | 2703.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 2707.90 | 2702.44 | 2704.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 2717.35 | 2705.42 | 2705.46 | SL hit (close>static) qty=1.00 sl=2712.95 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 2723.00 | 2708.94 | 2707.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 2727.60 | 2712.67 | 2708.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 2692.00 | 2708.54 | 2707.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 2692.00 | 2708.54 | 2707.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 2692.00 | 2708.54 | 2707.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 2692.00 | 2708.54 | 2707.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 2680.30 | 2702.89 | 2704.92 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 2737.75 | 2707.42 | 2704.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 2761.80 | 2718.30 | 2710.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 2766.50 | 2776.74 | 2759.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:00:00 | 2766.50 | 2776.74 | 2759.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 2765.35 | 2774.46 | 2759.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 2765.35 | 2774.46 | 2759.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2739.50 | 2766.32 | 2758.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 2739.50 | 2766.32 | 2758.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 2746.05 | 2762.26 | 2757.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 2745.80 | 2762.26 | 2757.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 2755.40 | 2760.26 | 2757.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:45:00 | 2756.90 | 2760.26 | 2757.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 2758.75 | 2759.96 | 2757.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 2774.00 | 2759.91 | 2757.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 10:15:00 | 2750.65 | 2758.06 | 2757.26 | SL hit (close<static) qty=1.00 sl=2754.65 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 2744.65 | 2755.38 | 2756.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 2730.85 | 2750.47 | 2753.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 2769.50 | 2743.95 | 2748.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 2769.50 | 2743.95 | 2748.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2769.50 | 2743.95 | 2748.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 2753.35 | 2743.95 | 2748.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 2761.00 | 2747.36 | 2749.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 2749.80 | 2747.36 | 2749.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 14:15:00 | 2612.31 | 2644.40 | 2672.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 2624.65 | 2624.20 | 2652.57 | SL hit (close>ema200) qty=0.50 sl=2624.20 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 2687.50 | 2659.39 | 2658.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 2700.85 | 2667.68 | 2662.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 2789.65 | 2793.91 | 2766.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 2785.55 | 2793.91 | 2766.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 2773.60 | 2789.85 | 2766.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 2773.60 | 2789.85 | 2766.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 2777.00 | 2787.28 | 2767.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 2764.25 | 2787.28 | 2767.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 2779.90 | 2786.26 | 2770.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 2772.85 | 2786.26 | 2770.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2824.45 | 2799.43 | 2782.21 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 2754.95 | 2777.13 | 2778.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 2729.70 | 2749.10 | 2760.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 2735.80 | 2732.10 | 2744.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 2735.80 | 2732.10 | 2744.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 2737.55 | 2733.19 | 2743.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 2738.20 | 2733.19 | 2743.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 2745.00 | 2736.49 | 2743.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 2722.00 | 2736.49 | 2743.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2719.95 | 2733.18 | 2741.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 2706.25 | 2721.68 | 2728.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 2704.90 | 2713.67 | 2718.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 2711.00 | 2717.38 | 2718.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:00:00 | 2709.40 | 2716.88 | 2718.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 2717.45 | 2716.99 | 2718.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:45:00 | 2717.05 | 2716.99 | 2718.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 2738.55 | 2721.31 | 2719.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 2738.55 | 2721.31 | 2719.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 2757.50 | 2729.54 | 2723.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 2748.85 | 2755.80 | 2743.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 2748.85 | 2755.80 | 2743.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2748.85 | 2755.80 | 2743.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 2743.50 | 2755.80 | 2743.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 2731.60 | 2753.54 | 2744.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 2731.60 | 2753.54 | 2744.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 2720.75 | 2746.99 | 2742.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:00:00 | 2720.75 | 2746.99 | 2742.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 2688.80 | 2735.35 | 2737.84 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 2759.55 | 2738.65 | 2736.10 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 2732.15 | 2735.02 | 2735.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 2716.60 | 2730.07 | 2732.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 2687.00 | 2658.29 | 2676.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 2687.00 | 2658.29 | 2676.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2687.00 | 2658.29 | 2676.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 2687.00 | 2658.29 | 2676.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 2691.15 | 2664.87 | 2677.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 2691.15 | 2664.87 | 2677.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 2674.90 | 2666.87 | 2677.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 2663.70 | 2666.87 | 2677.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 2668.55 | 2666.62 | 2675.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 2675.00 | 2647.42 | 2647.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 2675.00 | 2647.42 | 2647.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 2679.30 | 2661.32 | 2654.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 2667.30 | 2679.97 | 2670.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 14:15:00 | 2667.30 | 2679.97 | 2670.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 2667.30 | 2679.97 | 2670.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 2667.30 | 2679.97 | 2670.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 2678.75 | 2679.73 | 2671.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 2660.00 | 2679.73 | 2671.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2675.10 | 2678.80 | 2671.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 2656.95 | 2678.80 | 2671.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 2684.50 | 2679.94 | 2672.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:45:00 | 2679.50 | 2679.94 | 2672.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 2674.85 | 2680.11 | 2674.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 2674.85 | 2680.11 | 2674.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 2680.95 | 2680.28 | 2674.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 2697.10 | 2683.64 | 2676.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:30:00 | 2692.90 | 2688.91 | 2681.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 2651.25 | 2681.38 | 2678.47 | SL hit (close<static) qty=1.00 sl=2672.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2609.00 | 2666.90 | 2672.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 2592.60 | 2642.46 | 2659.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 2628.00 | 2619.89 | 2641.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:00:00 | 2628.00 | 2619.89 | 2641.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 2651.25 | 2624.42 | 2634.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 2651.25 | 2624.42 | 2634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 2655.65 | 2630.67 | 2636.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 2652.10 | 2630.67 | 2636.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 2622.30 | 2631.77 | 2636.33 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 2648.90 | 2640.12 | 2639.46 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 2583.90 | 2630.25 | 2635.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 2575.00 | 2619.20 | 2629.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2530.00 | 2529.26 | 2551.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:00:00 | 2530.00 | 2529.26 | 2551.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 2530.00 | 2510.41 | 2522.02 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 2531.70 | 2521.57 | 2521.55 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 2508.10 | 2518.88 | 2520.32 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 2530.00 | 2522.99 | 2522.06 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 2510.05 | 2521.11 | 2521.55 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 2537.00 | 2524.48 | 2522.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 2548.50 | 2529.28 | 2525.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 2502.20 | 2526.05 | 2524.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 2502.20 | 2526.05 | 2524.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 2502.20 | 2526.05 | 2524.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 2502.20 | 2526.05 | 2524.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 2500.85 | 2521.01 | 2522.50 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 12:15:00 | 2553.65 | 2528.86 | 2525.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 2581.40 | 2541.27 | 2532.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 2612.85 | 2614.78 | 2592.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 2612.85 | 2614.78 | 2592.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2607.45 | 2613.10 | 2602.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 2604.85 | 2613.10 | 2602.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 2601.40 | 2610.76 | 2602.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 2602.20 | 2610.76 | 2602.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 2598.00 | 2608.21 | 2602.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 2601.00 | 2608.21 | 2602.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 2627.85 | 2612.14 | 2604.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:15:00 | 2632.30 | 2614.76 | 2606.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 2589.60 | 2609.20 | 2607.27 | SL hit (close<static) qty=1.00 sl=2597.95 alert=retest2 |

### Cycle 50 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 2586.00 | 2604.56 | 2605.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 2562.60 | 2596.17 | 2601.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 2593.65 | 2591.04 | 2597.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 2593.65 | 2591.04 | 2597.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 2593.65 | 2591.04 | 2597.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 2599.95 | 2591.04 | 2597.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2605.75 | 2593.98 | 2598.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:45:00 | 2607.80 | 2593.98 | 2598.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2606.70 | 2596.53 | 2598.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 2606.70 | 2596.53 | 2598.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 2611.15 | 2599.45 | 2599.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:30:00 | 2618.30 | 2599.45 | 2599.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 2617.00 | 2602.96 | 2601.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 2657.35 | 2614.15 | 2607.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 2712.00 | 2713.83 | 2695.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:15:00 | 2703.00 | 2713.83 | 2695.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2690.10 | 2709.08 | 2695.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 2690.10 | 2709.08 | 2695.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2683.05 | 2703.88 | 2694.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 2680.35 | 2703.88 | 2694.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 2713.80 | 2705.86 | 2696.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 2669.65 | 2705.86 | 2696.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 2717.20 | 2711.74 | 2701.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 2683.20 | 2711.74 | 2701.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 2705.55 | 2709.30 | 2702.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 2705.55 | 2709.30 | 2702.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 2701.65 | 2707.77 | 2702.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:15:00 | 2699.90 | 2707.77 | 2702.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 2710.35 | 2708.29 | 2703.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:30:00 | 2718.95 | 2708.63 | 2704.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 2677.55 | 2700.69 | 2701.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 2677.55 | 2700.69 | 2701.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 2675.10 | 2688.66 | 2694.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 2689.15 | 2664.88 | 2674.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 2689.15 | 2664.88 | 2674.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2689.15 | 2664.88 | 2674.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 2660.90 | 2673.05 | 2675.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:00:00 | 2660.90 | 2664.04 | 2670.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:30:00 | 2643.00 | 2661.68 | 2668.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:45:00 | 2655.75 | 2662.08 | 2667.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 2656.95 | 2661.05 | 2666.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 2647.95 | 2661.05 | 2666.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 2669.50 | 2655.75 | 2662.12 | SL hit (close>static) qty=1.00 sl=2666.90 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 2685.80 | 2666.78 | 2666.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 2697.40 | 2672.90 | 2669.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 2675.60 | 2679.51 | 2673.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 11:00:00 | 2675.60 | 2679.51 | 2673.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 2676.20 | 2678.85 | 2673.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 2687.40 | 2681.66 | 2676.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 2645.10 | 2674.08 | 2673.94 | SL hit (close<static) qty=1.00 sl=2672.25 alert=retest2 |

### Cycle 54 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 2653.70 | 2670.00 | 2672.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 2629.75 | 2661.95 | 2668.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 2523.10 | 2516.27 | 2544.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 2523.10 | 2516.27 | 2544.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2545.00 | 2522.01 | 2544.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 2545.00 | 2522.01 | 2544.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 2524.10 | 2522.43 | 2542.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:15:00 | 2520.55 | 2522.43 | 2542.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 2520.00 | 2521.11 | 2536.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:00:00 | 2519.70 | 2521.67 | 2533.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:00:00 | 2521.65 | 2521.66 | 2532.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 2529.30 | 2518.77 | 2527.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:30:00 | 2496.05 | 2511.81 | 2522.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 2497.15 | 2497.56 | 2507.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 2494.15 | 2497.05 | 2506.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 2489.40 | 2460.37 | 2458.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 2489.40 | 2460.37 | 2458.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 2502.90 | 2468.88 | 2462.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 2510.90 | 2516.49 | 2501.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 2510.90 | 2516.49 | 2501.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2510.90 | 2516.49 | 2501.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2510.90 | 2516.49 | 2501.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2510.90 | 2515.13 | 2503.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2503.40 | 2515.13 | 2503.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2471.15 | 2506.34 | 2500.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2471.15 | 2506.34 | 2500.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2475.35 | 2500.14 | 2498.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 2456.80 | 2500.14 | 2498.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2462.65 | 2490.30 | 2493.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 2462.40 | 2484.72 | 2491.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2488.75 | 2481.27 | 2488.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 2488.75 | 2481.27 | 2488.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2488.75 | 2481.27 | 2488.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 2488.75 | 2481.27 | 2488.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 2477.45 | 2480.51 | 2487.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 2478.70 | 2480.51 | 2487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 2475.65 | 2479.54 | 2486.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 2477.45 | 2479.54 | 2486.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2297.60 | 2307.90 | 2329.22 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 2352.20 | 2333.54 | 2332.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 2359.00 | 2345.68 | 2339.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 2384.55 | 2392.96 | 2379.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 2384.55 | 2392.96 | 2379.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 2389.45 | 2392.26 | 2380.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 2384.20 | 2392.26 | 2380.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 2386.45 | 2391.10 | 2381.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:45:00 | 2394.50 | 2384.61 | 2381.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 2395.35 | 2385.68 | 2382.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 2425.15 | 2444.75 | 2447.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 2425.15 | 2444.75 | 2447.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 14:15:00 | 2413.85 | 2438.57 | 2444.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 2438.75 | 2434.84 | 2441.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 2438.75 | 2434.84 | 2441.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2438.75 | 2434.84 | 2441.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 2438.75 | 2434.84 | 2441.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2450.95 | 2438.06 | 2442.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 2450.95 | 2438.06 | 2442.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 2447.40 | 2439.93 | 2442.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:15:00 | 2450.15 | 2439.93 | 2442.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 2449.15 | 2441.77 | 2443.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:15:00 | 2453.00 | 2441.77 | 2443.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 2460.00 | 2445.42 | 2444.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2469.65 | 2452.54 | 2448.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 2487.90 | 2495.01 | 2482.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 2487.90 | 2495.01 | 2482.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2487.90 | 2495.01 | 2482.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:45:00 | 2509.50 | 2496.80 | 2484.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 2444.40 | 2486.03 | 2481.89 | SL hit (close<static) qty=1.00 sl=2482.50 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 2438.50 | 2476.53 | 2477.94 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 2480.35 | 2468.31 | 2467.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 2485.00 | 2471.64 | 2468.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2481.15 | 2484.27 | 2478.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 2481.15 | 2484.27 | 2478.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2481.15 | 2484.27 | 2478.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 14:30:00 | 2504.45 | 2491.00 | 2484.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 2504.40 | 2493.70 | 2486.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 2513.00 | 2494.37 | 2488.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:30:00 | 2514.05 | 2496.25 | 2490.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 2488.20 | 2494.64 | 2490.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:45:00 | 2480.95 | 2494.64 | 2490.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 2485.20 | 2492.75 | 2489.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 2464.75 | 2492.75 | 2489.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 2460.95 | 2486.39 | 2487.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 2460.95 | 2486.39 | 2487.09 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 2530.95 | 2485.29 | 2484.74 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 2442.55 | 2483.32 | 2486.15 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 2513.00 | 2486.80 | 2485.23 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 2471.95 | 2488.05 | 2488.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 2465.35 | 2483.51 | 2486.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 2449.70 | 2443.61 | 2459.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 2449.70 | 2443.61 | 2459.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 2449.70 | 2443.61 | 2459.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 2462.55 | 2443.61 | 2459.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 2450.30 | 2444.95 | 2458.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:30:00 | 2463.90 | 2444.95 | 2458.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 2451.30 | 2446.22 | 2458.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:30:00 | 2449.60 | 2446.22 | 2458.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 2455.30 | 2448.03 | 2457.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 2457.60 | 2448.03 | 2457.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2465.10 | 2451.45 | 2458.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:30:00 | 2468.15 | 2451.45 | 2458.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 2460.00 | 2453.16 | 2458.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 2423.05 | 2453.16 | 2458.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 2438.85 | 2450.30 | 2456.91 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 2465.60 | 2455.83 | 2455.13 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 2447.10 | 2455.98 | 2456.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2405.50 | 2445.02 | 2451.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 11:15:00 | 2443.95 | 2441.55 | 2448.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 12:00:00 | 2443.95 | 2441.55 | 2448.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 2435.00 | 2440.24 | 2447.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 13:30:00 | 2423.40 | 2437.73 | 2445.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 2427.80 | 2436.98 | 2444.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 15:15:00 | 2417.15 | 2436.98 | 2444.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 14:15:00 | 2306.41 | 2329.40 | 2350.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 15:15:00 | 2302.23 | 2328.49 | 2348.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 09:15:00 | 2349.95 | 2332.78 | 2348.67 | SL hit (close>ema200) qty=0.50 sl=2332.78 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2375.10 | 2355.98 | 2355.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2384.90 | 2370.35 | 2363.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 10:15:00 | 2379.95 | 2384.10 | 2374.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:00:00 | 2379.95 | 2384.10 | 2374.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 2394.60 | 2385.65 | 2378.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 10:15:00 | 2396.50 | 2385.67 | 2379.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 11:45:00 | 2398.70 | 2390.83 | 2383.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:45:00 | 2396.80 | 2392.27 | 2385.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 2400.95 | 2391.46 | 2386.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2414.95 | 2396.16 | 2388.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 2417.35 | 2403.18 | 2396.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 2386.20 | 2395.64 | 2395.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 2386.20 | 2395.64 | 2395.70 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 11:15:00 | 2407.40 | 2396.23 | 2395.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 2410.65 | 2400.35 | 2397.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 2390.90 | 2402.59 | 2399.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 2390.90 | 2402.59 | 2399.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2390.90 | 2402.59 | 2399.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 2390.90 | 2402.59 | 2399.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 2378.35 | 2397.74 | 2397.84 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 2407.00 | 2399.59 | 2398.67 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 2389.10 | 2398.17 | 2398.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 2375.75 | 2390.91 | 2394.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 2384.80 | 2380.72 | 2387.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 2384.80 | 2380.72 | 2387.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 2384.80 | 2380.72 | 2387.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 2396.90 | 2380.72 | 2387.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 2377.30 | 2380.03 | 2386.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 2384.85 | 2380.03 | 2386.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 2392.95 | 2382.62 | 2387.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 2392.95 | 2382.62 | 2387.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 2386.95 | 2383.48 | 2387.24 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 2400.00 | 2391.12 | 2390.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 2421.00 | 2397.09 | 2392.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 2461.85 | 2464.05 | 2448.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 2474.20 | 2464.05 | 2448.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2466.30 | 2480.17 | 2465.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2466.30 | 2480.17 | 2465.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2478.65 | 2479.87 | 2466.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2489.70 | 2479.87 | 2466.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 2510.45 | 2605.24 | 2617.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2510.45 | 2605.24 | 2617.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 2485.00 | 2567.15 | 2597.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2581.50 | 2538.90 | 2567.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2581.50 | 2538.90 | 2567.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2581.50 | 2538.90 | 2567.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:45:00 | 2567.00 | 2538.90 | 2567.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 2582.30 | 2547.58 | 2568.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 2560.95 | 2551.75 | 2568.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:30:00 | 2543.95 | 2566.67 | 2571.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:30:00 | 2558.60 | 2565.58 | 2570.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:00:00 | 2563.50 | 2565.17 | 2569.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2645.00 | 2580.67 | 2575.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2645.00 | 2580.67 | 2575.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 2652.90 | 2602.10 | 2586.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2697.50 | 2698.09 | 2665.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 2697.50 | 2698.09 | 2665.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 2740.00 | 2751.51 | 2745.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 2743.20 | 2751.51 | 2745.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2726.00 | 2746.41 | 2743.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2726.00 | 2746.41 | 2743.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 2705.10 | 2738.15 | 2739.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 11:15:00 | 2692.60 | 2729.04 | 2735.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 10:15:00 | 2716.70 | 2708.07 | 2719.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 10:15:00 | 2716.70 | 2708.07 | 2719.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 2716.70 | 2708.07 | 2719.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 2716.70 | 2708.07 | 2719.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 2732.50 | 2712.96 | 2720.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:45:00 | 2733.80 | 2712.96 | 2720.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 2728.60 | 2716.09 | 2721.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 2723.40 | 2716.09 | 2721.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:30:00 | 2722.50 | 2717.71 | 2721.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:00:00 | 2718.70 | 2711.24 | 2715.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 2739.10 | 2716.81 | 2717.95 | SL hit (close>static) qty=1.00 sl=2734.60 alert=retest2 |

### Cycle 79 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 2729.20 | 2719.29 | 2718.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 2769.00 | 2729.23 | 2723.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 2742.60 | 2748.22 | 2738.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 2742.60 | 2748.22 | 2738.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2742.60 | 2748.22 | 2738.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 2745.60 | 2748.22 | 2738.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 2729.50 | 2744.48 | 2737.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 2729.50 | 2744.48 | 2737.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 2725.00 | 2740.58 | 2736.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:30:00 | 2731.40 | 2740.58 | 2736.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 2737.10 | 2738.47 | 2736.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 2736.30 | 2738.47 | 2736.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2742.30 | 2739.23 | 2737.15 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 2722.10 | 2736.88 | 2737.18 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 2767.50 | 2743.00 | 2739.93 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 2710.90 | 2736.50 | 2737.51 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2757.40 | 2736.80 | 2735.71 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 2727.90 | 2740.25 | 2741.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 2722.00 | 2736.60 | 2739.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 14:15:00 | 2703.20 | 2696.21 | 2707.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 2703.20 | 2696.21 | 2707.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 2680.60 | 2693.08 | 2705.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 2664.20 | 2693.08 | 2705.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 2719.80 | 2665.76 | 2678.82 | SL hit (close>static) qty=1.00 sl=2709.60 alert=retest2 |

### Cycle 85 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2739.00 | 2691.70 | 2689.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 2775.00 | 2744.20 | 2730.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 2776.10 | 2783.43 | 2759.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 2776.10 | 2783.43 | 2759.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2788.60 | 2795.39 | 2776.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 2781.00 | 2795.39 | 2776.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 2776.40 | 2791.59 | 2776.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 2776.40 | 2791.59 | 2776.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 2777.00 | 2788.68 | 2776.68 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 2724.60 | 2767.00 | 2769.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 2718.10 | 2735.96 | 2749.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2735.90 | 2732.91 | 2745.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:30:00 | 2732.40 | 2732.91 | 2745.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2743.50 | 2735.03 | 2745.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2746.60 | 2735.03 | 2745.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2727.40 | 2733.51 | 2743.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 2724.20 | 2733.51 | 2743.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 2721.40 | 2730.82 | 2741.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 2720.90 | 2689.99 | 2705.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:15:00 | 2587.99 | 2608.44 | 2632.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:15:00 | 2585.33 | 2608.44 | 2632.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:15:00 | 2584.86 | 2608.44 | 2632.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 2579.00 | 2578.51 | 2596.00 | SL hit (close>ema200) qty=0.50 sl=2578.51 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 2585.30 | 2556.79 | 2553.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2609.50 | 2576.30 | 2568.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 2710.90 | 2711.60 | 2683.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 2678.40 | 2704.73 | 2689.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2678.40 | 2704.73 | 2689.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2678.40 | 2704.73 | 2689.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 2687.20 | 2701.22 | 2689.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:45:00 | 2689.00 | 2701.22 | 2689.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2684.30 | 2697.84 | 2688.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2672.30 | 2697.84 | 2688.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2672.90 | 2692.85 | 2687.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 2678.60 | 2688.04 | 2685.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 12:15:00 | 2667.80 | 2681.91 | 2683.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 2667.80 | 2681.91 | 2683.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 2660.30 | 2675.36 | 2679.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 2696.90 | 2674.43 | 2677.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 2696.90 | 2674.43 | 2677.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2696.90 | 2674.43 | 2677.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 2696.90 | 2674.43 | 2677.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 2704.00 | 2680.34 | 2679.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 2710.90 | 2686.45 | 2682.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 2691.30 | 2692.02 | 2686.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 2691.30 | 2692.02 | 2686.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2691.30 | 2692.02 | 2686.54 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 2668.60 | 2684.37 | 2684.89 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 2687.10 | 2685.19 | 2685.04 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 2677.00 | 2683.55 | 2684.31 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 2693.00 | 2685.16 | 2684.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 13:15:00 | 2703.00 | 2689.76 | 2686.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 2681.30 | 2688.86 | 2686.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 15:15:00 | 2681.30 | 2688.86 | 2686.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 2681.30 | 2688.86 | 2686.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 2706.10 | 2688.86 | 2686.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 2713.90 | 2705.18 | 2699.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 2836.00 | 2846.96 | 2847.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 2836.00 | 2846.96 | 2847.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 2829.90 | 2842.55 | 2845.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2807.50 | 2807.04 | 2821.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:30:00 | 2805.80 | 2807.04 | 2821.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2802.60 | 2805.49 | 2817.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 2802.60 | 2805.49 | 2817.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2811.00 | 2793.65 | 2804.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2811.00 | 2793.65 | 2804.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2816.00 | 2798.12 | 2805.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 2816.00 | 2798.12 | 2805.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2816.90 | 2808.58 | 2808.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 2828.20 | 2808.58 | 2808.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 2823.80 | 2811.62 | 2810.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2824.60 | 2814.69 | 2811.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 2811.20 | 2814.46 | 2812.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 12:15:00 | 2811.20 | 2814.46 | 2812.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 2811.20 | 2814.46 | 2812.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 2811.20 | 2814.46 | 2812.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2810.00 | 2813.57 | 2812.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 2803.90 | 2813.57 | 2812.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 2797.20 | 2810.30 | 2810.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 13:15:00 | 2788.10 | 2800.02 | 2804.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 2792.20 | 2778.52 | 2786.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 2792.20 | 2778.52 | 2786.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2792.20 | 2778.52 | 2786.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 2792.20 | 2778.52 | 2786.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 2782.00 | 2779.22 | 2786.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 2772.70 | 2779.22 | 2786.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 2787.90 | 2784.98 | 2784.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 2787.90 | 2784.98 | 2784.71 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 2779.60 | 2783.91 | 2784.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 2774.40 | 2781.38 | 2782.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 12:15:00 | 2764.10 | 2759.44 | 2767.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 2764.10 | 2759.44 | 2767.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 2764.10 | 2759.44 | 2767.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 2764.10 | 2759.44 | 2767.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 2771.30 | 2761.81 | 2768.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 2770.90 | 2761.81 | 2768.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 2768.70 | 2763.19 | 2768.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 2760.00 | 2763.19 | 2768.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 2740.00 | 2722.20 | 2719.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 2740.00 | 2722.20 | 2719.79 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 2714.60 | 2721.54 | 2722.11 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 2742.40 | 2723.61 | 2722.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 2786.90 | 2739.74 | 2730.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 2760.60 | 2762.70 | 2747.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 2760.60 | 2762.70 | 2747.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2729.30 | 2755.43 | 2746.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 2759.60 | 2752.71 | 2747.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 2760.00 | 2753.33 | 2748.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:00:00 | 2761.00 | 2755.93 | 2750.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 2740.80 | 2747.12 | 2747.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 2740.80 | 2747.12 | 2747.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 2717.90 | 2741.28 | 2745.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 2765.50 | 2744.57 | 2745.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 2765.50 | 2744.57 | 2745.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2765.50 | 2744.57 | 2745.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 2762.20 | 2744.57 | 2745.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 2766.90 | 2749.04 | 2747.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 2774.10 | 2757.63 | 2752.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 2784.50 | 2790.72 | 2779.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 2784.50 | 2790.72 | 2779.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2784.50 | 2790.72 | 2779.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 2784.50 | 2790.72 | 2779.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 2764.10 | 2785.40 | 2777.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 2764.10 | 2785.40 | 2777.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2762.90 | 2780.90 | 2776.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 2762.90 | 2780.90 | 2776.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 2767.00 | 2776.41 | 2774.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 2767.00 | 2776.41 | 2774.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 2767.60 | 2774.64 | 2774.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 2765.00 | 2774.64 | 2774.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 2764.00 | 2772.52 | 2773.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2731.30 | 2762.97 | 2768.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2755.60 | 2714.09 | 2728.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2755.60 | 2714.09 | 2728.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2755.60 | 2714.09 | 2728.42 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2753.80 | 2737.93 | 2736.31 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2716.30 | 2734.28 | 2735.32 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2750.00 | 2738.03 | 2736.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 2845.00 | 2770.55 | 2754.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 2829.50 | 2829.68 | 2799.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 2831.50 | 2829.68 | 2799.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 2809.40 | 2822.69 | 2803.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 2809.40 | 2822.69 | 2803.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2823.40 | 2823.83 | 2810.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 2811.40 | 2823.83 | 2810.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2845.90 | 2869.15 | 2855.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 2848.70 | 2869.15 | 2855.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 2821.00 | 2859.52 | 2851.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 2821.00 | 2859.52 | 2851.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 2818.60 | 2845.17 | 2846.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 2813.80 | 2835.05 | 2841.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 2817.60 | 2816.06 | 2827.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 2817.60 | 2816.06 | 2827.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 2805.70 | 2807.62 | 2818.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 2812.30 | 2807.62 | 2818.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 2791.00 | 2804.95 | 2814.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:30:00 | 2808.80 | 2804.95 | 2814.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2789.50 | 2783.30 | 2790.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 2789.50 | 2783.30 | 2790.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 2785.90 | 2783.95 | 2789.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 2785.20 | 2783.95 | 2789.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2805.10 | 2788.18 | 2791.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 2805.10 | 2788.18 | 2791.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 2804.00 | 2791.34 | 2792.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 2805.00 | 2791.34 | 2792.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 2816.80 | 2796.43 | 2794.55 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 2780.90 | 2794.69 | 2795.37 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 2802.10 | 2796.27 | 2795.71 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 2779.50 | 2792.92 | 2794.24 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 2841.70 | 2798.77 | 2795.89 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 2798.30 | 2807.92 | 2808.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 2791.20 | 2802.94 | 2806.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2804.60 | 2803.27 | 2805.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2804.60 | 2803.27 | 2805.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2804.60 | 2803.27 | 2805.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 2811.80 | 2803.27 | 2805.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2786.90 | 2800.00 | 2804.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 2780.30 | 2796.22 | 2801.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 2773.30 | 2792.32 | 2798.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 2782.50 | 2788.54 | 2795.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 2782.00 | 2787.10 | 2793.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2799.40 | 2789.41 | 2793.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 2802.60 | 2789.41 | 2793.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 2796.10 | 2790.75 | 2794.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 2794.30 | 2790.75 | 2794.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 2813.60 | 2796.80 | 2796.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 2813.60 | 2796.80 | 2796.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 2817.70 | 2805.65 | 2801.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 2801.70 | 2804.89 | 2802.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 14:15:00 | 2801.70 | 2804.89 | 2802.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 2801.70 | 2804.89 | 2802.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 2801.70 | 2804.89 | 2802.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 2801.40 | 2804.19 | 2802.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 2812.90 | 2804.19 | 2802.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 2825.60 | 2810.64 | 2805.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 2833.90 | 2865.50 | 2866.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 2833.90 | 2865.50 | 2866.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 2815.00 | 2855.40 | 2861.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 2837.70 | 2829.14 | 2840.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 2837.70 | 2829.14 | 2840.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2818.20 | 2820.23 | 2831.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 2829.60 | 2820.23 | 2831.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 2766.10 | 2789.54 | 2808.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:15:00 | 2746.10 | 2789.54 | 2808.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 2752.90 | 2770.63 | 2794.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:45:00 | 2753.60 | 2757.21 | 2775.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 2752.20 | 2756.23 | 2770.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2753.70 | 2754.73 | 2767.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 2734.10 | 2757.06 | 2762.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2787.20 | 2762.58 | 2763.17 | SL hit (close>static) qty=1.00 sl=2768.90 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 2788.70 | 2767.80 | 2765.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 2792.00 | 2772.64 | 2767.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 2778.60 | 2782.04 | 2774.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 12:15:00 | 2778.60 | 2782.04 | 2774.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 2778.60 | 2782.04 | 2774.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 2778.60 | 2782.04 | 2774.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 2775.80 | 2780.80 | 2774.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 2773.30 | 2780.80 | 2774.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2793.10 | 2783.26 | 2776.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 2799.30 | 2783.26 | 2776.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 2797.10 | 2790.45 | 2780.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 2771.30 | 2800.28 | 2798.58 | SL hit (close<static) qty=1.00 sl=2773.30 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 2772.60 | 2794.75 | 2796.21 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 2798.10 | 2792.90 | 2792.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 2811.70 | 2796.66 | 2794.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 12:15:00 | 2801.70 | 2808.25 | 2801.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 12:15:00 | 2801.70 | 2808.25 | 2801.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2801.70 | 2808.25 | 2801.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 2802.40 | 2808.25 | 2801.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 2808.70 | 2808.34 | 2802.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 2813.60 | 2808.34 | 2802.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 15:15:00 | 2812.00 | 2808.63 | 2803.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 2788.70 | 2805.18 | 2802.64 | SL hit (close<static) qty=1.00 sl=2797.60 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 2792.80 | 2801.04 | 2801.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2775.90 | 2792.05 | 2796.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2793.70 | 2783.06 | 2788.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 2793.70 | 2783.06 | 2788.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2793.70 | 2783.06 | 2788.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2793.10 | 2783.06 | 2788.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2795.90 | 2785.63 | 2789.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 2802.20 | 2785.63 | 2789.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 2820.00 | 2796.39 | 2793.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 2841.70 | 2813.27 | 2803.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 2848.80 | 2857.78 | 2839.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 2848.80 | 2857.78 | 2839.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2834.20 | 2853.06 | 2838.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 2835.00 | 2853.06 | 2838.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2832.80 | 2849.01 | 2838.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:15:00 | 2841.00 | 2849.01 | 2838.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 2844.80 | 2859.28 | 2860.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 2844.80 | 2859.28 | 2860.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 2824.90 | 2849.69 | 2855.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 2866.60 | 2851.00 | 2854.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 2866.60 | 2851.00 | 2854.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2866.60 | 2851.00 | 2854.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 2866.60 | 2851.00 | 2854.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2871.50 | 2855.10 | 2855.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:15:00 | 2876.00 | 2855.10 | 2855.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 2892.30 | 2862.54 | 2859.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 2922.50 | 2874.53 | 2865.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2934.30 | 2947.47 | 2930.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2934.30 | 2947.47 | 2930.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2934.30 | 2947.47 | 2930.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 2930.70 | 2947.47 | 2930.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2925.20 | 2945.30 | 2938.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 2925.20 | 2945.30 | 2938.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2917.90 | 2939.82 | 2936.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 2917.90 | 2939.82 | 2936.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 2925.80 | 2933.53 | 2933.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 2914.60 | 2929.74 | 2932.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 2905.00 | 2903.65 | 2913.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 2893.00 | 2903.65 | 2913.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2773.00 | 2737.31 | 2762.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 2773.00 | 2737.31 | 2762.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2777.30 | 2745.31 | 2763.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 2777.50 | 2745.31 | 2763.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2762.00 | 2757.79 | 2764.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 2760.10 | 2757.79 | 2764.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2753.60 | 2756.95 | 2763.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:45:00 | 2749.10 | 2754.60 | 2761.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2780.90 | 2767.60 | 2765.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2780.90 | 2767.60 | 2765.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 2798.90 | 2787.83 | 2782.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 2770.40 | 2784.34 | 2781.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2770.40 | 2784.34 | 2781.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2770.40 | 2784.34 | 2781.34 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 2764.20 | 2776.34 | 2777.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 2755.00 | 2769.05 | 2773.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 2763.40 | 2762.62 | 2769.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 2763.40 | 2762.62 | 2769.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 2760.30 | 2762.15 | 2768.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:30:00 | 2756.30 | 2760.98 | 2767.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 2739.20 | 2720.01 | 2717.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 2739.20 | 2720.01 | 2717.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 2744.80 | 2724.97 | 2720.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 2727.00 | 2728.54 | 2724.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 2727.00 | 2728.54 | 2724.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 2727.00 | 2728.54 | 2724.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 2722.40 | 2728.54 | 2724.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 2724.80 | 2727.80 | 2724.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 2724.60 | 2727.80 | 2724.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2740.70 | 2733.19 | 2727.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 2732.30 | 2733.19 | 2727.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2741.60 | 2737.61 | 2732.95 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 2727.00 | 2730.87 | 2731.06 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 2734.20 | 2731.54 | 2731.35 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 2728.70 | 2730.97 | 2731.10 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 2735.10 | 2731.80 | 2731.44 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 2716.00 | 2729.79 | 2730.73 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 2736.90 | 2729.93 | 2729.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 2742.70 | 2732.49 | 2730.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 2745.90 | 2746.02 | 2739.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 2745.90 | 2746.02 | 2739.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 2734.00 | 2745.79 | 2740.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:45:00 | 2735.50 | 2745.79 | 2740.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2743.50 | 2745.33 | 2740.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:15:00 | 2761.60 | 2742.61 | 2740.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:45:00 | 2757.90 | 2749.87 | 2744.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 2767.80 | 2752.25 | 2747.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 2750.00 | 2757.40 | 2752.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 2746.40 | 2755.20 | 2751.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 2746.40 | 2755.20 | 2751.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2746.70 | 2753.50 | 2751.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 2744.60 | 2753.50 | 2751.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 2748.00 | 2752.40 | 2750.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 2753.10 | 2752.40 | 2750.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 2749.90 | 2753.94 | 2751.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 2786.20 | 2806.79 | 2807.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 2786.20 | 2806.79 | 2807.78 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 2817.00 | 2804.26 | 2803.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 2819.00 | 2807.20 | 2805.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 2805.10 | 2806.78 | 2805.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 2805.10 | 2806.78 | 2805.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2805.10 | 2806.78 | 2805.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 2805.10 | 2806.78 | 2805.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2807.00 | 2806.83 | 2805.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 2806.90 | 2806.83 | 2805.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 2805.00 | 2806.46 | 2805.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 2825.50 | 2806.46 | 2805.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2817.00 | 2808.57 | 2806.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 2843.70 | 2811.96 | 2809.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:45:00 | 2839.30 | 2820.06 | 2813.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 2836.70 | 2819.26 | 2815.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 2833.80 | 2826.08 | 2820.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 2831.90 | 2833.65 | 2827.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 2830.50 | 2833.65 | 2827.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 2826.90 | 2832.30 | 2827.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 2826.50 | 2832.30 | 2827.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 2830.60 | 2831.96 | 2828.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 2831.20 | 2831.96 | 2828.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2822.90 | 2830.15 | 2827.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 2822.90 | 2830.15 | 2827.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2827.00 | 2829.52 | 2827.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 2827.00 | 2829.52 | 2827.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2828.60 | 2829.34 | 2827.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 2818.20 | 2825.87 | 2826.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 2818.20 | 2825.87 | 2826.25 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 2846.00 | 2828.93 | 2827.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 2854.50 | 2842.80 | 2837.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 11:15:00 | 2844.00 | 2845.09 | 2839.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 2844.00 | 2845.09 | 2839.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 2827.40 | 2841.91 | 2839.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 2825.80 | 2841.91 | 2839.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 2833.00 | 2840.13 | 2838.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 2832.10 | 2840.13 | 2838.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 2832.10 | 2838.18 | 2838.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 2829.00 | 2836.35 | 2837.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 13:15:00 | 2836.00 | 2835.20 | 2836.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 13:15:00 | 2836.00 | 2835.20 | 2836.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 2836.00 | 2835.20 | 2836.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 2839.00 | 2835.20 | 2836.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 2852.70 | 2838.70 | 2838.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 2874.50 | 2847.94 | 2842.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 2867.00 | 2869.00 | 2859.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 2867.00 | 2869.00 | 2859.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2851.50 | 2865.50 | 2858.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 2851.50 | 2865.50 | 2858.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2849.40 | 2862.28 | 2858.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 2849.40 | 2862.28 | 2858.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2852.40 | 2858.29 | 2856.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2850.20 | 2858.29 | 2856.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2863.70 | 2859.37 | 2857.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 2869.10 | 2861.32 | 2858.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 2865.30 | 2862.38 | 2859.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 2869.80 | 2862.38 | 2859.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 2848.70 | 2858.73 | 2858.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 2848.70 | 2858.73 | 2858.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 2840.10 | 2855.00 | 2857.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2780.50 | 2772.07 | 2789.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2777.90 | 2772.07 | 2789.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 2794.70 | 2776.60 | 2790.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 2794.70 | 2776.60 | 2790.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 2807.70 | 2782.82 | 2791.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 2818.90 | 2782.82 | 2791.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 2797.50 | 2790.41 | 2793.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:45:00 | 2801.50 | 2790.41 | 2793.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2785.90 | 2789.50 | 2792.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 2774.90 | 2787.82 | 2791.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:15:00 | 2776.80 | 2787.82 | 2791.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 2815.00 | 2788.30 | 2789.48 | SL hit (close>static) qty=1.00 sl=2798.50 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 2809.20 | 2792.48 | 2791.27 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 2784.90 | 2798.83 | 2799.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 2780.00 | 2795.06 | 2798.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 2743.00 | 2741.27 | 2761.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 2743.00 | 2741.27 | 2761.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2788.00 | 2748.71 | 2756.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 2788.00 | 2748.71 | 2756.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2775.00 | 2753.97 | 2758.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 2769.00 | 2760.24 | 2760.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 2766.70 | 2761.53 | 2761.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 2766.70 | 2761.53 | 2761.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 2792.20 | 2767.66 | 2764.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 2771.00 | 2775.88 | 2769.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 2771.00 | 2775.88 | 2769.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 2771.00 | 2775.88 | 2769.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 2770.00 | 2775.88 | 2769.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 2772.80 | 2775.26 | 2770.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:45:00 | 2775.20 | 2775.26 | 2770.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 2758.60 | 2771.93 | 2769.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 2753.40 | 2771.93 | 2769.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 2759.80 | 2769.50 | 2768.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 2759.80 | 2769.50 | 2768.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2814.50 | 2835.78 | 2823.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 2814.50 | 2835.78 | 2823.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2824.80 | 2833.58 | 2824.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 2830.00 | 2829.85 | 2824.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 2830.50 | 2831.61 | 2826.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 2832.30 | 2831.61 | 2826.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:45:00 | 2830.60 | 2830.27 | 2826.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2831.60 | 2830.54 | 2826.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-30 14:15:00 | 2817.90 | 2824.54 | 2824.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 2817.90 | 2824.54 | 2824.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2795.60 | 2815.25 | 2820.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 2759.90 | 2755.33 | 2777.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 2756.60 | 2755.33 | 2777.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2773.60 | 2761.76 | 2776.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2824.10 | 2761.76 | 2776.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2829.40 | 2775.29 | 2781.71 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2835.80 | 2787.39 | 2786.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2852.50 | 2813.63 | 2801.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 10:15:00 | 2844.20 | 2851.02 | 2838.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 10:15:00 | 2844.20 | 2851.02 | 2838.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2844.20 | 2851.02 | 2838.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 2844.20 | 2851.02 | 2838.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2841.90 | 2849.20 | 2839.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 2839.90 | 2849.20 | 2839.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 2828.90 | 2845.14 | 2838.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 2829.10 | 2845.14 | 2838.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2835.30 | 2843.17 | 2837.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 2826.50 | 2843.17 | 2837.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 2845.80 | 2842.55 | 2838.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 2864.10 | 2842.55 | 2838.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 2904.30 | 2918.20 | 2919.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 2904.30 | 2918.20 | 2919.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 2886.30 | 2911.82 | 2916.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 2910.40 | 2904.24 | 2909.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 2910.40 | 2904.24 | 2909.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2910.40 | 2904.24 | 2909.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 2910.40 | 2904.24 | 2909.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 2907.00 | 2904.79 | 2909.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 2912.40 | 2904.79 | 2909.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 2911.80 | 2906.19 | 2909.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 2914.80 | 2906.19 | 2909.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2896.20 | 2904.19 | 2908.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 2891.00 | 2903.82 | 2906.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 2932.30 | 2908.13 | 2907.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 2932.30 | 2908.13 | 2907.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 2932.80 | 2913.06 | 2910.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 2914.20 | 2924.13 | 2918.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 2914.20 | 2924.13 | 2918.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 2914.20 | 2924.13 | 2918.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 2914.20 | 2924.13 | 2918.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 2892.50 | 2917.80 | 2915.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 2892.50 | 2917.80 | 2915.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 2891.20 | 2912.48 | 2913.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2867.60 | 2898.24 | 2906.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 2853.40 | 2852.85 | 2871.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:30:00 | 2865.20 | 2852.85 | 2871.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 2872.00 | 2859.27 | 2869.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 2876.30 | 2859.27 | 2869.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2867.40 | 2860.90 | 2868.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 2873.30 | 2860.90 | 2868.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 2865.00 | 2861.72 | 2868.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 2864.00 | 2861.72 | 2868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 2862.00 | 2861.77 | 2867.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 2850.10 | 2861.58 | 2867.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 2879.30 | 2870.05 | 2869.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 2879.30 | 2870.05 | 2869.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2898.00 | 2875.64 | 2872.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 2878.20 | 2883.22 | 2877.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 2878.20 | 2883.22 | 2877.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 2878.20 | 2883.22 | 2877.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 2878.20 | 2883.22 | 2877.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 2889.60 | 2884.50 | 2878.21 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 2862.00 | 2874.20 | 2875.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2824.40 | 2860.97 | 2868.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2782.30 | 2781.26 | 2807.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:30:00 | 2783.40 | 2781.26 | 2807.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2717.10 | 2691.02 | 2717.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 2717.10 | 2691.02 | 2717.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2717.00 | 2696.21 | 2717.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2722.80 | 2702.81 | 2718.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2741.70 | 2710.59 | 2720.57 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2738.10 | 2728.00 | 2727.10 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 2714.00 | 2725.20 | 2725.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 2661.50 | 2711.31 | 2719.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2716.20 | 2688.49 | 2699.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 2716.20 | 2688.49 | 2699.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2716.20 | 2688.49 | 2699.81 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 2743.10 | 2708.55 | 2706.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 2745.00 | 2721.13 | 2713.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2725.00 | 2738.75 | 2728.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 2725.00 | 2738.75 | 2728.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2725.00 | 2738.75 | 2728.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 2725.00 | 2738.75 | 2728.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2725.30 | 2736.06 | 2728.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2680.70 | 2736.06 | 2728.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2691.30 | 2727.11 | 2724.97 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 2699.90 | 2721.67 | 2722.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 2677.40 | 2700.41 | 2711.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 2632.90 | 2619.23 | 2653.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:00:00 | 2632.90 | 2619.23 | 2653.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2641.10 | 2627.33 | 2651.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 2635.00 | 2627.33 | 2651.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 2659.00 | 2633.55 | 2648.41 | SL hit (close>static) qty=1.00 sl=2653.50 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 2679.80 | 2657.53 | 2656.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 2686.60 | 2663.34 | 2658.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2646.40 | 2696.59 | 2686.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2646.40 | 2696.59 | 2686.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2646.40 | 2696.59 | 2686.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 2648.50 | 2696.59 | 2686.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 2638.80 | 2685.03 | 2682.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 2638.80 | 2685.03 | 2682.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 2630.50 | 2674.12 | 2677.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 2622.20 | 2663.74 | 2672.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2649.20 | 2640.36 | 2656.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:15:00 | 2643.80 | 2640.36 | 2656.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 2649.10 | 2642.31 | 2654.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 2627.80 | 2644.05 | 2653.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 2636.90 | 2587.43 | 2586.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2636.90 | 2587.43 | 2586.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 2651.60 | 2606.85 | 2596.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2623.10 | 2628.68 | 2611.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2623.10 | 2628.68 | 2611.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2623.10 | 2628.68 | 2611.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 2617.00 | 2628.68 | 2611.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2618.20 | 2625.34 | 2613.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 2608.90 | 2625.34 | 2613.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2623.70 | 2625.01 | 2614.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:15:00 | 2642.10 | 2623.87 | 2615.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 2561.50 | 2614.32 | 2612.80 | SL hit (close<static) qty=1.00 sl=2613.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 2560.90 | 2603.63 | 2608.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 2541.00 | 2576.05 | 2591.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2593.70 | 2579.58 | 2592.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2593.70 | 2579.58 | 2592.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2593.70 | 2579.58 | 2592.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 2582.10 | 2579.58 | 2592.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 2578.50 | 2579.36 | 2590.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 2580.90 | 2589.33 | 2593.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 2580.00 | 2590.02 | 2593.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 2580.00 | 2588.02 | 2591.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2525.90 | 2588.02 | 2591.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 2619.70 | 2576.03 | 2572.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2619.70 | 2576.03 | 2572.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2738.70 | 2631.71 | 2605.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 2741.30 | 2741.30 | 2709.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2766.50 | 2741.30 | 2709.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 2747.60 | 2744.96 | 2726.21 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2696.20 | 2737.08 | 2726.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2696.20 | 2737.08 | 2726.00 | SL hit (close<ema400) qty=1.00 sl=2726.00 alert=retest1 |

### Cycle 160 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 2710.90 | 2721.95 | 2722.61 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2750.30 | 2727.62 | 2725.12 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 2699.90 | 2728.79 | 2730.81 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 2745.30 | 2727.00 | 2726.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 2756.00 | 2732.80 | 2729.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 2763.90 | 2769.42 | 2757.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 2763.90 | 2769.42 | 2757.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2763.90 | 2769.42 | 2757.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 2763.90 | 2769.42 | 2757.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 2774.00 | 2775.95 | 2766.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 2760.30 | 2775.95 | 2766.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2762.30 | 2773.22 | 2766.35 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 2749.20 | 2762.44 | 2762.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 2730.90 | 2756.13 | 2759.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 2757.40 | 2752.84 | 2757.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 2757.40 | 2752.84 | 2757.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 2757.40 | 2752.84 | 2757.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:15:00 | 2760.00 | 2752.84 | 2757.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 2752.80 | 2752.83 | 2757.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 2763.50 | 2752.83 | 2757.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 2750.00 | 2752.27 | 2756.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 2742.90 | 2748.81 | 2754.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 2738.90 | 2748.81 | 2754.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 2742.30 | 2746.81 | 2752.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 2763.20 | 2749.64 | 2752.98 | SL hit (close>static) qty=1.00 sl=2756.60 alert=retest2 |

### Cycle 165 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 2785.00 | 2756.71 | 2755.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 2793.00 | 2767.60 | 2761.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 2796.80 | 2802.82 | 2786.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 2796.80 | 2802.82 | 2786.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 2784.10 | 2799.07 | 2786.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 2784.10 | 2799.07 | 2786.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2775.90 | 2794.44 | 2785.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 2793.20 | 2794.44 | 2785.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 2749.00 | 2793.97 | 2792.78 | SL hit (close<static) qty=1.00 sl=2772.00 alert=retest2 |

### Cycle 166 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 2760.60 | 2787.29 | 2789.86 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 2825.40 | 2794.58 | 2791.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 2841.70 | 2804.00 | 2795.82 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 2350.00 | 2024-05-13 15:15:00 | 2392.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-22 13:30:00 | 2437.40 | 2024-05-23 09:15:00 | 2404.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-06-03 11:45:00 | 2384.95 | 2024-06-04 11:15:00 | 2265.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:45:00 | 2384.95 | 2024-06-05 12:15:00 | 2262.20 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2024-06-24 10:30:00 | 2483.45 | 2024-07-01 14:15:00 | 2731.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-24 11:00:00 | 2487.60 | 2024-07-01 14:15:00 | 2736.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-16 09:15:00 | 2822.25 | 2024-07-18 11:15:00 | 2792.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-28 13:30:00 | 2702.40 | 2024-08-28 14:15:00 | 2720.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-08-28 15:15:00 | 2704.10 | 2024-09-02 09:15:00 | 2704.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-30 15:15:00 | 2696.95 | 2024-09-02 09:15:00 | 2704.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-09-09 14:15:00 | 2694.50 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-09-10 09:45:00 | 2693.40 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-09-10 10:15:00 | 2682.90 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-09-11 09:15:00 | 2692.70 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-09-18 10:00:00 | 2774.00 | 2024-09-18 10:15:00 | 2750.65 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-19 11:15:00 | 2749.80 | 2024-09-24 14:15:00 | 2612.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 11:15:00 | 2749.80 | 2024-09-25 11:15:00 | 2624.65 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2024-10-11 09:15:00 | 2706.25 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-10-14 10:45:00 | 2704.90 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-10-15 10:00:00 | 2711.00 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-15 13:00:00 | 2709.40 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-10-24 12:15:00 | 2663.70 | 2024-10-29 09:15:00 | 2675.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-10-24 13:45:00 | 2668.55 | 2024-10-29 09:15:00 | 2675.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-10-31 15:00:00 | 2697.10 | 2024-11-04 09:15:00 | 2651.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-11-01 18:30:00 | 2692.90 | 2024-11-04 09:15:00 | 2651.25 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-27 14:15:00 | 2632.30 | 2024-11-28 11:15:00 | 2589.60 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-12-06 13:30:00 | 2718.95 | 2024-12-09 09:15:00 | 2677.55 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-12-12 09:15:00 | 2660.90 | 2024-12-13 11:15:00 | 2669.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-12 12:00:00 | 2660.90 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-12 12:30:00 | 2643.00 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-12-12 14:45:00 | 2655.75 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-13 09:15:00 | 2647.95 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-12-16 15:00:00 | 2687.40 | 2024-12-17 09:15:00 | 2645.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-12-23 12:15:00 | 2520.55 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-12-23 15:15:00 | 2520.00 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2024-12-24 11:00:00 | 2519.70 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2024-12-24 12:00:00 | 2521.65 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2024-12-26 10:30:00 | 2496.05 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-12-27 10:45:00 | 2497.15 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-12-27 11:45:00 | 2494.15 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-01-22 14:45:00 | 2394.50 | 2025-01-28 13:15:00 | 2425.15 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-01-23 09:15:00 | 2395.35 | 2025-01-28 13:15:00 | 2425.15 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-02-01 10:45:00 | 2509.50 | 2025-02-01 12:15:00 | 2444.40 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-02-06 14:30:00 | 2504.45 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-02-07 09:30:00 | 2504.40 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-02-07 11:30:00 | 2513.00 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-02-07 13:30:00 | 2514.05 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-02-21 13:30:00 | 2423.40 | 2025-02-28 14:15:00 | 2306.41 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2025-02-21 14:45:00 | 2427.80 | 2025-02-28 15:15:00 | 2302.23 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-02-21 13:30:00 | 2423.40 | 2025-03-03 09:15:00 | 2349.95 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-02-21 14:45:00 | 2427.80 | 2025-03-03 09:15:00 | 2349.95 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-02-21 15:15:00 | 2417.15 | 2025-03-03 14:15:00 | 2375.10 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2025-03-06 10:15:00 | 2396.50 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-03-06 11:45:00 | 2398.70 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-03-06 14:45:00 | 2396.80 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-03-07 09:15:00 | 2400.95 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-03-10 09:15:00 | 2417.35 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-03-24 09:15:00 | 2489.70 | 2025-04-07 09:15:00 | 2510.45 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-04-08 11:30:00 | 2560.95 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-04-09 10:30:00 | 2543.95 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-04-09 12:30:00 | 2558.60 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-04-09 14:00:00 | 2563.50 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-04-24 13:15:00 | 2723.40 | 2025-04-25 14:15:00 | 2739.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-04-25 09:30:00 | 2722.50 | 2025-04-25 14:15:00 | 2739.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-04-25 14:00:00 | 2718.70 | 2025-04-25 14:15:00 | 2739.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-05-09 09:15:00 | 2664.20 | 2025-05-12 09:15:00 | 2719.80 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-05-21 12:15:00 | 2724.20 | 2025-05-28 11:15:00 | 2587.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 12:45:00 | 2721.40 | 2025-05-28 11:15:00 | 2585.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 09:30:00 | 2720.90 | 2025-05-28 11:15:00 | 2584.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 12:15:00 | 2724.20 | 2025-05-29 15:15:00 | 2579.00 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2025-05-21 12:45:00 | 2721.40 | 2025-05-29 15:15:00 | 2579.00 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2025-05-23 09:30:00 | 2720.90 | 2025-05-29 15:15:00 | 2579.00 | STOP_HIT | 0.50 | 5.22% |
| BUY | retest2 | 2025-06-13 10:30:00 | 2678.60 | 2025-06-13 12:15:00 | 2667.80 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-06-20 09:15:00 | 2706.10 | 2025-07-03 09:15:00 | 2836.00 | STOP_HIT | 1.00 | 4.80% |
| BUY | retest2 | 2025-06-23 10:30:00 | 2713.90 | 2025-07-03 09:15:00 | 2836.00 | STOP_HIT | 1.00 | 4.50% |
| SELL | retest2 | 2025-07-14 11:15:00 | 2772.70 | 2025-07-15 12:15:00 | 2787.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-17 15:15:00 | 2760.00 | 2025-07-28 10:15:00 | 2740.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-07-31 13:00:00 | 2759.60 | 2025-08-01 13:15:00 | 2740.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-31 15:15:00 | 2760.00 | 2025-08-01 13:15:00 | 2740.80 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-01 10:00:00 | 2761.00 | 2025-08-01 13:15:00 | 2740.80 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-10 14:15:00 | 2780.30 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-11 09:15:00 | 2773.30 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-11 11:30:00 | 2782.50 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-11 12:45:00 | 2782.00 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-16 09:15:00 | 2812.90 | 2025-09-23 09:15:00 | 2833.90 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2025-09-16 12:00:00 | 2825.60 | 2025-09-23 09:15:00 | 2833.90 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-09-26 10:15:00 | 2746.10 | 2025-10-01 13:15:00 | 2787.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-26 13:15:00 | 2752.90 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-29 11:45:00 | 2753.60 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-29 14:45:00 | 2752.20 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-01 10:15:00 | 2734.10 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-03 15:15:00 | 2799.30 | 2025-10-08 10:15:00 | 2771.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-06 09:30:00 | 2797.10 | 2025-10-08 10:15:00 | 2771.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-10 14:15:00 | 2813.60 | 2025-10-13 09:15:00 | 2788.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-10 15:15:00 | 2812.00 | 2025-10-13 09:15:00 | 2788.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-17 15:15:00 | 2841.00 | 2025-10-24 11:15:00 | 2844.80 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-11-11 10:45:00 | 2749.10 | 2025-11-12 09:15:00 | 2780.90 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-11-19 13:30:00 | 2756.30 | 2025-11-26 13:15:00 | 2739.20 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-12-09 10:15:00 | 2761.60 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-12-09 11:45:00 | 2757.90 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-12-10 09:15:00 | 2767.80 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-12-10 13:15:00 | 2750.00 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-12-11 09:15:00 | 2753.10 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-12-11 09:45:00 | 2749.90 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-12-22 09:15:00 | 2843.70 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-22 10:45:00 | 2839.30 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-23 09:45:00 | 2836.70 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-23 14:15:00 | 2833.80 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-01-06 13:00:00 | 2869.10 | 2026-01-07 11:15:00 | 2848.70 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-06 14:45:00 | 2865.30 | 2026-01-07 11:15:00 | 2848.70 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-01-06 15:15:00 | 2869.80 | 2026-01-07 11:15:00 | 2848.70 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-13 12:45:00 | 2774.90 | 2026-01-14 10:15:00 | 2815.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-13 13:15:00 | 2776.80 | 2026-01-14 10:15:00 | 2815.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-22 12:30:00 | 2769.00 | 2026-01-22 13:15:00 | 2766.70 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2026-01-29 13:30:00 | 2830.00 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-30 09:30:00 | 2830.50 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-30 10:15:00 | 2832.30 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-01-30 10:45:00 | 2830.60 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-02-09 09:15:00 | 2864.10 | 2026-02-13 13:15:00 | 2904.30 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2026-02-17 15:15:00 | 2891.00 | 2026-02-18 10:15:00 | 2932.30 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-24 10:45:00 | 2850.10 | 2026-02-24 15:15:00 | 2879.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-16 12:15:00 | 2635.00 | 2026-03-16 14:15:00 | 2659.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-20 14:15:00 | 2627.80 | 2026-03-25 10:15:00 | 2636.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-03-27 15:15:00 | 2642.10 | 2026-03-30 09:15:00 | 2561.50 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-04-01 10:15:00 | 2582.10 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-01 11:00:00 | 2578.50 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-01 13:30:00 | 2580.90 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-04-01 15:15:00 | 2580.00 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2525.90 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest1 | 2026-04-10 09:15:00 | 2766.50 | 2026-04-13 09:15:00 | 2696.20 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2026-04-10 15:00:00 | 2747.60 | 2026-04-13 09:15:00 | 2696.20 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-04-13 10:30:00 | 2719.70 | 2026-04-13 15:15:00 | 2710.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2026-04-13 14:15:00 | 2717.00 | 2026-04-13 15:15:00 | 2710.90 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-04-13 14:45:00 | 2718.90 | 2026-04-13 15:15:00 | 2710.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-04-24 13:30:00 | 2742.90 | 2026-04-27 09:15:00 | 2763.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-04-24 14:00:00 | 2738.90 | 2026-04-27 09:15:00 | 2763.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-24 14:45:00 | 2742.30 | 2026-04-27 09:15:00 | 2763.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-04-29 09:15:00 | 2793.20 | 2026-04-30 09:15:00 | 2749.00 | STOP_HIT | 1.00 | -1.58% |
