# Adani Enterprises Ltd. (ADANIENT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 47 |
| ALERT2 | 48 |
| ALERT2_SKIP | 21 |
| ALERT3 | 123 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 58 |
| PARTIAL | 7 |
| TARGET_HIT | 11 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 35
- **Target hits / Stop hits / Partials:** 11 / 47 / 7
- **Avg / median % per leg:** 1.99% / -0.37%
- **Sum % (uncompounded):** 129.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 13 | 56.5% | 7 | 16 | 0 | 3.00% | 69.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 13 | 56.5% | 7 | 16 | 0 | 3.00% | 69.0% |
| SELL (all) | 42 | 17 | 40.5% | 4 | 31 | 7 | 1.43% | 60.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 17 | 40.5% | 4 | 31 | 7 | 1.43% | 60.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 65 | 30 | 46.2% | 11 | 47 | 7 | 1.99% | 129.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2341.69 | 2263.49 | 2253.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2347.90 | 2290.88 | 2268.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 2462.78 | 2470.92 | 2448.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 2462.78 | 2470.92 | 2448.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2456.29 | 2466.88 | 2454.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 2460.17 | 2466.88 | 2454.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 2447.08 | 2462.92 | 2453.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 2444.85 | 2462.92 | 2453.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 2436.41 | 2457.62 | 2452.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 2436.41 | 2457.62 | 2452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 2429.43 | 2446.99 | 2448.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 2417.22 | 2439.53 | 2444.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 2442.23 | 2430.31 | 2437.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2424.10 | 2429.07 | 2435.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2422.36 | 2429.07 | 2435.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 2446.98 | 2423.85 | 2427.61 | SL hit (close>static) qty=1.00 sl=2437.29 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 2461.52 | 2436.02 | 2432.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2482.17 | 2453.79 | 2442.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 2465.98 | 2466.56 | 2455.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 2464.24 | 2466.56 | 2455.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2456.58 | 2464.56 | 2455.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2456.58 | 2464.56 | 2455.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2471.22 | 2465.89 | 2457.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 2483.82 | 2465.89 | 2457.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 2473.35 | 2470.13 | 2462.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 2445.53 | 2460.02 | 2460.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 2445.53 | 2460.02 | 2460.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 2443.10 | 2456.64 | 2458.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 2472.38 | 2452.93 | 2454.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 2448.43 | 2452.03 | 2454.06 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 2465.40 | 2456.94 | 2456.08 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 2444.36 | 2454.24 | 2454.99 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 2473.16 | 2456.65 | 2454.68 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2430.50 | 2450.60 | 2453.03 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 2447.76 | 2425.88 | 2424.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 2463.95 | 2437.78 | 2431.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 2525.02 | 2528.51 | 2505.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:00:00 | 2525.02 | 2528.51 | 2505.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2495.75 | 2518.25 | 2507.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2500.59 | 2518.25 | 2507.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 2504.86 | 2515.57 | 2507.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 2514.46 | 2513.51 | 2507.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 2484.31 | 2505.21 | 2504.46 | SL hit (close<static) qty=1.00 sl=2494.10 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 2481.11 | 2500.39 | 2502.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 2465.01 | 2493.75 | 2499.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2458.81 | 2438.67 | 2456.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2460.75 | 2443.09 | 2456.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 2445.14 | 2455.84 | 2459.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 2442.81 | 2453.31 | 2457.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 2436.12 | 2448.94 | 2455.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 2402.29 | 2386.28 | 2384.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 2402.29 | 2386.28 | 2384.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2446.40 | 2402.55 | 2393.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 2427.69 | 2429.06 | 2412.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 2427.69 | 2429.06 | 2412.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 2531.62 | 2539.92 | 2528.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 2531.62 | 2539.92 | 2528.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 2535.01 | 2538.94 | 2529.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 2529.97 | 2538.94 | 2529.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 2538.89 | 2538.93 | 2530.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 2533.46 | 2538.93 | 2530.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2543.15 | 2547.84 | 2541.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 2535.30 | 2547.84 | 2541.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2539.57 | 2546.19 | 2541.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 2539.57 | 2546.19 | 2541.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 2536.37 | 2544.22 | 2540.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 2535.59 | 2544.22 | 2540.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2537.05 | 2542.79 | 2540.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:30:00 | 2546.35 | 2541.95 | 2540.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 2528.42 | 2537.84 | 2538.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 2528.42 | 2537.84 | 2538.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2512.91 | 2530.79 | 2534.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 2508.64 | 2506.31 | 2515.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 2514.17 | 2506.31 | 2515.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2505.05 | 2506.06 | 2514.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 2500.30 | 2504.91 | 2513.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 2495.55 | 2503.91 | 2512.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 2521.82 | 2511.00 | 2512.27 | SL hit (close>static) qty=1.00 sl=2520.66 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 2511.84 | 2500.70 | 2500.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 2522.99 | 2505.16 | 2502.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 2535.11 | 2537.37 | 2529.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 2537.72 | 2537.37 | 2529.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2526.96 | 2535.28 | 2528.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 2526.96 | 2535.28 | 2528.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2514.65 | 2531.16 | 2527.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 2510.00 | 2531.16 | 2527.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 2513.10 | 2524.35 | 2524.88 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 2537.53 | 2524.98 | 2524.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 2540.15 | 2531.50 | 2528.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 2530.74 | 2532.27 | 2529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2524.83 | 2530.78 | 2528.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 2526.48 | 2530.78 | 2528.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 2521.53 | 2528.93 | 2528.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 2521.53 | 2528.93 | 2528.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 2519.40 | 2527.02 | 2527.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 2510.38 | 2523.70 | 2525.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 2531.33 | 2527.03 | 2526.71 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 2525.51 | 2527.26 | 2527.49 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 2530.36 | 2527.88 | 2527.75 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 2502.05 | 2523.29 | 2525.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2480.82 | 2514.80 | 2521.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 2496.42 | 2490.71 | 2504.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2484.69 | 2489.50 | 2502.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 2476.84 | 2486.97 | 2499.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 2474.80 | 2482.75 | 2496.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 2353.00 | 2415.51 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 2351.06 | 2415.51 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-06 11:15:00 | 2229.16 | 2255.62 | 2284.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2216.73 | 2184.54 | 2182.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 2241.26 | 2203.36 | 2192.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 2212.07 | 2216.59 | 2205.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 2214.89 | 2216.59 | 2205.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2187.93 | 2210.86 | 2203.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2187.93 | 2210.86 | 2203.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2208.39 | 2210.36 | 2204.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 2213.14 | 2210.36 | 2204.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 2212.27 | 2210.27 | 2205.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 2217.21 | 2210.27 | 2205.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 2210.81 | 2211.49 | 2206.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2218.18 | 2213.26 | 2208.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 2224.87 | 2214.85 | 2209.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 2252.79 | 2213.79 | 2210.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 2259.77 | 2291.08 | 2293.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 2259.77 | 2291.08 | 2293.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 2256.86 | 2279.18 | 2287.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 2226.13 | 2216.27 | 2233.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:45:00 | 2226.91 | 2216.27 | 2233.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 2254.83 | 2223.98 | 2235.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 2254.54 | 2223.98 | 2235.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2233.69 | 2225.93 | 2235.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 2222.45 | 2225.93 | 2235.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 2224.87 | 2213.10 | 2211.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2224.87 | 2213.10 | 2211.91 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 2208.49 | 2211.53 | 2211.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 2200.92 | 2208.95 | 2210.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 11:15:00 | 2209.36 | 2209.03 | 2210.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 2209.36 | 2209.03 | 2210.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2206.35 | 2208.50 | 2210.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:30:00 | 2208.49 | 2208.50 | 2210.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2209.46 | 2208.69 | 2210.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 2209.46 | 2208.69 | 2210.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 2220.70 | 2211.09 | 2211.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 10:15:00 | 2222.64 | 2216.13 | 2213.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 2209.26 | 2212.99 | 2213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2203.54 | 2211.10 | 2212.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 2215.76 | 2204.97 | 2208.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 2209.07 | 2205.79 | 2208.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 2215.37 | 2205.79 | 2208.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 2215.37 | 2207.70 | 2208.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 2241.74 | 2207.70 | 2208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 2252.50 | 2216.66 | 2212.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2261.81 | 2245.09 | 2237.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 2318.04 | 2323.30 | 2304.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:30:00 | 2320.85 | 2323.30 | 2304.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 2312.03 | 2318.03 | 2309.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 2309.02 | 2318.03 | 2309.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 2311.83 | 2316.79 | 2310.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:15:00 | 2309.80 | 2316.79 | 2310.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 2309.80 | 2315.39 | 2310.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 2322.21 | 2315.39 | 2310.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 13:15:00 | 2554.43 | 2501.78 | 2442.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 2494.78 | 2529.17 | 2533.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 2484.98 | 2514.05 | 2525.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 2463.66 | 2434.41 | 2446.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 2510.29 | 2449.59 | 2452.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 2510.29 | 2449.59 | 2452.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2514.84 | 2462.64 | 2458.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 2523.67 | 2501.87 | 2492.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 2498.36 | 2502.21 | 2494.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 2501.08 | 2501.98 | 2495.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:15:00 | 2494.19 | 2501.98 | 2495.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2493.61 | 2500.31 | 2494.95 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 2477.81 | 2489.15 | 2490.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 2458.32 | 2482.99 | 2487.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 2467.92 | 2455.06 | 2466.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2474.13 | 2458.88 | 2467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2474.13 | 2458.88 | 2467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2469.67 | 2461.03 | 2467.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 2463.46 | 2461.03 | 2467.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 2463.75 | 2461.58 | 2467.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 2461.33 | 2461.53 | 2466.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 2473.83 | 2469.30 | 2468.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 2473.83 | 2469.30 | 2468.83 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 2447.76 | 2465.57 | 2467.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2427.11 | 2445.98 | 2454.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 2440.58 | 2434.91 | 2445.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:30:00 | 2435.73 | 2434.91 | 2445.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2458.90 | 2440.87 | 2446.60 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 2471.12 | 2453.94 | 2451.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 2484.01 | 2463.19 | 2457.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 2475.19 | 2474.98 | 2467.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2492.74 | 2478.53 | 2469.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 2469.86 | 2478.53 | 2469.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2476.74 | 2478.56 | 2472.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 2473.93 | 2478.56 | 2472.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2471.12 | 2477.07 | 2472.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 2472.09 | 2477.07 | 2472.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 2467.34 | 2475.12 | 2471.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 2490.32 | 2475.12 | 2471.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 2480.23 | 2474.34 | 2471.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 2478.97 | 2473.29 | 2472.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 2477.71 | 2473.29 | 2472.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 2470.73 | 2472.78 | 2472.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 2470.73 | 2472.78 | 2472.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2475.77 | 2480.11 | 2477.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 2475.77 | 2480.11 | 2477.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2464.24 | 2476.93 | 2475.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 2464.24 | 2476.93 | 2475.86 | SL hit (close<static) qty=1.00 sl=2467.34 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 2467.34 | 2475.01 | 2475.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 2456.00 | 2471.21 | 2473.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 11:15:00 | 2422.74 | 2420.61 | 2432.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:45:00 | 2424.00 | 2420.61 | 2432.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2427.49 | 2421.98 | 2432.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 2427.49 | 2421.98 | 2432.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 2413.44 | 2420.27 | 2430.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 2406.36 | 2420.27 | 2430.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 2437.38 | 2424.10 | 2429.74 | SL hit (close>static) qty=1.00 sl=2433.41 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 2477.91 | 2434.86 | 2434.12 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 2426.43 | 2446.94 | 2448.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 2404.81 | 2434.15 | 2441.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 2416.25 | 2413.56 | 2426.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 2416.25 | 2413.56 | 2426.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2300.30 | 2290.19 | 2314.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 2322.60 | 2290.19 | 2314.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2295.55 | 2293.55 | 2310.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 2282.17 | 2293.38 | 2303.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2436.22 | 2320.07 | 2309.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2436.22 | 2320.07 | 2309.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 2462.78 | 2422.61 | 2393.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 2438.25 | 2439.51 | 2412.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 2438.25 | 2439.51 | 2412.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2443.50 | 2448.49 | 2437.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 2438.50 | 2448.49 | 2437.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2435.50 | 2445.56 | 2438.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2435.50 | 2445.56 | 2438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2439.70 | 2444.38 | 2438.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 2434.20 | 2444.38 | 2438.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2437.60 | 2443.03 | 2438.24 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 2429.60 | 2435.86 | 2435.88 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2462.10 | 2441.11 | 2438.26 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 2431.70 | 2439.89 | 2440.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 2425.90 | 2434.40 | 2437.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2297.90 | 2287.37 | 2316.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 2297.90 | 2287.37 | 2316.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 2320.80 | 2297.47 | 2316.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 2323.50 | 2297.47 | 2316.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2308.70 | 2299.71 | 2315.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 2303.10 | 2298.49 | 2313.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 13:15:00 | 2187.94 | 2219.22 | 2244.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 2216.40 | 2211.06 | 2234.01 | SL hit (close>ema200) qty=0.50 sl=2211.06 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2256.60 | 2230.43 | 2229.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 2266.50 | 2237.65 | 2232.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2242.60 | 2242.69 | 2236.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 2242.60 | 2242.69 | 2236.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 2236.10 | 2243.50 | 2237.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 2236.10 | 2243.50 | 2237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 2227.00 | 2240.20 | 2236.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 2227.00 | 2240.20 | 2236.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 2208.20 | 2233.80 | 2234.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2199.90 | 2221.61 | 2228.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 2244.20 | 2225.74 | 2228.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2248.80 | 2230.36 | 2230.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 2248.80 | 2230.36 | 2230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 2249.00 | 2234.08 | 2232.30 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 2213.00 | 2233.45 | 2234.70 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 2244.10 | 2235.07 | 2234.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 2278.80 | 2243.82 | 2238.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 2279.00 | 2282.12 | 2272.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 2279.00 | 2282.12 | 2272.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2253.40 | 2275.43 | 2271.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 2255.00 | 2275.43 | 2271.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2249.00 | 2270.14 | 2269.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 2249.00 | 2270.14 | 2269.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 2246.20 | 2265.35 | 2266.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 2235.30 | 2259.34 | 2264.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 2242.50 | 2235.03 | 2242.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2242.60 | 2236.55 | 2242.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 2241.50 | 2236.55 | 2242.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2229.70 | 2235.18 | 2241.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 2222.30 | 2235.18 | 2241.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 2222.70 | 2233.96 | 2240.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 2224.70 | 2230.75 | 2237.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 2243.90 | 2231.84 | 2234.72 | SL hit (close>static) qty=1.00 sl=2243.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2257.90 | 2237.05 | 2236.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 2268.50 | 2254.26 | 2246.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 2259.60 | 2260.48 | 2252.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:30:00 | 2261.10 | 2260.48 | 2252.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2251.60 | 2258.71 | 2252.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 2251.60 | 2258.71 | 2252.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2255.50 | 2258.06 | 2252.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 2252.50 | 2258.06 | 2252.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2251.40 | 2256.73 | 2252.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 2251.90 | 2256.73 | 2252.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 2246.00 | 2254.59 | 2252.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 2257.20 | 2254.59 | 2252.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 2248.10 | 2252.25 | 2251.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 2247.20 | 2252.25 | 2251.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 2240.70 | 2249.94 | 2250.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 2238.50 | 2247.65 | 2249.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 2238.10 | 2236.55 | 2242.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2235.10 | 2236.26 | 2242.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 2246.60 | 2236.26 | 2242.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2230.00 | 2233.17 | 2237.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 2233.70 | 2233.17 | 2237.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2217.30 | 2211.47 | 2218.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 2217.30 | 2211.47 | 2218.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2226.20 | 2215.11 | 2219.00 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2235.60 | 2222.13 | 2221.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 2238.70 | 2225.45 | 2223.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2280.00 | 2280.67 | 2268.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 2280.00 | 2280.67 | 2268.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2265.00 | 2277.07 | 2270.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2265.00 | 2277.07 | 2270.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2271.10 | 2275.87 | 2270.84 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 2262.00 | 2267.21 | 2267.87 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 2276.10 | 2268.58 | 2268.36 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2253.10 | 2268.31 | 2268.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 2234.90 | 2261.63 | 2265.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2176.00 | 2162.44 | 2189.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2175.00 | 2162.44 | 2189.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2163.70 | 2166.49 | 2182.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 2148.10 | 2161.32 | 2175.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2151.20 | 2158.81 | 2170.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2151.70 | 2159.16 | 2167.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 2152.00 | 2157.20 | 2165.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2179.30 | 2160.79 | 2165.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 2179.30 | 2160.79 | 2165.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2185.80 | 2165.79 | 2167.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 2184.40 | 2165.79 | 2167.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 2156.30 | 2167.79 | 2168.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2146.60 | 2162.15 | 2165.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2092.10 | 2055.79 | 2079.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2070.20 | 2058.67 | 2078.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 2063.50 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 2064.00 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.32 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.80 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 14:15:00 | 1857.15 | 1985.50 | 2032.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 2018.50 | 1991.22 | 1990.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 2030.00 | 2006.04 | 1998.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 2027.00 | 2015.07 | 2008.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1943.60 | 1988.98 | 1997.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1982.80 | 1971.89 | 1983.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1978.70 | 1973.25 | 1983.36 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2181.90 | 2021.22 | 2003.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 2239.10 | 2189.37 | 2124.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 2214.50 | 2215.40 | 2172.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 2214.50 | 2215.40 | 2172.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2215.70 | 2219.19 | 2200.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 2208.90 | 2219.19 | 2200.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2217.50 | 2240.51 | 2229.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 2209.00 | 2240.51 | 2229.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 2245.50 | 2241.51 | 2230.80 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2222.50 | 2227.25 | 2227.74 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 2235.40 | 2229.41 | 2228.68 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2215.00 | 2226.62 | 2227.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 2208.30 | 2222.96 | 2225.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2171.70 | 2154.15 | 2174.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2171.70 | 2154.15 | 2174.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2180.30 | 2159.38 | 2175.20 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2199.80 | 2183.36 | 2181.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 2234.20 | 2193.53 | 2186.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 2210.40 | 2214.87 | 2200.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 2210.40 | 2214.87 | 2200.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2197.60 | 2211.41 | 2200.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 2198.60 | 2211.41 | 2200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2207.70 | 2210.67 | 2200.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:00:00 | 2211.80 | 2209.74 | 2202.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2193.30 | 2206.98 | 2202.81 | SL hit (close<static) qty=1.00 sl=2193.80 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2166.90 | 2198.96 | 2199.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2156.20 | 2178.44 | 2188.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 2165.00 | 2164.43 | 2175.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 2165.00 | 2164.43 | 2175.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2192.20 | 2169.31 | 2174.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 2198.60 | 2169.31 | 2174.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2188.90 | 2173.23 | 2176.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 2180.00 | 2174.76 | 2176.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 2180.60 | 2176.93 | 2177.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2200.90 | 2185.55 | 2182.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 2208.90 | 2211.94 | 2201.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 2207.40 | 2211.94 | 2201.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 2203.40 | 2209.51 | 2201.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 2203.40 | 2209.51 | 2201.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2216.60 | 2210.93 | 2203.15 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2190.90 | 2200.52 | 2200.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 2160.20 | 2191.39 | 2196.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2127.50 | 2126.24 | 2153.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 2127.50 | 2126.24 | 2153.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2080.50 | 2064.03 | 2088.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 2080.50 | 2064.03 | 2088.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2080.00 | 2067.23 | 2087.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2084.20 | 2069.38 | 2086.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2062.90 | 2068.09 | 2084.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 2059.00 | 2068.27 | 2081.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 2061.50 | 2066.45 | 2079.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1956.05 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1958.42 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 2000.10 | 1997.12 | 2016.67 | SL hit (close>ema200) qty=0.50 sl=1997.12 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2010.00 | 1978.91 | 1976.35 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1948.40 | 1984.01 | 1984.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1941.40 | 1970.47 | 1977.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1964.50 | 1955.14 | 1966.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1852.20 | 1856.22 | 1892.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1848.30 | 1856.22 | 1892.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1907.00 | 1857.89 | 1870.56 | SL hit (close>static) qty=1.00 sl=1893.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1891.50 | 1877.76 | 1877.72 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1849.00 | 1874.30 | 1876.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 1832.80 | 1861.87 | 1870.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1850.40 | 1803.28 | 1822.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1844.60 | 1811.54 | 1824.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 1856.00 | 1811.54 | 1824.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1854.90 | 1835.31 | 1833.76 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1786.20 | 1827.32 | 1830.69 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1839.20 | 1827.57 | 1826.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1863.40 | 1834.74 | 1829.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1868.70 | 1870.30 | 1852.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:45:00 | 1866.90 | 1870.30 | 1852.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2124.80 | 2082.59 | 2043.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2144.90 | 2082.59 | 2043.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2132.10 | 2093.87 | 2052.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 2130.00 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2130.20 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2249.80 | 2247.03 | 2235.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 2261.90 | 2247.03 | 2235.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 2261.70 | 2263.29 | 2259.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-28 09:15:00 | 2359.39 | 2328.13 | 2301.77 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 11:15:00 | 2422.36 | 2025-05-23 10:15:00 | 2446.98 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-27 11:15:00 | 2483.82 | 2025-05-28 12:15:00 | 2445.53 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-05-27 14:30:00 | 2473.35 | 2025-05-28 12:15:00 | 2445.53 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-12 09:15:00 | 2514.46 | 2025-06-12 10:15:00 | 2484.31 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-06-17 10:00:00 | 2445.14 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-06-17 10:45:00 | 2442.81 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-06-17 11:45:00 | 2436.12 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-07-03 13:30:00 | 2546.35 | 2025-07-03 15:15:00 | 2528.42 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-08 11:00:00 | 2500.30 | 2025-07-09 10:15:00 | 2521.82 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-08 11:30:00 | 2495.55 | 2025-07-09 10:15:00 | 2521.82 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-09 13:00:00 | 2498.56 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-09 13:30:00 | 2500.30 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-10 10:45:00 | 2501.17 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-07-10 12:45:00 | 2504.37 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-10 14:45:00 | 2501.85 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-14 11:15:00 | 2502.63 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-07-28 12:00:00 | 2476.84 | 2025-07-31 14:15:00 | 2353.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2474.80 | 2025-07-31 14:15:00 | 2351.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:00:00 | 2476.84 | 2025-08-06 11:15:00 | 2229.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2474.80 | 2025-08-06 14:15:00 | 2227.32 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-13 13:15:00 | 2213.14 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-08-13 14:30:00 | 2212.27 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-08-13 15:15:00 | 2217.21 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2025-08-14 10:00:00 | 2210.81 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-08-14 12:45:00 | 2224.87 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2252.79 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-28 12:15:00 | 2222.45 | 2025-09-02 10:15:00 | 2224.87 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-16 09:15:00 | 2322.21 | 2025-09-22 13:15:00 | 2554.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-09 12:15:00 | 2463.46 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-09 13:00:00 | 2463.75 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 14:00:00 | 2461.33 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2490.32 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-20 10:15:00 | 2480.23 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-21 13:45:00 | 2478.97 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-21 14:15:00 | 2477.71 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-28 14:15:00 | 2406.36 | 2025-10-29 09:15:00 | 2437.38 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-11 09:30:00 | 2282.17 | 2025-11-12 09:15:00 | 2436.22 | STOP_HIT | 1.00 | -6.75% |
| SELL | retest2 | 2025-11-28 13:30:00 | 2303.10 | 2025-12-03 13:15:00 | 2187.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 13:30:00 | 2303.10 | 2025-12-04 09:15:00 | 2216.40 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-12-18 14:15:00 | 2222.30 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-18 15:15:00 | 2222.70 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-19 11:15:00 | 2224.70 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-13 14:00:00 | 2148.10 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2151.20 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2151.70 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-14 15:15:00 | 2152.00 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 12:15:00 | 1960.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 12:15:00 | 1960.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 14:15:00 | 1857.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 14:15:00 | 1857.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 11:15:00 | 2027.00 | 2026-02-01 13:15:00 | 1965.80 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-02-18 15:00:00 | 2211.80 | 2026-02-19 09:15:00 | 2193.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-23 11:45:00 | 2180.00 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-23 13:15:00 | 2180.60 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-09 09:15:00 | 1956.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-09 09:15:00 | 1958.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-24 10:15:00 | 1848.30 | 2026-03-25 10:15:00 | 1907.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2144.90 | 2026-04-28 09:15:00 | 2359.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2132.10 | 2026-04-28 09:15:00 | 2345.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 14:45:00 | 2130.00 | 2026-04-28 09:15:00 | 2343.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:15:00 | 2130.20 | 2026-04-28 09:15:00 | 2343.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-23 10:15:00 | 2261.90 | 2026-05-04 09:15:00 | 2488.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 14:00:00 | 2261.70 | 2026-05-04 09:15:00 | 2487.87 | TARGET_HIT | 1.00 | 10.00% |
