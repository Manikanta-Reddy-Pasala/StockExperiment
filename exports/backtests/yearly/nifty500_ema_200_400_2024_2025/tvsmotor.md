# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3701.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 10 |
| TARGET_HIT | 9 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 3
- **Target hits / Stop hits / Partials:** 9 / 9 / 10
- **Avg / median % per leg:** 5.16% / 5.00%
- **Sum % (uncompounded):** 144.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| SELL (all) | 23 | 20 | 87.0% | 4 | 9 | 10 | 4.11% | 94.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 20 | 87.0% | 4 | 9 | 10 | 4.11% | 94.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 25 | 89.3% | 9 | 9 | 10 | 5.16% | 144.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 2125.25 | 2050.47 | 2050.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 2150.85 | 2058.62 | 2054.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 2107.00 | 2142.41 | 2105.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 2107.00 | 2142.41 | 2105.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 2107.00 | 2142.41 | 2105.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 2107.00 | 2142.41 | 2105.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 2192.30 | 2142.91 | 2105.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 2201.35 | 2142.91 | 2105.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 14:30:00 | 2216.15 | 2144.36 | 2106.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:00:00 | 2221.95 | 2144.36 | 2106.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-07 09:15:00 | 2421.49 | 2175.22 | 2125.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 2477.25 | 2656.79 | 2657.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2444.40 | 2647.08 | 2652.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 2494.60 | 2493.72 | 2549.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 2494.60 | 2493.72 | 2549.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 2557.20 | 2494.24 | 2548.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 2557.20 | 2494.24 | 2548.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 2555.50 | 2494.85 | 2548.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 2554.10 | 2494.85 | 2548.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 2547.75 | 2495.38 | 2548.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:30:00 | 2539.45 | 2495.70 | 2548.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 13:00:00 | 2528.15 | 2495.70 | 2548.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 14:15:00 | 2561.15 | 2496.86 | 2548.83 | SL hit (close>static) qty=1.00 sl=2556.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 10:15:00 | 2565.30 | 2451.25 | 2451.12 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 15:15:00 | 2362.65 | 2452.26 | 2452.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 2350.75 | 2451.25 | 2451.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 11:15:00 | 2342.00 | 2341.67 | 2380.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:45:00 | 2336.45 | 2341.67 | 2380.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2424.55 | 2342.69 | 2379.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:00:00 | 2424.55 | 2342.69 | 2379.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 2425.05 | 2343.51 | 2379.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:30:00 | 2423.30 | 2343.51 | 2379.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 2451.95 | 2365.44 | 2386.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 2451.95 | 2365.44 | 2386.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2369.65 | 2393.84 | 2398.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 2368.85 | 2393.82 | 2398.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 12:15:00 | 2432.15 | 2394.31 | 2398.50 | SL hit (close>static) qty=1.00 sl=2421.35 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2525.40 | 2402.55 | 2402.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2594.00 | 2409.75 | 2406.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 2726.30 | 2735.14 | 2655.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:45:00 | 2720.90 | 2735.14 | 2655.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2716.10 | 2739.52 | 2672.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 2731.90 | 2738.70 | 2673.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:00:00 | 2742.60 | 2738.73 | 2674.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-07 14:15:00 | 3005.09 | 2853.35 | 2808.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 3423.80 | 3687.62 | 3687.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 11:15:00 | 3383.60 | 3684.59 | 3686.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3690.90 | 3550.12 | 3604.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3690.90 | 3550.12 | 3604.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3690.90 | 3550.12 | 3604.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 3702.00 | 3550.12 | 3604.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 3688.90 | 3551.50 | 3604.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:15:00 | 3716.10 | 3551.50 | 3604.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 3751.50 | 3644.27 | 3643.97 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 3494.10 | 3642.59 | 3643.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 3468.90 | 3634.31 | 3639.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 3618.00 | 3588.00 | 3611.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 3618.00 | 3588.00 | 3611.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 3618.00 | 3588.00 | 3611.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:45:00 | 3618.00 | 3588.00 | 3611.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 3618.00 | 3588.30 | 3611.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 3658.30 | 3588.30 | 3611.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 3696.10 | 3590.17 | 3612.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 3696.10 | 3590.17 | 3612.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:15:00 | 2201.35 | 2024-06-07 09:15:00 | 2421.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 14:30:00 | 2216.15 | 2024-06-10 09:15:00 | 2437.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 15:00:00 | 2221.95 | 2024-06-10 09:15:00 | 2444.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-03 12:30:00 | 2539.45 | 2024-12-03 14:15:00 | 2561.15 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-03 13:00:00 | 2528.15 | 2024-12-03 14:15:00 | 2561.15 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-12-04 09:45:00 | 2538.00 | 2024-12-20 14:15:00 | 2411.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 11:45:00 | 2512.45 | 2024-12-20 14:15:00 | 2386.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 2515.05 | 2024-12-20 14:15:00 | 2389.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:00:00 | 2516.05 | 2024-12-20 14:15:00 | 2390.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:30:00 | 2516.00 | 2024-12-20 14:15:00 | 2390.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 2507.15 | 2024-12-20 15:15:00 | 2381.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 09:45:00 | 2538.00 | 2025-01-02 11:15:00 | 2478.10 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2024-12-04 11:45:00 | 2512.45 | 2025-01-02 11:15:00 | 2478.10 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2024-12-16 10:15:00 | 2515.05 | 2025-01-02 11:15:00 | 2478.10 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2024-12-16 12:00:00 | 2516.05 | 2025-01-02 11:15:00 | 2478.10 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2024-12-16 12:30:00 | 2516.00 | 2025-01-02 11:15:00 | 2478.10 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2024-12-17 09:15:00 | 2507.15 | 2025-01-02 11:15:00 | 2478.10 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2025-01-03 10:15:00 | 2475.05 | 2025-01-08 11:15:00 | 2352.96 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-01-03 12:15:00 | 2476.80 | 2025-01-08 11:15:00 | 2356.47 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-01-03 14:15:00 | 2480.50 | 2025-01-08 11:15:00 | 2353.77 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-01-06 09:15:00 | 2477.65 | 2025-01-08 12:15:00 | 2351.30 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-01-03 10:15:00 | 2475.05 | 2025-01-13 09:15:00 | 2227.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 2476.80 | 2025-01-13 09:15:00 | 2229.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 14:15:00 | 2480.50 | 2025-01-13 09:15:00 | 2232.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 2477.65 | 2025-01-13 09:15:00 | 2229.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-08 10:30:00 | 2368.85 | 2025-04-08 12:15:00 | 2432.15 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-06-13 13:15:00 | 2731.90 | 2025-08-07 14:15:00 | 3005.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 10:00:00 | 2742.60 | 2025-08-07 14:15:00 | 3016.86 | TARGET_HIT | 1.00 | 10.00% |
