# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2273.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 9 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -0.71% / -2.25%
- **Sum % (uncompounded):** -6.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 0 | 9 | 0 | -0.71% | -6.4% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 3.85% | 7.7% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 0 | 7 | 0 | -2.01% | -14.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 3.85% | 7.7% |
| retest2 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -2.01% | -14.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 2306.71 | 2276.05 | 2275.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2328.45 | 2276.89 | 2276.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.38 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.38 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 2346.55 | 2287.10 | 2282.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 2340.55 | 2287.63 | 2282.29 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-14 14:15:00 | 2314.77 | 2293.68 | 2285.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 2313.10 | 2293.87 | 2286.08 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-20 15:15:00 | 2309.66 | 2302.50 | 2291.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 2327.76 | 2302.75 | 2291.88 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-05-23 09:15:00 | 2312.12 | 2303.37 | 2292.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 2322.45 | 2303.56 | 2293.10 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 2307.79 | 2322.80 | 2308.90 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2260.67 | 2313.71 | 2305.87 | SL hit (close<static) qty=1.00 sl=2264.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2260.67 | 2313.71 | 2305.87 | SL hit (close<static) qty=1.00 sl=2264.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2260.67 | 2313.71 | 2305.87 | SL hit (close<static) qty=1.00 sl=2264.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2260.67 | 2313.71 | 2305.87 | SL hit (close<static) qty=1.00 sl=2264.71 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-07 09:15:00 | 2345.76 | 2283.62 | 2289.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 2360.61 | 2284.39 | 2290.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 2391.99 | 2296.04 | 2295.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2391.99 | 2296.04 | 2295.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2475.01 | 2306.18 | 2301.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2368.88 | 2375.75 | 2344.52 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-28 09:15:00 | 2400.75 | 2375.98 | 2345.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:15:00 | 2403.11 | 2376.25 | 2345.69 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-30 12:15:00 | 2396.62 | 2379.31 | 2349.63 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 13:15:00 | 2400.55 | 2379.52 | 2349.88 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2494.20 | 2546.03 | 2499.48 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2494.20 | 2546.03 | 2499.48 | SL hit (close<ema400) qty=1.00 sl=2499.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2494.20 | 2546.03 | 2499.48 | SL hit (close<ema400) qty=1.00 sl=2499.48 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-24 11:15:00 | 2524.10 | 2542.07 | 2499.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:15:00 | 2530.40 | 2541.95 | 2499.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 2473.44 | 2538.13 | 2499.75 | SL hit (close<static) qty=1.00 sl=2491.74 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 14:15:00 | 2519.97 | 2499.89 | 2490.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 2518.20 | 2500.07 | 2490.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 2469.01 | 2512.61 | 2498.59 | SL hit (close<static) qty=1.00 sl=2491.74 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 2465.00 | 2379.87 | 2379.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 2466.90 | 2382.31 | 2380.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 10:15:00 | 2340.55 | 2025-06-18 11:15:00 | 2260.67 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-05-14 15:15:00 | 2313.10 | 2025-06-18 11:15:00 | 2260.67 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-05-21 09:15:00 | 2327.76 | 2025-06-18 11:15:00 | 2260.67 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-05-23 10:15:00 | 2322.45 | 2025-06-18 11:15:00 | 2260.67 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-07-07 10:15:00 | 2360.61 | 2025-07-09 11:15:00 | 2391.99 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest1 | 2025-07-28 10:15:00 | 2403.11 | 2025-09-23 10:15:00 | 2494.20 | STOP_HIT | 1.00 | 3.79% |
| BUY | retest1 | 2025-07-30 13:15:00 | 2400.55 | 2025-09-23 10:15:00 | 2494.20 | STOP_HIT | 1.00 | 3.90% |
| BUY | retest2 | 2025-09-24 12:15:00 | 2530.40 | 2025-09-26 09:15:00 | 2473.44 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-16 15:15:00 | 2518.20 | 2025-10-24 09:15:00 | 2469.01 | STOP_HIT | 1.00 | -1.95% |
