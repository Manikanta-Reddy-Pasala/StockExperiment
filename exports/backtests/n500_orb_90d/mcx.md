# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3098.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 3
- **Avg / median % per leg:** 0.19% / -0.24%
- **Sum % (uncompounded):** 2.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.33% | 3.0% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.33% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.07% | -0.4% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.07% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 2 | 9 | 3 | 0.19% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 2486.00 | 2463.21 | 0.00 | ORB-long ORB[2435.00,2470.00] vol=3.7x ATR=10.55 |
| Stop hit — per-position SL triggered | 2026-02-10 09:35:00 | 2475.45 | 2466.61 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 2315.20 | 2303.93 | 0.00 | ORB-long ORB[2280.00,2306.00] vol=1.7x ATR=7.19 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 2308.01 | 2305.31 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 2434.50 | 2422.89 | 0.00 | ORB-long ORB[2400.20,2433.00] vol=2.2x ATR=6.32 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 2428.18 | 2424.03 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:50:00 | 2604.00 | 2589.87 | 0.00 | ORB-long ORB[2570.00,2593.80] vol=1.9x ATR=11.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:10:00 | 2620.58 | 2596.85 | 0.00 | T1 1.5R @ 2620.58 |
| Target hit | 2026-03-17 15:20:00 | 2676.80 | 2640.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-03-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 09:40:00 | 2472.60 | 2445.04 | 0.00 | ORB-long ORB[2423.20,2451.70] vol=1.9x ATR=11.59 |
| Stop hit — per-position SL triggered | 2026-03-27 10:55:00 | 2461.01 | 2460.40 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 2808.00 | 2850.86 | 0.00 | ORB-short ORB[2848.00,2884.00] vol=2.7x ATR=11.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:00:00 | 2791.22 | 2845.71 | 0.00 | T1 1.5R @ 2791.22 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 2808.00 | 2827.02 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 2818.00 | 2836.05 | 0.00 | ORB-short ORB[2826.00,2865.70] vol=2.6x ATR=6.76 |
| Stop hit — per-position SL triggered | 2026-04-21 11:05:00 | 2824.76 | 2835.55 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 2796.00 | 2813.28 | 0.00 | ORB-short ORB[2806.30,2834.80] vol=1.6x ATR=9.00 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 2805.00 | 2806.19 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:00:00 | 2871.80 | 2853.30 | 0.00 | ORB-long ORB[2833.10,2854.90] vol=1.8x ATR=7.27 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 2864.53 | 2858.84 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:05:00 | 2945.20 | 2965.34 | 0.00 | ORB-short ORB[2951.00,2985.10] vol=1.5x ATR=11.89 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 2957.09 | 2960.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 2946.50 | 2933.02 | 0.00 | ORB-long ORB[2911.00,2939.90] vol=1.9x ATR=9.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:15:00 | 2961.25 | 2937.21 | 0.00 | T1 1.5R @ 2961.25 |
| Target hit | 2026-05-06 15:20:00 | 2970.00 | 2961.78 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 2486.00 | 2026-02-10 09:35:00 | 2475.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-18 11:10:00 | 2315.20 | 2026-02-18 11:45:00 | 2308.01 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-25 09:30:00 | 2434.50 | 2026-02-25 09:40:00 | 2428.18 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-17 09:50:00 | 2604.00 | 2026-03-17 10:10:00 | 2620.58 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-03-17 09:50:00 | 2604.00 | 2026-03-17 15:20:00 | 2676.80 | TARGET_HIT | 0.50 | 2.80% |
| BUY | retest1 | 2026-03-27 09:40:00 | 2472.60 | 2026-03-27 10:55:00 | 2461.01 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-17 10:55:00 | 2808.00 | 2026-04-17 11:00:00 | 2791.22 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-17 10:55:00 | 2808.00 | 2026-04-17 12:15:00 | 2808.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 11:00:00 | 2818.00 | 2026-04-21 11:05:00 | 2824.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-22 09:35:00 | 2796.00 | 2026-04-22 10:00:00 | 2805.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 10:00:00 | 2871.80 | 2026-04-28 10:40:00 | 2864.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-30 10:05:00 | 2945.20 | 2026-04-30 10:40:00 | 2957.09 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-06 10:00:00 | 2946.50 | 2026-05-06 10:15:00 | 2961.25 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-06 10:00:00 | 2946.50 | 2026-05-06 15:20:00 | 2970.00 | TARGET_HIT | 0.50 | 0.80% |
