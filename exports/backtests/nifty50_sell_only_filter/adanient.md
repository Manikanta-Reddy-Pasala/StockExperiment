# ADANIENT (ADANIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2540.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 14 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 7 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / Stop hits / Partials:** 2 / 8 / 2
- **Avg / median % per leg:** 8.25% / 5.80%
- **Sum % (uncompounded):** 98.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 0 | 7 | 0 | 1.51% | 10.6% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.28% | -0.6% |
| BUY @ 3rd Alert (retest2) | 5 | 3 | 60.0% | 0 | 5 | 0 | 2.23% | 11.1% |
| SELL (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 17.68% | 88.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.60% | -1.6% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 22.50% | 90.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.72% | -2.2% |
| retest2 (combined) | 9 | 7 | 77.8% | 2 | 5 | 2 | 11.24% | 101.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 2529.00 | 2480.99 | 2480.79 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2023-10-16 10:15:00 | 2449.00 | 2480.71 | 2480.74 | slope filter: EMA200 not falling 0.50% over 350 bars |
| First Alert — break + close above crossover candle high | 2023-12-04 12:15:00 | 2541.40 | 2306.17 | 2346.09 | Break + close above crossover candle high |

### Cycle 2 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2891.00 | 2383.34 | 2382.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 2924.20 | 2546.51 | 2472.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 2895.00 | 2908.15 | 2769.51 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-29 09:15:00 | 3046.70 | 2905.26 | 2778.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:15:00 | 3045.95 | 2906.66 | 2779.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 3033.35 | 3197.75 | 3076.27 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 3076.27 | 3197.75 | 3076.27 | SL hit qty=1.00 sl=3076.27 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-26 09:15:00 | 3141.00 | 3141.69 | 3073.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-26 10:15:00 | 3130.00 | 3141.57 | 3073.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-28 10:15:00 | 3155.60 | 3139.21 | 3077.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 3156.40 | 3139.38 | 3077.55 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 2981.40 | 3167.87 | 3119.19 | SL hit qty=1.00 sl=2981.40 alert=retest2 |
| CROSSOVER_SKIP | 2024-05-07 12:15:00 | 2835.00 | 3088.49 | 3089.46 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-05-21 11:15:00 | 3145.95 | 3025.97 | 3050.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:15:00 | 3159.85 | 3027.30 | 3051.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-22 14:15:00 | 3142.75 | 3035.18 | 3054.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-22 15:15:00 | 3137.00 | 3036.19 | 3054.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-23 09:15:00 | 3165.05 | 3037.47 | 3055.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 3200.05 | 3039.09 | 3056.05 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 3410.90 | 3136.60 | 3107.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-07-19 09:15:00 | 3064.50 | 3145.00 | 3145.17 | HTF filter: close above htf_sma |

### Cycle 4 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3128.80 | 3077.40 | 3077.34 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 3006.60 | 3077.15 | 3077.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 2998.90 | 3075.78 | 3076.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2980.00 | 2975.07 | 3020.06 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-30 15:15:00 | 2959.00 | 2974.78 | 3019.02 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-31 09:15:00 | 2963.95 | 2974.67 | 3018.75 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-10-31 10:15:00 | 2957.00 | 2974.50 | 3018.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 2961.00 | 2974.36 | 3018.15 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 3022.65 | 2964.08 | 3008.34 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 3008.34 | 2964.08 | 3008.34 | SL hit qty=1.00 sl=3008.34 alert=retest1 |
| Cross detected — sustain check pending | 2024-11-07 09:15:00 | 2965.95 | 2967.16 | 3008.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:15:00 | 2965.10 | 2967.14 | 3008.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-07 13:15:00 | 2976.35 | 2967.38 | 3008.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:15:00 | 2969.75 | 2967.40 | 3007.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 2520.34 | 2921.50 | 2973.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 2524.29 | 2921.50 | 2973.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-11-22 09:15:00 | 2075.57 | 2873.45 | 2947.66 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-11-22 09:15:00 | 2078.82 | 2873.45 | 2947.66 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 2451.50 | 2342.97 | 2342.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2478.70 | 2347.46 | 2345.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-05-09 15:15:00 | 2255.00 | 2344.08 | 2344.24 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 7 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2387.20 | 2344.51 | 2344.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 2416.70 | 2345.23 | 2344.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 2454.80 | 2463.35 | 2420.16 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-04 10:15:00 | 2478.00 | 2463.50 | 2420.45 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-04 11:15:00 | 2472.00 | 2463.58 | 2420.71 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-04 12:15:00 | 2478.60 | 2463.73 | 2420.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 13:15:00 | 2495.00 | 2464.05 | 2421.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 2459.00 | 2501.24 | 2456.09 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 2456.09 | 2501.24 | 2456.09 | SL hit qty=1.00 sl=2456.09 alert=retest1 |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 2475.60 | 2491.46 | 2455.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 2479.50 | 2491.35 | 2455.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 2450.10 | 2563.59 | 2533.26 | SL hit qty=1.00 sl=2450.10 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-07 09:15:00 | 2265.20 | 2507.17 | 2507.39 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 2497.40 | 2363.72 | 2395.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 2493.40 | 2365.01 | 2395.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 2616.50 | 2422.51 | 2422.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2616.50 | 2422.51 | 2422.42 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-11-20 10:15:00 | 2456.80 | 2469.85 | 2469.85 | HTF filter: close above htf_sma |

### Cycle 9 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2101.36 | 2100.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.45 | 2103.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-29 10:15:00 | 3045.95 | 2024-03-13 09:15:00 | 3076.27 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-03-28 11:15:00 | 3156.40 | 2024-04-19 09:15:00 | 2981.40 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2024-05-21 12:15:00 | 3159.85 | 2024-05-24 14:15:00 | 3385.60 | STOP_HIT | 1.00 | 7.14% |
| BUY | retest2 | 2024-05-23 10:15:00 | 3200.05 | 2024-05-24 14:15:00 | 3385.60 | STOP_HIT | 1.00 | 5.80% |
| SELL | retest1 | 2024-10-31 11:15:00 | 2961.00 | 2024-11-06 11:15:00 | 3008.34 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-11-07 10:15:00 | 2965.10 | 2024-11-21 09:15:00 | 2520.34 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 14:15:00 | 2969.75 | 2024-11-21 09:15:00 | 2524.29 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 10:15:00 | 2965.10 | 2024-11-22 09:15:00 | 2075.57 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2024-11-07 14:15:00 | 2969.75 | 2024-11-22 09:15:00 | 2078.82 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest1 | 2025-06-04 13:15:00 | 2495.00 | 2025-06-18 12:15:00 | 2456.09 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-06-23 11:15:00 | 2479.50 | 2025-07-31 14:15:00 | 2450.10 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-09-19 10:15:00 | 2493.40 | 2025-09-24 14:15:00 | 2616.50 | STOP_HIT | 1.00 | 4.94% |
