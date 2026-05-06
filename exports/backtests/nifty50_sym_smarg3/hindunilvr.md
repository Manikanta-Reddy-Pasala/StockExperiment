# HINDUNILVR (HINDUNILVR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2317.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 5 |
| PENDING | 18 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 0 / 15 / 0
- **Avg / median % per leg:** -0.76% / -0.94%
- **Sum % (uncompounded):** -11.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 8 | 0 | 0.23% | 1.8% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 2.40% | 7.2% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.08% | -5.4% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.89% | -13.3% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 2.28% | 6.8% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.03% | -20.1% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 6 | 0 | 2.34% | 14.0% |
| retest2 (combined) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.83% | -25.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2566.65 | 2524.61 | 2524.50 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2023-12-11 09:15:00 | 2513.00 | 2524.34 | 2524.38 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 2561.15 | 2524.04 | 2523.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 13:15:00 | 2576.30 | 2527.95 | 2525.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 12:15:00 | 2566.55 | 2572.18 | 2552.63 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-08 14:15:00 | 2577.75 | 2572.23 | 2552.85 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 15:15:00 | 2579.00 | 2572.29 | 2552.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 2544.75 | 2572.46 | 2554.84 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-11 13:15:00 | 2554.84 | 2572.46 | 2554.84 | SL hit qty=1.00 sl=2554.84 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-15 13:15:00 | 2563.55 | 2568.58 | 2554.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 14:15:00 | 2572.65 | 2568.62 | 2554.11 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 10:15:00 | 2544.00 | 2568.77 | 2554.90 | SL hit qty=1.00 sl=2544.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-17 11:15:00 | 2560.60 | 2568.69 | 2554.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-17 12:15:00 | 2552.95 | 2568.54 | 2554.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-17 13:15:00 | 2559.75 | 2568.45 | 2554.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 14:15:00 | 2564.15 | 2568.41 | 2554.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 2544.00 | 2568.14 | 2554.99 | SL hit qty=1.00 sl=2544.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-19 13:15:00 | 2559.70 | 2566.26 | 2554.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 14:15:00 | 2567.65 | 2566.28 | 2554.79 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 2544.00 | 2564.99 | 2554.26 | SL hit qty=1.00 sl=2544.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 2457.00 | 2543.39 | 2543.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 2440.00 | 2542.36 | 2543.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 2436.35 | 2430.06 | 2465.98 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-05 09:15:00 | 2399.85 | 2428.65 | 2462.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:15:00 | 2397.05 | 2428.33 | 2462.49 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 2402.75 | 2423.57 | 2456.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 2402.30 | 2423.35 | 2456.13 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2355.00 | 2260.49 | 2311.07 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 2311.07 | 2260.49 | 2311.07 | SL hit qty=1.00 sl=2311.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 2311.07 | 2260.49 | 2311.07 | SL hit qty=1.00 sl=2311.07 alert=retest1 |
| CROSSOVER_SKIP | 2024-06-04 11:15:00 | 2438.75 | 2334.12 | 2333.96 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2024-10-25 14:15:00 | 2527.95 | 2766.03 | 2766.38 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-02-19 12:15:00 | 2250.70 | 2362.89 | 2401.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:15:00 | 2253.05 | 2361.80 | 2401.13 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-25 14:15:00 | 2260.25 | 2332.58 | 2380.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 2254.00 | 2331.80 | 2379.51 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-26 13:15:00 | 2248.60 | 2250.63 | 2301.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 2246.90 | 2250.60 | 2301.19 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-28 15:15:00 | 2257.00 | 2251.29 | 2297.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-01 09:15:00 | 2269.85 | 2251.47 | 2297.74 | ENTRY2 sustain failed after 5400m |
| Cross detected — sustain check pending | 2025-04-01 10:15:00 | 2248.00 | 2251.44 | 2297.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 2226.00 | 2251.18 | 2297.13 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 2300.00 | 2248.49 | 2288.21 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2357.75 | 2257.41 | 2290.47 | SL hit qty=1.00 sl=2357.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2357.75 | 2257.41 | 2290.47 | SL hit qty=1.00 sl=2357.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2357.75 | 2257.41 | 2290.47 | SL hit qty=1.00 sl=2357.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2357.75 | 2257.41 | 2290.47 | SL hit qty=1.00 sl=2357.75 alert=retest2 |
| CROSSOVER_SKIP | 2025-05-05 13:15:00 | 2347.20 | 2313.33 | 2313.19 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-06-23 09:15:00 | 2265.00 | 2342.87 | 2339.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-23 10:15:00 | 2266.80 | 2342.11 | 2339.56 | ENTRY2 sustain failed after 60m |

### Cycle 4 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2278.10 | 2336.79 | 2336.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2269.80 | 2336.12 | 2336.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2321.00 | 2320.27 | 2327.52 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-03 14:15:00 | 2314.20 | 2320.21 | 2327.38 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 15:15:00 | 2314.80 | 2320.16 | 2327.32 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2321.00 | 2320.17 | 2327.29 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 2327.29 | 2320.17 | 2327.29 | SL hit qty=1.00 sl=2327.29 alert=retest1 |

### Cycle 5 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2431.70 | 2334.16 | 2333.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2516.00 | 2344.48 | 2339.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2408.50 | 2415.21 | 2383.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-28 09:15:00 | 2440.60 | 2415.45 | 2384.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:15:00 | 2443.00 | 2415.72 | 2384.60 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-30 12:15:00 | 2436.30 | 2418.83 | 2388.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 13:15:00 | 2440.40 | 2419.04 | 2388.87 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2535.60 | 2588.37 | 2541.03 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2541.03 | 2588.37 | 2541.03 | SL hit qty=1.00 sl=2541.03 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2541.03 | 2588.37 | 2541.03 | SL hit qty=1.00 sl=2541.03 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-24 11:15:00 | 2566.00 | 2584.31 | 2540.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:15:00 | 2572.40 | 2584.19 | 2540.97 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 14:15:00 | 2533.30 | 2581.28 | 2541.38 | SL hit qty=1.00 sl=2533.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 14:15:00 | 2562.30 | 2541.38 | 2532.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 2560.00 | 2541.56 | 2532.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 2533.30 | 2554.25 | 2540.03 | SL hit qty=1.00 sl=2533.30 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-03 13:15:00 | 2456.10 | 2528.67 | 2528.99 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-08 15:15:00 | 2579.00 | 2024-01-11 13:15:00 | 2554.84 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-01-15 14:15:00 | 2572.65 | 2024-01-17 10:15:00 | 2544.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-01-17 14:15:00 | 2564.15 | 2024-01-18 09:15:00 | 2544.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-01-19 14:15:00 | 2567.65 | 2024-01-23 09:15:00 | 2544.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest1 | 2024-03-05 10:15:00 | 2397.05 | 2024-05-07 09:15:00 | 2311.07 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest1 | 2024-03-11 12:15:00 | 2402.30 | 2024-05-07 09:15:00 | 2311.07 | STOP_HIT | 1.00 | 3.80% |
| SELL | retest2 | 2025-02-19 13:15:00 | 2253.05 | 2025-04-11 09:15:00 | 2357.75 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2025-02-25 15:15:00 | 2254.00 | 2025-04-11 09:15:00 | 2357.75 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-03-26 14:15:00 | 2246.90 | 2025-04-11 09:15:00 | 2357.75 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2025-04-01 11:15:00 | 2226.00 | 2025-04-11 09:15:00 | 2357.75 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest1 | 2025-07-03 15:15:00 | 2314.80 | 2025-07-04 09:15:00 | 2327.29 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-07-28 10:15:00 | 2443.00 | 2025-09-23 10:15:00 | 2541.03 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest1 | 2025-07-30 13:15:00 | 2440.40 | 2025-09-23 10:15:00 | 2541.03 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2025-09-24 12:15:00 | 2572.40 | 2025-09-25 14:15:00 | 2533.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-16 15:15:00 | 2560.00 | 2025-10-24 09:15:00 | 2533.30 | STOP_HIT | 1.00 | -1.04% |
