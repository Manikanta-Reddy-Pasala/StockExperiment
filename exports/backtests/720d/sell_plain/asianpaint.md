# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2536.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 24 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 2 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 12
- **Target hits / Stop hits / Partials:** 0 / 13 / 0
- **Avg / median % per leg:** -2.25% / -1.06%
- **Sum % (uncompounded):** -29.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 1 | 7.7% | 0 | 13 | 0 | -2.25% | -29.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.27% | -8.5% |
| SELL @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 0 | 11 | 0 | -1.89% | -20.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.27% | -8.5% |
| retest2 (combined) | 11 | 1 | 9.1% | 0 | 11 | 0 | -1.89% | -20.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 2986.70 | 3116.44 | 3116.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2959.85 | 3114.88 | 3115.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 2301.20 | 2298.91 | 2414.22 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-04 09:15:00 | 2274.20 | 2301.41 | 2406.68 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-04 10:15:00 | 2293.40 | 2301.33 | 2406.11 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 13:15:00 | 2396.00 | 2302.55 | 2405.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 2396.00 | 2302.55 | 2405.16 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-05 09:15:00 | 2262.65 | 2303.15 | 2403.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 2276.05 | 2302.53 | 2402.63 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-24 15:15:00 | 2317.95 | 2257.86 | 2298.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-25 09:15:00 | 2325.00 | 2258.52 | 2298.77 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-03-26 11:15:00 | 2307.10 | 2264.00 | 2299.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-26 13:15:00 | 2320.80 | 2265.04 | 2299.96 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-03-26 15:15:00 | 2315.20 | 2266.13 | 2300.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 2313.30 | 2266.60 | 2300.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-04-01 13:15:00 | 2310.25 | 2277.82 | 2303.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 2316.70 | 2278.59 | 2303.34 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 2308.85 | 2284.65 | 2304.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-04 10:15:00 | 2321.40 | 2285.01 | 2304.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-04 11:15:00 | 2320.00 | 2285.36 | 2304.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-04 12:15:00 | 2335.00 | 2285.85 | 2304.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 2317.50 | 2288.15 | 2305.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-07 10:15:00 | 2342.45 | 2288.69 | 2305.95 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.00 | 2300.47 | 2310.84 | SL hit (close>static) qty=1.00 sl=2418.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.00 | 2300.47 | 2310.84 | SL hit (close>static) qty=1.00 sl=2418.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.00 | 2300.47 | 2310.84 | SL hit (close>static) qty=1.00 sl=2418.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-08 14:15:00 | 2306.10 | 2386.25 | 2363.20 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 2304.70 | 2384.59 | 2362.60 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1140m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2363.80 | 2379.71 | 2360.86 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-14 09:15:00 | 2292.90 | 2374.30 | 2359.37 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 2290.00 | 2372.67 | 2358.70 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-16 09:15:00 | 2315.90 | 2364.61 | 2355.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-16 11:15:00 | 2325.60 | 2363.76 | 2355.01 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-05-20 11:15:00 | 2320.80 | 2360.79 | 2354.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 2304.40 | 2359.74 | 2353.61 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-23 12:15:00 | 2319.60 | 2349.87 | 2349.00 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:15:00 | 2314.50 | 2349.20 | 2348.68 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-26 10:15:00 | 2319.40 | 2348.44 | 2348.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 2307.90 | 2347.65 | 2347.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.46 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-27 13:15:00 | 2329.60 | 2346.58 | 2347.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 2327.50 | 2346.19 | 2347.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 2351.80 | 2281.93 | 2300.90 | SL hit (close>static) qty=1.00 sl=2351.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-30 11:15:00 | 2334.90 | 2285.18 | 2302.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 2333.80 | 2286.08 | 2302.36 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 2369.00 | 2288.00 | 2303.08 | SL hit (close>static) qty=1.00 sl=2351.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-25 11:15:00 | 2332.00 | 2368.94 | 2351.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 13:15:00 | 2333.30 | 2368.23 | 2351.23 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 2358.10 | 2367.31 | 2351.10 | SL hit (close>static) qty=1.00 sl=2351.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 09:15:00 | 2322.70 | 2465.98 | 2456.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:15:00 | 2331.30 | 2463.28 | 2455.11 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 2353.90 | 2453.76 | 2450.58 | SL hit (close>static) qty=1.00 sl=2351.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.74 | 2447.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.28 | 2441.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.32 | 2422.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.96 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-29 14:15:00 | 2417.00 | 2732.73 | 2727.18 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-30 09:15:00 | 2423.60 | 2726.52 | 2724.11 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 2419.80 | 2723.47 | 2722.59 | ENTRY2 cross detected — sustain check pending (75m) |

### Cycle 4 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.80 | 2720.57 | 2721.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2374.00 | 2700.46 | 2710.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2263.38 | 2367.23 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 10:15:00 | 2269.10 | 2263.44 | 2366.74 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 12:15:00 | 2267.00 | 2263.40 | 2365.69 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 2271.40 | 2264.01 | 2363.98 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-09 11:15:00 | 2278.90 | 2264.24 | 2363.10 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 2272.10 | 2264.32 | 2362.64 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 14:15:00 | 2268.70 | 2264.37 | 2361.69 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-05 11:15:00 | 2276.05 | 2025-04-09 10:15:00 | 2419.00 | STOP_HIT | 1.00 | -6.28% |
| SELL | retest2 | 2025-03-27 09:15:00 | 2313.30 | 2025-04-09 10:15:00 | 2419.00 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-04-01 15:15:00 | 2316.70 | 2025-04-09 10:15:00 | 2419.00 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2025-05-09 09:15:00 | 2304.70 | 2025-05-26 11:15:00 | 2308.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-05-14 11:15:00 | 2290.00 | 2025-05-26 11:15:00 | 2308.60 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-20 13:15:00 | 2304.40 | 2025-05-26 11:15:00 | 2308.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-05-23 14:15:00 | 2314.50 | 2025-05-26 11:15:00 | 2308.60 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-05-27 15:15:00 | 2327.50 | 2025-06-27 13:15:00 | 2351.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-06-30 13:15:00 | 2333.80 | 2025-07-01 09:15:00 | 2369.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-07-25 13:15:00 | 2333.30 | 2025-07-28 10:15:00 | 2358.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-01 11:15:00 | 2331.30 | 2025-10-03 12:15:00 | 2353.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest1 | 2026-04-08 12:15:00 | 2267.00 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest1 | 2026-04-09 14:15:00 | 2268.70 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.23% |
