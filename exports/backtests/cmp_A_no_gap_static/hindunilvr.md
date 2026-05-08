# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 2286.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 22 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 1 |
| ENTRY2 | 12 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 7
- **Target hits / Stop hits / Partials:** 2 / 11 / 5
- **Avg / median % per leg:** 2.22% / 3.46%
- **Sum % (uncompounded):** 39.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 11 | 61.1% | 2 | 11 | 5 | 2.22% | 39.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.79% | -1.8% |
| SELL @ 3rd Alert (retest2) | 17 | 11 | 64.7% | 2 | 10 | 5 | 2.45% | 41.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.79% | -1.8% |
| retest2 (combined) | 17 | 11 | 64.7% | 2 | 10 | 5 | 2.45% | 41.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 2409.99 | 2485.30 | 2485.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 2398.68 | 2477.39 | 2481.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2418.95 | 2418.53 | 2442.74 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-27 12:15:00 | 2411.67 | 2418.46 | 2442.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 13:15:00 | 2402.32 | 2418.30 | 2442.38 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2437.54 | 2418.77 | 2441.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 2445.41 | 2419.04 | 2441.46 | SL hit (close>ema400) qty=1.00 sl=2441.46 alert=retest1 |
| Cross detected — sustain check pending | 2025-12-01 14:15:00 | 2421.80 | 2419.54 | 2441.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 2418.85 | 2419.53 | 2441.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 09:15:00 | 2414.81 | 2420.39 | 2440.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-03 10:15:00 | 2429.18 | 2420.48 | 2440.70 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 14:15:00 | 2409.80 | 2420.50 | 2440.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 2401.63 | 2420.32 | 2440.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-04 14:15:00 | 2422.29 | 2420.22 | 2439.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-04 15:15:00 | 2423.86 | 2420.25 | 2439.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-05 09:15:00 | 2361.60 | 2419.67 | 2439.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2297.91 | 2419.67 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 2351.10 | 2418.99 | 2438.57 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 09:15:00 | 2281.55 | 2388.13 | 2418.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2368.82 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2368.82 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 15:15:00 | 2421.00 | 2337.37 | 2369.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2411.10 | 2338.10 | 2370.14 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 2370.70 | 2346.49 | 2371.92 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-09 13:15:00 | 2364.10 | 2346.66 | 2371.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 14:15:00 | 2372.70 | 2346.92 | 2371.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-14 10:15:00 | 2351.40 | 2353.00 | 2373.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 2360.40 | 2353.07 | 2372.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 12:15:00 | 2362.00 | 2353.43 | 2372.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 2357.20 | 2353.47 | 2372.29 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2353.91 | 2372.23 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2353.91 | 2372.23 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-21 13:15:00 | 2364.90 | 2361.61 | 2374.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-21 14:15:00 | 2368.90 | 2361.68 | 2374.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-21 15:15:00 | 2365.90 | 2361.72 | 2374.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-22 09:15:00 | 2381.20 | 2361.91 | 2374.66 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-01-22 12:15:00 | 2366.70 | 2362.30 | 2374.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-22 13:15:00 | 2369.60 | 2362.37 | 2374.64 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 2362.60 | 2368.85 | 2377.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-28 10:15:00 | 2370.10 | 2368.86 | 2377.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 11:15:00 | 2361.90 | 2368.79 | 2376.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 2363.40 | 2368.74 | 2376.85 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 2386.00 | 2368.81 | 2376.77 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-29 09:15:00 | 2326.40 | 2368.39 | 2376.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 2323.10 | 2367.94 | 2376.25 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 2372.10 | 2366.85 | 2375.45 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-01 15:15:00 | 2331.40 | 2365.60 | 2374.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 2319.90 | 2365.15 | 2374.00 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 2382.40 | 2363.90 | 2372.97 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2385.80 | 2364.28 | 2373.07 | SL hit (close>static) qty=1.00 sl=2384.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 2339.00 | 2364.96 | 2372.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 10:15:00 | 2362.40 | 2364.93 | 2372.66 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2449.90 | 2371.47 | 2375.53 | SL hit (close>static) qty=1.00 sl=2447.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2449.90 | 2371.47 | 2375.53 | SL hit (close>static) qty=1.00 sl=2447.37 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 2465.00 | 2379.87 | 2379.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 2466.90 | 2382.31 | 2380.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.83 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2296.30 | 2379.30 | 2379.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 2283.40 | 2364.72 | 2371.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 11:15:00 | 2360.10 | 2358.38 | 2367.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 12:15:00 | 2367.80 | 2358.47 | 2367.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 2367.80 | 2358.47 | 2367.69 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-25 11:15:00 | 2352.20 | 2358.60 | 2367.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 12:15:00 | 2359.50 | 2358.61 | 2367.44 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-27 09:15:00 | 2343.30 | 2359.86 | 2367.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 2351.30 | 2359.78 | 2367.53 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-27 14:15:00 | 2339.50 | 2359.44 | 2367.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 2340.00 | 2359.25 | 2367.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 2233.74 | 2350.04 | 2361.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:15:00 | 2223.00 | 2341.64 | 2357.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-12 12:15:00 | 2116.17 | 2299.81 | 2332.56 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-19 09:15:00 | 2106.00 | 2259.96 | 2306.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2353.00 | 2199.62 | 2230.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2343.40 | 2201.05 | 2230.91 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 2226.23 | 2229.75 | 2242.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 2230.20 | 2229.75 | 2242.32 | SL hit (close>static) qty=0.50 sl=2229.75 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-11-27 13:15:00 | 2402.32 | 2025-12-01 10:15:00 | 2445.41 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-01 15:15:00 | 2418.85 | 2025-12-05 09:15:00 | 2297.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 15:15:00 | 2401.63 | 2025-12-12 09:15:00 | 2281.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 2418.85 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-12-03 15:15:00 | 2401.63 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-12-05 10:15:00 | 2351.10 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2411.10 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2026-01-14 11:15:00 | 2360.40 | 2026-01-28 15:15:00 | 2386.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-16 13:15:00 | 2357.20 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-28 12:15:00 | 2363.40 | 2026-02-03 13:15:00 | 2385.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-29 10:15:00 | 2323.10 | 2026-02-10 09:15:00 | 2449.90 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2026-02-02 09:15:00 | 2319.90 | 2026-02-10 09:15:00 | 2449.90 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2026-02-27 10:15:00 | 2351.30 | 2026-03-05 09:15:00 | 2233.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 2340.00 | 2026-03-06 10:15:00 | 2223.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:15:00 | 2351.30 | 2026-03-12 12:15:00 | 2116.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 2340.00 | 2026-03-19 09:15:00 | 2106.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 2343.40 | 2026-04-30 10:15:00 | 2226.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 2343.40 | 2026-04-30 10:15:00 | 2230.20 | STOP_HIT | 0.50 | 4.83% |
