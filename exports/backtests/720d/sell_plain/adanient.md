# ADANIENT (ADANIENT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2519.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 14 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 7 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 5
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 7.34% / 4.17%
- **Sum % (uncompounded):** 102.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 9 | 64.3% | 2 | 8 | 4 | 7.34% | 102.8% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 2.72% | 10.9% |
| SELL @ 3rd Alert (retest2) | 10 | 7 | 70.0% | 2 | 5 | 3 | 9.19% | 91.9% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 2.72% | 10.9% |
| retest2 (combined) | 10 | 7 | 70.0% | 2 | 5 | 3 | 9.19% | 91.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 2921.06 | 2983.90 | 2984.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 2877.19 | 2977.82 | 2981.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2889.07 | 2884.28 | 2928.05 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-30 15:15:00 | 2868.71 | 2883.99 | 2927.03 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-31 09:15:00 | 2872.20 | 2883.87 | 2926.76 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-10-31 10:15:00 | 2866.77 | 2883.70 | 2926.46 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 12:15:00 | 2851.89 | 2883.26 | 2925.81 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-11-01 18:15:00 | 2851.30 | 2882.15 | 2924.20 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 09:15:00 | 2830.90 | 2881.64 | 2923.73 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 3780m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 2930.41 | 2873.29 | 2916.09 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 2930.41 | 2873.29 | 2916.09 | SL hit (close>ema400) qty=1.00 sl=2916.09 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 2930.41 | 2873.29 | 2916.09 | SL hit (close>ema400) qty=1.00 sl=2916.09 alert=retest1 |
| Cross detected — sustain check pending | 2024-11-07 09:15:00 | 2874.81 | 2876.27 | 2916.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 2876.37 | 2876.26 | 2916.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-11-07 13:15:00 | 2885.53 | 2876.49 | 2915.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 15:15:00 | 2872.24 | 2876.48 | 2915.46 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 2444.91 | 2832.00 | 2882.71 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 2441.40 | 2832.00 | 2882.71 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2024-11-22 09:15:00 | 2013.46 | 2785.40 | 2857.27 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-11-22 09:15:00 | 2010.57 | 2785.40 | 2857.27 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 2196.08 | 2430.63 | 2430.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2172.13 | 2428.06 | 2429.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2308.54 | 2272.40 | 2320.63 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 2333.36 | 2273.01 | 2320.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2333.36 | 2273.01 | 2320.70 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-06 09:15:00 | 2290.31 | 2428.82 | 2410.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 2265.69 | 2425.65 | 2408.67 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-11-07 13:15:00 | 2300.30 | 2412.63 | 2402.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-07 15:15:00 | 2302.53 | 2410.35 | 2401.70 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-11-10 09:15:00 | 2295.55 | 2409.21 | 2401.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-10 10:15:00 | 2311.74 | 2408.24 | 2400.73 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-10 11:15:00 | 2298.65 | 2407.15 | 2400.22 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 2301.56 | 2405.04 | 2399.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2436.22 | 2395.36 | 2394.53 | SL hit (close>static) qty=1.00 sl=2343.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2436.22 | 2395.36 | 2394.53 | SL hit (close>static) qty=1.00 sl=2343.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-27 10:15:00 | 2301.20 | 2401.34 | 2399.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 2279.30 | 2399.04 | 2398.37 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-27 14:15:00 | 2256.30 | 2396.43 | 2397.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2256.30 | 2396.43 | 2397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2249.00 | 2394.97 | 2396.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 2272.00 | 2270.61 | 2310.39 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-08 10:15:00 | 2234.90 | 2270.37 | 2303.72 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 12:15:00 | 2230.00 | 2269.48 | 2302.94 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-23 13:15:00 | 1895.50 | 2190.12 | 2247.85 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2181.90 | 2117.03 | 2195.32 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2181.90 | 2117.03 | 2195.32 | SL hit (close>ema200) qty=0.50 sl=2117.03 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-13 15:15:00 | 2127.00 | 2161.91 | 2200.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2130.00 | 2161.60 | 2200.35 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2163.97 | 2199.66 | SL hit (close>static) qty=1.00 sl=2233.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-02 09:15:00 | 2093.00 | 2175.42 | 2197.02 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 11:15:00 | 2091.20 | 2173.96 | 2196.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-30 14:15:00 | 1777.52 | 1997.24 | 2076.39 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2004.00 | 1960.46 | 2045.42 | SL hit (close>ema200) qty=0.50 sl=1960.46 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-13 14:15:00 | 2124.00 | 1986.82 | 2048.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-15 09:15:00 | 2181.70 | 1990.17 | 2050.00 | ENTRY2 sustain failed after 2580m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-10-31 12:15:00 | 2851.89 | 2024-11-06 11:15:00 | 2930.41 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest1 | 2024-11-04 09:15:00 | 2830.90 | 2024-11-06 11:15:00 | 2930.41 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-11-07 11:15:00 | 2876.37 | 2024-11-21 09:15:00 | 2444.91 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 15:15:00 | 2872.24 | 2024-11-21 09:15:00 | 2441.40 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 11:15:00 | 2876.37 | 2024-11-22 09:15:00 | 2013.46 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2024-11-07 15:15:00 | 2872.24 | 2024-11-22 09:15:00 | 2010.57 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-11-06 11:15:00 | 2265.69 | 2025-11-12 09:15:00 | 2436.22 | STOP_HIT | 1.00 | -7.53% |
| SELL | retest2 | 2025-11-10 13:15:00 | 2301.56 | 2025-11-12 09:15:00 | 2436.22 | STOP_HIT | 1.00 | -5.85% |
| SELL | retest2 | 2025-11-27 12:15:00 | 2279.30 | 2025-11-27 14:15:00 | 2256.30 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest1 | 2026-01-08 12:15:00 | 2230.00 | 2026-01-23 13:15:00 | 1895.50 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2026-01-08 12:15:00 | 2230.00 | 2026-02-03 09:15:00 | 2181.90 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2026-02-16 09:15:00 | 2130.00 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-03-02 11:15:00 | 2091.20 | 2026-03-30 14:15:00 | 1777.52 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-03-02 11:15:00 | 2091.20 | 2026-04-08 09:15:00 | 2004.00 | STOP_HIT | 0.50 | 4.17% |
