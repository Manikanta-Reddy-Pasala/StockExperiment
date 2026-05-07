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
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 16 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 3 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -0.95% / -1.53%
- **Sum % (uncompounded):** -7.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 8 | 0 | -0.95% | -7.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.44% | -7.3% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.06% | -0.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.44% | -7.3% |
| retest2 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.06% | -0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 2436.50 | 2321.02 | 2320.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 2442.20 | 2322.22 | 2321.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2377.60 | 2391.17 | 2364.14 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2377.60 | 2391.17 | 2364.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2377.60 | 2391.17 | 2364.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-02 14:15:00 | 2419.90 | 2297.93 | 2307.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:15:00 | 2426.30 | 2299.21 | 2307.90 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-04 15:15:00 | 2428.00 | 2316.53 | 2316.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 2428.00 | 2316.53 | 2316.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 2448.00 | 2317.84 | 2316.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 2369.70 | 2371.85 | 2350.18 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 15:15:00 | 2378.80 | 2371.91 | 2350.53 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-22 09:15:00 | 2367.80 | 2371.87 | 2350.62 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-07-23 15:15:00 | 2381.20 | 2371.11 | 2351.56 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-24 09:15:00 | 2365.00 | 2371.05 | 2351.63 | ENTRY1 sustain failed after 1080m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 2344.00 | 2370.73 | 2351.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 2344.00 | 2370.73 | 2351.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-29 13:15:00 | 2398.10 | 2366.29 | 2351.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 2399.80 | 2366.62 | 2351.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 2322.70 | 2465.98 | 2456.38 | SL hit (close<static) qty=1.00 sl=2341.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 2369.90 | 2440.69 | 2444.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 2368.70 | 2439.98 | 2443.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 2337.10 | 2436.28 | 2441.71 | SL hit (close<static) qty=1.00 sl=2341.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-15 12:15:00 | 2373.60 | 2406.21 | 2424.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:15:00 | 2379.00 | 2405.94 | 2424.11 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 15:15:00 | 2514.60 | 2438.15 | 2437.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2514.60 | 2438.15 | 2437.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.44 | 2440.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.69 | 2683.47 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 2819.20 | 2797.87 | 2684.70 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-10 10:15:00 | 2805.20 | 2797.95 | 2685.30 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 2818.50 | 2791.64 | 2708.80 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-22 10:15:00 | 2803.70 | 2791.76 | 2709.27 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-22 14:15:00 | 2809.00 | 2792.15 | 2711.10 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-22 15:15:00 | 2801.90 | 2792.25 | 2711.56 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-23 13:15:00 | 2814.30 | 2792.70 | 2713.78 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-23 14:15:00 | 2806.50 | 2792.84 | 2714.24 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-23 15:15:00 | 2808.00 | 2792.99 | 2714.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 2819.60 | 2793.26 | 2715.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 2812.00 | 2784.90 | 2727.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 2823.20 | 2785.28 | 2728.24 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 2829.00 | 2791.82 | 2738.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 2823.40 | 2792.14 | 2739.16 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.00 | 2805.32 | 2754.25 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 2765.90 | 2804.45 | 2754.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2779.20 | 2804.20 | 2754.44 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.66 | SL hit (close<ema400) qty=1.00 sl=2754.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.66 | SL hit (close<ema400) qty=1.00 sl=2754.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.66 | SL hit (close<ema400) qty=1.00 sl=2754.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 2736.80 | 2802.03 | 2754.57 | SL hit (close<static) qty=1.00 sl=2750.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-23 10:15:00 | 2782.00 | 2779.98 | 2748.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 11:15:00 | 2755.60 | 2779.74 | 2748.08 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-23 12:15:00 | 2764.10 | 2779.58 | 2748.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 13:15:00 | 2729.00 | 2779.08 | 2748.06 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 2552.30 | 2410.96 | 2410.85 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-02 15:15:00 | 2426.30 | 2025-07-04 15:15:00 | 2428.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-07-29 14:15:00 | 2399.80 | 2025-10-01 09:15:00 | 2322.70 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-10-07 13:15:00 | 2368.70 | 2025-10-08 10:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-15 13:15:00 | 2379.00 | 2025-10-27 15:15:00 | 2514.60 | STOP_HIT | 1.00 | 5.70% |
| BUY | retest1 | 2025-12-24 09:15:00 | 2819.60 | 2026-01-19 14:15:00 | 2753.30 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2026-01-05 11:15:00 | 2823.20 | 2026-01-19 14:15:00 | 2753.30 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2026-01-09 10:15:00 | 2823.40 | 2026-01-19 14:15:00 | 2753.30 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-19 15:15:00 | 2736.80 | STOP_HIT | 1.00 | -1.53% |
