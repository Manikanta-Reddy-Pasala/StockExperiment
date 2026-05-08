# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 2600.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 13 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -3.16% / -2.48%
- **Sum % (uncompounded):** -22.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.21% | -8.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.44% | -7.3% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.53% | -1.5% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.43% | -13.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.43% | -13.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.43% | -20.6% |
| retest2 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.53% | -1.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 2342.40 | 2437.29 | 2437.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 2337.10 | 2436.29 | 2437.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.32 | 2418.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 14:15:00 | 2408.40 | 2404.32 | 2418.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 2408.40 | 2404.32 | 2418.97 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 15:15:00 | 2501.80 | 2432.18 | 2431.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2520.10 | 2433.06 | 2432.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.69 | 2682.43 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 2819.20 | 2797.87 | 2683.67 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-10 10:15:00 | 2805.20 | 2797.95 | 2684.27 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 2818.50 | 2791.64 | 2708.02 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-22 10:15:00 | 2803.70 | 2791.76 | 2708.50 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-22 14:15:00 | 2809.00 | 2792.15 | 2710.34 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-22 15:15:00 | 2801.90 | 2792.25 | 2710.80 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-23 13:15:00 | 2814.30 | 2792.70 | 2713.04 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-23 14:15:00 | 2806.50 | 2792.84 | 2713.51 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-23 15:15:00 | 2808.00 | 2792.99 | 2713.98 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 2819.60 | 2793.26 | 2714.50 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 2812.00 | 2784.90 | 2727.20 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 2823.20 | 2785.28 | 2727.67 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 2829.00 | 2791.82 | 2738.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 2823.40 | 2792.14 | 2738.67 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.00 | 2805.32 | 2753.83 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 2765.90 | 2804.45 | 2753.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2779.20 | 2804.20 | 2754.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.26 | SL hit (close<ema400) qty=1.00 sl=2754.26 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.26 | SL hit (close<ema400) qty=1.00 sl=2754.26 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.26 | SL hit (close<ema400) qty=1.00 sl=2754.26 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 2736.80 | 2802.03 | 2754.17 | SL hit (close<static) qty=1.00 sl=2750.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-23 10:15:00 | 2782.00 | 2779.98 | 2747.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 11:15:00 | 2755.60 | 2779.74 | 2747.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-23 12:15:00 | 2764.10 | 2779.58 | 2747.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 13:15:00 | 2729.00 | 2779.08 | 2747.71 | ENTRY2 sustain failed after 60m |

### Cycle 3 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.80 | 2720.57 | 2720.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2374.00 | 2700.46 | 2710.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2263.38 | 2367.16 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 10:15:00 | 2269.10 | 2263.44 | 2366.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:15:00 | 2255.70 | 2263.36 | 2366.12 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 2271.40 | 2264.01 | 2363.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 2272.10 | 2264.10 | 2363.46 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 2272.10 | 2264.32 | 2362.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 2265.00 | 2264.33 | 2362.09 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.18 | SL hit (close>ema400) qty=1.00 sl=2361.18 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.18 | SL hit (close>ema400) qty=1.00 sl=2361.18 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.18 | SL hit (close>ema400) qty=1.00 sl=2361.18 alert=retest1 |
| CROSSOVER_SKIP | 2026-05-07 12:15:00 | 2552.30 | 2410.96 | 2410.81 | min_gap filter: gap=0.006% < 0.010% |
| TREND_RESET | 2026-05-07 12:15:00 | 2552.30 | 2410.96 | 2410.81 | EMA inversion without crossover edge (EMA200=2410.96 EMA400=2410.81) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-24 09:15:00 | 2819.60 | 2026-01-19 14:15:00 | 2753.30 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2026-01-05 11:15:00 | 2823.20 | 2026-01-19 14:15:00 | 2753.30 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2026-01-09 10:15:00 | 2823.40 | 2026-01-19 14:15:00 | 2753.30 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-19 15:15:00 | 2736.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest1 | 2026-04-08 11:15:00 | 2255.70 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest1 | 2026-04-09 10:15:00 | 2272.10 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest1 | 2026-04-09 13:15:00 | 2265.00 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.40% |
