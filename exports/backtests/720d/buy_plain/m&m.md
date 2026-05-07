# M&M (M&M)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 3366.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 15 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 9
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 8.95% / 15.00%
- **Sum % (uncompounded):** 35.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 3 | 1 | 8.95% | 35.8% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 5.33% | 5.3% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 2 | 1 | 10.15% | 30.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 5.33% | 5.3% |
| retest2 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 10.15% | 30.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 12:15:00 | 3170.05 | 2975.77 | 2975.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 13:15:00 | 3181.35 | 2977.82 | 2976.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3012.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3012.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3012.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 3102.00 | 2794.77 | 2800.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 3084.70 | 2797.66 | 2802.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 3071.40 | 2808.28 | 2807.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 3071.40 | 2808.28 | 2807.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3097.30 | 2816.19 | 2811.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2977.50 | 2983.48 | 2922.99 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-02 10:15:00 | 3018.70 | 2983.47 | 2925.35 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-02 11:15:00 | 3017.40 | 2983.81 | 2925.81 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-02 12:15:00 | 3028.10 | 2984.25 | 2926.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 13:15:00 | 3031.60 | 2984.72 | 2926.84 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2989.30 | 3020.83 | 2962.28 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 13:15:00 | 3003.40 | 3019.76 | 2962.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 3005.70 | 3019.62 | 2963.11 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 3193.10 | 3267.93 | 3200.06 | SL hit (close<ema400) qty=1.00 sl=3200.06 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-04 09:15:00 | 3456.55 | 3271.28 | 3209.15 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 3483.20 | 3490.84 | 3376.06 | SL hit (close<ema200) qty=0.50 sl=3490.84 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-16 13:15:00 | 3001.00 | 3347.68 | 3447.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:15:00 | 3035.70 | 3344.58 | 3445.51 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 12:15:00 | 3033.30 | 3254.37 | 3378.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 13:15:00 | 3046.70 | 3252.30 | 3376.64 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 3047.90 | 3209.02 | 3339.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 3033.70 | 3207.27 | 3338.30 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 3053.90 | 3205.75 | 3336.88 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-01 12:15:00 | 3065.80 | 3204.36 | 3335.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 13:15:00 | 3046.00 | 3202.78 | 3334.08 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 3221.60 | 3160.50 | 3296.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 3224.20 | 3161.14 | 3296.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 3060.00 | 3181.43 | 3267.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 3086.10 | 3180.48 | 3266.19 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-04-24 11:15:00 | 3066.80 | 3178.05 | 3264.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-24 12:15:00 | 3047.80 | 3176.76 | 3263.04 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-27 09:15:00 | 3084.40 | 3171.85 | 3258.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 10:15:00 | 3083.30 | 3170.96 | 3257.97 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-30 11:15:00 | 3068.00 | 3160.18 | 3243.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 3081.40 | 3159.40 | 3242.40 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 3097.40 | 3158.78 | 3241.68 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-05-04 09:15:00 | 3123.80 | 3157.21 | 3239.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 3140.00 | 3157.04 | 3239.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 12:15:00 | 3174.20 | 3152.90 | 3233.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:15:00 | 3148.00 | 3152.85 | 3233.00 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-06 10:15:00 | 3084.70 | 2025-05-06 14:15:00 | 3071.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-02 13:15:00 | 3031.60 | 2025-08-29 14:15:00 | 3193.10 | STOP_HIT | 1.00 | 5.33% |
| BUY | retest2 | 2025-06-13 14:15:00 | 3005.70 | 2025-09-04 09:15:00 | 3456.55 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-13 14:15:00 | 3005.70 | 2025-09-26 09:15:00 | 3483.20 | STOP_HIT | 0.50 | 15.89% |
