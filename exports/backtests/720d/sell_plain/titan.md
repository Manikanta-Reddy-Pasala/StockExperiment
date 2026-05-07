# TITAN (TITAN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 4310.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 14 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** -2.06% / -2.04%
- **Sum % (uncompounded):** -20.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 10 | 0 | -2.06% | -20.6% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.74% | -11.2% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.34% | -9.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.74% | -11.2% |
| retest2 (combined) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.34% | -9.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 3314.05 | 3548.58 | 3548.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 3287.00 | 3539.17 | 3544.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 3313.80 | 3297.76 | 3384.13 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 3251.80 | 3299.58 | 3375.37 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 12:15:00 | 3215.70 | 3298.04 | 3373.84 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 3368.10 | 3295.17 | 3362.87 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-04 10:15:00 | 3368.10 | 3295.17 | 3362.87 | SL hit (close>ema400) qty=1.00 sl=3362.87 alert=retest1 |
| Cross detected — sustain check pending | 2024-12-20 15:15:00 | 3344.90 | 3371.29 | 3386.06 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-23 09:15:00 | 3352.15 | 3371.10 | 3385.89 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2024-12-26 11:15:00 | 3345.30 | 3371.69 | 3385.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 3330.50 | 3371.00 | 3384.61 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 3388.00 | 3345.37 | 3368.46 | SL hit (close>static) qty=1.00 sl=3375.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-14 11:15:00 | 3335.05 | 3389.72 | 3388.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 3321.50 | 3388.49 | 3388.04 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 3335.00 | 3387.25 | 3387.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 3335.00 | 3387.25 | 3387.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 3299.75 | 3386.38 | 3386.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 3394.00 | 3374.61 | 3380.66 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 3394.00 | 3374.61 | 3380.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 3394.00 | 3374.61 | 3380.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-22 13:15:00 | 3344.10 | 3374.47 | 3380.10 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-22 14:15:00 | 3358.30 | 3374.31 | 3379.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-27 13:15:00 | 3316.85 | 3376.73 | 3380.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 15:15:00 | 3312.00 | 3375.51 | 3380.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-28 14:15:00 | 3329.45 | 3373.37 | 3378.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-29 09:15:00 | 3354.00 | 3372.72 | 3378.49 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 3485.40 | 3373.12 | 3378.28 | SL hit (close>static) qty=1.00 sl=3395.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-10 11:15:00 | 3346.10 | 3419.65 | 3404.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 3311.45 | 3417.74 | 3403.34 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-02-13 14:15:00 | 3229.85 | 3390.32 | 3390.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 3229.85 | 3390.32 | 3390.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 3207.70 | 3385.38 | 3387.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 3167.65 | 3154.29 | 3234.98 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-24 09:15:00 | 3092.10 | 3156.34 | 3230.98 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 11:15:00 | 3081.50 | 3154.96 | 3229.55 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 3088.00 | 3120.15 | 3191.99 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 11:15:00 | 3085.20 | 3119.59 | 3190.99 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3133.45 | 3110.42 | 3182.11 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 14:15:00 | 3120.60 | 3111.76 | 3181.02 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 3117.25 | 3111.94 | 3180.42 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2025-04-09 13:15:00 | 3183.20 | 3114.13 | 3180.17 | SL hit (close>ema400) qty=1.00 sl=3180.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-09 13:15:00 | 3183.20 | 3114.13 | 3180.17 | SL hit (close>ema400) qty=1.00 sl=3180.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 3228.75 | 3118.61 | 3180.80 | SL hit (close>static) qty=1.00 sl=3227.25 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 3355.10 | 3464.89 | 3465.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3328.50 | 3460.14 | 3462.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 3449.00 | 3441.68 | 3452.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 12:15:00 | 3453.00 | 3441.79 | 3452.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 3453.00 | 3441.79 | 3452.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-06 13:15:00 | 3429.00 | 3441.67 | 3452.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 3400.00 | 3441.03 | 3451.89 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 3469.40 | 3439.12 | 3450.49 | SL hit (close>static) qty=1.00 sl=3462.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-11 09:15:00 | 3419.00 | 3440.53 | 3450.82 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-11 10:15:00 | 3450.00 | 3440.62 | 3450.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 3413.00 | 3552.31 | 3530.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 3420.00 | 3549.67 | 3529.58 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 3380.00 | 3511.17 | 3511.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 3380.00 | 3511.17 | 3511.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 3361.00 | 3496.94 | 3504.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-28 12:15:00 | 3215.70 | 2024-12-04 10:15:00 | 3368.10 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2024-12-26 13:15:00 | 3330.50 | 2025-01-02 14:15:00 | 3388.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-01-14 13:15:00 | 3321.50 | 2025-01-14 15:15:00 | 3335.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-01-27 15:15:00 | 3312.00 | 2025-01-31 09:15:00 | 3485.40 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-02-10 13:15:00 | 3311.45 | 2025-02-13 14:15:00 | 3229.85 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest1 | 2025-03-24 11:15:00 | 3081.50 | 2025-04-09 13:15:00 | 3183.20 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest1 | 2025-04-04 11:15:00 | 3085.20 | 2025-04-09 13:15:00 | 3183.20 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-04-09 09:15:00 | 3117.25 | 2025-04-11 11:15:00 | 3228.75 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-08-06 15:15:00 | 3400.00 | 2025-08-08 09:15:00 | 3469.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-09-23 11:15:00 | 3420.00 | 2025-09-29 11:15:00 | 3380.00 | STOP_HIT | 1.00 | 1.17% |
