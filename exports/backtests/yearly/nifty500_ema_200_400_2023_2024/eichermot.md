# Eicher Motors Ltd. (EICHERMOT)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 7309.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 12 |
| ALERT2 | 13 |
| ALERT2_SKIP | 8 |
| ALERT3 | 75 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 4 |
| TARGET_HIT | 9 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 46
- **Target hits / Stop hits / Partials:** 9 / 51 / 4
- **Avg / median % per leg:** 0.73% / -0.78%
- **Sum % (uncompounded):** 46.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 9 | 26.5% | 9 | 25 | 0 | 1.40% | 47.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 9 | 26.5% | 9 | 25 | 0 | 1.40% | 47.5% |
| SELL (all) | 30 | 9 | 30.0% | 0 | 26 | 4 | -0.02% | -0.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 9 | 30.0% | 0 | 26 | 4 | -0.02% | -0.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 64 | 18 | 28.1% | 9 | 51 | 4 | 0.73% | 46.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 3313.05 | 3434.02 | 3434.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 09:15:00 | 3289.95 | 3400.06 | 3415.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 09:15:00 | 3413.25 | 3380.47 | 3401.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 3413.25 | 3380.47 | 3401.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 3413.25 | 3380.47 | 3401.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 3413.25 | 3380.47 | 3401.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 3418.85 | 3380.85 | 3401.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:00:00 | 3418.85 | 3380.85 | 3401.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 3427.00 | 3381.31 | 3402.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:45:00 | 3428.65 | 3381.31 | 3402.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 3416.00 | 3381.94 | 3402.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 13:45:00 | 3414.85 | 3381.94 | 3402.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 3415.20 | 3382.27 | 3402.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:45:00 | 3415.20 | 3382.27 | 3402.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 3403.95 | 3379.65 | 3399.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:45:00 | 3385.80 | 3379.71 | 3399.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 09:45:00 | 3383.60 | 3377.16 | 3395.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 11:15:00 | 3382.00 | 3379.76 | 3395.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 12:00:00 | 3385.45 | 3379.82 | 3395.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 3400.00 | 3380.15 | 3395.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 14:00:00 | 3400.00 | 3380.15 | 3395.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 3386.05 | 3380.21 | 3395.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 14:30:00 | 3400.70 | 3380.21 | 3395.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 3388.05 | 3360.93 | 3378.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:30:00 | 3393.50 | 3360.93 | 3378.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 3389.05 | 3361.21 | 3378.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:30:00 | 3399.60 | 3361.21 | 3378.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 3374.25 | 3363.13 | 3379.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:45:00 | 3378.45 | 3363.13 | 3379.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 3372.75 | 3363.23 | 3379.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:30:00 | 3387.00 | 3363.23 | 3379.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 3367.85 | 3362.07 | 3377.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:30:00 | 3360.05 | 3362.07 | 3377.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 3386.00 | 3362.31 | 3377.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:00:00 | 3386.00 | 3362.31 | 3377.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 3399.35 | 3362.68 | 3378.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 11:45:00 | 3375.85 | 3371.62 | 3381.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 12:30:00 | 3378.95 | 3371.74 | 3381.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-07 09:15:00 | 3383.40 | 3372.19 | 3381.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 11:15:00 | 3383.40 | 3374.26 | 3382.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 11:15:00 | 3376.90 | 3374.29 | 3382.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 13:00:00 | 3371.15 | 3374.25 | 3382.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 14:15:00 | 3374.80 | 3374.27 | 3382.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 11:30:00 | 3374.55 | 3374.10 | 3381.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 12:00:00 | 3371.90 | 3374.10 | 3381.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 3394.00 | 3374.30 | 3381.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-11 12:15:00 | 3394.00 | 3374.30 | 3381.87 | SL hit (close>static) qty=1.00 sl=3385.85 alert=retest2 |

### Cycle 2 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 3401.25 | 3386.28 | 3386.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 3510.75 | 3387.85 | 3387.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 3326.70 | 3405.60 | 3396.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 3326.70 | 3405.60 | 3396.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 3326.70 | 3405.60 | 3396.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 11:30:00 | 3437.85 | 3402.12 | 3395.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 13:15:00 | 3435.00 | 3402.43 | 3395.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 15:00:00 | 3444.75 | 3403.19 | 3396.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 09:45:00 | 3435.55 | 3447.68 | 3425.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 11:15:00 | 3409.75 | 3447.17 | 3425.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:30:00 | 3405.10 | 3447.17 | 3425.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 12:15:00 | 3370.00 | 3446.40 | 3424.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 13:00:00 | 3370.00 | 3446.40 | 3424.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 3344.35 | 3432.20 | 3419.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-30 11:00:00 | 3344.35 | 3432.20 | 3419.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-31 14:15:00 | 3296.20 | 3421.90 | 3414.89 | SL hit (close<static) qty=1.00 sl=3302.90 alert=retest2 |

### Cycle 3 — SELL (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 12:15:00 | 3322.10 | 3408.26 | 3408.27 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 10:15:00 | 3526.85 | 3408.86 | 3408.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 3530.25 | 3410.07 | 3409.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 3952.30 | 3962.90 | 3818.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:00:00 | 3952.30 | 3962.90 | 3818.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 3816.70 | 3935.63 | 3831.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:00:00 | 3816.70 | 3935.63 | 3831.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 3827.00 | 3934.55 | 3831.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:30:00 | 3814.35 | 3934.55 | 3831.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 3838.75 | 3930.33 | 3831.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 14:30:00 | 3830.00 | 3930.33 | 3831.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 3824.75 | 3922.32 | 3834.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:45:00 | 3815.00 | 3922.32 | 3834.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 3835.00 | 3921.45 | 3834.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 3841.35 | 3800.00 | 3791.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:30:00 | 3858.50 | 3808.24 | 3796.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 14:30:00 | 3841.40 | 3810.26 | 3797.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:15:00 | 3889.40 | 3810.51 | 3797.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 3798.30 | 3831.44 | 3810.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 3798.30 | 3831.44 | 3810.30 | SL hit (close<static) qty=1.00 sl=3821.95 alert=retest2 |

### Cycle 5 — SELL (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 15:15:00 | 3747.50 | 3826.02 | 3826.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 3702.20 | 3824.79 | 3825.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 3858.70 | 3810.82 | 3818.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 3858.70 | 3810.82 | 3818.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 3858.70 | 3810.82 | 3818.43 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 3998.25 | 3826.69 | 3826.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 4008.20 | 3849.60 | 3838.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4540.00 | 4614.41 | 4424.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 4540.00 | 4614.41 | 4424.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 4392.50 | 4612.20 | 4424.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 4392.50 | 4612.20 | 4424.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 4462.65 | 4610.71 | 4424.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:45:00 | 4518.95 | 4609.71 | 4425.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 4548.95 | 4608.67 | 4425.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 09:45:00 | 4516.90 | 4608.00 | 4427.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 4968.59 | 4680.14 | 4510.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 4675.00 | 4805.59 | 4806.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 4567.35 | 4803.22 | 4805.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4843.25 | 4791.25 | 4798.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 4843.25 | 4791.25 | 4798.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 4843.25 | 4791.25 | 4798.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 4842.45 | 4791.25 | 4798.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 4795.70 | 4791.29 | 4798.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 4793.65 | 4791.45 | 4798.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 4770.70 | 4791.87 | 4798.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:30:00 | 4794.00 | 4791.99 | 4798.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 13:00:00 | 4790.80 | 4791.98 | 4798.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 4786.75 | 4791.93 | 4798.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:45:00 | 4797.60 | 4791.93 | 4798.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4553.97 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4532.16 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4554.30 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4551.26 | 4771.64 | 4787.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 4735.30 | 4755.89 | 4778.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 4823.00 | 4756.50 | 4778.37 | SL hit (close>ema200) qty=0.50 sl=4756.50 alert=retest2 |

### Cycle 8 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 4939.50 | 4797.53 | 4796.98 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 4583.25 | 4796.25 | 4796.87 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 4897.40 | 4797.27 | 4797.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 4975.00 | 4802.32 | 4799.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4820.80 | 4851.28 | 4827.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 4820.80 | 4851.28 | 4827.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4820.80 | 4851.28 | 4827.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 4830.10 | 4851.28 | 4827.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 4835.85 | 4851.12 | 4827.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 4819.65 | 4851.12 | 4827.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 4817.15 | 4850.78 | 4827.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 4817.15 | 4850.78 | 4827.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 4835.50 | 4850.63 | 4827.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 4835.50 | 4850.63 | 4827.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 4796.00 | 4850.09 | 4827.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 4796.00 | 4850.09 | 4827.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 4849.00 | 4850.08 | 4827.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:00:00 | 4857.75 | 4846.06 | 4827.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 4852.15 | 4846.09 | 4827.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 4874.15 | 4845.67 | 4827.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 4787.00 | 4844.48 | 4827.10 | SL hit (close<static) qty=1.00 sl=4793.80 alert=retest2 |

### Cycle 11 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 4750.10 | 4818.14 | 4818.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 4724.15 | 4817.20 | 4817.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 4935.45 | 4810.36 | 4814.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 4935.45 | 4810.36 | 4814.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 4935.45 | 4810.36 | 4814.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 4935.45 | 4810.36 | 4814.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 4883.95 | 4811.10 | 4814.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 4879.20 | 4811.10 | 4814.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:00:00 | 4880.35 | 4812.81 | 4815.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 4876.50 | 4813.44 | 4815.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:30:00 | 4881.90 | 4815.46 | 4816.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 4881.75 | 4817.60 | 4817.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 4881.75 | 4817.60 | 4817.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 4897.40 | 4821.83 | 4819.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.93 | 4899.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 11:00:00 | 4943.40 | 4960.93 | 4899.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5184.26 | 5069.13 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 14:15:00 | 4905.00 | 4993.00 | 4993.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 09:15:00 | 4847.00 | 4990.64 | 4992.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4986.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4986.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4986.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 4986.40 | 4980.63 | 4986.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 4970.00 | 4980.52 | 4986.88 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 14:15:00 | 5099.00 | 4993.25 | 4992.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 15:15:00 | 5130.00 | 4994.61 | 4993.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 4996.50 | 4997.07 | 4994.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 14:15:00 | 4996.50 | 4997.07 | 4994.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 4996.50 | 4997.07 | 4994.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 4975.05 | 4997.07 | 4994.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 4972.50 | 4996.82 | 4994.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 4940.00 | 4996.82 | 4994.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 4975.50 | 4996.61 | 4994.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 4954.60 | 4996.61 | 4994.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 4968.55 | 4996.33 | 4994.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:45:00 | 4970.35 | 4996.33 | 4994.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 4987.15 | 4996.02 | 4994.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 4980.90 | 4996.02 | 4994.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 4983.40 | 4995.90 | 4994.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 4983.40 | 4995.90 | 4994.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 5005.00 | 4995.99 | 4994.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 4971.30 | 4995.99 | 4994.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 4969.00 | 4995.72 | 4994.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 4971.30 | 4995.72 | 4994.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 4999.75 | 4995.76 | 4994.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 5020.25 | 4995.85 | 4994.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 5013.00 | 4996.95 | 4994.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 5010.25 | 4997.08 | 4995.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 5023.25 | 4997.41 | 4995.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 5046.30 | 4998.28 | 4995.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 5079.40 | 5001.44 | 4997.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-01 09:15:00 | 5522.28 | 5153.58 | 5084.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 6854.00 | 7352.51 | 7353.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.07 | 7342.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.48 | 7177.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:00:00 | 7072.00 | 7049.48 | 7177.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 7060.00 | 7052.41 | 7176.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 7026.00 | 7123.00 | 7172.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:00:00 | 7009.50 | 7121.87 | 7171.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 7213.50 | 7120.51 | 7169.49 | SL hit (close>static) qty=1.00 sl=7180.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-08-04 10:45:00 | 3385.80 | 2023-09-11 12:15:00 | 3394.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2023-08-10 09:45:00 | 3383.60 | 2023-09-11 12:15:00 | 3394.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-08-11 11:15:00 | 3382.00 | 2023-09-11 12:15:00 | 3394.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2023-08-11 12:00:00 | 3385.45 | 2023-09-11 12:15:00 | 3394.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2023-09-06 11:45:00 | 3375.85 | 2023-09-15 09:15:00 | 3428.25 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-09-06 12:30:00 | 3378.95 | 2023-09-15 09:15:00 | 3428.25 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-09-07 09:15:00 | 3383.40 | 2023-09-15 09:15:00 | 3428.25 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-09-08 11:15:00 | 3383.40 | 2023-09-15 09:15:00 | 3428.25 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-09-08 13:00:00 | 3371.15 | 2023-09-20 13:15:00 | 3443.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2023-09-08 14:15:00 | 3374.80 | 2023-09-20 13:15:00 | 3443.20 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-09-11 11:30:00 | 3374.55 | 2023-09-25 09:15:00 | 3401.25 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-09-11 12:00:00 | 3371.90 | 2023-09-25 09:15:00 | 3401.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-09-20 10:30:00 | 3405.80 | 2023-09-25 09:15:00 | 3401.25 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2023-09-20 11:30:00 | 3402.50 | 2023-09-25 09:15:00 | 3401.25 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2023-09-22 09:30:00 | 3410.10 | 2023-09-25 09:15:00 | 3401.25 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2023-09-22 14:15:00 | 3410.00 | 2023-09-25 09:15:00 | 3401.25 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2023-10-05 11:30:00 | 3437.85 | 2023-10-31 14:15:00 | 3296.20 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2023-10-05 13:15:00 | 3435.00 | 2023-10-31 14:15:00 | 3296.20 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2023-10-05 15:00:00 | 3444.75 | 2023-10-31 14:15:00 | 3296.20 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2023-10-25 09:45:00 | 3435.55 | 2023-10-31 14:15:00 | 3296.20 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2024-01-31 15:15:00 | 3841.35 | 2024-02-08 13:15:00 | 3798.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-02-02 09:30:00 | 3858.50 | 2024-02-08 13:15:00 | 3798.30 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-02-02 14:30:00 | 3841.40 | 2024-02-08 13:15:00 | 3798.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-02-05 09:15:00 | 3889.40 | 2024-02-08 13:15:00 | 3798.30 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-02-09 12:00:00 | 3815.00 | 2024-02-29 12:15:00 | 3772.55 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-02-20 10:30:00 | 3815.65 | 2024-02-29 12:15:00 | 3772.55 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-02-29 11:00:00 | 3823.30 | 2024-02-29 12:15:00 | 3772.55 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-03-01 10:15:00 | 3831.30 | 2024-03-04 09:15:00 | 3788.40 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-03-01 13:15:00 | 3839.20 | 2024-03-04 09:15:00 | 3788.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-03-12 11:45:00 | 3844.70 | 2024-03-12 13:15:00 | 3807.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-06-04 13:45:00 | 4518.95 | 2024-06-18 09:15:00 | 4968.59 | TARGET_HIT | 1.00 | 9.95% |
| BUY | retest2 | 2024-06-04 15:15:00 | 4548.95 | 2024-06-18 10:15:00 | 4970.85 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2024-06-05 09:45:00 | 4516.90 | 2024-07-26 09:15:00 | 5003.85 | TARGET_HIT | 1.00 | 10.78% |
| SELL | retest2 | 2024-10-21 12:30:00 | 4793.65 | 2024-10-25 10:15:00 | 4553.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 4770.70 | 2024-10-25 10:15:00 | 4532.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 11:30:00 | 4794.00 | 2024-10-25 10:15:00 | 4554.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 13:00:00 | 4790.80 | 2024-10-25 10:15:00 | 4551.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 4793.65 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2024-10-22 10:00:00 | 4770.70 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -1.10% |
| SELL | retest2 | 2024-10-22 11:30:00 | 4794.00 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -0.60% |
| SELL | retest2 | 2024-10-22 13:00:00 | 4790.80 | 2024-10-29 11:15:00 | 4823.00 | STOP_HIT | 0.50 | -0.67% |
| BUY | retest2 | 2024-12-03 10:00:00 | 4857.75 | 2024-12-04 11:15:00 | 4787.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-12-03 10:45:00 | 4852.15 | 2024-12-04 11:15:00 | 4787.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-12-04 09:15:00 | 4874.15 | 2024-12-04 11:15:00 | 4787.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-05 14:15:00 | 4864.30 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-06 09:15:00 | 4925.70 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-12-09 10:45:00 | 4853.05 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-09 11:30:00 | 4855.95 | 2024-12-10 12:15:00 | 4818.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-12-09 15:15:00 | 4857.40 | 2024-12-12 09:15:00 | 4780.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-12-13 13:30:00 | 4833.60 | 2024-12-16 10:15:00 | 4802.95 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-12-13 14:45:00 | 4842.50 | 2024-12-16 10:15:00 | 4802.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-12-16 14:15:00 | 4837.65 | 2024-12-17 10:15:00 | 4808.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-12-27 11:15:00 | 4879.20 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-12-27 14:00:00 | 4880.35 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-12-27 14:45:00 | 4876.50 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-12-30 10:30:00 | 4881.90 | 2024-12-30 13:15:00 | 4881.75 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-03-12 13:15:00 | 5020.25 | 2025-04-01 09:15:00 | 5522.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 09:45:00 | 5013.00 | 2025-04-01 09:15:00 | 5514.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 11:00:00 | 5010.25 | 2025-04-01 09:15:00 | 5511.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 14:00:00 | 5023.25 | 2025-04-15 14:15:00 | 5525.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 09:15:00 | 5079.40 | 2025-04-16 09:15:00 | 5587.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 15:15:00 | 5078.40 | 2025-04-16 09:15:00 | 5586.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:30:00 | 7026.00 | 2026-05-04 09:15:00 | 7213.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-04-30 11:00:00 | 7009.50 | 2026-05-04 09:15:00 | 7213.50 | STOP_HIT | 1.00 | -2.91% |
