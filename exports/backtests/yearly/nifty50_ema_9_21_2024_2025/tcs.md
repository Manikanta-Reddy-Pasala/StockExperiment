# TCS (TCS)

## Backtest Summary

- **Window:** 2024-03-12 09:15:00 → 2026-05-08 15:15:00 (3717 bars)
- **Last close:** 2397.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 105 |
| ALERT2 | 103 |
| ALERT2_SKIP | 58 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 111 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 94
- **Target hits / Stop hits / Partials:** 2 / 114 / 5
- **Avg / median % per leg:** -0.25% / -0.84%
- **Sum % (uncompounded):** -30.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 12 | 22.2% | 0 | 54 | 0 | -0.55% | -29.8% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.32% | -1.3% |
| BUY @ 3rd Alert (retest2) | 50 | 11 | 22.0% | 0 | 50 | 0 | -0.57% | -28.5% |
| SELL (all) | 67 | 15 | 22.4% | 2 | 60 | 5 | -0.00% | -0.2% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL @ 3rd Alert (retest2) | 66 | 14 | 21.2% | 1 | 60 | 5 | -0.15% | -10.2% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 4 | 0 | 1.74% | 8.7% |
| retest2 (combined) | 116 | 25 | 21.6% | 1 | 110 | 5 | -0.33% | -38.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 3945.65 | 3931.02 | 3929.26 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 09:15:00 | 3908.00 | 3926.41 | 3927.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 14:15:00 | 3903.50 | 3919.84 | 3923.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 3896.55 | 3891.67 | 3901.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 3896.55 | 3891.67 | 3901.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 3852.50 | 3883.83 | 3896.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 14:15:00 | 3842.75 | 3872.97 | 3884.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-18 09:45:00 | 3847.00 | 3857.19 | 3873.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 3820.90 | 3855.59 | 3869.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:00:00 | 3846.00 | 3833.57 | 3838.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 3864.55 | 3839.77 | 3841.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:00:00 | 3864.55 | 3839.77 | 3841.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 3870.80 | 3845.97 | 3843.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 3870.80 | 3845.97 | 3843.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 3897.35 | 3863.62 | 3853.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 3854.15 | 3867.38 | 3857.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 3854.15 | 3867.38 | 3857.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 3854.15 | 3867.38 | 3857.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 3854.15 | 3867.38 | 3857.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 3860.15 | 3865.94 | 3857.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:45:00 | 3863.90 | 3863.03 | 3857.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 13:15:00 | 3849.05 | 3860.23 | 3856.62 | SL hit (close<static) qty=1.00 sl=3853.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 3850.65 | 3862.00 | 3862.28 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 3865.10 | 3862.62 | 3862.54 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 3857.00 | 3862.26 | 3862.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 3846.85 | 3859.18 | 3861.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 3709.90 | 3708.68 | 3743.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 3709.90 | 3708.68 | 3743.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 3709.90 | 3708.68 | 3743.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 3705.25 | 3708.84 | 3740.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:15:00 | 3702.00 | 3708.84 | 3740.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:15:00 | 3704.75 | 3708.22 | 3734.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 3703.25 | 3707.22 | 3731.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 3713.85 | 3707.82 | 3725.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:45:00 | 3707.65 | 3707.82 | 3725.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 3652.30 | 3693.59 | 3715.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:15:00 | 3625.50 | 3693.59 | 3715.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 3729.15 | 3702.62 | 3714.58 | SL hit (close>static) qty=1.00 sl=3722.50 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 3775.80 | 3729.11 | 3724.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 3795.05 | 3759.92 | 3746.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 3855.90 | 3870.44 | 3831.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 3855.90 | 3870.44 | 3831.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 3860.00 | 3862.78 | 3846.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 3879.95 | 3860.43 | 3852.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 14:15:00 | 3830.95 | 3855.99 | 3854.66 | SL hit (close<static) qty=1.00 sl=3841.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 3837.00 | 3852.19 | 3853.06 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 3877.00 | 3857.15 | 3855.23 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 3839.95 | 3858.62 | 3859.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 12:15:00 | 3836.40 | 3851.80 | 3855.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 3859.00 | 3806.48 | 3809.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 3859.00 | 3806.48 | 3809.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 3859.00 | 3806.48 | 3809.74 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 3839.75 | 3813.13 | 3812.47 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 3804.00 | 3812.60 | 3813.17 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 3839.60 | 3818.00 | 3815.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 3847.30 | 3823.86 | 3818.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 13:15:00 | 3825.65 | 3831.41 | 3823.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 13:15:00 | 3825.65 | 3831.41 | 3823.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 3825.65 | 3831.41 | 3823.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 3825.65 | 3831.41 | 3823.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 3815.10 | 3828.15 | 3823.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 3815.10 | 3828.15 | 3823.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 3815.00 | 3825.52 | 3822.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 3813.55 | 3825.52 | 3822.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 3818.15 | 3820.32 | 3820.32 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 3818.80 | 3820.02 | 3820.18 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 14:15:00 | 3839.65 | 3823.64 | 3821.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 3857.10 | 3832.95 | 3826.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 3847.45 | 3850.35 | 3840.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 3847.45 | 3850.35 | 3840.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 3847.45 | 3850.35 | 3840.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 3830.20 | 3850.35 | 3840.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 3870.30 | 3854.34 | 3843.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 11:15:00 | 3881.55 | 3854.34 | 3843.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 13:15:00 | 3988.30 | 3997.59 | 3998.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 3988.30 | 3997.59 | 3998.30 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 4004.15 | 3998.33 | 3998.32 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 3992.95 | 3997.25 | 3997.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 15:15:00 | 3983.70 | 3992.96 | 3995.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 3932.25 | 3931.54 | 3955.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 3932.25 | 3931.54 | 3955.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 3932.25 | 3931.54 | 3955.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 3950.30 | 3931.54 | 3955.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 3931.30 | 3922.16 | 3940.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 3940.75 | 3922.16 | 3940.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 4038.10 | 3942.12 | 3946.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 4037.70 | 3942.12 | 3946.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 4085.00 | 3970.70 | 3959.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 4147.00 | 4005.96 | 3976.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 4166.00 | 4170.51 | 4128.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 4166.00 | 4170.51 | 4128.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 4269.90 | 4292.80 | 4273.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 4269.90 | 4292.80 | 4273.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 4288.95 | 4292.03 | 4274.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 4293.40 | 4289.21 | 4276.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:45:00 | 4306.00 | 4294.97 | 4279.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 4312.50 | 4299.03 | 4285.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:30:00 | 4315.20 | 4304.24 | 4291.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 4296.65 | 4305.48 | 4296.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 13:45:00 | 4289.35 | 4305.48 | 4296.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 4306.15 | 4305.61 | 4297.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:00:00 | 4329.00 | 4312.77 | 4303.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:45:00 | 4327.00 | 4318.08 | 4307.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 4327.85 | 4318.41 | 4308.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 4332.25 | 4318.41 | 4308.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 4345.55 | 4376.77 | 4367.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 4345.55 | 4376.77 | 4367.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 4350.90 | 4371.59 | 4365.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 4331.65 | 4371.59 | 4365.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 4355.65 | 4367.31 | 4364.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:00:00 | 4355.65 | 4367.31 | 4364.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 4362.40 | 4366.33 | 4364.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 4362.40 | 4366.33 | 4364.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 4364.10 | 4365.88 | 4364.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:15:00 | 4372.00 | 4365.88 | 4364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 4372.00 | 4367.11 | 4365.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 4367.15 | 4367.11 | 4365.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 4393.90 | 4372.46 | 4367.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:15:00 | 4404.95 | 4372.46 | 4367.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:00:00 | 4397.55 | 4383.26 | 4374.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 14:00:00 | 4398.50 | 4386.31 | 4376.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:30:00 | 4407.00 | 4386.29 | 4381.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 4387.00 | 4386.43 | 4381.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 4349.05 | 4386.43 | 4381.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 4331.10 | 4375.36 | 4377.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 4331.10 | 4375.36 | 4377.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 4307.45 | 4355.22 | 4367.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4229.65 | 4196.63 | 4248.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4229.65 | 4196.63 | 4248.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4229.65 | 4196.63 | 4248.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 4229.65 | 4196.63 | 4248.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 4210.00 | 4189.22 | 4218.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:15:00 | 4187.00 | 4189.96 | 4216.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:45:00 | 4191.00 | 4190.39 | 4214.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:30:00 | 4188.10 | 4190.61 | 4212.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 4191.20 | 4194.18 | 4207.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 4192.85 | 4193.91 | 4206.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 4180.95 | 4197.91 | 4206.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 4232.30 | 4195.84 | 4201.40 | SL hit (close>static) qty=1.00 sl=4209.90 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 4242.20 | 4205.11 | 4205.11 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 4194.25 | 4213.00 | 4213.66 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 4229.10 | 4216.22 | 4215.06 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 4192.10 | 4212.00 | 4213.74 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 09:15:00 | 4277.90 | 4220.61 | 4216.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 10:15:00 | 4290.25 | 4234.54 | 4223.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 4528.15 | 4529.23 | 4501.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 11:30:00 | 4529.80 | 4529.23 | 4501.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 4501.90 | 4523.76 | 4501.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 4501.90 | 4523.76 | 4501.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 4509.00 | 4520.81 | 4502.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:45:00 | 4503.00 | 4520.81 | 4502.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 4503.30 | 4517.31 | 4502.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 4503.30 | 4517.31 | 4502.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 4502.00 | 4514.25 | 4502.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 4482.85 | 4514.25 | 4502.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 4481.00 | 4507.60 | 4500.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 4475.20 | 4507.60 | 4500.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 4469.50 | 4499.98 | 4497.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 4469.50 | 4499.98 | 4497.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 4465.05 | 4492.99 | 4494.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 4463.65 | 4487.12 | 4491.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 4540.45 | 4488.30 | 4489.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 4540.45 | 4488.30 | 4489.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 4540.45 | 4488.30 | 4489.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 4535.00 | 4488.30 | 4489.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 4517.00 | 4494.04 | 4492.02 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 4491.00 | 4496.41 | 4496.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 4470.35 | 4491.20 | 4494.12 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 4517.45 | 4496.45 | 4496.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 11:15:00 | 4544.75 | 4506.11 | 4500.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 4507.25 | 4514.28 | 4506.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 14:15:00 | 4507.25 | 4514.28 | 4506.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 4507.25 | 4514.28 | 4506.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:45:00 | 4502.65 | 4514.28 | 4506.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 4513.00 | 4514.02 | 4507.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 4514.65 | 4514.02 | 4507.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 4539.00 | 4519.02 | 4509.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:00:00 | 4551.60 | 4522.16 | 4515.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 4549.10 | 4527.55 | 4518.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 13:00:00 | 4547.05 | 4534.21 | 4523.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 4554.55 | 4538.56 | 4527.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 4537.45 | 4547.01 | 4537.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 4537.45 | 4547.01 | 4537.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 4522.40 | 4542.09 | 4536.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 4522.40 | 4542.09 | 4536.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 4526.35 | 4538.94 | 4535.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 4503.70 | 4538.94 | 4535.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 4521.20 | 4531.41 | 4532.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 4521.20 | 4531.41 | 4532.47 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 4543.95 | 4532.98 | 4532.94 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 4513.75 | 4530.35 | 4531.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 4462.00 | 4513.85 | 4523.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 14:15:00 | 4478.05 | 4471.27 | 4485.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 15:00:00 | 4478.05 | 4471.27 | 4485.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 4490.00 | 4475.01 | 4486.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 4509.70 | 4475.01 | 4486.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 4466.90 | 4473.39 | 4484.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:30:00 | 4481.55 | 4473.39 | 4484.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 4445.10 | 4467.73 | 4480.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 4477.00 | 4467.73 | 4480.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 4471.55 | 4462.83 | 4471.83 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 4500.75 | 4473.44 | 4471.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 4531.20 | 4484.99 | 4477.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 4493.60 | 4510.82 | 4498.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 4493.60 | 4510.82 | 4498.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 4493.60 | 4510.82 | 4498.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 4493.60 | 4510.82 | 4498.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 4475.00 | 4503.65 | 4496.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 4475.00 | 4503.65 | 4496.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 4495.50 | 4502.02 | 4496.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 4479.30 | 4497.20 | 4494.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 4461.35 | 4490.03 | 4491.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 11:15:00 | 4453.35 | 4482.69 | 4488.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 4493.20 | 4479.40 | 4485.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 13:15:00 | 4493.20 | 4479.40 | 4485.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 4493.20 | 4479.40 | 4485.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 4493.20 | 4479.40 | 4485.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 4521.55 | 4487.83 | 4488.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 4521.55 | 4487.83 | 4488.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 4495.30 | 4489.33 | 4489.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 4538.95 | 4502.97 | 4495.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 4516.85 | 4519.45 | 4510.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:00:00 | 4516.85 | 4519.45 | 4510.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 4509.10 | 4517.38 | 4509.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 4509.10 | 4517.38 | 4509.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 4508.35 | 4515.57 | 4509.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 4515.70 | 4515.57 | 4509.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:00:00 | 4516.85 | 4515.83 | 4510.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:45:00 | 4516.70 | 4512.82 | 4510.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 4386.80 | 4486.37 | 4499.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 4386.80 | 4486.37 | 4499.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 4364.90 | 4444.44 | 4476.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 4296.75 | 4290.19 | 4330.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 15:00:00 | 4296.75 | 4290.19 | 4330.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 4269.30 | 4266.43 | 4283.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 4285.55 | 4266.43 | 4283.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 4282.00 | 4269.54 | 4283.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 4257.25 | 4269.54 | 4283.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:00:00 | 4260.05 | 4267.64 | 4280.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 4300.00 | 4273.90 | 4276.46 | SL hit (close>static) qty=1.00 sl=4285.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 4290.90 | 4278.81 | 4278.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 4339.85 | 4295.06 | 4286.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 15:15:00 | 4305.00 | 4307.29 | 4297.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 09:15:00 | 4284.25 | 4307.29 | 4297.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 4283.90 | 4302.62 | 4296.63 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 4272.50 | 4290.80 | 4291.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 4262.35 | 4275.00 | 4282.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 11:15:00 | 4272.80 | 4270.03 | 4278.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 12:00:00 | 4272.80 | 4270.03 | 4278.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 4260.00 | 4268.02 | 4277.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 4253.70 | 4275.14 | 4278.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 4295.95 | 4253.94 | 4257.59 | SL hit (close>static) qty=1.00 sl=4282.40 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 09:15:00 | 4275.90 | 4257.70 | 4257.69 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 4242.00 | 4254.56 | 4256.26 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 14:15:00 | 4273.55 | 4259.37 | 4257.95 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 09:15:00 | 4212.00 | 4252.24 | 4255.10 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 4272.00 | 4253.69 | 4252.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 11:15:00 | 4291.45 | 4270.30 | 4263.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 4232.30 | 4262.70 | 4260.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 4232.30 | 4262.70 | 4260.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 4232.30 | 4262.70 | 4260.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:45:00 | 4229.40 | 4262.70 | 4260.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 4230.30 | 4256.22 | 4257.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 4179.00 | 4232.64 | 4245.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 4104.00 | 4101.44 | 4123.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 14:00:00 | 4104.00 | 4101.44 | 4123.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 4113.40 | 4101.46 | 4117.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:30:00 | 4109.10 | 4101.46 | 4117.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 4120.00 | 4105.17 | 4117.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 4120.00 | 4105.17 | 4117.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 4097.10 | 4103.55 | 4116.06 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 4129.70 | 4119.28 | 4118.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 15:15:00 | 4130.50 | 4121.82 | 4119.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 09:15:00 | 4106.40 | 4118.73 | 4118.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 4106.40 | 4118.73 | 4118.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 4106.40 | 4118.73 | 4118.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 4091.80 | 4118.73 | 4118.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 4107.15 | 4116.42 | 4117.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 4097.70 | 4112.67 | 4115.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 4086.25 | 4058.98 | 4075.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 4086.25 | 4058.98 | 4075.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 4086.25 | 4058.98 | 4075.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 4086.25 | 4058.98 | 4075.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 4095.00 | 4066.18 | 4077.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 4106.00 | 4066.18 | 4077.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 4062.25 | 4069.69 | 4077.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 4051.90 | 4068.45 | 4075.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:30:00 | 4051.50 | 4055.45 | 4065.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:45:00 | 4058.30 | 4060.99 | 4064.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:45:00 | 4059.40 | 4059.45 | 4063.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 4054.25 | 4058.41 | 4062.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:45:00 | 4060.80 | 4058.41 | 4062.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 4056.65 | 4057.24 | 4061.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 4056.65 | 4057.24 | 4061.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 4060.00 | 4057.79 | 4061.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 4072.75 | 4057.79 | 4061.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 4072.55 | 4060.74 | 4062.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 4103.55 | 4069.31 | 4066.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 4103.55 | 4069.31 | 4066.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 4124.05 | 4080.25 | 4071.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 15:15:00 | 4090.05 | 4091.73 | 4080.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 09:15:00 | 4076.70 | 4091.73 | 4080.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 4073.10 | 4088.00 | 4080.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 4073.10 | 4088.00 | 4080.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 4085.00 | 4087.40 | 4080.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:30:00 | 4070.95 | 4087.40 | 4080.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 4087.45 | 4087.41 | 4081.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:45:00 | 4079.80 | 4087.41 | 4081.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 4070.95 | 4084.12 | 4080.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:00:00 | 4070.95 | 4084.12 | 4080.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 4082.00 | 4083.70 | 4080.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:45:00 | 4065.00 | 4083.70 | 4080.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 4073.55 | 4081.67 | 4079.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:45:00 | 4078.30 | 4081.67 | 4079.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 4080.00 | 4081.33 | 4079.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 4087.00 | 4081.33 | 4079.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 15:15:00 | 4071.20 | 4086.67 | 4085.03 | SL hit (close<static) qty=1.00 sl=4071.95 alert=retest2 |

### Cycle 48 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 3993.80 | 4068.10 | 4076.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 3963.80 | 4047.24 | 4066.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 14:15:00 | 3963.70 | 3963.11 | 3992.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 3963.70 | 3963.11 | 3992.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 3968.05 | 3968.47 | 3981.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 3968.05 | 3968.47 | 3981.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 4083.55 | 3991.96 | 3989.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 4103.80 | 4014.33 | 4000.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 4130.35 | 4133.87 | 4103.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:00:00 | 4130.35 | 4133.87 | 4103.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 4199.75 | 4148.84 | 4122.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 4208.95 | 4148.84 | 4122.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 4210.75 | 4193.06 | 4171.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 12:15:00 | 4137.55 | 4166.03 | 4166.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 4137.55 | 4166.03 | 4166.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 09:15:00 | 4124.80 | 4151.67 | 4159.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 14:15:00 | 4147.25 | 4146.98 | 4153.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 14:15:00 | 4147.25 | 4146.98 | 4153.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 4147.25 | 4146.98 | 4153.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 4150.00 | 4146.98 | 4153.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 4040.45 | 4125.84 | 4142.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 4012.85 | 4125.84 | 4142.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:00:00 | 4021.00 | 4073.37 | 4111.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 4021.45 | 4064.67 | 4104.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:00:00 | 4019.25 | 4055.59 | 4096.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 4095.35 | 4057.39 | 4089.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 4095.35 | 4057.39 | 4089.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 4114.00 | 4068.71 | 4092.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 4114.00 | 4068.71 | 4092.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 4115.00 | 4077.97 | 4094.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 4129.00 | 4077.97 | 4094.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 4090.05 | 4083.45 | 4094.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 4090.05 | 4083.45 | 4094.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 4030.90 | 4072.94 | 4088.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 13:15:00 | 4028.75 | 4057.22 | 4074.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 4131.85 | 4075.99 | 4077.61 | SL hit (close>static) qty=1.00 sl=4096.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 4145.00 | 4089.79 | 4083.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 4216.10 | 4133.16 | 4106.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 4340.60 | 4343.79 | 4303.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:45:00 | 4350.05 | 4343.79 | 4303.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 4313.80 | 4334.86 | 4309.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 4309.15 | 4334.86 | 4309.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4277.45 | 4323.38 | 4306.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 4274.85 | 4323.38 | 4306.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 4250.55 | 4308.81 | 4301.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 4250.55 | 4308.81 | 4301.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 4240.00 | 4295.05 | 4295.85 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 4292.60 | 4277.26 | 4275.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 4314.55 | 4286.21 | 4280.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 4342.00 | 4346.75 | 4322.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:15:00 | 4399.15 | 4346.75 | 4322.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 4441.00 | 4439.26 | 4418.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:00:00 | 4441.00 | 4439.26 | 4418.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 4482.45 | 4451.70 | 4431.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 4431.30 | 4446.74 | 4434.04 | SL hit (close<ema400) qty=1.00 sl=4434.04 alert=retest1 |

### Cycle 54 — SELL (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 12:15:00 | 4418.20 | 4430.10 | 4430.52 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 4477.85 | 4437.29 | 4433.37 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 11:15:00 | 4408.25 | 4447.60 | 4449.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 13:15:00 | 4407.65 | 4435.18 | 4442.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 13:15:00 | 4344.95 | 4338.01 | 4366.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 14:00:00 | 4344.95 | 4338.01 | 4366.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 4330.00 | 4340.32 | 4360.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:45:00 | 4340.85 | 4340.32 | 4360.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 4274.25 | 4295.52 | 4324.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 4236.10 | 4289.72 | 4319.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 4148.65 | 4121.41 | 4121.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 4148.65 | 4121.41 | 4121.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 4161.85 | 4130.82 | 4125.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 4118.10 | 4142.87 | 4134.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 4118.10 | 4142.87 | 4134.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 4118.10 | 4142.87 | 4134.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 4118.10 | 4142.87 | 4134.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 4109.80 | 4136.25 | 4132.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 4106.00 | 4136.25 | 4132.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 4118.10 | 4130.43 | 4130.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:30:00 | 4111.05 | 4130.43 | 4130.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 4108.00 | 4125.94 | 4128.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 4099.80 | 4120.72 | 4125.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 4134.05 | 4119.59 | 4124.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 4134.05 | 4119.59 | 4124.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 4134.05 | 4119.59 | 4124.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 4119.95 | 4119.59 | 4124.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 4096.85 | 4115.04 | 4121.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:30:00 | 4078.35 | 4109.83 | 4118.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 15:15:00 | 4090.00 | 4106.42 | 4114.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 09:30:00 | 4086.30 | 4091.31 | 4106.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 4123.65 | 4079.28 | 4074.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 4123.65 | 4079.28 | 4074.51 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 4040.90 | 4071.52 | 4072.20 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 09:15:00 | 4199.00 | 4086.40 | 4077.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 10:15:00 | 4235.40 | 4116.20 | 4091.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 10:15:00 | 4274.00 | 4281.41 | 4236.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:45:00 | 4278.00 | 4281.41 | 4236.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 4235.75 | 4269.26 | 4238.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 4235.75 | 4269.26 | 4238.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 4221.10 | 4259.63 | 4237.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 4221.10 | 4259.63 | 4237.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 4231.65 | 4254.03 | 4236.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 4251.95 | 4236.94 | 4233.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 4204.10 | 4229.74 | 4231.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 11:15:00 | 4204.10 | 4229.74 | 4231.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 12:15:00 | 4199.30 | 4223.65 | 4228.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 4098.60 | 4073.51 | 4098.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 4098.60 | 4073.51 | 4098.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 4098.60 | 4073.51 | 4098.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 4089.90 | 4073.51 | 4098.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 4128.05 | 4084.42 | 4101.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:45:00 | 4124.05 | 4084.42 | 4101.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 4129.95 | 4093.52 | 4103.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:45:00 | 4129.25 | 4093.52 | 4103.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 13:15:00 | 4144.70 | 4112.48 | 4111.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 14:15:00 | 4157.35 | 4121.46 | 4115.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 14:15:00 | 4142.65 | 4148.56 | 4135.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-23 14:30:00 | 4146.00 | 4148.56 | 4135.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 4161.00 | 4150.96 | 4139.11 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 4069.45 | 4129.33 | 4135.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 4051.20 | 4102.18 | 4121.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 4095.85 | 4064.62 | 4081.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 4095.85 | 4064.62 | 4081.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 4095.85 | 4064.62 | 4081.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 4104.35 | 4064.62 | 4081.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 4090.65 | 4069.83 | 4082.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 4079.75 | 4069.83 | 4082.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 4102.00 | 4083.31 | 4085.05 | SL hit (close>static) qty=1.00 sl=4101.55 alert=retest2 |

### Cycle 65 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 4105.00 | 4087.65 | 4086.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 4117.30 | 4093.58 | 4089.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 4091.65 | 4096.62 | 4091.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 11:15:00 | 4091.65 | 4096.62 | 4091.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 4091.65 | 4096.62 | 4091.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 4091.60 | 4096.62 | 4091.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 4108.80 | 4099.06 | 4093.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 4116.85 | 4096.64 | 4093.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:45:00 | 4114.40 | 4104.66 | 4099.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 4078.60 | 4099.45 | 4097.99 | SL hit (close<static) qty=1.00 sl=4090.30 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 4064.65 | 4092.49 | 4094.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 4022.95 | 4070.91 | 4083.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 4063.25 | 4059.99 | 4074.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 13:00:00 | 4063.25 | 4059.99 | 4074.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 4071.95 | 4062.37 | 4073.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:30:00 | 4065.50 | 4062.37 | 4073.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 4090.40 | 4068.24 | 4073.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 4090.40 | 4068.24 | 4073.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 4087.30 | 4072.05 | 4075.06 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 4093.90 | 4078.40 | 4077.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 4110.50 | 4087.43 | 4081.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 10:15:00 | 4088.35 | 4091.73 | 4085.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 11:00:00 | 4088.35 | 4091.73 | 4085.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 4083.95 | 4090.18 | 4085.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 4083.95 | 4090.18 | 4085.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 4087.75 | 4089.69 | 4085.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:30:00 | 4081.65 | 4089.69 | 4085.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 4078.85 | 4087.52 | 4085.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:30:00 | 4077.95 | 4087.52 | 4085.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 4089.45 | 4087.91 | 4085.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 4102.10 | 4087.53 | 4085.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 11:15:00 | 4068.80 | 4083.02 | 4083.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 4068.80 | 4083.02 | 4083.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 4062.45 | 4078.90 | 4081.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 4084.50 | 4076.99 | 4080.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 4084.50 | 4076.99 | 4080.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 4084.50 | 4076.99 | 4080.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 4084.50 | 4076.99 | 4080.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 4087.00 | 4078.99 | 4080.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 4032.55 | 4078.99 | 4080.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 09:15:00 | 3830.92 | 3875.07 | 3898.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 15:15:00 | 3788.00 | 3785.43 | 3814.74 | SL hit (close>ema200) qty=0.50 sl=3785.43 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 3578.10 | 3541.38 | 3539.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 14:15:00 | 3603.45 | 3569.66 | 3558.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 3599.70 | 3605.22 | 3593.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 3599.70 | 3605.22 | 3593.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 3586.80 | 3601.54 | 3592.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 3593.65 | 3601.54 | 3592.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 3589.00 | 3599.03 | 3592.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 3582.75 | 3599.03 | 3592.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 3563.30 | 3584.47 | 3586.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 3500.85 | 3561.71 | 3574.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 13:15:00 | 3513.65 | 3509.28 | 3528.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 14:00:00 | 3513.65 | 3509.28 | 3528.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 3503.15 | 3507.87 | 3522.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 3487.20 | 3503.74 | 3519.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 3486.35 | 3501.56 | 3517.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:30:00 | 3485.55 | 3498.02 | 3514.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 3545.35 | 3507.30 | 3513.23 | SL hit (close>static) qty=1.00 sl=3530.45 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 3549.00 | 3519.45 | 3517.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 3553.90 | 3533.26 | 3525.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 3477.50 | 3524.09 | 3522.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 09:15:00 | 3477.50 | 3524.09 | 3522.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 3477.50 | 3524.09 | 3522.63 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 10:15:00 | 3476.65 | 3514.60 | 3518.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 11:15:00 | 3464.70 | 3504.62 | 3513.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 14:15:00 | 3496.30 | 3496.11 | 3506.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 14:30:00 | 3501.15 | 3496.11 | 3506.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 3534.95 | 3505.30 | 3509.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:45:00 | 3560.50 | 3505.30 | 3509.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 10:15:00 | 3566.70 | 3517.58 | 3514.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 3570.90 | 3551.92 | 3535.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 15:15:00 | 3641.00 | 3651.74 | 3625.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:15:00 | 3662.45 | 3653.73 | 3629.24 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 12:00:00 | 3663.20 | 3657.45 | 3635.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 14:00:00 | 3663.95 | 3659.47 | 3640.10 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 3638.45 | 3655.27 | 3639.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 3638.45 | 3655.27 | 3639.95 | SL hit (close<ema400) qty=1.00 sl=3639.95 alert=retest1 |

### Cycle 74 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 3629.00 | 3638.49 | 3639.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 3552.75 | 3604.50 | 3620.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 3554.05 | 3553.92 | 3575.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:15:00 | 3459.65 | 3549.54 | 3567.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 3113.68 | 3296.69 | 3381.01 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 75 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 3274.00 | 3258.34 | 3258.30 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 09:15:00 | 3230.70 | 3256.96 | 3258.06 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 12:15:00 | 3276.40 | 3258.69 | 3258.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 3296.00 | 3266.15 | 3261.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 3310.50 | 3315.13 | 3298.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 3310.50 | 3315.13 | 3298.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 3310.50 | 3315.13 | 3298.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:15:00 | 3331.80 | 3316.23 | 3300.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 12:30:00 | 3327.50 | 3317.50 | 3303.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 3384.40 | 3314.41 | 3305.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 13:15:00 | 3438.10 | 3449.06 | 3450.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 13:15:00 | 3438.10 | 3449.06 | 3450.54 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 3484.80 | 3454.32 | 3452.44 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 3429.40 | 3451.18 | 3453.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 3414.20 | 3439.36 | 3445.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 3433.10 | 3432.49 | 3439.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 3433.10 | 3432.49 | 3439.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 3440.30 | 3434.05 | 3439.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 3440.30 | 3434.05 | 3439.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 3439.90 | 3435.22 | 3439.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 3510.60 | 3435.22 | 3439.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3536.60 | 3455.50 | 3448.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 3602.10 | 3509.33 | 3476.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 3562.10 | 3567.24 | 3527.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:00:00 | 3562.10 | 3567.24 | 3527.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 3515.00 | 3551.28 | 3526.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 3515.00 | 3551.28 | 3526.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 3514.30 | 3543.89 | 3525.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 3530.30 | 3538.71 | 3524.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 3549.00 | 3537.22 | 3533.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 11:15:00 | 3519.10 | 3544.59 | 3547.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 3519.10 | 3544.59 | 3547.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 3515.00 | 3525.80 | 3534.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 3530.30 | 3517.52 | 3525.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 3530.30 | 3517.52 | 3525.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 3530.30 | 3517.52 | 3525.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 3530.30 | 3517.52 | 3525.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 3519.90 | 3518.00 | 3525.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 3518.00 | 3518.00 | 3525.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 3479.70 | 3522.14 | 3525.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 3516.00 | 3493.46 | 3503.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 3517.80 | 3500.68 | 3505.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 3515.70 | 3503.68 | 3506.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 3515.70 | 3503.68 | 3506.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 3517.40 | 3506.43 | 3507.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:00:00 | 3517.40 | 3506.43 | 3507.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 3514.10 | 3507.96 | 3507.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 3514.10 | 3507.96 | 3507.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 3542.90 | 3514.96 | 3511.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 3498.00 | 3524.34 | 3519.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 3498.00 | 3524.34 | 3519.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 3498.00 | 3524.34 | 3519.96 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 3498.30 | 3516.21 | 3517.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 3490.00 | 3499.18 | 3503.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 3400.00 | 3392.01 | 3408.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 12:00:00 | 3400.00 | 3392.01 | 3408.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 3388.00 | 3380.58 | 3388.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 3420.00 | 3380.58 | 3388.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 3423.20 | 3389.11 | 3391.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 3423.20 | 3389.11 | 3391.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 3423.60 | 3396.01 | 3394.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 3428.50 | 3406.31 | 3399.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 3454.00 | 3465.60 | 3452.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 3454.00 | 3465.60 | 3452.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 3454.00 | 3465.60 | 3452.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 3454.00 | 3465.60 | 3452.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 3452.50 | 3462.98 | 3452.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 3459.10 | 3462.98 | 3452.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:15:00 | 3461.90 | 3461.96 | 3453.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 3428.60 | 3455.01 | 3451.47 | SL hit (close<static) qty=1.00 sl=3447.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 3432.00 | 3446.96 | 3448.22 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 3472.10 | 3449.57 | 3447.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 3504.00 | 3460.46 | 3452.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 3494.00 | 3504.49 | 3490.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 3494.00 | 3504.49 | 3490.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 3494.00 | 3504.49 | 3490.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 3494.00 | 3504.49 | 3490.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 3489.70 | 3501.53 | 3490.64 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 3453.40 | 3484.93 | 3485.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 3432.90 | 3465.30 | 3475.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 3438.80 | 3432.26 | 3447.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 3438.80 | 3432.26 | 3447.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 3440.70 | 3435.26 | 3446.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 3435.70 | 3435.26 | 3446.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3433.00 | 3406.91 | 3419.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 3413.70 | 3414.19 | 3420.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:00:00 | 3412.00 | 3414.19 | 3420.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 14:15:00 | 3443.50 | 3420.87 | 3418.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 3443.50 | 3420.87 | 3418.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 3445.10 | 3435.36 | 3427.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 3444.00 | 3444.50 | 3435.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 3444.00 | 3444.50 | 3435.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 3438.30 | 3443.26 | 3436.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 3433.90 | 3443.26 | 3436.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 3436.40 | 3441.89 | 3436.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 3436.40 | 3441.89 | 3436.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 3445.00 | 3442.51 | 3436.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 3451.10 | 3442.51 | 3436.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 3440.30 | 3442.07 | 3437.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 3463.50 | 3452.58 | 3444.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 3473.40 | 3454.06 | 3446.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 3420.00 | 3440.90 | 3442.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 3420.00 | 3440.90 | 3442.18 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 3469.80 | 3443.72 | 3442.87 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 3440.10 | 3442.08 | 3442.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 3435.00 | 3440.66 | 3441.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 3419.80 | 3414.43 | 3423.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 3419.80 | 3414.43 | 3423.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 3419.10 | 3415.37 | 3422.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 3404.30 | 3413.66 | 3417.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 3409.10 | 3410.60 | 3415.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 3404.10 | 3410.96 | 3414.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 3234.09 | 3286.81 | 3327.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 3238.64 | 3286.81 | 3327.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 3233.89 | 3286.81 | 3327.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 3257.90 | 3240.52 | 3274.49 | SL hit (close>ema200) qty=0.50 sl=3240.52 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 3074.10 | 3037.97 | 3036.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 3078.70 | 3046.11 | 3040.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 3042.10 | 3053.21 | 3048.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 3042.10 | 3053.21 | 3048.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 3042.10 | 3053.21 | 3048.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 3042.10 | 3053.21 | 3048.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 3030.90 | 3048.75 | 3046.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 3030.90 | 3048.75 | 3046.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 3031.80 | 3045.36 | 3045.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 3014.00 | 3032.41 | 3038.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 3033.00 | 3030.11 | 3035.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 3033.00 | 3030.11 | 3035.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 3048.30 | 3033.75 | 3036.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 3048.30 | 3033.75 | 3036.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3049.00 | 3036.80 | 3037.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 3037.90 | 3036.80 | 3037.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 3042.00 | 3038.80 | 3038.75 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 3034.70 | 3039.08 | 3039.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 3031.90 | 3037.65 | 3038.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 3042.70 | 3037.59 | 3038.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 3042.70 | 3037.59 | 3038.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 3042.70 | 3037.59 | 3038.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 3042.70 | 3037.59 | 3038.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 3042.60 | 3038.60 | 3038.62 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 3040.50 | 3038.98 | 3038.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3075.00 | 3047.20 | 3042.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 3043.30 | 3048.06 | 3044.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 3043.30 | 3048.06 | 3044.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 3043.30 | 3048.06 | 3044.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 3043.30 | 3048.06 | 3044.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 3040.00 | 3046.45 | 3044.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 3040.00 | 3046.45 | 3044.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 3039.80 | 3045.12 | 3043.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 3029.40 | 3045.12 | 3043.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 3034.00 | 3042.40 | 3042.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 3029.10 | 3037.72 | 3039.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 3020.80 | 3016.85 | 3023.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 3020.80 | 3016.85 | 3023.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 3020.80 | 3016.85 | 3023.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 3020.80 | 3016.85 | 3023.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 3018.70 | 3017.82 | 3022.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 3015.90 | 3017.48 | 3021.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 3016.40 | 3017.16 | 3021.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 3036.20 | 3021.10 | 3022.42 | SL hit (close>static) qty=1.00 sl=3022.90 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 3059.40 | 3028.76 | 3025.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 3085.70 | 3040.15 | 3031.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 3065.90 | 3092.84 | 3077.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 3065.90 | 3092.84 | 3077.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 3065.90 | 3092.84 | 3077.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 3065.90 | 3092.84 | 3077.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 3065.70 | 3087.41 | 3076.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 3065.50 | 3087.41 | 3076.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3052.10 | 3072.46 | 3071.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 3052.10 | 3072.46 | 3071.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 3051.00 | 3068.17 | 3070.04 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 3126.00 | 3079.73 | 3075.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 3147.40 | 3093.27 | 3081.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 3104.20 | 3138.10 | 3125.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 3104.20 | 3138.10 | 3125.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 3104.20 | 3138.10 | 3125.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 3104.20 | 3138.10 | 3125.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 3126.00 | 3135.68 | 3125.20 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 3092.70 | 3118.44 | 3119.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 3086.20 | 3098.49 | 3107.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 3115.60 | 3099.56 | 3105.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 3115.60 | 3099.56 | 3105.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3115.60 | 3099.56 | 3105.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 3123.60 | 3099.56 | 3105.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 3125.90 | 3104.83 | 3107.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 3125.90 | 3104.83 | 3107.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 3117.30 | 3110.61 | 3110.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 3131.30 | 3115.50 | 3112.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3110.70 | 3123.81 | 3118.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3110.70 | 3123.81 | 3118.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3110.70 | 3123.81 | 3118.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3110.70 | 3123.81 | 3118.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3111.60 | 3121.37 | 3117.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 3112.00 | 3121.37 | 3117.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 3112.00 | 3119.50 | 3117.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 3129.90 | 3119.50 | 3117.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 3091.20 | 3112.32 | 3114.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 3091.20 | 3112.32 | 3114.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 3049.90 | 3090.03 | 3099.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 3041.20 | 3032.60 | 3049.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 3041.20 | 3032.60 | 3049.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3041.20 | 3032.60 | 3049.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 3056.60 | 3032.60 | 3049.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 3050.00 | 3036.08 | 3049.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 3050.00 | 3036.08 | 3049.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 3050.60 | 3038.99 | 3049.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 3052.40 | 3038.99 | 3049.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 3049.50 | 3041.09 | 3049.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 3050.10 | 3041.09 | 3049.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 3047.90 | 3042.45 | 3049.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 3047.90 | 3042.45 | 3049.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 3049.90 | 3043.94 | 3049.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 3048.80 | 3043.94 | 3049.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 3052.00 | 3045.55 | 3049.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 3103.90 | 3045.55 | 3049.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 3106.50 | 3057.74 | 3054.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 3115.80 | 3069.35 | 3060.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 3106.00 | 3125.04 | 3114.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 3106.00 | 3125.04 | 3114.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 3106.00 | 3125.04 | 3114.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 3106.00 | 3125.04 | 3114.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 3099.70 | 3119.97 | 3113.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 3102.10 | 3119.97 | 3113.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 3112.40 | 3113.18 | 3111.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 3114.50 | 3113.18 | 3111.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3125.10 | 3115.57 | 3113.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 3132.50 | 3120.62 | 3115.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 3130.00 | 3124.48 | 3118.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 3100.10 | 3152.88 | 3156.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 3100.10 | 3152.88 | 3156.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 3057.50 | 3088.73 | 3116.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 2907.70 | 2907.60 | 2932.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 11:00:00 | 2907.70 | 2907.60 | 2932.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 2910.10 | 2898.91 | 2909.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:45:00 | 2916.50 | 2898.91 | 2909.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 2915.10 | 2902.15 | 2909.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 2907.50 | 2902.15 | 2909.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2903.00 | 2902.32 | 2908.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 2897.00 | 2900.76 | 2907.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 2895.00 | 2900.74 | 2907.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 2946.60 | 2909.53 | 2908.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 2946.60 | 2909.53 | 2908.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 2953.50 | 2918.32 | 2912.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 2961.00 | 2961.20 | 2943.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:15:00 | 2958.60 | 2961.20 | 2943.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3045.00 | 3041.94 | 3019.11 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 3003.40 | 3016.17 | 3017.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 2998.40 | 3009.68 | 3014.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 11:15:00 | 2970.80 | 2968.15 | 2984.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 12:00:00 | 2970.80 | 2968.15 | 2984.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 2969.20 | 2964.73 | 2972.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 2969.20 | 2964.73 | 2972.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 2971.80 | 2966.15 | 2972.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 2972.30 | 2966.15 | 2972.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2974.30 | 2967.78 | 2972.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:15:00 | 2977.80 | 2967.78 | 2972.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2982.80 | 2970.78 | 2973.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 2982.80 | 2970.78 | 2973.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 2972.20 | 2971.07 | 2973.08 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3000.30 | 2972.90 | 2972.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 3009.00 | 2984.20 | 2977.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 3000.00 | 3005.00 | 2993.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:30:00 | 3000.00 | 3005.00 | 2993.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3070.00 | 3079.00 | 3068.80 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 3044.60 | 3062.21 | 3063.15 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 3070.20 | 3062.94 | 3062.77 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 3060.10 | 3062.37 | 3062.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 3056.70 | 3061.23 | 3062.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 3059.90 | 3045.82 | 3051.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 3059.90 | 3045.82 | 3051.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 3059.90 | 3045.82 | 3051.07 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 3056.00 | 3053.67 | 3053.50 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 3037.50 | 3051.45 | 3052.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 3036.00 | 3048.36 | 3051.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 3005.30 | 2997.92 | 3012.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 3005.30 | 2997.92 | 3012.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 3005.30 | 2997.92 | 3012.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 2972.90 | 3008.09 | 3012.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 3019.70 | 2994.65 | 2999.34 | SL hit (close>static) qty=1.00 sl=3017.50 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 3044.30 | 3004.58 | 3003.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 3049.00 | 3035.73 | 3024.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 3108.20 | 3112.82 | 3089.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 3108.20 | 3112.82 | 3089.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 3085.90 | 3105.41 | 3091.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 3098.00 | 3105.38 | 3092.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 3100.30 | 3104.51 | 3093.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 3080.10 | 3099.63 | 3092.20 | SL hit (close<static) qty=1.00 sl=3080.50 alert=retest2 |

### Cycle 116 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 3092.90 | 3095.73 | 3095.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 3080.10 | 3090.49 | 3093.40 | Break + close below crossover candle low |

### Cycle 117 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 3130.90 | 3098.58 | 3096.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 10:15:00 | 3137.50 | 3106.36 | 3100.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 3145.70 | 3147.34 | 3135.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 3145.70 | 3147.34 | 3135.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3145.70 | 3147.34 | 3135.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 3145.70 | 3147.34 | 3135.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 3151.30 | 3152.28 | 3145.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 3154.50 | 3152.28 | 3145.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 3150.30 | 3151.88 | 3145.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 3147.50 | 3151.88 | 3145.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 3139.60 | 3154.40 | 3148.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 3139.60 | 3154.40 | 3148.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 3136.00 | 3150.72 | 3147.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 3141.70 | 3150.72 | 3147.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 3122.30 | 3142.49 | 3144.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 3120.50 | 3133.46 | 3139.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 3140.30 | 3132.13 | 3137.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 3140.30 | 3132.13 | 3137.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 3140.30 | 3132.13 | 3137.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 3140.30 | 3132.13 | 3137.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 3153.20 | 3136.35 | 3138.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 3153.20 | 3136.35 | 3138.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 3168.50 | 3142.78 | 3141.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 3172.70 | 3148.76 | 3144.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 3157.50 | 3158.07 | 3151.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 3157.50 | 3158.07 | 3151.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 3145.70 | 3155.59 | 3150.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 3145.70 | 3155.59 | 3150.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 3130.00 | 3150.47 | 3149.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 3130.00 | 3150.47 | 3149.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 3127.70 | 3145.92 | 3147.12 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 3165.00 | 3141.28 | 3139.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3151.11 | 3143.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 3231.00 | 3236.35 | 3214.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 3231.00 | 3236.35 | 3214.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3211.50 | 3237.77 | 3227.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 3211.50 | 3237.77 | 3227.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 3237.50 | 3237.71 | 3227.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 3222.40 | 3237.71 | 3227.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3184.20 | 3227.04 | 3224.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 3184.20 | 3227.04 | 3224.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 3188.00 | 3219.23 | 3221.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 3181.00 | 3190.23 | 3200.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 3187.30 | 3185.11 | 3196.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:00:00 | 3187.30 | 3185.11 | 3196.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 3185.00 | 3185.09 | 3195.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 3187.10 | 3185.09 | 3195.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 3191.20 | 3185.82 | 3193.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 3191.20 | 3185.82 | 3193.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 3193.90 | 3187.43 | 3193.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 3196.60 | 3187.43 | 3193.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 3190.00 | 3187.95 | 3193.33 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 3214.00 | 3198.08 | 3196.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 3220.10 | 3205.17 | 3200.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3210.00 | 3218.22 | 3212.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 3210.00 | 3218.22 | 3212.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 3210.00 | 3218.22 | 3212.21 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 3200.40 | 3207.63 | 3208.46 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 3224.00 | 3210.29 | 3209.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 3240.00 | 3220.10 | 3215.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 14:15:00 | 3280.70 | 3284.37 | 3263.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 3280.70 | 3284.37 | 3263.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3313.30 | 3308.03 | 3298.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 3318.70 | 3312.59 | 3304.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 3289.70 | 3306.05 | 3303.56 | SL hit (close<static) qty=1.00 sl=3296.10 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 3283.00 | 3301.44 | 3301.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 15:15:00 | 3276.90 | 3289.09 | 3295.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 3262.40 | 3256.95 | 3268.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 12:45:00 | 3255.10 | 3256.95 | 3268.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 3252.50 | 3256.06 | 3267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:30:00 | 3265.10 | 3256.06 | 3267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 3225.80 | 3223.88 | 3233.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:30:00 | 3231.50 | 3223.88 | 3233.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 3234.70 | 3226.22 | 3232.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 3235.80 | 3226.22 | 3232.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 3239.60 | 3228.90 | 3233.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 3236.60 | 3228.90 | 3233.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 3240.90 | 3231.30 | 3233.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 3242.10 | 3231.30 | 3233.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 3244.40 | 3233.92 | 3234.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 3244.40 | 3233.92 | 3234.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 3243.80 | 3235.90 | 3235.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 3250.40 | 3238.80 | 3236.99 | Break + close above crossover candle high |

### Cycle 128 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 3211.10 | 3234.36 | 3235.36 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 3239.90 | 3231.88 | 3231.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 3255.50 | 3237.94 | 3234.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 3216.10 | 3267.31 | 3257.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 3216.10 | 3267.31 | 3257.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 3216.10 | 3267.31 | 3257.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 3216.10 | 3267.31 | 3257.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 3196.50 | 3253.15 | 3252.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 3197.50 | 3253.15 | 3252.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 3221.20 | 3246.76 | 3249.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 3188.00 | 3208.35 | 3220.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 3213.20 | 3205.91 | 3215.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 3213.20 | 3205.91 | 3215.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 3213.20 | 3205.91 | 3215.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 3221.60 | 3205.91 | 3215.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 3211.40 | 3207.00 | 3215.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 3214.40 | 3207.00 | 3215.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 3227.20 | 3211.04 | 3216.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 3232.80 | 3211.04 | 3216.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3243.00 | 3217.43 | 3218.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 3212.00 | 3217.43 | 3218.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 3240.00 | 3221.95 | 3220.90 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 3205.40 | 3226.56 | 3229.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 3189.30 | 3219.11 | 3225.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 12:15:00 | 3209.90 | 3208.19 | 3215.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 12:30:00 | 3212.70 | 3208.19 | 3215.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 3209.00 | 3207.52 | 3213.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 3191.80 | 3207.52 | 3213.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 3185.00 | 3203.02 | 3210.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:00:00 | 3169.30 | 3196.27 | 3207.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 3180.80 | 3148.14 | 3144.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 3180.80 | 3148.14 | 3144.85 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 3140.90 | 3151.77 | 3152.49 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 3167.00 | 3154.89 | 3153.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 3183.90 | 3160.70 | 3156.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3135.50 | 3176.14 | 3169.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3135.50 | 3176.14 | 3169.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3135.50 | 3176.14 | 3169.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 3135.50 | 3176.14 | 3169.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3130.90 | 3167.10 | 3166.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 3132.30 | 3167.10 | 3166.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 3132.20 | 3160.12 | 3163.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 3113.40 | 3142.16 | 3152.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 3153.00 | 3131.80 | 3138.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 3153.00 | 3131.80 | 3138.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3153.00 | 3131.80 | 3138.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 3153.00 | 3131.80 | 3138.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3187.00 | 3142.84 | 3142.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 3187.00 | 3142.84 | 3142.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 3200.00 | 3154.27 | 3148.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 3235.00 | 3192.55 | 3174.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 3048.70 | 3178.39 | 3177.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3048.70 | 3178.39 | 3177.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3048.70 | 3178.39 | 3177.10 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 3032.00 | 3149.11 | 3163.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 11:15:00 | 3008.90 | 3121.07 | 3149.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 2953.40 | 2949.54 | 2986.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 2953.40 | 2949.54 | 2986.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 2980.60 | 2954.63 | 2972.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 2980.60 | 2954.63 | 2972.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2993.70 | 2962.44 | 2974.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 2993.20 | 2962.44 | 2974.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 2984.80 | 2979.42 | 2979.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 2976.70 | 2979.42 | 2979.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2958.30 | 2975.20 | 2977.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 2948.00 | 2967.03 | 2973.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 2800.60 | 2908.76 | 2940.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 2653.20 | 2758.55 | 2838.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 2651.80 | 2639.17 | 2639.03 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 2618.00 | 2637.12 | 2639.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 09:15:00 | 2559.20 | 2588.78 | 2605.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 2571.60 | 2570.91 | 2588.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 2571.60 | 2570.91 | 2588.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2571.60 | 2570.91 | 2588.59 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2477.20 | 2422.49 | 2417.57 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2381.00 | 2421.40 | 2422.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 2370.50 | 2404.90 | 2414.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 2387.40 | 2379.78 | 2391.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 2387.40 | 2379.78 | 2391.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 2387.40 | 2379.78 | 2391.08 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 2424.30 | 2395.24 | 2393.17 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 2379.80 | 2396.63 | 2398.20 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 2413.70 | 2401.86 | 2400.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 12:15:00 | 2420.20 | 2407.71 | 2403.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 2398.40 | 2405.85 | 2403.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 13:15:00 | 2398.40 | 2405.85 | 2403.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2398.40 | 2405.85 | 2403.01 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 2387.50 | 2399.31 | 2400.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2379.80 | 2395.41 | 2398.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 2393.40 | 2390.28 | 2395.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 11:15:00 | 2393.40 | 2390.28 | 2395.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 2393.40 | 2390.28 | 2395.33 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2446.50 | 2403.01 | 2397.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 2480.00 | 2450.58 | 2432.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 2511.00 | 2564.49 | 2547.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 2511.00 | 2564.49 | 2547.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2511.00 | 2564.49 | 2547.75 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 12:15:00 | 2512.60 | 2538.27 | 2538.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 2480.60 | 2520.77 | 2529.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 2549.30 | 2499.98 | 2510.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 2549.30 | 2499.98 | 2510.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 2549.30 | 2499.98 | 2510.57 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 2545.50 | 2521.14 | 2518.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 2554.50 | 2532.00 | 2524.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 2574.90 | 2577.46 | 2565.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 2574.90 | 2577.46 | 2565.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 2574.90 | 2577.46 | 2565.56 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 2517.60 | 2571.12 | 2576.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 2460.20 | 2519.82 | 2540.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 2439.90 | 2433.65 | 2473.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 2479.10 | 2450.62 | 2465.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 2479.10 | 2450.62 | 2465.18 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 2481.10 | 2462.80 | 2462.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 2485.30 | 2467.30 | 2464.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 2465.00 | 2468.19 | 2465.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 15:15:00 | 2465.00 | 2468.19 | 2465.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 2465.00 | 2468.19 | 2465.85 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 2457.10 | 2463.92 | 2464.39 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 12:15:00 | 2468.70 | 2464.87 | 2464.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 2480.30 | 2467.96 | 2466.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 09:15:00 | 2449.00 | 2465.91 | 2465.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 2449.00 | 2465.91 | 2465.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2449.00 | 2465.91 | 2465.85 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 10:15:00 | 2445.60 | 2461.85 | 2464.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 11:15:00 | 2435.70 | 2456.62 | 2461.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 09:15:00 | 2445.90 | 2442.38 | 2451.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 2445.90 | 2442.38 | 2451.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 2445.90 | 2442.38 | 2451.26 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 14:15:00 | 3842.75 | 2024-05-23 11:15:00 | 3870.80 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-05-18 09:45:00 | 3847.00 | 2024-05-23 11:15:00 | 3870.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-05-21 09:15:00 | 3820.90 | 2024-05-23 11:15:00 | 3870.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-05-23 10:00:00 | 3846.00 | 2024-05-23 11:15:00 | 3870.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-05-24 12:45:00 | 3863.90 | 2024-05-24 13:15:00 | 3849.05 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-05-27 09:30:00 | 3868.30 | 2024-05-27 14:15:00 | 3845.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-06-03 10:45:00 | 3705.25 | 2024-06-04 14:15:00 | 3729.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-06-03 11:15:00 | 3702.00 | 2024-06-05 09:15:00 | 3752.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-06-03 13:15:00 | 3704.75 | 2024-06-05 09:15:00 | 3752.80 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-06-03 14:00:00 | 3703.25 | 2024-06-05 09:15:00 | 3752.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-06-04 12:15:00 | 3625.50 | 2024-06-05 09:15:00 | 3752.80 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-06-12 09:15:00 | 3879.95 | 2024-06-12 14:15:00 | 3830.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-06-27 11:15:00 | 3881.55 | 2024-07-08 13:15:00 | 3988.30 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2024-07-23 10:15:00 | 4293.40 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-07-23 10:45:00 | 4306.00 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-07-23 13:30:00 | 4312.50 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2024-07-24 09:30:00 | 4315.20 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-07-25 12:00:00 | 4329.00 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-07-25 13:45:00 | 4327.00 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-07-25 14:45:00 | 4327.85 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-07-25 15:15:00 | 4332.25 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-07-31 10:15:00 | 4404.95 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-07-31 13:00:00 | 4397.55 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-31 14:00:00 | 4398.50 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-08-01 14:30:00 | 4407.00 | 2024-08-02 09:15:00 | 4331.10 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-08-07 11:15:00 | 4187.00 | 2024-08-09 09:15:00 | 4232.30 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-08-07 11:45:00 | 4191.00 | 2024-08-09 10:15:00 | 4242.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-07 12:30:00 | 4188.10 | 2024-08-09 10:15:00 | 4242.20 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-08-08 10:00:00 | 4191.20 | 2024-08-09 10:15:00 | 4242.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-08 13:15:00 | 4180.95 | 2024-08-09 10:15:00 | 4242.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-08-30 10:00:00 | 4551.60 | 2024-09-03 10:15:00 | 4521.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-30 11:00:00 | 4549.10 | 2024-09-03 10:15:00 | 4521.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-30 13:00:00 | 4547.05 | 2024-09-03 10:15:00 | 4521.20 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-08-30 15:00:00 | 4554.55 | 2024-09-03 10:15:00 | 4521.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-09-16 13:15:00 | 4515.70 | 2024-09-18 09:15:00 | 4386.80 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-09-16 14:00:00 | 4516.85 | 2024-09-18 09:15:00 | 4386.80 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-09-17 10:45:00 | 4516.70 | 2024-09-18 09:15:00 | 4386.80 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-09-25 09:15:00 | 4257.25 | 2024-09-26 09:15:00 | 4300.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-09-25 10:00:00 | 4260.05 | 2024-09-26 09:15:00 | 4300.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-10-03 09:15:00 | 4253.70 | 2024-10-04 11:15:00 | 4295.95 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-04 12:30:00 | 4250.25 | 2024-10-07 09:15:00 | 4275.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-04 15:15:00 | 4251.90 | 2024-10-07 09:15:00 | 4275.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-10-24 09:15:00 | 4051.90 | 2024-10-28 10:15:00 | 4103.55 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-10-24 13:30:00 | 4051.50 | 2024-10-28 10:15:00 | 4103.55 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-25 10:45:00 | 4058.30 | 2024-10-28 10:15:00 | 4103.55 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-25 11:45:00 | 4059.40 | 2024-10-28 10:15:00 | 4103.55 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-10-30 09:15:00 | 4087.00 | 2024-10-30 15:15:00 | 4071.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-11-11 10:15:00 | 4208.95 | 2024-11-13 12:15:00 | 4137.55 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-11-12 14:15:00 | 4210.75 | 2024-11-13 12:15:00 | 4137.55 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-18 10:15:00 | 4012.85 | 2024-11-22 09:15:00 | 4131.85 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-11-18 13:00:00 | 4021.00 | 2024-11-22 10:15:00 | 4145.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-11-18 14:15:00 | 4021.45 | 2024-11-22 10:15:00 | 4145.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-11-18 15:00:00 | 4019.25 | 2024-11-22 10:15:00 | 4145.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-11-21 13:15:00 | 4028.75 | 2024-11-22 10:15:00 | 4145.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2024-12-05 09:15:00 | 4399.15 | 2024-12-10 12:15:00 | 4431.30 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-12-20 11:15:00 | 4236.10 | 2025-01-02 10:15:00 | 4148.65 | STOP_HIT | 1.00 | 2.06% |
| SELL | retest2 | 2025-01-06 11:30:00 | 4078.35 | 2025-01-09 09:15:00 | 4123.65 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-06 15:15:00 | 4090.00 | 2025-01-09 09:15:00 | 4123.65 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-07 09:30:00 | 4086.30 | 2025-01-09 09:15:00 | 4123.65 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-01-15 15:00:00 | 4251.95 | 2025-01-16 11:15:00 | 4204.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-01-29 11:15:00 | 4079.75 | 2025-01-29 14:15:00 | 4102.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-01-31 09:15:00 | 4116.85 | 2025-02-01 11:15:00 | 4078.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-02-01 10:45:00 | 4114.40 | 2025-02-01 11:15:00 | 4078.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-06 09:15:00 | 4102.10 | 2025-02-06 11:15:00 | 4068.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-02-07 09:15:00 | 4032.55 | 2025-02-19 09:15:00 | 3830.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 4032.55 | 2025-02-20 15:15:00 | 3788.00 | STOP_HIT | 0.50 | 6.06% |
| SELL | retest2 | 2025-03-17 11:00:00 | 3487.20 | 2025-03-18 09:15:00 | 3545.35 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-03-17 11:45:00 | 3486.35 | 2025-03-18 09:15:00 | 3545.35 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-17 12:30:00 | 3485.55 | 2025-03-18 09:15:00 | 3545.35 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-03-26 10:15:00 | 3662.45 | 2025-03-26 14:15:00 | 3638.45 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2025-03-26 12:00:00 | 3663.20 | 2025-03-26 14:15:00 | 3638.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2025-03-26 14:00:00 | 3663.95 | 2025-03-26 14:15:00 | 3638.45 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-03-27 13:15:00 | 3655.40 | 2025-03-28 09:15:00 | 3604.25 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest1 | 2025-04-03 09:15:00 | 3459.65 | 2025-04-07 09:15:00 | 3113.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 3244.85 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-09 12:15:00 | 3250.00 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-04-09 14:00:00 | 3251.60 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-11 10:00:00 | 3241.85 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-04-15 11:30:00 | 3243.40 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-04-15 13:45:00 | 3247.10 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-04-15 14:45:00 | 3243.70 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-04-16 09:45:00 | 3245.80 | 2025-04-16 13:15:00 | 3274.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-04-22 11:15:00 | 3331.80 | 2025-05-02 13:15:00 | 3438.10 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-04-22 12:30:00 | 3327.50 | 2025-05-02 13:15:00 | 3438.10 | STOP_HIT | 1.00 | 3.32% |
| BUY | retest2 | 2025-04-23 09:15:00 | 3384.40 | 2025-05-02 13:15:00 | 3438.10 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2025-05-14 09:15:00 | 3530.30 | 2025-05-19 11:15:00 | 3519.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-15 10:30:00 | 3549.00 | 2025-05-19 11:15:00 | 3519.10 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-05-21 12:15:00 | 3518.00 | 2025-05-23 14:15:00 | 3514.10 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-05-22 09:15:00 | 3479.70 | 2025-05-23 14:15:00 | 3514.10 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-05-23 10:15:00 | 3516.00 | 2025-05-23 14:15:00 | 3514.10 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-05-23 11:30:00 | 3517.80 | 2025-05-23 14:15:00 | 3514.10 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-06-12 11:15:00 | 3459.10 | 2025-06-12 13:15:00 | 3428.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-12 12:15:00 | 3461.90 | 2025-06-12 13:15:00 | 3428.60 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-24 12:30:00 | 3413.70 | 2025-06-25 14:15:00 | 3443.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-24 13:00:00 | 3412.00 | 2025-06-25 14:15:00 | 3443.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-06-30 15:00:00 | 3463.50 | 2025-07-01 13:15:00 | 3420.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-01 09:15:00 | 3473.40 | 2025-07-01 13:15:00 | 3420.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-07-08 09:30:00 | 3404.30 | 2025-07-14 09:15:00 | 3234.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 11:30:00 | 3409.10 | 2025-07-14 09:15:00 | 3238.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 14:00:00 | 3404.10 | 2025-07-14 09:15:00 | 3233.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:30:00 | 3404.30 | 2025-07-15 10:15:00 | 3257.90 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2025-07-08 11:30:00 | 3409.10 | 2025-07-15 10:15:00 | 3257.90 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2025-07-08 14:00:00 | 3404.10 | 2025-07-15 10:15:00 | 3257.90 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-08-19 13:45:00 | 3015.90 | 2025-08-20 09:15:00 | 3036.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-08-19 14:30:00 | 3016.40 | 2025-08-20 09:15:00 | 3036.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-03 09:15:00 | 3129.90 | 2025-09-03 10:15:00 | 3091.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-16 12:00:00 | 3132.50 | 2025-09-22 09:15:00 | 3100.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-16 14:00:00 | 3130.00 | 2025-09-22 09:15:00 | 3100.10 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-03 10:30:00 | 2897.00 | 2025-10-06 09:15:00 | 2946.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-10-03 12:15:00 | 2895.00 | 2025-10-06 09:15:00 | 2946.60 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-11-07 09:15:00 | 2972.90 | 2025-11-10 09:15:00 | 3019.70 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-14 10:45:00 | 3098.00 | 2025-11-14 12:15:00 | 3080.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-14 11:45:00 | 3100.30 | 2025-11-14 12:15:00 | 3080.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-11-14 15:00:00 | 3107.00 | 2025-11-18 09:15:00 | 3085.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-11-17 10:00:00 | 3097.90 | 2025-11-18 09:15:00 | 3085.50 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-11-17 12:00:00 | 3103.00 | 2025-11-18 12:15:00 | 3092.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-17 12:30:00 | 3106.30 | 2025-11-18 12:15:00 | 3092.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-24 14:45:00 | 3318.70 | 2025-12-26 10:15:00 | 3289.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-19 11:00:00 | 3169.30 | 2026-01-23 09:15:00 | 3180.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-11 11:45:00 | 2948.00 | 2026-02-12 09:15:00 | 2800.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 11:45:00 | 2948.00 | 2026-02-13 09:15:00 | 2653.20 | TARGET_HIT | 0.50 | 10.00% |
