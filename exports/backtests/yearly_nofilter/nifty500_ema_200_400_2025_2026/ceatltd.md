# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 27
- **Target hits / Stop hits / Partials:** 1 / 27 / 0
- **Avg / median % per leg:** -1.63% / -1.84%
- **Sum % (uncompounded):** -45.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -1.13% | -18.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -1.13% | -18.0% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.30% | -27.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.30% | -27.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 1 | 3.6% | 1 | 27 | 0 | -1.63% | -45.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 3245.30 | 3535.01 | 3535.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 3235.10 | 3529.21 | 3532.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 3293.50 | 3273.04 | 3363.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3365.70 | 3274.80 | 3363.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3365.70 | 3274.80 | 3363.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3365.70 | 3274.80 | 3363.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3379.00 | 3275.83 | 3363.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 3389.00 | 3275.83 | 3363.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3403.10 | 3280.33 | 3361.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 3367.10 | 3281.29 | 3361.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 3376.70 | 3281.29 | 3361.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:45:00 | 3362.00 | 3283.83 | 3361.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 3378.30 | 3290.73 | 3361.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 3375.00 | 3293.19 | 3361.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 3375.40 | 3293.19 | 3361.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 3353.40 | 3297.09 | 3361.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 3364.90 | 3297.09 | 3361.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 3368.60 | 3298.33 | 3361.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 3367.90 | 3298.33 | 3361.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 3379.20 | 3299.13 | 3361.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 3379.20 | 3299.13 | 3361.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 3366.10 | 3299.80 | 3361.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 3369.90 | 3299.80 | 3361.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3378.90 | 3301.22 | 3361.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 3378.90 | 3301.22 | 3361.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3362.00 | 3301.82 | 3361.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 3335.00 | 3304.69 | 3362.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 3353.00 | 3305.26 | 3362.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 3355.00 | 3306.23 | 3362.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 3407.90 | 3307.87 | 3357.86 | SL hit (close>static) qty=1.00 sl=3382.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 3457.40 | 3388.68 | 3388.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 3474.10 | 3391.54 | 3390.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 3874.00 | 3902.36 | 3742.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 15:00:00 | 3874.00 | 3902.36 | 3742.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3765.00 | 3890.89 | 3798.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 3733.20 | 3890.89 | 3798.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 3798.90 | 3889.97 | 3798.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 3819.00 | 3889.97 | 3798.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 12:45:00 | 3810.00 | 3884.96 | 3799.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:30:00 | 3805.20 | 3883.41 | 3799.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:00:00 | 3806.30 | 3883.41 | 3799.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 3803.80 | 3882.62 | 3799.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 3791.40 | 3882.62 | 3799.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 3767.70 | 3881.47 | 3799.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 3754.40 | 3881.47 | 3799.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 3771.80 | 3880.38 | 3799.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 3771.80 | 3880.38 | 3799.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 3785.00 | 3877.68 | 3799.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 3791.00 | 3877.68 | 3799.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 3768.60 | 3873.63 | 3798.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 3760.80 | 3873.63 | 3798.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 3736.30 | 3871.18 | 3798.31 | SL hit (close<static) qty=1.00 sl=3753.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 3710.00 | 3790.93 | 3791.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 3681.00 | 3787.10 | 3789.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 3780.90 | 3768.29 | 3779.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 3791.10 | 3768.52 | 3779.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 3791.10 | 3768.52 | 3779.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3755.00 | 3768.39 | 3778.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 3729.40 | 3768.39 | 3778.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:00:00 | 3742.60 | 3768.13 | 3778.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 3742.00 | 3767.82 | 3778.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 3830.20 | 3764.21 | 3776.14 | SL hit (close>static) qty=1.00 sl=3808.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 3878.10 | 3787.59 | 3787.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 3920.40 | 3791.20 | 3789.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 3851.30 | 3861.66 | 3828.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:45:00 | 3852.70 | 3861.66 | 3828.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3820.40 | 3865.04 | 3834.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3820.40 | 3865.04 | 3834.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3804.80 | 3864.44 | 3833.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 3825.90 | 3858.83 | 3832.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 3767.90 | 3857.92 | 3831.94 | SL hit (close<static) qty=1.00 sl=3770.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3460.00 | 3808.15 | 3809.54 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 15:00:00 | 3508.40 | 2025-07-08 14:15:00 | 3859.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-23 09:15:00 | 3485.50 | 2025-07-24 09:15:00 | 3424.40 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-23 09:45:00 | 3510.80 | 2025-07-24 09:15:00 | 3424.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-07-23 10:45:00 | 3486.50 | 2025-07-24 09:15:00 | 3424.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-04 10:30:00 | 3367.10 | 2025-09-16 09:15:00 | 3407.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-04 11:00:00 | 3376.70 | 2025-09-16 09:15:00 | 3407.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-04 13:45:00 | 3362.00 | 2025-09-16 09:15:00 | 3407.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-08 11:15:00 | 3378.30 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-09-11 09:15:00 | 3335.00 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-09-11 10:15:00 | 3353.00 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-11 11:30:00 | 3355.00 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-29 11:15:00 | 3354.60 | 2025-09-30 09:15:00 | 3433.80 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-09-30 13:00:00 | 3406.30 | 2025-10-01 09:15:00 | 3465.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-12-09 11:15:00 | 3819.00 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-12-10 12:45:00 | 3810.00 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-12-10 14:30:00 | 3805.20 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-10 15:00:00 | 3806.30 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-01-02 10:30:00 | 3837.90 | 2026-01-06 09:15:00 | 3744.70 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-01-08 09:30:00 | 3846.00 | 2026-01-09 10:15:00 | 3800.70 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-09 09:45:00 | 3845.00 | 2026-01-09 10:15:00 | 3800.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-13 14:45:00 | 3838.30 | 2026-01-16 12:15:00 | 3796.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-19 09:15:00 | 3879.20 | 2026-01-20 12:15:00 | 3773.80 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-01-20 10:00:00 | 3875.00 | 2026-01-20 12:15:00 | 3773.80 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-01-20 10:30:00 | 3834.10 | 2026-01-20 12:15:00 | 3773.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-01 12:15:00 | 3729.40 | 2026-02-03 10:15:00 | 3830.20 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-02-01 13:00:00 | 3742.60 | 2026-02-03 10:15:00 | 3830.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-02-01 15:00:00 | 3742.00 | 2026-02-03 10:15:00 | 3830.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-02-23 09:30:00 | 3825.90 | 2026-02-23 10:15:00 | 3767.90 | STOP_HIT | 1.00 | -1.52% |
