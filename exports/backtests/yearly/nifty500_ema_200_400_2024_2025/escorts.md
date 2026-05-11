# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3148.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 59 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 5 |
| TARGET_HIT | 6 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 15 / 42
- **Target hits / Stop hits / Partials:** 6 / 46 / 5
- **Avg / median % per leg:** -0.18% / -1.69%
- **Sum % (uncompounded):** -10.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 5 | 13.9% | 2 | 34 | 0 | -1.59% | -57.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 36 | 5 | 13.9% | 2 | 34 | 0 | -1.59% | -57.1% |
| SELL (all) | 21 | 10 | 47.6% | 4 | 12 | 5 | 2.24% | 47.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 10 | 47.6% | 4 | 12 | 5 | 2.24% | 47.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 15 | 26.3% | 6 | 46 | 5 | -0.18% | -10.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 3787.45 | 3856.86 | 3857.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 3768.65 | 3848.49 | 3852.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 11:15:00 | 3813.00 | 3811.61 | 3831.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 11:15:00 | 3813.00 | 3811.61 | 3831.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 3813.00 | 3811.61 | 3831.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 3813.00 | 3811.61 | 3831.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 3833.75 | 3811.79 | 3830.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 3833.75 | 3811.79 | 3830.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 3855.00 | 3812.22 | 3830.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 3855.00 | 3812.22 | 3830.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 3846.00 | 3812.56 | 3830.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:15:00 | 3818.10 | 3819.78 | 3833.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:00:00 | 3820.20 | 3819.10 | 3832.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 3919.25 | 3818.99 | 3832.14 | SL hit (close>static) qty=1.00 sl=3855.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 4202.75 | 3845.97 | 3845.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 4228.20 | 3853.30 | 3848.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 4019.50 | 4023.28 | 3948.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 4019.50 | 4023.28 | 3948.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 3953.85 | 4022.56 | 3948.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:45:00 | 3983.95 | 4022.56 | 3948.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 3900.00 | 4021.35 | 3948.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 3900.00 | 4021.35 | 3948.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 3909.45 | 4020.23 | 3948.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:45:00 | 3896.65 | 4020.23 | 3948.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 3952.25 | 4004.51 | 3948.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 3940.15 | 4004.51 | 3948.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3939.90 | 4003.33 | 3948.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 3936.00 | 4003.33 | 3948.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 3912.50 | 4002.43 | 3948.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:00:00 | 3912.50 | 4002.43 | 3948.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 3909.60 | 4001.51 | 3948.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:45:00 | 3922.70 | 4001.51 | 3948.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 3903.55 | 3999.43 | 3947.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 3903.55 | 3999.43 | 3947.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 3917.15 | 3998.61 | 3947.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 3908.35 | 3998.61 | 3947.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 3930.85 | 3997.69 | 3947.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:00:00 | 3930.85 | 3997.69 | 3947.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 3967.05 | 3997.38 | 3947.86 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 3498.25 | 3912.12 | 3912.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 3342.85 | 3543.93 | 3639.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 3403.25 | 3390.59 | 3514.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 3403.25 | 3390.59 | 3514.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 3472.00 | 3380.57 | 3478.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:30:00 | 3426.15 | 3454.96 | 3497.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 3492.15 | 3450.26 | 3492.47 | SL hit (close>static) qty=1.00 sl=3490.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 13:15:00 | 3345.00 | 3222.27 | 3221.97 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 3131.00 | 3225.57 | 3225.83 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3358.80 | 3226.89 | 3226.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 3388.30 | 3232.14 | 3229.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 3378.20 | 3389.83 | 3326.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 3378.20 | 3389.83 | 3326.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 3318.00 | 3388.96 | 3326.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 3318.00 | 3388.96 | 3326.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 3329.70 | 3388.37 | 3326.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 3358.30 | 3388.37 | 3326.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 3310.00 | 3386.56 | 3326.49 | SL hit (close<static) qty=1.00 sl=3312.20 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 3135.80 | 3296.50 | 3296.68 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 3363.30 | 3295.45 | 3295.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 3383.90 | 3314.47 | 3306.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 3312.50 | 3322.60 | 3311.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 3312.50 | 3322.60 | 3311.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 3312.50 | 3322.60 | 3311.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 3312.50 | 3322.60 | 3311.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 3307.00 | 3322.44 | 3311.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 3307.00 | 3322.44 | 3311.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 3296.20 | 3322.18 | 3311.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 3296.20 | 3322.18 | 3311.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 3268.10 | 3320.39 | 3310.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:15:00 | 3263.70 | 3320.39 | 3310.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3352.60 | 3381.90 | 3350.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 3352.60 | 3381.90 | 3350.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 3377.80 | 3381.86 | 3350.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 3380.80 | 3381.86 | 3350.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 3331.40 | 3381.07 | 3350.82 | SL hit (close<static) qty=1.00 sl=3344.10 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 3452.20 | 3680.58 | 3681.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 3422.60 | 3674.03 | 3678.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 3600.10 | 3598.29 | 3635.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 10:15:00 | 3600.10 | 3598.29 | 3635.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 3600.10 | 3598.29 | 3635.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 3583.70 | 3644.17 | 3652.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 3404.51 | 3600.35 | 3627.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 3591.90 | 3589.76 | 3619.84 | SL hit (close>ema200) qty=0.50 sl=3589.76 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-22 10:30:00 | 3997.90 | 2024-08-05 09:15:00 | 3796.25 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest2 | 2024-07-23 12:30:00 | 4058.25 | 2024-08-05 09:15:00 | 3796.25 | STOP_HIT | 1.00 | -6.46% |
| SELL | retest2 | 2024-09-18 13:15:00 | 3818.10 | 2024-09-20 09:15:00 | 3919.25 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-09-19 10:00:00 | 3820.20 | 2024-09-20 09:15:00 | 3919.25 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-01-27 09:30:00 | 3426.15 | 2025-01-28 13:15:00 | 3492.15 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-02-03 09:15:00 | 3356.00 | 2025-02-11 09:15:00 | 3188.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 09:15:00 | 3356.00 | 2025-02-12 09:15:00 | 3020.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-24 09:30:00 | 3423.50 | 2025-04-25 13:15:00 | 3345.00 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2025-05-29 11:15:00 | 3358.30 | 2025-05-29 13:15:00 | 3310.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-05-30 13:45:00 | 3343.40 | 2025-06-03 12:15:00 | 3295.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-06-02 09:45:00 | 3344.50 | 2025-06-03 12:15:00 | 3295.10 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-01 09:15:00 | 3380.80 | 2025-08-01 10:15:00 | 3331.40 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-08-04 14:00:00 | 3417.40 | 2025-08-07 09:15:00 | 3334.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-08-05 14:15:00 | 3388.00 | 2025-08-07 09:15:00 | 3334.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-08-12 14:45:00 | 3380.90 | 2025-08-18 10:15:00 | 3718.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 12:15:00 | 3562.10 | 2025-10-07 09:15:00 | 3500.60 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-10-01 13:15:00 | 3588.00 | 2025-10-07 09:15:00 | 3500.60 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-10-08 09:15:00 | 3647.90 | 2025-11-18 09:15:00 | 3567.60 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-10 14:30:00 | 3561.30 | 2025-11-18 09:15:00 | 3567.60 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-11-17 10:30:00 | 3630.00 | 2025-11-18 09:15:00 | 3567.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-17 12:15:00 | 3615.20 | 2025-11-19 14:15:00 | 3581.50 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-11-17 14:30:00 | 3617.00 | 2025-12-17 12:15:00 | 3644.90 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-11-18 15:15:00 | 3616.10 | 2025-12-17 12:15:00 | 3644.90 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-11-24 10:30:00 | 3681.80 | 2025-12-17 12:15:00 | 3644.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-26 15:00:00 | 3693.30 | 2025-12-18 10:15:00 | 3610.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-12-08 12:45:00 | 3680.30 | 2025-12-18 10:15:00 | 3610.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-12-08 13:45:00 | 3682.60 | 2025-12-18 10:15:00 | 3610.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-12-11 12:15:00 | 3675.00 | 2025-12-18 10:15:00 | 3610.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-12-11 13:15:00 | 3671.60 | 2025-12-22 09:15:00 | 3581.60 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-12-11 14:00:00 | 3671.90 | 2025-12-22 09:15:00 | 3581.60 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-12-11 15:00:00 | 3671.30 | 2025-12-22 09:15:00 | 3581.60 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-15 14:00:00 | 3683.60 | 2025-12-22 09:15:00 | 3581.60 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-12-17 09:15:00 | 3681.80 | 2026-01-05 09:15:00 | 3917.43 | TARGET_HIT | 1.00 | 6.40% |
| BUY | retest2 | 2025-12-17 10:45:00 | 3694.00 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-23 09:15:00 | 3686.60 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-23 10:15:00 | 3755.00 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-12-24 11:00:00 | 3750.90 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-01-01 09:15:00 | 3807.90 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2026-01-12 12:15:00 | 3754.90 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2026-01-14 11:30:00 | 3693.60 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-01-14 13:30:00 | 3690.00 | 2026-01-19 09:15:00 | 3631.60 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-16 14:15:00 | 3693.70 | 2026-01-20 09:15:00 | 3507.60 | STOP_HIT | 1.00 | -5.04% |
| SELL | retest2 | 2026-02-13 15:00:00 | 3583.70 | 2026-02-20 15:15:00 | 3404.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 3583.70 | 2026-02-25 09:15:00 | 3591.90 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2026-02-25 14:15:00 | 3586.40 | 2026-03-02 12:15:00 | 3407.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 3584.90 | 2026-03-02 12:15:00 | 3405.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:15:00 | 3587.90 | 2026-03-02 12:15:00 | 3408.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:15:00 | 3586.40 | 2026-03-09 09:15:00 | 3227.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 3584.90 | 2026-03-09 09:15:00 | 3226.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 11:15:00 | 3587.90 | 2026-03-09 09:15:00 | 3229.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 12:30:00 | 3242.00 | 2026-04-28 14:15:00 | 3310.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-04-24 14:45:00 | 3240.60 | 2026-04-28 14:15:00 | 3310.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-04-30 11:00:00 | 3221.00 | 2026-05-04 09:15:00 | 3290.40 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-30 14:00:00 | 3233.80 | 2026-05-04 09:15:00 | 3290.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-05-04 10:45:00 | 3270.60 | 2026-05-07 09:15:00 | 3326.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-05-06 10:45:00 | 3275.00 | 2026-05-07 09:15:00 | 3326.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-05-06 12:45:00 | 3276.60 | 2026-05-07 09:15:00 | 3326.10 | STOP_HIT | 1.00 | -1.51% |
