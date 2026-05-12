# Supreme Industries Ltd. (SUPREMEIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3654.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 2 / 16 / 3
- **Avg / median % per leg:** 0.88% / -0.99%
- **Sum % (uncompounded):** 18.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.49% | -6.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.49% | -6.8% |
| SELL (all) | 7 | 6 | 85.7% | 0 | 4 | 3 | 3.62% | 25.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 0 | 4 | 3 | 3.62% | 25.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 8 | 38.1% | 2 | 16 | 3 | 0.88% | 18.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 4130.00 | 3633.09 | 3632.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 12:15:00 | 4168.60 | 3643.36 | 3637.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 4253.30 | 4269.74 | 4088.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 15:00:00 | 4253.30 | 4269.74 | 4088.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4107.00 | 4252.93 | 4110.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 4107.00 | 4252.93 | 4110.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 4100.10 | 4251.40 | 4110.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 4100.10 | 4251.40 | 4110.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 4122.30 | 4248.78 | 4110.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:00:00 | 4151.10 | 4234.02 | 4110.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:30:00 | 4156.90 | 4233.49 | 4110.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 4095.80 | 4232.47 | 4130.56 | SL hit (close<static) qty=1.00 sl=4110.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 4177.70 | 4321.74 | 4322.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 4163.70 | 4301.39 | 4311.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 3445.00 | 3410.99 | 3603.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 3445.00 | 3410.99 | 3603.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 3603.90 | 3425.71 | 3596.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 3603.90 | 3425.71 | 3596.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 3612.20 | 3427.57 | 3596.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 3612.20 | 3427.57 | 3596.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 3596.50 | 3431.02 | 3597.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 3612.70 | 3431.02 | 3597.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 3591.20 | 3432.61 | 3597.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 3591.20 | 3432.61 | 3597.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 3587.30 | 3434.15 | 3596.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 3603.40 | 3434.15 | 3596.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3616.70 | 3435.97 | 3597.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 3616.70 | 3435.97 | 3597.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 3609.90 | 3437.70 | 3597.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 3609.90 | 3437.70 | 3597.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 3631.80 | 3441.67 | 3597.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 3625.00 | 3445.50 | 3597.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 3621.50 | 3450.98 | 3598.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 3608.80 | 3459.04 | 3598.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:15:00 | 3443.75 | 3471.78 | 3589.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:15:00 | 3440.42 | 3471.78 | 3589.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:15:00 | 3428.36 | 3471.78 | 3589.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 3472.10 | 3471.78 | 3589.34 | SL hit (close>ema200) qty=0.50 sl=3471.78 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 13:15:00 | 3792.50 | 3597.33 | 3596.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 3840.30 | 3609.16 | 3602.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 3787.50 | 3818.44 | 3734.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 12:00:00 | 3787.50 | 3818.44 | 3734.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 3694.00 | 3866.78 | 3787.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 3694.00 | 3866.78 | 3787.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 3682.10 | 3864.94 | 3787.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 3682.10 | 3864.94 | 3787.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3771.80 | 3849.20 | 3786.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 3771.80 | 3849.20 | 3786.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 3762.40 | 3848.33 | 3786.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:15:00 | 3773.20 | 3848.33 | 3786.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 3797.90 | 3847.42 | 3786.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:30:00 | 3803.50 | 3847.42 | 3786.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 3745.90 | 3846.41 | 3786.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 3745.90 | 3846.41 | 3786.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 3762.00 | 3845.57 | 3786.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 3708.20 | 3845.57 | 3786.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3792.70 | 3836.46 | 3783.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 3756.10 | 3836.46 | 3783.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 3703.80 | 3835.14 | 3783.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 3703.80 | 3835.14 | 3783.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 3710.60 | 3833.90 | 3783.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:45:00 | 3690.30 | 3833.90 | 3783.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 3775.00 | 3796.92 | 3768.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:00:00 | 3775.00 | 3796.92 | 3768.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 3773.20 | 3796.68 | 3768.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:30:00 | 3772.30 | 3796.68 | 3768.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 3760.00 | 3796.32 | 3768.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 3885.30 | 3796.32 | 3768.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 15:00:00 | 3794.60 | 3796.91 | 3770.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:00:00 | 3788.00 | 3796.48 | 3770.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 3779.20 | 3795.87 | 3770.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 3757.00 | 3795.49 | 3770.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-09 15:15:00 | 3757.00 | 3795.49 | 3770.36 | SL hit (close<static) qty=1.00 sl=3760.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 3658.30 | 3755.46 | 3755.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 3648.80 | 3748.56 | 3752.11 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-15 10:00:00 | 4151.10 | 2025-07-22 11:15:00 | 4095.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-07-15 10:30:00 | 4156.90 | 2025-07-22 11:15:00 | 4095.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-07-24 13:30:00 | 4159.30 | 2025-08-19 13:15:00 | 4575.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:30:00 | 4157.50 | 2025-08-19 13:15:00 | 4573.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 09:15:00 | 4451.00 | 2025-09-25 14:15:00 | 4264.40 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-09-23 10:30:00 | 4388.90 | 2025-09-25 14:15:00 | 4264.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-01-06 15:15:00 | 3625.00 | 2026-01-13 11:15:00 | 3443.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:15:00 | 3621.50 | 2026-01-13 11:15:00 | 3440.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 3608.80 | 2026-01-13 11:15:00 | 3428.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:15:00 | 3625.00 | 2026-01-13 12:15:00 | 3472.10 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2026-01-07 11:15:00 | 3621.50 | 2026-01-13 12:15:00 | 3472.10 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2026-01-08 09:15:00 | 3608.80 | 2026-01-13 12:15:00 | 3472.10 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2026-02-03 13:15:00 | 3621.10 | 2026-02-03 13:15:00 | 3685.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-04-08 09:15:00 | 3885.30 | 2026-04-09 15:15:00 | 3757.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-04-08 15:00:00 | 3794.60 | 2026-04-09 15:15:00 | 3757.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-04-09 11:00:00 | 3788.00 | 2026-04-09 15:15:00 | 3757.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-04-09 15:00:00 | 3779.20 | 2026-04-09 15:15:00 | 3757.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-04-13 11:15:00 | 3829.90 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3833.40 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-04-15 13:30:00 | 3831.60 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-04-15 14:15:00 | 3830.10 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.79% |
