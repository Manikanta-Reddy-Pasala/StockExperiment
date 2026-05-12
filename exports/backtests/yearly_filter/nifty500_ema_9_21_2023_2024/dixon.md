# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 10825.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 220 |
| ALERT1 | 146 |
| ALERT2 | 145 |
| ALERT2_SKIP | 72 |
| ALERT3 | 383 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 164 |
| PARTIAL | 28 |
| TARGET_HIT | 8 |
| STOP_HIT | 160 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 196 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 69 / 127
- **Target hits / Stop hits / Partials:** 8 / 160 / 28
- **Avg / median % per leg:** 0.53% / -0.69%
- **Sum % (uncompounded):** 103.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 83 | 15 | 18.1% | 8 | 75 | 0 | -0.08% | -6.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.17% | -2.3% |
| BUY @ 3rd Alert (retest2) | 81 | 15 | 18.5% | 8 | 73 | 0 | -0.05% | -4.1% |
| SELL (all) | 113 | 54 | 47.8% | 0 | 85 | 28 | 0.97% | 109.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.46% | -4.4% |
| SELL @ 3rd Alert (retest2) | 110 | 54 | 49.1% | 0 | 82 | 28 | 1.04% | 114.3% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.34% | -6.7% |
| retest2 (combined) | 191 | 69 | 36.1% | 8 | 155 | 28 | 0.58% | 110.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 2934.00 | 2907.21 | 2906.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 14:15:00 | 2943.95 | 2914.55 | 2910.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 11:15:00 | 2920.25 | 2922.20 | 2915.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 11:15:00 | 2920.25 | 2922.20 | 2915.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 11:15:00 | 2920.25 | 2922.20 | 2915.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 11:30:00 | 2917.10 | 2922.20 | 2915.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 14:15:00 | 2909.95 | 2920.07 | 2916.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 15:00:00 | 2909.95 | 2920.07 | 2916.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 2915.00 | 2919.06 | 2916.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 09:15:00 | 3016.85 | 2919.06 | 2916.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-05-24 09:15:00 | 3318.54 | 3285.97 | 3199.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 14:15:00 | 3862.90 | 3919.78 | 3923.55 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 09:15:00 | 3965.00 | 3927.55 | 3923.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 10:15:00 | 3982.25 | 3938.49 | 3928.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 15:15:00 | 4060.00 | 4063.48 | 4025.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-12 09:15:00 | 4030.75 | 4063.48 | 4025.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 4075.00 | 4065.79 | 4029.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 10:15:00 | 4107.05 | 4065.79 | 4029.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-15 14:15:00 | 4517.76 | 4452.52 | 4368.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 4543.00 | 4564.09 | 4565.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 13:15:00 | 4474.15 | 4546.10 | 4557.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 4432.95 | 4388.67 | 4444.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 4432.95 | 4388.67 | 4444.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 4432.95 | 4388.67 | 4444.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:45:00 | 4432.80 | 4388.67 | 4444.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 4467.90 | 4404.52 | 4446.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 4467.90 | 4404.52 | 4446.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 4440.60 | 4411.73 | 4445.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 12:00:00 | 4440.60 | 4411.73 | 4445.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 4428.00 | 4414.99 | 4444.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 09:30:00 | 4406.00 | 4419.50 | 4437.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:15:00 | 4398.00 | 4417.82 | 4433.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 10:15:00 | 4413.20 | 4389.90 | 4402.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 11:00:00 | 4394.65 | 4390.85 | 4401.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 4392.00 | 4391.08 | 4400.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 12:00:00 | 4392.00 | 4391.08 | 4400.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 4410.60 | 4393.44 | 4397.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:30:00 | 4436.80 | 4393.44 | 4397.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 4373.15 | 4389.38 | 4395.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 13:15:00 | 4364.00 | 4384.36 | 4391.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 14:30:00 | 4370.55 | 4377.38 | 4387.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 12:15:00 | 4367.45 | 4374.62 | 4382.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 13:30:00 | 4370.00 | 4368.62 | 4378.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4185.70 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4178.10 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4192.54 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4174.92 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4145.80 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4152.02 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4149.08 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 4151.50 | 4306.45 | 4345.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 4267.80 | 4242.69 | 4288.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-06 09:15:00 | 4267.80 | 4242.69 | 4288.72 | SL hit (close>ema200) qty=0.50 sl=4242.69 alert=retest2 |

### Cycle 5 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 4318.60 | 4258.65 | 4257.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 13:15:00 | 4359.90 | 4308.91 | 4284.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 12:15:00 | 4329.00 | 4344.46 | 4317.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 12:45:00 | 4334.20 | 4344.46 | 4317.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 4330.00 | 4339.30 | 4321.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 4358.90 | 4339.30 | 4321.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:00:00 | 4348.05 | 4345.01 | 4327.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 4268.70 | 4331.15 | 4325.74 | SL hit (close<static) qty=1.00 sl=4321.45 alert=retest2 |

### Cycle 6 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 4299.00 | 4319.36 | 4320.99 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 09:15:00 | 4343.05 | 4324.10 | 4322.99 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 4302.00 | 4319.68 | 4321.08 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 4380.00 | 4331.20 | 4325.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 10:15:00 | 4394.75 | 4343.91 | 4331.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 12:15:00 | 4338.95 | 4351.04 | 4337.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 12:15:00 | 4338.95 | 4351.04 | 4337.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 4338.95 | 4351.04 | 4337.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:00:00 | 4338.95 | 4351.04 | 4337.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 4338.00 | 4348.43 | 4337.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:15:00 | 4341.45 | 4348.43 | 4337.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 4325.95 | 4343.93 | 4336.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 15:00:00 | 4325.95 | 4343.93 | 4336.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 4344.00 | 4343.95 | 4337.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:45:00 | 4359.90 | 4347.07 | 4339.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 10:15:00 | 4317.05 | 4341.06 | 4337.25 | SL hit (close<static) qty=1.00 sl=4323.80 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 4291.20 | 4331.09 | 4333.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 13:15:00 | 4268.15 | 4305.39 | 4315.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 4079.20 | 4059.72 | 4130.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-25 09:45:00 | 4092.55 | 4059.72 | 4130.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 4115.00 | 4077.46 | 4112.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:30:00 | 4114.00 | 4077.46 | 4112.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 4125.05 | 4086.98 | 4113.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:15:00 | 4100.00 | 4086.98 | 4113.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 4139.65 | 4097.51 | 4115.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:45:00 | 4141.55 | 4097.51 | 4115.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 4103.45 | 4098.70 | 4114.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 11:15:00 | 4085.55 | 4098.70 | 4114.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 15:15:00 | 4087.00 | 4103.63 | 4112.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:00:00 | 4088.25 | 4098.00 | 4107.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:45:00 | 4082.95 | 4093.90 | 4104.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 4059.00 | 4043.80 | 4064.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 4059.00 | 4043.80 | 4064.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 4065.00 | 4048.04 | 4064.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 4068.45 | 4048.04 | 4064.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 4107.25 | 4059.88 | 4068.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 4107.25 | 4059.88 | 4068.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 4121.85 | 4072.27 | 4073.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:00:00 | 4121.85 | 4072.27 | 4073.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-31 11:15:00 | 4117.90 | 4081.40 | 4077.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 4117.90 | 4081.40 | 4077.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 11:15:00 | 4135.00 | 4116.13 | 4099.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 4134.40 | 4144.52 | 4124.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 11:15:00 | 4134.40 | 4144.52 | 4124.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 4134.40 | 4144.52 | 4124.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 4133.00 | 4144.52 | 4124.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 4101.00 | 4135.81 | 4122.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 4101.00 | 4135.81 | 4122.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 4096.85 | 4128.02 | 4120.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 15:00:00 | 4132.05 | 4128.83 | 4121.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-04 09:15:00 | 4545.26 | 4411.77 | 4281.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 4730.45 | 4780.46 | 4783.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 11:15:00 | 4679.05 | 4747.78 | 4767.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 4693.20 | 4683.90 | 4714.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 13:00:00 | 4693.20 | 4683.90 | 4714.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 4747.20 | 4696.79 | 4715.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 14:45:00 | 4744.55 | 4696.79 | 4715.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 4750.00 | 4707.44 | 4718.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:15:00 | 4775.75 | 4707.44 | 4718.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 11:15:00 | 4768.00 | 4732.67 | 4728.30 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 4681.70 | 4724.12 | 4725.77 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 12:15:00 | 4735.40 | 4725.77 | 4724.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 13:15:00 | 4762.50 | 4733.11 | 4728.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 14:15:00 | 4731.90 | 4732.87 | 4728.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-18 15:00:00 | 4731.90 | 4732.87 | 4728.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 4727.70 | 4731.84 | 4728.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 4820.00 | 4731.84 | 4728.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 10:15:00 | 4885.00 | 4945.86 | 4947.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 4885.00 | 4945.86 | 4947.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 4844.85 | 4891.19 | 4916.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 4894.00 | 4891.75 | 4914.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 4894.00 | 4891.75 | 4914.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 4894.00 | 4891.75 | 4914.47 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 15:15:00 | 4940.00 | 4921.10 | 4920.48 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 10:15:00 | 4892.95 | 4918.49 | 4919.59 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 4936.95 | 4921.30 | 4920.62 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 4914.00 | 4919.84 | 4920.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 15:15:00 | 4910.00 | 4917.65 | 4918.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 4922.60 | 4918.64 | 4919.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 4922.60 | 4918.64 | 4919.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 4922.60 | 4918.64 | 4919.31 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 10:15:00 | 4938.00 | 4922.51 | 4921.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 12:15:00 | 4949.00 | 4930.61 | 4925.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 5121.55 | 5136.51 | 5096.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 5121.55 | 5136.51 | 5096.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 5121.55 | 5136.51 | 5096.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:00:00 | 5121.55 | 5136.51 | 5096.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 5111.00 | 5131.41 | 5097.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:45:00 | 5081.05 | 5131.41 | 5097.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 5055.40 | 5113.51 | 5097.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 10:00:00 | 5055.40 | 5113.51 | 5097.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 5056.30 | 5102.07 | 5093.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:15:00 | 5029.00 | 5102.07 | 5093.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 11:15:00 | 5032.00 | 5088.05 | 5088.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 15:15:00 | 4978.00 | 5035.93 | 5060.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 09:15:00 | 5069.55 | 5042.66 | 5061.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 5069.55 | 5042.66 | 5061.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 5069.55 | 5042.66 | 5061.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:00:00 | 5069.55 | 5042.66 | 5061.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 5049.50 | 5044.03 | 5060.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:45:00 | 5059.05 | 5044.03 | 5060.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 5085.00 | 5052.22 | 5062.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 11:45:00 | 5087.00 | 5052.22 | 5062.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 5073.95 | 5056.57 | 5063.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-07 13:15:00 | 5071.00 | 5056.57 | 5063.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 15:15:00 | 5082.00 | 5068.65 | 5068.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 15:15:00 | 5082.00 | 5068.65 | 5068.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 5127.45 | 5080.41 | 5073.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 12:15:00 | 5080.55 | 5089.35 | 5080.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-08 13:00:00 | 5080.55 | 5089.35 | 5080.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 5109.20 | 5093.32 | 5082.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 14:45:00 | 5129.00 | 5097.87 | 5085.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:30:00 | 5132.00 | 5106.11 | 5091.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 11:15:00 | 5037.30 | 5107.70 | 5109.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 5037.30 | 5107.70 | 5109.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 10:15:00 | 5017.90 | 5059.58 | 5081.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 5060.70 | 5055.75 | 5072.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 14:45:00 | 5066.00 | 5055.75 | 5072.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 5063.60 | 5057.32 | 5071.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 5099.05 | 5057.32 | 5071.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 5086.60 | 5063.18 | 5072.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:45:00 | 5070.00 | 5073.69 | 5075.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:30:00 | 5066.30 | 5070.25 | 5074.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 09:15:00 | 5108.55 | 5081.39 | 5078.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 5108.55 | 5081.39 | 5078.61 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 12:15:00 | 5039.95 | 5072.38 | 5075.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 10:15:00 | 4999.55 | 5046.89 | 5060.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 4901.80 | 4871.36 | 4903.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 4901.80 | 4871.36 | 4903.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 4901.80 | 4871.36 | 4903.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 4880.35 | 4871.36 | 4903.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 4867.00 | 4870.49 | 4900.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 15:00:00 | 4804.35 | 4850.91 | 4881.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 11:15:00 | 4933.75 | 4859.53 | 4874.01 | SL hit (close>static) qty=1.00 sl=4908.90 alert=retest2 |

### Cycle 27 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 4998.60 | 4901.98 | 4890.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 10:15:00 | 5100.00 | 4941.58 | 4909.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 5215.20 | 5225.99 | 5146.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 13:00:00 | 5215.20 | 5225.99 | 5146.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 5206.55 | 5214.33 | 5170.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:30:00 | 5185.00 | 5214.33 | 5170.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 5228.40 | 5271.02 | 5246.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 5228.40 | 5271.02 | 5246.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 5228.00 | 5262.42 | 5244.47 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 5190.00 | 5232.25 | 5233.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 10:15:00 | 5177.95 | 5213.51 | 5224.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 13:15:00 | 5150.15 | 5136.57 | 5166.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 14:00:00 | 5150.15 | 5136.57 | 5166.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 5145.00 | 5137.26 | 5159.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:45:00 | 5145.45 | 5137.26 | 5159.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 5163.00 | 5119.24 | 5135.98 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 5167.65 | 5147.21 | 5145.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 5193.00 | 5163.70 | 5154.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 12:15:00 | 5266.45 | 5288.06 | 5258.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 13:00:00 | 5266.45 | 5288.06 | 5258.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 5516.05 | 5476.49 | 5439.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 12:00:00 | 5555.00 | 5498.85 | 5456.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 5605.00 | 5514.79 | 5495.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-23 10:15:00 | 5541.55 | 5518.65 | 5499.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 13:15:00 | 5435.70 | 5485.26 | 5488.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 13:15:00 | 5435.70 | 5485.26 | 5488.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 5411.40 | 5470.49 | 5481.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 5499.95 | 5469.92 | 5479.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 5499.95 | 5469.92 | 5479.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 5499.95 | 5469.92 | 5479.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:30:00 | 5387.50 | 5444.31 | 5462.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 11:00:00 | 5402.50 | 5391.77 | 5415.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 11:45:00 | 5409.55 | 5393.12 | 5413.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 5118.12 | 5328.21 | 5374.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 5132.38 | 5328.21 | 5374.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 5139.07 | 5328.21 | 5374.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 5179.35 | 5177.56 | 5259.37 | SL hit (close>ema200) qty=0.50 sl=5177.56 alert=retest2 |

### Cycle 31 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 5334.65 | 5198.54 | 5198.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 10:15:00 | 5356.55 | 5230.14 | 5212.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 09:15:00 | 5294.15 | 5298.80 | 5262.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 09:45:00 | 5297.80 | 5298.80 | 5262.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 5292.00 | 5297.44 | 5265.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 10:30:00 | 5254.30 | 5297.44 | 5265.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 5273.05 | 5292.34 | 5273.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 15:00:00 | 5273.05 | 5292.34 | 5273.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 5271.50 | 5288.18 | 5273.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 5325.30 | 5288.18 | 5273.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 12:15:00 | 5252.75 | 5279.51 | 5274.40 | SL hit (close<static) qty=1.00 sl=5261.10 alert=retest2 |

### Cycle 32 — SELL (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 13:15:00 | 5254.95 | 5273.05 | 5274.16 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 5346.85 | 5281.68 | 5277.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 11:15:00 | 5350.00 | 5302.40 | 5287.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 12:15:00 | 5305.05 | 5332.64 | 5316.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 12:15:00 | 5305.05 | 5332.64 | 5316.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 5305.05 | 5332.64 | 5316.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:00:00 | 5305.05 | 5332.64 | 5316.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 5313.05 | 5328.72 | 5316.01 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 5245.00 | 5305.59 | 5308.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 10:15:00 | 5209.05 | 5255.44 | 5274.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 5307.80 | 5250.78 | 5260.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 5307.80 | 5250.78 | 5260.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 5307.80 | 5250.78 | 5260.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 10:00:00 | 5307.80 | 5250.78 | 5260.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 5330.00 | 5266.63 | 5267.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 10:45:00 | 5354.00 | 5266.63 | 5267.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 11:15:00 | 5282.40 | 5269.78 | 5268.55 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 5252.85 | 5266.66 | 5267.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 09:15:00 | 5234.00 | 5257.59 | 5263.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 10:15:00 | 5261.15 | 5258.30 | 5262.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 10:15:00 | 5261.15 | 5258.30 | 5262.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 5261.15 | 5258.30 | 5262.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:45:00 | 5257.25 | 5258.30 | 5262.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 11:15:00 | 5335.00 | 5273.64 | 5269.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 12:15:00 | 5429.65 | 5304.84 | 5284.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 14:15:00 | 5441.40 | 5446.90 | 5411.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 15:00:00 | 5441.40 | 5446.90 | 5411.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 5450.25 | 5470.08 | 5452.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:00:00 | 5450.25 | 5470.08 | 5452.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 5424.55 | 5460.98 | 5449.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:45:00 | 5413.70 | 5460.98 | 5449.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 5401.00 | 5448.98 | 5445.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 13:00:00 | 5401.00 | 5448.98 | 5445.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 5393.50 | 5437.88 | 5440.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 09:15:00 | 5375.00 | 5419.22 | 5430.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 13:15:00 | 5301.05 | 5297.60 | 5323.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 13:45:00 | 5303.80 | 5297.60 | 5323.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 5362.70 | 5308.35 | 5321.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:30:00 | 5345.75 | 5308.35 | 5321.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 5360.00 | 5318.68 | 5325.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:00:00 | 5360.00 | 5318.68 | 5325.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 5361.45 | 5333.69 | 5331.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 5389.40 | 5344.83 | 5336.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 11:15:00 | 5999.35 | 6015.74 | 5941.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-06 12:00:00 | 5999.35 | 6015.74 | 5941.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 5988.75 | 5995.26 | 5957.58 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 5918.05 | 5962.43 | 5964.65 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 6106.55 | 5990.85 | 5976.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 10:15:00 | 6133.00 | 6019.28 | 5990.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 6386.85 | 6393.45 | 6246.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 13:00:00 | 6386.85 | 6393.45 | 6246.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 6291.00 | 6356.38 | 6274.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 6263.45 | 6356.38 | 6274.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 6289.10 | 6342.93 | 6275.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 10:30:00 | 6209.40 | 6342.93 | 6275.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 6240.50 | 6322.44 | 6272.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 11:30:00 | 6249.85 | 6322.44 | 6272.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 6208.40 | 6299.63 | 6266.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:00:00 | 6208.40 | 6299.63 | 6266.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 6269.90 | 6290.36 | 6268.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 14:45:00 | 6271.00 | 6290.36 | 6268.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 6324.00 | 6297.09 | 6273.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 09:15:00 | 6350.00 | 6297.09 | 6273.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 09:15:00 | 6355.65 | 6333.23 | 6311.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 11:30:00 | 6336.00 | 6341.38 | 6321.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 12:15:00 | 6334.50 | 6341.38 | 6321.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 6345.00 | 6342.10 | 6323.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 12:30:00 | 6345.80 | 6342.10 | 6323.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 6318.55 | 6336.46 | 6324.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 6318.55 | 6336.46 | 6324.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 6330.00 | 6335.17 | 6324.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 6287.45 | 6335.17 | 6324.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 6285.00 | 6325.14 | 6321.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-18 10:15:00 | 6290.00 | 6318.11 | 6318.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 10:15:00 | 6290.00 | 6318.11 | 6318.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 12:15:00 | 6273.95 | 6308.30 | 6313.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 09:15:00 | 6340.00 | 6300.89 | 6307.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 6340.00 | 6300.89 | 6307.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 6340.00 | 6300.89 | 6307.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:30:00 | 6411.45 | 6300.89 | 6307.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 10:15:00 | 6409.10 | 6322.53 | 6316.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 6502.15 | 6386.33 | 6353.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 6372.45 | 6429.28 | 6388.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 6372.45 | 6429.28 | 6388.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 6372.45 | 6429.28 | 6388.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 6372.45 | 6429.28 | 6388.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 6252.70 | 6393.97 | 6376.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 6252.70 | 6393.97 | 6376.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 6252.00 | 6365.57 | 6365.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 6290.65 | 6365.57 | 6365.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 10:15:00 | 6417.35 | 6382.66 | 6373.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 10:30:00 | 6422.00 | 6382.66 | 6373.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 6450.00 | 6450.27 | 6416.89 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 10:15:00 | 6439.00 | 6446.73 | 6447.04 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 11:15:00 | 6455.00 | 6448.38 | 6447.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 12:15:00 | 6467.85 | 6452.28 | 6449.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 6450.00 | 6458.18 | 6454.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 6450.00 | 6458.18 | 6454.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 6450.00 | 6458.18 | 6454.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:15:00 | 6508.00 | 6458.18 | 6454.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:00:00 | 6477.80 | 6506.49 | 6498.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 15:15:00 | 6444.70 | 6487.83 | 6491.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 6444.70 | 6487.83 | 6491.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 09:15:00 | 6301.50 | 6450.56 | 6473.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 11:15:00 | 6379.40 | 6346.68 | 6387.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 12:00:00 | 6379.40 | 6346.68 | 6387.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 12:15:00 | 6368.75 | 6351.09 | 6385.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:00:00 | 6368.75 | 6351.09 | 6385.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 13:15:00 | 6400.85 | 6361.04 | 6386.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 14:00:00 | 6400.85 | 6361.04 | 6386.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 6352.20 | 6359.28 | 6383.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-03 15:15:00 | 6351.00 | 6359.28 | 6383.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-04 09:15:00 | 6415.00 | 6369.10 | 6383.83 | SL hit (close>static) qty=1.00 sl=6408.90 alert=retest2 |

### Cycle 47 — BUY (started 2024-01-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 13:15:00 | 6404.00 | 6390.56 | 6390.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 6445.55 | 6401.54 | 6395.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 6411.50 | 6416.03 | 6405.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 14:00:00 | 6411.50 | 6416.03 | 6405.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 6379.60 | 6408.75 | 6403.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 15:15:00 | 6370.50 | 6408.75 | 6403.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 15:15:00 | 6370.50 | 6401.10 | 6400.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:15:00 | 6344.00 | 6401.10 | 6400.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 6349.15 | 6390.71 | 6395.45 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 13:15:00 | 6396.20 | 6374.72 | 6374.34 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 6352.55 | 6370.28 | 6372.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 6334.00 | 6360.09 | 6367.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 15:15:00 | 6326.45 | 6324.74 | 6342.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 15:15:00 | 6326.45 | 6324.74 | 6342.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 15:15:00 | 6326.45 | 6324.74 | 6342.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:15:00 | 6495.25 | 6324.74 | 6342.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 6433.05 | 6346.40 | 6351.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 6428.40 | 6346.40 | 6351.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 10:15:00 | 6437.95 | 6364.71 | 6358.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 11:15:00 | 6475.20 | 6386.81 | 6369.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 6366.40 | 6499.61 | 6468.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 6366.40 | 6499.61 | 6468.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 6366.40 | 6499.61 | 6468.42 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 12:15:00 | 6374.95 | 6445.49 | 6448.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 13:15:00 | 6356.90 | 6427.77 | 6440.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 09:15:00 | 6387.40 | 6363.53 | 6387.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 6387.40 | 6363.53 | 6387.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 6387.40 | 6363.53 | 6387.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:45:00 | 6378.55 | 6363.53 | 6387.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 6371.05 | 6365.04 | 6386.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:30:00 | 6368.60 | 6365.04 | 6386.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 6363.20 | 6364.67 | 6383.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 09:15:00 | 6322.80 | 6370.03 | 6380.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 15:15:00 | 6006.66 | 6065.39 | 6137.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-24 14:15:00 | 5923.60 | 5891.42 | 5963.20 | SL hit (close>ema200) qty=0.50 sl=5891.42 alert=retest2 |

### Cycle 53 — BUY (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 12:15:00 | 5966.70 | 5905.98 | 5900.29 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 5870.00 | 5894.90 | 5896.64 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 5912.60 | 5898.44 | 5898.09 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 10:15:00 | 5866.00 | 5891.95 | 5895.17 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 5940.05 | 5903.95 | 5900.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 14:15:00 | 5980.05 | 5919.17 | 5907.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 6262.00 | 6271.98 | 6198.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 14:45:00 | 6250.60 | 6271.98 | 6198.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 6296.05 | 6325.04 | 6286.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:30:00 | 6296.10 | 6325.04 | 6286.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 6275.10 | 6306.85 | 6291.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 6275.10 | 6306.85 | 6291.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 6236.00 | 6292.68 | 6286.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 6236.00 | 6292.68 | 6286.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 12:15:00 | 6238.45 | 6278.21 | 6280.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 13:15:00 | 6215.15 | 6265.59 | 6274.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 09:15:00 | 6260.35 | 6256.32 | 6267.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 6260.35 | 6256.32 | 6267.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 6260.35 | 6256.32 | 6267.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:30:00 | 6235.00 | 6256.32 | 6267.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 6155.45 | 6236.14 | 6257.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 11:15:00 | 6138.25 | 6236.14 | 6257.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 14:15:00 | 6276.00 | 6243.53 | 6253.77 | SL hit (close>static) qty=1.00 sl=6264.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 13:15:00 | 6305.00 | 6263.53 | 6259.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 6335.35 | 6286.66 | 6275.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 6344.30 | 6368.60 | 6343.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 12:15:00 | 6344.30 | 6368.60 | 6343.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 6344.30 | 6368.60 | 6343.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:00:00 | 6344.30 | 6368.60 | 6343.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 6347.85 | 6364.45 | 6343.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 14:15:00 | 6371.00 | 6364.45 | 6343.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-26 09:15:00 | 7008.10 | 6842.49 | 6787.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 6731.60 | 6830.04 | 6835.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 6684.05 | 6765.66 | 6799.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 6650.45 | 6643.81 | 6711.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 6650.45 | 6643.81 | 6711.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 6672.20 | 6661.39 | 6695.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 6685.00 | 6661.39 | 6695.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 6751.90 | 6679.49 | 6701.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:00:00 | 6751.90 | 6679.49 | 6701.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 6856.45 | 6714.89 | 6715.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:00:00 | 6856.45 | 6714.89 | 6715.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 6972.00 | 6766.31 | 6738.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 6988.45 | 6842.55 | 6779.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 7082.00 | 7098.57 | 7007.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:15:00 | 7023.00 | 7098.57 | 7007.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 6996.65 | 7060.19 | 7031.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 6996.65 | 7060.19 | 7031.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 6963.25 | 7040.81 | 7025.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 6963.25 | 7040.81 | 7025.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 11:15:00 | 6938.35 | 7020.31 | 7017.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 12:00:00 | 6938.35 | 7020.31 | 7017.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 6994.00 | 7015.05 | 7015.16 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 13:15:00 | 7023.85 | 7016.81 | 7015.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 14:15:00 | 7133.15 | 7040.08 | 7026.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-07 15:15:00 | 7125.00 | 7128.81 | 7092.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:15:00 | 7189.40 | 7128.81 | 7092.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 11:15:00 | 7161.00 | 7136.27 | 7102.80 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 7091.00 | 7124.16 | 7102.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-11 12:15:00 | 7091.00 | 7124.16 | 7102.92 | SL hit (close<ema400) qty=1.00 sl=7102.92 alert=retest1 |

### Cycle 64 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 7053.80 | 7085.50 | 7089.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 12:15:00 | 6970.10 | 7041.15 | 7066.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 6794.20 | 6759.49 | 6857.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 6784.25 | 6759.49 | 6857.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 6830.45 | 6790.87 | 6837.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 6817.75 | 6790.87 | 6837.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 6775.70 | 6787.84 | 6831.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:30:00 | 6753.05 | 6794.27 | 6830.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 13:15:00 | 6917.65 | 6833.38 | 6840.51 | SL hit (close>static) qty=1.00 sl=6879.80 alert=retest2 |

### Cycle 65 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 6960.20 | 6858.74 | 6851.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 09:15:00 | 6981.15 | 6898.11 | 6871.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 6951.45 | 6970.24 | 6928.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-19 10:00:00 | 6951.45 | 6970.24 | 6928.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 6960.00 | 6968.19 | 6931.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 13:15:00 | 6984.15 | 6961.70 | 6934.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 6850.60 | 6932.03 | 6928.91 | SL hit (close<static) qty=1.00 sl=6921.15 alert=retest2 |

### Cycle 66 — SELL (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 11:15:00 | 6901.30 | 6923.99 | 6925.68 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 14:15:00 | 6946.85 | 6929.48 | 6927.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 15:15:00 | 6962.00 | 6935.99 | 6931.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 12:15:00 | 7772.00 | 7797.94 | 7697.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 12:45:00 | 7783.55 | 7797.94 | 7697.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 7711.35 | 7780.62 | 7698.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:45:00 | 7713.15 | 7780.62 | 7698.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 7707.85 | 7766.06 | 7699.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:45:00 | 7700.00 | 7766.06 | 7699.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 7700.00 | 7752.85 | 7699.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:15:00 | 7661.25 | 7752.85 | 7699.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 7688.85 | 7740.05 | 7698.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:30:00 | 7660.80 | 7740.05 | 7698.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 7659.35 | 7723.91 | 7694.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 7668.80 | 7723.91 | 7694.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 7631.20 | 7705.37 | 7689.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 11:45:00 | 7638.65 | 7705.37 | 7689.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 7624.90 | 7677.70 | 7678.55 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 7715.65 | 7678.91 | 7675.36 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 11:15:00 | 7651.15 | 7673.55 | 7673.55 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 12:15:00 | 7698.90 | 7678.62 | 7675.86 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 14:15:00 | 7622.95 | 7665.31 | 7670.15 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 09:15:00 | 7752.35 | 7678.67 | 7675.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 10:15:00 | 7908.95 | 7826.03 | 7770.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 7847.00 | 7865.72 | 7811.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 15:00:00 | 7847.00 | 7865.72 | 7811.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 7829.95 | 7858.56 | 7812.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 7801.40 | 7858.56 | 7812.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 7769.95 | 7840.84 | 7809.00 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 7682.65 | 7770.63 | 7781.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 7627.35 | 7725.34 | 7758.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 7695.00 | 7619.24 | 7667.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 7695.00 | 7619.24 | 7667.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 7695.00 | 7619.24 | 7667.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:00:00 | 7695.00 | 7619.24 | 7667.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 7661.45 | 7627.68 | 7666.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 11:30:00 | 7621.45 | 7624.49 | 7661.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 12:45:00 | 7629.00 | 7625.85 | 7659.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 7599.60 | 7625.85 | 7659.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 10:15:00 | 7240.38 | 7461.46 | 7562.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 10:15:00 | 7247.55 | 7461.46 | 7562.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 10:15:00 | 7219.62 | 7461.46 | 7562.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 13:15:00 | 7512.95 | 7444.64 | 7526.96 | SL hit (close>ema200) qty=0.50 sl=7444.64 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 7743.75 | 7568.58 | 7563.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 7834.30 | 7690.35 | 7628.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 13:15:00 | 7765.70 | 7794.04 | 7718.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 14:00:00 | 7765.70 | 7794.04 | 7718.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 7774.50 | 7790.13 | 7723.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:30:00 | 7745.00 | 7790.13 | 7723.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 8381.10 | 8427.57 | 8366.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:30:00 | 8362.85 | 8427.57 | 8366.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 8353.75 | 8406.91 | 8367.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 8353.75 | 8406.91 | 8367.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 8333.00 | 8392.12 | 8364.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 8381.45 | 8392.12 | 8364.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 8350.20 | 8369.16 | 8357.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:45:00 | 8384.00 | 8365.25 | 8357.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 13:30:00 | 8380.50 | 8368.92 | 8360.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 14:30:00 | 8380.00 | 8376.40 | 8364.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 11:30:00 | 8388.00 | 8391.69 | 8376.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 8434.70 | 8407.82 | 8387.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:45:00 | 8391.95 | 8407.82 | 8387.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 8342.95 | 8405.32 | 8392.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 8342.95 | 8405.32 | 8392.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 8408.10 | 8405.88 | 8393.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 8474.95 | 8403.63 | 8397.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 11:15:00 | 8306.50 | 8382.75 | 8389.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 8306.50 | 8382.75 | 8389.40 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 8434.50 | 8389.14 | 8384.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 12:15:00 | 8482.15 | 8407.74 | 8393.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 13:15:00 | 8401.00 | 8406.39 | 8394.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 13:45:00 | 8412.05 | 8406.39 | 8394.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 8400.05 | 8405.12 | 8394.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 8437.65 | 8405.12 | 8394.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:30:00 | 8429.65 | 8415.34 | 8401.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 10:45:00 | 8429.75 | 8415.91 | 8402.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 11:15:00 | 8355.20 | 8403.77 | 8398.63 | SL hit (close<static) qty=1.00 sl=8385.90 alert=retest2 |

### Cycle 78 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 8340.20 | 8391.06 | 8393.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 8322.80 | 8377.41 | 8386.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 8343.00 | 8340.84 | 8365.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 09:45:00 | 8331.30 | 8340.84 | 8365.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 8366.55 | 8345.98 | 8365.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:45:00 | 8379.95 | 8345.98 | 8365.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 8323.05 | 8341.39 | 8361.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 8374.00 | 8341.39 | 8361.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 8341.90 | 8341.50 | 8359.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:30:00 | 8364.00 | 8341.50 | 8359.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 8379.80 | 8349.16 | 8361.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:30:00 | 8377.90 | 8349.16 | 8361.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 8398.25 | 8358.98 | 8364.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 8398.25 | 8358.98 | 8364.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-05-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 15:15:00 | 8447.00 | 8376.58 | 8372.27 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 8217.00 | 8344.66 | 8358.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 09:15:00 | 8124.90 | 8280.19 | 8319.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 14:15:00 | 8098.00 | 8066.16 | 8139.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 8098.00 | 8066.16 | 8139.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 8098.00 | 8066.16 | 8139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:45:00 | 8134.80 | 8066.16 | 8139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 8324.95 | 8117.95 | 8150.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:00:00 | 8324.95 | 8117.95 | 8150.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 8290.00 | 8152.36 | 8163.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 11:15:00 | 8198.55 | 8152.36 | 8163.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 8256.00 | 8169.98 | 8168.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 12:15:00 | 8256.00 | 8169.98 | 8168.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 13:15:00 | 8271.95 | 8190.37 | 8178.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 10:15:00 | 9221.05 | 9241.05 | 9103.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 10:30:00 | 9221.30 | 9241.05 | 9103.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 9237.90 | 9278.02 | 9226.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 9237.90 | 9278.02 | 9226.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 9237.10 | 9269.84 | 9227.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 14:30:00 | 9310.00 | 9256.42 | 9233.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 9151.75 | 9236.06 | 9228.00 | SL hit (close<static) qty=1.00 sl=9222.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 9150.00 | 9218.85 | 9220.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 9090.05 | 9193.09 | 9209.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 9155.85 | 9145.71 | 9175.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 9155.85 | 9145.71 | 9175.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 9196.00 | 9151.66 | 9170.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:00:00 | 9196.00 | 9151.66 | 9170.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 9242.40 | 9169.81 | 9176.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 9242.40 | 9169.81 | 9176.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 9244.75 | 9184.80 | 9183.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 15:15:00 | 9253.00 | 9198.44 | 9189.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 9172.00 | 9193.15 | 9187.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 9172.00 | 9193.15 | 9187.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 9172.00 | 9193.15 | 9187.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:15:00 | 9251.40 | 9206.37 | 9194.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 11:15:00 | 9334.90 | 9625.34 | 9532.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 8897.40 | 9479.75 | 9474.77 | SL hit (close<static) qty=1.00 sl=9155.55 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 8913.05 | 9366.41 | 9423.70 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 9465.65 | 9320.05 | 9312.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 9732.00 | 9402.44 | 9350.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 11:15:00 | 11396.20 | 11407.70 | 11208.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 12:00:00 | 11396.20 | 11407.70 | 11208.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 11444.20 | 11390.37 | 11273.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:30:00 | 11497.00 | 11411.13 | 11293.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:00:00 | 11481.85 | 11446.97 | 11341.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 15:15:00 | 11496.00 | 11452.58 | 11353.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 12:15:00 | 11491.40 | 11564.94 | 11568.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 11491.40 | 11564.94 | 11568.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 13:15:00 | 11426.50 | 11537.25 | 11555.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 10:15:00 | 11554.90 | 11497.44 | 11525.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 11554.90 | 11497.44 | 11525.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 11554.90 | 11497.44 | 11525.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 11560.00 | 11497.44 | 11525.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 11562.05 | 11510.36 | 11528.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 11562.05 | 11510.36 | 11528.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 11582.90 | 11524.87 | 11533.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:30:00 | 11594.60 | 11524.87 | 11533.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 13:15:00 | 11608.00 | 11541.49 | 11540.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 11731.05 | 11579.41 | 11557.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 12356.40 | 12364.86 | 12154.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 12356.40 | 12364.86 | 12154.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 12579.70 | 12653.09 | 12579.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 12553.50 | 12653.09 | 12579.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 12525.00 | 12627.47 | 12574.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 12525.00 | 12627.47 | 12574.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 12513.45 | 12604.67 | 12568.74 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 14:15:00 | 12498.90 | 12546.46 | 12548.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 12415.45 | 12507.83 | 12528.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 12478.00 | 12476.93 | 12504.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 15:00:00 | 12478.00 | 12476.93 | 12504.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 12498.00 | 12481.15 | 12504.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 12602.00 | 12481.15 | 12504.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 12550.50 | 12495.02 | 12508.46 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 12533.20 | 12519.13 | 12518.11 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 12330.95 | 12486.31 | 12504.22 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 12575.00 | 12471.01 | 12466.07 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 12408.45 | 12477.49 | 12484.24 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 12566.30 | 12490.17 | 12487.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 14:15:00 | 12678.60 | 12538.38 | 12511.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 12582.00 | 12600.25 | 12555.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 12582.00 | 12600.25 | 12555.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 12559.45 | 12591.56 | 12558.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 12559.45 | 12591.56 | 12558.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 12554.15 | 12584.08 | 12558.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 12530.70 | 12584.08 | 12558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 12550.00 | 12577.26 | 12557.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 12230.50 | 12577.26 | 12557.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 12160.40 | 12493.89 | 12521.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 11723.00 | 12339.71 | 12449.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 11380.00 | 11367.88 | 11576.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 11225.00 | 11367.88 | 11576.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 11112.65 | 10947.57 | 11010.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 11144.20 | 10947.57 | 11010.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 11125.65 | 10983.19 | 11020.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 11125.65 | 10983.19 | 11020.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 11178.95 | 11064.01 | 11051.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 11266.30 | 11104.47 | 11071.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 11890.20 | 12026.36 | 11846.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 10:00:00 | 11890.20 | 12026.36 | 11846.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 11698.70 | 11960.82 | 11833.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 11698.70 | 11960.82 | 11833.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 11697.75 | 11908.21 | 11821.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:30:00 | 11693.40 | 11908.21 | 11821.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 11700.00 | 11766.49 | 11771.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 11288.00 | 11616.40 | 11690.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 11529.15 | 11297.58 | 11442.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 11529.15 | 11297.58 | 11442.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 11529.15 | 11297.58 | 11442.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 11529.15 | 11297.58 | 11442.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 11379.00 | 11313.86 | 11436.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:30:00 | 11333.75 | 11311.43 | 11415.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 11535.00 | 11405.99 | 11397.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 11535.00 | 11405.99 | 11397.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 11588.00 | 11442.39 | 11415.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 11464.25 | 11557.15 | 11500.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 13:15:00 | 11464.25 | 11557.15 | 11500.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 11464.25 | 11557.15 | 11500.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 11464.25 | 11557.15 | 11500.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 11464.25 | 11538.57 | 11497.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 11464.25 | 11538.57 | 11497.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 11440.00 | 11518.85 | 11491.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 11697.00 | 11518.85 | 11491.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-20 09:15:00 | 12866.70 | 12634.70 | 12419.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 13135.00 | 13242.36 | 13245.00 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 13345.25 | 13236.74 | 13235.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 13418.85 | 13282.89 | 13257.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 13083.05 | 13279.04 | 13265.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 13083.05 | 13279.04 | 13265.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 13083.05 | 13279.04 | 13265.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 13083.05 | 13279.04 | 13265.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 12910.00 | 13205.23 | 13232.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 12735.85 | 13111.35 | 13187.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 12774.70 | 12749.81 | 12922.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 11:15:00 | 12877.05 | 12796.57 | 12914.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 12877.05 | 12796.57 | 12914.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:30:00 | 12920.00 | 12796.57 | 12914.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 12945.70 | 12826.39 | 12917.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:45:00 | 12946.70 | 12826.39 | 12917.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 12984.35 | 12857.98 | 12923.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:45:00 | 12972.00 | 12857.98 | 12923.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 12399.05 | 12235.15 | 12331.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 12399.05 | 12235.15 | 12331.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 12418.00 | 12271.72 | 12339.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 12616.00 | 12271.72 | 12339.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 12519.85 | 12400.55 | 12386.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 12602.40 | 12485.66 | 12435.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 13964.95 | 13982.11 | 13754.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:00:00 | 13964.95 | 13982.11 | 13754.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 13780.30 | 13954.68 | 13815.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 13780.30 | 13954.68 | 13815.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 13811.00 | 13925.94 | 13815.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:15:00 | 13653.50 | 13925.94 | 13815.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 13660.45 | 13872.84 | 13801.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 13564.65 | 13872.84 | 13801.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 13477.50 | 13793.77 | 13771.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 13477.50 | 13793.77 | 13771.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 13410.55 | 13717.13 | 13738.95 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 13835.75 | 13764.66 | 13756.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 13955.20 | 13802.77 | 13774.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 14203.00 | 14287.75 | 14203.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 14203.00 | 14287.75 | 14203.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 14203.00 | 14287.75 | 14203.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 14225.00 | 14287.75 | 14203.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 14126.95 | 14255.59 | 14196.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 14150.05 | 14255.59 | 14196.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 14075.50 | 14219.57 | 14185.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:15:00 | 14068.50 | 14219.57 | 14185.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 14165.05 | 14205.63 | 14185.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 14138.15 | 14205.63 | 14185.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 14033.10 | 14171.13 | 14171.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 10:15:00 | 13980.00 | 14132.90 | 14153.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 14090.05 | 14045.96 | 14096.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 14090.05 | 14045.96 | 14096.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 14090.05 | 14045.96 | 14096.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 14090.05 | 14045.96 | 14096.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 14115.50 | 14059.87 | 14098.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 14099.00 | 14059.87 | 14098.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 14018.20 | 14051.54 | 14091.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 14010.10 | 14051.54 | 14091.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 13820.10 | 14033.34 | 14061.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 14216.25 | 13970.81 | 13979.46 | SL hit (close>static) qty=1.00 sl=14195.65 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 14099.40 | 13996.53 | 13990.36 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 13700.60 | 13965.09 | 13996.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 14:15:00 | 13631.10 | 13823.53 | 13917.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 13782.90 | 13764.71 | 13862.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 13782.90 | 13764.71 | 13862.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 13895.00 | 13790.77 | 13865.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 13895.00 | 13790.77 | 13865.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 13809.35 | 13794.48 | 13860.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:30:00 | 13847.65 | 13794.48 | 13860.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 13884.00 | 13635.39 | 13699.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 13884.00 | 13635.39 | 13699.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 14035.00 | 13715.32 | 13729.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 14035.00 | 13715.32 | 13729.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 14226.00 | 13817.45 | 13775.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 12:15:00 | 14400.00 | 13933.96 | 13831.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 15346.00 | 15346.55 | 15225.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 10:45:00 | 15363.40 | 15346.55 | 15225.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 15075.40 | 15287.37 | 15251.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 15075.40 | 15287.37 | 15251.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 15110.70 | 15252.04 | 15238.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:45:00 | 15270.05 | 15245.47 | 15237.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 13:15:00 | 15158.50 | 15228.07 | 15230.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 15158.50 | 15228.07 | 15230.38 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 15365.60 | 15224.80 | 15222.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 10:15:00 | 15467.40 | 15351.00 | 15294.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 15375.35 | 15379.12 | 15328.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 14:15:00 | 15375.35 | 15379.12 | 15328.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 15375.35 | 15379.12 | 15328.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 15365.65 | 15379.12 | 15328.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 15271.60 | 15354.79 | 15326.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 15271.60 | 15354.79 | 15326.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 15234.85 | 15330.80 | 15317.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 15162.55 | 15330.80 | 15317.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 15100.00 | 15284.64 | 15298.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 15043.40 | 15236.39 | 15275.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 15312.40 | 15139.30 | 15206.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 15312.40 | 15139.30 | 15206.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 15312.40 | 15139.30 | 15206.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 15312.40 | 15139.30 | 15206.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 15357.00 | 15182.84 | 15219.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:15:00 | 15369.00 | 15182.84 | 15219.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 15396.80 | 15254.67 | 15247.93 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 15044.95 | 15244.04 | 15265.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 13682.30 | 14931.69 | 15121.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 14136.45 | 14033.09 | 14442.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:45:00 | 14145.75 | 14033.09 | 14442.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 14470.00 | 14199.62 | 14423.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 14470.00 | 14199.62 | 14423.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 14400.95 | 14239.89 | 14421.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:30:00 | 14314.00 | 14241.11 | 14405.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 14619.90 | 14367.85 | 14414.84 | SL hit (close>static) qty=1.00 sl=14498.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 14963.75 | 14544.97 | 14490.79 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 14276.00 | 14487.95 | 14511.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 14184.50 | 14427.26 | 14481.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 14220.00 | 14154.49 | 14264.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 14220.00 | 14154.49 | 14264.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 14220.00 | 14154.49 | 14264.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 14220.00 | 14154.49 | 14264.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 14105.80 | 14098.96 | 14199.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:30:00 | 14161.05 | 14098.96 | 14199.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 14360.00 | 14160.46 | 14210.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 14360.00 | 14160.46 | 14210.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 14471.85 | 14222.74 | 14234.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 14471.85 | 14222.74 | 14234.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 14350.00 | 14248.19 | 14245.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 14501.40 | 14298.83 | 14268.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 11:15:00 | 14210.00 | 14293.63 | 14272.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 11:15:00 | 14210.00 | 14293.63 | 14272.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 14210.00 | 14293.63 | 14272.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 14210.00 | 14293.63 | 14272.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 14205.30 | 14275.96 | 14266.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 14295.00 | 14275.96 | 14266.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 10:15:00 | 15724.50 | 14818.52 | 14536.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 15349.90 | 15456.87 | 15460.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 15028.50 | 15356.42 | 15413.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 14843.90 | 14829.54 | 15011.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 10:15:00 | 14831.05 | 14829.84 | 14994.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 14831.05 | 14829.84 | 14994.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 14960.00 | 14829.84 | 14994.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 15032.85 | 14871.67 | 14985.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 15032.85 | 14871.67 | 14985.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 14842.10 | 14865.75 | 14972.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 14826.50 | 14865.75 | 14972.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 14818.80 | 14846.98 | 14953.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 14818.25 | 14849.04 | 14935.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 14763.65 | 14841.31 | 14906.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 15247.40 | 14914.28 | 14922.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 15247.40 | 14914.28 | 14922.60 | SL hit (close>static) qty=1.00 sl=15088.50 alert=retest2 |

### Cycle 117 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 15124.90 | 14956.40 | 14940.99 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 14775.25 | 14952.71 | 14954.16 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 15212.40 | 14993.61 | 14970.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 15309.55 | 15140.43 | 15065.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 15505.05 | 15562.32 | 15390.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 15505.05 | 15562.32 | 15390.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 15504.00 | 15510.41 | 15441.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 15476.50 | 15510.41 | 15441.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 15474.30 | 15501.88 | 15449.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 15475.00 | 15501.88 | 15449.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 15594.00 | 15651.43 | 15583.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 15594.00 | 15651.43 | 15583.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 15635.00 | 15648.14 | 15588.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 15935.40 | 15648.14 | 15588.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 09:15:00 | 17528.94 | 17278.03 | 16957.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 13:15:00 | 18465.10 | 18623.12 | 18641.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 14:15:00 | 18406.75 | 18579.85 | 18620.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 11:15:00 | 18058.15 | 18053.50 | 18237.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:45:00 | 18070.00 | 18053.50 | 18237.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 18124.95 | 17986.66 | 18019.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 18289.90 | 17986.66 | 18019.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 18000.00 | 17989.33 | 18017.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 17781.00 | 17945.86 | 17995.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:00:00 | 17794.80 | 17915.65 | 17977.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:45:00 | 17881.20 | 17950.13 | 17971.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 17922.35 | 17973.49 | 17978.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 18065.70 | 17962.45 | 17970.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 18065.70 | 17962.45 | 17970.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 18000.00 | 17969.96 | 17972.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 17787.75 | 17969.96 | 17972.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 17956.15 | 17915.02 | 17917.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 17974.30 | 17927.48 | 17923.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 17974.30 | 17927.48 | 17923.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 15:15:00 | 18018.00 | 17956.63 | 17937.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 18234.40 | 18357.50 | 18257.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 18234.40 | 18357.50 | 18257.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 18234.40 | 18357.50 | 18257.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 18234.40 | 18357.50 | 18257.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 18253.65 | 18336.73 | 18256.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 18130.00 | 18336.73 | 18256.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 18267.60 | 18322.90 | 18257.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:15:00 | 18204.35 | 18322.90 | 18257.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 18190.00 | 18296.32 | 18251.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 18162.50 | 18296.32 | 18251.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 17979.90 | 18233.04 | 18227.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 17979.90 | 18233.04 | 18227.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 18024.95 | 18191.42 | 18208.65 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 18263.50 | 18225.78 | 18222.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 18478.10 | 18286.77 | 18251.46 | Break + close above crossover candle high |

### Cycle 124 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 17237.30 | 18139.71 | 18203.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 17021.70 | 17760.06 | 18010.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 16204.95 | 16107.87 | 16414.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 12:00:00 | 16204.95 | 16107.87 | 16414.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 16300.00 | 16180.65 | 16352.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 16218.55 | 16180.65 | 16352.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 16335.95 | 16211.71 | 16351.41 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 16847.00 | 16481.32 | 16451.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 17112.05 | 16774.21 | 16614.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 16870.00 | 16940.13 | 16809.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 16870.00 | 16940.13 | 16809.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 16870.00 | 16940.13 | 16809.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 16870.00 | 16940.13 | 16809.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 16912.70 | 16934.65 | 16818.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 16925.00 | 16934.65 | 16818.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 15630.15 | 17026.67 | 17026.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 15630.15 | 17026.67 | 17026.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 15260.10 | 16673.36 | 16865.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 14974.45 | 15545.81 | 16126.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 15084.20 | 14897.50 | 15423.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 15084.20 | 14897.50 | 15423.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 15288.25 | 15030.50 | 15395.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:45:00 | 15268.65 | 15030.50 | 15395.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 15378.30 | 15140.94 | 15385.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 15367.50 | 15140.94 | 15385.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 15498.55 | 15212.46 | 15395.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 15498.55 | 15212.46 | 15395.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 15430.00 | 15255.97 | 15398.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 15562.10 | 15255.97 | 15398.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 15400.00 | 15294.22 | 15391.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 15400.00 | 15294.22 | 15391.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 15559.00 | 15347.18 | 15407.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 15559.00 | 15347.18 | 15407.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 15514.00 | 15380.54 | 15416.81 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 14:15:00 | 15614.15 | 15446.35 | 15441.61 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 15358.95 | 15437.46 | 15438.93 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 10:15:00 | 15475.00 | 15444.96 | 15442.20 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 15415.00 | 15438.97 | 15439.73 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 12:15:00 | 15528.85 | 15456.95 | 15447.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-27 13:15:00 | 15542.95 | 15474.15 | 15456.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 14:15:00 | 15422.15 | 15463.75 | 15453.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 15422.15 | 15463.75 | 15453.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 15422.15 | 15463.75 | 15453.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 15422.15 | 15463.75 | 15453.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 15381.00 | 15447.20 | 15446.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 15030.00 | 15447.20 | 15446.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 14612.15 | 15280.19 | 15370.91 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 15018.65 | 14842.24 | 14825.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 15106.60 | 14895.11 | 14850.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 14392.10 | 14853.84 | 14847.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 14392.10 | 14853.84 | 14847.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 14392.10 | 14853.84 | 14847.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 14392.10 | 14853.84 | 14847.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 15004.55 | 14883.98 | 14862.20 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 14664.75 | 14824.39 | 14837.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 14511.85 | 14761.88 | 14808.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 14876.00 | 14605.72 | 14674.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 14876.00 | 14605.72 | 14674.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 14876.00 | 14605.72 | 14674.76 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 14802.00 | 14711.06 | 14708.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 14933.55 | 14755.56 | 14728.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 14925.55 | 15055.23 | 14953.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 14925.55 | 15055.23 | 14953.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 14925.55 | 15055.23 | 14953.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 14993.75 | 15055.23 | 14953.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 14798.60 | 15003.90 | 14939.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 14798.60 | 15003.90 | 14939.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 14723.85 | 14947.89 | 14919.84 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 14642.50 | 14886.81 | 14894.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 14526.55 | 14814.76 | 14861.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 14967.05 | 14791.04 | 14834.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 14967.05 | 14791.04 | 14834.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 14967.05 | 14791.04 | 14834.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 14936.80 | 14791.04 | 14834.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 15106.00 | 14854.03 | 14859.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 15106.00 | 14854.03 | 14859.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 11:15:00 | 15120.00 | 14907.23 | 14882.79 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 14725.75 | 14894.03 | 14915.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 14698.60 | 14812.09 | 14866.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 14679.85 | 14668.18 | 14768.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 14679.85 | 14668.18 | 14768.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 14747.10 | 14600.83 | 14696.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 14823.50 | 14600.83 | 14696.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 14808.30 | 14642.32 | 14706.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 14775.20 | 14642.32 | 14706.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 14669.95 | 14647.85 | 14703.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 14779.95 | 14647.85 | 14703.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 14785.05 | 14675.29 | 14710.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 14785.05 | 14675.29 | 14710.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 14720.00 | 14684.23 | 14711.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 14904.60 | 14684.23 | 14711.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 15000.00 | 14747.38 | 14737.71 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 14231.40 | 14655.27 | 14711.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 14114.15 | 14547.04 | 14657.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 14056.35 | 14012.55 | 14217.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 13:15:00 | 14056.35 | 14012.55 | 14217.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 14056.35 | 14012.55 | 14217.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 14036.95 | 14012.55 | 14217.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 14212.70 | 14052.58 | 14217.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 14212.70 | 14052.58 | 14217.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 14200.00 | 14082.06 | 14215.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 14172.45 | 14082.06 | 14215.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 14148.05 | 14095.26 | 14209.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 14059.35 | 14095.26 | 14209.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:45:00 | 14074.90 | 14109.44 | 14125.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 14025.65 | 14109.44 | 14125.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 13983.30 | 14107.88 | 14122.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 14144.80 | 14115.27 | 14124.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 14132.45 | 14115.27 | 14124.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 14162.75 | 14124.76 | 14127.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:00:00 | 14162.75 | 14124.76 | 14127.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-20 11:15:00 | 14157.30 | 14131.27 | 14130.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 14157.30 | 14131.27 | 14130.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 12:15:00 | 14195.60 | 14144.14 | 14136.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 14062.50 | 14162.18 | 14150.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 14062.50 | 14162.18 | 14150.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 14062.50 | 14162.18 | 14150.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 14062.50 | 14162.18 | 14150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 14121.35 | 14154.02 | 14148.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 14216.00 | 14171.01 | 14156.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:00:00 | 14239.00 | 14171.01 | 14156.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 13998.25 | 14132.94 | 14143.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 13998.25 | 14132.94 | 14143.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 13764.25 | 14032.19 | 14094.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 14:15:00 | 14010.30 | 13954.91 | 14021.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 14:15:00 | 14010.30 | 13954.91 | 14021.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 14010.30 | 13954.91 | 14021.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 14010.30 | 13954.91 | 14021.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 14018.60 | 13967.65 | 14021.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 14094.35 | 13967.65 | 14021.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 14283.90 | 14030.90 | 14045.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 14265.95 | 14030.90 | 14045.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 10:15:00 | 14199.00 | 14064.52 | 14059.41 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 13851.10 | 14064.52 | 14070.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 13670.05 | 13985.62 | 14034.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 13990.45 | 13782.24 | 13849.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 13990.45 | 13782.24 | 13849.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 13990.45 | 13782.24 | 13849.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 13990.45 | 13782.24 | 13849.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 14000.00 | 13825.79 | 13863.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 13786.85 | 13825.79 | 13863.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 14240.00 | 13827.44 | 13841.03 | SL hit (close>static) qty=1.00 sl=14090.40 alert=retest2 |

### Cycle 145 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 14022.25 | 13866.40 | 13857.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 14549.40 | 14241.89 | 14102.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 14297.75 | 14430.53 | 14310.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 14297.75 | 14430.53 | 14310.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 14297.75 | 14430.53 | 14310.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 14297.75 | 14430.53 | 14310.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 14395.20 | 14423.47 | 14317.76 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 11:15:00 | 14025.00 | 14234.67 | 14261.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 12:15:00 | 13976.95 | 14183.12 | 14235.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 13288.35 | 13246.71 | 13497.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:45:00 | 13346.15 | 13246.71 | 13497.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 13195.50 | 13076.22 | 13231.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:45:00 | 13195.95 | 13076.22 | 13231.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 13300.00 | 13120.97 | 13238.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 13302.05 | 13120.97 | 13238.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 13438.70 | 13184.52 | 13256.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:00:00 | 13438.70 | 13184.52 | 13256.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 13527.45 | 13253.10 | 13281.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 13527.45 | 13253.10 | 13281.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 13327.05 | 13301.77 | 13300.41 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 13140.00 | 13284.91 | 13296.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 13089.00 | 13232.92 | 13269.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 13284.95 | 13218.07 | 13255.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 13284.95 | 13218.07 | 13255.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 13284.95 | 13218.07 | 13255.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 13284.95 | 13218.07 | 13255.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 13306.30 | 13235.72 | 13259.73 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 13403.85 | 13287.45 | 13280.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 13472.05 | 13324.37 | 13297.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 13989.95 | 14373.46 | 14197.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 13989.95 | 14373.46 | 14197.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 13989.95 | 14373.46 | 14197.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 13989.95 | 14373.46 | 14197.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 13892.00 | 14277.16 | 14169.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 13916.75 | 14277.16 | 14169.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 13809.95 | 14066.30 | 14090.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 13709.65 | 13994.97 | 14055.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 13574.95 | 13501.03 | 13682.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 11:00:00 | 13574.95 | 13501.03 | 13682.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 13430.00 | 13501.95 | 13652.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 13:15:00 | 13400.30 | 13501.95 | 13652.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:00:00 | 13407.80 | 13468.73 | 13576.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:45:00 | 13417.10 | 13245.31 | 13255.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 13435.00 | 13283.25 | 13271.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 13435.00 | 13283.25 | 13271.38 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 13186.85 | 13248.89 | 13256.89 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 13391.35 | 13277.38 | 13269.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 13454.95 | 13328.30 | 13294.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 13138.65 | 13333.22 | 13309.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 13138.65 | 13333.22 | 13309.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 13138.65 | 13333.22 | 13309.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 13138.65 | 13333.22 | 13309.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 13207.55 | 13308.09 | 13300.08 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 13121.95 | 13270.86 | 13283.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 12539.85 | 13066.38 | 13176.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 12857.40 | 12719.86 | 12897.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 12857.40 | 12719.86 | 12897.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 12857.40 | 12719.86 | 12897.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 12795.50 | 12898.23 | 12925.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 13213.20 | 12979.28 | 12956.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 13213.20 | 12979.28 | 12956.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 13284.70 | 13079.68 | 13008.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 16416.00 | 16472.53 | 16071.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:45:00 | 16348.00 | 16472.53 | 16071.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 16459.00 | 16563.00 | 16433.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 16459.00 | 16563.00 | 16433.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 16217.00 | 16493.80 | 16413.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 16217.00 | 16493.80 | 16413.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 16335.00 | 16462.04 | 16406.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:15:00 | 16358.00 | 16462.04 | 16406.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 16164.00 | 16343.31 | 16365.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 16164.00 | 16343.31 | 16365.11 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 16608.00 | 16396.24 | 16387.19 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 16372.00 | 16464.12 | 16468.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 16293.00 | 16429.90 | 16452.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 14:15:00 | 16449.00 | 16433.72 | 16452.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 14:15:00 | 16449.00 | 16433.72 | 16452.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 16449.00 | 16433.72 | 16452.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 16449.00 | 16433.72 | 16452.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 16400.00 | 16426.97 | 16447.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 16607.00 | 16426.97 | 16447.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 16655.00 | 16472.58 | 16466.20 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 16350.00 | 16457.92 | 16470.80 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 16500.00 | 16476.53 | 16476.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 16666.00 | 16527.46 | 16500.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 16455.00 | 16564.39 | 16528.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 16455.00 | 16564.39 | 16528.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 16455.00 | 16564.39 | 16528.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 16455.00 | 16564.39 | 16528.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 16306.00 | 16512.71 | 16508.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 16306.00 | 16512.71 | 16508.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 16282.00 | 16466.57 | 16488.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 16122.00 | 16359.80 | 16433.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 16276.00 | 16144.79 | 16228.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 16276.00 | 16144.79 | 16228.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 16276.00 | 16144.79 | 16228.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 16276.00 | 16144.79 | 16228.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 16231.00 | 16162.03 | 16228.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 16195.00 | 16161.82 | 16222.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:00:00 | 16161.00 | 16161.82 | 16222.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 15385.25 | 15754.92 | 15989.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 15352.95 | 15754.92 | 15989.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 16136.00 | 15488.23 | 15677.61 | SL hit (close>ema200) qty=0.50 sl=15488.23 alert=retest2 |

### Cycle 163 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 15993.00 | 15808.05 | 15793.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 16028.00 | 15852.04 | 15814.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 16141.00 | 16210.19 | 16103.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 16141.00 | 16210.19 | 16103.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 16141.00 | 16210.19 | 16103.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 16115.00 | 16210.19 | 16103.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 16120.00 | 16192.15 | 16104.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 16063.00 | 16192.15 | 16104.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 16087.00 | 16171.12 | 16103.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 16087.00 | 16171.12 | 16103.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 16106.00 | 16158.09 | 16103.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 16076.00 | 16158.09 | 16103.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 16145.00 | 16155.48 | 16107.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 16186.00 | 16155.48 | 16107.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 16172.00 | 16151.14 | 16113.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 16075.00 | 16130.77 | 16110.44 | SL hit (close<static) qty=1.00 sl=16101.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 15690.00 | 16505.67 | 16537.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 15211.00 | 15720.04 | 16053.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 12:15:00 | 15130.00 | 15124.13 | 15452.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 15130.00 | 15124.13 | 15452.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 15027.00 | 15084.53 | 15214.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:15:00 | 15009.00 | 15084.53 | 15214.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 14998.00 | 15069.25 | 15146.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 14967.00 | 14684.23 | 14655.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 14967.00 | 14684.23 | 14655.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 15012.00 | 14895.44 | 14830.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 14928.00 | 14960.86 | 14904.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 14928.00 | 14960.86 | 14904.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 14928.00 | 14960.86 | 14904.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 14926.00 | 14960.86 | 14904.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 14891.00 | 14946.89 | 14903.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 14905.00 | 14946.89 | 14903.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 14954.00 | 14948.31 | 14908.26 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 14788.00 | 14874.57 | 14883.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 14743.00 | 14848.26 | 14870.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 14390.00 | 14316.30 | 14433.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 14390.00 | 14316.30 | 14433.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 14425.00 | 14342.57 | 14408.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 14332.00 | 14327.26 | 14395.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 14333.00 | 14376.80 | 14380.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 14317.00 | 14373.23 | 14377.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 13615.40 | 14141.78 | 14184.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 14378.00 | 14141.78 | 14184.95 | SL hit (close>static) qty=0.50 sl=14141.78 alert=retest2 |

### Cycle 167 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 14514.00 | 14216.23 | 14214.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 14557.00 | 14401.06 | 14313.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 14484.00 | 14541.06 | 14442.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 14484.00 | 14541.06 | 14442.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 14225.00 | 14471.28 | 14427.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 14281.00 | 14471.28 | 14427.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 14123.00 | 14401.62 | 14400.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 14123.00 | 14401.62 | 14400.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 11:15:00 | 14160.00 | 14353.30 | 14378.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 14063.00 | 14189.11 | 14277.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 14203.00 | 14184.03 | 14258.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:15:00 | 14242.00 | 14184.03 | 14258.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 14150.00 | 14177.22 | 14249.08 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 14823.00 | 14344.21 | 14305.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 13:15:00 | 14982.00 | 14623.63 | 14516.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 14517.00 | 14697.84 | 14585.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 14517.00 | 14697.84 | 14585.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 14517.00 | 14697.84 | 14585.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 14535.00 | 14697.84 | 14585.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 14540.00 | 14666.27 | 14581.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 14604.00 | 14579.98 | 14557.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 09:15:00 | 16064.40 | 15925.72 | 15860.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 15950.00 | 16041.37 | 16043.93 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 16180.00 | 16069.10 | 16056.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 16270.00 | 16175.77 | 16119.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 16189.00 | 16198.20 | 16145.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 16189.00 | 16198.20 | 16145.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 16140.00 | 16189.70 | 16155.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 16140.00 | 16189.70 | 16155.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 16148.00 | 16181.36 | 16154.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 16148.00 | 16181.36 | 16154.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 16905.00 | 16762.03 | 16662.65 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 16303.00 | 16666.14 | 16697.59 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 16856.00 | 16717.46 | 16708.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 17077.00 | 16822.55 | 16761.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 16833.00 | 16907.84 | 16837.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 16833.00 | 16907.84 | 16837.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 16833.00 | 16907.84 | 16837.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 16833.00 | 16907.84 | 16837.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 16856.00 | 16897.48 | 16839.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 16956.00 | 16897.48 | 16839.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:45:00 | 16890.00 | 16904.85 | 16857.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:00:00 | 16888.00 | 16931.10 | 16898.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 16882.00 | 16909.54 | 16898.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 16620.00 | 16839.71 | 16867.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 16599.00 | 16791.56 | 16843.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 16648.00 | 16553.85 | 16640.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 16648.00 | 16553.85 | 16640.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 16648.00 | 16553.85 | 16640.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 16648.00 | 16553.85 | 16640.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 16675.00 | 16578.08 | 16643.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 16305.00 | 16578.08 | 16643.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 15870.00 | 15922.64 | 16031.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 15681.00 | 15922.64 | 16031.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 15997.00 | 15945.41 | 16023.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 16001.00 | 15945.41 | 16023.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 16024.00 | 15966.35 | 16013.64 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 16174.00 | 16043.88 | 16040.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 13:15:00 | 16182.00 | 16071.50 | 16052.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 16870.00 | 16912.75 | 16767.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 16870.00 | 16912.75 | 16767.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 16818.00 | 16870.87 | 16791.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 16805.00 | 16870.87 | 16791.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 16780.00 | 16852.70 | 16790.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 16919.00 | 16798.95 | 16783.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:00:00 | 16874.00 | 16852.53 | 16815.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 16839.00 | 16964.50 | 16936.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 16780.00 | 16912.32 | 16916.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 16780.00 | 16912.32 | 16916.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 16668.00 | 16835.72 | 16879.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 16794.00 | 16787.52 | 16842.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 16794.00 | 16787.52 | 16842.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 16794.00 | 16787.52 | 16842.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 16794.00 | 16787.52 | 16842.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 16799.00 | 16789.82 | 16838.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 16762.00 | 16789.82 | 16838.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 16830.00 | 16749.36 | 16792.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 16830.00 | 16749.36 | 16792.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 16775.00 | 16754.49 | 16791.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 16816.00 | 16754.49 | 16791.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 17072.00 | 16796.10 | 16793.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 17413.00 | 17028.35 | 16910.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 14:15:00 | 17861.00 | 17864.54 | 17772.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 15:00:00 | 17861.00 | 17864.54 | 17772.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 17870.00 | 17959.69 | 17896.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 17881.00 | 17959.69 | 17896.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 17998.00 | 17967.36 | 17905.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 18014.00 | 17974.88 | 17914.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 18038.00 | 17974.88 | 17914.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 18016.00 | 17994.86 | 17945.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 18010.00 | 17955.53 | 17944.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 18021.00 | 18014.39 | 17985.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 17963.00 | 18014.39 | 17985.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 17998.00 | 18011.12 | 17986.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 18052.00 | 18016.29 | 17990.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 18040.00 | 18021.03 | 17995.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 18046.00 | 18050.10 | 18023.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 18056.00 | 18050.10 | 18023.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 18040.00 | 18048.87 | 18027.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 18040.00 | 18048.87 | 18027.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 17994.00 | 18037.89 | 18024.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 17994.00 | 18037.89 | 18024.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 17970.00 | 18024.31 | 18019.47 | SL hit (close<static) qty=1.00 sl=17984.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 18075.00 | 18198.62 | 18206.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 18038.00 | 18149.11 | 18181.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 18154.00 | 18137.91 | 18169.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 18154.00 | 18137.91 | 18169.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 18111.00 | 18132.53 | 18164.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 18068.00 | 18127.92 | 18154.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 18178.00 | 18142.59 | 18151.26 | SL hit (close>static) qty=1.00 sl=18166.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 18339.00 | 18188.46 | 18170.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 18409.00 | 18232.56 | 18192.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 18175.00 | 18257.87 | 18221.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 18175.00 | 18257.87 | 18221.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 18175.00 | 18257.87 | 18221.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 18175.00 | 18257.87 | 18221.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 18187.00 | 18243.69 | 18218.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 18166.00 | 18243.69 | 18218.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 18020.00 | 18171.64 | 18188.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 17833.00 | 18103.92 | 18156.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 16503.00 | 16437.44 | 16801.09 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 16355.00 | 16421.95 | 16760.99 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 12:15:00 | 16370.00 | 16429.36 | 16733.54 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 16609.00 | 16527.51 | 16593.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 16609.00 | 16527.51 | 16593.43 | SL hit (close>ema400) qty=1.00 sl=16593.43 alert=retest1 |

### Cycle 181 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 16894.00 | 16665.43 | 16641.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 16997.00 | 16731.74 | 16673.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 16985.00 | 17106.27 | 16980.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 16985.00 | 17106.27 | 16980.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 16985.00 | 17106.27 | 16980.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 16985.00 | 17106.27 | 16980.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 17028.00 | 17090.61 | 16985.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 17067.00 | 17085.89 | 16992.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 16848.00 | 17028.09 | 16981.77 | SL hit (close<static) qty=1.00 sl=16960.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 16655.00 | 17111.47 | 17142.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 16549.00 | 16929.78 | 17049.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 16796.00 | 16732.82 | 16886.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 16857.00 | 16732.82 | 16886.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 16666.00 | 16719.45 | 16866.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 16727.00 | 16719.45 | 16866.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 16776.00 | 16722.15 | 16819.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 16792.00 | 16722.15 | 16819.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 16781.00 | 16746.38 | 16814.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 16685.00 | 16719.56 | 16790.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 16835.00 | 16815.78 | 16814.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 16835.00 | 16815.78 | 16814.09 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 16754.00 | 16803.43 | 16808.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 16697.00 | 16770.71 | 16792.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 15465.00 | 15421.88 | 15513.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 15465.00 | 15421.88 | 15513.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 15442.00 | 15425.90 | 15507.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 15486.00 | 15425.90 | 15507.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 15484.00 | 15440.62 | 15494.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 15484.00 | 15440.62 | 15494.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 15539.00 | 15460.30 | 15498.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 15562.00 | 15460.30 | 15498.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 15526.00 | 15473.44 | 15500.71 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 15656.00 | 15542.18 | 15528.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 15670.00 | 15622.54 | 15583.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 15473.00 | 15595.11 | 15577.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 15473.00 | 15595.11 | 15577.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 15473.00 | 15595.11 | 15577.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 15473.00 | 15595.11 | 15577.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 15504.00 | 15576.89 | 15571.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 15384.00 | 15576.89 | 15571.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 15336.00 | 15528.71 | 15549.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 15242.00 | 15395.92 | 15460.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 14911.00 | 14881.54 | 15013.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:30:00 | 14790.00 | 14877.43 | 14956.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 14991.00 | 14892.55 | 14948.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 14991.00 | 14892.55 | 14948.77 | SL hit (close>ema400) qty=1.00 sl=14948.77 alert=retest1 |

### Cycle 187 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 15125.00 | 14984.07 | 14980.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 15252.00 | 15070.27 | 15023.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 15330.00 | 15371.79 | 15254.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 15330.00 | 15371.79 | 15254.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 15475.00 | 15385.75 | 15281.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:45:00 | 15622.00 | 15453.14 | 15369.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 15623.00 | 15487.11 | 15392.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:00:00 | 15571.00 | 15587.19 | 15493.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 15382.00 | 15567.51 | 15580.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 15382.00 | 15567.51 | 15580.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 15332.00 | 15445.42 | 15508.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 14688.00 | 14575.35 | 14754.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 14688.00 | 14575.35 | 14754.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 14763.00 | 14612.88 | 14755.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 14760.00 | 14612.88 | 14755.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 14827.00 | 14655.70 | 14761.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 14834.00 | 14655.70 | 14761.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 14786.00 | 14704.21 | 14766.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 14750.00 | 14750.90 | 14775.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 14740.00 | 14741.30 | 14766.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 14:15:00 | 14012.50 | 14204.91 | 14369.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 14:15:00 | 14003.00 | 14204.91 | 14369.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 12:15:00 | 13763.00 | 13750.56 | 13944.23 | SL hit (close>ema200) qty=0.50 sl=13750.56 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 13350.00 | 13167.08 | 13163.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 13391.00 | 13211.86 | 13184.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 13589.00 | 13608.30 | 13457.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 13589.00 | 13608.30 | 13457.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 13389.00 | 13577.63 | 13521.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 13389.00 | 13577.63 | 13521.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 13319.00 | 13525.90 | 13503.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 13274.00 | 13525.90 | 13503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 13322.00 | 13485.12 | 13487.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 13271.00 | 13414.60 | 13452.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 13411.00 | 13323.30 | 13389.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 13411.00 | 13323.30 | 13389.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 13411.00 | 13323.30 | 13389.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 13411.00 | 13323.30 | 13389.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 13416.00 | 13341.84 | 13392.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 13416.00 | 13341.84 | 13392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 13384.00 | 13350.27 | 13391.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:00:00 | 13277.00 | 13335.62 | 13380.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:45:00 | 13295.00 | 13304.43 | 13354.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 13267.00 | 13251.69 | 13289.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 12613.15 | 12977.79 | 13121.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 12630.25 | 12977.79 | 13121.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 12603.65 | 12977.79 | 13121.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 13098.00 | 12905.27 | 12995.26 | SL hit (close>ema200) qty=0.50 sl=12905.27 alert=retest2 |

### Cycle 191 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 12231.00 | 12044.34 | 12030.73 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 11993.00 | 12079.86 | 12083.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 12:15:00 | 11866.00 | 11992.97 | 12038.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 11885.00 | 11866.93 | 11947.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 11885.00 | 11866.93 | 11947.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 11885.00 | 11866.93 | 11947.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 11885.00 | 11866.93 | 11947.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 11800.00 | 11792.07 | 11865.51 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 14:15:00 | 11994.00 | 11891.79 | 11890.41 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 11783.00 | 11876.06 | 11888.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 09:15:00 | 11566.00 | 11784.11 | 11832.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 11427.00 | 11419.60 | 11586.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:45:00 | 11513.00 | 11419.60 | 11586.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 11015.00 | 10912.62 | 11035.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 11015.00 | 10912.62 | 11035.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 10975.00 | 10942.27 | 11029.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:00:00 | 10883.00 | 10930.42 | 11015.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:30:00 | 10877.00 | 10904.54 | 10996.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 10338.85 | 10655.70 | 10822.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 10333.15 | 10655.70 | 10822.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 10680.00 | 10636.93 | 10783.49 | SL hit (close>ema200) qty=0.50 sl=10636.93 alert=retest2 |

### Cycle 195 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 10777.00 | 10344.48 | 10302.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 10846.00 | 10533.61 | 10431.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 10417.00 | 10516.75 | 10441.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 10417.00 | 10516.75 | 10441.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 10417.00 | 10516.75 | 10441.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 10447.00 | 10516.75 | 10441.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 10215.00 | 10456.40 | 10421.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 10215.00 | 10456.40 | 10421.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 10207.00 | 10406.52 | 10401.70 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 10158.00 | 10356.82 | 10379.54 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 10905.00 | 10432.68 | 10391.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 11446.00 | 10983.45 | 10736.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 11196.00 | 11413.99 | 11132.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 11218.00 | 11413.99 | 11132.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 11247.00 | 11349.66 | 11231.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 11385.00 | 11352.93 | 11243.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 11429.00 | 11347.25 | 11267.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 11378.00 | 11601.76 | 11626.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 11378.00 | 11601.76 | 11626.32 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 11780.00 | 11608.80 | 11589.29 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 11314.00 | 11603.62 | 11628.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 11289.00 | 11464.34 | 11533.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 11199.00 | 11192.72 | 11300.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:30:00 | 11175.00 | 11192.72 | 11300.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 10232.00 | 10227.73 | 10376.94 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 15:15:00 | 10520.00 | 10440.31 | 10434.73 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 10355.00 | 10423.25 | 10427.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 9926.00 | 10175.67 | 10283.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 10147.00 | 10041.85 | 10171.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 10147.00 | 10041.85 | 10171.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 10151.00 | 10073.14 | 10164.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 10190.00 | 10073.14 | 10164.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 10080.00 | 10074.51 | 10156.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 10051.00 | 10074.51 | 10156.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 10056.00 | 10077.85 | 10138.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 10037.00 | 10064.48 | 10126.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 10045.00 | 10126.09 | 10139.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 10196.00 | 9931.21 | 9990.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 10375.00 | 10019.97 | 10025.17 | SL hit (close>static) qty=1.00 sl=10284.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 10314.00 | 10078.77 | 10051.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 10890.00 | 10327.48 | 10179.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 10585.00 | 10624.26 | 10428.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 10585.00 | 10624.26 | 10428.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 10320.00 | 10552.20 | 10443.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 10589.00 | 10571.76 | 10462.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 10566.00 | 10629.96 | 10562.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 10536.00 | 10629.96 | 10562.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 10357.00 | 10504.27 | 10518.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 10357.00 | 10504.27 | 10518.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 10218.00 | 10418.97 | 10475.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 10262.00 | 10253.70 | 10359.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 10302.00 | 10253.70 | 10359.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 10396.00 | 10278.69 | 10351.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 10434.00 | 10278.69 | 10351.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 10310.00 | 10284.95 | 10348.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 10250.00 | 10284.95 | 10348.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 10297.00 | 10282.58 | 10326.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 10472.00 | 10321.65 | 10336.36 | SL hit (close>static) qty=1.00 sl=10411.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 10670.00 | 10391.32 | 10366.69 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 10323.00 | 10427.81 | 10433.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 10264.00 | 10395.05 | 10418.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 10415.00 | 10338.70 | 10381.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 10415.00 | 10338.70 | 10381.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 10415.00 | 10338.70 | 10381.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 10100.00 | 10349.46 | 10370.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 10475.00 | 10184.65 | 10163.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 10475.00 | 10184.65 | 10163.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 10553.00 | 10258.32 | 10199.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 10095.00 | 10309.45 | 10264.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 10095.00 | 10309.45 | 10264.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 10095.00 | 10309.45 | 10264.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 10095.00 | 10309.45 | 10264.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 10115.00 | 10270.56 | 10251.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 10094.00 | 10270.56 | 10251.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 10157.00 | 10232.24 | 10236.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 10134.00 | 10212.59 | 10226.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 10095.00 | 9898.41 | 10007.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 10095.00 | 9898.41 | 10007.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 10095.00 | 9898.41 | 10007.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 10095.00 | 9898.41 | 10007.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 10025.50 | 9923.82 | 10009.19 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 10373.00 | 10065.85 | 10062.16 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 9785.00 | 10079.98 | 10090.81 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 10100.50 | 10024.82 | 10021.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 10542.00 | 10154.98 | 10083.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 10485.00 | 10491.53 | 10326.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 10435.50 | 10491.53 | 10326.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 10413.00 | 10626.13 | 10564.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 10508.00 | 10595.70 | 10556.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 10515.00 | 10541.20 | 10538.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 10482.50 | 10529.46 | 10533.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 10482.50 | 10529.46 | 10533.27 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 10575.00 | 10538.57 | 10537.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 10860.00 | 10602.85 | 10566.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 11255.00 | 11307.50 | 11173.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 11255.00 | 11307.50 | 11173.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 11255.00 | 11307.50 | 11173.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 11219.00 | 11307.50 | 11173.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 11234.00 | 11295.82 | 11220.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 11234.00 | 11295.82 | 11220.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 11194.00 | 11275.46 | 11218.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 11318.00 | 11275.46 | 11218.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 11242.00 | 11286.65 | 11264.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 11096.00 | 11246.84 | 11258.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 11096.00 | 11246.84 | 11258.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 10995.50 | 11196.57 | 11234.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 10990.00 | 10986.60 | 11093.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:00:00 | 10990.00 | 10986.60 | 11093.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 10866.00 | 10962.48 | 11072.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:30:00 | 10845.50 | 10945.88 | 11055.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:15:00 | 10821.00 | 10945.88 | 11055.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 10856.00 | 10880.24 | 10994.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 11160.00 | 10966.05 | 10998.68 | SL hit (close>static) qty=1.00 sl=11086.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 11139.50 | 11037.69 | 11027.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 11306.00 | 11117.88 | 11067.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 11340.00 | 11348.22 | 11230.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 11340.00 | 11348.22 | 11230.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 11318.50 | 11383.43 | 11314.09 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 11153.00 | 11278.20 | 11279.21 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 11363.00 | 11265.26 | 11254.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 11460.00 | 11304.21 | 11272.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 11310.00 | 11327.73 | 11290.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:45:00 | 11304.00 | 11327.73 | 11290.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 11274.00 | 11316.99 | 11288.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 11258.00 | 11316.99 | 11288.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 11313.00 | 11316.19 | 11291.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 11335.00 | 11293.66 | 11285.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 11336.00 | 11309.22 | 11294.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 11267.00 | 11300.78 | 11292.23 | SL hit (close<static) qty=1.00 sl=11268.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 11220.00 | 11284.62 | 11285.66 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 11297.00 | 11285.96 | 11285.75 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 11204.00 | 11277.26 | 11282.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 12:15:00 | 11130.00 | 11247.81 | 11268.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 11195.00 | 11172.82 | 11220.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 11195.00 | 11172.82 | 11220.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 11195.00 | 11172.82 | 11220.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 11216.00 | 11172.82 | 11220.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 11170.00 | 11172.26 | 11215.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 11139.00 | 11172.26 | 11215.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 14:15:00 | 10582.05 | 11004.54 | 11118.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-17 09:15:00 | 3016.85 | 2023-05-24 09:15:00 | 3318.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-12 10:15:00 | 4107.05 | 2023-06-15 14:15:00 | 4517.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-27 09:30:00 | 4406.00 | 2023-07-05 10:15:00 | 4185.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-27 12:15:00 | 4398.00 | 2023-07-05 10:15:00 | 4178.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-30 10:15:00 | 4413.20 | 2023-07-05 10:15:00 | 4192.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-30 11:00:00 | 4394.65 | 2023-07-05 10:15:00 | 4174.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 13:15:00 | 4364.00 | 2023-07-05 10:15:00 | 4145.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 14:30:00 | 4370.55 | 2023-07-05 10:15:00 | 4152.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-04 12:15:00 | 4367.45 | 2023-07-05 10:15:00 | 4149.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-04 13:30:00 | 4370.00 | 2023-07-05 10:15:00 | 4151.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-27 09:30:00 | 4406.00 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2023-06-27 12:15:00 | 4398.00 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2023-06-30 10:15:00 | 4413.20 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2023-06-30 11:00:00 | 4394.65 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2023-07-03 13:15:00 | 4364.00 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2023-07-03 14:30:00 | 4370.55 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2023-07-04 12:15:00 | 4367.45 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2023-07-04 13:30:00 | 4370.00 | 2023-07-06 09:15:00 | 4267.80 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2023-07-07 10:45:00 | 4241.05 | 2023-07-11 09:15:00 | 4318.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2023-07-07 12:00:00 | 4246.15 | 2023-07-11 09:15:00 | 4318.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-07-07 15:15:00 | 4241.00 | 2023-07-11 09:15:00 | 4318.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2023-07-10 10:15:00 | 4247.00 | 2023-07-11 09:15:00 | 4318.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2023-07-10 12:00:00 | 4233.00 | 2023-07-11 09:15:00 | 4318.60 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2023-07-10 14:15:00 | 4234.00 | 2023-07-11 09:15:00 | 4318.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-07-13 09:15:00 | 4358.90 | 2023-07-13 13:15:00 | 4268.70 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-07-13 11:00:00 | 4348.05 | 2023-07-13 13:15:00 | 4268.70 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-07-18 09:45:00 | 4359.90 | 2023-07-18 10:15:00 | 4317.05 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-07-26 11:15:00 | 4085.55 | 2023-07-31 11:15:00 | 4117.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-07-26 15:15:00 | 4087.00 | 2023-07-31 11:15:00 | 4117.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-07-27 11:00:00 | 4088.25 | 2023-07-31 11:15:00 | 4117.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-07-27 11:45:00 | 4082.95 | 2023-07-31 11:15:00 | 4117.90 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-08-02 15:00:00 | 4132.05 | 2023-08-04 09:15:00 | 4545.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-21 09:15:00 | 4820.00 | 2023-08-25 10:15:00 | 4885.00 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2023-09-07 13:15:00 | 5071.00 | 2023-09-07 15:15:00 | 5082.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-09-08 14:45:00 | 5129.00 | 2023-09-12 11:15:00 | 5037.30 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2023-09-11 09:30:00 | 5132.00 | 2023-09-12 11:15:00 | 5037.30 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2023-09-14 12:45:00 | 5070.00 | 2023-09-15 09:15:00 | 5108.55 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-09-14 13:30:00 | 5066.30 | 2023-09-15 09:15:00 | 5108.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-09-22 15:00:00 | 4804.35 | 2023-09-25 11:15:00 | 4933.75 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2023-10-19 12:00:00 | 5555.00 | 2023-10-23 13:15:00 | 5435.70 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2023-10-23 09:15:00 | 5605.00 | 2023-10-23 13:15:00 | 5435.70 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2023-10-23 10:15:00 | 5541.55 | 2023-10-23 13:15:00 | 5435.70 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-10-26 09:30:00 | 5387.50 | 2023-10-30 09:15:00 | 5118.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-27 11:00:00 | 5402.50 | 2023-10-30 09:15:00 | 5132.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-27 11:45:00 | 5409.55 | 2023-10-30 09:15:00 | 5139.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-26 09:30:00 | 5387.50 | 2023-10-31 09:15:00 | 5179.35 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2023-10-27 11:00:00 | 5402.50 | 2023-10-31 09:15:00 | 5179.35 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2023-10-27 11:45:00 | 5409.55 | 2023-10-31 09:15:00 | 5179.35 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2023-11-06 09:15:00 | 5325.30 | 2023-11-06 12:15:00 | 5252.75 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-11-07 09:15:00 | 5310.25 | 2023-11-07 12:15:00 | 5255.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-12-14 09:15:00 | 6350.00 | 2023-12-18 10:15:00 | 6290.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-12-15 09:15:00 | 6355.65 | 2023-12-18 10:15:00 | 6290.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-12-15 11:30:00 | 6336.00 | 2023-12-18 10:15:00 | 6290.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-15 12:15:00 | 6334.50 | 2023-12-18 10:15:00 | 6290.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-12-29 10:15:00 | 6508.00 | 2024-01-01 15:15:00 | 6444.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-01-01 14:00:00 | 6477.80 | 2024-01-01 15:15:00 | 6444.70 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-01-03 15:15:00 | 6351.00 | 2024-01-04 09:15:00 | 6415.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-01-18 09:15:00 | 6322.80 | 2024-01-20 15:15:00 | 6006.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-18 09:15:00 | 6322.80 | 2024-01-24 14:15:00 | 5923.60 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2024-02-09 11:15:00 | 6138.25 | 2024-02-09 14:15:00 | 6276.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-02-16 14:15:00 | 6371.00 | 2024-02-26 09:15:00 | 7008.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-03-11 09:15:00 | 7189.40 | 2024-03-11 12:15:00 | 7091.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest1 | 2024-03-11 11:15:00 | 7161.00 | 2024-03-11 12:15:00 | 7091.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-03-15 10:30:00 | 6753.05 | 2024-03-15 13:15:00 | 6917.65 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-03-19 13:15:00 | 6984.15 | 2024-03-20 09:15:00 | 6850.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-04-18 11:30:00 | 7621.45 | 2024-04-19 10:15:00 | 7240.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 12:45:00 | 7629.00 | 2024-04-19 10:15:00 | 7247.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 13:15:00 | 7599.60 | 2024-04-19 10:15:00 | 7219.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 11:30:00 | 7621.45 | 2024-04-19 13:15:00 | 7512.95 | STOP_HIT | 0.50 | 1.42% |
| SELL | retest2 | 2024-04-18 12:45:00 | 7629.00 | 2024-04-19 13:15:00 | 7512.95 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2024-04-18 13:15:00 | 7599.60 | 2024-04-19 13:15:00 | 7512.95 | STOP_HIT | 0.50 | 1.14% |
| BUY | retest2 | 2024-05-02 12:45:00 | 8384.00 | 2024-05-07 11:15:00 | 8306.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-05-02 13:30:00 | 8380.50 | 2024-05-07 11:15:00 | 8306.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-05-02 14:30:00 | 8380.00 | 2024-05-07 11:15:00 | 8306.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-05-03 11:30:00 | 8388.00 | 2024-05-07 11:15:00 | 8306.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-05-07 09:15:00 | 8474.95 | 2024-05-07 11:15:00 | 8306.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-05-08 15:15:00 | 8437.65 | 2024-05-09 11:15:00 | 8355.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-05-09 09:30:00 | 8429.65 | 2024-05-09 11:15:00 | 8355.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-05-09 10:45:00 | 8429.75 | 2024-05-09 11:15:00 | 8355.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-05-16 11:15:00 | 8198.55 | 2024-05-16 12:15:00 | 8256.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-05-27 14:30:00 | 9310.00 | 2024-05-28 09:15:00 | 9151.75 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-30 12:15:00 | 9251.40 | 2024-06-04 11:15:00 | 8897.40 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2024-06-04 11:15:00 | 9334.90 | 2024-06-04 11:15:00 | 8897.40 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2024-06-20 10:30:00 | 11497.00 | 2024-06-26 12:15:00 | 11491.40 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-06-20 14:00:00 | 11481.85 | 2024-06-26 12:15:00 | 11491.40 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-06-20 15:15:00 | 11496.00 | 2024-06-26 12:15:00 | 11491.40 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-08-06 12:30:00 | 11333.75 | 2024-08-07 13:15:00 | 11535.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-08-09 09:15:00 | 11697.00 | 2024-08-20 09:15:00 | 12866.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-27 10:15:00 | 14010.10 | 2024-10-01 10:15:00 | 14216.25 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-09-30 09:15:00 | 13820.10 | 2024-10-01 10:15:00 | 14216.25 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-10-17 12:45:00 | 15270.05 | 2024-10-17 13:15:00 | 15158.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-10-28 14:30:00 | 14314.00 | 2024-10-29 11:15:00 | 14619.90 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-11-05 13:15:00 | 14295.00 | 2024-11-06 10:15:00 | 15724.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-14 14:15:00 | 14826.50 | 2024-11-19 09:15:00 | 15247.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-11-14 14:45:00 | 14818.80 | 2024-11-19 09:15:00 | 15247.40 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-11-18 09:30:00 | 14818.25 | 2024-11-19 09:15:00 | 15247.40 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-11-18 14:00:00 | 14763.65 | 2024-11-19 09:15:00 | 15247.40 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-11-29 09:15:00 | 15935.40 | 2024-12-05 09:15:00 | 17528.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-27 10:45:00 | 17781.00 | 2025-01-01 13:15:00 | 17974.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-12-27 12:00:00 | 17794.80 | 2025-01-01 13:15:00 | 17974.30 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-30 09:45:00 | 17881.20 | 2025-01-01 13:15:00 | 17974.30 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-30 12:15:00 | 17922.35 | 2025-01-01 13:15:00 | 17974.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-12-31 09:15:00 | 17787.75 | 2025-01-01 13:15:00 | 17974.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-01-01 11:15:00 | 17956.15 | 2025-01-01 13:15:00 | 17974.30 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-02-18 10:15:00 | 14059.35 | 2025-02-20 11:15:00 | 14157.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-02-19 13:45:00 | 14074.90 | 2025-02-20 11:15:00 | 14157.30 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-02-19 14:15:00 | 14025.65 | 2025-02-20 11:15:00 | 14157.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-02-20 09:15:00 | 13983.30 | 2025-02-20 11:15:00 | 14157.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-02-21 11:30:00 | 14216.00 | 2025-02-21 14:15:00 | 13998.25 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-21 12:00:00 | 14239.00 | 2025-02-21 14:15:00 | 13998.25 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-03 09:15:00 | 13786.85 | 2025-03-03 12:15:00 | 14240.00 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-03-27 13:15:00 | 13400.30 | 2025-04-02 15:15:00 | 13435.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-03-28 11:00:00 | 13407.80 | 2025-04-02 15:15:00 | 13435.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-04-02 14:45:00 | 13417.10 | 2025-04-02 15:15:00 | 13435.00 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-04-09 09:45:00 | 12795.50 | 2025-04-09 12:15:00 | 13213.20 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-04-25 12:15:00 | 16358.00 | 2025-04-25 15:15:00 | 16164.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-05-08 11:30:00 | 16195.00 | 2025-05-09 09:15:00 | 15385.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 12:00:00 | 16161.00 | 2025-05-09 09:15:00 | 15352.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 16195.00 | 2025-05-12 09:15:00 | 16136.00 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2025-05-08 12:00:00 | 16161.00 | 2025-05-12 09:15:00 | 16136.00 | STOP_HIT | 0.50 | 0.15% |
| BUY | retest2 | 2025-05-15 09:15:00 | 16186.00 | 2025-05-15 12:15:00 | 16075.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-05-15 10:30:00 | 16172.00 | 2025-05-15 12:15:00 | 16075.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-05-15 13:45:00 | 16162.00 | 2025-05-21 09:15:00 | 15690.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-05-27 10:15:00 | 15009.00 | 2025-06-04 14:15:00 | 14967.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-05-28 09:30:00 | 14998.00 | 2025-06-04 14:15:00 | 14967.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-06-17 11:30:00 | 14332.00 | 2025-06-23 09:15:00 | 13615.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 14332.00 | 2025-06-23 09:15:00 | 14378.00 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2025-06-18 14:15:00 | 14333.00 | 2025-06-23 09:15:00 | 13616.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 14:15:00 | 14333.00 | 2025-06-23 09:15:00 | 14378.00 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2025-06-19 10:15:00 | 14317.00 | 2025-06-23 09:15:00 | 13601.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 10:15:00 | 14317.00 | 2025-06-23 09:15:00 | 14378.00 | STOP_HIT | 0.50 | -0.43% |
| BUY | retest2 | 2025-07-01 14:15:00 | 14604.00 | 2025-07-16 09:15:00 | 16064.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 16956.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-04 11:45:00 | 16890.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-08-05 11:00:00 | 16888.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-08-05 14:30:00 | 16882.00 | 2025-08-06 09:15:00 | 16620.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-08-22 11:15:00 | 16919.00 | 2025-08-26 12:15:00 | 16780.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-22 14:00:00 | 16874.00 | 2025-08-26 12:15:00 | 16780.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-26 11:00:00 | 16839.00 | 2025-08-26 12:15:00 | 16780.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-09-09 12:30:00 | 18014.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-09-09 13:15:00 | 18038.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-10 10:00:00 | 18016.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-09-11 09:45:00 | 18010.00 | 2025-09-15 14:15:00 | 17970.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-09-12 11:45:00 | 18052.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-12 13:00:00 | 18040.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-15 10:45:00 | 18046.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-09-15 11:15:00 | 18056.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-09-16 09:15:00 | 18030.00 | 2025-09-22 14:15:00 | 18075.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-24 09:15:00 | 18068.00 | 2025-09-24 13:15:00 | 18178.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-10-01 10:45:00 | 16355.00 | 2025-10-03 15:15:00 | 16609.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2025-10-01 12:15:00 | 16370.00 | 2025-10-03 15:15:00 | 16609.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-08 12:45:00 | 17067.00 | 2025-10-08 14:15:00 | 16848.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-09 10:00:00 | 17059.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-10-09 11:30:00 | 17054.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-10-09 12:00:00 | 17055.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-10-16 11:45:00 | 16685.00 | 2025-10-17 11:15:00 | 16835.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest1 | 2025-11-11 10:30:00 | 14790.00 | 2025-11-11 12:15:00 | 14991.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-11-17 09:45:00 | 15622.00 | 2025-11-20 09:15:00 | 15382.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-17 11:00:00 | 15623.00 | 2025-11-20 09:15:00 | 15382.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-18 10:00:00 | 15571.00 | 2025-11-20 09:15:00 | 15382.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-27 10:15:00 | 14750.00 | 2025-12-03 14:15:00 | 14012.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:45:00 | 14740.00 | 2025-12-03 14:15:00 | 14003.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 14750.00 | 2025-12-05 12:15:00 | 13763.00 | STOP_HIT | 0.50 | 6.69% |
| SELL | retest2 | 2025-11-27 11:45:00 | 14740.00 | 2025-12-05 12:15:00 | 13763.00 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2025-12-18 14:00:00 | 13277.00 | 2025-12-23 09:15:00 | 12613.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 09:45:00 | 13295.00 | 2025-12-23 09:15:00 | 12630.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 10:15:00 | 13267.00 | 2025-12-23 09:15:00 | 12603.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 14:00:00 | 13277.00 | 2025-12-24 09:15:00 | 13098.00 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-12-19 09:45:00 | 13295.00 | 2025-12-24 09:15:00 | 13098.00 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-12-22 10:15:00 | 13267.00 | 2025-12-24 09:15:00 | 13098.00 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2026-01-20 11:00:00 | 10883.00 | 2026-01-21 10:15:00 | 10338.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:30:00 | 10877.00 | 2026-01-21 10:15:00 | 10333.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:00:00 | 10883.00 | 2026-01-21 12:15:00 | 10680.00 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2026-01-20 11:30:00 | 10877.00 | 2026-01-21 12:15:00 | 10680.00 | STOP_HIT | 0.50 | 1.81% |
| BUY | retest2 | 2026-02-06 11:15:00 | 11385.00 | 2026-02-13 09:15:00 | 11378.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2026-02-06 14:15:00 | 11429.00 | 2026-02-13 09:15:00 | 11378.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-03-05 10:15:00 | 10051.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2026-03-05 12:45:00 | 10056.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2026-03-05 13:45:00 | 10037.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-03-06 14:45:00 | 10045.00 | 2026-03-10 10:15:00 | 10375.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-03-12 10:45:00 | 10589.00 | 2026-03-13 14:15:00 | 10357.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-03-13 10:45:00 | 10566.00 | 2026-03-13 14:15:00 | 10357.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-03-13 11:15:00 | 10536.00 | 2026-03-13 14:15:00 | 10357.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-17 11:15:00 | 10250.00 | 2026-03-18 09:15:00 | 10472.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-03-17 15:00:00 | 10297.00 | 2026-03-18 09:15:00 | 10472.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-23 09:15:00 | 10100.00 | 2026-03-25 09:15:00 | 10475.00 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-04-13 10:45:00 | 10508.00 | 2026-04-13 14:15:00 | 10482.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-04-13 13:45:00 | 10515.00 | 2026-04-13 14:15:00 | 10482.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-21 09:15:00 | 11318.00 | 2026-04-23 09:15:00 | 11096.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-22 10:15:00 | 11242.00 | 2026-04-23 09:15:00 | 11096.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-04-24 11:30:00 | 10845.50 | 2026-04-27 11:15:00 | 11160.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2026-04-24 12:15:00 | 10821.00 | 2026-04-27 11:15:00 | 11160.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-04-24 14:45:00 | 10856.00 | 2026-04-27 11:15:00 | 11160.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-05-06 09:15:00 | 11335.00 | 2026-05-06 11:15:00 | 11267.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-05-06 11:00:00 | 11336.00 | 2026-05-06 11:15:00 | 11267.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-05-08 11:15:00 | 11139.00 | 2026-05-08 14:15:00 | 10582.05 | PARTIAL | 0.50 | 5.00% |
