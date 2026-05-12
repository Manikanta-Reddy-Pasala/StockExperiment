# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 4160.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 96 |
| ALERT2 | 96 |
| ALERT2_SKIP | 43 |
| ALERT3 | 278 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 120 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 35 / 96
- **Target hits / Stop hits / Partials:** 6 / 117 / 8
- **Avg / median % per leg:** 0.01% / -0.86%
- **Sum % (uncompounded):** 1.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 11 | 21.2% | 4 | 48 | 0 | 0.07% | 3.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.09% | -4.2% |
| BUY @ 3rd Alert (retest2) | 50 | 11 | 22.0% | 4 | 46 | 0 | 0.15% | 7.7% |
| SELL (all) | 79 | 24 | 30.4% | 2 | 69 | 8 | -0.02% | -1.6% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.97% | 11.9% |
| SELL @ 3rd Alert (retest2) | 75 | 21 | 28.0% | 2 | 66 | 7 | -0.18% | -13.5% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.28% | 7.7% |
| retest2 (combined) | 125 | 32 | 25.6% | 6 | 112 | 7 | -0.05% | -5.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 4257.75 | 4312.56 | 4314.49 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 4346.20 | 4308.26 | 4304.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 4368.80 | 4333.08 | 4319.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 4352.05 | 4357.13 | 4340.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:15:00 | 4398.45 | 4357.13 | 4340.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 4340.00 | 4353.71 | 4340.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 4340.00 | 4353.71 | 4340.23 | SL hit (close<ema400) qty=1.00 sl=4340.23 alert=retest1 |

### Cycle 3 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 4345.00 | 4397.04 | 4400.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 4316.05 | 4363.82 | 4379.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 15:15:00 | 4200.00 | 4195.63 | 4226.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 09:15:00 | 4182.00 | 4195.63 | 4226.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 4096.95 | 4118.45 | 4149.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 4048.90 | 4118.45 | 4149.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 4064.10 | 4020.98 | 4057.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 4050.00 | 4037.59 | 4059.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 12:15:00 | 4186.00 | 4093.88 | 4081.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 4186.00 | 4093.88 | 4081.85 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 14:15:00 | 4041.15 | 4099.65 | 4103.64 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 11:15:00 | 4128.40 | 4094.72 | 4092.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 4141.50 | 4116.01 | 4105.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 10:15:00 | 4115.05 | 4115.82 | 4106.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 10:15:00 | 4115.05 | 4115.82 | 4106.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 4115.05 | 4115.82 | 4106.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:30:00 | 4102.85 | 4115.82 | 4106.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 4119.75 | 4124.54 | 4113.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 4119.75 | 4124.54 | 4113.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 4113.85 | 4122.40 | 4113.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 4136.55 | 4121.31 | 4113.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 09:15:00 | 4091.20 | 4115.29 | 4111.55 | SL hit (close<static) qty=1.00 sl=4105.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 4107.15 | 4129.24 | 4130.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 14:15:00 | 4090.45 | 4113.77 | 4120.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 4110.65 | 4107.73 | 4114.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 12:00:00 | 4110.65 | 4107.73 | 4114.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 4106.55 | 4107.49 | 4114.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:45:00 | 4110.30 | 4107.49 | 4114.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 4097.75 | 4105.54 | 4112.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 4109.65 | 4105.54 | 4112.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 4103.75 | 4105.19 | 4111.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 4103.75 | 4105.19 | 4111.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 4098.00 | 4103.75 | 4110.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 4125.20 | 4103.75 | 4110.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 4139.60 | 4110.92 | 4113.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 4130.80 | 4110.92 | 4113.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 4132.00 | 4115.14 | 4114.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 14:15:00 | 4156.60 | 4131.38 | 4123.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 4137.45 | 4157.98 | 4140.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 11:15:00 | 4137.45 | 4157.98 | 4140.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 4137.45 | 4157.98 | 4140.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 4137.45 | 4157.98 | 4140.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 4133.80 | 4153.15 | 4140.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 4133.80 | 4153.15 | 4140.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 4199.60 | 4222.36 | 4204.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 4199.60 | 4222.36 | 4204.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 4190.00 | 4215.89 | 4203.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 4178.00 | 4215.89 | 4203.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 4200.05 | 4208.19 | 4201.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:15:00 | 4180.00 | 4208.19 | 4201.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 4173.85 | 4201.32 | 4199.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 4173.85 | 4201.32 | 4199.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 4163.15 | 4193.69 | 4195.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 4152.20 | 4180.33 | 4189.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 4169.55 | 4167.17 | 4179.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 12:00:00 | 4169.55 | 4167.17 | 4179.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 4203.25 | 4136.25 | 4155.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 4185.45 | 4136.25 | 4155.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 4252.00 | 4159.40 | 4164.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 4248.45 | 4159.40 | 4164.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 4206.80 | 4168.88 | 4167.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 14:15:00 | 4259.05 | 4203.54 | 4185.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 4366.35 | 4384.02 | 4313.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 10:00:00 | 4366.35 | 4384.02 | 4313.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 4335.10 | 4369.71 | 4324.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 4335.10 | 4369.71 | 4324.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 4299.40 | 4348.86 | 4328.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 4285.40 | 4348.86 | 4328.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 4286.05 | 4336.30 | 4324.69 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 4290.00 | 4313.10 | 4315.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 15:15:00 | 4276.00 | 4301.79 | 4309.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 4310.80 | 4303.59 | 4309.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 4310.80 | 4303.59 | 4309.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 4310.80 | 4303.59 | 4309.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 4251.55 | 4305.98 | 4308.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:45:00 | 4263.50 | 4291.70 | 4301.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 4261.80 | 4278.25 | 4290.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 4271.30 | 4266.50 | 4282.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 4282.15 | 4259.22 | 4269.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 4282.15 | 4259.22 | 4269.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 4266.00 | 4260.58 | 4269.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 4280.55 | 4260.58 | 4269.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 4273.40 | 4263.14 | 4269.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:15:00 | 4287.55 | 4263.14 | 4269.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 4308.30 | 4272.17 | 4273.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 4308.30 | 4272.17 | 4273.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-09 13:15:00 | 4318.80 | 4281.50 | 4277.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 4318.80 | 4281.50 | 4277.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 14:15:00 | 4351.35 | 4295.47 | 4284.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 4291.75 | 4303.45 | 4290.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 4291.75 | 4303.45 | 4290.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 4291.75 | 4303.45 | 4290.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 4291.75 | 4303.45 | 4290.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 4297.40 | 4302.24 | 4290.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 4313.55 | 4302.24 | 4290.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 4300.65 | 4301.92 | 4291.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:30:00 | 4320.40 | 4301.34 | 4292.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 4314.00 | 4302.87 | 4295.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 4283.70 | 4295.13 | 4293.30 | SL hit (close<static) qty=1.00 sl=4288.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 4279.90 | 4291.12 | 4291.74 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 4298.90 | 4292.67 | 4292.39 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 15:15:00 | 4287.00 | 4291.54 | 4291.90 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 4345.15 | 4302.40 | 4296.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 4536.00 | 4366.69 | 4331.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 4469.25 | 4490.68 | 4439.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:45:00 | 4480.45 | 4490.68 | 4439.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 4437.05 | 4479.96 | 4439.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:30:00 | 4433.35 | 4479.96 | 4439.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 4454.00 | 4474.77 | 4440.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:15:00 | 4465.80 | 4474.77 | 4440.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 4465.80 | 4472.97 | 4443.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 4370.55 | 4472.97 | 4443.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 4354.90 | 4449.36 | 4435.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 4349.35 | 4449.36 | 4435.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 4304.60 | 4420.41 | 4423.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 4298.25 | 4372.11 | 4397.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 4295.80 | 4290.67 | 4330.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 4295.80 | 4290.67 | 4330.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 4295.80 | 4290.67 | 4330.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 4332.60 | 4290.67 | 4330.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 4260.25 | 4283.57 | 4307.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 4171.00 | 4277.18 | 4300.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 4231.10 | 4272.89 | 4293.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:30:00 | 4234.80 | 4270.09 | 4284.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:00:00 | 4243.35 | 4260.93 | 4276.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 4268.35 | 4251.90 | 4262.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 4268.35 | 4251.90 | 4262.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 4261.00 | 4253.72 | 4262.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 4243.85 | 4253.72 | 4262.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 4242.65 | 4251.50 | 4260.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-29 10:15:00 | 4320.00 | 4267.06 | 4261.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 4320.00 | 4267.06 | 4261.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 15:15:00 | 4341.00 | 4303.51 | 4283.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 4377.00 | 4389.22 | 4361.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 4377.00 | 4389.22 | 4361.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 4390.00 | 4391.70 | 4369.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 4390.00 | 4391.70 | 4369.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 4364.30 | 4386.22 | 4368.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 4364.30 | 4386.22 | 4368.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 4332.30 | 4375.44 | 4365.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 4332.95 | 4375.44 | 4365.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 4336.00 | 4359.34 | 4359.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 4294.00 | 4346.27 | 4353.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 13:15:00 | 4340.00 | 4337.61 | 4346.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 15:00:00 | 4324.05 | 4334.89 | 4344.40 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4278.95 | 4261.91 | 4291.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 4213.25 | 4241.55 | 4272.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 4262.05 | 4226.83 | 4256.05 | SL hit (close>ema400) qty=1.00 sl=4256.05 alert=retest1 |

### Cycle 20 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 4318.00 | 4273.15 | 4269.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 4331.45 | 4284.81 | 4275.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 4404.85 | 4408.27 | 4370.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 15:00:00 | 4404.85 | 4408.27 | 4370.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 4432.40 | 4414.97 | 4380.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:30:00 | 4389.85 | 4414.97 | 4380.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 4423.70 | 4501.82 | 4471.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 4423.70 | 4501.82 | 4471.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 4414.45 | 4484.35 | 4466.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 4414.45 | 4484.35 | 4466.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 4441.00 | 4457.94 | 4457.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 4493.15 | 4457.94 | 4457.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 13:15:00 | 4535.85 | 4561.06 | 4564.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 4535.85 | 4561.06 | 4564.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 14:15:00 | 4521.80 | 4553.21 | 4560.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 12:15:00 | 4544.95 | 4538.04 | 4548.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 12:15:00 | 4544.95 | 4538.04 | 4548.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 4544.95 | 4538.04 | 4548.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 4544.95 | 4538.04 | 4548.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 4531.00 | 4536.63 | 4547.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:30:00 | 4546.20 | 4536.63 | 4547.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 4540.95 | 4537.49 | 4546.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 4540.95 | 4537.49 | 4546.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 4531.40 | 4536.27 | 4545.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 4525.00 | 4536.27 | 4545.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 4528.65 | 4534.75 | 4543.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 13:45:00 | 4490.05 | 4522.77 | 4535.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 15:00:00 | 4483.25 | 4514.87 | 4530.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 4598.45 | 4532.41 | 4535.72 | SL hit (close>static) qty=1.00 sl=4572.40 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 4576.55 | 4541.24 | 4539.43 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 4519.85 | 4535.84 | 4537.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 4500.00 | 4522.21 | 4529.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 11:15:00 | 4532.40 | 4522.20 | 4528.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 11:15:00 | 4532.40 | 4522.20 | 4528.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 4532.40 | 4522.20 | 4528.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:00:00 | 4508.50 | 4524.18 | 4527.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:30:00 | 4500.95 | 4519.00 | 4524.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:30:00 | 4502.00 | 4504.96 | 4510.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:45:00 | 4510.00 | 4505.71 | 4509.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 4539.90 | 4503.27 | 4506.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:00:00 | 4539.90 | 4503.27 | 4506.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 4567.95 | 4516.21 | 4512.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 4567.95 | 4516.21 | 4512.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 4597.95 | 4546.13 | 4530.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 12:15:00 | 4588.95 | 4604.24 | 4580.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 13:00:00 | 4588.95 | 4604.24 | 4580.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 4592.60 | 4601.91 | 4581.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:30:00 | 4580.00 | 4601.91 | 4581.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 4589.85 | 4608.16 | 4594.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:00:00 | 4589.85 | 4608.16 | 4594.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 4600.00 | 4606.53 | 4594.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 4726.80 | 4600.24 | 4593.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:30:00 | 4626.00 | 4613.06 | 4603.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 4539.60 | 4600.17 | 4600.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 4539.60 | 4600.17 | 4600.42 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 4609.45 | 4588.60 | 4587.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 4726.50 | 4621.18 | 4603.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 4645.20 | 4655.68 | 4631.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 4645.20 | 4655.68 | 4631.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 4645.20 | 4655.68 | 4631.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 4645.20 | 4655.68 | 4631.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 4637.90 | 4651.05 | 4633.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 4637.90 | 4651.05 | 4633.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 4652.95 | 4651.43 | 4635.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 4646.75 | 4651.43 | 4635.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 4639.45 | 4649.04 | 4635.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 4644.95 | 4649.04 | 4635.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 4613.90 | 4642.01 | 4633.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 4613.90 | 4642.01 | 4633.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 4612.00 | 4636.01 | 4631.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:45:00 | 4612.55 | 4636.01 | 4631.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 4639.20 | 4646.94 | 4638.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 4634.55 | 4646.94 | 4638.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 4667.45 | 4651.04 | 4641.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 4680.10 | 4651.04 | 4641.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:30:00 | 4684.95 | 4675.99 | 4660.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:00:00 | 4722.45 | 4679.27 | 4666.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 4679.50 | 4707.94 | 4708.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 4679.50 | 4707.94 | 4708.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 4641.85 | 4689.51 | 4699.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 4665.00 | 4664.16 | 4679.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:30:00 | 4664.85 | 4664.16 | 4679.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 4657.30 | 4662.79 | 4677.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 4664.55 | 4662.79 | 4677.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 4692.10 | 4663.00 | 4671.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 4686.10 | 4663.00 | 4671.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 4680.00 | 4666.40 | 4672.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:30:00 | 4646.40 | 4661.12 | 4669.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 4649.05 | 4585.58 | 4585.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 4649.05 | 4585.58 | 4585.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 4664.05 | 4628.40 | 4609.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 4630.95 | 4635.37 | 4619.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 12:00:00 | 4630.95 | 4635.37 | 4619.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 4634.40 | 4635.17 | 4620.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 4634.40 | 4635.17 | 4620.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 4641.45 | 4638.80 | 4624.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:30:00 | 4634.85 | 4638.80 | 4624.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 4582.35 | 4628.90 | 4622.93 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 4497.30 | 4602.58 | 4611.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 4445.50 | 4514.51 | 4557.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 4344.00 | 4331.54 | 4395.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:15:00 | 4408.70 | 4331.54 | 4395.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 4411.30 | 4347.49 | 4396.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:15:00 | 4425.30 | 4347.49 | 4396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 4417.00 | 4361.39 | 4398.64 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 4490.00 | 4426.75 | 4421.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 4578.85 | 4457.17 | 4435.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 4495.65 | 4502.85 | 4476.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 4495.65 | 4502.85 | 4476.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 4476.60 | 4498.76 | 4479.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 4476.60 | 4498.76 | 4479.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 4490.95 | 4497.20 | 4480.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:45:00 | 4475.15 | 4497.20 | 4480.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 4495.00 | 4496.76 | 4481.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:30:00 | 4523.15 | 4510.93 | 4490.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-17 09:15:00 | 4975.47 | 4764.47 | 4712.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 4691.90 | 4747.21 | 4748.84 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 4847.45 | 4751.91 | 4748.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 10:15:00 | 4927.40 | 4787.01 | 4764.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 4846.50 | 4872.47 | 4835.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 11:15:00 | 4846.50 | 4872.47 | 4835.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 4846.50 | 4872.47 | 4835.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 12:00:00 | 4846.50 | 4872.47 | 4835.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 4839.05 | 4866.65 | 4847.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 11:15:00 | 4888.65 | 4857.68 | 4845.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-30 09:15:00 | 5377.52 | 5220.14 | 5156.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 14:15:00 | 5346.70 | 5390.02 | 5391.68 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 5480.00 | 5400.81 | 5395.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 5549.60 | 5450.62 | 5420.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 5483.30 | 5487.05 | 5460.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:45:00 | 5491.95 | 5487.05 | 5460.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 5555.30 | 5510.01 | 5480.96 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 5449.85 | 5516.96 | 5518.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 5258.05 | 5456.71 | 5490.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 5336.05 | 5310.93 | 5373.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 5336.05 | 5310.93 | 5373.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 5216.80 | 5240.24 | 5280.06 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 10:15:00 | 5439.45 | 5312.97 | 5296.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 11:15:00 | 5536.30 | 5357.64 | 5317.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 11:15:00 | 5502.05 | 5522.16 | 5441.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 11:45:00 | 5506.95 | 5522.16 | 5441.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 5455.35 | 5503.17 | 5458.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 5521.40 | 5503.17 | 5458.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:00:00 | 5499.25 | 5502.38 | 5462.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 5255.55 | 5449.34 | 5455.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 5255.55 | 5449.34 | 5455.55 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 5450.30 | 5353.45 | 5340.46 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 5275.05 | 5331.81 | 5338.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 14:15:00 | 5274.80 | 5313.43 | 5328.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 5304.55 | 5272.92 | 5292.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 5304.55 | 5272.92 | 5292.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 5304.55 | 5272.92 | 5292.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 5313.05 | 5272.92 | 5292.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 5341.80 | 5286.70 | 5297.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 5341.80 | 5286.70 | 5297.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 11:15:00 | 5380.00 | 5305.36 | 5304.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 5386.45 | 5330.27 | 5317.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 14:15:00 | 5358.60 | 5360.99 | 5341.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 5358.60 | 5360.99 | 5341.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 5358.60 | 5360.99 | 5341.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 5353.80 | 5360.99 | 5341.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 5360.00 | 5360.80 | 5343.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 5342.70 | 5357.18 | 5343.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 5345.00 | 5354.74 | 5343.27 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 5317.00 | 5340.15 | 5340.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 5281.00 | 5328.32 | 5335.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 5328.25 | 5307.22 | 5317.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 5328.25 | 5307.22 | 5317.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 5328.25 | 5307.22 | 5317.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 5328.25 | 5307.22 | 5317.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 5343.30 | 5314.44 | 5319.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:45:00 | 5352.65 | 5314.44 | 5319.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 12:15:00 | 5392.35 | 5333.31 | 5327.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 13:15:00 | 5422.00 | 5351.05 | 5336.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 5796.55 | 5803.77 | 5738.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:45:00 | 5807.60 | 5803.77 | 5738.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 5768.90 | 5792.00 | 5753.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 11:00:00 | 5840.20 | 5801.64 | 5760.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 5826.00 | 5808.73 | 5798.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:45:00 | 5826.45 | 5824.04 | 5806.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 5730.80 | 5818.23 | 5811.13 | SL hit (close<static) qty=1.00 sl=5740.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 5566.00 | 5767.78 | 5788.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 09:15:00 | 5515.00 | 5587.50 | 5636.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 5584.10 | 5570.17 | 5618.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 5584.10 | 5570.17 | 5618.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 5634.95 | 5589.20 | 5610.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 5634.95 | 5589.20 | 5610.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 5635.20 | 5598.40 | 5612.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:30:00 | 5644.70 | 5598.40 | 5612.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 5739.00 | 5641.58 | 5630.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 5993.25 | 5712.78 | 5664.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 6267.60 | 6351.28 | 6197.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:30:00 | 6256.00 | 6351.28 | 6197.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 6194.70 | 6271.57 | 6212.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 6194.70 | 6271.57 | 6212.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 6155.00 | 6248.26 | 6207.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 6265.00 | 6248.26 | 6207.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 6226.00 | 6241.68 | 6211.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:15:00 | 6223.10 | 6241.68 | 6211.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 6222.90 | 6237.93 | 6212.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 6207.85 | 6237.93 | 6212.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 6200.00 | 6230.34 | 6211.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:45:00 | 6200.00 | 6230.34 | 6211.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 6185.30 | 6221.33 | 6208.87 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 6114.85 | 6200.04 | 6200.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 6100.00 | 6180.03 | 6191.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 6244.15 | 6192.85 | 6196.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 6244.15 | 6192.85 | 6196.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 6244.15 | 6192.85 | 6196.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 6244.15 | 6192.85 | 6196.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 6106.35 | 6175.55 | 6187.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:30:00 | 6086.65 | 6149.01 | 6174.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:15:00 | 6095.80 | 6143.59 | 6169.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:15:00 | 5791.01 | 5923.03 | 6005.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:15:00 | 5782.32 | 5897.43 | 5986.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 5772.95 | 5735.21 | 5801.84 | SL hit (close>ema200) qty=0.50 sl=5735.21 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-13 12:15:00 | 5878.10 | 5798.64 | 5797.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 09:15:00 | 5962.90 | 5831.67 | 5813.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 14:15:00 | 5874.00 | 5876.86 | 5847.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-14 15:00:00 | 5874.00 | 5876.86 | 5847.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 5882.85 | 5880.40 | 5854.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:30:00 | 5898.10 | 5880.40 | 5854.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 5860.85 | 5877.23 | 5857.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 5860.85 | 5877.23 | 5857.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 5797.00 | 5861.18 | 5852.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 5797.00 | 5861.18 | 5852.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 5795.40 | 5848.03 | 5847.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:30:00 | 5786.05 | 5848.03 | 5847.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 5775.70 | 5833.56 | 5840.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 09:15:00 | 5754.80 | 5809.24 | 5827.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 5599.80 | 5586.64 | 5646.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 12:45:00 | 5584.30 | 5586.64 | 5646.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 5395.65 | 5437.42 | 5510.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 5489.10 | 5437.42 | 5510.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 5308.00 | 5325.33 | 5376.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:45:00 | 5259.00 | 5316.78 | 5354.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 5178.25 | 5325.42 | 5355.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 4996.05 | 5090.81 | 5192.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 4919.34 | 5090.81 | 5192.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 5120.00 | 5051.64 | 5136.18 | SL hit (close>ema200) qty=0.50 sl=5051.64 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 5285.25 | 5179.92 | 5167.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 5433.40 | 5240.70 | 5198.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 5233.05 | 5274.64 | 5231.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 11:15:00 | 5233.05 | 5274.64 | 5231.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 5233.05 | 5274.64 | 5231.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:45:00 | 5227.70 | 5274.64 | 5231.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 5271.40 | 5273.99 | 5235.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 5313.05 | 5272.27 | 5244.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 5291.65 | 5376.73 | 5385.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 09:15:00 | 5291.65 | 5376.73 | 5385.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 5201.70 | 5296.08 | 5329.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 5272.45 | 5242.96 | 5280.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 12:15:00 | 5272.45 | 5242.96 | 5280.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 5272.45 | 5242.96 | 5280.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 5272.45 | 5242.96 | 5280.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 5239.85 | 5242.34 | 5276.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 5274.95 | 5242.34 | 5276.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 5190.80 | 5232.03 | 5263.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:30:00 | 5177.15 | 5218.28 | 5254.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-11 09:15:00 | 4659.43 | 5081.33 | 5167.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 15:15:00 | 5395.00 | 5181.16 | 5174.61 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 5082.45 | 5161.42 | 5166.24 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 10:15:00 | 5214.95 | 5172.12 | 5170.66 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 12:15:00 | 5136.65 | 5168.47 | 5169.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 13:15:00 | 5072.90 | 5149.36 | 5160.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 11:15:00 | 5136.15 | 5110.38 | 5132.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 11:15:00 | 5136.15 | 5110.38 | 5132.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 5136.15 | 5110.38 | 5132.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:30:00 | 5183.50 | 5110.38 | 5132.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 5179.70 | 5124.24 | 5136.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 13:00:00 | 5179.70 | 5124.24 | 5136.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 5164.80 | 5132.35 | 5139.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:00:00 | 5164.80 | 5132.35 | 5139.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 5103.60 | 5112.94 | 5127.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 5103.60 | 5112.94 | 5127.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 5102.05 | 5090.80 | 5108.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:45:00 | 5155.00 | 5090.80 | 5108.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 5080.00 | 5088.64 | 5106.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 4991.00 | 5088.64 | 5106.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 10:15:00 | 4741.45 | 4858.31 | 4927.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 4660.00 | 4652.29 | 4708.30 | SL hit (close>ema200) qty=0.50 sl=4652.29 alert=retest2 |

### Cycle 54 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 4527.30 | 4466.67 | 4463.82 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 10:15:00 | 4453.45 | 4462.75 | 4463.78 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 4527.75 | 4470.37 | 4464.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 13:15:00 | 4540.60 | 4498.00 | 4479.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 4532.85 | 4538.22 | 4509.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 4532.85 | 4538.22 | 4509.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 4501.10 | 4530.80 | 4508.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 4501.10 | 4530.80 | 4508.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 4504.30 | 4525.50 | 4508.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:30:00 | 4504.50 | 4525.50 | 4508.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 4487.35 | 4517.87 | 4506.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 4487.35 | 4517.87 | 4506.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 4480.00 | 4510.29 | 4504.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 4450.00 | 4510.29 | 4504.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 4436.70 | 4487.73 | 4494.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 4422.00 | 4460.49 | 4479.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 4391.25 | 4389.17 | 4421.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 4395.50 | 4389.17 | 4421.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 4345.40 | 4380.41 | 4414.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 4320.00 | 4363.56 | 4387.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:15:00 | 4327.05 | 4357.83 | 4382.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 4327.45 | 4351.76 | 4377.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 4395.45 | 4384.26 | 4383.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 4395.45 | 4384.26 | 4383.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 4496.00 | 4409.54 | 4395.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 11:15:00 | 4411.75 | 4411.96 | 4399.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 11:45:00 | 4411.40 | 4411.96 | 4399.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 4357.95 | 4401.16 | 4395.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 4357.95 | 4401.16 | 4395.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 13:15:00 | 4315.20 | 4383.97 | 4388.03 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 13:15:00 | 4400.40 | 4388.25 | 4387.75 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 15:15:00 | 4370.15 | 4385.04 | 4386.40 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 4431.50 | 4394.33 | 4390.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 4461.90 | 4407.84 | 4396.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 4405.00 | 4413.60 | 4404.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 4529.00 | 4413.60 | 4404.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 4399.70 | 4418.16 | 4409.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-21 11:15:00 | 4399.70 | 4418.16 | 4409.73 | SL hit (close<ema400) qty=1.00 sl=4409.73 alert=retest1 |

### Cycle 63 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 4196.25 | 4366.47 | 4387.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 15:15:00 | 4175.00 | 4328.18 | 4368.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 4188.00 | 4179.48 | 4224.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 09:30:00 | 4182.05 | 4179.48 | 4224.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 4159.80 | 4168.96 | 4197.07 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 4236.00 | 4184.93 | 4180.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 4251.50 | 4202.55 | 4189.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 4255.55 | 4264.03 | 4235.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 12:15:00 | 4185.05 | 4246.80 | 4234.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 4185.05 | 4246.80 | 4234.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 4185.05 | 4246.80 | 4234.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 4178.20 | 4233.08 | 4229.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 4181.00 | 4233.08 | 4229.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 4191.10 | 4224.68 | 4226.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4008.10 | 4175.98 | 4203.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 4138.10 | 4115.71 | 4156.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 4138.10 | 4115.71 | 4156.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4149.95 | 4124.84 | 4154.11 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 4233.95 | 4170.98 | 4168.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 14:15:00 | 4336.35 | 4204.05 | 4183.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 4187.00 | 4219.98 | 4195.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 4187.00 | 4219.98 | 4195.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 4187.00 | 4219.98 | 4195.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 4187.00 | 4219.98 | 4195.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 4202.35 | 4216.46 | 4196.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 11:15:00 | 4273.90 | 4203.36 | 4198.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 15:00:00 | 4219.90 | 4205.42 | 4200.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 12:15:00 | 4641.89 | 4582.75 | 4499.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 4610.00 | 4694.70 | 4695.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 4544.90 | 4643.57 | 4671.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 12:15:00 | 4447.00 | 4444.49 | 4493.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 12:15:00 | 4447.00 | 4444.49 | 4493.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 4447.00 | 4444.49 | 4493.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:45:00 | 4452.20 | 4444.49 | 4493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 4429.90 | 4441.57 | 4487.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:30:00 | 4413.60 | 4441.57 | 4487.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 4441.60 | 4441.58 | 4483.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:45:00 | 4479.50 | 4441.58 | 4483.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 4530.00 | 4461.09 | 4485.22 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 4735.50 | 4522.20 | 4509.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 4853.40 | 4709.13 | 4665.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 4836.10 | 4873.35 | 4796.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 11:00:00 | 4836.10 | 4873.35 | 4796.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 5079.60 | 5094.09 | 5062.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 5079.60 | 5094.09 | 5062.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 5063.00 | 5087.87 | 5062.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 5063.00 | 5087.87 | 5062.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 5049.80 | 5080.26 | 5060.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 5049.80 | 5080.26 | 5060.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 5020.20 | 5068.25 | 5057.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 5007.80 | 5068.25 | 5057.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 5023.50 | 5052.23 | 5051.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 5020.70 | 5052.23 | 5051.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 5019.10 | 5045.61 | 5048.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 14:15:00 | 5010.40 | 5033.80 | 5041.95 | Break + close below crossover candle low |

### Cycle 70 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 5159.70 | 5054.05 | 5049.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 5170.00 | 5077.24 | 5060.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 5095.30 | 5124.92 | 5098.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 5095.30 | 5124.92 | 5098.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 5095.30 | 5124.92 | 5098.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 5095.80 | 5124.92 | 5098.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 5066.50 | 5113.24 | 5095.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 5066.50 | 5113.24 | 5095.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 5045.30 | 5099.65 | 5091.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 5045.30 | 5099.65 | 5091.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5012.90 | 5073.04 | 5079.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 5003.90 | 5045.14 | 5062.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 5099.80 | 5053.52 | 5061.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 5099.80 | 5053.52 | 5061.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 5099.80 | 5053.52 | 5061.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 5099.80 | 5053.52 | 5061.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 5086.20 | 5060.06 | 5063.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 5072.40 | 5060.06 | 5063.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 5080.20 | 5064.09 | 5065.23 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 5092.80 | 5069.83 | 5067.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 5129.80 | 5107.64 | 5094.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 5124.30 | 5125.29 | 5110.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:00:00 | 5124.30 | 5125.29 | 5110.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 5165.00 | 5132.39 | 5115.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 5154.90 | 5132.39 | 5115.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 5150.50 | 5155.56 | 5134.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 5150.50 | 5155.56 | 5134.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 5249.20 | 5265.90 | 5246.16 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 5182.00 | 5227.63 | 5232.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 5162.50 | 5192.37 | 5208.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 5296.50 | 5201.75 | 5206.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 5296.50 | 5201.75 | 5206.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 5296.50 | 5201.75 | 5206.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 5274.00 | 5201.75 | 5206.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 5312.00 | 5223.80 | 5215.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 5415.00 | 5352.83 | 5314.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 5654.50 | 5683.18 | 5580.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 5654.50 | 5683.18 | 5580.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 5588.00 | 5643.45 | 5586.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 5588.00 | 5643.45 | 5586.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 5538.50 | 5622.46 | 5581.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 5538.50 | 5622.46 | 5581.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 5536.50 | 5605.27 | 5577.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 5536.50 | 5605.27 | 5577.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 5356.50 | 5546.67 | 5555.41 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 5673.50 | 5495.85 | 5475.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 5687.00 | 5604.72 | 5540.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 5600.50 | 5615.92 | 5557.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:00:00 | 5600.50 | 5615.92 | 5557.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 5556.50 | 5604.04 | 5557.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 5556.50 | 5604.04 | 5557.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 5560.00 | 5595.23 | 5557.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 5549.00 | 5595.23 | 5557.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 5486.50 | 5573.48 | 5550.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 5486.50 | 5573.48 | 5550.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 5465.50 | 5551.89 | 5543.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 5465.50 | 5551.89 | 5543.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5617.00 | 5616.80 | 5585.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 5595.00 | 5616.80 | 5585.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 5585.00 | 5613.86 | 5591.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 5592.50 | 5613.86 | 5591.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 5702.00 | 5631.49 | 5601.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 5735.00 | 5631.49 | 5601.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 5725.00 | 5688.36 | 5638.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 5500.00 | 5674.65 | 5646.44 | SL hit (close<static) qty=1.00 sl=5533.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 5948.00 | 5998.81 | 6000.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 5925.00 | 5982.77 | 5992.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 6008.00 | 5973.22 | 5985.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 6008.00 | 5973.22 | 5985.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 6008.00 | 5973.22 | 5985.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 6008.00 | 5973.22 | 5985.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 6057.00 | 5989.97 | 5992.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 6050.00 | 5989.97 | 5992.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 6056.50 | 6003.28 | 5998.11 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 5874.50 | 6001.06 | 6004.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 12:15:00 | 5847.00 | 5970.25 | 5990.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 5830.00 | 5813.49 | 5870.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 5830.00 | 5813.49 | 5870.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 5830.00 | 5813.49 | 5870.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 5769.50 | 5804.69 | 5861.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 5880.00 | 5832.12 | 5837.07 | SL hit (close>static) qty=1.00 sl=5877.50 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 5879.50 | 5841.59 | 5840.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 5892.00 | 5851.58 | 5845.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 5917.00 | 5921.77 | 5891.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:45:00 | 5918.50 | 5921.77 | 5891.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 5904.50 | 5918.32 | 5892.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 5904.50 | 5918.32 | 5892.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 5900.00 | 5914.65 | 5893.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 5900.00 | 5914.65 | 5893.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5959.00 | 6010.41 | 5988.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 5959.00 | 6010.41 | 5988.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 5983.50 | 6005.03 | 5988.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 5985.00 | 6005.03 | 5988.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 5983.00 | 6000.62 | 5987.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 6095.00 | 6000.62 | 5987.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 6008.00 | 6004.52 | 5993.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 5948.00 | 5993.22 | 5989.77 | SL hit (close<static) qty=1.00 sl=5961.50 alert=retest2 |

### Cycle 81 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 5940.00 | 5982.57 | 5985.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 5934.00 | 5972.86 | 5980.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 5980.00 | 5965.83 | 5975.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 5980.00 | 5965.83 | 5975.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 5980.00 | 5965.83 | 5975.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 5980.00 | 5965.83 | 5975.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5934.50 | 5959.56 | 5971.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 5901.50 | 5935.58 | 5953.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 5606.43 | 5664.33 | 5732.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 09:15:00 | 5311.35 | 5428.14 | 5544.46 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 5313.00 | 5283.41 | 5283.00 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 5197.00 | 5292.53 | 5297.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 5180.00 | 5224.98 | 5254.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 5212.00 | 5188.66 | 5214.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 12:00:00 | 5212.00 | 5188.66 | 5214.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 5210.00 | 5192.93 | 5214.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 5236.00 | 5192.93 | 5214.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 5202.50 | 5196.37 | 5212.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 5202.50 | 5196.37 | 5212.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 5229.50 | 5203.00 | 5213.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 5221.50 | 5203.00 | 5213.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 5205.00 | 5203.40 | 5212.90 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 5231.00 | 5216.79 | 5216.47 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 5185.00 | 5210.94 | 5213.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 5158.00 | 5200.35 | 5208.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 14:15:00 | 5206.00 | 5150.20 | 5165.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 5206.00 | 5150.20 | 5165.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 5206.00 | 5150.20 | 5165.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 5206.00 | 5150.20 | 5165.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 5207.50 | 5161.66 | 5169.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 5221.50 | 5161.66 | 5169.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 5217.00 | 5182.78 | 5178.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 5267.00 | 5199.63 | 5186.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 13:15:00 | 5302.50 | 5304.45 | 5262.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 14:00:00 | 5302.50 | 5304.45 | 5262.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 5363.00 | 5383.31 | 5355.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 5363.00 | 5383.31 | 5355.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 5345.00 | 5375.65 | 5354.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 5376.00 | 5375.65 | 5354.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 5329.50 | 5375.72 | 5375.68 | SL hit (close<static) qty=1.00 sl=5341.50 alert=retest2 |

### Cycle 87 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 5309.50 | 5362.47 | 5369.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 5273.50 | 5335.48 | 5355.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 5018.50 | 4992.00 | 5074.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 5018.50 | 4992.00 | 5074.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 5038.20 | 5005.78 | 5037.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 5039.80 | 5005.78 | 5037.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 5031.40 | 5010.90 | 5036.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 5031.40 | 5010.90 | 5036.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 5030.90 | 5014.90 | 5035.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 5034.70 | 5014.90 | 5035.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 5088.00 | 5029.52 | 5040.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 5088.00 | 5029.52 | 5040.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 5058.90 | 5035.40 | 5042.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 5026.30 | 5035.40 | 5042.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 5076.30 | 5051.39 | 5048.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 5076.30 | 5051.39 | 5048.92 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 5012.40 | 5044.20 | 5047.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 5001.00 | 5029.02 | 5039.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 5045.50 | 5032.31 | 5040.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 5045.50 | 5032.31 | 5040.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 5045.50 | 5032.31 | 5040.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 5045.50 | 5032.31 | 5040.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 5044.10 | 5034.67 | 5040.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 5028.00 | 5034.67 | 5040.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 5031.50 | 5034.04 | 5039.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 5011.00 | 5028.65 | 5035.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 5019.00 | 5019.68 | 5030.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:00:00 | 5011.40 | 5014.96 | 5026.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 5019.10 | 5018.82 | 5026.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 5016.90 | 5018.44 | 5026.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 5062.90 | 5031.89 | 5031.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 5062.90 | 5031.89 | 5031.08 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 5010.10 | 5027.53 | 5029.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 11:15:00 | 4983.90 | 5012.03 | 5020.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 4982.10 | 4981.21 | 4998.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 4982.10 | 4981.21 | 4998.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 4982.10 | 4981.21 | 4998.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 4982.10 | 4981.21 | 4998.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 4989.40 | 4982.67 | 4993.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 4989.40 | 4982.67 | 4993.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 4981.80 | 4982.50 | 4992.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 5010.30 | 4982.50 | 4992.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 5029.90 | 4991.98 | 4996.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 5046.40 | 4991.98 | 4996.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 5078.10 | 5009.20 | 5003.65 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 5009.80 | 5023.68 | 5024.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 5005.90 | 5017.70 | 5021.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 5012.50 | 5012.30 | 5017.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 5012.50 | 5012.30 | 5017.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 5020.40 | 5013.92 | 5017.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 5020.40 | 5013.92 | 5017.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 5013.80 | 5013.90 | 5017.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 5013.80 | 5013.90 | 5017.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 5000.00 | 5011.12 | 5015.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 5043.90 | 5011.12 | 5015.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 5028.00 | 5014.49 | 5016.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 5041.10 | 5014.49 | 5016.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 5016.00 | 5014.80 | 5016.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 4993.90 | 5011.13 | 5014.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:30:00 | 5007.70 | 5010.84 | 5014.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 5009.70 | 5010.84 | 5014.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 5046.20 | 5017.91 | 5017.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 5046.20 | 5017.91 | 5017.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 15:15:00 | 5051.90 | 5024.71 | 5020.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 5034.80 | 5034.85 | 5028.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 13:45:00 | 5039.90 | 5034.85 | 5028.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 5080.00 | 5043.88 | 5032.83 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 5018.60 | 5031.92 | 5032.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 5015.30 | 5028.60 | 5031.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 4818.00 | 4800.23 | 4846.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 12:00:00 | 4765.40 | 4793.26 | 4839.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 09:15:00 | 4527.13 | 4601.41 | 4663.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 4479.90 | 4464.31 | 4516.89 | SL hit (close>ema200) qty=0.50 sl=4464.31 alert=retest1 |

### Cycle 96 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 4632.90 | 4549.71 | 4540.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 4680.10 | 4613.10 | 4586.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 11:15:00 | 4654.20 | 4667.03 | 4628.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 12:00:00 | 4654.20 | 4667.03 | 4628.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 4629.70 | 4659.56 | 4628.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:00:00 | 4629.70 | 4659.56 | 4628.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 4624.30 | 4652.51 | 4628.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 4626.50 | 4652.51 | 4628.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 4633.40 | 4648.69 | 4628.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 4605.40 | 4648.69 | 4628.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 4636.10 | 4646.17 | 4629.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 4608.00 | 4646.17 | 4629.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 4598.70 | 4636.68 | 4626.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 4616.00 | 4636.68 | 4626.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 4623.30 | 4634.00 | 4626.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 4598.00 | 4634.00 | 4626.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 4630.00 | 4633.20 | 4626.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 4639.00 | 4631.42 | 4626.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 4635.80 | 4631.01 | 4627.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:00:00 | 4638.20 | 4633.21 | 4628.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 4655.30 | 4640.67 | 4632.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4612.60 | 4659.15 | 4648.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 4606.10 | 4642.52 | 4642.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 4606.10 | 4642.52 | 4642.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 4582.00 | 4624.25 | 4634.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 4591.60 | 4588.97 | 4607.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 13:15:00 | 4591.60 | 4588.97 | 4607.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 4591.60 | 4588.97 | 4607.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 4591.60 | 4588.97 | 4607.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 4585.00 | 4587.87 | 4603.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 4662.30 | 4587.87 | 4603.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 4736.20 | 4617.54 | 4615.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 4786.40 | 4729.46 | 4709.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 4948.10 | 5001.29 | 4971.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 4948.10 | 5001.29 | 4971.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 4948.10 | 5001.29 | 4971.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 4948.10 | 5001.29 | 4971.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 4958.70 | 4992.77 | 4970.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 4934.40 | 4992.77 | 4970.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 4985.00 | 4991.18 | 4977.00 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 4952.30 | 4969.04 | 4970.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 4930.10 | 4948.16 | 4954.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 4808.70 | 4800.90 | 4837.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 4808.70 | 4800.90 | 4837.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 4806.00 | 4804.59 | 4833.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 4735.70 | 4805.67 | 4831.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 10:00:00 | 4769.10 | 4798.36 | 4825.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 11:00:00 | 4757.00 | 4790.09 | 4819.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 4736.20 | 4680.50 | 4674.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 4736.20 | 4680.50 | 4674.93 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 4636.10 | 4680.90 | 4681.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 4628.50 | 4649.28 | 4659.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 4548.20 | 4545.97 | 4571.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:15:00 | 4544.00 | 4545.97 | 4571.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 4485.00 | 4530.32 | 4556.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 4463.70 | 4510.42 | 4539.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 4471.30 | 4509.48 | 4519.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:45:00 | 4446.40 | 4434.96 | 4454.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 4485.30 | 4465.18 | 4464.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 4485.30 | 4465.18 | 4464.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 14:15:00 | 4493.80 | 4470.91 | 4467.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 4442.10 | 4473.11 | 4470.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 4442.10 | 4473.11 | 4470.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 4442.10 | 4473.11 | 4470.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 4442.10 | 4473.11 | 4470.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 4431.10 | 4464.71 | 4466.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 4396.20 | 4451.01 | 4460.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 4432.60 | 4429.01 | 4444.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 4432.60 | 4429.01 | 4444.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 4432.60 | 4429.01 | 4444.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 4435.30 | 4429.01 | 4444.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 4435.80 | 4430.37 | 4444.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 4438.80 | 4430.37 | 4444.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 4464.90 | 4437.28 | 4446.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 4481.40 | 4437.28 | 4446.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 4439.90 | 4437.80 | 4445.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 4415.00 | 4435.62 | 4442.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 4427.00 | 4429.21 | 4430.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 4462.60 | 4435.89 | 4433.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 4462.60 | 4435.89 | 4433.11 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 4384.80 | 4424.21 | 4428.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 4358.30 | 4411.03 | 4422.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 4369.20 | 4361.84 | 4387.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 4369.20 | 4361.84 | 4387.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 4376.50 | 4364.77 | 4386.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 4374.00 | 4364.77 | 4386.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 4396.00 | 4371.02 | 4387.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 4396.00 | 4371.02 | 4387.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 4398.50 | 4376.51 | 4388.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 4439.50 | 4376.51 | 4388.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 4365.10 | 4357.57 | 4370.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 4324.50 | 4350.32 | 4365.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 4343.40 | 4350.65 | 4364.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 4376.10 | 4359.15 | 4365.43 | SL hit (close>static) qty=1.00 sl=4376.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 4389.70 | 4369.14 | 4369.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 4468.80 | 4389.07 | 4378.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 4453.70 | 4462.35 | 4428.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:30:00 | 4438.90 | 4462.35 | 4428.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 4450.60 | 4465.38 | 4449.12 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 4417.40 | 4440.90 | 4441.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 4391.00 | 4430.32 | 4436.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 4424.00 | 4420.75 | 4427.94 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:15:00 | 4366.40 | 4420.75 | 4427.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4354.70 | 4351.40 | 4377.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 4354.70 | 4351.40 | 4377.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 4389.60 | 4359.04 | 4378.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 4389.60 | 4359.04 | 4378.62 | SL hit (close>ema400) qty=1.00 sl=4378.62 alert=retest1 |

### Cycle 108 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 4307.00 | 4296.68 | 4295.84 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 4261.30 | 4289.61 | 4292.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 4213.80 | 4274.45 | 4285.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4290.90 | 4241.34 | 4258.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4290.90 | 4241.34 | 4258.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4290.90 | 4241.34 | 4258.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 4290.90 | 4241.34 | 4258.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 4307.00 | 4254.47 | 4263.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 4307.00 | 4254.47 | 4263.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 4298.60 | 4271.70 | 4270.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 4323.70 | 4287.99 | 4278.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 4662.20 | 4687.52 | 4613.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 4662.20 | 4687.52 | 4613.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 4715.80 | 4708.47 | 4661.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 4756.00 | 4715.42 | 4669.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:45:00 | 4751.00 | 4721.32 | 4686.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:15:00 | 4751.10 | 4722.40 | 4697.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:30:00 | 4761.40 | 4732.84 | 4707.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 4818.90 | 4804.14 | 4770.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 4855.90 | 4812.43 | 4780.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 4742.30 | 4793.66 | 4781.74 | SL hit (close<static) qty=1.00 sl=4769.50 alert=retest2 |

### Cycle 111 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 4696.00 | 4765.03 | 4770.23 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 4800.00 | 4770.38 | 4767.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 4800.80 | 4778.80 | 4772.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 4787.30 | 4791.92 | 4781.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 10:45:00 | 4785.50 | 4791.92 | 4781.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 4765.90 | 4786.72 | 4780.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 4765.90 | 4786.72 | 4780.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 4743.90 | 4778.15 | 4776.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 4743.90 | 4778.15 | 4776.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 4757.80 | 4774.08 | 4775.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 4589.80 | 4730.25 | 4754.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 4602.90 | 4495.56 | 4517.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 4602.90 | 4495.56 | 4517.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4602.90 | 4495.56 | 4517.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 4602.90 | 4495.56 | 4517.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 4620.10 | 4520.47 | 4527.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:30:00 | 4610.00 | 4520.47 | 4527.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 4607.40 | 4537.85 | 4534.46 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 4500.00 | 4544.30 | 4545.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 12:15:00 | 4494.10 | 4534.26 | 4540.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 4548.50 | 4530.37 | 4536.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 4548.50 | 4530.37 | 4536.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 4548.50 | 4530.37 | 4536.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 4548.50 | 4530.37 | 4536.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 4588.10 | 4541.91 | 4540.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 4653.60 | 4564.25 | 4551.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 4598.00 | 4667.38 | 4635.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 4598.00 | 4667.38 | 4635.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 4598.00 | 4667.38 | 4635.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 4561.90 | 4667.38 | 4635.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 4635.50 | 4661.00 | 4635.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 4684.70 | 4654.85 | 4636.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 4690.30 | 4661.94 | 4641.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:00:00 | 4689.70 | 4754.27 | 4736.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 4689.60 | 4723.33 | 4725.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 4689.60 | 4723.33 | 4725.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 4671.80 | 4709.11 | 4718.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 4689.30 | 4665.89 | 4687.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 4689.30 | 4665.89 | 4687.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4689.30 | 4665.89 | 4687.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 4690.00 | 4665.89 | 4687.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 4724.20 | 4677.55 | 4690.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 4724.20 | 4677.55 | 4690.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 4689.20 | 4679.88 | 4690.32 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 4757.20 | 4707.78 | 4701.45 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 12:15:00 | 4694.00 | 4699.12 | 4699.72 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 4714.60 | 4699.69 | 4699.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 4725.60 | 4704.87 | 4702.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 4678.00 | 4707.48 | 4704.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 14:15:00 | 4678.00 | 4707.48 | 4704.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 4678.00 | 4707.48 | 4704.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:45:00 | 4680.00 | 4707.48 | 4704.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 4673.30 | 4700.65 | 4701.83 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 4706.90 | 4703.26 | 4702.88 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 4690.00 | 4700.61 | 4701.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 4638.30 | 4686.36 | 4694.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 4666.80 | 4663.69 | 4680.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:30:00 | 4670.00 | 4663.69 | 4680.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 4675.60 | 4666.07 | 4680.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 4429.80 | 4665.68 | 4677.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 4637.60 | 4536.11 | 4572.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 4610.00 | 4584.54 | 4583.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 4610.00 | 4584.54 | 4583.26 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 4555.00 | 4581.48 | 4582.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 4512.90 | 4553.89 | 4566.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 15:15:00 | 4549.00 | 4530.79 | 4546.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 4549.00 | 4530.79 | 4546.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 4549.00 | 4530.79 | 4546.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 4594.70 | 4530.79 | 4546.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 4567.10 | 4538.05 | 4548.20 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 4582.60 | 4555.42 | 4554.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 4605.30 | 4566.40 | 4559.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 15:15:00 | 4594.80 | 4595.30 | 4580.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:15:00 | 4524.90 | 4595.30 | 4580.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 4514.00 | 4579.04 | 4574.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 4496.20 | 4579.04 | 4574.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 4522.30 | 4567.69 | 4569.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4465.80 | 4536.16 | 4554.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 4523.60 | 4520.82 | 4539.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 4523.60 | 4520.82 | 4539.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 4495.90 | 4512.46 | 4530.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 4511.00 | 4512.46 | 4530.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 4536.50 | 4517.27 | 4531.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 4536.50 | 4517.27 | 4531.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 4539.60 | 4521.74 | 4532.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 4539.60 | 4521.74 | 4532.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 4536.50 | 4524.69 | 4532.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 4497.20 | 4524.69 | 4532.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4272.34 | 4416.79 | 4452.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 4437.20 | 4416.79 | 4452.04 | SL hit (close>static) qty=0.50 sl=4416.79 alert=retest2 |

### Cycle 128 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 4304.60 | 4239.27 | 4237.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 4318.70 | 4255.16 | 4245.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4217.80 | 4276.38 | 4262.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 4217.80 | 4276.38 | 4262.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 4217.80 | 4276.38 | 4262.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 4212.60 | 4276.38 | 4262.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 4262.40 | 4273.58 | 4262.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 4296.00 | 4278.06 | 4265.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 4245.00 | 4256.16 | 4257.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 4245.00 | 4256.16 | 4257.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 4217.00 | 4248.33 | 4253.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 4023.80 | 4018.86 | 4055.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 11:15:00 | 4026.70 | 4018.86 | 4055.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 4060.10 | 4027.11 | 4055.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 4060.10 | 4027.11 | 4055.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 4098.50 | 4041.39 | 4059.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 4098.50 | 4041.39 | 4059.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 4047.20 | 4042.55 | 4058.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 4033.20 | 4040.68 | 4056.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 4019.30 | 4039.94 | 4054.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:30:00 | 4031.20 | 4013.40 | 4026.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 4050.50 | 3995.25 | 3989.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 4050.50 | 3995.25 | 3989.61 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 3894.80 | 3970.64 | 3980.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 3849.00 | 3924.37 | 3953.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3847.90 | 3804.30 | 3855.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3847.90 | 3804.30 | 3855.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3847.90 | 3804.30 | 3855.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 3751.50 | 3816.78 | 3840.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 14:30:00 | 3775.20 | 3764.61 | 3773.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 3772.80 | 3774.59 | 3777.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 10:45:00 | 3770.30 | 3775.11 | 3777.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 3779.00 | 3775.89 | 3777.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 12:15:00 | 3770.40 | 3775.89 | 3777.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3937.20 | 3805.10 | 3789.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 3937.20 | 3805.10 | 3789.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 4010.70 | 3915.85 | 3867.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 3924.70 | 3929.36 | 3887.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 15:00:00 | 3924.70 | 3929.36 | 3887.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4006.10 | 4039.44 | 3982.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4058.80 | 4043.47 | 3989.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 4063.50 | 4046.73 | 4004.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 4085.00 | 4042.98 | 4009.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 4319.80 | 4344.21 | 4346.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 4319.80 | 4344.21 | 4346.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 4274.70 | 4324.83 | 4336.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 14:15:00 | 4295.00 | 4286.11 | 4304.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 4295.00 | 4286.11 | 4304.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 4290.00 | 4286.89 | 4303.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 4298.10 | 4286.89 | 4303.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4294.00 | 4288.31 | 4302.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 4242.00 | 4282.85 | 4298.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 4247.50 | 4275.78 | 4293.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 4258.70 | 4270.03 | 4287.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 4354.80 | 4300.32 | 4295.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 4354.80 | 4300.32 | 4295.49 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 4266.60 | 4291.86 | 4295.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 4246.50 | 4278.70 | 4288.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 4181.40 | 4165.01 | 4186.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 4181.40 | 4165.01 | 4186.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 4181.40 | 4165.01 | 4186.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 4195.50 | 4165.01 | 4186.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 4190.00 | 4170.01 | 4186.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 4147.10 | 4171.02 | 4182.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 4167.20 | 4170.81 | 4179.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 4201.90 | 4179.30 | 4181.65 | SL hit (close>static) qty=1.00 sl=4195.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:15:00 | 4398.45 | 2024-05-16 09:15:00 | 4340.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-17 11:45:00 | 4379.00 | 2024-05-23 13:15:00 | 4380.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-05-17 12:30:00 | 4381.70 | 2024-05-23 13:15:00 | 4380.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-05-17 14:00:00 | 4378.90 | 2024-05-23 14:15:00 | 4345.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-05-18 09:15:00 | 4379.95 | 2024-05-23 14:15:00 | 4345.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-05-22 14:15:00 | 4434.30 | 2024-05-23 14:15:00 | 4345.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-05-23 12:30:00 | 4420.60 | 2024-05-23 14:15:00 | 4345.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-06-03 10:15:00 | 4048.90 | 2024-06-05 12:15:00 | 4186.00 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-06-04 15:15:00 | 4064.10 | 2024-06-05 12:15:00 | 4186.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-06-05 09:30:00 | 4050.00 | 2024-06-05 12:15:00 | 4186.00 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-06-12 09:15:00 | 4136.55 | 2024-06-12 09:15:00 | 4091.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-06-12 11:45:00 | 4134.20 | 2024-06-14 13:15:00 | 4107.15 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-06-12 15:00:00 | 4136.45 | 2024-06-14 13:15:00 | 4107.15 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-06-13 09:15:00 | 4143.25 | 2024-06-14 13:15:00 | 4107.15 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-07-05 09:15:00 | 4251.55 | 2024-07-09 13:15:00 | 4318.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-07-05 10:45:00 | 4263.50 | 2024-07-09 13:15:00 | 4318.80 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-05 15:15:00 | 4261.80 | 2024-07-09 13:15:00 | 4318.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-08 10:30:00 | 4271.30 | 2024-07-09 13:15:00 | 4318.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-07-10 12:30:00 | 4320.40 | 2024-07-11 11:15:00 | 4283.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-07-11 09:15:00 | 4314.00 | 2024-07-11 11:15:00 | 4283.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-07-23 12:15:00 | 4171.00 | 2024-07-29 10:15:00 | 4320.00 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-07-23 13:30:00 | 4231.10 | 2024-07-29 10:15:00 | 4320.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-07-24 11:30:00 | 4234.80 | 2024-07-29 10:15:00 | 4320.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-07-24 15:00:00 | 4243.35 | 2024-07-29 10:15:00 | 4320.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest1 | 2024-08-02 15:00:00 | 4324.05 | 2024-08-07 09:15:00 | 4262.05 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2024-08-06 13:30:00 | 4213.25 | 2024-08-07 10:15:00 | 4331.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-08-16 09:15:00 | 4493.15 | 2024-08-22 13:15:00 | 4535.85 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-08-26 13:45:00 | 4490.05 | 2024-08-27 09:15:00 | 4598.45 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-08-26 15:00:00 | 4483.25 | 2024-08-27 09:15:00 | 4598.45 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-08-29 10:00:00 | 4508.50 | 2024-09-02 10:15:00 | 4567.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-08-29 10:30:00 | 4500.95 | 2024-09-02 10:15:00 | 4567.95 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-08-30 11:30:00 | 4502.00 | 2024-09-02 10:15:00 | 4567.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-08-30 13:45:00 | 4510.00 | 2024-09-02 10:15:00 | 4567.95 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-09-06 09:15:00 | 4726.80 | 2024-09-09 09:15:00 | 4539.60 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-09-06 13:30:00 | 4626.00 | 2024-09-09 09:15:00 | 4539.60 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-09-13 11:15:00 | 4680.10 | 2024-09-19 09:15:00 | 4679.50 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-09-16 09:30:00 | 4684.95 | 2024-09-19 09:15:00 | 4679.50 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-09-16 14:00:00 | 4722.45 | 2024-09-19 09:15:00 | 4679.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-09-23 11:30:00 | 4646.40 | 2024-09-30 09:15:00 | 4649.05 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-10-11 09:30:00 | 4523.15 | 2024-10-17 09:15:00 | 4975.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 11:15:00 | 4888.65 | 2024-10-30 09:15:00 | 5377.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-25 09:15:00 | 5521.40 | 2024-11-25 14:15:00 | 5255.55 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-11-25 10:00:00 | 5499.25 | 2024-11-25 14:15:00 | 5255.55 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2024-12-18 11:00:00 | 5840.20 | 2024-12-20 14:15:00 | 5730.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-12-20 09:15:00 | 5826.00 | 2024-12-20 14:15:00 | 5730.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-12-20 10:45:00 | 5826.45 | 2024-12-20 14:15:00 | 5730.80 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-01-06 11:30:00 | 6086.65 | 2025-01-08 09:15:00 | 5791.01 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-01-06 13:15:00 | 6095.80 | 2025-01-08 10:15:00 | 5782.32 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-01-06 11:30:00 | 6086.65 | 2025-01-10 10:15:00 | 5772.95 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2025-01-06 13:15:00 | 6095.80 | 2025-01-10 10:15:00 | 5772.95 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2025-01-24 14:45:00 | 5259.00 | 2025-01-28 09:15:00 | 4996.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 5178.25 | 2025-01-28 09:15:00 | 4919.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:45:00 | 5259.00 | 2025-01-28 13:15:00 | 5120.00 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-01-27 09:15:00 | 5178.25 | 2025-01-28 13:15:00 | 5120.00 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2025-01-29 11:30:00 | 5267.75 | 2025-01-29 12:15:00 | 5285.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-01-31 09:15:00 | 5313.05 | 2025-02-05 09:15:00 | 5291.65 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-02-10 10:30:00 | 5177.15 | 2025-02-11 09:15:00 | 4659.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 4991.00 | 2025-02-19 10:15:00 | 4741.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 4991.00 | 2025-02-21 14:15:00 | 4660.00 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2025-03-13 10:45:00 | 4320.00 | 2025-03-17 13:15:00 | 4395.45 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-03-13 12:15:00 | 4327.05 | 2025-03-17 13:15:00 | 4395.45 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-03-13 13:00:00 | 4327.45 | 2025-03-17 13:15:00 | 4395.45 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest1 | 2025-03-21 09:15:00 | 4529.00 | 2025-03-21 11:15:00 | 4399.70 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-04-11 11:15:00 | 4273.90 | 2025-04-21 12:15:00 | 4641.89 | TARGET_HIT | 1.00 | 8.61% |
| BUY | retest2 | 2025-04-11 15:00:00 | 4219.90 | 2025-04-21 14:15:00 | 4701.29 | TARGET_HIT | 1.00 | 11.41% |
| BUY | retest2 | 2025-06-20 10:15:00 | 5735.00 | 2025-06-20 15:15:00 | 5500.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-06-20 12:45:00 | 5725.00 | 2025-06-20 15:15:00 | 5500.00 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2025-06-23 09:45:00 | 5727.00 | 2025-07-02 11:15:00 | 5948.00 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2025-07-08 11:00:00 | 5769.50 | 2025-07-09 14:15:00 | 5880.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-16 09:15:00 | 6095.00 | 2025-07-16 12:15:00 | 5948.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-16 11:30:00 | 6008.00 | 2025-07-16 12:15:00 | 5948.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 5901.50 | 2025-07-25 14:15:00 | 5606.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 5901.50 | 2025-07-29 09:15:00 | 5311.35 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-21 09:15:00 | 5376.00 | 2025-08-22 13:15:00 | 5329.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-02 09:15:00 | 5026.30 | 2025-09-02 10:15:00 | 5076.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-04 13:00:00 | 5011.00 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-04 14:45:00 | 5019.00 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-05 10:00:00 | 5011.40 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-05 11:15:00 | 5019.10 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-16 13:00:00 | 4993.90 | 2025-09-16 14:15:00 | 5046.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-16 13:30:00 | 5007.70 | 2025-09-16 14:15:00 | 5046.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-16 14:00:00 | 5009.70 | 2025-09-16 14:15:00 | 5046.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest1 | 2025-09-25 12:00:00 | 4765.40 | 2025-09-30 09:15:00 | 4527.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-25 12:00:00 | 4765.40 | 2025-10-01 15:15:00 | 4479.90 | STOP_HIT | 0.50 | 5.99% |
| BUY | retest2 | 2025-10-09 13:15:00 | 4639.00 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-09 15:15:00 | 4635.80 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-10 10:00:00 | 4638.20 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-10-10 10:45:00 | 4655.30 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4735.70 | 2025-11-17 10:15:00 | 4736.20 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-11-07 10:00:00 | 4769.10 | 2025-11-17 10:15:00 | 4736.20 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-11-07 11:00:00 | 4757.00 | 2025-11-17 10:15:00 | 4736.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-11-25 13:00:00 | 4463.70 | 2025-12-01 13:15:00 | 4485.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-27 11:00:00 | 4471.30 | 2025-12-01 13:15:00 | 4485.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-01 09:45:00 | 4446.40 | 2025-12-01 13:15:00 | 4485.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-04 10:30:00 | 4415.00 | 2025-12-05 15:15:00 | 4462.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-05 14:30:00 | 4427.00 | 2025-12-05 15:15:00 | 4462.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-11 09:30:00 | 4324.50 | 2025-12-11 13:15:00 | 4376.10 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-11 10:45:00 | 4343.40 | 2025-12-11 13:15:00 | 4376.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2025-12-18 09:15:00 | 4366.40 | 2025-12-19 10:15:00 | 4389.60 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-19 12:15:00 | 4358.60 | 2025-12-29 15:15:00 | 4307.00 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-12-22 09:30:00 | 4353.00 | 2025-12-29 15:15:00 | 4307.00 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2025-12-22 10:30:00 | 4356.20 | 2025-12-29 15:15:00 | 4307.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-01-09 10:45:00 | 4756.00 | 2026-01-14 15:15:00 | 4742.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-01-09 14:45:00 | 4751.00 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-01-12 12:15:00 | 4751.10 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-01-12 13:30:00 | 4761.40 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-01-14 11:30:00 | 4855.90 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-02-02 12:15:00 | 4684.70 | 2026-02-04 14:15:00 | 4689.60 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2026-02-02 13:00:00 | 4690.30 | 2026-02-04 14:15:00 | 4689.60 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-02-04 12:00:00 | 4689.70 | 2026-02-04 14:15:00 | 4689.60 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 4429.80 | 2026-02-17 13:15:00 | 4610.00 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2026-02-16 10:30:00 | 4637.60 | 2026-02-17 13:15:00 | 4610.00 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2026-02-26 09:15:00 | 4497.20 | 2026-03-02 09:15:00 | 4272.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 4497.20 | 2026-03-02 09:15:00 | 4437.20 | STOP_HIT | 0.50 | 1.33% |
| BUY | retest2 | 2026-03-12 12:00:00 | 4296.00 | 2026-03-12 14:15:00 | 4245.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-18 15:00:00 | 4033.20 | 2026-03-25 10:15:00 | 4050.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-03-19 09:15:00 | 4019.30 | 2026-03-25 10:15:00 | 4050.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-03-20 10:30:00 | 4031.20 | 2026-03-25 10:15:00 | 4050.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-04-02 09:15:00 | 3751.50 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-04-06 14:30:00 | 3775.20 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2026-04-07 09:30:00 | 3772.80 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-04-07 10:45:00 | 3770.30 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2026-04-07 12:15:00 | 3770.40 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4058.80 | 2026-04-24 11:15:00 | 4319.80 | STOP_HIT | 1.00 | 6.43% |
| BUY | retest2 | 2026-04-13 13:45:00 | 4063.50 | 2026-04-24 11:15:00 | 4319.80 | STOP_HIT | 1.00 | 6.31% |
| BUY | retest2 | 2026-04-15 09:15:00 | 4085.00 | 2026-04-24 11:15:00 | 4319.80 | STOP_HIT | 1.00 | 5.75% |
| SELL | retest2 | 2026-04-28 11:15:00 | 4242.00 | 2026-04-29 12:15:00 | 4354.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-04-28 12:00:00 | 4247.50 | 2026-04-29 12:15:00 | 4354.80 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-28 13:30:00 | 4258.70 | 2026-04-29 12:15:00 | 4354.80 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-05-07 15:00:00 | 4147.10 | 2026-05-08 12:15:00 | 4201.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-05-08 11:15:00 | 4167.20 | 2026-05-08 12:15:00 | 4201.90 | STOP_HIT | 1.00 | -0.83% |
