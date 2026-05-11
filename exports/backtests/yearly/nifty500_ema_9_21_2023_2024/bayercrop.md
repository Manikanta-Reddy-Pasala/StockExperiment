# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 4600.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 206 |
| ALERT1 | 138 |
| ALERT2 | 134 |
| ALERT2_SKIP | 64 |
| ALERT3 | 369 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 202 |
| PARTIAL | 23 |
| TARGET_HIT | 8 |
| STOP_HIT | 201 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 232 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 87 / 145
- **Target hits / Stop hits / Partials:** 8 / 201 / 23
- **Avg / median % per leg:** 0.54% / -0.71%
- **Sum % (uncompounded):** 126.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 32 | 37.2% | 7 | 78 | 1 | 0.75% | 64.7% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | 1.39% | 8.3% |
| BUY @ 3rd Alert (retest2) | 80 | 30 | 37.5% | 7 | 73 | 0 | 0.70% | 56.4% |
| SELL (all) | 146 | 55 | 37.7% | 1 | 123 | 22 | 0.42% | 61.4% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.86% | -17.1% |
| SELL @ 3rd Alert (retest2) | 140 | 55 | 39.3% | 1 | 117 | 22 | 0.56% | 78.5% |
| retest1 (combined) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.73% | -8.8% |
| retest2 (combined) | 220 | 85 | 38.6% | 8 | 190 | 22 | 0.61% | 134.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 11:15:00 | 4161.40 | 4151.12 | 4151.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 15:15:00 | 4171.25 | 4158.81 | 4155.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 13:15:00 | 4140.00 | 4161.48 | 4158.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 13:15:00 | 4140.00 | 4161.48 | 4158.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 4140.00 | 4161.48 | 4158.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 14:00:00 | 4140.00 | 4161.48 | 4158.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 4134.10 | 4156.00 | 4156.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 13:15:00 | 4131.00 | 4142.86 | 4148.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 4147.90 | 4140.41 | 4145.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 4147.90 | 4140.41 | 4145.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 4147.90 | 4140.41 | 4145.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 4140.85 | 4140.41 | 4145.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 4139.90 | 4140.31 | 4145.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:45:00 | 4150.95 | 4140.31 | 4145.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 4134.95 | 4139.24 | 4144.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:30:00 | 4152.00 | 4139.24 | 4144.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 4116.50 | 4127.86 | 4136.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 10:15:00 | 4104.10 | 4127.86 | 4136.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 15:15:00 | 4104.00 | 4115.75 | 4125.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 09:30:00 | 4112.00 | 4115.70 | 4123.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 09:15:00 | 4172.05 | 4132.62 | 4128.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 4172.05 | 4132.62 | 4128.65 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 4112.05 | 4129.36 | 4130.38 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 4142.00 | 4126.60 | 4125.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 10:15:00 | 4145.00 | 4130.28 | 4127.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 13:15:00 | 4136.70 | 4141.05 | 4133.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 13:15:00 | 4136.70 | 4141.05 | 4133.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 4136.70 | 4141.05 | 4133.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 13:45:00 | 4130.00 | 4141.05 | 4133.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 4390.70 | 4401.95 | 4383.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 14:30:00 | 4390.40 | 4401.95 | 4383.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 15:15:00 | 4390.10 | 4399.58 | 4384.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 09:15:00 | 4403.35 | 4399.58 | 4384.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 12:15:00 | 4359.60 | 4387.22 | 4383.04 | SL hit (close<static) qty=1.00 sl=4376.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 13:15:00 | 4349.95 | 4379.77 | 4380.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 4327.00 | 4359.36 | 4369.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 10:15:00 | 4365.95 | 4316.70 | 4335.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 10:15:00 | 4365.95 | 4316.70 | 4335.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 4365.95 | 4316.70 | 4335.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:00:00 | 4365.95 | 4316.70 | 4335.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 4250.00 | 4303.36 | 4327.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 13:30:00 | 4230.00 | 4280.15 | 4312.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 15:15:00 | 4241.10 | 4275.12 | 4306.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 11:45:00 | 4233.90 | 4244.06 | 4280.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 15:15:00 | 4245.00 | 4238.09 | 4268.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 4257.90 | 4243.16 | 4265.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 12:45:00 | 4216.05 | 4230.51 | 4253.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 13:00:00 | 4215.00 | 4229.94 | 4241.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 13:45:00 | 4199.95 | 4222.95 | 4237.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 12:15:00 | 4276.70 | 4239.60 | 4237.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 12:15:00 | 4276.70 | 4239.60 | 4237.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 09:15:00 | 4281.95 | 4264.10 | 4255.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 10:15:00 | 4306.35 | 4313.59 | 4300.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 10:45:00 | 4313.90 | 4313.59 | 4300.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 4305.00 | 4311.87 | 4300.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:45:00 | 4297.00 | 4311.87 | 4300.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 4307.05 | 4310.91 | 4301.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 13:15:00 | 4316.00 | 4310.91 | 4301.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 14:15:00 | 4317.45 | 4310.52 | 4301.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 15:00:00 | 4314.10 | 4311.23 | 4303.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 4264.95 | 4302.43 | 4300.52 | SL hit (close<static) qty=1.00 sl=4282.05 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 4278.55 | 4297.65 | 4298.52 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 11:15:00 | 4318.05 | 4301.73 | 4300.30 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 12:15:00 | 4289.60 | 4299.30 | 4299.33 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 4324.95 | 4301.44 | 4299.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 10:15:00 | 4331.05 | 4314.02 | 4306.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 13:15:00 | 4320.25 | 4321.16 | 4312.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 14:00:00 | 4320.25 | 4321.16 | 4312.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 12:15:00 | 4330.00 | 4333.15 | 4323.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 12:45:00 | 4330.00 | 4333.15 | 4323.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 4328.00 | 4332.12 | 4323.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 13:30:00 | 4329.20 | 4332.12 | 4323.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 4355.00 | 4336.70 | 4326.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 14:30:00 | 4311.00 | 4336.70 | 4326.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 4384.85 | 4362.04 | 4346.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 14:45:00 | 4338.00 | 4362.04 | 4346.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 4609.45 | 4627.72 | 4592.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 4595.00 | 4627.72 | 4592.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 4589.80 | 4620.13 | 4592.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:30:00 | 4583.00 | 4620.13 | 4592.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 4628.65 | 4621.84 | 4595.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 13:45:00 | 4632.90 | 4624.16 | 4599.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 14:00:00 | 4642.90 | 4657.59 | 4657.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 14:15:00 | 4639.85 | 4654.04 | 4655.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 4639.85 | 4654.04 | 4655.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 15:15:00 | 4616.30 | 4646.49 | 4651.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 12:15:00 | 4642.00 | 4639.12 | 4645.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 12:15:00 | 4642.00 | 4639.12 | 4645.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 4642.00 | 4639.12 | 4645.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:30:00 | 4642.55 | 4639.12 | 4645.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 4633.55 | 4638.01 | 4644.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:00:00 | 4633.55 | 4638.01 | 4644.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 4636.70 | 4623.75 | 4635.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:00:00 | 4636.70 | 4623.75 | 4635.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 4674.05 | 4633.81 | 4638.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:00:00 | 4674.05 | 4633.81 | 4638.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 4677.60 | 4642.57 | 4642.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 4700.00 | 4680.59 | 4669.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 12:15:00 | 4679.35 | 4681.54 | 4672.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-19 12:30:00 | 4680.80 | 4681.54 | 4672.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 4674.80 | 4680.19 | 4672.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 09:45:00 | 4690.90 | 4680.82 | 4674.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 11:30:00 | 4689.40 | 4683.41 | 4677.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 12:00:00 | 4690.95 | 4683.41 | 4677.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 13:15:00 | 4689.45 | 4683.85 | 4677.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 4675.00 | 4682.26 | 4678.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:45:00 | 4663.15 | 4682.26 | 4678.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 4667.75 | 4679.36 | 4677.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-20 15:15:00 | 4667.75 | 4679.36 | 4677.30 | SL hit (close<static) qty=1.00 sl=4672.55 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 4633.70 | 4670.23 | 4673.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 09:15:00 | 4535.10 | 4600.50 | 4623.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 4445.50 | 4443.60 | 4493.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:45:00 | 4441.75 | 4443.60 | 4493.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 4430.25 | 4408.81 | 4430.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 4430.25 | 4408.81 | 4430.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 4444.35 | 4415.92 | 4431.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:00:00 | 4444.35 | 4415.92 | 4431.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 4448.35 | 4422.40 | 4433.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:30:00 | 4443.95 | 4422.40 | 4433.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 4437.90 | 4425.50 | 4433.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 13:00:00 | 4437.90 | 4425.50 | 4433.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 13:15:00 | 4450.00 | 4430.40 | 4435.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:00:00 | 4450.00 | 4430.40 | 4435.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 4445.90 | 4433.50 | 4436.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:45:00 | 4445.85 | 4433.50 | 4436.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 4449.00 | 4436.60 | 4437.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:15:00 | 4476.55 | 4436.60 | 4437.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 4483.00 | 4445.88 | 4441.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 11:15:00 | 4495.05 | 4461.65 | 4449.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-03 14:15:00 | 4554.70 | 4563.39 | 4537.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-03 14:30:00 | 4544.25 | 4563.39 | 4537.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 4592.80 | 4569.27 | 4542.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:15:00 | 4537.00 | 4569.27 | 4542.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 4537.90 | 4563.00 | 4542.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:00:00 | 4537.90 | 4563.00 | 4542.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 4568.10 | 4564.02 | 4544.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 12:15:00 | 4595.00 | 4575.29 | 4561.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 15:00:00 | 4594.95 | 4580.14 | 4567.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 09:15:00 | 4662.20 | 4580.32 | 4568.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 15:15:00 | 4740.00 | 4757.62 | 4758.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 4740.00 | 4757.62 | 4758.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 4700.00 | 4746.09 | 4753.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 4719.00 | 4715.93 | 4727.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 12:15:00 | 4719.00 | 4715.93 | 4727.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 4719.00 | 4715.93 | 4727.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:45:00 | 4725.00 | 4715.93 | 4727.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 4727.70 | 4718.28 | 4727.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:45:00 | 4738.10 | 4718.28 | 4727.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 4767.45 | 4728.12 | 4731.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 15:00:00 | 4767.45 | 4728.12 | 4731.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 4770.00 | 4736.49 | 4734.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 4818.95 | 4752.98 | 4742.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 09:15:00 | 4775.30 | 4801.27 | 4778.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 4775.30 | 4801.27 | 4778.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 4775.30 | 4801.27 | 4778.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:30:00 | 4783.00 | 4801.27 | 4778.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 4765.25 | 4794.06 | 4777.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:30:00 | 4763.10 | 4794.06 | 4777.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 4767.00 | 4788.65 | 4776.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 13:45:00 | 4781.45 | 4781.94 | 4775.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 14:15:00 | 4735.65 | 4772.68 | 4771.80 | SL hit (close<static) qty=1.00 sl=4760.05 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 15:15:00 | 4750.00 | 4768.14 | 4769.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 15:15:00 | 4721.10 | 4754.56 | 4760.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 15:15:00 | 4748.85 | 4746.35 | 4752.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-23 09:15:00 | 4748.55 | 4746.35 | 4752.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 4750.10 | 4747.10 | 4752.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 12:15:00 | 4732.35 | 4748.47 | 4752.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 11:15:00 | 4739.25 | 4726.63 | 4737.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 09:15:00 | 4733.40 | 4717.46 | 4715.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 4733.40 | 4717.46 | 4715.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 4775.00 | 4749.64 | 4738.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 10:15:00 | 4799.00 | 4813.58 | 4787.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-01 11:00:00 | 4799.00 | 4813.58 | 4787.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 4792.10 | 4804.58 | 4789.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:45:00 | 4793.05 | 4804.58 | 4789.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 4790.85 | 4801.84 | 4789.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 15:00:00 | 4790.85 | 4801.84 | 4789.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 4805.00 | 4802.47 | 4790.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 09:45:00 | 4810.75 | 4807.47 | 4794.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 12:15:00 | 5291.83 | 5017.25 | 4908.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 5175.55 | 5232.27 | 5237.16 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 09:15:00 | 5261.95 | 5195.79 | 5194.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 09:15:00 | 5295.10 | 5266.79 | 5252.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 10:15:00 | 5326.55 | 5335.07 | 5302.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-25 11:00:00 | 5326.55 | 5335.07 | 5302.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 5305.80 | 5325.21 | 5303.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 5305.80 | 5325.21 | 5303.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 5383.10 | 5336.79 | 5310.51 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 11:15:00 | 5218.45 | 5291.68 | 5298.81 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 11:15:00 | 5288.30 | 5270.33 | 5270.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 14:15:00 | 5350.00 | 5300.09 | 5287.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 10:15:00 | 5377.00 | 5399.40 | 5366.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 5373.60 | 5394.24 | 5366.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 5373.60 | 5394.24 | 5366.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:45:00 | 5370.05 | 5394.24 | 5366.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 5370.00 | 5389.39 | 5367.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:30:00 | 5372.00 | 5389.39 | 5367.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 5385.00 | 5388.51 | 5368.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 14:15:00 | 5388.50 | 5388.51 | 5368.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 15:00:00 | 5388.00 | 5388.41 | 5370.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 5390.90 | 5384.73 | 5370.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 09:15:00 | 5349.95 | 5377.77 | 5368.53 | SL hit (close<static) qty=1.00 sl=5367.50 alert=retest2 |

### Cycle 24 — SELL (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 11:15:00 | 5310.00 | 5353.01 | 5358.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 5280.40 | 5338.49 | 5351.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 5322.05 | 5316.22 | 5334.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 5322.05 | 5316.22 | 5334.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 5322.05 | 5316.22 | 5334.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:30:00 | 5305.30 | 5316.22 | 5334.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 5341.10 | 5321.20 | 5335.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:00:00 | 5341.10 | 5321.20 | 5335.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 5309.90 | 5318.94 | 5332.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:30:00 | 5335.20 | 5318.94 | 5332.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 5300.05 | 5315.16 | 5329.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:00:00 | 5300.05 | 5315.16 | 5329.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 12:15:00 | 5313.80 | 5295.13 | 5309.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:00:00 | 5313.80 | 5295.13 | 5309.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 5358.90 | 5307.88 | 5313.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:30:00 | 5340.00 | 5307.88 | 5313.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 14:15:00 | 5374.70 | 5321.24 | 5319.16 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 14:15:00 | 5286.35 | 5339.99 | 5340.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 5275.00 | 5299.02 | 5315.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 5288.00 | 5280.73 | 5295.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 5288.00 | 5280.73 | 5295.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 5288.00 | 5280.73 | 5295.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 14:45:00 | 5242.30 | 5270.24 | 5284.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 11:45:00 | 5240.25 | 5270.91 | 5280.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 12:30:00 | 5241.10 | 5266.71 | 5277.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:15:00 | 5240.05 | 5266.71 | 5277.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 5225.00 | 5253.29 | 5267.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 10:15:00 | 5218.30 | 5253.29 | 5267.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 4980.18 | 5035.65 | 5090.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 4978.24 | 5035.65 | 5090.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 4979.05 | 5035.65 | 5090.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 4978.05 | 5035.65 | 5090.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 4957.39 | 5035.65 | 5090.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 15:15:00 | 4915.00 | 4888.53 | 4948.56 | SL hit (close>ema200) qty=0.50 sl=4888.53 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 11:15:00 | 4929.95 | 4902.26 | 4900.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 09:15:00 | 4950.00 | 4924.82 | 4915.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 10:15:00 | 4893.35 | 4918.52 | 4913.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 10:15:00 | 4893.35 | 4918.52 | 4913.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 4893.35 | 4918.52 | 4913.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 4893.35 | 4918.52 | 4913.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 4900.00 | 4914.82 | 4912.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 4900.35 | 4914.82 | 4912.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 4901.30 | 4912.11 | 4911.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:15:00 | 4917.25 | 4912.11 | 4911.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 13:15:00 | 4897.85 | 4909.26 | 4909.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 13:15:00 | 4897.85 | 4909.26 | 4909.91 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 4949.90 | 4916.30 | 4912.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 5120.00 | 4970.60 | 4939.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 13:15:00 | 5387.35 | 5436.42 | 5381.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 13:15:00 | 5387.35 | 5436.42 | 5381.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 5387.35 | 5436.42 | 5381.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:30:00 | 5377.30 | 5436.42 | 5381.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 5370.90 | 5423.32 | 5380.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:30:00 | 5372.80 | 5423.32 | 5380.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 5400.00 | 5418.65 | 5381.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 5453.10 | 5418.65 | 5381.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 15:15:00 | 5470.00 | 5487.38 | 5489.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-11-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 15:15:00 | 5470.00 | 5487.38 | 5489.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 09:15:00 | 5450.95 | 5480.10 | 5485.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 15:15:00 | 5388.50 | 5369.08 | 5400.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 09:15:00 | 5290.00 | 5369.08 | 5400.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 5330.70 | 5361.41 | 5393.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 14:00:00 | 5271.65 | 5329.34 | 5367.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 14:30:00 | 5283.90 | 5319.95 | 5359.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:30:00 | 5283.80 | 5302.22 | 5316.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:30:00 | 5273.00 | 5295.28 | 5310.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 5268.55 | 5273.37 | 5287.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 10:30:00 | 5251.00 | 5267.53 | 5283.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:00:00 | 5253.60 | 5264.32 | 5270.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 11:00:00 | 5253.50 | 5250.43 | 5258.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 11:15:00 | 5323.90 | 5265.12 | 5264.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 5323.90 | 5265.12 | 5264.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 12:15:00 | 5346.10 | 5281.32 | 5271.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 09:15:00 | 5366.00 | 5374.88 | 5344.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-04 09:45:00 | 5365.85 | 5374.88 | 5344.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 5356.20 | 5374.14 | 5352.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 13:00:00 | 5356.20 | 5374.14 | 5352.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 5344.90 | 5368.30 | 5351.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 13:45:00 | 5344.50 | 5368.30 | 5351.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 5330.95 | 5360.83 | 5349.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 15:00:00 | 5330.95 | 5360.83 | 5349.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 5349.70 | 5354.05 | 5348.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 13:00:00 | 5359.00 | 5350.11 | 5347.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 12:15:00 | 5407.60 | 5440.56 | 5441.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 12:15:00 | 5407.60 | 5440.56 | 5441.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 14:15:00 | 5391.45 | 5426.63 | 5432.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 5437.30 | 5428.66 | 5432.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 5437.30 | 5428.66 | 5432.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 5437.30 | 5428.66 | 5432.75 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 5449.80 | 5435.39 | 5434.95 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 14:15:00 | 5367.50 | 5422.55 | 5429.23 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 13:15:00 | 5462.65 | 5426.95 | 5423.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 12:15:00 | 5496.65 | 5460.54 | 5446.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 5457.35 | 5460.81 | 5449.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 15:00:00 | 5457.35 | 5460.81 | 5449.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 5435.00 | 5455.65 | 5447.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 09:45:00 | 5480.95 | 5459.55 | 5450.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 10:45:00 | 5472.05 | 5462.59 | 5452.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:00:00 | 5467.35 | 5463.54 | 5453.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:30:00 | 5470.00 | 5465.87 | 5455.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 5471.05 | 5466.91 | 5457.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 5463.40 | 5466.91 | 5457.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 5499.00 | 5473.33 | 5461.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 14:30:00 | 5463.05 | 5473.33 | 5461.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 5493.65 | 5496.34 | 5477.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:45:00 | 5487.40 | 5496.34 | 5477.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 5505.65 | 5498.20 | 5480.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 13:15:00 | 5520.00 | 5498.20 | 5480.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 15:15:00 | 5546.00 | 5624.90 | 5634.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 15:15:00 | 5546.00 | 5624.90 | 5634.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 15:15:00 | 5502.45 | 5583.32 | 5607.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-01 11:15:00 | 5588.00 | 5560.76 | 5589.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 11:15:00 | 5588.00 | 5560.76 | 5589.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 5588.00 | 5560.76 | 5589.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 12:00:00 | 5588.00 | 5560.76 | 5589.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 5495.10 | 5547.63 | 5580.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 13:30:00 | 5490.50 | 5536.12 | 5572.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 15:15:00 | 5464.10 | 5533.04 | 5567.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 15:15:00 | 5461.05 | 5518.88 | 5541.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-04 15:15:00 | 5487.00 | 5521.88 | 5527.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 5512.80 | 5514.48 | 5523.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-05 14:15:00 | 5589.80 | 5536.65 | 5530.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 14:15:00 | 5589.80 | 5536.65 | 5530.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 15:15:00 | 5600.00 | 5549.32 | 5536.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 11:15:00 | 5546.55 | 5559.77 | 5545.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 11:15:00 | 5546.55 | 5559.77 | 5545.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 5546.55 | 5559.77 | 5545.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 5541.75 | 5559.77 | 5545.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 5615.35 | 5570.89 | 5552.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 5655.90 | 5572.01 | 5557.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:00:00 | 5639.85 | 5585.58 | 5565.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:45:00 | 5643.00 | 5597.86 | 5572.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 11:15:00 | 5803.30 | 5825.82 | 5828.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 11:15:00 | 5803.30 | 5825.82 | 5828.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 14:15:00 | 5785.75 | 5815.42 | 5822.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 5824.50 | 5816.37 | 5821.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 5824.50 | 5816.37 | 5821.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 5824.50 | 5816.37 | 5821.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:00:00 | 5824.50 | 5816.37 | 5821.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 5822.00 | 5817.49 | 5821.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:00:00 | 5822.00 | 5817.49 | 5821.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 5800.00 | 5813.99 | 5819.99 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 14:15:00 | 5975.85 | 5848.06 | 5834.05 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-01-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 12:15:00 | 5809.40 | 5857.76 | 5863.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 14:15:00 | 5791.45 | 5840.99 | 5854.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 09:15:00 | 5819.80 | 5784.42 | 5810.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 5819.80 | 5784.42 | 5810.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 5819.80 | 5784.42 | 5810.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:45:00 | 5835.00 | 5784.42 | 5810.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 5804.90 | 5788.51 | 5810.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 10:30:00 | 5823.70 | 5788.51 | 5810.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 11:15:00 | 5818.00 | 5794.41 | 5811.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 12:45:00 | 5780.00 | 5792.84 | 5808.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 09:45:00 | 5774.95 | 5785.56 | 5799.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 5823.50 | 5797.06 | 5802.52 | SL hit (close>static) qty=1.00 sl=5820.55 alert=retest2 |

### Cycle 41 — BUY (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 15:15:00 | 5820.00 | 5805.94 | 5805.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 11:15:00 | 5849.85 | 5820.33 | 5812.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 13:15:00 | 5896.55 | 5920.56 | 5885.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 13:15:00 | 5896.55 | 5920.56 | 5885.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 5896.55 | 5920.56 | 5885.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:00:00 | 5896.55 | 5920.56 | 5885.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 5851.45 | 5906.74 | 5881.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 5851.45 | 5906.74 | 5881.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 5815.00 | 5888.39 | 5875.87 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 11:15:00 | 5786.00 | 5859.87 | 5865.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 13:15:00 | 5762.95 | 5824.98 | 5842.69 | Break + close below crossover candle low |

### Cycle 43 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 6042.00 | 5849.08 | 5847.37 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 10:15:00 | 5799.40 | 5950.79 | 5953.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 15:15:00 | 5760.00 | 5853.29 | 5899.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 5786.80 | 5764.16 | 5819.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 15:15:00 | 5666.05 | 5756.30 | 5795.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 5837.85 | 5758.17 | 5788.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-07 09:15:00 | 5837.85 | 5758.17 | 5788.84 | SL hit (close>ema400) qty=1.00 sl=5788.84 alert=retest1 |

### Cycle 45 — BUY (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 10:15:00 | 5850.00 | 5798.51 | 5797.10 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 5786.00 | 5801.02 | 5801.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 12:15:00 | 5751.75 | 5788.39 | 5795.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 5829.00 | 5794.84 | 5797.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 14:15:00 | 5829.00 | 5794.84 | 5797.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 5829.00 | 5794.84 | 5797.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 15:00:00 | 5829.00 | 5794.84 | 5797.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 5845.00 | 5804.87 | 5801.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 09:15:00 | 5864.35 | 5816.77 | 5807.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 10:15:00 | 5987.50 | 5998.21 | 5947.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-14 11:00:00 | 5987.50 | 5998.21 | 5947.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 5973.95 | 5993.20 | 5958.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:45:00 | 5979.20 | 5993.20 | 5958.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 5956.90 | 5985.94 | 5957.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:45:00 | 5944.30 | 5985.94 | 5957.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 5980.90 | 5984.93 | 5960.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 5904.90 | 5984.93 | 5960.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 5849.55 | 5957.86 | 5950.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:00:00 | 5849.55 | 5957.86 | 5950.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 5848.45 | 5935.98 | 5940.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 11:15:00 | 5810.85 | 5865.44 | 5895.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 14:15:00 | 5861.95 | 5859.41 | 5884.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 5861.95 | 5859.41 | 5884.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 5932.75 | 5872.18 | 5883.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:00:00 | 5932.75 | 5872.18 | 5883.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 5970.95 | 5891.93 | 5891.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 13:15:00 | 6001.55 | 5928.92 | 5909.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 09:15:00 | 6068.00 | 6077.67 | 6026.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-21 09:30:00 | 6057.00 | 6077.67 | 6026.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 6021.00 | 6080.55 | 6054.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 5978.00 | 6080.55 | 6054.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 5995.00 | 6063.44 | 6048.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:30:00 | 5994.00 | 6063.44 | 6048.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 6017.40 | 6054.23 | 6046.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 11:30:00 | 6056.95 | 6054.77 | 6047.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 15:15:00 | 6015.00 | 6041.88 | 6043.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 15:15:00 | 6015.00 | 6041.88 | 6043.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 09:15:00 | 5986.00 | 6030.70 | 6037.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 11:15:00 | 5961.95 | 5954.91 | 5984.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 11:15:00 | 5961.95 | 5954.91 | 5984.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 5961.95 | 5954.91 | 5984.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:45:00 | 5979.35 | 5954.91 | 5984.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 5763.00 | 5748.71 | 5797.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:45:00 | 5766.10 | 5748.71 | 5797.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 5690.60 | 5670.18 | 5711.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 5690.65 | 5672.28 | 5708.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 5627.15 | 5660.33 | 5696.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:45:00 | 5600.80 | 5639.51 | 5680.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 5320.76 | 5396.62 | 5444.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 13:15:00 | 5040.72 | 5185.02 | 5289.33 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 5065.00 | 5035.22 | 5034.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 5170.60 | 5077.07 | 5057.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 5113.85 | 5115.48 | 5087.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 15:00:00 | 5113.85 | 5115.48 | 5087.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 5081.25 | 5108.63 | 5086.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 5120.25 | 5108.63 | 5086.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:45:00 | 5156.45 | 5114.51 | 5091.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 14:15:00 | 5440.15 | 5465.48 | 5466.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 5440.15 | 5465.48 | 5466.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 15:15:00 | 5404.00 | 5453.18 | 5460.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 14:15:00 | 5429.80 | 5426.67 | 5441.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-05 15:00:00 | 5429.80 | 5426.67 | 5441.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 5429.05 | 5427.67 | 5439.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 14:30:00 | 5395.00 | 5429.23 | 5437.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 15:00:00 | 5417.00 | 5372.87 | 5382.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 09:15:00 | 5464.00 | 5389.04 | 5387.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 5464.00 | 5389.04 | 5387.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 12:15:00 | 5497.10 | 5434.85 | 5411.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 14:15:00 | 5493.45 | 5508.47 | 5472.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 14:15:00 | 5493.45 | 5508.47 | 5472.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 5493.45 | 5508.47 | 5472.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 14:45:00 | 5471.55 | 5508.47 | 5472.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 5500.00 | 5506.77 | 5475.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 5476.20 | 5506.77 | 5475.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 5453.35 | 5496.09 | 5473.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:30:00 | 5449.15 | 5496.09 | 5473.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 5454.10 | 5487.69 | 5471.57 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 12:15:00 | 5360.60 | 5456.73 | 5459.95 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 5518.95 | 5455.41 | 5451.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 13:15:00 | 5570.35 | 5478.40 | 5462.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 11:15:00 | 5688.45 | 5689.10 | 5619.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-22 12:00:00 | 5688.45 | 5689.10 | 5619.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 5635.15 | 5665.04 | 5624.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 14:45:00 | 5650.00 | 5665.04 | 5624.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 5586.20 | 5643.99 | 5621.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:45:00 | 5585.00 | 5643.99 | 5621.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 5544.90 | 5624.17 | 5614.51 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 12:15:00 | 5541.20 | 5595.71 | 5602.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 15:15:00 | 5523.00 | 5564.46 | 5580.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 10:15:00 | 5566.40 | 5564.65 | 5578.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:30:00 | 5558.20 | 5564.65 | 5578.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 5578.65 | 5568.36 | 5577.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:45:00 | 5596.50 | 5568.36 | 5577.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 5565.00 | 5567.69 | 5576.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 13:45:00 | 5570.00 | 5567.69 | 5576.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 5578.60 | 5569.87 | 5576.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 15:00:00 | 5578.60 | 5569.87 | 5576.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 5583.00 | 5572.50 | 5577.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 5608.00 | 5572.50 | 5577.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 5559.00 | 5569.80 | 5575.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 11:30:00 | 5530.15 | 5564.18 | 5571.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:30:00 | 5514.50 | 5519.84 | 5523.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 5533.25 | 5488.10 | 5497.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 14:15:00 | 5256.59 | 5333.95 | 5366.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 15:15:00 | 5253.64 | 5312.17 | 5353.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 15:15:00 | 5238.77 | 5312.17 | 5353.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 5224.95 | 5222.04 | 5263.75 | SL hit (close>ema200) qty=0.50 sl=5222.04 alert=retest2 |

### Cycle 57 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 5304.00 | 5274.15 | 5271.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 5327.00 | 5289.50 | 5278.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 5422.20 | 5431.58 | 5386.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 5422.20 | 5431.58 | 5386.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 5393.95 | 5424.05 | 5386.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 5393.95 | 5424.05 | 5386.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 5442.00 | 5427.64 | 5391.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:15:00 | 5472.00 | 5431.81 | 5397.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 5551.95 | 5518.08 | 5464.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 5575.05 | 5598.19 | 5598.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 5575.05 | 5598.19 | 5598.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 5293.10 | 5529.85 | 5566.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 5234.70 | 5210.01 | 5273.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 15:00:00 | 5234.70 | 5210.01 | 5273.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 5188.40 | 5204.54 | 5259.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 5183.75 | 5204.54 | 5259.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 5152.05 | 5094.87 | 5088.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 5152.05 | 5094.87 | 5088.40 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 5000.00 | 5095.87 | 5096.87 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 5143.85 | 5106.13 | 5101.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 5170.40 | 5116.27 | 5107.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 6020.10 | 6034.48 | 5900.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 11:00:00 | 6020.10 | 6034.48 | 5900.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 6043.95 | 6099.69 | 6044.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 6030.10 | 6099.69 | 6044.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 6059.95 | 6091.74 | 6045.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:00:00 | 6100.00 | 6093.39 | 6050.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-27 09:15:00 | 6710.00 | 6507.61 | 6461.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 6546.60 | 6637.69 | 6649.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 11:15:00 | 6479.00 | 6565.64 | 6604.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 6520.10 | 6492.89 | 6546.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 6520.10 | 6492.89 | 6546.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 6520.10 | 6492.89 | 6546.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:45:00 | 6515.00 | 6492.89 | 6546.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 6577.70 | 6514.81 | 6547.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:45:00 | 6571.95 | 6514.81 | 6547.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 6585.00 | 6528.85 | 6550.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:00:00 | 6585.00 | 6528.85 | 6550.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 6500.05 | 6539.95 | 6550.38 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 6570.00 | 6556.33 | 6556.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 6587.00 | 6562.46 | 6558.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 12:15:00 | 6617.95 | 6651.94 | 6625.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 12:15:00 | 6617.95 | 6651.94 | 6625.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 6617.95 | 6651.94 | 6625.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 6613.70 | 6651.94 | 6625.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 6611.00 | 6643.75 | 6623.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 6611.00 | 6643.75 | 6623.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 6622.05 | 6638.00 | 6624.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 6582.00 | 6638.00 | 6624.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 6504.60 | 6611.32 | 6613.60 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 6600.00 | 6574.53 | 6573.34 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 6561.40 | 6571.91 | 6572.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 15:15:00 | 6550.90 | 6567.70 | 6570.31 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 6609.85 | 6576.13 | 6573.91 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 6556.70 | 6573.94 | 6574.20 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 14:15:00 | 6596.40 | 6578.43 | 6576.22 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 6560.00 | 6574.75 | 6574.75 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 6634.90 | 6586.78 | 6580.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 10:15:00 | 6657.60 | 6600.94 | 6587.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 13:15:00 | 6595.95 | 6608.56 | 6595.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 6595.95 | 6608.56 | 6595.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 6595.95 | 6608.56 | 6595.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 6549.95 | 6608.56 | 6595.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 6636.50 | 6614.15 | 6598.86 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 6524.00 | 6587.61 | 6592.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 6484.00 | 6566.89 | 6582.19 | Break + close below crossover candle low |

### Cycle 73 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 6712.60 | 6596.02 | 6592.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 6754.00 | 6627.61 | 6607.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 6664.35 | 6684.98 | 6657.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 6664.35 | 6684.98 | 6657.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 6664.35 | 6684.98 | 6657.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 6685.00 | 6684.98 | 6657.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 6689.50 | 6685.89 | 6660.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 6655.70 | 6685.89 | 6660.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 6703.50 | 6689.41 | 6664.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 6703.50 | 6689.41 | 6664.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 6700.10 | 6691.55 | 6667.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 6635.00 | 6680.24 | 6664.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 6654.85 | 6675.16 | 6663.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 6642.50 | 6675.16 | 6663.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 6654.95 | 6666.16 | 6661.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:00:00 | 6654.95 | 6666.16 | 6661.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 6648.90 | 6662.71 | 6660.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:00:00 | 6648.90 | 6662.71 | 6660.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 15:15:00 | 6628.00 | 6655.77 | 6657.63 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 6671.25 | 6651.90 | 6650.15 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 6613.60 | 6646.39 | 6648.28 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 6669.00 | 6650.91 | 6650.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 6994.60 | 6719.65 | 6681.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 6953.40 | 6987.01 | 6928.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 6953.40 | 6987.01 | 6928.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 6999.55 | 6982.46 | 6935.83 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 6860.00 | 6921.92 | 6923.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 6834.50 | 6901.22 | 6912.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 15:15:00 | 6900.00 | 6888.82 | 6902.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 09:15:00 | 6700.00 | 6888.82 | 6902.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 6787.80 | 6743.71 | 6781.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-06 12:15:00 | 6787.80 | 6743.71 | 6781.04 | SL hit (close>ema400) qty=1.00 sl=6781.04 alert=retest1 |

### Cycle 79 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 6934.35 | 6808.15 | 6793.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 6962.00 | 6838.92 | 6809.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 6934.30 | 6964.71 | 6900.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 6934.30 | 6964.71 | 6900.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 6964.00 | 6964.57 | 6906.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 6502.65 | 6964.57 | 6906.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 6605.30 | 6892.72 | 6879.00 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 10:15:00 | 6557.15 | 6825.60 | 6849.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 6481.60 | 6632.71 | 6728.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 6285.60 | 6275.04 | 6393.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 6285.60 | 6275.04 | 6393.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 6137.00 | 6223.28 | 6321.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 6112.00 | 6201.03 | 6302.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 6395.00 | 6255.55 | 6238.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 6395.00 | 6255.55 | 6238.58 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 6287.60 | 6298.27 | 6298.32 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 6367.15 | 6312.05 | 6304.57 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 6304.00 | 6347.02 | 6348.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 10:15:00 | 6284.30 | 6329.97 | 6339.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 12:15:00 | 6332.95 | 6317.77 | 6331.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 12:15:00 | 6332.95 | 6317.77 | 6331.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 6332.95 | 6317.77 | 6331.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:45:00 | 6346.65 | 6317.77 | 6331.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 6312.45 | 6316.70 | 6330.04 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 6420.70 | 6340.47 | 6333.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 10:15:00 | 6476.30 | 6397.07 | 6370.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 6452.30 | 6465.58 | 6426.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 11:00:00 | 6452.30 | 6465.58 | 6426.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 6445.80 | 6456.50 | 6434.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:30:00 | 6429.15 | 6456.50 | 6434.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 6365.90 | 6435.60 | 6428.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 6365.90 | 6435.60 | 6428.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 6363.55 | 6421.19 | 6422.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 6336.45 | 6388.03 | 6404.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 6499.95 | 6332.94 | 6348.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 6499.95 | 6332.94 | 6348.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 6499.95 | 6332.94 | 6348.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 6499.95 | 6332.94 | 6348.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 6469.30 | 6360.21 | 6359.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 6515.75 | 6494.87 | 6462.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 10:15:00 | 6458.60 | 6502.88 | 6483.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 10:15:00 | 6458.60 | 6502.88 | 6483.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 6458.60 | 6502.88 | 6483.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:00:00 | 6458.60 | 6502.88 | 6483.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 6494.45 | 6501.19 | 6484.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:45:00 | 6499.75 | 6500.95 | 6485.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:45:00 | 6498.85 | 6500.16 | 6486.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 15:00:00 | 6499.95 | 6500.12 | 6487.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 6500.30 | 6496.50 | 6487.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 6509.80 | 6499.16 | 6489.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-17 12:15:00 | 6466.60 | 6489.49 | 6490.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 6466.60 | 6489.49 | 6490.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 6434.35 | 6474.27 | 6483.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 6464.25 | 6403.09 | 6430.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 6464.25 | 6403.09 | 6430.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 6464.25 | 6403.09 | 6430.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 6464.25 | 6403.09 | 6430.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 6405.30 | 6403.53 | 6427.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 6341.30 | 6403.53 | 6427.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 6341.10 | 6258.85 | 6252.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 6341.10 | 6258.85 | 6252.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 6367.50 | 6301.23 | 6282.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 6535.45 | 6566.90 | 6481.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 6535.45 | 6566.90 | 6481.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 6508.45 | 6555.21 | 6484.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:30:00 | 6470.00 | 6555.21 | 6484.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 6452.05 | 6525.58 | 6482.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 6452.05 | 6525.58 | 6482.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 6506.55 | 6521.78 | 6484.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 6431.10 | 6521.78 | 6484.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 6492.25 | 6512.55 | 6486.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 6374.95 | 6512.55 | 6486.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 6510.15 | 6512.07 | 6488.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 6528.45 | 6534.65 | 6501.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 13:00:00 | 6527.00 | 6533.12 | 6503.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 6613.65 | 6691.12 | 6701.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 6613.65 | 6691.12 | 6701.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 13:15:00 | 6600.00 | 6648.91 | 6677.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 6780.00 | 6592.05 | 6603.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 14:15:00 | 6780.00 | 6592.05 | 6603.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 6780.00 | 6592.05 | 6603.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 6780.00 | 6592.05 | 6603.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 15:15:00 | 6800.00 | 6633.64 | 6621.34 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 6590.30 | 6632.67 | 6636.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 6576.30 | 6621.40 | 6631.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 6636.55 | 6608.95 | 6620.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 6636.55 | 6608.95 | 6620.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 6636.55 | 6608.95 | 6620.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 6636.55 | 6608.95 | 6620.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 6615.35 | 6610.23 | 6620.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 6638.65 | 6610.23 | 6620.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 6615.10 | 6611.21 | 6619.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:15:00 | 6641.60 | 6611.21 | 6619.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 6618.05 | 6612.57 | 6619.46 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 15:15:00 | 6690.00 | 6628.93 | 6625.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-22 09:15:00 | 6701.35 | 6643.41 | 6632.13 | Break + close above crossover candle high |

### Cycle 94 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 6493.20 | 6613.37 | 6619.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 6390.00 | 6540.90 | 6580.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 09:15:00 | 6366.45 | 6364.99 | 6406.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 6366.45 | 6364.99 | 6406.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 6366.45 | 6364.99 | 6406.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:15:00 | 6320.70 | 6368.24 | 6395.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 6481.20 | 6390.09 | 6395.88 | SL hit (close>static) qty=1.00 sl=6415.45 alert=retest2 |

### Cycle 95 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 6468.90 | 6405.85 | 6402.52 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 6412.55 | 6447.99 | 6449.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 6374.15 | 6433.22 | 6443.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 6528.45 | 6445.34 | 6446.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 6528.45 | 6445.34 | 6446.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 6528.45 | 6445.34 | 6446.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 6528.45 | 6445.34 | 6446.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 6561.15 | 6468.50 | 6456.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 6585.90 | 6491.98 | 6468.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 11:15:00 | 6719.40 | 6720.01 | 6652.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 11:45:00 | 6716.20 | 6720.01 | 6652.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 6714.25 | 6733.92 | 6706.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 6714.25 | 6733.92 | 6706.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 6674.00 | 6721.94 | 6703.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 6685.25 | 6721.94 | 6703.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 6645.55 | 6706.66 | 6698.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 6645.55 | 6706.66 | 6698.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 6677.70 | 6700.87 | 6696.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 6625.95 | 6700.87 | 6696.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 6656.85 | 6692.06 | 6693.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 12:15:00 | 6640.00 | 6681.65 | 6688.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 5757.70 | 5743.49 | 5885.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:30:00 | 5730.05 | 5743.49 | 5885.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 5901.80 | 5729.80 | 5800.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 5901.80 | 5729.80 | 5800.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 5810.00 | 5745.84 | 5801.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 5918.25 | 5745.84 | 5801.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 5780.00 | 5752.67 | 5799.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:30:00 | 5747.20 | 5756.00 | 5796.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 13:15:00 | 5752.30 | 5756.00 | 5796.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:30:00 | 5760.50 | 5763.82 | 5793.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:30:00 | 5756.80 | 5771.69 | 5790.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 5746.75 | 5764.83 | 5784.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 5768.00 | 5764.83 | 5784.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 5767.75 | 5765.41 | 5782.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 5767.75 | 5765.41 | 5782.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 5865.10 | 5777.29 | 5783.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 5865.10 | 5777.29 | 5783.06 | SL hit (close>static) qty=1.00 sl=5835.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 5815.00 | 5789.07 | 5787.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 5845.00 | 5817.62 | 5807.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 12:15:00 | 5795.90 | 5826.57 | 5815.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 12:15:00 | 5795.90 | 5826.57 | 5815.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 5795.90 | 5826.57 | 5815.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:45:00 | 5796.40 | 5826.57 | 5815.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 5799.95 | 5821.25 | 5813.89 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 5775.05 | 5805.27 | 5807.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 5696.05 | 5783.43 | 5797.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 5705.70 | 5702.53 | 5743.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 5705.70 | 5702.53 | 5743.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 5705.70 | 5702.53 | 5743.27 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 5817.15 | 5745.10 | 5738.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 5890.00 | 5786.42 | 5759.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 14:15:00 | 6098.90 | 6099.57 | 6019.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 15:00:00 | 6098.90 | 6099.57 | 6019.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 6098.75 | 6091.91 | 6054.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:30:00 | 6058.20 | 6091.91 | 6054.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 6072.55 | 6085.97 | 6061.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:45:00 | 6056.80 | 6085.97 | 6061.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 6070.00 | 6082.78 | 6061.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 6090.00 | 6057.71 | 6055.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:00:00 | 6100.00 | 6072.29 | 6063.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 13:15:00 | 6041.20 | 6065.35 | 6061.46 | SL hit (close<static) qty=1.00 sl=6060.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 14:15:00 | 6191.95 | 6213.35 | 6213.81 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 6224.85 | 6215.65 | 6214.81 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 6169.85 | 6206.38 | 6210.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 6125.00 | 6190.11 | 6202.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 5855.00 | 5810.04 | 5912.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 5855.00 | 5810.04 | 5912.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 5862.05 | 5820.44 | 5907.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:45:00 | 5870.05 | 5820.44 | 5907.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 5868.70 | 5818.63 | 5872.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:30:00 | 5863.70 | 5818.63 | 5872.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 5863.15 | 5834.15 | 5870.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:30:00 | 5860.00 | 5834.15 | 5870.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 5860.05 | 5839.33 | 5869.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:30:00 | 5860.05 | 5839.33 | 5869.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 5860.05 | 5843.48 | 5868.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 5785.80 | 5843.48 | 5868.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 5855.40 | 5845.86 | 5867.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 5855.40 | 5845.86 | 5867.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 5903.65 | 5857.42 | 5870.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:45:00 | 5917.40 | 5857.42 | 5870.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 5893.95 | 5864.72 | 5872.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 5876.40 | 5864.72 | 5872.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 5899.70 | 5871.72 | 5875.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:00:00 | 5899.70 | 5871.72 | 5875.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 5752.05 | 5849.11 | 5864.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:30:00 | 5849.65 | 5849.11 | 5864.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 5756.20 | 5740.99 | 5776.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 5756.20 | 5740.99 | 5776.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 5809.80 | 5754.75 | 5779.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 5809.80 | 5754.75 | 5779.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 5785.10 | 5760.82 | 5779.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 5784.70 | 5760.82 | 5779.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 5777.20 | 5764.10 | 5779.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 5743.30 | 5768.49 | 5774.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 5817.25 | 5780.35 | 5779.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 5817.25 | 5780.35 | 5779.25 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 5551.70 | 5746.28 | 5765.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 5462.60 | 5689.54 | 5737.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 5550.00 | 5547.35 | 5629.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 5550.00 | 5547.35 | 5629.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 5585.20 | 5540.69 | 5562.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 14:00:00 | 5585.20 | 5540.69 | 5562.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 5641.85 | 5560.92 | 5569.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 5641.85 | 5560.92 | 5569.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 5617.10 | 5575.34 | 5574.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 11:15:00 | 5645.05 | 5596.57 | 5584.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 5572.60 | 5606.82 | 5596.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 5572.60 | 5606.82 | 5596.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 5572.60 | 5606.82 | 5596.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 5587.70 | 5606.82 | 5596.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 5497.50 | 5584.96 | 5587.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 14:15:00 | 5470.00 | 5513.02 | 5530.16 | Break + close below crossover candle low |

### Cycle 109 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 5718.20 | 5547.01 | 5542.20 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 5420.50 | 5538.54 | 5553.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 5400.25 | 5494.33 | 5529.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 5251.45 | 5228.42 | 5318.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 5251.45 | 5228.42 | 5318.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 5231.60 | 5214.98 | 5267.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:30:00 | 5250.25 | 5214.98 | 5267.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 5233.75 | 5218.74 | 5264.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 5233.75 | 5218.74 | 5264.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 5215.70 | 5192.80 | 5232.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:00:00 | 5176.05 | 5189.45 | 5227.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 5160.50 | 5184.56 | 5221.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 5140.65 | 5164.83 | 5192.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 12:15:00 | 5092.05 | 5028.29 | 5027.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 5092.05 | 5028.29 | 5027.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 5177.30 | 5058.09 | 5041.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 5021.60 | 5070.50 | 5052.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 5021.60 | 5070.50 | 5052.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 5021.60 | 5070.50 | 5052.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 5021.60 | 5070.50 | 5052.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 5032.00 | 5062.80 | 5050.75 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 4971.60 | 5030.58 | 5037.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 4956.05 | 5015.68 | 5030.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 4914.15 | 4885.02 | 4941.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 4914.15 | 4885.02 | 4941.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 4900.60 | 4836.83 | 4874.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 4900.60 | 4836.83 | 4874.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 4916.90 | 4852.85 | 4878.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 4933.95 | 4852.85 | 4878.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 4924.70 | 4872.43 | 4883.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 4927.45 | 4872.43 | 4883.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 4987.75 | 4895.49 | 4892.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 5008.15 | 4918.02 | 4903.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 5109.10 | 5118.30 | 5075.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 5109.10 | 5118.30 | 5075.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 5109.10 | 5118.30 | 5075.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 5103.05 | 5118.30 | 5075.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 5135.80 | 5117.70 | 5082.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:30:00 | 5142.90 | 5120.60 | 5087.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 5195.50 | 5120.68 | 5090.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 5173.90 | 5138.23 | 5120.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:00:00 | 5149.00 | 5166.78 | 5143.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 5112.00 | 5155.82 | 5140.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 5112.00 | 5155.82 | 5140.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 5121.95 | 5149.05 | 5138.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 5077.30 | 5149.05 | 5138.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-05 11:15:00 | 5092.15 | 5126.03 | 5129.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 5092.15 | 5126.03 | 5129.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 12:15:00 | 5055.00 | 5111.83 | 5123.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 15:15:00 | 4450.00 | 4420.67 | 4527.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 09:15:00 | 4325.35 | 4420.67 | 4527.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 09:45:00 | 4375.30 | 4413.01 | 4514.02 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 12:30:00 | 4380.05 | 4407.36 | 4486.21 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 13:45:00 | 4349.15 | 4398.37 | 4474.95 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 4444.55 | 4407.60 | 4472.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 15:00:00 | 4444.55 | 4407.60 | 4472.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 4444.00 | 4414.88 | 4469.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 4496.80 | 4426.39 | 4465.05 | SL hit (close>ema400) qty=1.00 sl=4465.05 alert=retest1 |

### Cycle 115 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 4595.70 | 4491.84 | 4486.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 4703.70 | 4582.71 | 4537.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 4742.00 | 4744.71 | 4695.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 14:30:00 | 4744.20 | 4744.71 | 4695.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 4742.00 | 4784.20 | 4748.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 4709.95 | 4784.20 | 4748.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 4765.90 | 4780.54 | 4750.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 4764.40 | 4780.54 | 4750.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 4752.25 | 4774.88 | 4750.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:00:00 | 4752.25 | 4774.88 | 4750.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 4746.85 | 4769.27 | 4750.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:15:00 | 4744.60 | 4769.27 | 4750.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 4747.65 | 4764.95 | 4750.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 4740.00 | 4764.95 | 4750.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 4737.00 | 4759.36 | 4748.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 4737.00 | 4759.36 | 4748.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 4730.00 | 4744.07 | 4743.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 4750.40 | 4744.07 | 4743.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 4727.55 | 4740.77 | 4741.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 4727.55 | 4740.77 | 4741.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 12:15:00 | 4725.00 | 4736.03 | 4739.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 4759.90 | 4735.63 | 4737.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 4759.90 | 4735.63 | 4737.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 4759.90 | 4735.63 | 4737.40 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 4754.45 | 4739.40 | 4738.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 11:15:00 | 4879.10 | 4767.34 | 4751.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 4662.50 | 4820.89 | 4794.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 4662.50 | 4820.89 | 4794.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 4662.50 | 4820.89 | 4794.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 4662.50 | 4820.89 | 4794.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 4692.50 | 4795.21 | 4785.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:45:00 | 4660.00 | 4795.21 | 4785.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 11:15:00 | 4628.40 | 4761.85 | 4770.92 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 4830.40 | 4742.89 | 4732.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 4883.80 | 4784.34 | 4753.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 12:15:00 | 4827.20 | 4827.82 | 4793.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 13:00:00 | 4827.20 | 4827.82 | 4793.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 4886.00 | 4879.11 | 4851.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:45:00 | 4906.00 | 4885.25 | 4856.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:45:00 | 4910.00 | 4887.78 | 4871.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:15:00 | 4915.10 | 4887.94 | 4874.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:45:00 | 4926.25 | 4891.44 | 4878.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 4882.15 | 4894.15 | 4882.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 4873.65 | 4894.15 | 4882.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 4875.20 | 4890.36 | 4881.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 4891.35 | 4890.36 | 4881.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 4873.75 | 4887.04 | 4880.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:30:00 | 4884.90 | 4887.04 | 4880.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 4845.95 | 4878.82 | 4877.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 4845.95 | 4878.82 | 4877.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 4836.75 | 4871.56 | 4874.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 14:15:00 | 4836.75 | 4871.56 | 4874.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 4710.05 | 4823.36 | 4850.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 10:15:00 | 4778.45 | 4759.87 | 4797.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 11:00:00 | 4778.45 | 4759.87 | 4797.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 4755.75 | 4750.08 | 4774.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 4779.60 | 4750.08 | 4774.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 4779.00 | 4755.86 | 4775.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 4779.00 | 4755.86 | 4775.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 4769.15 | 4758.52 | 4774.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 4772.10 | 4758.52 | 4774.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 4793.80 | 4765.57 | 4776.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 4793.80 | 4765.57 | 4776.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 4805.00 | 4773.46 | 4779.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:45:00 | 4822.40 | 4773.46 | 4779.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 4824.00 | 4789.99 | 4786.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 4940.60 | 4820.11 | 4800.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 14:15:00 | 4860.80 | 4861.85 | 4832.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 15:00:00 | 4860.80 | 4861.85 | 4832.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 4937.95 | 4952.06 | 4925.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 4952.95 | 4952.06 | 4925.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 4905.35 | 4942.72 | 4923.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:00:00 | 4905.35 | 4942.72 | 4923.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 4892.40 | 4932.66 | 4920.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:00:00 | 4892.40 | 4932.66 | 4920.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 4876.00 | 4910.00 | 4912.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 15:15:00 | 4825.00 | 4893.00 | 4904.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 4987.05 | 4911.81 | 4911.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 09:15:00 | 4987.05 | 4911.81 | 4911.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 4987.05 | 4911.81 | 4911.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:00:00 | 4987.05 | 4911.81 | 4911.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 5013.95 | 4932.24 | 4921.14 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 4861.90 | 4915.40 | 4918.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 11:15:00 | 4847.65 | 4894.22 | 4908.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 14:15:00 | 4874.75 | 4870.97 | 4892.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-25 14:45:00 | 4869.95 | 4870.97 | 4892.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 4864.80 | 4868.14 | 4887.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:45:00 | 4850.45 | 4866.57 | 4879.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 15:15:00 | 4850.00 | 4866.57 | 4879.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 4831.20 | 4857.68 | 4871.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:45:00 | 4842.85 | 4856.22 | 4869.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 4876.30 | 4860.23 | 4870.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 4870.00 | 4860.23 | 4870.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 4880.00 | 4864.19 | 4871.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 4880.60 | 4864.19 | 4871.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 4694.15 | 4830.18 | 4855.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 4890.00 | 4830.18 | 4855.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 5230.00 | 4900.03 | 4881.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 5230.00 | 4900.03 | 4881.90 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 4819.85 | 4891.75 | 4896.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 4785.00 | 4845.76 | 4871.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 4816.70 | 4804.36 | 4834.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 15:00:00 | 4816.70 | 4804.36 | 4834.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 4794.90 | 4805.77 | 4829.91 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 4948.05 | 4853.76 | 4844.85 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 4785.55 | 4837.04 | 4840.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4659.65 | 4789.45 | 4815.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 4748.20 | 4727.09 | 4762.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 4748.20 | 4727.09 | 4762.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4748.20 | 4727.09 | 4762.68 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 4800.00 | 4780.51 | 4779.90 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 4710.20 | 4767.14 | 4774.41 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 4806.15 | 4768.53 | 4766.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 4908.30 | 4857.37 | 4831.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 4933.10 | 4969.18 | 4916.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 10:00:00 | 4933.10 | 4969.18 | 4916.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 4930.30 | 4961.40 | 4918.03 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 13:15:00 | 4896.60 | 4908.53 | 4909.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 14:15:00 | 4894.30 | 4905.68 | 4907.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 10:15:00 | 4924.80 | 4900.09 | 4904.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 10:15:00 | 4924.80 | 4900.09 | 4904.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 4924.80 | 4900.09 | 4904.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:00:00 | 4924.80 | 4900.09 | 4904.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 4966.00 | 4913.27 | 4909.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 4995.00 | 4956.21 | 4934.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 4845.80 | 4954.99 | 4947.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 4845.80 | 4954.99 | 4947.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 4845.80 | 4954.99 | 4947.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 4845.80 | 4954.99 | 4947.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 4803.80 | 4924.75 | 4934.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 15:15:00 | 4785.10 | 4817.16 | 4857.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 4840.00 | 4821.72 | 4856.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 4840.00 | 4821.72 | 4856.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 4840.00 | 4821.72 | 4856.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:45:00 | 4760.70 | 4794.80 | 4829.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 4724.80 | 4795.84 | 4826.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:00:00 | 4760.30 | 4772.71 | 4803.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 4522.66 | 4729.52 | 4770.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 4488.56 | 4729.52 | 4770.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 4522.28 | 4729.52 | 4770.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-02 10:15:00 | 4731.00 | 4729.81 | 4766.95 | SL hit (close>ema200) qty=0.50 sl=4729.81 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 4731.10 | 4656.42 | 4646.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 4764.50 | 4685.68 | 4661.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 4723.50 | 4728.49 | 4693.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 4812.00 | 4755.21 | 4724.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 14:15:00 | 5052.60 | 4993.03 | 4935.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 5110.20 | 5071.07 | 5018.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 5168.00 | 5106.87 | 5064.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:30:00 | 5150.00 | 5132.01 | 5091.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 5127.20 | 5135.91 | 5104.13 | SL hit (close<ema200) qty=0.50 sl=5135.91 alert=retest1 |

### Cycle 136 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 5047.50 | 5087.34 | 5091.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 5033.00 | 5066.51 | 5079.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 5086.00 | 5037.51 | 5058.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 5086.00 | 5037.51 | 5058.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5086.00 | 5037.51 | 5058.23 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 5136.40 | 5080.86 | 5073.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 5754.00 | 5220.15 | 5138.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 11:15:00 | 5693.00 | 5707.12 | 5590.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 12:00:00 | 5693.00 | 5707.12 | 5590.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5712.90 | 5699.89 | 5629.87 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 5613.00 | 5637.08 | 5640.30 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 5694.00 | 5648.47 | 5645.18 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 5607.00 | 5641.32 | 5644.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 5596.50 | 5622.60 | 5634.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 5496.50 | 5490.50 | 5526.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:45:00 | 5497.50 | 5490.50 | 5526.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 5484.50 | 5490.82 | 5520.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:30:00 | 5438.50 | 5483.19 | 5507.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:00:00 | 5469.00 | 5481.77 | 5494.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:00:00 | 5465.00 | 5442.10 | 5457.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:45:00 | 5460.00 | 5444.28 | 5456.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 5450.00 | 5445.43 | 5456.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 5439.00 | 5446.47 | 5454.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 5439.50 | 5449.38 | 5455.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 5518.00 | 5460.08 | 5459.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 5518.00 | 5460.08 | 5459.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 5586.00 | 5529.50 | 5498.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 5607.00 | 5609.68 | 5563.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 12:00:00 | 5607.00 | 5609.68 | 5563.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5600.00 | 5646.71 | 5612.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5600.00 | 5646.71 | 5612.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5591.50 | 5635.67 | 5610.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 5589.00 | 5635.67 | 5610.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 5577.00 | 5610.14 | 5605.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 5568.00 | 5610.14 | 5605.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 5576.50 | 5603.41 | 5602.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 5569.00 | 5603.41 | 5602.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 12:15:00 | 5545.50 | 5591.83 | 5597.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 10:15:00 | 5537.50 | 5565.96 | 5580.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 10:15:00 | 5595.00 | 5542.43 | 5556.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 5595.00 | 5542.43 | 5556.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 5595.00 | 5542.43 | 5556.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 5595.00 | 5542.43 | 5556.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 5659.00 | 5565.74 | 5565.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:30:00 | 5659.50 | 5565.74 | 5565.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 5690.00 | 5590.59 | 5577.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 5750.00 | 5689.71 | 5646.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 15:15:00 | 6350.00 | 6370.40 | 6286.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:00:00 | 6429.00 | 6382.12 | 6299.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 12:30:00 | 6432.00 | 6404.30 | 6331.16 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 14:30:00 | 6430.50 | 6411.81 | 6347.38 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:15:00 | 6436.50 | 6407.05 | 6351.08 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 6380.50 | 6437.92 | 6419.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 6380.50 | 6437.92 | 6419.28 | SL hit (close<ema400) qty=1.00 sl=6419.28 alert=retest1 |

### Cycle 144 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 6320.00 | 6430.75 | 6437.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 6283.00 | 6401.20 | 6423.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 6225.50 | 6197.36 | 6237.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 14:15:00 | 6225.50 | 6197.36 | 6237.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 6225.50 | 6197.36 | 6237.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 6225.50 | 6197.36 | 6237.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 6233.00 | 6208.75 | 6235.78 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 6344.50 | 6258.58 | 6255.14 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 6280.00 | 6314.81 | 6315.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 6272.50 | 6300.03 | 6308.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 6304.00 | 6297.22 | 6305.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 6336.50 | 6297.22 | 6305.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 6285.00 | 6294.78 | 6303.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 6270.00 | 6294.78 | 6303.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 6268.00 | 6274.90 | 6289.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 6257.50 | 6277.74 | 6288.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 6396.00 | 6302.47 | 6298.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 6396.00 | 6302.47 | 6298.02 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 6218.00 | 6306.89 | 6315.42 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 6470.00 | 6326.31 | 6316.84 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 6314.50 | 6352.85 | 6352.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 13:15:00 | 6266.50 | 6335.58 | 6345.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 6408.00 | 6345.96 | 6347.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 6408.00 | 6345.96 | 6347.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6408.00 | 6345.96 | 6347.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 6417.50 | 6345.96 | 6347.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 6419.50 | 6360.67 | 6353.77 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 6327.00 | 6350.12 | 6350.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 6317.50 | 6341.16 | 6346.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 6265.00 | 6247.64 | 6276.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 6230.00 | 6247.64 | 6276.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 6223.50 | 6242.81 | 6271.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 6200.00 | 6266.87 | 6273.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 6333.00 | 6280.10 | 6279.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 6333.00 | 6280.10 | 6279.26 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5994.50 | 6226.00 | 6255.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 5946.50 | 6170.10 | 6226.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 5643.00 | 5630.88 | 5720.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 15:00:00 | 5643.00 | 5630.88 | 5720.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 5599.50 | 5628.90 | 5666.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 5592.00 | 5619.10 | 5652.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 5592.00 | 5626.05 | 5646.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 5688.00 | 5644.45 | 5645.59 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |

### Cycle 155 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 5614.00 | 5581.71 | 5579.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 5643.50 | 5594.07 | 5585.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 5604.50 | 5604.66 | 5593.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 5604.50 | 5604.66 | 5593.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 5589.50 | 5601.63 | 5592.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 5589.50 | 5601.63 | 5592.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 5624.00 | 5606.10 | 5595.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 5565.00 | 5600.78 | 5594.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 5524.50 | 5585.53 | 5587.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 5510.00 | 5570.42 | 5580.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 5565.00 | 5543.81 | 5560.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 5565.00 | 5543.81 | 5560.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 5565.00 | 5543.81 | 5560.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 5568.00 | 5543.81 | 5560.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 5520.00 | 5539.05 | 5557.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 5499.50 | 5527.09 | 5546.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:15:00 | 5224.52 | 5285.00 | 5353.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 5243.00 | 5240.48 | 5297.05 | SL hit (close>ema200) qty=0.50 sl=5240.48 alert=retest2 |

### Cycle 157 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 5125.00 | 5108.78 | 5107.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 5134.50 | 5119.01 | 5113.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 5115.00 | 5120.05 | 5114.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 5115.00 | 5120.05 | 5114.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 5115.00 | 5120.05 | 5114.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 5115.00 | 5120.05 | 5114.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 5109.00 | 5117.84 | 5114.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 5139.00 | 5117.84 | 5114.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 5103.00 | 5120.79 | 5117.86 | SL hit (close<static) qty=1.00 sl=5103.50 alert=retest2 |

### Cycle 158 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 5082.00 | 5113.03 | 5114.60 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 5175.00 | 5112.75 | 5108.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 5209.50 | 5132.10 | 5117.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 5196.50 | 5197.11 | 5163.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:00:00 | 5196.50 | 5197.11 | 5163.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 5185.00 | 5194.69 | 5165.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 5214.50 | 5200.80 | 5180.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 5223.50 | 5200.82 | 5188.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 5135.50 | 5222.11 | 5228.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 5135.50 | 5222.11 | 5228.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 5099.50 | 5197.59 | 5216.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 5001.00 | 4994.09 | 5039.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 5001.00 | 4994.09 | 5039.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 5021.00 | 4999.47 | 5037.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 5021.00 | 4999.47 | 5037.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 5051.50 | 5009.88 | 5038.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 5005.00 | 5030.66 | 5040.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 5006.50 | 5030.66 | 5040.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 4964.50 | 4911.43 | 4911.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 4964.50 | 4911.43 | 4911.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 5032.00 | 4958.50 | 4935.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 5000.00 | 5006.70 | 4972.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:45:00 | 4998.00 | 5006.70 | 4972.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 4989.00 | 5004.19 | 4980.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 5007.30 | 5004.81 | 4982.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 4977.00 | 4996.88 | 4982.93 | SL hit (close<static) qty=1.00 sl=4980.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 4987.30 | 5003.71 | 5003.93 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 5046.10 | 5000.95 | 4996.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 5104.30 | 5021.62 | 5006.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 5051.40 | 5073.06 | 5051.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 5051.40 | 5073.06 | 5051.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 5036.30 | 5065.71 | 5049.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 5036.30 | 5065.71 | 5049.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 5035.30 | 5059.63 | 5048.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 5038.20 | 5059.63 | 5048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 5027.90 | 5053.28 | 5046.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 5025.00 | 5053.28 | 5046.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 4999.50 | 5038.80 | 5041.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 4984.50 | 5020.12 | 5031.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 5017.20 | 5004.96 | 5018.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 5017.20 | 5004.96 | 5018.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 5017.20 | 5004.96 | 5018.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 5014.90 | 5004.96 | 5018.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5041.70 | 5014.40 | 5020.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 5041.70 | 5014.40 | 5020.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 5018.00 | 5015.12 | 5020.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:45:00 | 4982.40 | 5010.47 | 5017.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:00:00 | 4999.90 | 5008.36 | 5016.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:00:00 | 4996.80 | 5006.05 | 5014.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 4999.70 | 5004.78 | 5013.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 5020.00 | 5004.98 | 5011.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 5020.00 | 5004.98 | 5011.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 5007.40 | 5005.46 | 5011.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 5000.60 | 5005.46 | 5011.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:45:00 | 4999.90 | 5009.82 | 5012.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 5053.30 | 5021.55 | 5017.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 5053.30 | 5021.55 | 5017.23 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 4999.40 | 5019.79 | 5020.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 4977.90 | 5005.48 | 5013.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 15:15:00 | 4959.00 | 4934.77 | 4966.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 09:15:00 | 4913.80 | 4934.77 | 4966.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 4905.80 | 4928.97 | 4960.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 4881.80 | 4910.81 | 4940.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 4879.50 | 4901.73 | 4925.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 4878.40 | 4901.73 | 4925.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 4878.10 | 4896.38 | 4920.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 4896.60 | 4888.70 | 4908.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 4909.60 | 4888.70 | 4908.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 4869.90 | 4879.36 | 4896.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:45:00 | 4883.90 | 4879.36 | 4896.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4637.71 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4635.52 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4634.48 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4634.19 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4642.10 | 4617.21 | 4668.49 | SL hit (close>ema200) qty=0.50 sl=4617.21 alert=retest2 |

### Cycle 167 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 4555.00 | 4523.98 | 4521.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 4557.90 | 4535.57 | 4527.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 4512.60 | 4537.12 | 4529.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 4512.60 | 4537.12 | 4529.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4512.60 | 4537.12 | 4529.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 4514.50 | 4537.12 | 4529.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 4503.90 | 4530.48 | 4527.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 4503.30 | 4530.48 | 4527.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 4549.00 | 4549.29 | 4539.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:15:00 | 4580.00 | 4552.25 | 4545.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 4577.00 | 4556.02 | 4548.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 4574.30 | 4553.36 | 4550.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 15:15:00 | 4528.30 | 4547.15 | 4547.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 4528.30 | 4547.15 | 4547.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 4512.90 | 4539.59 | 4544.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 4534.90 | 4532.91 | 4539.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 4534.90 | 4532.91 | 4539.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 4534.90 | 4532.91 | 4539.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 4534.90 | 4532.91 | 4539.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 4534.00 | 4533.13 | 4539.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 4530.10 | 4533.13 | 4539.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 4535.80 | 4533.66 | 4538.72 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 4590.00 | 4548.83 | 4544.06 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 4466.50 | 4549.36 | 4554.70 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 4576.00 | 4554.08 | 4553.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 10:15:00 | 4593.90 | 4562.04 | 4557.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 4612.40 | 4620.57 | 4597.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:00:00 | 4612.40 | 4620.57 | 4597.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 4585.00 | 4613.46 | 4596.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 4585.00 | 4613.46 | 4596.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 4565.30 | 4603.82 | 4593.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 4565.30 | 4603.82 | 4593.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 4599.00 | 4602.25 | 4594.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 4605.00 | 4602.25 | 4594.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 4614.60 | 4606.21 | 4599.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 4607.40 | 4606.45 | 4600.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 4571.00 | 4598.17 | 4597.80 | SL hit (close<static) qty=1.00 sl=4581.50 alert=retest2 |

### Cycle 172 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 4568.30 | 4592.19 | 4595.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 4545.00 | 4582.76 | 4590.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 4556.20 | 4539.72 | 4559.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 4556.20 | 4539.72 | 4559.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 4550.80 | 4541.94 | 4558.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 4511.60 | 4536.33 | 4554.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 15:15:00 | 4286.02 | 4316.70 | 4343.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 4315.00 | 4312.73 | 4334.88 | SL hit (close>ema200) qty=0.50 sl=4312.73 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 4355.30 | 4336.83 | 4334.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 4369.80 | 4343.42 | 4337.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 4555.50 | 4560.03 | 4524.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:15:00 | 4552.50 | 4560.03 | 4524.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 4513.10 | 4550.65 | 4523.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 4513.10 | 4550.65 | 4523.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 4545.40 | 4549.60 | 4525.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 4532.10 | 4549.60 | 4525.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 4530.00 | 4545.68 | 4526.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 4538.40 | 4542.34 | 4526.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 4481.90 | 4530.25 | 4522.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 4481.90 | 4530.25 | 4522.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 4518.40 | 4527.88 | 4521.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 4507.50 | 4527.88 | 4521.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 4512.70 | 4524.85 | 4521.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 4528.80 | 4524.85 | 4521.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 4551.20 | 4530.12 | 4523.84 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 4530.00 | 4545.33 | 4546.09 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 14:15:00 | 4567.10 | 4548.99 | 4547.58 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 4520.00 | 4543.39 | 4545.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 4503.00 | 4531.57 | 4539.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 4525.20 | 4524.39 | 4533.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 15:00:00 | 4525.20 | 4524.39 | 4533.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 4529.80 | 4521.66 | 4529.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 4537.10 | 4521.66 | 4529.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 4532.10 | 4523.75 | 4529.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 4528.00 | 4523.75 | 4529.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 4529.80 | 4524.96 | 4529.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 4515.70 | 4528.88 | 4530.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 4515.00 | 4525.32 | 4528.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 4513.20 | 4521.02 | 4526.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 4289.91 | 4368.39 | 4384.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 4398.00 | 4368.39 | 4384.80 | SL hit (close>static) qty=0.50 sl=4368.39 alert=retest2 |

### Cycle 177 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 4399.90 | 4387.18 | 4385.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 4435.50 | 4396.85 | 4390.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 10:15:00 | 4391.60 | 4395.80 | 4390.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 10:15:00 | 4391.60 | 4395.80 | 4390.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 4391.60 | 4395.80 | 4390.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 4391.60 | 4395.80 | 4390.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 4375.00 | 4391.64 | 4389.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:45:00 | 4361.10 | 4391.64 | 4389.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 4360.70 | 4385.45 | 4386.56 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 4404.90 | 4384.00 | 4383.37 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 4368.40 | 4381.52 | 4382.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 4363.80 | 4377.97 | 4380.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 4369.00 | 4368.94 | 4375.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 4369.00 | 4368.94 | 4375.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 4369.00 | 4368.94 | 4375.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 4394.60 | 4368.94 | 4375.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4372.90 | 4369.73 | 4375.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:45:00 | 4334.90 | 4365.41 | 4371.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:45:00 | 4337.90 | 4364.19 | 4370.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 4430.00 | 4377.92 | 4373.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 4430.00 | 4377.92 | 4373.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 4479.10 | 4398.15 | 4382.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 4409.20 | 4426.98 | 4406.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 4409.20 | 4426.98 | 4406.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 4409.20 | 4426.98 | 4406.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:45:00 | 4413.50 | 4426.98 | 4406.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 4386.50 | 4418.88 | 4404.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 4391.60 | 4418.88 | 4404.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 4385.40 | 4412.19 | 4402.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 4372.90 | 4412.19 | 4402.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 4351.20 | 4396.26 | 4397.06 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 4440.00 | 4399.52 | 4395.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4450.00 | 4409.61 | 4400.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 4423.80 | 4427.26 | 4414.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 14:15:00 | 4423.80 | 4427.26 | 4414.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 4423.80 | 4427.26 | 4414.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:30:00 | 4423.00 | 4427.26 | 4414.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 4415.40 | 4424.89 | 4414.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:30:00 | 4452.30 | 4426.95 | 4416.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:30:00 | 4445.10 | 4430.70 | 4419.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 4447.50 | 4434.73 | 4422.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 4484.00 | 4467.08 | 4453.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 4464.00 | 4466.46 | 4454.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 4428.50 | 4466.46 | 4454.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4436.40 | 4460.45 | 4452.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:00:00 | 4485.30 | 4463.80 | 4455.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-12 09:15:00 | 4897.53 | 4743.46 | 4684.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 12:15:00 | 4690.00 | 4743.11 | 4746.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 15:15:00 | 4670.00 | 4711.52 | 4729.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 4751.00 | 4719.42 | 4731.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 4751.00 | 4719.42 | 4731.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 4751.00 | 4719.42 | 4731.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 4751.00 | 4719.42 | 4731.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 4749.40 | 4725.41 | 4733.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 4751.80 | 4725.41 | 4733.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 4758.00 | 4740.01 | 4738.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 4833.90 | 4763.69 | 4750.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 4772.30 | 4802.54 | 4782.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 4772.30 | 4802.54 | 4782.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 4772.30 | 4802.54 | 4782.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 4773.10 | 4802.54 | 4782.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 4770.30 | 4796.10 | 4781.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 4761.50 | 4796.10 | 4781.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 4751.10 | 4781.32 | 4776.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 4751.10 | 4781.32 | 4776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 4775.00 | 4779.99 | 4776.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 4775.00 | 4779.99 | 4776.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 4779.80 | 4779.95 | 4777.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 4740.00 | 4779.95 | 4777.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 4766.80 | 4777.32 | 4776.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 4757.80 | 4777.32 | 4776.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 4752.70 | 4772.40 | 4774.09 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 4800.10 | 4775.31 | 4774.96 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 4747.70 | 4782.96 | 4784.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4707.50 | 4751.91 | 4768.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 4738.50 | 4735.26 | 4753.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 4738.50 | 4735.26 | 4753.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 4736.20 | 4730.05 | 4740.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 4743.30 | 4730.05 | 4740.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 4567.40 | 4558.16 | 4588.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 4500.80 | 4547.28 | 4575.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 4600.00 | 4566.58 | 4566.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 14:15:00 | 4600.00 | 4566.58 | 4566.30 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 4488.30 | 4556.64 | 4562.18 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 4615.60 | 4550.56 | 4544.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 4625.00 | 4585.99 | 4564.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 4580.20 | 4587.07 | 4569.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:45:00 | 4575.10 | 4587.07 | 4569.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 4607.40 | 4591.14 | 4572.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:45:00 | 4607.20 | 4591.14 | 4572.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 4584.70 | 4589.85 | 4573.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 4602.00 | 4589.85 | 4573.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 4514.70 | 4574.82 | 4568.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 4514.70 | 4574.82 | 4568.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 4500.00 | 4559.86 | 4562.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 4463.00 | 4507.94 | 4529.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 4513.90 | 4493.12 | 4513.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 13:15:00 | 4513.90 | 4493.12 | 4513.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 4513.90 | 4493.12 | 4513.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 4500.00 | 4493.12 | 4513.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 4511.00 | 4496.70 | 4513.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 4499.80 | 4496.70 | 4513.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 4511.00 | 4499.56 | 4513.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 4473.60 | 4499.56 | 4513.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 4446.80 | 4478.05 | 4486.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 4466.40 | 4472.52 | 4479.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 4499.40 | 4483.53 | 4482.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 4499.40 | 4483.53 | 4482.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-19 09:15:00 | 4550.10 | 4497.58 | 4489.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 11:15:00 | 4495.00 | 4498.96 | 4491.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 12:00:00 | 4495.00 | 4498.96 | 4491.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 4511.30 | 4501.43 | 4493.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:30:00 | 4500.80 | 4501.43 | 4493.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 4496.00 | 4500.34 | 4493.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:30:00 | 4492.50 | 4500.34 | 4493.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 4460.60 | 4492.39 | 4490.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 4460.60 | 4492.39 | 4490.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 4478.00 | 4489.51 | 4489.65 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 4584.80 | 4507.30 | 4496.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 4686.00 | 4543.04 | 4514.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 4553.00 | 4589.07 | 4546.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 4553.00 | 4589.07 | 4546.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 4490.00 | 4569.25 | 4541.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 4475.20 | 4569.25 | 4541.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 4550.60 | 4565.52 | 4542.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:30:00 | 4486.00 | 4565.52 | 4542.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 4426.90 | 4537.80 | 4531.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 4426.90 | 4537.80 | 4531.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 4500.00 | 4530.24 | 4528.71 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 4500.00 | 4524.19 | 4526.10 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 4539.10 | 4526.71 | 4526.01 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 12:15:00 | 4507.70 | 4522.91 | 4524.35 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 4594.00 | 4537.13 | 4530.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4608.00 | 4555.97 | 4541.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 4537.20 | 4558.19 | 4546.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 12:15:00 | 4537.20 | 4558.19 | 4546.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 4537.20 | 4558.19 | 4546.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:45:00 | 4542.90 | 4558.19 | 4546.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 4533.00 | 4553.15 | 4545.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:00:00 | 4533.00 | 4553.15 | 4545.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 4505.10 | 4543.54 | 4541.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 4505.10 | 4543.54 | 4541.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 4494.80 | 4533.79 | 4537.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 4449.80 | 4516.99 | 4529.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 4512.20 | 4488.47 | 4507.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 4512.20 | 4488.47 | 4507.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 4512.20 | 4488.47 | 4507.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 4512.20 | 4488.47 | 4507.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 4460.00 | 4482.78 | 4502.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 4422.00 | 4482.78 | 4502.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 13:15:00 | 4565.00 | 4471.54 | 4485.59 | SL hit (close>static) qty=1.00 sl=4515.30 alert=retest2 |

### Cycle 201 — BUY (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 14:15:00 | 4641.50 | 4505.54 | 4499.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 4695.40 | 4606.37 | 4556.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 4611.90 | 4661.13 | 4604.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 10:00:00 | 4611.90 | 4661.13 | 4604.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 4597.90 | 4648.48 | 4603.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 4597.90 | 4648.48 | 4603.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 4632.20 | 4645.23 | 4606.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 4638.70 | 4661.78 | 4617.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 4645.80 | 4655.40 | 4656.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 15:15:00 | 4645.80 | 4655.40 | 4656.16 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 4730.00 | 4670.32 | 4662.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 4818.00 | 4748.99 | 4708.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 4797.50 | 4798.74 | 4758.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:30:00 | 4820.00 | 4798.74 | 4758.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4809.60 | 4802.07 | 4766.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4865.60 | 4823.59 | 4798.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 14:15:00 | 4894.70 | 4931.67 | 4933.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 14:15:00 | 4894.70 | 4931.67 | 4933.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 4857.60 | 4912.91 | 4924.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 4880.00 | 4868.24 | 4890.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 4880.00 | 4868.24 | 4890.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 4880.00 | 4868.24 | 4890.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 4902.60 | 4868.24 | 4890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 4860.50 | 4866.70 | 4887.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 4869.70 | 4866.70 | 4887.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4805.30 | 4837.35 | 4863.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:45:00 | 4783.30 | 4827.70 | 4856.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:30:00 | 4795.80 | 4812.33 | 4844.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 13:45:00 | 4790.50 | 4807.04 | 4838.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 4792.10 | 4799.95 | 4827.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4706.40 | 4692.75 | 4715.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:15:00 | 4728.20 | 4692.75 | 4715.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 4719.50 | 4698.10 | 4715.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 4701.00 | 4695.34 | 4713.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:45:00 | 4698.10 | 4694.63 | 4711.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:00:00 | 4684.00 | 4670.69 | 4687.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 4733.00 | 4695.17 | 4693.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 4733.00 | 4695.17 | 4693.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 4759.50 | 4708.03 | 4699.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 4729.50 | 4730.28 | 4713.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 4729.50 | 4730.28 | 4713.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 4798.00 | 4743.82 | 4720.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 4824.00 | 4743.82 | 4720.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 4700.50 | 4737.86 | 4726.45 | SL hit (close<static) qty=1.00 sl=4720.30 alert=retest2 |

### Cycle 206 — SELL (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 15:15:00 | 4699.00 | 4718.02 | 4719.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 4682.30 | 4710.88 | 4715.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 15:15:00 | 4652.00 | 4645.48 | 4663.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 4670.00 | 4645.48 | 4663.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 4673.00 | 4650.98 | 4664.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 4648.80 | 4651.94 | 4663.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 4648.00 | 4653.42 | 4662.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:00:00 | 4647.40 | 4650.47 | 4659.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 4634.00 | 4646.69 | 4655.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 14:30:00 | 4150.00 | 2023-05-17 11:15:00 | 4161.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2023-05-16 12:45:00 | 4148.40 | 2023-05-17 11:15:00 | 4161.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-05-16 13:15:00 | 4151.30 | 2023-05-17 11:15:00 | 4161.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2023-05-23 10:15:00 | 4104.10 | 2023-05-25 09:15:00 | 4172.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-05-23 15:15:00 | 4104.00 | 2023-05-25 09:15:00 | 4172.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-05-24 09:30:00 | 4112.00 | 2023-05-25 09:15:00 | 4172.05 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-06-08 09:15:00 | 4403.35 | 2023-06-08 12:15:00 | 4359.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-06-12 13:30:00 | 4230.00 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-06-12 15:15:00 | 4241.10 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-06-13 11:45:00 | 4233.90 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-06-13 15:15:00 | 4245.00 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-06-14 12:45:00 | 4216.05 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-06-15 13:00:00 | 4215.00 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-06-15 13:45:00 | 4199.95 | 2023-06-16 12:15:00 | 4276.70 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2023-06-22 13:15:00 | 4316.00 | 2023-06-23 09:15:00 | 4264.95 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-06-22 14:15:00 | 4317.45 | 2023-06-23 09:15:00 | 4264.95 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-06-22 15:00:00 | 4314.10 | 2023-06-23 09:15:00 | 4264.95 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-07-07 13:45:00 | 4632.90 | 2023-07-13 14:15:00 | 4639.85 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2023-07-13 14:00:00 | 4642.90 | 2023-07-13 14:15:00 | 4639.85 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-07-20 09:45:00 | 4690.90 | 2023-07-20 15:15:00 | 4667.75 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-07-20 11:30:00 | 4689.40 | 2023-07-20 15:15:00 | 4667.75 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-07-20 12:00:00 | 4690.95 | 2023-07-20 15:15:00 | 4667.75 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-07-20 13:15:00 | 4689.45 | 2023-07-20 15:15:00 | 4667.75 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-08-07 12:15:00 | 4595.00 | 2023-08-11 15:15:00 | 4740.00 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2023-08-07 15:00:00 | 4594.95 | 2023-08-11 15:15:00 | 4740.00 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2023-08-08 09:15:00 | 4662.20 | 2023-08-11 15:15:00 | 4740.00 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2023-08-18 13:45:00 | 4781.45 | 2023-08-18 14:15:00 | 4735.65 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-08-23 12:15:00 | 4732.35 | 2023-08-29 09:15:00 | 4733.40 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2023-08-24 11:15:00 | 4739.25 | 2023-08-29 09:15:00 | 4733.40 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2023-09-04 09:45:00 | 4810.75 | 2023-09-05 12:15:00 | 5291.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-04 14:15:00 | 5388.50 | 2023-10-05 09:15:00 | 5349.95 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-10-04 15:00:00 | 5388.00 | 2023-10-05 09:15:00 | 5349.95 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-10-05 09:15:00 | 5390.90 | 2023-10-05 09:15:00 | 5349.95 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-10-16 14:45:00 | 5242.30 | 2023-10-25 09:15:00 | 4980.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 11:45:00 | 5240.25 | 2023-10-25 09:15:00 | 4978.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 12:30:00 | 5241.10 | 2023-10-25 09:15:00 | 4979.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 13:15:00 | 5240.05 | 2023-10-25 09:15:00 | 4978.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 10:15:00 | 5218.30 | 2023-10-25 09:15:00 | 4957.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-16 14:45:00 | 5242.30 | 2023-10-26 15:15:00 | 4915.00 | STOP_HIT | 0.50 | 6.24% |
| SELL | retest2 | 2023-10-17 11:45:00 | 5240.25 | 2023-10-26 15:15:00 | 4915.00 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2023-10-17 12:30:00 | 5241.10 | 2023-10-26 15:15:00 | 4915.00 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2023-10-17 13:15:00 | 5240.05 | 2023-10-26 15:15:00 | 4915.00 | STOP_HIT | 0.50 | 6.20% |
| SELL | retest2 | 2023-10-18 10:15:00 | 5218.30 | 2023-10-26 15:15:00 | 4915.00 | STOP_HIT | 0.50 | 5.81% |
| BUY | retest2 | 2023-11-02 13:15:00 | 4917.25 | 2023-11-02 13:15:00 | 4897.85 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-11-10 09:15:00 | 5453.10 | 2023-11-16 15:15:00 | 5470.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2023-11-21 14:00:00 | 5271.65 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-11-21 14:30:00 | 5283.90 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-11-23 12:30:00 | 5283.80 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-11-23 14:30:00 | 5273.00 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-11-28 10:30:00 | 5251.00 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-11-29 12:00:00 | 5253.60 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-11-30 11:00:00 | 5253.50 | 2023-11-30 11:15:00 | 5323.90 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-12-05 13:00:00 | 5359.00 | 2023-12-12 12:15:00 | 5407.60 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2023-12-21 09:45:00 | 5480.95 | 2023-12-28 15:15:00 | 5546.00 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2023-12-21 10:45:00 | 5472.05 | 2023-12-28 15:15:00 | 5546.00 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2023-12-21 12:00:00 | 5467.35 | 2023-12-28 15:15:00 | 5546.00 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2023-12-21 12:30:00 | 5470.00 | 2023-12-28 15:15:00 | 5546.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2023-12-22 13:15:00 | 5520.00 | 2023-12-28 15:15:00 | 5546.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2024-01-01 13:30:00 | 5490.50 | 2024-01-05 14:15:00 | 5589.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-01-01 15:15:00 | 5464.10 | 2024-01-05 14:15:00 | 5589.80 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-01-02 15:15:00 | 5461.05 | 2024-01-05 14:15:00 | 5589.80 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-01-04 15:15:00 | 5487.00 | 2024-01-05 14:15:00 | 5589.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-01-09 09:15:00 | 5655.90 | 2024-01-15 11:15:00 | 5803.30 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2024-01-09 10:00:00 | 5639.85 | 2024-01-15 11:15:00 | 5803.30 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2024-01-09 10:45:00 | 5643.00 | 2024-01-15 11:15:00 | 5803.30 | STOP_HIT | 1.00 | 2.84% |
| SELL | retest2 | 2024-01-20 12:45:00 | 5780.00 | 2024-01-23 11:15:00 | 5823.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-01-23 09:45:00 | 5774.95 | 2024-01-23 11:15:00 | 5823.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-01-23 13:15:00 | 5796.95 | 2024-01-23 15:15:00 | 5820.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-01-23 14:00:00 | 5804.75 | 2024-01-23 15:15:00 | 5820.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-02-06 15:15:00 | 5666.05 | 2024-02-07 09:15:00 | 5837.85 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-02-07 13:00:00 | 5785.35 | 2024-02-08 10:15:00 | 5850.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-02-07 13:30:00 | 5784.10 | 2024-02-08 10:15:00 | 5850.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-02-08 09:15:00 | 5768.10 | 2024-02-08 10:15:00 | 5850.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-02-08 10:00:00 | 5781.70 | 2024-02-08 10:15:00 | 5850.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-02-22 11:30:00 | 6056.95 | 2024-02-22 15:15:00 | 6015.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-03-04 11:45:00 | 5600.80 | 2024-03-12 09:15:00 | 5320.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 11:45:00 | 5600.80 | 2024-03-13 13:15:00 | 5040.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-27 09:15:00 | 5120.25 | 2024-04-04 14:15:00 | 5440.15 | STOP_HIT | 1.00 | 6.25% |
| BUY | retest2 | 2024-03-27 09:45:00 | 5156.45 | 2024-04-04 14:15:00 | 5440.15 | STOP_HIT | 1.00 | 5.50% |
| SELL | retest2 | 2024-04-08 14:30:00 | 5395.00 | 2024-04-12 09:15:00 | 5464.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-04-10 15:00:00 | 5417.00 | 2024-04-12 09:15:00 | 5464.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-04-26 11:30:00 | 5530.15 | 2024-05-09 14:15:00 | 5256.59 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-04-30 14:30:00 | 5514.50 | 2024-05-09 15:15:00 | 5253.64 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2024-05-03 10:15:00 | 5533.25 | 2024-05-09 15:15:00 | 5238.77 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2024-04-26 11:30:00 | 5530.15 | 2024-05-13 13:15:00 | 5224.95 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2024-04-30 14:30:00 | 5514.50 | 2024-05-13 13:15:00 | 5224.95 | STOP_HIT | 0.50 | 5.25% |
| SELL | retest2 | 2024-05-03 10:15:00 | 5533.25 | 2024-05-13 13:15:00 | 5224.95 | STOP_HIT | 0.50 | 5.57% |
| BUY | retest2 | 2024-05-17 10:15:00 | 5472.00 | 2024-05-23 12:15:00 | 5575.05 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2024-05-18 09:15:00 | 5551.95 | 2024-05-23 12:15:00 | 5575.05 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-05-29 10:15:00 | 5183.75 | 2024-06-03 12:15:00 | 5152.05 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-06-14 12:00:00 | 6100.00 | 2024-06-27 09:15:00 | 6710.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-08-05 09:15:00 | 6700.00 | 2024-08-06 12:15:00 | 6787.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-07 09:45:00 | 6745.55 | 2024-08-07 13:15:00 | 6842.95 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-08-07 12:45:00 | 6760.00 | 2024-08-07 13:15:00 | 6842.95 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-08-16 11:00:00 | 6112.00 | 2024-08-21 09:15:00 | 6395.00 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2024-09-13 12:45:00 | 6499.75 | 2024-09-17 12:15:00 | 6466.60 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-13 13:45:00 | 6498.85 | 2024-09-17 12:15:00 | 6466.60 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-09-13 15:00:00 | 6499.95 | 2024-09-17 12:15:00 | 6466.60 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-16 09:15:00 | 6500.30 | 2024-09-17 12:15:00 | 6466.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-09-20 09:15:00 | 6341.30 | 2024-09-26 11:15:00 | 6341.10 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-10-04 11:30:00 | 6528.45 | 2024-10-14 10:15:00 | 6613.65 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2024-10-04 13:00:00 | 6527.00 | 2024-10-14 10:15:00 | 6613.65 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2024-10-25 14:15:00 | 6320.70 | 2024-10-28 10:15:00 | 6481.20 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-11-21 12:30:00 | 5747.20 | 2024-11-25 09:15:00 | 5865.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-11-21 13:15:00 | 5752.30 | 2024-11-25 09:15:00 | 5865.10 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-11-21 14:30:00 | 5760.50 | 2024-11-25 09:15:00 | 5865.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-11-22 10:30:00 | 5756.80 | 2024-11-25 09:15:00 | 5865.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-12-09 10:15:00 | 6090.00 | 2024-12-09 13:15:00 | 6041.20 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-12-09 12:00:00 | 6100.00 | 2024-12-09 13:15:00 | 6041.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-12-10 11:30:00 | 6102.30 | 2024-12-16 14:15:00 | 6191.95 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2024-12-30 09:15:00 | 5743.30 | 2024-12-30 10:15:00 | 5817.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-01-16 11:00:00 | 5176.05 | 2025-01-23 12:15:00 | 5092.05 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2025-01-16 11:45:00 | 5160.50 | 2025-01-23 12:15:00 | 5092.05 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2025-01-17 10:45:00 | 5140.65 | 2025-01-23 12:15:00 | 5092.05 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2025-02-01 14:30:00 | 5142.90 | 2025-02-05 11:15:00 | 5092.15 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-02-03 09:15:00 | 5195.50 | 2025-02-05 11:15:00 | 5092.15 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-02-04 09:15:00 | 5173.90 | 2025-02-05 11:15:00 | 5092.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-02-04 14:00:00 | 5149.00 | 2025-02-05 11:15:00 | 5092.15 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest1 | 2025-02-14 09:15:00 | 4325.35 | 2025-02-17 10:15:00 | 4496.80 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest1 | 2025-02-14 09:45:00 | 4375.30 | 2025-02-17 10:15:00 | 4496.80 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest1 | 2025-02-14 12:30:00 | 4380.05 | 2025-02-17 10:15:00 | 4496.80 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest1 | 2025-02-14 13:45:00 | 4349.15 | 2025-02-17 10:15:00 | 4496.80 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-02-25 10:15:00 | 4750.40 | 2025-02-25 10:15:00 | 4727.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-03-07 10:45:00 | 4906.00 | 2025-03-11 14:15:00 | 4836.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-03-10 10:45:00 | 4910.00 | 2025-03-11 14:15:00 | 4836.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-03-10 13:15:00 | 4915.10 | 2025-03-11 14:15:00 | 4836.75 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-10 14:45:00 | 4926.25 | 2025-03-11 14:15:00 | 4836.75 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-03-26 14:45:00 | 4850.45 | 2025-03-28 09:15:00 | 5230.00 | STOP_HIT | 1.00 | -7.83% |
| SELL | retest2 | 2025-03-26 15:15:00 | 4850.00 | 2025-03-28 09:15:00 | 5230.00 | STOP_HIT | 1.00 | -7.84% |
| SELL | retest2 | 2025-03-27 10:30:00 | 4831.20 | 2025-03-28 09:15:00 | 5230.00 | STOP_HIT | 1.00 | -8.25% |
| SELL | retest2 | 2025-03-27 11:45:00 | 4842.85 | 2025-03-28 09:15:00 | 5230.00 | STOP_HIT | 1.00 | -7.99% |
| SELL | retest2 | 2025-04-29 14:45:00 | 4760.70 | 2025-05-02 09:15:00 | 4522.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 4724.80 | 2025-05-02 09:15:00 | 4488.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 13:00:00 | 4760.30 | 2025-05-02 09:15:00 | 4522.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 14:45:00 | 4760.70 | 2025-05-02 10:15:00 | 4731.00 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2025-04-30 09:15:00 | 4724.80 | 2025-05-02 10:15:00 | 4731.00 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2025-04-30 13:00:00 | 4760.30 | 2025-05-02 10:15:00 | 4731.00 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2025-05-06 11:00:00 | 4749.50 | 2025-05-09 09:15:00 | 4512.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 4749.50 | 2025-05-09 14:15:00 | 4624.70 | STOP_HIT | 0.50 | 2.63% |
| SELL | retest2 | 2025-05-06 12:30:00 | 4698.10 | 2025-05-12 09:15:00 | 4751.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-05-06 14:00:00 | 4697.90 | 2025-05-12 09:15:00 | 4751.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2025-05-14 09:15:00 | 4812.00 | 2025-05-16 14:15:00 | 5052.60 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 09:15:00 | 4812.00 | 2025-05-22 09:15:00 | 5127.20 | STOP_HIT | 0.50 | 6.55% |
| BUY | retest2 | 2025-05-21 09:30:00 | 5168.00 | 2025-05-22 15:15:00 | 5047.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-05-21 13:30:00 | 5150.00 | 2025-05-22 15:15:00 | 5047.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-05-22 09:45:00 | 5126.90 | 2025-05-22 15:15:00 | 5047.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-10 13:30:00 | 5438.50 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-06-11 13:00:00 | 5469.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-13 11:00:00 | 5465.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-13 11:45:00 | 5460.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-13 15:00:00 | 5439.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-06-16 09:15:00 | 5439.50 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-07-03 10:00:00 | 6429.00 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest1 | 2025-07-03 12:30:00 | 6432.00 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2025-07-03 14:30:00 | 6430.50 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-07-04 09:15:00 | 6436.50 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-08 13:30:00 | 6475.50 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-07-08 14:45:00 | 6474.50 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-07-09 09:30:00 | 6475.00 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-07-09 10:45:00 | 6483.50 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-07-22 10:15:00 | 6270.00 | 2025-07-23 10:15:00 | 6396.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-22 13:45:00 | 6268.00 | 2025-07-23 10:15:00 | 6396.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-23 09:15:00 | 6257.50 | 2025-07-23 10:15:00 | 6396.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-08-06 11:30:00 | 6200.00 | 2025-08-06 12:15:00 | 6333.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-13 13:45:00 | 5592.00 | 2025-08-18 09:15:00 | 5688.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-08-14 11:15:00 | 5592.00 | 2025-08-18 09:15:00 | 5688.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-08-18 11:45:00 | 5580.00 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-20 10:15:00 | 5582.50 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-20 11:15:00 | 5577.00 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-20 15:00:00 | 5567.50 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-25 14:00:00 | 5499.50 | 2025-08-29 10:15:00 | 5224.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:00:00 | 5499.50 | 2025-09-01 09:15:00 | 5243.00 | STOP_HIT | 0.50 | 4.66% |
| BUY | retest2 | 2025-09-12 09:15:00 | 5139.00 | 2025-09-12 12:15:00 | 5103.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-18 09:30:00 | 5214.50 | 2025-09-23 10:15:00 | 5135.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-09-19 09:30:00 | 5223.50 | 2025-09-23 10:15:00 | 5135.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-29 09:30:00 | 5005.00 | 2025-10-03 10:15:00 | 4964.50 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-09-29 10:15:00 | 5006.50 | 2025-10-03 10:15:00 | 4964.50 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-10-07 10:00:00 | 5007.30 | 2025-10-07 11:15:00 | 4977.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-09 14:30:00 | 5003.10 | 2025-10-13 10:15:00 | 4987.30 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-10-10 13:00:00 | 5009.20 | 2025-10-13 10:15:00 | 4987.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-23 11:45:00 | 4982.40 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-23 13:00:00 | 4999.90 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-23 14:00:00 | 4996.80 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-23 15:00:00 | 4999.70 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-24 11:15:00 | 5000.60 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-24 13:45:00 | 4999.90 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-30 13:45:00 | 4881.80 | 2025-11-07 10:15:00 | 4637.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 11:45:00 | 4879.50 | 2025-11-07 10:15:00 | 4635.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 12:15:00 | 4878.40 | 2025-11-07 10:15:00 | 4634.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 12:45:00 | 4878.10 | 2025-11-07 10:15:00 | 4634.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 4881.80 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-10-31 11:45:00 | 4879.50 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-10-31 12:15:00 | 4878.40 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-31 12:45:00 | 4878.10 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-14 09:15:00 | 4454.00 | 2025-11-17 12:15:00 | 4555.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-14 14:45:00 | 4478.40 | 2025-11-17 12:15:00 | 4555.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-11-20 12:15:00 | 4580.00 | 2025-11-21 15:15:00 | 4528.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-11-20 14:15:00 | 4577.00 | 2025-11-21 15:15:00 | 4528.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-21 13:45:00 | 4574.30 | 2025-11-21 15:15:00 | 4528.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-02 09:15:00 | 4605.00 | 2025-12-03 09:15:00 | 4571.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-02 13:45:00 | 4614.60 | 2025-12-03 09:15:00 | 4571.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-02 15:00:00 | 4607.40 | 2025-12-03 09:15:00 | 4571.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-04 13:45:00 | 4511.60 | 2025-12-18 15:15:00 | 4286.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:45:00 | 4511.60 | 2025-12-19 11:15:00 | 4315.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-01-08 09:15:00 | 4515.70 | 2026-01-20 14:15:00 | 4289.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 4515.70 | 2026-01-20 14:15:00 | 4398.00 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2026-01-08 11:00:00 | 4515.00 | 2026-01-20 14:15:00 | 4289.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 4515.00 | 2026-01-20 14:15:00 | 4398.00 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2026-01-08 11:45:00 | 4513.20 | 2026-01-20 14:15:00 | 4287.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 4513.20 | 2026-01-20 14:15:00 | 4398.00 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2026-01-29 10:45:00 | 4334.90 | 2026-01-30 12:15:00 | 4430.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-01-29 11:45:00 | 4337.90 | 2026-01-30 12:15:00 | 4430.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-02-04 09:30:00 | 4452.30 | 2026-02-12 09:15:00 | 4897.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 10:30:00 | 4445.10 | 2026-02-12 09:15:00 | 4889.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 13:00:00 | 4447.50 | 2026-02-12 09:15:00 | 4892.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-05 14:45:00 | 4484.00 | 2026-02-12 09:15:00 | 4932.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-06 13:00:00 | 4485.30 | 2026-02-12 09:15:00 | 4933.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-05 12:45:00 | 4500.80 | 2026-03-06 14:15:00 | 4600.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-03-16 09:15:00 | 4473.60 | 2026-03-18 12:15:00 | 4499.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-03-17 11:15:00 | 4446.80 | 2026-03-18 12:15:00 | 4499.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-17 15:15:00 | 4466.40 | 2026-03-18 12:15:00 | 4499.40 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-03-30 09:15:00 | 4422.00 | 2026-03-30 13:15:00 | 4565.00 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-04-02 12:30:00 | 4638.70 | 2026-04-07 15:15:00 | 4645.80 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4865.60 | 2026-04-17 14:15:00 | 4894.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2026-04-22 10:45:00 | 4783.30 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2026-04-22 12:30:00 | 4795.80 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2026-04-22 13:45:00 | 4790.50 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2026-04-23 10:15:00 | 4792.10 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2026-04-28 11:45:00 | 4701.00 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-04-28 12:45:00 | 4698.10 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-04-29 12:00:00 | 4684.00 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-05-04 09:15:00 | 4824.00 | 2026-05-04 12:15:00 | 4700.50 | STOP_HIT | 1.00 | -2.56% |
